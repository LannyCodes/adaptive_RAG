"""
Agent 运行时模块
================
用 LangGraph 原生 create_react_agent + Checkpointer 替换 agent_rag.py 的手写 ReAct 循环。

与旧实现 (agent_rag.py) 的关键差异:
1. ReAct 循环: 手写 while 循环 → create_react_agent 预建图（内部等价循环，但获得
   状态持久化、中断恢复、流式事件等基础设施能力）
2. 工具共享状态: 实例级 self.documents → ThreadSafeDocStore 按 thread_id 隔离，
   工具经 RunnableConfig 取会话 ID，消除并发会话间的状态污染
3. 持久化: 无 → RedisSaver（有 REDIS_URL 时）/ MemorySaver（降级）
4. 迭代上限: 硬编码 15 → config 的 recursion_limit（AGENT_MAX_ITERATIONS）

依赖版本: langgraph==0.3.34（requirements.txt 锁定）
"""

import threading
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config import (
    AGENT_MAX_ITERATIONS,
    CONTEXT_COMPRESS_KEEP_RECENT,
    CONTEXT_COMPRESS_THRESHOLD,
    REDIS_URL,
)
from routers_and_graders import create_chat_model, initialize_graders_and_router


# ═══════════════════════════════════════════════════════════════
# System Prompt — 与 agent_rag.py 保持一致（策略编码为自然语言指令）
# ═══════════════════════════════════════════════════════════════

RAG_AGENT_SYSTEM_PROMPT = """你是一个自适应 RAG (检索增强生成) Agent。你的任务是用工具回答用户问题。

## 可用工具
- route_query: 判断问题应该用知识库(vectorstore)还是网络搜索(web_search)
- decompose_query: 将复杂问题分解为子问题
- retrieve_from_vectorstore: 从知识库检索文档
- search_web: 网络搜索
- grade_documents: 评估检索到的文档是否与问题相关，过滤不相关文档
- rewrite_query: 优化查询以获得更好的检索结果
- check_answerability: 判断已有文档是否足够回答问题
- generate_answer: 基于文档生成最终答案
- check_hallucination: 检查生成的答案是否有幻觉

## 策略指南
1. 首先调用 route_query 判断走知识库还是网络搜索
2. 知识库路线: decompose_query → retrieve_from_vectorstore → grade_documents
3. 如果文档质量不好: rewrite_query → retrieve_from_vectorstore → grade_documents (最多重试2次)
4. 如果多次检索仍无好结果: 回退到 search_web
5. 生成前用 check_answerability 确认文档足够
6. 用 generate_answer 生成答案
7. 用 check_hallucination 验证答案
8. 验证通过后，直接输出最终答案给用户（不要再调用工具）

## 约束
- 每次只调用一个工具
- 检索重试不超过2次
- 最终必须输出用户能读懂的答案
- 如果找不到相关信息，诚实告知用户"""


# ═══════════════════════════════════════════════════════════════
# ThreadSafeDocStore — 按会话隔离的检索文档暂存区
# ═══════════════════════════════════════════════════════════════

class ThreadSafeDocStore:
    """按 thread_id 隔离的文档暂存区（线程安全）。

    替代旧实现中 Agent 实例级的 self.documents:
    - 旧: 单实例全局共享，并发会话互相污染
    - 新: {thread_id: [Document, ...]}，每个会话独立暂存

    工具之间"检索 → 评分 → 生成"的文档传递通过本暂存区完成，
    与 LangGraph checkpoint 的消息历史解耦（文档对象较大，不进 checkpoint）。
    """

    def __init__(self):
        self._store: Dict[str, List[Document]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _tid(config: Optional[RunnableConfig]) -> str:
        return (config or {}).get("configurable", {}).get("thread_id", "default")

    def set(self, config: Optional[RunnableConfig], docs: List[Document]):
        with self._lock:
            self._store[self._tid(config)] = docs

    def get(self, config: Optional[RunnableConfig]) -> List[Document]:
        with self._lock:
            return list(self._store.get(self._tid(config), []))

    def clear(self, config: Optional[RunnableConfig]):
        with self._lock:
            self._store.pop(self._tid(config), None)


# ═══════════════════════════════════════════════════════════════
# 工具工厂 — 从 agent_rag.py 迁移的 9 个工具
# ═══════════════════════════════════════════════════════════════
#
# 每个工具做一件事，由 Agent 决策何时调用。
# 工具函数本身是"哑"的——不包含流程逻辑，只执行操作并返回观察结果。
#
# 与旧实现的唯一行为差异:
#   文档暂存从 self_ref.documents 改为 doc_store + RunnableConfig 注入。
#   RunnableConfig 参数由 LangChain 自动注入，不出现在 LLM 可见的 tool schema 中。

def build_rag_tools(doc_processor, graders, doc_store: ThreadSafeDocStore) -> list:
    """构建 RAG 工具集（9 个），返回 langchain @tool 列表。

    薄包装层: 核心逻辑在 agents/rag_tool_impl.py（与 MCP server 共享），
    本层只负责文档暂存（doc_store + RunnableConfig 注入 thread_id）。
    """
    from agents import rag_tool_impl as impl

    # ── 工具 1: 路由 ──
    @tool
    def route_query(question: str) -> str:
        """判断用户问题应该用知识库(vectorstore)还是网络搜索(web_search)。收到问题后应首先调用。"""
        return impl.route_query(graders, question)

    # ── 工具 2: 查询分解 ──
    @tool
    def decompose_query(question: str) -> str:
        """将复杂的多跳问题分解为多个简单的子问题。对需要多步推理的问题使用。"""
        return impl.decompose_query(graders, question)

    # ── 工具 3: 知识库检索 ──
    @tool
    def retrieve_from_vectorstore(
        query: str, top_k: int = 5, config: RunnableConfig = None
    ) -> str:
        """从本地知识库检索与查询相关的文档。返回文档内容和编号。"""
        text, docs = impl.retrieve_from_vectorstore(doc_processor, query, top_k)
        if docs:
            doc_store.set(config, docs)
        return text

    # ── 工具 4: 网络搜索 ──
    @tool
    def search_web(query: str, config: RunnableConfig = None) -> str:
        """通过网络搜索获取实时信息。知识库结果不足时使用。"""
        text, docs = impl.search_web(query)
        if docs:
            doc_store.set(config, docs)
        return text

    # ── 工具 5: 文档相关性评分 ──
    @tool
    def grade_documents(question: str, config: RunnableConfig = None) -> str:
        """评估当前检索到的文档与问题的相关性，过滤不相关文档。检索后应调用此工具。"""
        text, filtered = impl.grade_documents(question, doc_store.get(config))
        if filtered:
            doc_store.set(config, filtered)
        return text

    # ── 工具 6: 查询重写 ──
    @tool
    def rewrite_query(question: str, context_hint: str = "") -> str:
        """优化查询以获得更好的检索结果。检索质量差时使用。context_hint 可包含之前检索到的关键词。"""
        return impl.rewrite_query(graders, question, context_hint)

    # ── 工具 7: 可回答性检查 ──
    @tool
    def check_answerability(question: str, config: RunnableConfig = None) -> str:
        """检查已有文档是否足够回答用户问题。返回 yes 或 no。生成答案前使用。"""
        return impl.check_answerability(graders, question, doc_store.get(config))

    # ── 工具 8: 答案生成 ──
    @tool
    def generate_answer(question: str, config: RunnableConfig = None) -> str:
        """基于当前检索和筛选的文档生成最终答案。仅在文档经过评分后调用。"""
        return impl.generate_answer(question, doc_store.get(config))

    # ── 工具 9: 幻觉检查 ──
    @tool
    def check_hallucination(generated_answer: str, config: RunnableConfig = None) -> str:
        """检查答案内容是否都能在文档中找到支撑。生成答案后使用。返回 yes(无幻觉) 或 no(有幻觉)。"""
        return impl.check_hallucination(graders, generated_answer, doc_store.get(config))

    return [
        route_query,
        decompose_query,
        retrieve_from_vectorstore,
        search_web,
        grade_documents,
        rewrite_query,
        check_answerability,
        generate_answer,
        check_hallucination,
    ]


# ═══════════════════════════════════════════════════════════════
# Checkpointer — Redis 优先，Memory 降级
# ═══════════════════════════════════════════════════════════════

def build_checkpointer():
    """构建状态检查点存储。

    - 有 REDIS_URL: RedisSaver 持久化（重启/崩溃后会话可恢复）
    - 否则: MemorySaver（仅进程内存，重启丢状态，仅开发用）
    """
    if REDIS_URL:
        try:
            from langgraph.checkpoint.redis import RedisSaver

            saver = RedisSaver(REDIS_URL)
            saver.setup()  # 初始化 Redis 索引（幂等）
            print("✅ Agent 状态持久化: RedisSaver")
            return saver
        except Exception as e:
            print(f"⚠️ RedisSaver 初始化失败，降级 MemorySaver: {e}")
    else:
        print("⚠️ 未配置 REDIS_URL，Agent 状态仅保存在内存（重启丢失）")
    return MemorySaver()


# ═══════════════════════════════════════════════════════════════
# AgentRuntime — LangGraph 原生 Agent 运行时
# ═══════════════════════════════════════════════════════════════

class AgentRuntime:
    """LangGraph 原生 ReAct Agent 运行时。

    对比 agent_rag.py 的 AgentRAG:
    - AgentRAG._run_agent_loop(): 手写 while 循环 (~90 行) + 手写 stream_query (~40 行)
    - AgentRuntime: create_react_agent 预建图，invoke/stream 由 LangGraph 驱动
    """

    def __init__(self, doc_processor, vectorstore=None, retriever=None,
                 graph_retriever=None, checkpointer=None):
        print("初始化 LangGraph Agent 运行时...")

        self.doc_processor = doc_processor
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.graph_retriever = graph_retriever

        # 复用现有评分器/路由器作为工具底层实现
        self.graders = initialize_graders_and_router()

        # 会话隔离的文档暂存
        self.doc_store = ThreadSafeDocStore()

        # 状态持久化（可由外部注入共享实例）
        self.checkpointer = checkpointer if checkpointer is not None else build_checkpointer()

        # 长期记忆（Milvus user_memory collection；初始化失败降级为无记忆模式）
        self.user_memory = None
        embeddings = getattr(doc_processor, "embeddings", None)
        if embeddings is not None:
            try:
                from memory.user_memory import UserMemoryStore

                self.user_memory = UserMemoryStore(embeddings)
                print("✅ 长期记忆存储初始化完成")
            except Exception as e:
                print(f"⚠️ 长期记忆初始化失败，降级为无记忆模式: {e}")

        # 构建 create_react_agent 预建图
        # pre_model_hook: 工作记忆压缩（消息总长超阈值时压缩早期历史为摘要）
        tools = build_rag_tools(doc_processor, self.graders, self.doc_store)
        llm = create_chat_model(temperature=0.0)
        self.graph = create_react_agent(
            llm,
            tools,
            prompt=RAG_AGENT_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            pre_model_hook=self._build_compress_hook(),
        )

        print("✅ LangGraph Agent 运行时初始化完成")

    # ── 工作记忆压缩 hook ──

    @staticmethod
    def _build_compress_hook():
        """构建 pre_model_hook: 消息总长超阈值时，将早期历史压缩为摘要。

        保留最近 CONTEXT_COMPRESS_KEEP_RECENT 条消息，更早的用轻量模型
        压缩为一条摘要 SystemMessage，防止长任务撑爆上下文。
        """
        from langchain_core.messages import RemoveMessage, SystemMessage
        from langgraph.graph.message import REMOVE_ALL_MESSAGES

        def compress_hook(state: dict) -> dict:
            messages = state.get("messages", [])
            total_chars = sum(len(str(m.content or "")) for m in messages)
            if total_chars < CONTEXT_COMPRESS_THRESHOLD:
                return {}  # 未超阈值，不修改状态

            keep = CONTEXT_COMPRESS_KEEP_RECENT
            if len(messages) <= keep + 2:
                return {}  # 消息条数太少，压缩无意义

            old_messages = messages[:-keep]
            recent = messages[-keep:]
            old_text = "\n".join(
                f"{m.type}: {str(m.content)[:500]}" for m in old_messages
            )
            try:
                summary_llm = create_chat_model(temperature=0.0, light=True)
                summary = summary_llm.invoke(
                    "请将以下对话历史压缩为简洁摘要，保留关键事实、已确认结论"
                    f"和用户意图（500 字以内）：\n{old_text[:8000]}"
                ).content
            except Exception as e:
                print(f"⚠️ 上下文压缩失败（截断兜底）: {e}")
                summary = old_text[:1000] + "..."

            print(f"🗜️ 工作记忆压缩: {len(messages)} 条消息 → 摘要 + 最近 {keep} 条")
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    SystemMessage(content=f"[早期对话摘要] {summary}"),
                    *recent,
                ]
            }

        return compress_hook

    # ── 长期记忆读写 ──

    def _memory_prompt_section(self, question: str) -> str:
        """检索相关长期记忆并格式化（无记忆/无存储时返回空串）。"""
        if self.user_memory is None:
            return ""
        return self.user_memory.format_for_prompt(question)

    def _extract_memory_async(self, question: str, answer: str, thread_id: str):
        """对话结束后后台抽取事实记忆（不阻塞响应）。"""
        if self.user_memory is None:
            return
        import concurrent.futures

        executor = getattr(self, "_memory_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._memory_executor = executor
        executor.submit(
            self.user_memory.extract_and_store, question, answer, thread_id
        )

    # ── 调用配置 ──

    @staticmethod
    def _run_config(thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": AGENT_MAX_ITERATIONS,
        }

    # ── 查询接口 ──

    def query(self, question: str, thread_id: str = "default",
              verbose: bool = True) -> dict:
        """同步查询。thread_id 标识会话，相同 thread_id 自动接续历史。"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 [AgentRuntime] 处理: {question} (thread={thread_id})")
            print(f"{'='*60}")

        start_time = time.time()
        config = self._run_config(thread_id)

        # 注入长期记忆（作为额外 system 消息，随本轮输入进入图）
        memory_section = self._memory_prompt_section(question)
        input_messages = []
        if memory_section:
            from langchain_core.messages import SystemMessage

            input_messages.append(SystemMessage(content=memory_section.strip()))
        input_messages.append(HumanMessage(content=question))

        result = self.graph.invoke({"messages": input_messages}, config)

        elapsed = time.time() - start_time
        messages = result["messages"]
        answer = messages[-1].content if messages else ""
        tool_call_count = sum(
            len(getattr(m, "tool_calls", []) or []) for m in messages
        )

        if verbose:
            print(f"\n{'─'*40}")
            print(f"⏱️ 耗时: {elapsed:.1f}s | 工具调用: {tool_call_count} 次")
            print(f"📝 答案:\n{answer or '未能生成答案'}")
            print(f"{'='*60}")

        # 后台抽取事实记忆（不阻塞响应）
        if answer:
            self._extract_memory_async(question, answer, thread_id)

        return {
            "answer": answer,
            "messages": messages,
            "tool_call_count": tool_call_count,
            "elapsed": elapsed,
        }

    def stream_events(self, question: str, thread_id: str = "default"):
        """流式查询（updates 模式），逐步产出节点更新事件。

        事件格式与 server.py SSE 层约定:
          {"type": "tool_call", "name": ..., "args": ...}
          {"type": "tool_result", "name": ..., "preview": ...}
          {"type": "answer", "content": ...}
        """
        config = self._run_config(thread_id)
        inputs = {"messages": [HumanMessage(content=question)]}

        for chunk in self.graph.stream(inputs, config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                for msg in update.get("messages", []):
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            yield {
                                "type": "tool_call",
                                "node": node_name,
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                            }
                    elif msg.type == "tool":
                        preview = (msg.content or "")[:150].replace("\n", " ")
                        yield {
                            "type": "tool_result",
                            "node": node_name,
                            "name": getattr(msg, "name", ""),
                            "preview": preview,
                        }
                    elif msg.type == "ai" and msg.content:
                        yield {
                            "type": "answer",
                            "node": node_name,
                            "content": msg.content,
                        }

    def get_history(self, thread_id: str) -> list:
        """读取会话历史消息（来自 checkpointer）。"""
        state = self.graph.get_state(self._run_config(thread_id))
        if state and state.values:
            return state.values.get("messages", [])
        return []
