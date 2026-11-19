"""
Agent 运行时模块
================
用 LangGraph 原生 create_react_agent + Checkpointer 替换早期的手写 ReAct 循环。

与旧版手写 ReAct 实现的关键差异:
1. ReAct 循环: 手写 while 循环 → create_react_agent 预建图（内部等价循环，但获得
   状态持久化、中断恢复、流式事件等基础设施能力）
2. 工具共享状态: 实例级 self.documents → ThreadSafeDocStore 按 thread_id 隔离，
   工具经 RunnableConfig 取会话 ID，消除并发会话间的状态污染
3. 持久化: 无 → RedisSaver（有 REDIS_URL 时）/ MemorySaver（降级）
4. 迭代上限: 硬编码 15 → config 的 recursion_limit（AGENT_MAX_ITERATIONS）

依赖版本: langgraph==0.3.34（requirements.txt 锁定）
"""

import math
import threading
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agents.skill_loader import get_rag_system_prompt
from config import (
    AGENT_MAX_ITERATIONS,
    CONTEXT_COMPRESS_KEEP_RECENT,
    CONTEXT_COMPRESS_THRESHOLD,
    ENABLE_USER_MEMORY,
    REDIS_URL,
)
from routers_and_graders import create_chat_model, initialize_graders_and_router


# ═══════════════════════════════════════════════════════════════
# System Prompt — 从 Skill 资产加载（skills/rag_query/SKILL.md）
# 提示词与代码解耦：策略调整只改 SKILL.md，不动代码
# ═══════════════════════════════════════════════════════════════

RAG_AGENT_SYSTEM_PROMPT = get_rag_system_prompt()


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
        # 知识库检索结果单独保留一份，避免后续 web_search 覆盖后丢失引用来源
        self._vector_store: Dict[str, List[Document]] = {}
        # grade_documents 过滤后的文档，用于生成干净的引用来源和检索指标
        self._graded_store: Dict[str, List[Document]] = {}
        # 本轮送进 grade_documents 的候选文档，保证评分与指标同源
        self._candidate_store: Dict[str, List[Document]] = {}
        # 单次查询的检索/评分过程指标（不进入 LangGraph checkpoint）
        self._metrics: Dict[str, Dict[str, Any]] = {}
        # 最近一次塞入通用暂存区的来源通道: vectorstore / generic
        self._last_channel: Dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _tid(config: Optional[RunnableConfig]) -> str:
        return (config or {}).get("configurable", {}).get("thread_id", "default")

    def set(self, config: Optional[RunnableConfig], docs: List[Document],
            channel: str = "generic"):
        with self._lock:
            tid = self._tid(config)
            self._store[tid] = docs
            self._last_channel[tid] = channel
            if channel == "vectorstore":
                self._vector_store[tid] = docs

    def get(self, config: Optional[RunnableConfig]) -> List[Document]:
        with self._lock:
            return list(self._store.get(self._tid(config), []))

    def get_vector(self, config: Optional[RunnableConfig]) -> List[Document]:
        """返回最近一次知识库检索的文档；未检索过则为空。"""
        with self._lock:
            return list(self._vector_store.get(self._tid(config), []))

    def set_graded(self, config: Optional[RunnableConfig], docs: List[Document]):
        """保存 grade_documents 过滤后的文档。"""
        with self._lock:
            self._graded_store[self._tid(config)] = list(docs)

    def get_graded(self, config: Optional[RunnableConfig]) -> List[Document]:
        """返回最近一次评分后保留的文档；未评分则为空。"""
        with self._lock:
            return list(self._graded_store.get(self._tid(config), []))

    def get_candidates(self, config: Optional[RunnableConfig]) -> List[Document]:
        """返回最近一次被评分的候选文档；未评分则为空。"""
        with self._lock:
            return list(self._candidate_store.get(self._tid(config), []))

    def record_retrieval(self, config: Optional[RunnableConfig], docs: List[Document],
                         latency: float, query: str, top_k: int):
        """记录一次知识库检索的耗时、文档数和来源数。"""
        with self._lock:
            metrics = self._metrics.setdefault(self._tid(config), {})
            metrics["retrieval_latency"] = metrics.get("retrieval_latency", 0.0) + latency
            metrics["retrieval_calls"] = metrics.get("retrieval_calls", 0) + 1
            metrics["retrieved_docs_count"] = len(docs)
            metrics["unique_sources"] = len({
                d.metadata.get("source", "")
                for d in docs
                if getattr(d, "metadata", None)
            })
            metrics["last_query"] = query
            metrics["top_k"] = top_k

    def get_channel(self, config: Optional[RunnableConfig]) -> str:
        with self._lock:
            return self._last_channel.get(self._tid(config), "")

    def record_grading(self, config: Optional[RunnableConfig], docs: List[Document],
                       total_docs: int, fallback: bool = False,
                       channel: str = "",
                       candidates: Optional[List[Document]] = None):
        """记录评分后保留的文档，以及是否触发了“全不相关”兜底。"""
        with self._lock:
            tid = self._tid(config)
            self._graded_store[tid] = list(docs)
            if candidates is not None:
                self._candidate_store[tid] = list(candidates)
            metrics = self._metrics.setdefault(tid, {})
            metrics["graded_relevant_count"] = len(docs)
            metrics["graded_total_docs"] = total_docs
            metrics["grader_fallback"] = fallback
            metrics["grading_channel"] = channel

    def update_metrics(self, config: Optional[RunnableConfig], **kwargs):
        with self._lock:
            self._metrics.setdefault(self._tid(config), {}).update(kwargs)

    def increment_metric(self, config: Optional[RunnableConfig], key: str, amount: int = 1):
        with self._lock:
            metrics = self._metrics.setdefault(self._tid(config), {})
            metrics[key] = metrics.get(key, 0) + amount

    def get_metrics(self, config: Optional[RunnableConfig]) -> Dict[str, Any]:
        with self._lock:
            return dict(self._metrics.get(self._tid(config), {}))

    def clear(self, config: Optional[RunnableConfig]):
        with self._lock:
            tid = self._tid(config)
            self._store.pop(tid, None)
            self._vector_store.pop(tid, None)
            self._graded_store.pop(tid, None)
            self._metrics.pop(tid, None)
            self._last_channel.pop(tid, None)


# ═══════════════════════════════════════════════════════════════
# 工具工厂 — 从早期手写 Agent 迁移的 9 个工具
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

    薄包装层: 核心逻辑在 agents/rag_tool_impl.py，
    本层只负责文档暂存（doc_store + RunnableConfig 注入 thread_id）。
    """
    from agents import rag_tool_impl as impl

    # ── 工具 1: 路由 ──
    @tool
    def route_query(question: str, config: RunnableConfig = None) -> str:
        """判断用户问题应该用知识库(vectorstore)还是网络搜索(web_search)。收到问题后应首先调用。"""
        result = impl.route_query(graders, question)
        doc_store.update_metrics(config, route=result.split(":", 1)[-1].strip())
        return result

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
        start = time.perf_counter()
        text, docs = impl.retrieve_from_vectorstore(doc_processor, query, top_k)
        latency = time.perf_counter() - start
        if docs:
            doc_store.set(config, docs, channel="vectorstore")
        doc_store.record_retrieval(config, docs, latency, query, top_k)
        return text

    # ── 工具 4: 网络搜索 ──
    @tool
    def search_web(query: str, config: RunnableConfig = None) -> str:
        """通过网络搜索获取实时信息。知识库结果不足时使用。"""
        doc_store.increment_metric(config, "web_search_calls")
        text, docs = impl.search_web(query)
        if docs:
            doc_store.set(config, docs)
        return text

    # ── 工具 5: 文档相关性评分 ──
    @tool
    def grade_documents(question: str, config: RunnableConfig = None) -> str:
        """评估当前检索到的文档与问题的相关性，过滤不相关文档。检索后应调用此工 具。"""
        docs = doc_store.get(config)
        source_channel = doc_store.get_channel(config)
        text, filtered, grade_meta = impl.grade_documents(question, docs)
        if filtered:
            # 保留原始来源通道，避免第二次评分把 vectorstore 误标成 generic
            doc_store.set(config, filtered, channel=source_channel or "generic")
        doc_store.record_grading(
            config, filtered, len(docs),
            fallback=bool(grade_meta.get("fallback", False)),
            channel=source_channel,
            candidates=docs,
        )
        return text

    # ── 工具 6: 查询重写 ──
    @tool
    def rewrite_query(question: str, context_hint: str = "",
                      config: RunnableConfig = None) -> str:
        """优化查询以获得更好的检索结果。检索质量差时使用。context_hint 可包含之前检索到的关键词。"""
        doc_store.increment_metric(config, "rewrite_calls")
        return impl.rewrite_query(graders, question, context_hint)

    # ── 工具 7: 可回答性检查 ──
    @tool
    def check_answerability(question: str, config: RunnableConfig = None) -> str:
        """检查已有文档是否足够回答用户问题。返回 yes 或 no。生成答案前使用。"""
        result = impl.check_answerability(graders, question, doc_store.get(config))
        doc_store.update_metrics(config, answerability=result)
        return result

    # ── 工具 8: 答案生成 ──
    @tool
    def generate_answer(question: str, config: RunnableConfig = None) -> str:
        """基于当前检索和筛选的文档生成最终答案。仅在文档经过评分后调用。"""
        start = time.perf_counter()
        answer = impl.generate_answer(question, doc_store.get(config))
        doc_store.update_metrics(config, generation_latency=time.perf_counter() - start)
        return answer

    # ── 工具 9: 幻觉检查 ──
    @tool
    def check_hallucination(generated_answer: str, config: RunnableConfig = None) -> str:
        """检查答案内容是否都能在文档中找到支撑。生成答案后使用。返回 yes(无幻觉) 或 no(有幻觉)。"""
        result = impl.check_hallucination(graders, generated_answer, doc_store.get(config))
        doc_store.update_metrics(config, hallucination_check=result)
        return result

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

    相对早期手写 ReAct 实现的差异:
    - 旧实现: 手写 while 循环 (~90 行) + 手写 stream_query (~40 行)
    - 当前实现: create_react_agent 预建图，invoke/stream 由 LangGraph 驱动
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
        if ENABLE_USER_MEMORY:
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

    # ── 步数耗尽兜底 ──
    #
    # create_react_agent 在 recursion_limit 用尽时不抛 GraphRecursionError，
    # 而是直接把固定英文占位当作最终答案返回，所以无法用 try/except 捕获。

    _STEP_EXHAUSTED_MARKER = "Sorry, need more steps to process this request."

    @staticmethod
    def _content_to_text(content) -> str:
        """把消息 content 统一成纯文本。

        多模态/分段内容的 content 是 list，元素可能是 {"type": "text", "text": ...}，
        直接 str() 会拼出字典 repr，必须取出 text 字段。
        """
        if not isinstance(content, list):
            return str(content)
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return " ".join(parts)

    @classmethod
    def _is_step_exhausted(cls, messages: list) -> bool:
        """判断本轮是否因递归上限中断。

        两种信号任一成立即视为耗尽:
        1. 最后一条 AI 消息是 LangGraph 的固定占位文案
        2. 最后一条 AI 消息仍挂着未执行的 tool_calls，说明模型刚发出调用、图就被截断
        """
        if not messages:
            return False
        last = messages[-1]
        if getattr(last, "type", "") != "ai":
            return False
        if getattr(last, "tool_calls", None):
            return True
        content = getattr(last, "content", "")
        return cls._content_to_text(content).strip().startswith(cls._STEP_EXHAUSTED_MARKER)

    def _fallback_answer(self, question: str, config: Optional[RunnableConfig]) -> str:
        """用已检索到的文档直接生成答案，替换掉无信息的英文占位。

        优先用评分后保留的文档，其次用原始检索结果，最后用通用暂存区。
        没有任何文档时不编造内容，只告诉用户可做什么。
        """
        from agents import rag_tool_impl as impl

        docs = (
            self.doc_store.get_graded(config)
            or self.doc_store.get_vector(config)
            or self.doc_store.get(config)
        )
        if not docs:
            return (
                "抱歉，这个问题需要的推理步骤超出了当前上限，"
                "且没有检索到可用的知识库资料。建议换一种问法或缩小问题范围。"
            )
        try:
            answer = impl.generate_answer(question, docs, prefix=False)
        except Exception as e:
            print(f"⚠️ 步数耗尽兜底生成失败: {e}")
            return "抱歉，这个问题需要的推理步骤超出了当前上限，请稍后重试或换个问法。"
        return (
            f"{answer}\n\n（说明：本次推理步数达到上限，"
            f"以上答案基于已检索到的 {len(docs)} 篇资料生成）"
        )

    # ── 检索指标 ──

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """用文档内容前缀作为去重/匹配键，与 retrieval_evaluation.py 保持一致。"""
        return getattr(doc, "page_content", str(doc))[:200]

    @classmethod
    def _ranking_metrics(cls, retrieved_docs: List[Document],
                         relevant_docs: List[Document]) -> dict:
        """用 Agent 的评分结果作为伪相关标签计算排序指标。

        注意：这不是人工标注 ground truth 上的 Precision/Recall，而是
        “Agent 认为相关的文档，是否排在检索结果前列”的自洽评估。
        """
        metric_keys = (
            "precision_at_1", "precision_at_3", "precision_at_5",
            "recall_at_1", "recall_at_3", "recall_at_5",
            "map_score", "mrr",
            "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
        )
        if not retrieved_docs or not relevant_docs:
            return {key: 0.0 for key in metric_keys}

        relevant_ids = {cls._doc_key(d) for d in relevant_docs}
        if not relevant_ids:
            return {key: 0.0 for key in metric_keys}

        def hits_at(k: int) -> int:
            return sum(
                1 for doc in retrieved_docs[:k]
                if cls._doc_key(doc) in relevant_ids
            )

        precision = {k: hits_at(k) / min(k, len(retrieved_docs)) for k in (1, 3, 5)}
        recall = {k: hits_at(k) / len(relevant_ids) for k in (1, 3, 5)}

        # 二值相关 NDCG：DCG 按位置折损，IDCG 为理想排序（相关文档全部提前）的上界
        def ndcg_at(k: int) -> float:
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, doc in enumerate(retrieved_docs[:k], 1)
                if cls._doc_key(doc) in relevant_ids
            )
            ideal_hits = min(k, len(relevant_ids))
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            return dcg / idcg if idcg > 0 else 0.0

        # Average Precision：每个命中位置的 precision 取平均
        hit_count = 0
        precision_sum = 0.0
        for rank, doc in enumerate(retrieved_docs, 1):
            if cls._doc_key(doc) in relevant_ids:
                hit_count += 1
                precision_sum += hit_count / rank
        map_score = precision_sum / len(relevant_ids)

        # Mean Reciprocal Rank：第一个相关文档的倒数排名
        mrr = 0.0
        for rank, doc in enumerate(retrieved_docs, 1):
            if cls._doc_key(doc) in relevant_ids:
                mrr = 1.0 / rank
                break

        return {
            "precision_at_1": precision[1],
            "precision_at_3": precision[3],
            "precision_at_5": precision[5],
            "recall_at_1": recall[1],
            "recall_at_3": recall[3],
            "recall_at_5": recall[5],
            "map_score": map_score,
            "mrr": mrr,
            "ndcg_at_1": ndcg_at(1),
            "ndcg_at_3": ndcg_at(3),
            "ndcg_at_5": ndcg_at(5),
        }

    def _build_retrieval_metrics(self, config: RunnableConfig,
                                 tool_call_count: int) -> dict:
        """汇总本次 Agent 查询的真实检索过程指标。"""
        raw_docs = self.doc_store.get_vector(config)
        graded_docs = self.doc_store.get_graded(config)
        candidates = self.doc_store.get_candidates(config)
        trace = self.doc_store.get_metrics(config)
        # Agent 可能重写查询后再检索一次，此时 get_vector 只剩最后一次结果，
        # 而 graded 来自更早一次评分。指标必须用“被评分的那批候选”作基准，
        # 否则 graded 与 retrieved 不同源，命中数会被误算成 0。
        basis_docs = candidates or raw_docs
        fallback = bool(trace.get("grader_fallback", False))
        grading_channel = trace.get("grading_channel", "")

        # 评分兜底意味着“所有候选都被判为不相关”。此时不能用兜底保留的全量
        # 文档当相关标签——那会让 P@k/MAP/NDCG 恒等于 1.0，把最差的样本
        # 美化成满分。传空相关集让指标归零，由汇总层决定是否计入均值。
        ranking = self._ranking_metrics(basis_docs, [] if fallback else graded_docs)
        if grading_channel == "vectorstore" and graded_docs and not fallback:
            metric_source = "agent_graded"
        elif grading_channel == "vectorstore" and graded_docs and fallback:
            metric_source = "agent_graded_fallback"
        elif grading_channel == "generic":
            # 走的是 web_search，不能拿网络结果去算知识库排序指标
            metric_source = "web_search"
            ranking = {key: 0.0 for key in ranking}
        else:
            metric_source = "unavailable"

        if metric_source == "agent_graded":
            note = "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
        elif metric_source == "agent_graded_fallback":
            note = "评分兜底保留了全部文档，Precision/Recall/MAP 仅供参考"
        elif metric_source == "web_search":
            note = "本次回答使用 web_search，不计算知识库 Precision/Recall/MAP"
        else:
            note = "本次 Agent 未完成文档评分，Precision/Recall/MAP 不可用"

        return {
            **ranking,
            "latency": round(float(trace.get("retrieval_latency", 0.0)), 6),
            "retrieved_docs_count": len(basis_docs),
            "last_retrieval_docs": trace.get("retrieved_docs_count", len(raw_docs)),
            "graded_relevant_count": len(graded_docs),
            "graded_total_docs": trace.get("graded_total_docs", len(raw_docs)),
            "retrieval_calls": trace.get("retrieval_calls", 0),
            "unique_sources": trace.get("unique_sources", 0),
            "route": trace.get("route"),
            "answerability": trace.get("answerability"),
            "web_search_calls": trace.get("web_search_calls", 0),
            "rewrite_calls": trace.get("rewrite_calls", 0),
            "tool_call_count": tool_call_count,
            "grader_fallback": fallback,
            "generation_latency": round(float(trace.get("generation_latency", 0.0)), 6),
            "hallucination_check": trace.get("hallucination_check"),
            "metric_source": metric_source,
            "ground_truth": False,
            "step_exhausted": False,
            "evaluation_note": note,
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
        # 清理本 thread 上一轮的文档/指标暂存，避免引用或指标串轮
        self.doc_store.clear(config)

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

        # 步数耗尽兜底：LangGraph 不抛异常，只把固定英文占位当作最终答案返回。
        # 直接用它会让用户拿到一句无信息的话，因此改用已检索文档重新生成。
        step_exhausted = self._is_step_exhausted(messages)
        if step_exhausted:
            answer = self._fallback_answer(question, config)
            elapsed = time.time() - start_time
        else:
            answer = self._content_to_text(messages[-1].content) if messages else ""

        tool_call_count = sum(
            len(getattr(m, "tool_calls", []) or []) for m in messages
        )
        retrieval_metrics = self._build_retrieval_metrics(config, tool_call_count)
        if step_exhausted:
            retrieval_metrics["step_exhausted"] = True

        if verbose:
            print(f"\n{'─'*40}")
            print(f"⏱️ 耗时: {elapsed:.1f}s | 工具调用: {tool_call_count} 次")
            if step_exhausted:
                print("⚠️ 推理步数达到上限，已用已检索文档兜底生成")
            print(f"📝 答案:\n{answer or '未能生成答案'}")
            print(f"{'='*60}")

        # 后台抽取事实记忆（不阻塞响应）；兜底答案是降级产物，不参与抽取
        if answer and not step_exhausted:
            self._extract_memory_async(question, answer, thread_id)

        # 引用来源优先使用评分后保留的文档，避免把 top-k 中的无关文档
        # 全部当成最终引用；没有评分结果时再回退到原始检索/通用暂存区。
        source_documents = self.doc_store.get_graded(config)
        if not source_documents:
            source_documents = self.doc_store.get_vector(config)
        if not source_documents:
            source_documents = self.doc_store.get(config)

        return {
            "answer": answer,
            "messages": messages,
            "tool_call_count": tool_call_count,
            "elapsed": elapsed,
            "source_documents": source_documents,
            "retrieval_metrics": retrieval_metrics,
        }

    def stream_events(self, question: str, thread_id: str = "default"):
        """流式查询（updates 模式），逐步产出节点更新事件。

        事件格式与 server.py SSE 层约定:
          {"type": "tool_call", "name": ..., "args": ...}
          {"type": "tool_result", "name": ..., "preview": ...}
          {"type": "answer", "content": ...}
        """
        config = self._run_config(thread_id)
        # 清理本 thread 上一轮的文档/指标暂存，与 query() 保持一致
        self.doc_store.clear(config)
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
                        text = self._content_to_text(msg.content)
                        # 步数耗尽时 LangGraph 用固定英文占位冒充答案，不下发给前端
                        if text.strip().startswith(self._STEP_EXHAUSTED_MARKER):
                            continue
                        yield {
                            "type": "answer",
                            "node": node_name,
                            "content": text,
                        }

        # 图被递归上限截断时，用已检索文档补一条可用答案，避免前端停在半截
        state = self.graph.get_state(config)
        messages = state.values.get("messages", []) if state and state.values else []
        if self._is_step_exhausted(messages):
            yield {
                "type": "answer",
                "node": "step_exhausted_fallback",
                "content": self._fallback_answer(question, config),
            }

    def get_history(self, thread_id: str) -> list:
        """读取会话历史消息（来自 checkpointer）。"""
        state = self.graph.get_state(self._run_config(thread_id))
        if state and state.values:
            return state.values.get("messages", [])
        return []
