"""
Agent-based RAG System
=======================
将固定的 LangGraph 工作流转换为动态 Agent 架构。

架构对比:
  旧 (workflow): 预定义 DAG，硬编码边
    START → route_and_decompose → retrieve/web_search → grade_docs
          → decide_to_generate → (generate | transform_query | prepare_next_query)
          → grade_hallucination → END

  新 (agent): Agent 自主决定工具调用序列
    Agent 循环: LLM思考 → 选择工具 → 执行工具 → 观察结果 → 继续思考 → ... → 最终答案

核心区别:
  - 工作流: 每一步去哪个节点是代码写死的 (workflow.add_edge / add_conditional_edges)
  - Agent:  每一步调用哪个工具是 LLM 推理后自主决定的 (ReAct loop)
  - Agent 可以跳过步骤、改变顺序、根据中间结果动态调整策略
"""

import json
import time
from typing import List, Dict, Any

from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.documents import Document

from config import LLM_BACKEND
from routers_and_graders import create_chat_model, initialize_graders_and_router


# ═══════════════════════════════════════════════════════════════
# System Prompt — Agent 的"大脑"，编码 RAG 策略为自然语言指令
# 工作流版本中这些策略是写死在 conditional_edges 中的
# 从 Skill 资产加载（skills/rag_query/SKILL.md），与 agents/runtime.py 共享同一份
# ═══════════════════════════════════════════════════════════════

from agents.skill_loader import get_rag_system_prompt

RAG_AGENT_SYSTEM_PROMPT = get_rag_system_prompt()


# ═══════════════════════════════════════════════════════════════
# AgentRAG — Agent 版本的 RAG 系统
# ═══════════════════════════════════════════════════════════════

class AgentRAG:
    """
    Agent-based RAG 系统

    对比 main.py 的 AdaptiveRAGSystem:
    - AdaptiveRAGSystem._build_workflow():
      用 add_node + add_edge + add_conditional_edges 构建固定 DAG
      LLM 只在各节点内部做判断(路由/评分)，大流程由代码控制
      ≈ 60 行硬编码的图定义 + WorkflowNodes 类 1000+ 行

    - AgentRAG:
      所有能力暴露为 tool，LLM 自主选择调用顺序
      大流程由 LLM 推理决定，不是代码写死的
      ≈ 工具定义 + ReAct 循环，核心逻辑 ~100 行
    """

    def __init__(self, doc_processor, vectorstore, retriever, graph_retriever=None):
        print("初始化 Agent-based RAG 系统...")

        self.doc_processor = doc_processor
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.graph_retriever = graph_retriever
        self.documents: List[Document] = []

        # 复用现有的评分器/路由器作为工具的底层实现
        self.graders = initialize_graders_and_router()

        # Agent LLM — 需要支持 tool calling
        self.agent_llm = create_chat_model(temperature=0.0)

        # 构建工具列表 (openai 格式的 tool schema，bind_tools 需要)
        self._raw_tools, self._tool_schemas = self._build_tools()

        # 将工具绑定到 LLM（LLM 会返回 tool_calls 而不是直接生成文本）
        self.llm_with_tools = self.agent_llm.bind_tools(self._tool_schemas)

        print("✅ Agent-based RAG 系统初始化完成")

    # ── 工具定义 ──────────────────────────────────────────────
    #
    # 每个工具做一件事，由 Agent 决策何时调用。
    # 工具函数本身是"哑"的——它们不包含任何流程逻辑，
    # 只是执行操作并返回观察结果。
    #
    # 对比 workflow_nodes.py:
    #   每个节点函数 (~50-100行) 包含了执行逻辑 + 流程决策逻辑
    #   工具函数 (~15行) 只包含执行逻辑，决策权交还给 Agent

    def _build_tools(self):
        """将所有 RAG 能力包装为工具"""
        doc_processor = self.doc_processor
        graders = self.graders
        self_ref = self  # 工具函数内部需要通过 self 访问共享状态
        graph_retriever = self.graph_retriever

        # ── 工具 1: 路由 ──
        @tool
        def route_query(question: str) -> str:
            """判断用户问题应该用知识库(vectorstore)还是网络搜索(web_search)。收到问题后应首先调用。"""
            result = graders["query_router"].route(question)
            return f"路由决策: {result}"

        # ── 工具 2: 查询分解 ──
        @tool
        def decompose_query(question: str) -> str:
            """将复杂的多跳问题分解为多个简单的子问题。对需要多步推理的问题使用。"""
            sub_queries = graders["query_decomposer"].decompose(question)
            lines = [f"  {i}. {sq}" for i, sq in enumerate(sub_queries, 1)]
            return f"分解为 {len(sub_queries)} 个子问题:\n" + "\n".join(lines)

        # ── 工具 3: 知识库检索 ──
        @tool
        def retrieve_from_vectorstore(query: str, top_k: int = 5) -> str:
            """从本地知识库检索与查询相关的文档。返回文档内容和编号。"""
            try:
                docs = doc_processor.enhanced_retrieve(
                    query, top_k=top_k, use_query_expansion=False
                )
            except Exception:
                docs = doc_processor.retriever.invoke(query)

            if not docs:
                return "未检索到任何相关文档。"

            self_ref.documents = docs

            parts = []
            for i, doc in enumerate(docs):
                content = getattr(doc, 'page_content', str(doc))
                truncated = content[:800] + "..." if len(content) > 800 else content
                parts.append(f"[文档{i}] {truncated}")

            return f"检索到 {len(docs)} 个文档:\n\n" + "\n\n".join(parts)

        # ── 工具 4: 网络搜索 ──
        @tool
        def search_web(query: str) -> str:
            """通过网络搜索获取实时信息。知识库结果不足时使用。"""
            from langchain_tavily import TavilySearch
            from config import WEB_SEARCH_RESULTS_COUNT

            web_search = TavilySearch(k=WEB_SEARCH_RESULTS_COUNT)
            try:
                docs = web_search.invoke({"query": query})
                if isinstance(docs, list) and len(docs) > 0:
                    first = docs[0]
                    if isinstance(first, str):
                        content = "\n".join(docs)
                    elif isinstance(first, dict) and "content" in first:
                        content = "\n".join(d.get("content", str(d)) for d in docs)
                    else:
                        content = "\n".join(str(d) for d in docs)
                else:
                    content = str(docs) if not isinstance(docs, list) else "无结果"

                self_ref.documents = [Document(page_content=content)]
                return f"网络搜索结果:\n{content[:2000]}"
            except Exception as e:
                return f"网络搜索失败: {e}"

        # ── 工具 5: 文档相关性评分 ──
        @tool
        def grade_documents(question: str) -> str:
            """评估当前检索到的文档与问题的相关性，过滤不相关文档。检索后应调用此工具。"""
            docs = self_ref.documents
            if not docs:
                return "没有文档可供评分。请先调用 retrieve_from_vectorstore 或 search_web。"

            from langchain_core.output_parsers import JsonOutputParser
            from prompt_manager import get_prompt_manager

            docs_text = ""
            for i, doc in enumerate(docs):
                c = doc.page_content[:500] if hasattr(doc, 'page_content') else str(doc)[:500]
                docs_text += f"\n---文档{i+1}---\n{c}\n"

            batch_prompt = get_prompt_manager().get_template("grade_documents_batch")
            batch_llm = create_chat_model(format="json", temperature=0.0)
            chain = batch_prompt | batch_llm | JsonOutputParser()
            result = chain.invoke({
                "question": question,
                "documents": docs_text,
                "doc_count": len(docs),
                "retry_hint": "",
            })
            scores = result.get("scores", [])

            filtered = []
            lines = []
            for i, (doc, score) in enumerate(zip(docs, scores)):
                status = "相关" if score == "yes" else "不相关"
                lines.append(f"  文档{i+1}: {status}")
                if score == "yes":
                    filtered.append(doc)

            if not filtered and docs:
                filtered = list(docs)
                lines.append("  (兜底: 保留全部文档)")

            self_ref.documents = filtered
            return f"评分结果 ({len(filtered)}/{len(docs)} 相关):\n" + "\n".join(lines)

        # ── 工具 6: 查询重写 ──
        @tool
        def rewrite_query(question: str, context_hint: str = "") -> str:
            """优化查询以获得更好的检索结果。检索质量差时使用。context_hint 可包含之前检索到的关键词。"""
            better = graders["query_rewriter"].rewrite(question, context=context_hint)
            return f"优化后查询: {better}"

        # ── 工具 7: 可回答性检查 ──
        @tool
        def check_answerability(question: str) -> str:
            """检查已有文档是否足够回答用户问题。返回 yes 或 no。生成答案前使用。"""
            docs = self_ref.documents
            if not docs:
                return "无可评估文档，请先检索。"

            contents = [d.page_content if hasattr(d, 'page_content') else str(d) for d in docs]
            context = "\n---\n".join(contents)[:5000]
            score = graders["answerability_grader"].grade(question, context)
            return f"可回答性: {'足够' if score == 'yes' else '不足，需要更多信息'}"

        # ── 工具 8: 答案生成 ──
        @tool
        def generate_answer(question: str) -> str:
            """基于当前检索和筛选的文档生成最终答案。仅在文档经过评分后调用。"""
            docs = self_ref.documents
            if not docs:
                return "错误: 没有文档可生成答案。"

            from langchain_core.output_parsers import StrOutputParser
            from prompt_manager import get_prompt_manager

            context = "\n\n".join(
                d.page_content if hasattr(d, 'page_content') else str(d) for d in docs
            )
            rag_prompt = get_prompt_manager().get_template("generate_answer")
            gen_llm = create_chat_model(temperature=0.3)
            chain = rag_prompt | gen_llm | StrOutputParser()
            answer = chain.invoke({"question": question, "context": context})
            self_ref._generated_answer = answer
            return f"生成答案:\n\n{answer}"

        # ── 工具 9: 幻觉检查 ──
        @tool
        def check_hallucination(generated_answer: str) -> str:
            """检查答案内容是否都能在文档中找到支撑。生成答案后使用。返回 yes(无幻觉) 或 no(有幻觉)。"""
            docs = self_ref.documents
            if not docs:
                return "无参考文档，无法检查。"
            score = graders["hallucination_grader"].grade(generated_answer, docs)
            if score == "yes":
                return "通过: 答案有文档支撑，可以输出给用户。"
            return "未通过: 答案存在文档不能支撑的内容。应改写查询重新检索。"

        raw_tools = [
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

        # 同时生成 openai 格式的 tool schema（用于 bind_tools）
        tool_schemas = [t.get_tool_schema() for t in raw_tools]

        return raw_tools, tool_schemas

    # ── 工具执行 ──────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """根据工具名查找工具并执行"""
        for t in self._raw_tools:
            if t.name == tool_name:
                try:
                    return t.invoke(tool_args)
                except Exception as e:
                    return f"工具执行错误: {e}"
        return f"未找到工具: {tool_name}"

    # ── ReAct Agent 循环 ──────────────────────────────────────
    #
    # 这是 Agent 的核心: Thought → Action → Observation 循环
    # 工作流版本中这个循环是 LangGraph 的 graph.invoke/astream 驱动的
    # 这里是一个显式的 while 循环，每一步都由 LLM 自主决策

    def _run_agent_loop(
        self,
        question: str,
        max_iterations: int = 15,
        verbose: bool = True,
    ) -> dict:
        """
        ReAct Agent 主循环

        伪代码:
          messages = [SystemPrompt, UserQuestion]
          while iterations < max:
              response = LLM(messages)    ← LLM 决定下一步做什么
              if no tool_calls: break     ← LLM 认为可以结束了
              for each tool_call:
                  result = execute(tool)   ← 执行工具
                  messages.append(result)  ← 观察结果追加到对话
          return messages
        """
        self.documents = []
        self._generated_answer = None

        messages = [
            SystemMessage(content=RAG_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        tool_call_count = 0

        for iteration in range(max_iterations):
            # ── Thought/Action: LLM 决定下一步 ──
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            # 没有 tool_calls → LLM 认为任务完成，输出最终答案
            if not response.tool_calls:
                if verbose:
                    print(f"\n✅ Agent 完成，共调用 {tool_call_count} 次工具，"
                          f"迭代 {iteration + 1} 轮")
                return {
                    "answer": response.content,
                    "messages": messages,
                    "tool_call_count": tool_call_count,
                    "iterations": iteration + 1,
                }

            # ── Observation: 执行工具并收集结果 ──
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", "")

                if verbose:
                    args_preview = str(tool_args)
                    if len(args_preview) > 100:
                        args_preview = args_preview[:100] + "..."
                    print(f"  🔧 [{iteration+1}] {tool_name}({args_preview})")

                result_text = self._execute_tool(tool_name, tool_args)
                tool_call_count += 1

                if verbose:
                    preview = result_text[:150].replace("\n", " ")
                    print(f"     → {preview}...")

                messages.append(ToolMessage(
                    content=result_text,
                    tool_call_id=tool_id,
                ))

        # 达到最大迭代次数，强制要求 LLM 总结
        if verbose:
            print(f"\n⚠️ 达到最大迭代次数 ({max_iterations})，强制生成总结")
        messages.append(HumanMessage(
            content="已达到最大工具调用次数。请基于已有信息直接回答用户问题，"
                    "不要再调用工具。如果信息不足，请诚实告知。"
        ))
        final_response = self.llm_with_tools.invoke(messages)

        return {
            "answer": final_response.content,
            "messages": messages,
            "tool_call_count": tool_call_count,
            "iterations": max_iterations,
        }

    # ── 查询接口 ──────────────────────────────────────────────

    def query(self, question: str, verbose: bool = True) -> dict:
        """
        处理用户问题 (Agent 版本)

        与 AdaptiveRAGSystem.query() 的关键区别:
        - 旧: app.astream(inputs) → LangGraph 沿预定义 DAG 遍历节点
          (START → route_and_decompose → retrieve → grade → decide → ...)
          流程路径是代码写死的

        - 新: _run_agent_loop(question) → Agent 在 ReAct 循环中自主决定
          调用哪些工具、按什么顺序、何时停止
          流程路径是 LLM 实时推理的
        """
        print(f"\n{'='*60}")
        print(f"🔍 [AgentRAG] 处理: {question}")
        print(f"{'='*60}")

        start_time = time.time()

        result = self._run_agent_loop(question, verbose=verbose)

        elapsed = time.time() - start_time
        answer = result["answer"] or self._generated_answer

        print(f"\n{'─'*40}")
        print(f"⏱️ 耗时: {elapsed:.1f}s | 工具调用: {result['tool_call_count']} 次 | "
              f"迭代: {result['iterations']} 轮")
        print(f"📝 答案:\n{answer or '未能生成答案'}")
        print(f"{'='*60}")

        return {
            "answer": answer,
            "elapsed": elapsed,
            **result,
        }

    def stream_query(self, question: str):
        """
        流式查询 (简化版，同步)

        逐步 yield Agent 的思考和工具调用过程。
        """
        self.documents = []
        self._generated_answer = None

        yield {"type": "start", "content": f"处理: {question}"}

        messages = [
            SystemMessage(content=RAG_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        for iteration in range(15):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                yield {"type": "answer", "content": response.content}
                break

            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                yield {"type": "tool_call", "content": f"🔧 {tool_name}", "args": tool_args}

                result = self._execute_tool(tool_name, tool_args)
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tc.get("id", ""),
                ))

        yield {"type": "done", "content": "完成"}


# ═══════════════════════════════════════════════════════════════
# main.py 中的修改示例
# ═══════════════════════════════════════════════════════════════
#
# 【原版 AdaptiveRAGSystem._build_workflow() — 固定 DAG】
#
#   workflow = StateGraph(GraphState)
#   workflow.add_node("web_search", self.workflow_nodes.web_search)
#   workflow.add_node("retrieve", self.workflow_nodes.retrieve)
#   workflow.add_node("grade_documents", self.workflow_nodes.grade_documents)
#   workflow.add_node("generate", self.workflow_nodes.generate)
#   workflow.add_node("transform_query", self.workflow_nodes.transform_query)
#   workflow.add_node("route_and_decompose", self.workflow_nodes.route_and_decompose)
#   workflow.add_node("prepare_next_query", self.workflow_nodes.prepare_next_query)
#   workflow.add_conditional_edges(START, ..., {...})
#   workflow.add_edge("web_search", "generate")
#   workflow.add_edge("retrieve", "grade_documents")
#   workflow.add_conditional_edges("grade_documents", ..., {...})
#   workflow.add_edge("transform_query", "retrieve")
#   workflow.add_edge("prepare_next_query", "retrieve")
#   workflow.add_conditional_edges("generate", ..., {...})
#   self.app = workflow.compile(...)
#
# 【Agent 版 — 替代整个 _build_workflow】
#
#   self.agent_rag = AgentRAG(
#       doc_processor=self.doc_processor,
#       vectorstore=self.vectorstore,
#       retriever=self.retriever,
#   )
#
#   def query(self, question: str):
#       return self.agent_rag.query(question)
#
# 对比:
#   add_node × 7        →  工具函数 × 9
#   add_edge × 4        →  自然语言系统提示
#   add_conditional × 3  →  LLM 自主决策
#   固定 DAG             →  动态工具选择
#
# ═══════════════════════════════════════════════════════════════
