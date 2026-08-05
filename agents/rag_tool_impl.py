"""
RAG 工具纯函数实现层
====================
9 个 RAG 能力的核心逻辑，以普通函数形式实现，
由 agents/runtime.py 包装为 LangChain @tool（进程内 ReAct Agent 使用）。

设计约定:
- 函数不持有状态；检索到的文档通过返回值显式传递（调用方负责暂存）
- 返回 tuple[str, list] 的函数: (展示给 LLM 的文本, 需暂存的文档列表)
- 返回 str 的函数: 纯文本观察结果
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config import WEB_SEARCH_RESULTS_COUNT
from guardrails import scan_external_content


# ── 1. 路由 ──

def route_query(graders, question: str) -> str:
    """判断问题应走知识库(vectorstore)还是网络搜索(web_search)。"""
    result = graders["query_router"].route(question)
    return f"路由决策: {result}"


# ── 2. 查询分解 ──

def decompose_query(graders, question: str) -> str:
    """将复杂多跳问题分解为子问题。"""
    sub_queries = graders["query_decomposer"].decompose(question)
    lines = [f"  {i}. {sq}" for i, sq in enumerate(sub_queries, 1)]
    return f"分解为 {len(sub_queries)} 个子问题:\n" + "\n".join(lines)


# ── 3. 知识库检索 ──

def retrieve_from_vectorstore(
    doc_processor, query: str, top_k: int = 5
) -> Tuple[str, List[Document]]:
    """从本地知识库检索文档，返回 (文本, 文档列表)。"""
    try:
        docs = doc_processor.enhanced_retrieve(
            query, top_k=top_k, use_query_expansion=False
        )
    except Exception:
        docs = doc_processor.retriever.invoke(query)

    if not docs:
        return "未检索到任何相关文档。", []

    # 外部内容注入扫描: 过滤含注入指令的文档
    doc_texts = [getattr(d, "page_content", str(d)) for d in docs]
    _, blocked_count, blocked_snippets = scan_external_content(doc_texts)
    if blocked_count > 0:
        clean_texts, _, _ = scan_external_content(doc_texts)
        clean_set = set(clean_texts)
        docs = [d for d in docs if getattr(d, "page_content", str(d)) in clean_set]
        print(f"⚠️ [安全] 知识库检索: 过滤 {blocked_count} 篇含疑似注入的文档")
        if not docs:
            return "检索到的文档因安全原因被过滤，无可用内容。", []

    parts = []
    for i, doc in enumerate(docs):
        content = getattr(doc, "page_content", str(doc))
        truncated = content[:800] + "..." if len(content) > 800 else content
        parts.append(f"[文档{i}] {truncated}")

    return f"检索到 {len(docs)} 个文档:\n\n" + "\n\n".join(parts), docs


# ── 4. 网络搜索 ──

def search_web(query: str) -> Tuple[str, List[Document]]:
    """Tavily 网络搜索，返回 (文本, 文档列表)。"""
    from langchain_tavily import TavilySearch

    web_search = TavilySearch(k=WEB_SEARCH_RESULTS_COUNT)
    try:
        docs = web_search.invoke({"query": query})

        # 解析搜索结果为独立条目
        items: List[str] = []
        if isinstance(docs, list) and len(docs) > 0:
            first = docs[0]
            if isinstance(first, str):
                items = list(docs)
            elif isinstance(first, dict) and "content" in first:
                items = [d.get("content", str(d)) for d in docs]
            else:
                items = [str(d) for d in docs]
        else:
            content = str(docs) if not isinstance(docs, list) else "无结果"
            return f"网络搜索结果:\n{content[:2000]}", [Document(page_content=content)]

        # 外部内容注入扫描: 过滤含注入指令的搜索结果
        clean_items, blocked_count, _ = scan_external_content(items)
        if blocked_count > 0:
            print(f"⚠️ [安全] 网络搜索: 过滤 {blocked_count} 条含疑似注入的结果")
        if not clean_items:
            return "网络搜索结果因安全原因被过滤，无可用内容。", []

        content = "\n".join(clean_items)
        warning = f"\n[安全] 已过滤 {blocked_count} 条含疑似注入的搜索结果\n" if blocked_count > 0 else ""
        return f"网络搜索结果:\n{warning}{content[:2000]}", [Document(page_content=content)]
    except Exception as e:
        return f"网络搜索失败: {e}", []


# ── 5. 文档相关性评分 ──

def grade_documents(
    question: str, docs: List[Document]
) -> Tuple[str, List[Document]]:
    """批量评估文档相关性并过滤，返回 (文本, 过滤后文档)。"""
    if not docs:
        return "没有文档可供评分。请先调用 retrieve_from_vectorstore 或 search_web。", []

    from langchain_core.output_parsers import JsonOutputParser
    from prompt_manager import get_prompt_manager
    from routers_and_graders import create_chat_model

    docs_text = ""
    for i, doc in enumerate(docs):
        c = doc.page_content[:500] if hasattr(doc, "page_content") else str(doc)[:500]
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

    return f"评分结果 ({len(filtered)}/{len(docs)} 相关):\n" + "\n".join(lines), filtered


# ── 6. 查询重写 ──

def rewrite_query(graders, question: str, context_hint: str = "") -> str:
    """优化查询以获得更好的检索结果。"""
    better = graders["query_rewriter"].rewrite(question, context=context_hint)
    return f"优化后查询: {better}"


# ── 7. 可回答性检查 ──

def check_answerability(graders, question: str, docs: List[Document]) -> str:
    """检查现有文档是否足以回答问题。"""
    if not docs:
        return "无可评估文档，请先检索。"

    contents = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    context = "\n---\n".join(contents)[:5000]
    score = graders["answerability_grader"].grade(question, context)
    return f"可回答性: {'足够' if score == 'yes' else '不足，需要更多信息'}"


# ── 8. 答案生成 ──

def generate_answer(question: str, docs: List[Document]) -> str:
    """基于筛选后的文档生成最终答案。"""
    if not docs:
        return "错误: 没有文档可生成答案。"

    from langchain_core.output_parsers import StrOutputParser
    from prompt_manager import get_prompt_manager
    from routers_and_graders import create_chat_model

    context = "\n\n".join(
        d.page_content if hasattr(d, "page_content") else str(d) for d in docs
    )
    rag_prompt = get_prompt_manager().get_template("generate_answer")
    gen_llm = create_chat_model(temperature=0.3)
    chain = rag_prompt | gen_llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return f"生成答案:\n\n{answer}"


# ── 9. 幻觉检查 ──

def check_hallucination(
    graders, generated_answer: str, docs: List[Document]
) -> str:
    """检查答案是否都有文档支撑。"""
    if not docs:
        return "无参考文档，无法检查。"
    score = graders["hallucination_grader"].grade(generated_answer, docs)
    if score == "yes":
        return "通过: 答案有文档支撑，可以输出给用户。"
    return "未通过: 答案存在文档不能支撑的内容。应改写查询重新检索。"
