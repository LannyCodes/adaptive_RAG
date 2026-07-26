"""
RAG 工具 MCP Server
===================
以 MCP (Model Context Protocol) 标准协议暴露 9 个 RAG 工具，
核心逻辑复用 agents/rag_tool_impl.py（与进程内 LangChain 工具同一实现）。

运行: python -m mcp_servers.rag_server  (stdio 传输，由 MCP client 拉起)

关键实现点:
1. stdio 协议占用 stdout 的 binary buffer，而项目模块存在无条件 print。
   本模块在 import 项目代码前安装 stdout 代理: print → stderr，
   仅 .buffer 保留原始通道供 MCP 协议消息使用。
2. 重资源 (graders / doc_processor) 懒初始化，避免 client 握手超时。
3. 文档暂存按 session_id 隔离（跨进程无 RunnableConfig 可注入，
   session_id 作为显式工具参数，LLM 侧保持默认值即可）。
"""

import os
import sys

# 项目根目录加入 sys.path（作为独立脚本/模块运行时定位项目代码）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── stdout 代理: 保护 stdio 协议通道 ─────────────────────────
class _StdoutProxy:
    """print 写入导向 stderr；.buffer 保留原始 stdout 供 MCP 协议使用。"""

    def __init__(self, original):
        self._original = original
        self.buffer = original.buffer  # MCP SDK (stdio_server) 只使用此属性

    def write(self, s):
        return sys.stderr.write(s)

    def flush(self):
        return sys.stderr.flush()

 
if not isinstance(sys.stdout, _StdoutProxy):
    sys.stdout = _StdoutProxy(sys.stdout)


def _log(msg: str):
    print(f"[rag_mcp] {msg}", file=sys.stderr, flush=True)


# ── 重资源懒初始化 ───────────────────────────────────────────
_graders = None
_doc_processor = None
_doc_store: dict = {}  # {session_id: [Document]}


def _ensure_initialized():
    global _graders, _doc_processor
    if _graders is None:
        from routers_and_graders import initialize_graders_and_router
        _graders = initialize_graders_and_router()
        _log("评分器/路由器初始化完成")
    if _doc_processor is None:
        from document_processor import initialize_document_processor
        _doc_processor, _, _, _ = initialize_document_processor()
        _log("文档处理器初始化完成")


def _get_docs(session_id: str) -> list:
    return list(_doc_store.get(session_id, []))


def _set_docs(session_id: str, docs: list):
    _doc_store[session_id] = docs


# ── MCP Server ───────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rag-tools")

from agents import rag_tool_impl as impl


@mcp.tool()
def route_query(question: str) -> str:
    """判断用户问题应该用知识库(vectorstore)还是网络搜索(web_search)。收到问题后应首先调用。"""
    _ensure_initialized()
    return impl.route_query(_graders, question)


@mcp.tool()
def decompose_query(question: str) -> str:
    """将复杂的多跳问题分解为多个简单的子问题。对需要多步推理的问题使用。"""
    _ensure_initialized()
    return impl.decompose_query(_graders, question)


@mcp.tool()
def retrieve_from_vectorstore(
    query: str, top_k: int = 5, session_id: str = "default"
) -> str:
    """从本地知识库检索与查询相关的文档。返回文档内容和编号。
    session_id 为会话标识，保持默认值即可。"""
    _ensure_initialized()
    text, docs = impl.retrieve_from_vectorstore(_doc_processor, query, top_k)
    if docs:
        _set_docs(session_id, docs)
    return text


@mcp.tool()
def search_web(query: str, session_id: str = "default") -> str:
    """通过网络搜索获取实时信息。知识库结果不足时使用。
    session_id 为会话标识，保持默认值即可。"""
    text, docs = impl.search_web(query)
    if docs:
        _set_docs(session_id, docs)
    return text


@mcp.tool()
def grade_documents(question: str, session_id: str = "default") -> str:
    """评估当前检索到的文档与问题的相关性，过滤不相关文档。检索后应调用此工具。
    session_id 为会话标识，保持默认值即可。"""
    text, filtered = impl.grade_documents(question, _get_docs(session_id))
    if filtered:
        _set_docs(session_id, filtered)
    return text


@mcp.tool()
def rewrite_query(question: str, context_hint: str = "") -> str:
    """优化查询以获得更好的检索结果。检索质量差时使用。context_hint 可包含之前检索到的关键词。"""
    _ensure_initialized()
    return impl.rewrite_query(_graders, question, context_hint)


@mcp.tool()
def check_answerability(question: str, session_id: str = "default") -> str:
    """检查已有文档是否足够回答用户问题。返回 yes 或 no。生成答案前使用。
    session_id 为会话标识，保持默认值即可。"""
    _ensure_initialized()
    return impl.check_answerability(_graders, question, _get_docs(session_id))


@mcp.tool()
def generate_answer(question: str, session_id: str = "default") -> str:
    """基于当前检索和筛选的文档生成最终答案。仅在文档经过评分后调用。
    session_id 为会话标识，保持默认值即可。"""
    return impl.generate_answer(question, _get_docs(session_id))


@mcp.tool()
def check_hallucination(generated_answer: str, session_id: str = "default") -> str:
    """检查答案内容是否都能在文档中找到支撑。生成答案后使用。返回 yes(无幻觉) 或 no(有幻觉)。
    session_id 为会话标识，保持默认值即可。"""
    _ensure_initialized()
    return impl.check_hallucination(_graders, generated_answer, _get_docs(session_id))


if __name__ == "__main__":
    # 双传输模式:
    #   stdio (默认): 由 MCP client 以子进程拉起，适合开发调试
    #   http: 常驻服务（重资源只初始化一次），适合生产。client 用 URL 连接
    from config import MCP_RAG_PORT, MCP_TRANSPORT

    if MCP_TRANSPORT == "http":
        _log(f"RAG MCP Server 启动 (streamable-http, port={MCP_RAG_PORT})")
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = MCP_RAG_PORT
        mcp.run(transport="streamable-http")
    else:
        _log("RAG MCP Server 启动 (stdio)")
        mcp.run(transport="stdio")
