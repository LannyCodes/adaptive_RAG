"""
MCP Server 模块
===============
将系统能力以 MCP (Model Context Protocol) 标准协议暴露，供任何兼容
MCP 的 Agent/客户端调用。

Server 列表:
- rag_server:    RAG 检索工具集（9 个工具，复用 agents/rag_tool_impl.py）
- action_server: 行动型工具集（代码沙箱 / 文件读写 / 网页抓取）

运行方式 (stdio 传输，由 MCP client 以子进程拉起):
    python -m mcp_servers.rag_server
    python -m mcp_servers.action_server

注意: stdio 传输占用 stdout，所有日志必须输出到 stderr。
"""
