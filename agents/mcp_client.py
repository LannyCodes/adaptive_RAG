"""
MCP 客户端工具加载模块
======================
通过 langchain-mcp-adapters 的 MultiServerMCPClient 以 stdio 子进程方式
拉起本项目的 MCP servers，并将其工具转换为 LangChain BaseTool。

使用方:
- agents/research_agent.py: 加载 "rag-tools" 工具
- agents/action_agent.py:  加载 "action-tools" 工具

注意: get_tools() 为异步方法；同步场景用 asyncio.run 包装
（每个 stdio server 进程随 client 会话生命周期管理）。
"""

import asyncio
from typing import List, Optional

from config import (
    MCP_ACTION_SERVER_URL,
    MCP_RAG_SERVER_URL,
    MCP_SERVER_COMMANDS,
    MCP_TRANSPORT,
)


def build_mcp_client():
    """构建 MultiServerMCPClient。

    连接方式按 config.MCP_TRANSPORT:
    - http: 连接常驻 MCP 服务（生产，server 重资源只初始化一次）
    - stdio: 拉起子进程（开发调试；注意每次工具调用新建会话）
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if MCP_TRANSPORT == "http":
        connections = {
            "rag-tools": {"url": MCP_RAG_SERVER_URL, "transport": "http"},
            "action-tools": {"url": MCP_ACTION_SERVER_URL, "transport": "http"},
        }
    else:
        connections = {}
        for name, cmd in MCP_SERVER_COMMANDS.items():
            connections[name] = {
                "command": cmd[0],
                "args": list(cmd[1:]),
                "transport": "stdio",
            }
    return MultiServerMCPClient(connections)


async def load_mcp_tools_async(server_names: Optional[List[str]] = None) -> list:
    """异步加载 MCP 工具。

    Args:
        server_names: 只加载指定 server 的工具；None 加载全部。
    """
    client = build_mcp_client()
    if not server_names:
        return await client.get_tools()

    tools = []
    for name in server_names:
        tools.extend(await client.get_tools(server_name=name))
    return tools


def load_mcp_tools(server_names: Optional[List[str]] = None) -> list:
    """同步加载 MCP 工具（asyncio.run 包装）。"""
    return asyncio.run(load_mcp_tools_async(server_names))
