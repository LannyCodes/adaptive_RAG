"""
Action Agent — 行动型子智能体
=============================
create_react_agent + action-tools（MCP）: 代码沙箱 / 文件读写 / 网页抓取。
在 Supervisor 架构中只接收任务描述，返回执行结果（上下文与 Supervisor 隔离）。
"""

from langgraph.prebuilt import create_react_agent

from agents.mcp_client import load_mcp_tools_async
from routers_and_graders import create_chat_model


ACTION_AGENT_PROMPT = """你是行动智能体（Action Agent），负责执行具体的操作型任务。

## 可用工具
- run_python_code: 在沙箱中执行 Python 代码（禁网络/subprocess/os.system，超时 30s）
- read_file: 读取工作目录内文件
- write_file: 写入工作目录内文件
- list_files: 列出工作目录内容
- browse_webpage: 用浏览器抓取网页正文

## 策略指南
1. 数据分析/计算类任务: 优先 run_python_code
2. 代码出错时: 分析错误信息，修正后重试（最多 3 次）
3. 需要保存中间结果时: write_file 写入工作目录，后续 read_file 读取
4. 需要网页详情时: browse_webpage
5. 任务完成后: 输出清晰的执行结果总结

## 约束
- 每次只调用一个工具
- 代码执行重试不超过 3 次
- 不要尝试绕过沙箱限制（禁网/禁 subprocess 是安全底线）
- 你的输出会被 Supervisor 汇总，请输出完整、自足的结果文本"""


async def build_action_agent():
    """构建 Action Agent（MCP action-tools，异步加载）。"""
    tools = await load_mcp_tools_async(["action-tools"])
    llm = create_chat_model(temperature=0.0)
    return create_react_agent(llm, tools, prompt=ACTION_AGENT_PROMPT)
