"""
Research Agent — 知识检索子智能体
=================================
create_react_agent + rag-tools（MCP），即现有 RAG 能力的 Agent 化封装。
在 Supervisor 架构中只接收任务描述，返回结论（上下文与 Supervisor 隔离）。
"""

from langgraph.prebuilt import create_react_agent

from agents.mcp_client import load_mcp_tools_async
from routers_and_graders import create_chat_model


RESEARCH_AGENT_PROMPT = """你是研究智能体（Research Agent），负责回答知识型问题。

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
2. 知识库路线: retrieve_from_vectorstore → grade_documents
3. 如果文档质量不好: rewrite_query → retrieve_from_vectorstore → grade_documents (最多重试2次)
4. 如果多次检索仍无好结果: 回退到 search_web
5. 生成前用 check_answerability 确认文档足够
6. 用 generate_answer 生成答案，用 check_hallucination 验证
7. 验证通过后，直接输出最终答案（不要再调用工具）

## 约束
- 每次只调用一个工具
- 检索重试不超过2次
- 如果找不到相关信息，诚实告知
- 你的输出会被 Supervisor 汇总，请输出完整、自足的答案文本"""


async def build_research_agent():
    """构建 Research Agent（MCP rag-tools，异步加载）。"""
    tools = await load_mcp_tools_async(["rag-tools"])
    llm = create_chat_model(temperature=0.0)
    return create_react_agent(llm, tools, prompt=RESEARCH_AGENT_PROMPT)
