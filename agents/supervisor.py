"""
Supervisor 多智能体编排
=======================
手写 Supervisor 路由（不用 langgraph-supervisor 库，避免额外依赖的版本风险）。

架构:
    START → supervisor ─┬→ research ─→ supervisor
                        ├→ action ───→ supervisor
                        ├→ verifier ─→ supervisor
                        └→ finalize → END

设计要点:
1. Supervisor 节点用 LLM JSON 模式决策（与项目 graders 风格一致，弱模型友好），
   输出 {"next", "task", "reason"}；硬性重试上限在代码层保证（不依赖 LLM 自觉）。
2. 上下文隔离: 子 Agent 只收到任务描述（plan），不共享 Supervisor 消息历史；
   子 Agent 结论存入 intermediate_results，Supervisor 只看摘要。
3. 子 Agent 的 MCP 工具为 async，全图节点均为 async，统一 ainvoke/astream。
4. 延迟初始化: 首次调用时才构建子 Agent（MCP 工具异步加载），
   避免在非 async 上下文中构造。
"""

import asyncio
import json
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langgraph.types import Command, interrupt

from agents.action_agent import build_action_agent
from agents.research_agent import build_research_agent
from agents.runtime import build_checkpointer
from agents.verifier_agent import build_verifier
from config import AGENT_MAX_ITERATIONS, HITL_ENABLED
from guardrails import check_input, sanitize_output
from routers_and_graders import create_chat_model

# Supervisor 分派失败时的硬性重试上限（沿用项目重试惯例）
SUPERVISOR_MAX_RETRY = 2


class SupervisorState(TypedDict, total=False):
    """Supervisor 图共享状态。

    messages: Supervisor 层的决策/进度消息（含用户问题与最终答案）
    question: 原始用户问题
    plan: Supervisor 给子 Agent 的当前任务描述
    intermediate_results: {agent名: 结论文本}，_candidate 键存当前候选答案
    final_answer: 最终答案
    retry_count: 验证不通过后的重试次数
    next: Supervisor 决策（research/action/verifier/finish）
    verification: 最近一次验证结果 {"passed", "score", "reason"}
    """

    messages: Annotated[list, add_messages]
    question: str
    plan: str
    intermediate_results: dict
    final_answer: str
    retry_count: int
    next: str
    verification: dict


SUPERVISOR_SYSTEM_PROMPT = """你是多智能体系统的 Supervisor（调度者）。根据用户问题和当前进展，决定下一步把任务分派给哪个智能体。

## 可分派的智能体
- research: 知识检索智能体。处理知识型问题（知识库检索、网络搜索、资料查证）
- action: 行动智能体。处理操作型任务（执行代码计算、读写文件、抓取网页正文）
- verifier: 验证器。在已有候选答案后，验证答案是否有效回答了用户问题
- finish: 任务完成。已有验证通过的答案，或重试已达上限，输出最终答案

## 决策规则
1. 知识型问题先分派 research；需要计算/文件/网页操作时先分派 action；复杂任务可分步（先 research 再 action 或反之）
2. 获得候选答案后，必须先分派 verifier 验证，再考虑 finish
3. verifier 未通过时: 分析原因，换策略重新分派（改写任务描述/换智能体）
4. 候选答案已验证通过，或确认无法获得更好结果时: finish
5. task 字段必须是对子智能体的完整、自足的任务描述（子智能体看不到对话历史）

## 输出格式（严格 JSON，不要输出其他内容）
{"next": "research|action|verifier|finish", "task": "给子智能体的任务描述（finish 时为空串）", "reason": "一句话决策理由"}"""


def _format_progress(state: SupervisorState) -> str:
    """汇总当前进展供 Supervisor 决策。"""
    parts = [f"原始问题: {state.get('question', '')}"]
    results = state.get("intermediate_results", {})
    real_results = {k: v for k, v in results.items() if not k.startswith("_")}
    if real_results:
        parts.append("已完成的子任务:")
        for name, text in real_results.items():
            parts.append(f"  [{name}] {str(text)[:300]}")
    else:
        parts.append("已完成的子任务: (无)")
    verification = state.get("verification")
    if verification:
        parts.append(
            f"最近验证: {'通过' if verification.get('passed') else '未通过'}"
            f"（{verification.get('reason', '')}）"
        )
    parts.append(f"当前重试次数: {state.get('retry_count', 0)}/{SUPERVISOR_MAX_RETRY}")
    return "\n".join(parts)


class SupervisorAgent:
    """Supervisor 多智能体系统。"""

    def __init__(self, checkpointer=None, graders: Optional[dict] = None):
        self.checkpointer = checkpointer if checkpointer is not None else build_checkpointer()
        self._graders = graders
        self.research_agent = None
        self.action_agent = None
        self.verifier = None
        self.graph = None
        self._init_lock = asyncio.Lock()

    # ── 延迟初始化（MCP 工具异步加载）──

    async def initialize(self):
        async with self._init_lock:
            if self.graph is not None:
                return
            print("初始化 Supervisor 多智能体系统...")

            self.research_agent = await build_research_agent()
            print("  ✅ Research Agent 就绪")
            self.action_agent = await build_action_agent()
            print("  ✅ Action Agent 就绪")
            self.verifier = build_verifier(self._graders)
            print("  ✅ Verifier 就绪")

            workflow = StateGraph(SupervisorState)
            workflow.add_node("supervisor", self._supervisor_node)
            workflow.add_node("research", self._research_node)
            workflow.add_node("action", self._action_node)
            workflow.add_node("verifier", self._verifier_node)
            workflow.add_node("finalize", self._finalize_node)

            workflow.add_edge(START, "supervisor")
            workflow.add_conditional_edges(
                "supervisor",
                self._route_from_supervisor,
                {
                    "research": "research",
                    "action": "action",
                    "verifier": "verifier",
                    "finish": "finalize",
                },
            )
            workflow.add_edge("research", "supervisor")
            workflow.add_edge("action", "supervisor")
            workflow.add_edge("verifier", "supervisor")
            workflow.add_edge("finalize", END)

            self.graph = workflow.compile(checkpointer=self.checkpointer)
            print("✅ Supervisor 图编译完成")

    # ── 节点实现 ──

    async def _supervisor_node(self, state: SupervisorState) -> dict:
        progress = _format_progress(state)
        llm = create_chat_model(format="json", temperature=0.0)
        response = llm.invoke([
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": progress},
        ])
        try:
            decision = json.loads(response.content)
        except (json.JSONDecodeError, AttributeError):
            # 决策解析失败: 无子任务结果→research，有→finish
            has_result = bool(state.get("intermediate_results"))
            decision = {
                "next": "finish" if has_result else "research",
                "task": state.get("question", ""),
                "reason": "决策解析失败，走默认路径",
            }

        next_step = str(decision.get("next", "finish")).strip().lower()
        if next_step not in ("research", "action", "verifier", "finish"):
            next_step = "finish"

        # 硬性约束 1: 无候选答案时不允许直接 finish（除非重试上限）
        results = state.get("intermediate_results", {})
        has_candidate = bool(results.get("_candidate"))
        retry_count = state.get("retry_count", 0)
        if next_step == "finish" and not has_candidate and retry_count < SUPERVISOR_MAX_RETRY:
            next_step = "research"
            decision["task"] = state.get("question", "")
            decision["reason"] = "尚无候选答案，先检索"

        # 硬性约束 2: 重试达上限强制 finish
        if retry_count >= SUPERVISOR_MAX_RETRY:
            next_step = "finish"

        reason = str(decision.get("reason", ""))[:200]
        return {
            "next": next_step,
            "plan": str(decision.get("task", ""))[:2000],
            "messages": [AIMessage(content=f"[Supervisor] → {next_step}: {reason}")],
        }

    def _route_from_supervisor(self, state: SupervisorState) -> str:
        nxt = state.get("next", "finish")
        return nxt if nxt in ("research", "action", "verifier", "finish") else "finish"

    async def _run_sub_agent(self, agent, task: str, name: str,
                             state: SupervisorState) -> dict:
        """执行子 Agent 并把结论写回 intermediate_results（上下文隔离）。"""
        tool_calls: list = []
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=task)]},
                {"recursion_limit": AGENT_MAX_ITERATIONS},
            )
            answer = result["messages"][-1].content if result.get("messages") else ""
            # 提取工具调用轨迹（供轨迹评估 agent_evals.py 使用）
            for m in result.get("messages", []):
                for tc in getattr(m, "tool_calls", None) or []:
                    tool_calls.append(tc.get("name", ""))
        except Exception as e:
            answer = f"{name} 执行失败: {e}"

        results = dict(state.get("intermediate_results", {}))
        results[name] = answer
        results["_candidate"] = answer  # 最新产出即候选答案
        results[f"_trace_{name}"] = tool_calls  # 工具调用轨迹（下划线前缀，不展示）
        return {
            "intermediate_results": results,
            "messages": [AIMessage(content=f"[{name} 完成]\n{str(answer)[:500]}")],
        }

    async def _research_node(self, state: SupervisorState) -> dict:
        task = state.get("plan") or state.get("question", "")
        return await self._run_sub_agent(self.research_agent, task, "research", state)

    async def _action_node(self, state: SupervisorState) -> dict:
        task = state.get("plan") or state.get("question", "")

        # HITL: 行动任务执行前中断等待人工审批（代码/文件/网页操作的网关）
        # interrupt 在 Supervisor 图节点内（有 checkpointer + thread_id），
        # 审批后经 aresume() 以 Command(resume=decision) 恢复执行。
        if HITL_ENABLED:
            decision = interrupt({
                "type": "approval_required",
                "agent": "action",
                "task": task,
                "message": "行动智能体将执行以下任务（可能包含代码执行/文件写入/"
                           "网页抓取），是否批准？",
            })
            if decision != "approve":
                results = dict(state.get("intermediate_results", {}))
                results["action"] = "用户拒绝了该行动任务的执行"
                return {
                    "intermediate_results": results,
                    "messages": [AIMessage(content="[action] 用户拒绝执行该任务")],
                }

        return await self._run_sub_agent(self.action_agent, task, "action", state)

    async def _verifier_node(self, state: SupervisorState) -> dict:
        candidate = state.get("intermediate_results", {}).get("_candidate", "")
        verification = self.verifier(state.get("question", ""), candidate)
        retry_count = state.get("retry_count", 0)
        if not verification["passed"]:
            retry_count += 1
        status = "通过" if verification["passed"] else "未通过"
        return {
            "verification": verification,
            "retry_count": retry_count,
            "messages": [
                AIMessage(content=f"[verifier] {status}: {verification['reason']}")
            ],
        }

    async def _finalize_node(self, state: SupervisorState) -> dict:
        """确定最终答案: 验证通过的候选答案优先，否则最新子任务结果兜底。"""
        results = state.get("intermediate_results", {})
        verification = state.get("verification") or {}
        candidate = results.get("_candidate", "")

        if candidate and verification.get("passed", True):
            final = candidate
        elif candidate:
            final = candidate
        else:
            real = [v for k, v in results.items() if not k.startswith("_")]
            final = real[-1] if real else "抱歉，未能生成有效答案。"

        return {
            "final_answer": final,
            "messages": [AIMessage(content=final)],
        }

    # ── 调用接口 ──

    @staticmethod
    def _run_config(thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": AGENT_MAX_ITERATIONS * 3,  # 含 supervisor 回环
        }

    async def aquery(self, question: str, thread_id: str = "default",
                     verbose: bool = True) -> dict:
        """异步查询主接口。"""
        await self.initialize()

        # 输入护栏: 提示注入检测
        input_check = check_input(question)
        if not input_check["safe"]:
            return {
                "answer": f"输入被安全护栏拦截: {input_check['reason']}",
                "messages": [],
                "intermediate_results": {},
                "traces": {},
                "verification": None,
                "elapsed": 0.0,
                "blocked": True,
            }

        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 [Supervisor] 处理: {question} (thread={thread_id})")
            print(f"{'='*60}")

        start_time = time.time()
        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=question)],
                "question": question,
                "intermediate_results": {},
                "retry_count": 0,
            },
            self._run_config(thread_id),
        )
        elapsed = time.time() - start_time

        answer = result.get("final_answer", "")

        # HITL: 图在 interrupt 处暂停（等待审批）
        if "__interrupt__" in result:
            interrupt_info = result["__interrupt__"][0].value
            return {
                "answer": "",
                "messages": result.get("messages", []),
                "intermediate_results": {},
                "traces": {},
                "verification": None,
                "elapsed": elapsed,
                "pending_approval": interrupt_info,
            }

        # 输出护栏: 敏感信息过滤
        answer = sanitize_output(answer)

        if verbose:
            print(f"\n⏱️ 耗时: {elapsed:.1f}s")
            print(f"📝 最终答案:\n{answer}")
            print(f"{'='*60}")

        return {
            "answer": answer,
            "messages": result.get("messages", []),
            "intermediate_results": {
                k: v for k, v in result.get("intermediate_results", {}).items()
                if not k.startswith("_")
            },
            "traces": self._extract_traces(result),
            "verification": result.get("verification"),
            "elapsed": elapsed,
        }

    @staticmethod
    def _extract_traces(result: dict) -> dict:
        """从 intermediate_results 提取各子 Agent 的工具调用轨迹。"""
        return {
            k[len("_trace_"):]: v
            for k, v in result.get("intermediate_results", {}).items()
            if k.startswith("_trace_")
        }

    async def aresume(self, decision: str, thread_id: str = "default") -> dict:
        """HITL 审批恢复接口。

        Args:
            decision: "approve" 批准执行 / 其他值视为拒绝
            thread_id: 待恢复会话的 thread_id（与 aquery 一致）
        """
        await self.initialize()
        start_time = time.time()
        result = await self.graph.ainvoke(
            Command(resume=decision), self._run_config(thread_id)
        )
        elapsed = time.time() - start_time

        # 可能再次中断（后续又有 action 任务）
        if "__interrupt__" in result:
            interrupt_info = result["__interrupt__"][0].value
            return {
                "answer": "",
                "messages": result.get("messages", []),
                "traces": {},
                "elapsed": elapsed,
                "pending_approval": interrupt_info,
            }

        answer = sanitize_output(result.get("final_answer", ""))
        return {
            "answer": answer,
            "messages": result.get("messages", []),
            "intermediate_results": {
                k: v for k, v in result.get("intermediate_results", {}).items()
                if not k.startswith("_")
            },
            "traces": self._extract_traces(result),
            "verification": result.get("verification"),
            "elapsed": elapsed,
        }

    def query(self, question: str, thread_id: str = "default",
              verbose: bool = True) -> dict:
        """同步包装（CLI 场景；async 环境请用 aquery）。"""
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass
        return asyncio.run(self.aquery(question, thread_id, verbose))

    async def astream(self, question: str, thread_id: str = "default"):
        """异步流式接口（updates 模式），供 SSE 层消费。

        事件格式:
          {"type": "thinking", "node": "supervisor", "content": ...}
          {"type": "agent_result", "node": "research"|"action", "content": ...}
          {"type": "verification", "content": ...}
          {"type": "answer", "content": ...}
        """
        await self.initialize()

        # 输入护栏: 提示注入检测
        input_check = check_input(question)
        if not input_check["safe"]:
            yield {"type": "blocked", "content": f"输入被安全护栏拦截: {input_check['reason']}"}
            return

        inputs = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "intermediate_results": {},
            "retry_count": 0,
        }
        async for chunk in self.graph.astream(
            inputs, self._run_config(thread_id), stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                # HITL: 中断事件 → 审批请求
                if node_name == "__interrupt__":
                    interrupt_value = update[0].value if isinstance(update, (list, tuple)) else getattr(update, "value", update)
                    yield {
                        "type": "approval_required",
                        "node": "action",
                        "content": interrupt_value,
                    }
                    return
                if not isinstance(update, dict):
                    continue
                if node_name == "supervisor":
                    yield {
                        "type": "thinking",
                        "node": "supervisor",
                        "content": update.get("next", ""),
                    }
                elif node_name == "verifier":
                    v = update.get("verification", {})
                    yield {
                        "type": "verification",
                        "node": "verifier",
                        "content": "通过" if v.get("passed") else "未通过",
                    }
                elif node_name == "finalize":
                    yield {
                        "type": "answer",
                        "node": "finalize",
                        "content": sanitize_output(update.get("final_answer", "")),
                    }
                else:  # research / action
                    results = update.get("intermediate_results", {})
                    yield {
                        "type": "agent_result",
                        "node": node_name,
                        "content": str(results.get(node_name, ""))[:500],
                    }
