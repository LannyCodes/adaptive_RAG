"""
Verifier — 答案验证器
=====================
对子智能体的产出做最终质量校验（确定性校验，非 LLM Agent 循环）。

复用现有评分器:
- answer_grader: 答案是否有用、是否切题（question + generation → yes/no）

Supervisor 层没有检索文档（文档在 Research Agent 的 MCP 会话内），
因此此处验证"答案是否回答了问题"，幻觉校验已在 Research Agent 内部
由 check_hallucination 工具完成。
"""

from typing import Optional


def build_verifier(graders: Optional[dict] = None):
    """构建验证函数。返回 verify(question, answer) -> dict。"""

    if graders is None:
        from routers_and_graders import initialize_graders_and_router

        graders = initialize_graders_and_router()

    def verify(question: str, answer: str) -> dict:
        """验证答案质量。返回 {"passed": bool, "score": str, "reason": str}。"""
        if not answer or not answer.strip():
            return {"passed": False, "score": "no", "reason": "答案为空"}

        try:
            score = graders["answer_grader"].grade(question, answer)
        except Exception as e:
            # 验证器故障时放行（不阻塞主流程），但标记原因
            return {"passed": True, "score": "unknown",
                    "reason": f"验证器异常，默认放行: {e}"}

        passed = score == "yes"
        return {
            "passed": passed,
            "score": score,
            "reason": "答案切题且有用" if passed else "答案未能有效回答问题",
        }

    return verify
