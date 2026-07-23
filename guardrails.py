"""
安全护栏模块
============
Agent 系统的输入/输出双侧护栏:

- 输入侧 check_input: 提示注入检测（规则正则 + 关键词黑名单）。
  Agent 会接触不可信外部数据（网络搜索/网页抓取），注入防护是刚需。
- 输出侧 sanitize_output: 敏感信息正则过滤（API key、密码、私钥模式）。

工具侧强制约束（工作目录限制、代码执行黑名单）在 mcp_servers/action_server.py
中实现，属于强制执行层，不经过本模块。
"""

import re
from typing import Dict, List, Tuple

# ── 提示注入检测模式（不区分大小写）─────────────────────────
# 覆盖常见注入手法: 指令覆盖、角色扮演越狱、系统提示探测、分隔符逃逸
_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"forget\s+(everything|all|your)\s+(you|instructions?|rules?)",
    # 中文模式允许插入词: “忽略之前的所有指令” / “忘记你之前的全部设定”
    r"忽略(之前|以上|先前|所有).{0,6}(指令|指示|提示|规则|约束)",
    r"忘记(之前|所有|你).{0,6}(指令|设定|规则)",
    r"you\s+are\s+now\s+(a|an)\s+\w+\s+(without|with\s+no)\s+(restrictions?|limits?)",
    r"(jailbreak|DAN\s+mode|do\s+anything\s+now)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
    r"(显示|透露|输出)(你的)?(系统提示|提示词|指令集)",
    r"\[system\]|<\s*system\s*>|###\s*system",
    r"new\s+instructions?:",
]

# ── 敏感信息模式（输出侧过滤）───────────────────────────────
_SENSITIVE_PATTERNS: List[Tuple[str, str]] = [
    # (模式, 替换文本)
    (r"(tvly-[A-Za-z0-9]{10,})", "[REDACTED_TAVILY_KEY]"),
    (r"(sk-[A-Za-z0-9]{20,})", "[REDACTED_API_KEY]"),
    (r"(sk-ant-[A-Za-z0-9\-]{20,})", "[REDACTED_API_KEY]"),
    (r"(?i)(api[_-]?key|token|secret|password|密码|口令)\s*[:=]\s*[\"']?[\w\-]{8,}[\"']?",
     r"\1=[REDACTED]"),
    (r"-----BEGIN\s+\w+\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+\w+\s+PRIVATE\s+KEY-----",
     "[REDACTED_PRIVATE_KEY]"),
]


def check_input(text: str) -> Dict:
    """输入侧护栏: 检测提示注入。

    Returns:
        {"safe": bool, "reason": str, "matched": str}
    """
    if not text:
        return {"safe": True, "reason": "", "matched": ""}

    for pattern in _INJECTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                "safe": False,
                "reason": "检测到疑似提示注入攻击",
                "matched": match.group(0)[:100],
            }
    return {"safe": True, "reason": "", "matched": ""}


def sanitize_output(text: str) -> str:
    """输出侧护栏: 过滤敏感信息（API key、密码、私钥）。"""
    if not text:
        return text
    sanitized = text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized


def check_and_sanitize_input(text: str) -> Dict:
    """输入护栏入口（check_input 的别名，保持语义完整）。"""
    return check_input(text)
