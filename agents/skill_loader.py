"""
Skill 资产加载器
================
把 Agent 的"操作手册"从代码字符串升级为独立资产（SKILL.md），
支持独立编辑、版本化、按需注入，实现提示词与代码解耦。

SKILL.md 格式:
    ---
    name: rag_query
    description: 技能的用途与触发条件
    version: "1.0"
    ---
    指令正文（注入给 LLM 的系统提示词）

用法:
    from agents.skill_loader import load_skill
    meta, instructions = load_skill("rag_query")
"""

import os
from typing import Dict, Optional, Tuple

# skills/ 目录定位在项目根（agents/ 的上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKILLS_DIR = os.path.join(_PROJECT_ROOT, "skills")


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """解析 SKILL.md 的 YAML frontmatter（轻量实现，仅支持 key: value 扁平结构）。

    Returns:
        (meta 字典, 指令正文)
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    meta: Dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped == "---":
            body = "\n".join(lines[i + 1:]).strip()
            return meta, body
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")

    # 未闭合的 frontmatter：整体当作正文处理
    return {}, text


def load_skill(name: str, skills_dir: Optional[str] = None) -> Tuple[Dict[str, str], str]:
    """加载指定技能资产。

    Args:
        name: 技能目录名（对应 skills/<name>/SKILL.md）
        skills_dir: 自定义技能目录（默认项目根 skills/）

    Returns:
        (meta 元数据字典, instructions 指令正文)

    Raises:
        FileNotFoundError: 技能文件缺失——视为部署问题，显式报错而非静默降级，
            避免 Agent 行为在无人察觉的情况下退化。
    """
    skill_path = os.path.join(skills_dir or _SKILLS_DIR, name, "SKILL.md")
    if not os.path.isfile(skill_path):
        raise FileNotFoundError(
            f"技能资产缺失: {skill_path} —— 请确认 skills/{name}/SKILL.md 已随项目部署"
        )

    with open(skill_path, "r", encoding="utf-8") as f:
        meta, instructions = _parse_frontmatter(f.read())

    if not instructions:
        raise ValueError(f"技能资产为空: {skill_path}")
    return meta, instructions


def get_rag_system_prompt() -> str:
    """加载 RAG 问答技能的指令正文（供 AgentRuntime / AgentRAG 作为系统提示词）。"""
    _, instructions = load_skill("rag_query")
    return instructions
