"""
Prompt 管理器
将所有 LLM 提示集中管理，支持 YAML 配置和运行时渲染
"""

import os
import yaml
from typing import Optional
from langchain.prompts import PromptTemplate as LangChainPromptTemplate


DEFAULT_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts.yaml")


class PromptManager:
    """集中式 Prompt 管理器"""

    def __init__(self, yaml_path: str = DEFAULT_PROMPTS_PATH):
        self._yaml_path = yaml_path
        self._prompts: dict = {}
        self._templates: dict[str, LangChainPromptTemplate] = {}
        self._load()

    def _load(self):
        """加载 YAML 配置"""
        if not os.path.exists(self._yaml_path):
            print(f"⚠️  Prompt 文件未找到: {self._yaml_path}")
            return

        with open(self._yaml_path, "r", encoding="utf-8") as f:
            self._prompts = yaml.safe_load(f)

        # 预编译为 LangChain PromptTemplate
        for key, config in self._prompts.items():
            if isinstance(config, dict) and "template" in config:
                self._templates[key] = LangChainPromptTemplate(
                    template=config["template"],
                    input_variables=config.get("input_vars", []),
                )

        print(f"  📝 Prompt 管理器已加载 ({len(self._templates)} 个模板)")

    def get(self, key: str, **kwargs) -> str:
        """获取渲染后的 Prompt 字符串"""
        template = self._templates.get(key)
        if template is None:
            raise KeyError(f"Prompt 模板 '{key}' 未找到")

        # 检查是否缺少必要变量
        missing = [v for v in template.input_variables if v not in kwargs]
        if missing:
            print(f"⚠️  Prompt '{key}' 缺少变量: {missing}")

        return template.format(**kwargs)

    def get_template(self, key: str) -> Optional[LangChainPromptTemplate]:
        """获取 LangChain PromptTemplate 对象"""
        return self._templates.get(key)

    def list_keys(self) -> list:
        """列出所有可用 Prompt 键"""
        return list(self._templates.keys())

    def reload(self):
        """热重载（开发时修改 YAML 后调用）"""
        self._templates.clear()
        self._load()
        print(f"  🔄 Prompt 已重载 ({len(self._templates)} 个模板)")


# 全局单例
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
