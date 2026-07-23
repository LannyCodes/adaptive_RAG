"""
长期用户记忆
============
基于 Milvus 独立 collection 的事实型长期记忆:
- extract_and_store(): 对话结束后用轻量模型抽取事实记忆（JSON）并写入
- retrieve(): 按问题语义检索 top-k 相关记忆，注入系统提示
- 冲突处理: 同 (category, thread_id) 新记忆覆盖旧记忆（先删后写）

实现要点:
1. 向量维度由 langchain-milvus 按嵌入模型实际输出自动确定（bge-m3=1024），
   不硬编码，避免 dimension mismatch（项目历史踩坑）。
2. langchain-milvus 0.1.10 默认将 metadata 每个 key 建为独立 schema 字段
   （按首次插入值推断类型），因此所有记忆的 metadata key 必须一致；
   过滤/删除表达式直接用普通字段语法: category == "x"。
3. 复用 pymilvus "default" 全局连接（与主向量库同一 Milvus 实例），
   嵌入实例由调用方注入（避免重复加载 bge-m3）。
"""

import json
import time
from typing import List, Optional

from config import MEMORY_TOP_K, USER_MEMORY_COLLECTION

# 记忆 metadata 的固定 key（首次插入决定 Milvus schema，不可变更）
_META_KEYS = ("category", "thread_id", "timestamp")


class UserMemoryStore:
    """事实型长期记忆存储（Milvus 后端）。"""

    def __init__(self, embeddings, collection_name: str = USER_MEMORY_COLLECTION):
        from langchain_milvus import Milvus

        self.store = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"alias": "default"},  # 复用主库全局连接
            drop_old=False,
            auto_id=True,
        )

    # ── 写入 ──

    def add_memory(self, content: str, category: str = "general",
                   thread_id: str = "") -> None:
        """写入记忆。同 (category, thread_id) 的旧记忆被覆盖。"""
        content = content.strip()
        if not content:
            return
        self._delete_by_category(category, thread_id)
        self.store.add_texts(
            [content],
            metadatas=[{
                "category": category[:128],
                "thread_id": thread_id[:128],
                "timestamp": int(time.time()),
            }],
        )

    def _delete_by_category(self, category: str, thread_id: str = "") -> None:
        """删除同 (category, thread_id) 的旧记忆（覆盖语义）。失败不阻塞写入。"""
        try:
            col = getattr(self.store, "col", None)
            if col is None:
                return
            expr = f'category == "{category}" and thread_id == "{thread_id}"'
            col.delete(expr)
        except Exception as e:
            print(f"⚠️ 旧记忆清理失败（忽略，继续写入）: {e}")

    # ── 检索 ──

    def retrieve(self, query: str, top_k: int = MEMORY_TOP_K) -> List[str]:
        """按语义检索相关记忆内容，按时间新→旧排序。"""
        try:
            docs = self.store.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"⚠️ 记忆检索失败（按无记忆处理）: {e}")
            return []
        # 按 timestamp 新→旧排序（schema 字段为独立 metadata key）
        docs.sort(
            key=lambda d: d.metadata.get("timestamp", 0) if d.metadata else 0,
            reverse=True,
        )
        return [d.page_content for d in docs]

    def format_for_prompt(self, query: str, top_k: int = MEMORY_TOP_K) -> str:
        """检索并格式化为系统提示注入文本；无记忆时返回空串。"""
        memories = self.retrieve(query, top_k)
        if not memories:
            return ""
        lines = "\n".join(f"- {m}" for m in memories)
        return f"\n\n## 关于用户的长期记忆（跨会话）\n{lines}\n"

    # ── 抽取 ──

    def extract_and_store(self, question: str, answer: str,
                          thread_id: str = "") -> None:
        """对话结束后用轻量模型抽取事实型记忆并写入。

        只记"值得长期记住"的内容: 用户偏好、身份信息、项目背景、明确结论。
        失败静默（记忆是增强能力，不影响主流程）。
        """
        if not question or not answer:
            return
        try:
            from routers_and_graders import create_chat_model

            prompt = (
                "请从以下对话中抽取值得长期记住的事实（用户偏好、身份信息、"
                "项目背景、用户确认过的结论）。\n"
                "以 JSON 数组返回，每项含 \"content\" 和 \"category\" 两个字段；"
                "category 从 preference/identity/project/fact 中选择。\n"
                "没有值得记忆的内容时返回空数组 []。仅输出 JSON，不要任何解释。\n\n"
                f"用户: {question}\n助手: {answer[:2000]}"
            )
            llm = create_chat_model(format="json", temperature=0.0, light=True)
            raw = llm.invoke(prompt).content
            items = json.loads(raw)
            if not isinstance(items, list):
                return
            for item in items[:5]:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "")).strip()
                category = str(item.get("category", "general")).strip() or "general"
                if content:
                    self.add_memory(content, category, thread_id)
        except Exception as e:  # 含 json.JSONDecodeError
            print(f"⚠️ 记忆抽取失败（忽略）: {e}")
