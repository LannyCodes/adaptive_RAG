"""
记忆模块
========
Agent 三层记忆体系中的长期记忆实现。

模块列表:
- user_memory: 基于 Milvus 独立 collection 的事实型长期记忆

三层记忆划分:
- 短期记忆: LangGraph Checkpointer 中的会话消息历史（agents/runtime.py）
- 工作记忆: 超长对话的摘要压缩（agents/runtime.py 的 pre_model_hook）
- 长期记忆: 本模块（跨会话的用户偏好/项目背景/历史结论）
"""
