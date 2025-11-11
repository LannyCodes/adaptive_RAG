"""
GraphRAG检索器
实现基于知识图谱的检索策略，包括本地查询和全局查询
"""

from typing import List, Dict, Set, Tuple
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from knowledge_graph import KnowledgeGraph
from config import LOCAL_LLM


class GraphRetriever:
    """基于知识图谱的检索器"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.llm = ChatOllama(model=LOCAL_LLM, temperature=0.3)
        
        # 实体识别提示
        self.entity_recognition_prompt = PromptTemplate(
            template="""从以下问题中识别关键实体和概念:

问题: {question}

已知实体示例: {sample_entities}

请识别问题中提到的实体，返回JSON格式:
{{
    "entities": ["实体1", "实体2", ...]
}}

只返回JSON，不要其他内容。
""",
            input_variables=["question", "sample_entities"]
        )
        
        # 全局查询生成提示
        self.global_query_prompt = PromptTemplate(
            template="""你是一个知识图谱分析专家。基于以下社区摘要，回答用户问题。

用户问题: {question}

相关社区摘要:
{community_summaries}

请基于这些摘要提供一个综合性的答案。如果摘要中没有相关信息，请说明。

答案:
""",
            input_variables=["question", "community_summaries"]
        )
        
        # 本地查询生成提示
        self.local_query_prompt = PromptTemplate(
            template="""基于以下实体及其关系信息，回答用户问题。

用户问题: {question}

相关实体信息:
{entity_info}

实体间的关系:
{relations}

请基于这些信息提供答案。

答案:
""",
            input_variables=["question", "entity_info", "relations"]
        )
        
        self.entity_recognition_chain = self.entity_recognition_prompt | self.llm | JsonOutputParser()
        self.global_query_chain = self.global_query_prompt | self.llm | StrOutputParser()
        self.local_query_chain = self.local_query_prompt | self.llm | StrOutputParser()
    
    def recognize_entities(self, question: str) -> List[str]:
        """
        从问题中识别实体
        
        Args:
            question: 用户问题
            
        Returns:
            识别到的实体列表
        """
        # 获取一些示例实体
        sample_entities = list(self.kg.entities.keys())[:10]
        sample_text = ", ".join(sample_entities)
        
        try:
            result = self.entity_recognition_chain.invoke({
                "question": question,
                "sample_entities": sample_text
            })
            entities = result.get("entities", [])
            
            # 匹配到图谱中的实体
            matched_entities = []
            for entity in entities:
                # 精确匹配
                if entity in self.kg.entities:
                    matched_entities.append(entity)
                else:
                    # 模糊匹配
                    for kg_entity in self.kg.entities.keys():
                        if entity.lower() in kg_entity.lower() or kg_entity.lower() in entity.lower():
                            matched_entities.append(kg_entity)
                            break
            
            print(f"🔍 识别到实体: {matched_entities}")
            return matched_entities
            
        except Exception as e:
            print(f"❌ 实体识别失败: {e}")
            return []
    
    def local_query(self, question: str, max_hops: int = 2, top_k: int = 10) -> str:
        """
        本地查询 - 基于问题中的实体及其邻域进行检索
        
        适用场景: 针对特定实体的详细问题
        例如: "AlphaCodium的作者是谁？"
        
        Args:
            question: 用户问题
            max_hops: 最大跳数
            top_k: 返回的最大实体数
            
        Returns:
            答案文本
        """
        print(f"\n🔎 执行本地查询...")
        
        # 1. 识别问题中的实体
        mentioned_entities = self.recognize_entities(question)
        
        if not mentioned_entities:
            return "未能在知识图谱中找到相关实体。"
        
        # 2. 获取实体的邻域
        relevant_entities = set()
        for entity in mentioned_entities:
            neighbors = self.kg.get_node_neighbors(entity, depth=max_hops)
            relevant_entities.update(neighbors)
        
        relevant_entities = list(relevant_entities)[:top_k]
        
        # 3. 收集实体信息
        entity_info_list = []
        for entity in relevant_entities:
            info = self.kg.get_entity_info(entity)
            if info:
                entity_info_list.append(
                    f"- {info['name']} ({info.get('type', 'UNKNOWN')}): {info.get('description', '无描述')}"
                )
        
        # 4. 收集关系信息
        relation_list = []
        for u, v, data in self.kg.graph.edges(data=True):
            if u in relevant_entities and v in relevant_entities:
                relation_list.append(
                    f"- {u} --[{data.get('relation_type', 'RELATED')}]--> {v}: {data.get('description', '')}"
                )
        
        entity_info_text = "\n".join(entity_info_list) if entity_info_list else "无相关实体信息"
        relations_text = "\n".join(relation_list[:20]) if relation_list else "无相关关系"
        
        # 5. 生成答案
        try:
            answer = self.local_query_chain.invoke({
                "question": question,
                "entity_info": entity_info_text,
                "relations": relations_text
            })
            print(f"✅ 本地查询完成")
            return answer.strip()
        except Exception as e:
            print(f"❌ 本地查询失败: {e}")
            return "查询失败，请重试。"
    
    def global_query(self, question: str, top_k_communities: int = 5) -> str:
        """
        全局查询 - 基于社区摘要进行高层次查询
        
        适用场景: 需要整体理解的概括性问题
        例如: "这些文档主要讨论什么主题？"
        
        Args:
            question: 用户问题
            top_k_communities: 使用的社区数量
            
        Returns:
            答案文本
        """
        print(f"\n🌍 执行全局查询...")
        
        if not self.kg.community_summaries:
            return "知识图谱尚未生成社区摘要，请先运行索引流程。"
        
        # 获取社区摘要
        community_summaries = []
        for cid, summary in list(self.kg.community_summaries.items())[:top_k_communities]:
            community_summaries.append(f"社区 {cid}:\n{summary}\n")
        
        summaries_text = "\n".join(community_summaries)
        
        # 生成答案
        try:
            answer = self.global_query_chain.invoke({
                "question": question,
                "community_summaries": summaries_text
            })
            print(f"✅ 全局查询完成")
            return answer.strip()
        except Exception as e:
            print(f"❌ 全局查询失败: {e}")
            return "查询失败，请重试。"
    
    def hybrid_query(self, question: str) -> Dict[str, str]:
        """
        混合查询 - 同时执行本地和全局查询，返回两种结果
        
        Args:
            question: 用户问题
            
        Returns:
            包含本地和全局查询结果的字典
        """
        print(f"\n🔀 执行混合查询...")
        
        local_answer = self.local_query(question)
        global_answer = self.global_query(question)
        
        return {
            "local": local_answer,
            "global": global_answer,
            "question": question
        }
    
    def smart_query(self, question: str) -> str:
        """
        智能查询 - 根据问题类型自动选择查询策略
        
        Args:
            question: 用户问题
            
        Returns:
            答案文本
        """
        # 判断问题类型
        question_lower = question.lower()
        
        # 包含具体实体名称的问题 -> 本地查询
        mentioned_entities = self.recognize_entities(question)
        if mentioned_entities:
            print("📍 检测到具体实体，使用本地查询")
            return self.local_query(question)
        
        # 概括性问题 -> 全局查询
        global_keywords = ["主要", "总体", "概述", "整体", "主题", "讨论", "内容", "what", "overview", "main", "topics"]
        if any(keyword in question_lower for keyword in global_keywords):
            print("🌐 检测到概括性问题，使用全局查询")
            return self.global_query(question)
        
        # 默认使用本地查询
        print("📍 使用本地查询作为默认策略")
        return self.local_query(question)


def initialize_graph_retriever(knowledge_graph: KnowledgeGraph):
    """初始化GraphRAG检索器"""
    return GraphRetriever(knowledge_graph)
