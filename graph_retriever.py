"""
GraphRAG检索器
实现基于知识图谱的检索策略，包括本地查询和全局查询
"""

from typing import List, Dict, Set, Tuple
import time
import networkx as nx
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from knowledge_graph import KnowledgeGraph
from retrieval_evaluation import RetrievalEvaluator, RetrievalResult
from routers_and_graders import HallucinationGrader, create_chat_model
from prompt_manager import get_prompt_manager


class GraphRetriever:
    """基于知识图谱的检索器"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.llm = create_chat_model(temperature=0.3)
        self.hallucination_grader = HallucinationGrader()
        
        # 实体识别提示
        self.entity_recognition_prompt = get_prompt_manager().get_template("recognize_entities")
        
        # 全局查询生成提示
        self.global_query_prompt = get_prompt_manager().get_template("global_query")
        
        # 本地查询生成提示
        self.local_query_prompt = get_prompt_manager().get_template("local_query")
        
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
    
    def _normalize_map(self, values: Dict[str, float], keys: List[str]) -> Dict[str, float]:
        arr = [values.get(k, 0.0) for k in keys]
        if not arr:
            return {k: 0.0 for k in keys}
        mn = min(arr)
        mx = max(arr)
        if mx == mn:
            return {k: 0.5 for k in keys}
        return {k: (values.get(k, 0.0) - mn) / (mx - mn) for k in keys}
    
    def _rank_entities(self, mentioned_entities: List[str], candidate_entities: List[str]) -> List[str]:
        G = self.kg.graph
        nodes = list(set(candidate_entities) | set(mentioned_entities))
        if not nodes:
            return []
        subG = G.subgraph(nodes)
        deg = nx.degree_centrality(subG)
        btw = nx.betweenness_centrality(subG, normalized=True)
        weight_to_mentioned = {}
        path_prox = {}
        for n in candidate_entities:
            w_sum = 0.0
            best_len = None
            for m in mentioned_entities:
                if G.has_edge(n, m):
                    data = G.get_edge_data(n, m)
                    if isinstance(data, dict):
                        w_sum += float(data.get('weight', 1.0))
                    else:
                        w_sum += 1.0
                try:
                    l = nx.shortest_path_length(G, source=m, target=n)
                    if best_len is None or l < best_len:
                        best_len = l
                except nx.NetworkXNoPath:
                    pass
            weight_to_mentioned[n] = w_sum
            path_prox[n] = 0.0 if best_len is None else 1.0 / (1.0 + best_len)
        deg_n = self._normalize_map(deg, candidate_entities)
        btw_n = self._normalize_map(btw, candidate_entities)
        w_n = self._normalize_map(weight_to_mentioned, candidate_entities)
        prox_n = self._normalize_map(path_prox, candidate_entities)
        scores = {}
        for n in candidate_entities:
            scores[n] = 0.3 * deg_n.get(n, 0.0) + 0.3 * btw_n.get(n, 0.0) + 0.2 * w_n.get(n, 0.0) + 0.2 * prox_n.get(n, 0.0)
        ranked = sorted(candidate_entities, key=lambda x: scores.get(x, 0.0), reverse=True)
        return ranked
    
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
        ranked_entities = self._rank_entities(mentioned_entities, list(relevant_entities))
        relevant_entities = ranked_entities[:top_k]
        
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
    
    def local_query_with_metrics(self, question: str, max_hops: int = 2, top_k: int = 10, k_values: List[int] = [1, 3, 5]) -> tuple:
        print(f"\n🔎 执行本地查询并评估...")
        start_t = time.time()
        mentioned_entities = self.recognize_entities(question)
        if not mentioned_entities:
            return "未能在知识图谱中找到相关实体。", {
                "error": "no_entities",
                "latency": 0.0,
                "retrieved_docs_count": 0
            }
        relevant_entities = set()
        for entity in mentioned_entities:
            neighbors = self.kg.get_node_neighbors(entity, depth=max_hops)
            relevant_entities.update(neighbors)
        ranked_entities = self._rank_entities(mentioned_entities, list(relevant_entities))
        relevant_entities = ranked_entities[:top_k]
        entity_info_list = []
        for entity in relevant_entities:
            info = self.kg.get_entity_info(entity)
            if info:
                entity_info_list.append(f"- {info['name']} ({info.get('type', 'UNKNOWN')}): {info.get('description', '无描述')}")
        relation_list = []
        for u, v, data in self.kg.graph.edges(data=True):
            if u in relevant_entities and v in relevant_entities:
                relation_list.append(f"- {u} --[{data.get('relation_type', 'RELATED')}]--> {v}: {data.get('description', '')}")
        entity_info_text = "\n".join(entity_info_list) if entity_info_list else "无相关实体信息"
        relations_text = "\n".join(relation_list[:20]) if relation_list else "无相关关系"
        try:
            answer = self.local_query_chain.invoke({
                "question": question,
                "entity_info": entity_info_text,
                "relations": relations_text
            }).strip()
        except Exception:
            answer = "查询失败，请重试。"
        retrieved_docs = []
        for entity in relevant_entities:
            info = self.kg.get_entity_info(entity) or {"name": entity}
            content = f"{info.get('name', entity)} {info.get('type', '')} {info.get('description', '')}".strip()
            retrieved_docs.append(Document(page_content=content, metadata={"entity": info.get('name', entity)}))
        try:
            hallucination_grade = self.hallucination_grader.grade(answer, retrieved_docs)
        except Exception:
            hallucination_grade = "unknown"
        relevant_docs = []
        for entity in mentioned_entities:
            info = self.kg.get_entity_info(entity) or {"name": entity}
            content = f"{info.get('name', entity)} {info.get('type', '')} {info.get('description', '')}".strip()
            relevant_docs.append(Document(page_content=content, metadata={"entity": info.get('name', entity)}))
        latency = time.time() - start_t
        try:
            evaluator = RetrievalEvaluator()
            result = RetrievalResult(query=question, retrieved_docs=retrieved_docs, relevant_docs=relevant_docs, retrieval_time=latency)
            metrics_obj = evaluator.evaluate_retrieval([result], k_values=k_values)
            metrics = {
                "precision_at_1": metrics_obj.precision_at_k.get(1, 0),
                "precision_at_3": metrics_obj.precision_at_k.get(3, 0),
                "precision_at_5": metrics_obj.precision_at_k.get(5, 0),
                "recall_at_1": metrics_obj.recall_at_k.get(1, 0),
                "recall_at_3": metrics_obj.recall_at_k.get(3, 0),
                "recall_at_5": metrics_obj.recall_at_k.get(5, 0),
                "map_score": metrics_obj.map_score,
                "mrr": metrics_obj.mrr,
                "latency": metrics_obj.latency,
                "retrieved_docs_count": len(retrieved_docs),
                "hallucination": hallucination_grade
            }
        except Exception:
            metrics = {"latency": latency, "retrieved_docs_count": len(retrieved_docs), "hallucination": hallucination_grade}
        return answer, metrics
    
    def global_query_with_metrics(self, question: str, top_k_communities: int = 5, k_values: List[int] = [1, 3, 5]) -> tuple:
        print(f"\n🌍 执行全局查询并评估...")
        start_t = time.time()
        mentioned_entities = self.recognize_entities(question)
        if not self.kg.community_summaries:
            return "知识图谱尚未生成社区摘要，请先运行索引流程。", {
                "error": "no_summaries",
                "latency": 0.0,
                "retrieved_docs_count": 0
            }
        community_summaries = []
        for cid, summary in list(self.kg.community_summaries.items())[:top_k_communities]:
            community_summaries.append((cid, summary))
        summaries_text = "\n".join([f"社区 {cid}:\n{summary}\n" for cid, summary in community_summaries])
        try:
            answer = self.global_query_chain.invoke({
                "question": question,
                "community_summaries": summaries_text
            }).strip()
        except Exception:
            answer = "查询失败，请重试。"
        retrieved_docs = []
        for cid, summary in community_summaries:
            retrieved_docs.append(Document(page_content=summary, metadata={"community_id": str(cid)}))
        try:
            hallucination_grade = self.hallucination_grader.grade(answer, retrieved_docs)
        except Exception:
            hallucination_grade = "unknown"
        relevant_docs = []
        query_tokens = [t for t in question.split() if t]
        for cid, summary in community_summaries:
            ok = False
            for ent in mentioned_entities:
                if ent and ent.lower() in summary.lower():
                    ok = True
                    break
            if not ok:
                for t in query_tokens:
                    if t and t.lower() in summary.lower():
                        ok = True
                        break
            if ok:
                relevant_docs.append(Document(page_content=summary, metadata={"community_id": str(cid)}))
        latency = time.time() - start_t
        try:
            evaluator = RetrievalEvaluator()
            result = RetrievalResult(query=question, retrieved_docs=retrieved_docs, relevant_docs=relevant_docs, retrieval_time=latency)
            metrics_obj = evaluator.evaluate_retrieval([result], k_values=k_values)
            metrics = {
                "precision_at_1": metrics_obj.precision_at_k.get(1, 0),
                "precision_at_3": metrics_obj.precision_at_k.get(3, 0),
                "precision_at_5": metrics_obj.precision_at_k.get(5, 0),
                "recall_at_1": metrics_obj.recall_at_k.get(1, 0),
                "recall_at_3": metrics_obj.recall_at_k.get(3, 0),
                "recall_at_5": metrics_obj.recall_at_k.get(5, 0),
                "map_score": metrics_obj.map_score,
                "mrr": metrics_obj.mrr,
                "latency": metrics_obj.latency,
                "retrieved_docs_count": len(retrieved_docs),
                "hallucination": hallucination_grade
            }
        except Exception:
            metrics = {"latency": latency, "retrieved_docs_count": len(retrieved_docs), "hallucination": hallucination_grade}
        return answer, metrics
    
    def hybrid_query_with_metrics(self, question: str) -> Dict[str, str]:
        print(f"\n🔀 执行混合查询并评估...")
        local_answer, local_metrics = self.local_query_with_metrics(question)
        global_answer, global_metrics = self.global_query_with_metrics(question)
        return {
            "local": local_answer,
            "global": global_answer,
            "local_hallucination": local_metrics.get("hallucination"),
            "global_hallucination": global_metrics.get("hallucination"),
            "local_metrics": local_metrics,
            "global_metrics": global_metrics,
            "question": question
        }
    
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
