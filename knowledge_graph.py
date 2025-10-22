"""
知识图谱模块
实现GraphRAG的核心功能：图谱构建、社区检测、层次化摘要
"""

import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import json

try:
    from community import community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("⚠️ python-louvain未安装，社区检测功能受限")

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from config import LOCAL_LLM


class KnowledgeGraph:
    """知识图谱类 - 使用NetworkX构建和管理图谱"""
    
    def __init__(self):
        self.graph = nx.Graph()  # 无向图
        self.entities = {}  # 实体详细信息
        self.communities = {}  # 社区划分结果
        self.community_summaries = {}  # 社区摘要
        
    def add_entity(self, name: str, entity_type: str, description: str = "", **kwargs):
        """添加实体节点"""
        self.graph.add_node(
            name,
            type=entity_type,
            description=description,
            **kwargs
        )
        self.entities[name] = {
            "name": name,
            "type": entity_type,
            "description": description,
            **kwargs
        }
    
    def add_relation(self, source: str, target: str, relation_type: str, 
                    description: str = "", weight: float = 1.0):
        """添加关系边"""
        self.graph.add_edge(
            source,
            target,
            relation_type=relation_type,
            description=description,
            weight=weight
        )
    
    def build_from_extractions(self, extraction_results: List[Dict]):
        """
        从实体提取结果构建图谱
        
        Args:
            extraction_results: 实体和关系提取结果列表
        """
        print("🔨 开始构建知识图谱...")
        
        total_entities = 0
        total_relations = 0
        
        for result in extraction_results:
            # 添加实体
            entities = result.get("entities", [])
            for entity in entities:
                self.add_entity(
                    name=entity["name"],
                    entity_type=entity.get("type", "UNKNOWN"),
                    description=entity.get("description", "")
                )
                total_entities += 1
            
            # 添加关系
            relations = result.get("relations", [])
            for relation in relations:
                source = relation.get("source")
                target = relation.get("target")
                
                # 确保节点存在
                if source in self.graph and target in self.graph:
                    self.add_relation(
                        source=source,
                        target=target,
                        relation_type=relation.get("relation_type", "RELATED_TO"),
                        description=relation.get("description", "")
                    )
                    total_relations += 1
        
        print(f"✅ 图谱构建完成: {total_entities} 个实体, {total_relations} 个关系")
        print(f"   实际节点数: {self.graph.number_of_nodes()}")
        print(f"   实际边数: {self.graph.number_of_edges()}")
    
    def detect_communities(self, algorithm: str = "louvain") -> Dict[str, int]:
        """
        社区检测 - GraphRAG的核心组件
        
        Args:
            algorithm: 社区检测算法 ('louvain', 'greedy', 'label_propagation')
            
        Returns:
            节点到社区ID的映射
        """
        print(f"🔍 开始社区检测 (算法: {algorithm})...")
        
        if self.graph.number_of_nodes() == 0:
            print("⚠️ 图谱为空，跳过社区检测")
            return {}
        
        try:
            if algorithm == "louvain" and LOUVAIN_AVAILABLE:
                communities = community_louvain.best_partition(self.graph)
            elif algorithm == "greedy":
                communities_generator = nx.community.greedy_modularity_communities(self.graph)
                communities = {}
                for idx, community_set in enumerate(communities_generator):
                    for node in community_set:
                        communities[node] = idx
            elif algorithm == "label_propagation":
                communities_generator = nx.community.label_propagation_communities(self.graph)
                communities = {}
                for idx, community_set in enumerate(communities_generator):
                    for node in community_set:
                        communities[node] = idx
            else:
                print(f"⚠️ 未知算法 {algorithm}，使用贪婪算法")
                communities_generator = nx.community.greedy_modularity_communities(self.graph)
                communities = {}
                for idx, community_set in enumerate(communities_generator):
                    for node in community_set:
                        communities[node] = idx
            
            self.communities = communities
            num_communities = len(set(communities.values()))
            print(f"✅ 检测到 {num_communities} 个社区")
            
            return communities
            
        except Exception as e:
            print(f"❌ 社区检测失败: {e}")
            return {}
    
    def get_community_members(self, community_id: int) -> List[str]:
        """获取指定社区的所有成员"""
        return [node for node, cid in self.communities.items() if cid == community_id]
    
    def get_community_subgraph(self, community_id: int) -> nx.Graph:
        """获取指定社区的子图"""
        members = self.get_community_members(community_id)
        return self.graph.subgraph(members)
    
    def get_node_neighbors(self, node: str, depth: int = 1) -> Set[str]:
        """获取节点的邻居（支持多跳）"""
        if node not in self.graph:
            return set()
        
        neighbors = {node}
        current_layer = {node}
        
        for _ in range(depth):
            next_layer = set()
            for n in current_layer:
                next_layer.update(self.graph.neighbors(n))
            neighbors.update(next_layer)
            current_layer = next_layer
        
        return neighbors
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict]:
        """获取实体详细信息"""
        return self.entities.get(entity_name)
    
    def search_entities_by_type(self, entity_type: str) -> List[str]:
        """按类型搜索实体"""
        return [
            name for name, data in self.entities.items()
            if data.get("type") == entity_type
        ]
    
    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_communities": len(set(self.communities.values())) if self.communities else 0,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            "entity_types": {}
        }
        
        # 统计实体类型分布
        for entity in self.entities.values():
            etype = entity.get("type", "UNKNOWN")
            stats["entity_types"][etype] = stats["entity_types"].get(etype, 0) + 1
        
        return stats
    
    def save_to_file(self, filepath: str):
        """保存图谱到文件"""
        data = {
            "entities": self.entities,
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "data": data
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "communities": self.communities,
            "community_summaries": self.community_summaries
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 图谱已保存到: {filepath}")
    
    def load_from_file(self, filepath: str):
        """从文件加载图谱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entities = data.get("entities", {})
        self.communities = data.get("communities", {})
        self.community_summaries = data.get("community_summaries", {})
        
        # 重建图
        self.graph.clear()
        for name, entity in self.entities.items():
            self.add_entity(**entity)
        
        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                **edge["data"]
            )
        
        print(f"✅ 图谱已从文件加载: {filepath}")


class CommunitySummarizer:
    """社区摘要生成器 - GraphRAG的关键组件"""
    
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, temperature=0.3)
        
        self.summary_prompt = PromptTemplate(
            template="""你是一个知识图谱分析专家。请为以下社区生成一个综合摘要。

社区成员（实体）:
{entities}

实体间的关系:
{relations}

请生成一个简洁的摘要，描述：
1. 这个社区的主题是什么
2. 主要包含哪些核心概念
3. 实体之间的关键关系

摘要（2-3句话）:
""",
            input_variables=["entities", "relations"]
        )
        
        self.summary_chain = self.summary_prompt | self.llm | StrOutputParser()
    
    def summarize_community(self, kg: KnowledgeGraph, community_id: int) -> str:
        """
        为指定社区生成摘要
        
        Args:
            kg: 知识图谱对象
            community_id: 社区ID
            
        Returns:
            社区摘要文本
        """
        members = kg.get_community_members(community_id)
        subgraph = kg.get_community_subgraph(community_id)
        
        # 准备实体信息
        entity_info = []
        for member in members[:20]:  # 限制数量
            info = kg.get_entity_info(member)
            if info:
                entity_info.append(
                    f"- {info['name']} ({info.get('type', 'UNKNOWN')}): {info.get('description', '无描述')}"
                )
        
        # 准备关系信息
        relation_info = []
        for u, v, data in subgraph.edges(data=True):
            relation_info.append(
                f"- {u} --[{data.get('relation_type', 'RELATED')}]--> {v}"
            )
        
        entities_text = "\n".join(entity_info) if entity_info else "无实体"
        relations_text = "\n".join(relation_info[:15]) if relation_info else "无关系"
        
        try:
            summary = self.summary_chain.invoke({
                "entities": entities_text,
                "relations": relations_text
            })
            return summary.strip()
        except Exception as e:
            print(f"❌ 社区 {community_id} 摘要生成失败: {e}")
            return f"社区{community_id}: 包含{len(members)}个实体"
    
    def summarize_all_communities(self, kg: KnowledgeGraph) -> Dict[int, str]:
        """为所有社区生成摘要"""
        if not kg.communities:
            print("⚠️ 未检测到社区，请先运行社区检测")
            return {}
        
        community_ids = set(kg.communities.values())
        print(f"📝 开始为 {len(community_ids)} 个社区生成摘要...")
        
        summaries = {}
        for cid in community_ids:
            print(f"   处理社区 {cid}...")
            summary = self.summarize_community(kg, cid)
            summaries[cid] = summary
            kg.community_summaries[cid] = summary
        
        print("✅ 所有社区摘要生成完成")
        return summaries


def initialize_knowledge_graph():
    """初始化知识图谱"""
    return KnowledgeGraph()


def initialize_community_summarizer():
    """初始化社区摘要生成器"""
    return CommunitySummarizer()
