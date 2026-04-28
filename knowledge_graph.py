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

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from config import LOCAL_LLM
from routers_and_graders import create_chat_model


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
    
    def load_from_jsonld(self, filepath: str):
        """
        从 JSON-LD 格式的知识图谱数据加载图谱
        
        支持从 GraphRAG 导出的 JSON-LD 文件中直接加载实体、关系和社区报告。
        
        Args:
            filepath: JSON-LD 文件路径
        """
        import re
        
        def _extract_name(id_uri: str) -> str:
            return id_uri.rstrip("/").split("/")[-1]
        
        def _extract_type_short(type_uri: str) -> str:
            return type_uri.rstrip("/").split("/")[-1]
        
        print(f"📂 正在从 JSON-LD 文件加载知识图谱: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        graph_nodes = data.get("@graph", [])
        
        entity_type_uris = {
            "http://schema.org/PERSON",
            "http://schema.org/ORGANIZATION",
            "http://schema.org/EVENT",
            "http://schema.org/GEO",
            "http://schema.org/INDUSTRY",
            "http://schema.org/PRODUCT",
            "http://schema.org/ORGANISM",
        }
        
        # 清空现有数据
        self.graph.clear()
        self.entities = {}
        self.communities = {}
        self.community_summaries = {}
        
        # 第一遍: 添加所有实体
        entity_count = 0
        for node in graph_nodes:
            node_type = node.get("@type", "")
            if node_type in entity_type_uris:
                name = node.get("schema:name") or node.get("rdfs:label") or _extract_name(node.get("@id", ""))
                entity_type_short = _extract_type_short(node_type)
                description = node.get("schema:description", "")
                frequency = node.get("kg:frequency", 0)
                degree = node.get("kg:degree", 0.0)
                
                self.add_entity(
                    name=name,
                    entity_type=entity_type_short,
                    description=description,
                    frequency=int(frequency) if frequency else 0,
                    degree=float(degree) if degree else 0.0,
                )
                entity_count += 1
        
        # 第二遍: 添加所有关系
        relation_count = 0
        for node in graph_nodes:
            node_type = node.get("@type", "")
            if node_type == "rdf:Statement":
                subject = node.get("rdf:subject", {})
                obj = node.get("rdf:object", {})
                predicate = node.get("rdf:predicate", {})
                description = node.get("schema:description", "")
                weight = node.get("kg:weight", 1.0)
                
                subject_name = _extract_name(subject.get("@id", "")) if isinstance(subject, dict) else ""
                object_name = _extract_name(obj.get("@id", "")) if isinstance(obj, dict) else ""
                predicate_name = _extract_name(predicate.get("@id", "")) if isinstance(predicate, dict) else ""
                
                if subject_name and object_name and subject_name in self.graph and object_name in self.graph:
                    self.add_relation(
                        source=subject_name,
                        target=object_name,
                        relation_type=predicate_name,
                        description=description,
                        weight=float(weight) if weight else 1.0,
                    )
                    relation_count += 1
        
        # 第三遍: 加载社区报告
        community_count = 0
        for node in graph_nodes:
            node_type = node.get("@type", "")
            if node_type == "schema:Article":
                headline = node.get("schema:headline", "")
                summary = node.get("schema:description", "")
                community_id = _extract_name(node.get("@id", ""))
                
                try:
                    cid = int(re.search(r'\d+', community_id).group()) if re.search(r'\d+', community_id) else 0
                except (ValueError, AttributeError):
                    cid = hash(community_id) % 10000
                
                self.community_summaries[cid] = summary or headline
                community_count += 1
        
        # 执行社区检测
        if self.graph.number_of_nodes() > 0:
            try:
                self.detect_communities(algorithm="greedy")
            except Exception as e:
                print(f"⚠️ 社区检测失败: {e}")
        
        print(f"✅ JSON-LD 图谱加载完成:")
        print(f"   实体: {entity_count} 个 (实际节点: {self.graph.number_of_nodes()})")
        print(f"   关系: {relation_count} 个 (实际边: {self.graph.number_of_edges()})")
        print(f"   社区报告: {community_count} 个")
    
    def load_from_csv_triples(self, filepath: str):
        """
        从 CSV 三元组文件加载图谱
        
        CSV 格式: subject, predicate, object
        
        Args:
            filepath: CSV 文件路径
        """
        import csv
        
        def _extract_name(id_uri: str) -> str:
            return id_uri.rstrip("/").split("/")[-1]
        
        print(f"📂 正在从 CSV 三元组文件加载知识图谱: {filepath}")
        
        # 清空现有数据
        self.graph.clear()
        self.entities = {}
        self.communities = {}
        self.community_summaries = {}
        
        entity_count = 0
        relation_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过标题行
            
            for row in reader:
                if len(row) < 3:
                    continue
                
                subject_uri, predicate_uri, obj_value = row[0], row[1], row[2]
                subject_name = _extract_name(subject_uri)
                predicate_name = _extract_name(predicate_uri)
                
                # 判断 object 是 URI 还是字面值
                if obj_value.startswith("http://"):
                    object_name = _extract_name(obj_value)
                else:
                    object_name = obj_value.strip('@"')
                
                # 根据谓词类型处理
                if predicate_name == "type":
                    # 类型三元组: 添加实体
                    entity_type = _extract_name(obj_value)
                    if subject_name not in self.entities:
                        self.add_entity(
                            name=subject_name,
                            entity_type=entity_type,
                        )
                        entity_count += 1
                    else:
                        self.entities[subject_name]["type"] = entity_type
                        self.graph.nodes[subject_name]["type"] = entity_type
                        
                elif predicate_name == "label":
                    # 标签三元组: 更新实体名称
                    if subject_name in self.entities:
                        self.entities[subject_name]["description"] = object_name
                        
                elif predicate_name == "description":
                    # 描述三元组: 更新实体描述
                    if subject_name in self.entities:
                        self.entities[subject_name]["description"] = object_name
                        
                elif predicate_name in ("frequency", "degree"):
                    # 数值属性
                    if subject_name in self.entities:
                        try:
                            self.entities[subject_name][predicate_name] = float(object_name)
                            self.graph.nodes[subject_name][predicate_name] = float(object_name)
                        except ValueError:
                            pass
                            
                else:
                    # 关系三元组: 添加关系
                    # 确保 source 和 target 实体存在
                    if subject_name not in self.graph:
                        self.add_entity(name=subject_name, entity_type="UNKNOWN")
                        entity_count += 1
                    if object_name not in self.graph and not obj_value.startswith("http://schema.org/"):
                        self.add_entity(name=object_name, entity_type="UNKNOWN")
                        entity_count += 1
                    
                    if subject_name in self.graph and object_name in self.graph:
                        self.add_relation(
                            source=subject_name,
                            target=object_name,
                            relation_type=predicate_name,
                        )
                        relation_count += 1
        
        # 执行社区检测
        if self.graph.number_of_nodes() > 0:
            try:
                self.detect_communities(algorithm="greedy")
            except Exception as e:
                print(f"⚠️ 社区检测失败: {e}")
        
        print(f"✅ CSV 图谱加载完成:")
        print(f"   实体: {entity_count} 个 (实际节点: {self.graph.number_of_nodes()})")
        print(f"   关系: {relation_count} 个 (实际边: {self.graph.number_of_edges()})")


class CommunitySummarizer:
    """社区摘要生成器 - GraphRAG的关键组件"""
    
    def __init__(self):
        self.llm = create_chat_model(temperature=0.3)
        
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
