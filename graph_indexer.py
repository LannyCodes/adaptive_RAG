"""
GraphRAG索引器
负责构建层次化的知识图谱索引，包括实体提取、图谱构建、社区检测和摘要生成
"""

from typing import List, Dict, Optional
import asyncio
try:
    from langchain_core.documents import Document
except ImportError:
    try:
    from langchain_core.documents import Document
except ImportError:
    try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from entity_extractor import EntityExtractor, EntityDeduplicator
from knowledge_graph import KnowledgeGraph, CommunitySummarizer


class GraphRAGIndexer:
    """GraphRAG索引器 - 实现Microsoft GraphRAG的索引流程"""
    
    def __init__(self, enable_async: bool = True, async_batch_size: int = 8):
        """初始化GraphRAG索引器
        
        Args:
            enable_async: 是否启用异步处理（默认启用）
            async_batch_size: 异步并发批次大小（默认5个文档并发）
        """
        print("🚀 初始化GraphRAG索引器...")
        
        self.entity_extractor = EntityExtractor(enable_async=enable_async)
        self.entity_deduplicator = EntityDeduplicator()
        self.knowledge_graph = KnowledgeGraph()
        self.community_summarizer = CommunitySummarizer()
        
        self.enable_async = enable_async
        self.async_batch_size = async_batch_size
        self.indexed = False
        
        mode = "异步模式" if enable_async else "同步模式"
        print(f"✅ GraphRAG索引器初始化完成 ({mode}, 并发数={async_batch_size})")
    
    def index_documents(self, documents: List[Document], 
                       batch_size: int = 10,
                       save_path: Optional[str] = None) -> KnowledgeGraph:
        """
        对文档集合建立GraphRAG索引
        
        工作流程（遵循Microsoft GraphRAG）:
        1. 文档分块（已在document_processor中完成）
        2. 实体和关系提取
        3. 实体去重和合并
        4. 构建知识图谱
        5. 社区检测
        6. 生成社区摘要
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
            save_path: 保存路径
            
        Returns:
            构建好的知识图谱
        """
        print(f"\n{'='*50}")
        print(f"📊 开始GraphRAG索引流程")
        print(f"   文档数量: {len(documents)}")
        print(f"{'='*50}\n")
        
        # 步骤1: 实体和关系提取
        print("📍 步骤 1/5: 实体和关系提取")
        extraction_results = []
        
        if self.enable_async:
            # 异步批量处理模式
            print(f"🚀 使用异步处理模式，并发数={self.async_batch_size}")
            extraction_results = self._extract_async(documents)
        else:
            # 同步处理模式（原有逻辑）
            print("🔄 使用同步处理模式")
            total_batches = (len(documents) - 1) // batch_size + 1
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                batch_num = i // batch_size + 1
                print(f"\n⚙️  === 批次 {batch_num}/{total_batches} (文档 {i+1}-{min(i+batch_size, len(documents))}) ===")
                
                for idx, doc in enumerate(batch):
                    doc_global_index = i + idx
                    try:
                        result = self.entity_extractor.extract_from_document(
                            doc.page_content, 
                            doc_index=doc_global_index
                        )
                        extraction_results.append(result)
                    except Exception as e:
                        print(f"   ❌ 文档 #{doc_global_index + 1} 处理失败: {e}")
                        # 添加空结果以保持索引一致
                        extraction_results.append({"entities": [], "relations": []})
                
                print(f"✅ 批次 {batch_num}/{total_batches} 完成")
        
        # 步骤2: 实体去重
        print("\n📍 步骤 2/5: 实体去重和合并")
        all_entities = []
        all_relations = []
        
        for result in extraction_results:
            all_entities.extend(result.get("entities", []))
            all_relations.extend(result.get("relations", []))
        
        dedup_result = self.entity_deduplicator.deduplicate_entities(all_entities)
        unique_entities = dedup_result["entities"]
        entity_mapping = dedup_result["mapping"]
        
        # 更新关系中的实体名称
        mapped_relations = []
        for relation in all_relations:
            source = entity_mapping.get(relation["source"], relation["source"])
            target = entity_mapping.get(relation["target"], relation["target"])
            mapped_relations.append({
                **relation,
                "source": source,
                "target": target
            })
        
        # 步骤3: 构建知识图谱
        print("\n📍 步骤 3/5: 构建知识图谱")
        cleaned_results = [{
            "entities": unique_entities,
            "relations": mapped_relations
        }]
        self.knowledge_graph.build_from_extractions(cleaned_results)
        
        # 步骤4: 社区检测
        print("\n📍 步骤 4/5: 社区检测")
        self.knowledge_graph.detect_communities(algorithm="louvain")
        
        # 步骤5: 生成社区摘要
        print("\n📍 步骤 5/5: 生成社区摘要")
        self.community_summarizer.summarize_all_communities(self.knowledge_graph)
        
        # 保存图谱
        if save_path:
            self.knowledge_graph.save_to_file(save_path)
        
        self.indexed = True
        
        # 打印统计信息
        print(f"\n{'='*50}")
        print("✅ GraphRAG索引构建完成!")
        stats = self.knowledge_graph.get_statistics()
        print(f"\n📊 统计信息:")
        print(f"   - 节点数: {stats['num_nodes']}")
        print(f"   - 边数: {stats['num_edges']}")
        print(f"   - 社区数: {stats['num_communities']}")
        print(f"   - 图密度: {stats['density']:.4f}")
        print(f"\n   实体类型分布:")
        for etype, count in stats['entity_types'].items():
            print(f"     • {etype}: {count}")
        print(f"{'='*50}\n")
        
        return self.knowledge_graph
    
    def _extract_async(self, documents: List[Document]) -> List[Dict]:
        """异步批量提取实体和关系
        
        Args:
            documents: 文档列表
            
        Returns:
            提取结果列表
        """
        total_docs = len(documents)
        extraction_results = []
        
        # 将文档分成多个异步批次
        for i in range(0, total_docs, self.async_batch_size):
            batch_end = min(i + self.async_batch_size, total_docs)
            batch_num = i // self.async_batch_size + 1
            total_batches = (total_docs - 1) // self.async_batch_size + 1
            
            print(f"\n⚡ === 异步批次 {batch_num}/{total_batches} (文档 {i+1}-{batch_end}) ===")
            
            # 准备异步批次数据
            async_batch = [
                (documents[idx].page_content, idx) 
                for idx in range(i, batch_end)
            ]
            
            # 异步执行当前批次
            try:
                batch_results = asyncio.run(
                    main=self.entity_extractor.extract_batch_async(async_batch)
                )
                extraction_results.extend(batch_results)
                print(f"✅ 异步批次 {batch_num}/{total_batches} 完成")
            except Exception as e:
                print(f"❌ 异步批次 {batch_num} 失败: {e}")
                # 添加空结果
                for _ in range(len(async_batch)):
                    extraction_results.append({"entities": [], "relations": []})
        
        return extraction_results
    
    def get_graph(self) -> KnowledgeGraph:
        """获取知识图谱"""
        if not self.indexed:
            print("⚠️ 图谱尚未构建，请先调用 index_documents()")
        return self.knowledge_graph
    
    def load_index(self, filepath: str) -> KnowledgeGraph:
        """加载已有的图谱索引"""
        print(f"📂 从文件加载图谱索引: {filepath}")
        self.knowledge_graph.load_from_file(filepath)
        self.indexed = True
        return self.knowledge_graph


def initialize_graph_indexer():
    """初始化GraphRAG索引器"""
    return GraphRAGIndexer()
