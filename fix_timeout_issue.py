"""
临时修复超时问题的脚本
在 Colab 中运行此脚本来增加超时时间
"""

import sys
import os

# 确保路径正确
sys.path.insert(0, '/content/drive/MyDrive/adaptive_RAG')

print("🔧 修复超时问题...")
print("="*60)

# 方案 1: 修改 entity_extractor 的超时设置
print("\n📝 方案 1: 增加超时时间和重试次数")
print("-"*60)

# 重新导入并修改
from entity_extractor import EntityExtractor, EntityDeduplicator
from graph_indexer import GraphRAGIndexer

# 创建自定义的 GraphRAG 索引器，使用更长的超时
class GraphRAGIndexerWithLongerTimeout(GraphRAGIndexer):
    """增加超时时间的 GraphRAG 索引器"""
    
    def __init__(self, timeout=180, max_retries=5):
        """
        初始化索引器，使用更长的超时时间
        
        Args:
            timeout: 超时时间（秒），默认180秒（3分钟）
            max_retries: 最大重试次数，默认5次
        """
        print(f"🚀 初始化GraphRAG索引器（超时: {timeout}秒, 重试: {max_retries}次）...")
        
        # 使用更长的超时初始化实体提取器
        self.entity_extractor = EntityExtractor(
            timeout=timeout,
            max_retries=max_retries
        )
        self.entity_deduplicator = EntityDeduplicator()
        
        # 导入其他必要的类
        from knowledge_graph import KnowledgeGraph, CommunitySummarizer
        self.knowledge_graph = KnowledgeGraph()
        self.community_summarizer = CommunitySummarizer()
        
        self.indexed = False
        
        print("✅ GraphRAG索引器初始化完成")


# 方案 2: 提供快速重启脚本
print("\n📝 方案 2: 重启 Ollama 服务")
print("-"*60)
print("运行以下命令:")
print("  !pkill -9 ollama")
print("  !sleep 2")
print("  !nohup ollama serve > /tmp/ollama.log 2>&1 &")
print("  !sleep 5")
print("  !curl http://localhost:11434/api/tags")


# 方案 3: 跳过当前文档
print("\n📝 方案 3: 跳过问题文档并继续")
print("-"*60)
print("如果某个文档持续失败，可以跳过它:")
print("""
# 示例：从文档 #57 开始继续处理
problem_doc_index = 55  # 文档 #56 的索引
doc_splits_filtered = doc_splits[:problem_doc_index] + doc_splits[problem_doc_index+1:]

# 使用过滤后的文档列表
graph = indexer.index_documents(
    documents=doc_splits_filtered,
    batch_size=3
)
""")


# 使用示例
print("\n" + "="*60)
print("✅ 修复方案准备完成")
print("="*60)
print("\n💡 推荐的使用方法:")
print("-"*60)

usage_example = """
# 1. 导入修复后的索引器
from fix_timeout_issue import GraphRAGIndexerWithLongerTimeout

# 2. 使用更长的超时时间（3分钟）创建索引器
indexer = GraphRAGIndexerWithLongerTimeout(
    timeout=180,      # 3分钟超时
    max_retries=5     # 5次重试
)

# 3. 减小批次大小，继续处理
# 如果已经处理了部分文档，可以跳过它们
processed_count = 55  # 已处理到文档 #55

remaining_docs = doc_splits[processed_count:]

graph = indexer.index_documents(
    documents=remaining_docs,
    batch_size=3,  # 更小的批次
    save_path="/content/drive/MyDrive/knowledge_graph.pkl"
)

# 4. 如果还是超时，考虑跳过问题文档
# problem_indices = [55]  # 文档 #56 的索引
# remaining_docs_filtered = [doc for i, doc in enumerate(doc_splits[processed_count:]) 
#                            if (processed_count + i) not in problem_indices]
"""

print(usage_example)

print("\n" + "="*60)
print("🎯 立即执行的步骤:")
print("="*60)
print("""
1️⃣ 首先重启 Ollama 服务:
   !pkill -9 ollama && sleep 2 && nohup ollama serve > /tmp/ollama.log 2>&1 & && sleep 5

2️⃣ 然后使用更长的超时时间继续:
   from fix_timeout_issue import GraphRAGIndexerWithLongerTimeout
   indexer = GraphRAGIndexerWithLongerTimeout(timeout=180, max_retries=5)
   
3️⃣ 从文档 #56 继续处理（减小批次大小）:
   remaining_docs = doc_splits[55:]  # 从文档 #56 开始
   graph = indexer.index_documents(remaining_docs, batch_size=3)

4️⃣ 如果文档 #56 仍然超时，跳过它:
   remaining_docs = doc_splits[56:]  # 跳过文档 #56，从 #57 开始
   graph = indexer.index_documents(remaining_docs, batch_size=3)
""")

print("\n⚠️ 注意:")
print("  • 超时通常说明文档内容复杂或 Ollama 负载过重")
print("  • 重启 Ollama 通常能解决负载问题")
print("  • 增加超时时间（180秒）能处理复杂文档")
print("  • 减小批次大小（3个文档/批次）能减轻负载")
print("  • 如果某个文档持续失败，可以考虑跳过它")
