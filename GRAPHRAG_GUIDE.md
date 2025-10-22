# GraphRAG 集成指南

## 📋 概述

本项目已集成**Microsoft GraphRAG**架构，通过知识图谱增强传统向量检索，提供更精准的信息提取和推理能力。

## 🏗️ GraphRAG 架构

### 核心组件

```
文档集合
    ↓
┌─────────────────────────────────────┐
│  实体和关系提取 (Entity Extraction)   │
│  - 使用LLM识别实体                    │
│  - 提取实体间关系                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  知识图谱构建 (Graph Construction)    │
│  - 实体去重                          │
│  - 构建图结构                        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  社区检测 (Community Detection)       │
│  - Louvain算法                       │
│  - 层次化聚类                        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  社区摘要生成 (Community Summaries)   │
│  - LLM生成摘要                       │
│  - 多层次索引                        │
└─────────────────────────────────────┘
    ↓
    查询阶段
    ↓
┌──────────────┬──────────────┐
│  本地查询     │   全局查询    │
│ (Local Query)│(Global Query)│
│              │              │
│ 实体邻域检索  │  社区摘要查询 │
└──────────────┴──────────────┘
```

## 📦 新增文件说明

### 1. **entity_extractor.py** - 实体提取器
```python
EntityExtractor
├── extract_entities()      # 从文本提取实体
├── extract_relations()     # 提取实体关系
└── extract_from_document() # 完整文档处理

EntityDeduplicator
└── deduplicate_entities()  # 实体去重
```

**功能**:
- 使用LLM识别6种实体类型 (PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, PAPER, EVENT)
- 提取8种关系类型 (AUTHOR_OF, USES, BASED_ON, etc.)
- 智能实体去重和合并

### 2. **knowledge_graph.py** - 知识图谱核心
```python
KnowledgeGraph
├── add_entity()                 # 添加节点
├── add_relation()               # 添加边
├── build_from_extractions()     # 构建图谱
├── detect_communities()         # 社区检测
├── get_community_members()      # 获取社区成员
└── get_statistics()             # 统计信息

CommunitySummarizer
├── summarize_community()        # 单社区摘要
└── summarize_all_communities()  # 全部社区摘要
```

**功能**:
- 基于NetworkX的图谱管理
- 支持3种社区检测算法 (Louvain, Greedy, Label Propagation)
- LLM驱动的社区摘要生成
- 图谱持久化存储

### 3. **graph_indexer.py** - 索引构建器
```python
GraphRAGIndexer
├── index_documents()  # 构建索引
├── get_graph()        # 获取图谱
└── load_index()       # 加载索引
```

**流程**:
1. 批量实体提取
2. 实体去重合并
3. 构建知识图谱
4. 社区检测
5. 生成摘要

### 4. **graph_retriever.py** - 图谱检索器
```python
GraphRetriever
├── recognize_entities()  # 识别问题中的实体
├── local_query()         # 本地查询
├── global_query()        # 全局查询
├── hybrid_query()        # 混合查询
└── smart_query()         # 智能查询
```

**查询模式**:
- **本地查询**: 针对特定实体的详细问题
- **全局查询**: 需要整体理解的概括性问题
- **智能查询**: 自动选择最佳策略

### 5. **main_graphrag.py** - GraphRAG集成示例
完整的使用示例和交互式界面

### 6. **requirements_graphrag.txt** - 额外依赖
GraphRAG所需的图处理库

## 🚀 快速开始

### 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装GraphRAG依赖
pip install -r requirements_graphrag.txt
```

### 首次使用

```python
# 方式1: 使用集成示例
python main_graphrag.py

# 方式2: 在代码中集成
from config import setup_environment
from document_processor import initialize_document_processor
from graph_indexer import initialize_graph_indexer
from graph_retriever import initialize_graph_retriever

# 初始化
setup_environment()
processor, vectorstore, retriever, doc_splits = initialize_document_processor()

# 构建GraphRAG索引
graph_indexer = initialize_graph_indexer()
knowledge_graph = graph_indexer.index_documents(
    documents=doc_splits,
    save_path="./data/knowledge_graph.json"
)

# 初始化检索器
graph_retriever = initialize_graph_retriever(knowledge_graph)

# 查询
answer = graph_retriever.smart_query("LLM Agent的核心组件是什么？")
print(answer)
```

## 🔧 配置说明

在 `config.py` 中添加了以下配置:

```python
# GraphRAG配置
ENABLE_GRAPHRAG = True                           # 是否启用GraphRAG
GRAPHRAG_INDEX_PATH = "./data/knowledge_graph.json"  # 图谱存储路径
GRAPHRAG_COMMUNITY_ALGORITHM = "louvain"         # 社区检测算法
GRAPHRAG_MAX_HOPS = 2                            # 本地查询最大跳数
GRAPHRAG_TOP_K_COMMUNITIES = 5                   # 全局查询使用的社区数
GRAPHRAG_BATCH_SIZE = 10                         # 实体提取批大小
```

## 📊 使用场景对比

### 传统向量检索 vs GraphRAG

| 场景 | 向量检索 | GraphRAG | 推荐 |
|-----|---------|----------|------|
| "AlphaCodium的作者是谁？" | ⚠️ 可能找到但不精确 | ✅ 直接查询实体关系 | GraphRAG本地查询 |
| "这些文档讨论什么主题？" | ⚠️ 需要读取多个片段 | ✅ 社区摘要直接回答 | GraphRAG全局查询 |
| "提示工程的应用场景" | ✅ 语义匹配效果好 | ✅ 可追踪关系链 | 混合查询 |
| "最新技术发展" | ✅ 适合模糊查询 | ❌ 需要明确实体 | 向量检索 |

## 🎯 查询策略选择

### 本地查询 (Local Query)
**适用**: 针对特定实体的详细问题

```python
# 示例问题
"LLM Agent包含哪些组件？"
"Transformer模型的作者是谁？"
"AlphaCodium使用了什么技术？"

# 代码
answer = graph_retriever.local_query(question, max_hops=2)
```

**工作原理**:
1. 识别问题中的实体
2. 扩展到邻居节点（支持多跳）
3. 收集实体信息和关系
4. 基于子图生成答案

### 全局查询 (Global Query)
**适用**: 需要整体视角的概括性问题

```python
# 示例问题
"这些文档的主要主题是什么？"
"涵盖了哪些研究领域？"
"关键的技术趋势有哪些？"

# 代码
answer = graph_retriever.global_query(question, top_k_communities=5)
```

**工作原理**:
1. 获取社区摘要
2. 基于摘要理解全局结构
3. 综合多个社区的信息
4. 生成高层次答案

### 智能查询 (Smart Query)
**适用**: 自动选择最佳策略

```python
# 自动判断使用本地还是全局查询
answer = graph_retriever.smart_query(question)
```

**决策逻辑**:
- 包含具体实体名称 → 本地查询
- 包含"主要"、"总体"、"概述"等关键词 → 全局查询
- 默认 → 本地查询

### 混合查询 (Hybrid Query)
**适用**: 需要多种视角的复杂问题

```python
result = graph_retriever.hybrid_query(question)
# 返回: {"local": "...", "global": "..."}
```

## 📈 性能优化

### 索引构建优化

```python
# 1. 批处理大小
graph_indexer.index_documents(
    documents=doc_splits,
    batch_size=20  # 增大批处理提高速度
)

# 2. 增量索引（开发中）
# 避免每次重建整个图谱

# 3. 缓存已有索引
if os.path.exists(GRAPHRAG_INDEX_PATH):
    knowledge_graph = graph_indexer.load_index(GRAPHRAG_INDEX_PATH)
```

### 查询优化

```python
# 1. 调整跳数
answer = graph_retriever.local_query(question, max_hops=1)  # 减少跳数提速

# 2. 限制社区数量
answer = graph_retriever.global_query(question, top_k_communities=3)  # 减少社区数

# 3. 实体识别缓存（开发中）
```

## 🔍 调试和可视化

### 查看图谱统计

```python
stats = knowledge_graph.get_statistics()
print(f"节点数: {stats['num_nodes']}")
print(f"边数: {stats['num_edges']}")
print(f"社区数: {stats['num_communities']}")
```

### 导出图谱

```python
# 保存为JSON
knowledge_graph.save_to_file("my_graph.json")

# 加载图谱
knowledge_graph.load_from_file("my_graph.json")
```

### 可视化（可选）

```python
# 需要额外安装: pip install pyvis
from pyvis.network import Network

def visualize_graph(kg, output="graph.html"):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    for node, data in kg.graph.nodes(data=True):
        net.add_node(node, label=node, title=data.get('description', ''))
    
    for u, v, data in kg.graph.edges(data=True):
        net.add_edge(u, v, title=data.get('relation_type', ''))
    
    net.show(output)
    print(f"图谱已保存到: {output}")
```

## ⚠️ 常见问题

### Q1: 实体提取质量不高？
**A**: 
- 调整LLM温度参数
- 优化实体提取提示词
- 使用更强大的LLM模型

### Q2: 索引构建时间长？
**A**:
- 增大批处理大小
- 减少文档数量进行测试
- 使用缓存的索引文件

### Q3: 查询结果不相关？
**A**:
- 检查实体识别是否准确
- 调整查询策略（本地/全局）
- 增加邻居跳数

### Q4: 内存占用过大？
**A**:
- 使用更轻量的图数据库
- 分批处理大文档集
- 限制社区检测的迭代次数

## 🔄 与现有系统集成

### 修改现有 main.py

```python
from config import ENABLE_GRAPHRAG
from graph_indexer import initialize_graph_indexer
from graph_retriever import initialize_graph_retriever

class AdaptiveRAGSystem:
    def __init__(self):
        # ... 现有初始化代码 ...
        
        # 添加GraphRAG支持
        if ENABLE_GRAPHRAG:
            self._setup_graphrag()
    
    def _setup_graphrag(self):
        self.graph_indexer = initialize_graph_indexer()
        # ... 索引构建 ...
        self.graph_retriever = initialize_graph_retriever(self.knowledge_graph)
    
    def query(self, question: str):
        # 混合使用向量检索和图谱查询
        vector_docs = self.retriever.get_relevant_documents(question)
        
        if ENABLE_GRAPHRAG:
            graph_answer = self.graph_retriever.smart_query(question)
            # 融合两种结果
            return self._merge_results(vector_docs, graph_answer)
        
        return self._generate_from_docs(vector_docs)
```

## 📚 参考资料

- [Microsoft GraphRAG 论文](https://arxiv.org/abs/2404.16130)
- [NetworkX 文档](https://networkx.org/)
- [Louvain 社区检测算法](https://en.wikipedia.org/wiki/Louvain_method)

## 🛣️ 未来增强

- [ ] 增量索引更新
- [ ] 多模态知识图谱
- [ ] 图谱可视化界面
- [ ] Neo4j集成（生产环境）
- [ ] 知识图谱推理引擎
- [ ] 实体链接优化
- [ ] 自动实体消歧

---

**提示**: 首次使用建议先在小数据集上测试，验证效果后再应用到完整数据集。
