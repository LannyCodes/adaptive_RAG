# 高级重排器集成指南

## 📋 概述

本次更新添加了两个强大的重排器：

1. **ContextAwareReranker（上下文感知重排器）**
   - 考虑对话历史、用户偏好、查询意图
   - 动态调整文档排序以匹配上下文
   - 适合多轮对话场景

2. **MultiTaskReranker（多任务重排器）**
   - 同时优化5个目标：相关性、多样性、新颖性、权威性、时效性
   - 可自定义权重配置
   - 适合需要平衡多个目标的场景

## 🔧 使用方法

### 方法1: 使用工厂函数（推荐）

```python
from reranker import create_reranker

# 上下文感知重排
reranker = create_reranker(
    'context_aware',
    model_name='BAAI/bge-reranker-base',
    context_weight=0.3  # 上下文权重 (0-1)
)

results = reranker.rerank(
    query=query,
    documents=documents,
    top_k=5,
    context={
        'conversation_history': [...],
        'user_preferences': {...},
        'query_intent': {...}
    }
)

# 多任务重排
reranker = create_reranker(
    'multi_task',
    embeddings_model=embeddings,
    weights={
        'relevance': 0.35,
        'diversity': 0.25,
        'novelty': 0.15,
        'authority': 0.15,
        'recency': 0.10
    }
)

results = reranker.rerank(query, documents, top_k=5)
```

### 方法2: 直接实例化

```python
from reranker import ContextAwareReranker, MultiTaskReranker

# 上下文感知重排器
reranker = ContextAwareReranker(
    model_name='BAAI/bge-reranker-base',
    max_length=1024,
    context_weight=0.3
)

# 多任务重排器
reranker = MultiTaskReranker(
    embeddings_model=embeddings,
    weights={'relevance': 0.4, 'diversity': 0.2, ...},
    diversity_lambda=0.5
)
```

## 🎯 集成到工作流

### 在 workflow_nodes.py 中使用

修改 `WorkflowNodes` 类的 `retrieve` 方法：

```python
async def retrieve(self, state):
    """检索文档（使用高级重排器）"""
    question = state["question"]
    
    # 1. 基础检索
    documents = await self.doc_processor.async_enhanced_retrieve(
        question, 
        top_k=5, 
        rerank_candidates=20  # 先检索更多候选
    )
    
    # 2. 使用上下文感知重排
    if hasattr(self, 'context_reranker') and self.context_reranker:
        # 构建上下文
        context = {
            'conversation_history': state.get('history', []),
            'user_preferences': state.get('user_prefs', {}),
            'previous_documents': state.get('previous_docs', []),
            'query_intent': state.get('intent', {})
        }
        
        # 重排
        reranked = self.context_reranker.rerank(
            query=question,
            documents=documents,
            top_k=5,
            context=context
        )
        
        # 提取文档
        documents = [doc for doc, score in reranked]
    
    return {"documents": documents, "question": question}
```

### 在 document_processor.py 中集成

修改 `enhanced_retrieve` 方法：

```python
def enhanced_retrieve(self, query, top_k=5, rerank_candidates=20, 
                     use_advanced_reranker=True, context=None):
    """增强检索（支持高级重排器）"""
    
    # 1. 混合检索
    all_docs = self.hybrid_retrieve(query, rerank_candidates)
    
    # 2. 选择重排策略
    if use_advanced_reranker and self.advanced_reranker:
        if isinstance(self.advanced_reranker, ContextAwareReranker):
            results = self.advanced_reranker.rerank(
                query, all_docs, top_k, context=context
            )
        elif isinstance(self.advanced_reranker, MultiTaskReranker):
            results = self.advanced_reranker.rerank(
                query, all_docs, top_k
            )
        return [doc for doc, score in results]
    
    # 3. 回退到基础重排
    if self.reranker:
        results = self.reranker.rerank(query, all_docs, top_k)
        return [doc for doc, score in results]
    
    return all_docs[:top_k]
```

## 📊 场景选择指南

### 使用上下文感知重排的场景

✅ **多轮对话系统**
- 用户在与AI进行连续对话
- 需要考虑之前的对话内容
- 用户偏好会逐渐明确

✅ **个性化推荐**
- 不同用户有不同偏好
- 需要根据用户历史调整排序
- 避免重复推荐相似内容

✅ **专业领域问答**
- 用户意图明确（技术/概念/实践）
- 需要根据难度级别调整
- 考虑用户的专业背景

### 使用多任务重排的场景

✅ **知识搜索系统**
- 需要平衡相关性和多样性
- 希望覆盖多个子主题
- 避免结果过于集中

✅ **新闻/文章推荐**
- 时效性很重要
- 来源权威性需要考虑
- 希望提供新颖的内容

✅ **学术研究辅助**
- 权威性（引用次数）重要
- 新颖性（最新研究）重要
- 需要多样化的视角

## ⚙️ 参数调优

### 上下文感知重排参数

```python
context_weight = 0.3  # 上下文权重

# 调优建议：
# 0.1-0.2: 上下文影响较小，适合单轮对话
# 0.3-0.4: 平衡上下文和基础相关性（推荐）
# 0.5-0.6: 强上下文依赖，适合连续对话
# 0.7+:    几乎完全基于上下文，适合个性化推荐
```

### 多任务重排权重

```python
# 场景1: 通用搜索（推荐配置）
weights = {
    'relevance': 0.35,    # 相关性为主
    'diversity': 0.25,    # 兼顾多样性
    'novelty': 0.15,      # 适度新颖
    'authority': 0.15,    # 适度权威
    'recency': 0.10       # 轻微时效
}

# 场景2: 新闻搜索
weights = {
    'relevance': 0.30,
    'diversity': 0.20,
    'novelty': 0.10,
    'authority': 0.15,
    'recency': 0.25       # 强调时效性
}

# 场景3: 学术研究
weights = {
    'relevance': 0.30,
    'diversity': 0.15,
    'novelty': 0.20,      # 强调新颖性
    'authority': 0.25,    # 强调权威性
    'recency': 0.10
}

# 场景4: 学习教程
weights = {
    'relevance': 0.40,    # 高度相关
    'diversity': 0.20,
    'novelty': 0.10,
    'authority': 0.20,    # 权威来源
    'recency': 0.10
}
```

## 🔍 上下文构建示例

### 对话历史

```python
context = {
    'conversation_history': [
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是AI的一个分支..."},
        {"role": "user", "content": "有哪些常用算法？"}
    ]
}
```

### 用户偏好

```python
context = {
    'user_preferences': {
        'preferred_topics': ['深度学习', '神经网络', 'Python'],
        'avoid_topics': ['数学推导', '理论证明'],
        'difficulty_level': 'intermediate',
        'content_type': 'tutorial'  # tutorial/research/practical
    }
}
```

### 查询意图

```python
context = {
    'query_intent': {
        'type': 'technical',  # technical/conceptual/practical
        'difficulty': 'beginner',  # beginner/intermediate/advanced
        'urgency': 'normal',  # normal/urgent
        'depth': 'overview'  # overview/detailed/comprehensive
    }
}
```

## 📈 性能优化建议

### 1. 缓存策略

```python
# 缓存重排结果
from functools import lru_cache

class CachedReranker:
    def __init__(self, reranker):
        self.reranker = reranker
        self.cache = {}
    
    def rerank(self, query, documents, top_k=5, **kwargs):
        # 生成缓存键
        cache_key = f"{query}_{len(documents)}_{top_k}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = self.reranker.rerank(query, documents, top_k, **kwargs)
        self.cache[cache_key] = results
        return results
```

### 2. 批量处理

```python
# 批量重排多个查询
def batch_rerank(reranker, queries, documents_list, top_k=5):
    """批量重排"""
    all_results = []
    
    for query, docs in zip(queries, documents_list):
        results = reranker.rerank(query, docs, top_k)
        all_results.append(results)
    
    return all_results
```

### 3. 异步重排

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_rerank(reranker, query, documents, top_k=5, **kwargs):
    """异步重排"""
    loop = asyncio.get_running_loop()
    
    with ThreadPoolExecutor() as executor:
        results = await loop.run_in_executor(
            executor,
            lambda: reranker.rerank(query, documents, top_k, **kwargs)
        )
    
    return results
```

## 🧪 测试和评估

### 运行示例

```bash
# 运行高级重排器示例
python advanced_reranker_demo.py
```

### 评估指标

```python
from retrieval_evaluation import RetrievalEvaluator

# 比较不同重排器
rerankers = {
    'crossencoder': create_reranker('crossencoder'),
    'context_aware': create_reranker('context_aware', context_weight=0.3),
    'multi_task': create_reranker('multi_task', embeddings_model=embeddings)
}

for name, reranker in rerankers.items():
    results = evaluate_reranker(reranker, test_queries, test_documents)
    print(f"{name}: Precision@5={results['precision']:.4f}")
```

## ⚠️ 注意事项

1. **模型下载**: 首次使用会下载重排模型（约400MB）
2. **GPU内存**: 使用GPU时注意内存占用
3. **上下文大小**: 对话历史过长会影响性能，建议限制在最近5-10轮
4. **权重调整**: 根据实际场景调整权重，没有银弹配置

## 🚀 下一步

- 尝试不同的权重配置
- 在实际对话场景中测试
- 收集用户反馈优化参数
- 考虑集成学习-to-Rank模型

## 📚 参考资料

- CrossEncoder论文: https://arxiv.org/abs/1908.10084
- MMR算法: https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
- BGE Reranker: https://github.com/FlagOpen/FlagEmbedding
