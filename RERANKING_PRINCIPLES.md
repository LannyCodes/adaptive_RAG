# 向量重排（Vector Reranking）原理详解

## 🎯 什么是向量重排

向量重排是检索增强生成（RAG）系统中的一种高级技术，用于在初始向量检索后对候选文档进行重新排序，以提高最终检索结果的质量和相关性。

## 🔍 为什么需要重排

### 初始检索的局限性

1. **语义距离偏差**
   - 向量相似度可能无法完全捕捉语义相关性
   - 某些相关文档可能因为表达方式不同而排名靠后

2. **上下文理解不足**
   - 简单的余弦相似度无法理解复杂的查询意图
   - 缺乏对查询和文档交互关系的深度理解

3. **多样性问题**
   - 初始检索可能返回内容相似的重复文档
   - 缺乏结果的多样性和全面性

## 🧠 重排的核心原理

### 1. 双阶段检索架构

```
查询 → 粗排（向量检索）→ 精排（重排模型）→ 最终结果
     ↓                  ↓
   召回候选集        重新排序打分
   (100-1000篇)      (选择前k篇)
```

### 2. 重排模型类型

#### A. 交叉编码器（Cross-Encoder）
```python
# 原理示意
def cross_encoder_rerank(query, documents):
    scores = []
    for doc in documents:
        # 查询和文档一起编码
        input_text = f"[CLS] {query} [SEP] {doc} [SEP]"
        score = model(input_text)  # 直接输出相关性分数
        scores.append(score)
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
```

#### B. 双编码器重排（Bi-Encoder Reranking）
```python
def bi_encoder_rerank(query, documents):
    query_embedding = query_encoder(query)
    doc_embeddings = [doc_encoder(doc) for doc in documents]
    
    # 使用更复杂的相似度计算
    scores = []
    for doc_emb in doc_embeddings:
        score = complex_similarity(query_embedding, doc_emb)
        scores.append(score)
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
```

## 🔬 重排算法详解

### 1. 基于机器学习的重排

#### Learning to Rank (LTR)
```python
class LearnToRankReranker:
    def __init__(self):
        self.model = None  # XGBoost, LambdaMART等
    
    def extract_features(self, query, document):
        """提取查询-文档特征"""
        features = [
            # 文本匹配特征
            jaccard_similarity(query, document),
            tf_idf_score(query, document),
            bm25_score(query, document),
            
            # 语义特征
            cosine_similarity(query_emb, doc_emb),
            bert_score(query, document),
            
            # 文档特征
            document_length(document),
            document_quality_score(document),
            
            # 查询特征
            query_complexity(query),
            query_type_classification(query)
        ]
        return features
    
    def rerank(self, query, documents):
        features_matrix = []
        for doc in documents:
            features = self.extract_features(query, doc)
            features_matrix.append(features)
        
        scores = self.model.predict(features_matrix)
        return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
```

### 2. 基于深度学习的重排

#### Transformer重排模型
```python
class TransformerReranker:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def rerank(self, query, documents, top_k=5):
        scores = []
        
        for doc in documents:
            # 构造输入
            inputs = self.tokenizer(
                query, doc,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 获取相关性分数
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = torch.softmax(outputs.logits, dim=-1)[0][1]  # 相关性概率
                scores.append(score.item())
        
        # 重新排序
        ranked_results = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked_results[:top_k]
```

### 3. 多策略融合重排

```python
class MultiStrategyReranker:
    def __init__(self):
        self.semantic_weight = 0.4
        self.lexical_weight = 0.3
        self.diversity_weight = 0.2
        self.freshness_weight = 0.1
    
    def rerank(self, query, documents):
        # 1. 语义相关性分数
        semantic_scores = self.compute_semantic_scores(query, documents)
        
        # 2. 词汇匹配分数
        lexical_scores = self.compute_lexical_scores(query, documents)
        
        # 3. 多样性分数
        diversity_scores = self.compute_diversity_scores(documents)
        
        # 4. 时效性分数
        freshness_scores = self.compute_freshness_scores(documents)
        
        # 5. 加权融合
        final_scores = []
        for i in range(len(documents)):
            score = (
                self.semantic_weight * semantic_scores[i] +
                self.lexical_weight * lexical_scores[i] +
                self.diversity_weight * diversity_scores[i] +
                self.freshness_weight * freshness_scores[i]
            )
            final_scores.append(score)
        
        return sorted(zip(documents, final_scores), key=lambda x: x[1], reverse=True)
```

## 🎛️ 重排特征工程

### 1. 文本匹配特征
```python
def extract_text_features(query, document):
    return {
        # 精确匹配
        'exact_match_ratio': exact_match_count(query, document) / len(query.split()),
        
        # 模糊匹配
        'fuzzy_match_score': fuzz.ratio(query, document) / 100,
        
        # N-gram重叠
        'bigram_overlap': ngram_overlap(query, document, n=2),
        'trigram_overlap': ngram_overlap(query, document, n=3),
        
        # TF-IDF相似度
        'tfidf_similarity': tfidf_cosine_similarity(query, document),
        
        # BM25分数
        'bm25_score': compute_bm25(query, document)
    }
```

### 2. 语义特征
```python
def extract_semantic_features(query, document, embeddings):
    query_emb = embeddings['query']
    doc_emb = embeddings['document']
    
    return {
        # 余弦相似度
        'cosine_similarity': cosine_sim(query_emb, doc_emb),
        
        # 欧几里得距离
        'euclidean_distance': euclidean_distance(query_emb, doc_emb),
        
        # 曼哈顿距离
        'manhattan_distance': manhattan_distance(query_emb, doc_emb),
        
        # BERT分数
        'bert_score': bert_score_f1(query, document),
        
        # 语义角度
        'semantic_angle': semantic_angle(query_emb, doc_emb)
    }
```

### 3. 文档质量特征
```python
def extract_quality_features(document):
    return {
        # 长度特征
        'doc_length': len(document.split()),
        'sentence_count': len(sent_tokenize(document)),
        
        # 可读性特征
        'readability_score': textstat.flesch_reading_ease(document),
        'complexity_score': textstat.flesch_kincaid_grade(document),
        
        # 信息密度
        'unique_word_ratio': len(set(document.split())) / len(document.split()),
        'stopword_ratio': stopword_count(document) / len(document.split()),
        
        # 结构特征
        'has_headers': bool(re.search(r'^#+\s', document, re.MULTILINE)),
        'has_lists': bool(re.search(r'^\s*[-*+]\s', document, re.MULTILINE))
    }
```

## 🚀 实际应用示例

### 集成到RAG系统中
```python
class AdaptiveRAGWithReranking:
    def __init__(self):
        self.initial_retriever = VectorRetriever()
        self.reranker = TransformerReranker()
        self.generator = LanguageModel()
    
    def query(self, question, top_k=5, rerank_candidates=20):
        # 1. 初始检索（获取更多候选）
        initial_docs = self.initial_retriever.retrieve(
            question, 
            top_k=rerank_candidates
        )
        
        # 2. 重排
        reranked_docs = self.reranker.rerank(
            question, 
            initial_docs, 
            top_k=top_k
        )
        
        # 3. 生成答案
        context = "\n\n".join([doc[0] for doc in reranked_docs])
        answer = self.generator.generate(question, context)
        
        return {
            'answer': answer,
            'sources': reranked_docs,
            'confidence': self.calculate_confidence(reranked_docs)
        }
```

## 📊 性能评估指标

### 1. 排序质量指标
```python
def evaluate_reranking(original_ranking, reranked_results, ground_truth):
    return {
        # NDCG (Normalized Discounted Cumulative Gain)
        'ndcg@5': ndcg_score(ground_truth, reranked_results, k=5),
        'ndcg@10': ndcg_score(ground_truth, reranked_results, k=10),
        
        # MAP (Mean Average Precision)
        'map': mean_average_precision(ground_truth, reranked_results),
        
        # MRR (Mean Reciprocal Rank)
        'mrr': mean_reciprocal_rank(ground_truth, reranked_results),
        
        # 排序改进度
        'ranking_improvement': kendall_tau(original_ranking, reranked_results)
    }
```

### 2. 端到端效果评估
```python
def evaluate_rag_with_reranking(test_questions, ground_truth_answers):
    results = []
    
    for question, gt_answer in zip(test_questions, ground_truth_answers):
        # 无重排
        original_answer = rag_without_rerank(question)
        
        # 有重排
        reranked_answer = rag_with_rerank(question)
        
        results.append({
            'question': question,
            'original_score': evaluate_answer(original_answer, gt_answer),
            'reranked_score': evaluate_answer(reranked_answer, gt_answer),
            'improvement': evaluate_answer(reranked_answer, gt_answer) - 
                          evaluate_answer(original_answer, gt_answer)
        })
    
    return results
```

## 💡 最佳实践

### 1. 重排策略选择
- **实时性要求高**: 使用轻量级规则或简单ML模型
- **精度要求高**: 使用深度学习重排模型
- **平衡性能**: 多策略融合 + 缓存优化

### 2. 特征选择原则
- **相关性特征**: 语义相似度、词汇匹配
- **质量特征**: 文档权威性、完整性
- **多样性特征**: 避免结果冗余
- **时效性特征**: 信息新鲜度

### 3. 系统优化
```python
class OptimizedReranker:
    def __init__(self):
        self.cache = LRUCache(maxsize=1000)
        self.batch_size = 32
    
    @lru_cache(maxsize=1000)
    def cached_rerank(self, query_hash, doc_hashes):
        """缓存重排结果"""
        pass
    
    def batch_rerank(self, queries, documents):
        """批量重排优化"""
        pass
```

重排向量是提升RAG系统检索精度的关键技术，通过多层次的相关性评估和智能排序，显著提高了最终答案的质量和准确性。