"""
向量重排模块
实现多种重排策略以提高检索质量
支持 CrossEncoder 深度重排
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import math

# CrossEncoder support
try:
    from sentence_transformers import CrossEncoder as SentenceTransformerCrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    print("⚠️ sentence-transformers not available. CrossEncoder reranking disabled.")


class DocumentReranker:
    """文档重排器基类"""
    
    def __init__(self):
        self.name = "BaseReranker"
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        """重排文档并返回top_k结果"""
        raise NotImplementedError


class TFIDFReranker(DocumentReranker):
    """基于TF-IDF的重排器"""
    
    def __init__(self):
        super().__init__()
        self.name = "TFIDFReranker"
        # 移除 stop_words 以支持中文，使用 char_wb 分词器
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # 字符级分词，支持中文
            ngram_range=(2, 4),  # 2-4 字符 n-gram
            max_features=5000
        )
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        """使用TF-IDF重新排序文档"""
        if not documents:
            return []
        
        # 提取文档内容
        doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        all_texts = [query] + doc_texts
        
        # 计算TF-IDF矩阵
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        query_vec = tfidf_matrix[0]
        doc_vecs = tfidf_matrix[1:]
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()
        
        # 排序并返回top_k
        ranked_indices = np.argsort(similarities)[::-1]
        results = []
        for i in ranked_indices[:top_k]:
            results.append((documents[i], float(similarities[i])))
        
        return results


class BM25Reranker(DocumentReranker):
    """基于BM25算法的重排器"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__()
        self.name = "BM25Reranker"
        self.k1 = k1
        self.b = b
    
    def _tokenize(self, text: str) -> List[str]:
        """
        改进的分词，支持中英文
        中文使用字符级分词，英文使用单词分词
        """
        # 检测是否包含中文
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if has_chinese:
            # 中文：使用字符级 + 2-gram
            chars = list(text.lower())
            # 生成 unigram 和 bigram
            tokens = chars + [chars[i] + chars[i+1] for i in range(len(chars)-1)]
            return [t for t in tokens if t.strip()]  # 移除空格
        else:
            # 英文：使用单词分词
            return re.findall(r'\b\w+\b', text.lower())
    
    def _compute_idf(self, documents: List[str], query_terms: List[str]) -> Dict[str, float]:
        """计算IDF值"""
        N = len(documents)
        idf = {}
        
        for term in query_terms:
            df = sum(1 for doc in documents if term in self._tokenize(doc))
            idf[term] = math.log((N - df + 0.5) / (df + 0.5))
        
        return idf
    
    def _bm25_score(self, query_terms: List[str], document: str, avg_doc_len: float, idf: Dict[str, float]) -> float:
        """计算BM25分数"""
        doc_terms = self._tokenize(document)
        doc_len = len(doc_terms)
        term_freq = Counter(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]
                score += idf.get(term, 0) * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)
                )
        
        return score
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        """使用BM25重新排序文档"""
        if not documents:
            return []
        
        query_terms = self._tokenize(query)
        doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        
        # 计算平均文档长度
        avg_doc_len = sum(len(self._tokenize(doc)) for doc in doc_texts) / len(doc_texts)
        
        # 计算IDF
        idf = self._compute_idf(doc_texts, query_terms)
        
        # 计算BM25分数
        scores = []
        for doc_text in doc_texts:
            score = self._bm25_score(query_terms, doc_text, avg_doc_len, idf)
            scores.append(score)
        
        # 排序并返回top_k
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for i in ranked_indices[:top_k]:
            results.append((documents[i], float(scores[i])))
        
        return results


class SemanticReranker(DocumentReranker):
    """基于语义相似度的重排器"""
    
    def __init__(self, embeddings_model):
        super().__init__()
        self.name = "SemanticReranker"
        self.embeddings_model = embeddings_model
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        """使用语义相似度重新排序文档"""
        if not documents:
            return []
        
        # 获取查询嵌入
        query_embedding = self.embeddings_model.embed_query(query)
        
        # 获取文档嵌入
        doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        doc_embeddings = self.embeddings_model.embed_documents(doc_texts)
        
        # 计算余弦相似度
        similarities = []
        for doc_emb in doc_embeddings:
            sim = cosine_similarity([query_embedding], [doc_emb])[0][0]
            similarities.append(sim)
        
        # 排序并返回top_k
        ranked_indices = np.argsort(similarities)[::-1]
        results = []
        for i in ranked_indices[:top_k]:
            results.append((documents[i], float(similarities[i])))
        
        return results


class CrossEncoderReranker(DocumentReranker):
    """
    基于 CrossEncoder 的重排器
    使用联合编码，相比 Bi-Encoder 准确率提升 15-20%
    适合精排阶段 (Top 20-100 文档)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512):
        """
        初始化 CrossEncoder 重排器
        
        Args:
            model_name: 模型名称，默认使用轻量级模型
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (轻量级，推荐)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (平衡)
                - "BAAI/bge-reranker-base" (中文优化)
                - "BAAI/bge-reranker-large" (高精度)
            max_length: 最大输入长度
        """
        super().__init__()
        self.name = "CrossEncoderReranker"
        self.model_name = model_name
        self.max_length = max_length
        
        # 加载模型
        if not CROSSENCODER_AVAILABLE:
            raise ImportError(
                "CrossEncoder requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
        
        try:
            print(f"🔧 加载 CrossEncoder 模型: {model_name}...")
            self.model = SentenceTransformerCrossEncoder(model_name, max_length=max_length)
            print(f"✅ CrossEncoder 模型加载成功")
        except Exception as e:
            print(f"❌ CrossEncoder 模型加载失败: {e}")
            raise
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        使用 CrossEncoder 重新排序文档
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回结果数量
            
        Returns:
            排序后的 (document, score) 元组列表
        """
        if not documents:
            return []
        
        # 提取文档内容
        doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        
        # 构造 [query, doc] 对
        query_doc_pairs = [[query, doc_text] for doc_text in doc_texts]
        
        # CrossEncoder 评分 - 联合编码
        try:
            scores = self.model.predict(query_doc_pairs)
            
            # 排序
            ranked_indices = np.argsort(scores)[::-1]
            
            # 返回 top_k 结果
            results = []
            for i in ranked_indices[:top_k]:
                results.append((documents[i], float(scores[i])))
            
            return results
            
        except Exception as e:
            print(f"⚠️ CrossEncoder 重排失败: {e}")
            # 回退到原始顺序
            return [(doc, 0.0) for doc in documents[:top_k]]


class HybridReranker(DocumentReranker):
    """混合重排器，融合多种策略"""
    
    def __init__(self, embeddings_model, weights: Dict[str, float] = None):
        super().__init__()
        self.name = "HybridReranker"
        
        # 初始化各种重排器
        self.tfidf_reranker = TFIDFReranker()
        self.bm25_reranker = BM25Reranker()
        self.semantic_reranker = SemanticReranker(embeddings_model)
        
        # 设置权重
        self.weights = weights or {
            'tfidf': 0.3,
            'bm25': 0.3,
            'semantic': 0.4
        }
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        """使用混合策略重新排序文档"""
        if not documents:
            return []
        
        # 获取各种重排结果
        tfidf_results = self.tfidf_reranker.rerank(query, documents, len(documents))
        bm25_results = self.bm25_reranker.rerank(query, documents, len(documents))
        semantic_results = self.semantic_reranker.rerank(query, documents, len(documents))
        
        # 创建文档到分数的映射
        doc_scores = {}
        for doc in documents:
            doc_id = id(doc)
            doc_scores[doc_id] = {'doc': doc, 'tfidf': 0, 'bm25': 0, 'semantic': 0}
        
        # 填充各种分数
        for doc, score in tfidf_results:
            doc_scores[id(doc)]['tfidf'] = score
        
        for doc, score in bm25_results:
            doc_scores[id(doc)]['bm25'] = score
        
        for doc, score in semantic_results:
            doc_scores[id(doc)]['semantic'] = score
        
        # 归一化分数
        for score_type in ['tfidf', 'bm25', 'semantic']:
            scores = [info[score_type] for info in doc_scores.values()]
            if max(scores) > 0:
                max_score = max(scores)
                for doc_id in doc_scores:
                    doc_scores[doc_id][score_type] /= max_score
        
        # 计算综合分数
        final_scores = []
        for doc_id, info in doc_scores.items():
            combined_score = (
                self.weights['tfidf'] * info['tfidf'] +
                self.weights['bm25'] * info['bm25'] +
                self.weights['semantic'] * info['semantic']
            )
            final_scores.append((info['doc'], combined_score))
        
        # 排序并返回top_k
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]


class DiversityReranker(DocumentReranker):
    """多样性重排器，避免结果重复"""
    
    def __init__(self, embeddings_model, diversity_lambda: float = 0.5):
        super().__init__()
        self.name = "DiversityReranker"
        self.embeddings_model = embeddings_model
        self.diversity_lambda = diversity_lambda
    
    def _calculate_diversity_penalty(self, candidate_doc: str, selected_docs: List[str]) -> float:
        """计算多样性惩罚"""
        if not selected_docs:
            return 0.0
        
        candidate_emb = self.embeddings_model.embed_documents([candidate_doc])[0]
        selected_embs = self.embeddings_model.embed_documents(selected_docs)
        
        max_similarity = 0.0
        for selected_emb in selected_embs:
            sim = cosine_similarity([candidate_emb], [selected_emb])[0][0]
            max_similarity = max(max_similarity, sim)
        
        return max_similarity
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        """使用多样性策略重新排序文档"""
        if not documents:
            return []
        
        # 首先使用语义相似度获取初始排序
        semantic_results = SemanticReranker(self.embeddings_model).rerank(
            query, documents, len(documents)
        )
        
        # MMR (Maximal Marginal Relevance) 算法
        selected_docs = []
        selected_texts = []
        remaining_docs = [doc for doc, _ in semantic_results]
        relevance_scores = {id(doc): score for doc, score in semantic_results}
        
        while len(selected_docs) < top_k and remaining_docs:
            best_score = -1
            best_doc = None
            best_idx = -1
            
            for i, doc in enumerate(remaining_docs):
                doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                relevance = relevance_scores[id(doc)]
                diversity_penalty = self._calculate_diversity_penalty(doc_text, selected_texts)
                
                # MMR分数 = λ * 相关性 - (1-λ) * 多样性惩罚
                mmr_score = (
                    self.diversity_lambda * relevance - 
                    (1 - self.diversity_lambda) * diversity_penalty
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
                    best_idx = i
            
            if best_doc is not None:
                selected_docs.append((best_doc, best_score))
                selected_texts.append(
                    best_doc.page_content if hasattr(best_doc, 'page_content') else str(best_doc)
                )
                remaining_docs.pop(best_idx)
        
        return selected_docs


class ContextAwareReranker(DocumentReranker):
    """
    上下文感知重排器
    考虑对话历史、用户偏好和查询上下文进行重排
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-reranker-base", 
                 max_length: int = 1024,
                 context_weight: float = 0.3):
        """
        初始化上下文感知重排器
        
        Args:
            model_name: CrossEncoder模型名称
            max_length: 最大输入长度
            context_weight: 上下文分数的权重 (0-1)
        """
        super().__init__()
        self.name = "ContextAwareReranker"
        self.context_weight = context_weight
        
        if not CROSSENCODER_AVAILABLE:
            raise ImportError("ContextAwareReranker requires sentence-transformers")
        
        try:
            print(f"🔧 加载上下文感知重排模型: {model_name}...")
            self.model = SentenceTransformerCrossEncoder(model_name, max_length=max_length)
            print(f"✅ 上下文感知重排模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def rerank(self, 
               query: str, 
               documents: List[dict], 
               top_k: int = 5,
               context: Optional[Dict[str, Any]] = None) -> List[Tuple[dict, float]]:
        """
        上下文感知重排
        
        Args:
            query: 当前查询
            documents: 候选文档列表
            top_k: 返回结果数量
            context: 上下文信息，包含:
                - conversation_history: 对话历史列表
                - user_preferences: 用户偏好
                - previous_documents: 之前检索到的文档
                - query_intent: 查询意图分类
        
        Returns:
            排序后的 (document, score) 元组列表
        """
        if not documents:
            return []
        
        # 1. 基础相关性评分 (CrossEncoder)
        doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        query_doc_pairs = [[query, doc_text] for doc_text in doc_texts]
        
        try:
            base_scores = self.model.predict(query_doc_pairs)
        except Exception as e:
            print(f"⚠️ CrossEncoder评分失败: {e}")
            base_scores = np.zeros(len(documents))
        
        # 2. 计算上下文相关分数
        context_scores = self._calculate_context_scores(query, documents, context)
        
        # 3. 归一化分数到 [0, 1]
        base_scores_norm = self._normalize_scores(base_scores)
        context_scores_norm = self._normalize_scores(context_scores)
        
        # 4. 融合分数
        final_scores = (1 - self.context_weight) * base_scores_norm + self.context_weight * context_scores_norm
        
        # 5. 排序并返回
        ranked_indices = np.argsort(final_scores)[::-1]
        results = []
        for i in ranked_indices[:top_k]:
            results.append((documents[i], float(final_scores[i])))
        
        return results
    
    def _calculate_context_scores(self, 
                                  query: str, 
                                  documents: List[dict], 
                                  context: Optional[Dict[str, Any]]) -> np.ndarray:
        """计算上下文相关分数"""
        n_docs = len(documents)
        context_scores = np.zeros(n_docs)
        
        if context is None:
            return context_scores
        
        # 2.1 对话历史相关性
        if 'conversation_history' in context and context['conversation_history']:
            history = context['conversation_history']
            history_text = ' '.join([str(h.get('content', '')) for h in history[-5:]])  # 最近5轮
            
            for i, doc in enumerate(documents):
                doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                # 计算与对话历史的词汇重叠
                overlap = self._calculate_text_overlap(history_text, doc_text)
                context_scores[i] += overlap * 0.4
        
        # 2.2 用户偏好匹配
        if 'user_preferences' in context and context['user_preferences']:
            preferences = context['user_preferences']
            for i, doc in enumerate(documents):
                doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                doc_lower = doc_text.lower()
                
                # 检查偏好关键词
                pref_score = 0.0
                for pref in preferences.get('preferred_topics', []):
                    if pref.lower() in doc_lower:
                        pref_score += 0.3
                
                for avoid in preferences.get('avoid_topics', []):
                    if avoid.lower() in doc_lower:
                        pref_score -= 0.3
                
                context_scores[i] += max(0, pref_score) * 0.3
        
        # 2.3 文档多样性惩罚
        if 'previous_documents' in context and context['previous_documents']:
            prev_docs = context['previous_documents']
            prev_texts = [d.page_content if hasattr(d, 'page_content') else str(d) for d in prev_docs]
            
            for i, doc in enumerate(documents):
                doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                max_similarity = 0.0
                
                for prev_text in prev_texts:
                    similarity = self._calculate_text_similarity(doc_text, prev_text)
                    max_similarity = max(max_similarity, similarity)
                
                # 与之前文档越相似，分数越低（鼓励多样性）
                context_scores[i] -= max_similarity * 0.3
        
        # 2.4 查询意图匹配
        if 'query_intent' in context and context['query_intent']:
            intent = context['query_intent']
            for i, doc in enumerate(documents):
                doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                
                # 根据意图类型调整分数
                if intent.get('type') == 'technical' and any(kw in doc_text.lower() for kw in ['实现', '算法', '代码', 'technical']):
                    context_scores[i] += 0.2
                elif intent.get('type') == 'conceptual' and any(kw in doc_text.lower() for kw in ['概念', '原理', '定义', 'concept']):
                    context_scores[i] += 0.2
        
        return context_scores
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """计算文本词汇重叠度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版Jaccard）"""
        # 使用n-gram相似度
        n = 3
        ngrams1 = set([text1[i:i+n] for i in range(len(text1)-n+1)])
        ngrams2 = set([text2[i:i+n] for i in range(len(text2)-n+1)])
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数到 [0, 1]"""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        
        return (scores - min_score) / (max_score - min_score)


class MultiTaskReranker(DocumentReranker):
    """
    多任务重排器
    同时优化多个目标：相关性、多样性、新颖性、权威性等
    """
    
    def __init__(self,
                 embeddings_model,
                 weights: Optional[Dict[str, float]] = None,
                 diversity_lambda: float = 0.5):
        """
        初始化多任务重排器
        
        Args:
            embeddings_model: 嵌入模型（用于语义计算）
            weights: 各任务权重
                - relevance: 相关性权重
                - diversity: 多样性权重
                - novelty: 新颖性权重
                - authority: 权威性权重
                - recency: 时效性权重
            diversity_lambda: MMR算法的多样性权衡参数 (0-1)
        """
        super().__init__()
        self.name = "MultiTaskReranker"
        self.embeddings_model = embeddings_model
        self.diversity_lambda = diversity_lambda
        
        # 默认权重
        self.weights = weights or {
            'relevance': 0.40,      # 相关性
            'diversity': 0.20,      # 多样性
            'novelty': 0.15,        # 新颖性
            'authority': 0.15,      # 权威性
            'recency': 0.10         # 时效性
        }
        
        # 验证权重和为1
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            print(f"⚠️ 权重和为 {weight_sum}，将自动归一化")
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}
    
    def rerank(self, 
               query: str, 
               documents: List[dict], 
               top_k: int = 5,
               metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[dict, float]]:
        """
        多任务重排
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回结果数量
            metadata: 文档元数据（包含权威性、时效性等信息）
        
        Returns:
            排序后的 (document, score) 元组列表
        """
        if not documents:
            return []
        
        n_docs = len(documents)
        doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        
        # 1. 相关性评分 (Semantic Similarity)
        relevance_scores = self._calculate_relevance_scores(query, doc_texts)
        
        # 2. 多样性评分 (使用MMR算法)
        diversity_scores = self._calculate_diversity_scores(query, documents, doc_texts)
        
        # 3. 新颖性评分 (与已知信息的新颖程度)
        novelty_scores = self._calculate_novelty_scores(doc_texts, metadata)
        
        # 4. 权威性评分 (基于来源可信度)
        authority_scores = self._calculate_authority_scores(documents, metadata)
        
        # 5. 时效性评分 (基于文档时间)
        recency_scores = self._calculate_recency_scores(documents, metadata)
        
        # 6. 归一化所有分数
        scores_dict = {
            'relevance': self._normalize_scores(relevance_scores),
            'diversity': self._normalize_scores(diversity_scores),
            'novelty': self._normalize_scores(novelty_scores),
            'authority': self._normalize_scores(authority_scores),
            'recency': self._normalize_scores(recency_scores)
        }
        
        # 7. 加权融合
        final_scores = np.zeros(n_docs)
        for task_name, weight in self.weights.items():
            if task_name in scores_dict:
                final_scores += weight * scores_dict[task_name]
        
        # 8. 排序并返回
        ranked_indices = np.argsort(final_scores)[::-1]
        results = []
        for i in ranked_indices[:top_k]:
            results.append((documents[i], float(final_scores[i])))
        
        return results
    
    def _calculate_relevance_scores(self, query: str, doc_texts: List[str]) -> np.ndarray:
        """计算语义相关性分数"""
        try:
            query_emb = self.embeddings_model.encode([query], convert_to_numpy=True)
            doc_embs = self.embeddings_model.encode(doc_texts, convert_to_numpy=True)
            
            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_emb, doc_embs).flatten()
            return similarities
        except Exception as e:
            print(f"⚠️ 相关性计算失败: {e}")
            return np.zeros(len(doc_texts))
    
    def _calculate_diversity_scores(self, query: str, documents: List[dict], doc_texts: List[str]) -> np.ndarray:
        """使用MMR算法计算多样性分数"""
        n_docs = len(doc_texts)
        if n_docs <= 1:
            return np.ones(n_docs)
        
        try:
            # 计算文档间的相似度矩阵
            doc_embs = self.embeddings_model.encode(doc_texts, convert_to_numpy=True)
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(doc_embs)
            
            # MMR分数: λ * relevance - (1-λ) * max_similarity_to_selected
            # 这里简化为与所有其他文档的平均相似度
            diversity_scores = np.zeros(n_docs)
            for i in range(n_docs):
                # 与其他文档的平均相似度（排除自己）
                avg_sim = (np.sum(sim_matrix[i]) - sim_matrix[i][i]) / (n_docs - 1)
                diversity_scores[i] = 1 - avg_sim  # 相似度越低，多样性越高
            
            return diversity_scores
        except Exception as e:
            print(f"⚠️ 多样性计算失败: {e}")
            return np.ones(n_docs) * 0.5
    
    def _calculate_novelty_scores(self, doc_texts: List[str], metadata: Optional[Dict]) -> np.ndarray:
        """计算新颖性分数（文档间的独特性）"""
        n_docs = len(doc_texts)
        if n_docs <= 1:
            return np.ones(n_docs)
        
        try:
            # 计算文档间的n-gram重叠
            novelty_scores = np.zeros(n_docs)
            
            for i in range(n_docs):
                # 提取n-grams
                n = 4
                ngrams_i = set([doc_texts[i][j:j+n] for j in range(len(doc_texts[i])-n+1)])
                
                # 计算与其他文档的最小重叠（越大越不新颖）
                min_overlap = 1.0
                for j in range(n_docs):
                    if i == j:
                        continue
                    
                    ngrams_j = set([doc_texts[j][k:k+n] for k in range(len(doc_texts[j])-n+1)])
                    
                    if ngrams_i and ngrams_j:
                        overlap = len(ngrams_i & ngrams_j) / len(ngrams_i | ngrams_j)
                        min_overlap = min(min_overlap, overlap)
                
                # 重叠越小，新颖性越高
                novelty_scores[i] = 1 - min_overlap
            
            return novelty_scores
        except Exception as e:
            print(f"⚠️ 新颖性计算失败: {e}")
            return np.ones(n_docs) * 0.5
    
    def _calculate_authority_scores(self, documents: List[dict], metadata: Optional[Dict]) -> np.ndarray:
        """计算权威性分数"""
        n_docs = len(documents)
        authority_scores = np.ones(n_docs) * 0.5  # 默认中等权威性
        
        try:
            for i, doc in enumerate(documents):
                score = 0.5  # 基础分数
                
                # 检查元数据中的权威性指标
                if hasattr(doc, 'metadata') and doc.metadata:
                    meta = doc.metadata
                    
                    # 来源权威性
                    source = meta.get('source', '')
                    if any(domain in source.lower() for domain in ['edu', 'gov', 'org', 'wikipedia', 'official']):
                        score += 0.3
                    
                    # 引用次数
                    citations = meta.get('citations', 0)
                    if citations > 0:
                        score += min(0.2, citations / 100)  # 最多+0.2
                    
                    # 作者权威性
                    author = meta.get('author', '')
                    if author and len(author) > 0:
                        score += 0.1
                
                authority_scores[i] = min(1.0, score)
        except Exception as e:
            print(f"⚠️ 权威性计算失败: {e}")
        
        return authority_scores
    
    def _calculate_recency_scores(self, documents: List[dict], metadata: Optional[Dict]) -> np.ndarray:
        """计算时效性分数"""
        import time
        from datetime import datetime
        
        n_docs = len(documents)
        recency_scores = np.ones(n_docs) * 0.5  # 默认中等时效性
        
        try:
            current_time = time.time()
            one_year_seconds = 365 * 24 * 3600
            
            for i, doc in enumerate(documents):
                score = 0.5
                
                if hasattr(doc, 'metadata') and doc.metadata:
                    meta = doc.metadata
                    
                    # 检查时间戳
                    timestamp = meta.get('timestamp') or meta.get('date') or meta.get('publish_date')
                    
                    if timestamp:
                        # 尝试解析不同格式的时间
                        try:
                            if isinstance(timestamp, str):
                                # 尝试解析日期字符串
                                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%Y-%m-%d %H:%M:%S']:
                                    try:
                                        doc_time = datetime.strptime(timestamp, fmt).timestamp()
                                        break
                                    except:
                                        continue
                                else:
                                    doc_time = current_time
                            else:
                                doc_time = float(timestamp)
                            
                            # 计算时间差（年）
                            years_diff = (current_time - doc_time) / one_year_seconds
                            
                            # 指数衰减：越近分数越高
                            score = np.exp(-0.5 * years_diff)
                        except:
                            score = 0.5
                
                recency_scores[i] = max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ 时效性计算失败: {e}")
        
        return recency_scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数到 [0, 1]"""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        
        return (scores - min_score) / (max_score - min_score)


def create_reranker(reranker_type: str, embeddings_model=None, **kwargs) -> DocumentReranker:
    """
    工厂函数：创建指定类型的重排器
    
    Args:
        reranker_type: 重排器类型
            - 'tfidf': TF-IDF 重排
            - 'bm25': BM25 重排
            - 'semantic': Bi-Encoder 语义重排
            - 'crossencoder': CrossEncoder 重排 (推荐) ⭐
            - 'context_aware': 上下文感知重排 ⭐⭐
            - 'multi_task': 多任务重排 ⭐⭐
            - 'hybrid': 混合重排
            - 'diversity': 多样性重排
        embeddings_model: 嵌入模型 (某些重排器需要)
        **kwargs: 其他参数
            - model_name: CrossEncoder 模型名称
            - max_length: CrossEncoder 最大长度
            - weights: 混合重排权重
            - context_weight: 上下文权重 (context_aware)
            - diversity_lambda: 多样性权衡 (multi_task)
    
    Returns:
        DocumentReranker: 重排器实例
    """
    
    if reranker_type.lower() == 'tfidf':
        return TFIDFReranker()
    
    elif reranker_type.lower() == 'bm25':
        return BM25Reranker(**kwargs)
    
    elif reranker_type.lower() == 'semantic':
        if embeddings_model is None:
            raise ValueError("SemanticReranker requires embeddings_model")
        return SemanticReranker(embeddings_model)
    
    elif reranker_type.lower() in ['crossencoder', 'cross_encoder', 'cross-encoder']:
        # CrossEncoder 不需要 embeddings_model，使用自己的模型
        model_name = kwargs.get('model_name', 'BAAI/bge-reranker-base')
        max_length = kwargs.get('max_length', 1024)
        return CrossEncoderReranker(model_name=model_name, max_length=max_length)
    
    elif reranker_type.lower() == 'context_aware':
        model_name = kwargs.get('model_name', 'BAAI/bge-reranker-base')
        max_length = kwargs.get('max_length', 1024)
        context_weight = kwargs.get('context_weight', 0.3)
        return ContextAwareReranker(
            model_name=model_name, 
            max_length=max_length,
            context_weight=context_weight
        )
    
    elif reranker_type.lower() == 'multi_task':
        if embeddings_model is None:
            raise ValueError("MultiTaskReranker requires embeddings_model")
        weights = kwargs.get('weights', None)
        diversity_lambda = kwargs.get('diversity_lambda', 0.5)
        return MultiTaskReranker(
            embeddings_model=embeddings_model,
            weights=weights,
            diversity_lambda=diversity_lambda
        )
    
    elif reranker_type.lower() == 'hybrid':
        if embeddings_model is None:
            raise ValueError("HybridReranker requires embeddings_model")
        return HybridReranker(embeddings_model, **kwargs)
    
    elif reranker_type.lower() == 'diversity':
        if embeddings_model is None:
            raise ValueError("DiversityReranker requires embeddings_model")
        return DiversityReranker(embeddings_model, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown reranker type: {reranker_type}. "
            f"Available types: tfidf, bm25, semantic, crossencoder, context_aware, multi_task, hybrid, diversity"
        )


# 使用示例
if __name__ == "__main__":
    # 模拟文档
    class MockDoc:
        def __init__(self, content):
            self.page_content = content
    
    docs = [
        MockDoc("人工智能是计算机科学的一个分支"),
        MockDoc("机器学习是人工智能的子领域"),
        MockDoc("深度学习使用神经网络"),
        MockDoc("自然语言处理处理文本数据"),
        MockDoc("今天天气很好")
    ]
    
    query = "什么是人工智能？"
    
    # 测试TF-IDF重排
    tfidf_reranker = TFIDFReranker()
    results = tfidf_reranker.rerank(query, docs, top_k=3)
    
    print("TF-IDF重排结果:")
    for doc, score in results:
        print(f"分数: {score:.4f} - 内容: {doc.page_content}")