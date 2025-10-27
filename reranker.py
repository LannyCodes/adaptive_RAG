"""
向量重排模块
实现多种重排策略以提高检索质量
支持 CrossEncoder 深度重排
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
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


def create_reranker(reranker_type: str, embeddings_model=None, **kwargs) -> DocumentReranker:
    """
    工厂函数：创建指定类型的重排器
    
    Args:
        reranker_type: 重排器类型
            - 'tfidf': TF-IDF 重排
            - 'bm25': BM25 重排
            - 'semantic': Bi-Encoder 语义重排
            - 'crossencoder': CrossEncoder 重排 (推荐) ⭐
            - 'hybrid': 混合重排
            - 'diversity': 多样性重排
        embeddings_model: 嵌入模型 (某些重排器需要)
        **kwargs: 其他参数
            - model_name: CrossEncoder 模型名称
            - max_length: CrossEncoder 最大长度
            - weights: 混合重排权重
    
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
        model_name = kwargs.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        max_length = kwargs.get('max_length', 512)
        return CrossEncoderReranker(model_name=model_name, max_length=max_length)
    
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
            f"Available types: tfidf, bm25, semantic, crossencoder, hybrid, diversity"
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