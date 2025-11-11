"""
文档处理和向量化模块
负责文档加载、文本分块、向量化和向量数据库初始化
"""

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from config import (
    KNOWLEDGE_BASE_URLS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    # 混合检索配置
    ENABLE_HYBRID_SEARCH,
    HYBRID_SEARCH_WEIGHTS,
    KEYWORD_SEARCH_K,
    BM25_K1,
    BM25_B,
    # 查询扩展配置
    ENABLE_QUERY_EXPANSION,
    QUERY_EXPANSION_MODEL,
    QUERY_EXPANSION_PROMPT,
    MAX_EXPANDED_QUERIES,
    # 多模态配置
    ENABLE_MULTIMODAL,
    MULTIMODAL_IMAGE_MODEL,
    SUPPORTED_IMAGE_FORMATS,
    IMAGE_EMBEDDING_DIM,
    MULTIMODAL_WEIGHTS
)
from reranker import create_reranker

# 多模态支持相关导入
import base64
import io
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union


class CustomEnsembleRetriever:
    """自定义集成检索器，结合向量检索和BM25检索"""
    
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights = weights
        
    def invoke(self, query):
        """执行检索并合并结果"""
        # 获取各检索器的结果
        all_results = []
        for i, retriever in enumerate(self.retrievers):
            results = retriever.invoke(query)
            for doc in results:
                # 添加检索器索引和权重信息
                doc.metadata["retriever_index"] = i
                doc.metadata["retriever_weight"] = self.weights[i]
                all_results.append(doc)
        
        # 根据权重排序并去重
        # 简单实现：先按检索器索引排序，再按权重排序
        all_results.sort(key=lambda x: (x.metadata["retriever_index"], -x.metadata["retriever_weight"]))
        
        # 去重（基于文档内容）
        unique_results = []
        seen_content = set()
        for doc in all_results:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_results.append(doc)
                
        return unique_results


class DocumentProcessor:
    """文档处理器类，负责文档加载、处理和向量化"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Try to initialize embeddings with error handling
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"✅ 检测到设备: {device}")
            if device == 'cuda':
                print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
                print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # 轻量级嵌入模型
                model_kwargs={'device': device},  # 自动选择GPU或CPU
                encode_kwargs={'normalize_embeddings': True}  # 标准化嵌入向量
            )
            print(f"✅ HuggingFace嵌入模型初始化成功 (设备: {device})")
        except Exception as e:
            print(f"⚠️ HuggingFace嵌入初始化失败: {e}")
            print("正在尝试备用嵌入方案...")
            # Fallback to OpenAI embeddings or other alternatives
            from langchain_community.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)  # For testing purposes
            print("✅ 使用测试嵌入模型")
            
        self.vectorstore = None
        self.retriever = None
        self.bm25_retriever = None  # BM25检索器
        self.ensemble_retriever = None  # 集成检索器
        
        # 初始化重排器
        self.reranker = None
        self._setup_reranker()
        
        # 初始化多模态支持
        self.image_embeddings_model = None
        self._setup_multimodal()
        
        # 初始化查询扩展
        self.query_expansion_model = None
        self._setup_query_expansion()
    
    def _setup_reranker(self):
        """
        设置重排器
        使用 CrossEncoder 提升重排准确率
        """
        try:
            # 使用 CrossEncoder 重排器 (准确率最高) ⭐
            print("🔧 正在初始化 CrossEncoder 重排器...")
            self.reranker = create_reranker(
                'crossencoder',
                model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',  # 轻量级模型
                max_length=512
            )
            print("✅ CrossEncoder 重排器初始化成功")
        except Exception as e:
            print(f"⚠️ CrossEncoder 初始化失败: {e}")
            print("🔄 尝试回退到混合重排器...")
            try:
                # 回退到混合重排器
                self.reranker = create_reranker('hybrid', self.embeddings)
                print("✅ 混合重排器初始化成功")
            except Exception as e2:
                print(f"⚠️ 重排器初始化完全失败: {e2}")
                print("⚠️ 将使用基础检索，不进行重排")
    
    def _setup_multimodal(self):
        """设置多模态支持"""
        if not ENABLE_MULTIMODAL:
            print("⚠️ 多模态支持已禁用")
            return
            
        try:
            print("🔧 正在初始化多模态支持...")
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.image_embeddings_model = CLIPModel.from_pretrained(MULTIMODAL_IMAGE_MODEL).to(device)
            self.image_processor = CLIPProcessor.from_pretrained(MULTIMODAL_IMAGE_MODEL)
            print(f"✅ 多模态支持初始化成功 (设备: {device})")
        except Exception as e:
            print(f"⚠️ 多模态支持初始化失败: {e}")
            print("⚠️ 将仅使用文本检索")
            self.image_embeddings_model = None
    
    def _setup_query_expansion(self):
        """设置查询扩展"""
        if not ENABLE_QUERY_EXPANSION:
            print("⚠️ 查询扩展已禁用")
            return
            
        try:
            print("🔧 正在初始化查询扩展...")
            from langchain_community.llms import Ollama
            
            self.query_expansion_model = Ollama(model=QUERY_EXPANSION_MODEL)
            print(f"✅ 查询扩展初始化成功 (模型: {QUERY_EXPANSION_MODEL})")
        except Exception as e:
            print(f"⚠️ 查询扩展初始化失败: {e}")
            print("⚠️ 将不使用查询扩展")
            self.query_expansion_model = None
    
    def load_documents(self, urls=None):
        """从URL加载文档"""
        if urls is None:
            urls = KNOWLEDGE_BASE_URLS
        
        print(f"正在加载 {len(urls)} 个URL的文档...")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        print(f"成功加载 {len(docs_list)} 个文档")
        return docs_list
    
    def split_documents(self, docs):
        """将文档分割成块"""
        print("正在分割文档...")
        doc_splits = self.text_splitter.split_documents(docs)
        print(f"文档分割完成，共 {len(doc_splits)} 个文档块")
        return doc_splits
    
    def create_vectorstore(self, doc_splits):
        """创建向量数据库"""
        print("正在创建向量数据库...")
        self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever()
        
        # 如果启用混合检索，创建BM25检索器和集成检索器
        if ENABLE_HYBRID_SEARCH:
            print("正在初始化混合检索...")
            try:
                # 创建BM25检索器
                self.bm25_retriever = BM25Retriever.from_documents(
                    doc_splits, 
                    k=KEYWORD_SEARCH_K,
                    k1=BM25_K1,
                    b=BM25_B
                )
                
                # 创建集成检索器，结合向量检索和BM25检索
                self.ensemble_retriever = CustomEnsembleRetriever(
                    retrievers=[self.retriever, self.bm25_retriever],
                    weights=[HYBRID_SEARCH_WEIGHTS["vector"], HYBRID_SEARCH_WEIGHTS["keyword"]]
                )
                print("✅ 混合检索初始化成功")
            except Exception as e:
                print(f"⚠️ 混合检索初始化失败: {e}")
                print("⚠️ 将仅使用向量检索")
                self.ensemble_retriever = None
        
        print("向量数据库创建完成")
        return self.vectorstore, self.retriever
    
    def setup_knowledge_base(self, urls=None, enable_graphrag=False):
        """设置完整的知识库（加载、分割、向量化）
        
        Args:
            urls: 文档URL列表
            enable_graphrag: 是否启用GraphRAG索引
            
        Returns:
            vectorstore, retriever, doc_splits
        """
        docs = self.load_documents(urls)
        doc_splits = self.split_documents(docs)
        vectorstore, retriever = self.create_vectorstore(doc_splits)
        
        # 返回doc_splits用于GraphRAG索引
        return vectorstore, retriever, doc_splits
    
    def expand_query(self, query: str) -> List[str]:
        """扩展查询，生成相关查询"""
        if not self.query_expansion_model:
            return [query]
            
        try:
            # 使用LLM生成扩展查询
            prompt = QUERY_EXPANSION_PROMPT.format(query=query)
            expanded_queries_text = self.query_expansion_model.invoke(prompt)
            
            # 解析扩展查询
            expanded_queries = [query]  # 包含原始查询
            for line in expanded_queries_text.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # 移除可能的编号前缀
                    if line[0].isdigit() and '.' in line[:5]:
                        line = line.split('.', 1)[1].strip()
                    expanded_queries.append(line)
            
            # 限制扩展查询数量
            return expanded_queries[:MAX_EXPANDED_QUERIES + 1]  # +1 因为包含原始查询
        except Exception as e:
            print(f"⚠️ 查询扩展失败: {e}")
            return [query]
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """编码图像为嵌入向量"""
        if not self.image_embeddings_model:
            raise ValueError("多模态支持未初始化")
            
        try:
            # 加载并处理图像
            image = Image.open(image_path).convert('RGB')
            inputs = self.image_processor(images=image, return_tensors="pt")
            
            # 获取图像嵌入
            with torch.no_grad():
                image_features = self.image_embeddings_model.get_image_features(**inputs)
                # 标准化嵌入向量
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ 图像编码失败: {e}")
            raise
    
    def multimodal_retrieve(self, query: str, image_paths: List[str] = None, top_k: int = 5) -> List:
        """多模态检索，结合文本和图像"""
        if not ENABLE_MULTIMODAL or not self.image_embeddings_model:
            # 如果多模态未启用，回退到文本检索
            return self.hybrid_retrieve(query, top_k) if ENABLE_HYBRID_SEARCH else self.retriever.invoke(query)[:top_k]
        
        # 文本检索
        text_docs = self.hybrid_retrieve(query, top_k) if ENABLE_HYBRID_SEARCH else self.retriever.invoke(query)[:top_k]
        
        # 如果没有提供图像，直接返回文本检索结果
        if not image_paths:
            return text_docs
            
        try:
            # 图像检索
            image_results = []
            for image_path in image_paths:
                # 检查文件格式
                file_ext = image_path.split('.')[-1].lower()
                if file_ext not in SUPPORTED_IMAGE_FORMATS:
                    print(f"⚠️ 不支持的图像格式: {file_ext}")
                    continue
                    
                # 编码图像
                image_embedding = self.encode_image(image_path)
                
                # 这里应该实现图像到文本的匹配逻辑
                # 由于原始实现中没有图像数据库，我们简化处理
                # 在实际应用中，应该有一个图像数据库和相应的检索逻辑
                
            # 合并文本和图像结果（简化版本）
            # 在实际应用中，应该有更复杂的融合逻辑
            final_docs = text_docs  # 简化版本，仅返回文本结果
            
            print(f"✅ 多模态检索完成，返回 {len(final_docs)} 个结果")
            return final_docs
        except Exception as e:
            print(f"⚠️ 多模态检索失败: {e}")
            print("回退到文本检索")
            return text_docs
    
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List:
        """混合检索，结合向量检索和关键词检索"""
        if not ENABLE_HYBRID_SEARCH or not self.ensemble_retriever:
            # 如果混合检索未启用，回退到向量检索
            return self.retriever.invoke(query)[:top_k]
            
        try:
            # 使用集成检索器进行混合检索
            results = self.ensemble_retriever.invoke(query)
            return results[:top_k]
        except Exception as e:
            print(f"⚠️ 混合检索失败: {e}")
            print("回退到向量检索")
            return self.retriever.invoke(query)[:top_k]
    
    def enhanced_retrieve(self, query: str, top_k: int = 5, rerank_candidates: int = 20, 
                         image_paths: List[str] = None, use_query_expansion: bool = None):
        """增强检索：先检索更多候选，然后重排，支持查询扩展和多模态
        
        Args:
            query: 查询字符串
            top_k: 返回的文档数量
            rerank_candidates: 重排前的候选文档数量
            image_paths: 图像路径列表，用于多模态检索
            use_query_expansion: 是否使用查询扩展，None表示使用配置默认值
        """
        # 确定是否使用查询扩展
        if use_query_expansion is None:
            use_query_expansion = ENABLE_QUERY_EXPANSION
            
        # 如果启用查询扩展，生成扩展查询
        if use_query_expansion:
            expanded_queries = self.expand_query(query)
            print(f"查询扩展: {len(expanded_queries)} 个查询")
        else:
            expanded_queries = [query]
            
        # 多模态检索（如果提供了图像）
        if image_paths and ENABLE_MULTIMODAL:
            return self.multimodal_retrieve(query, image_paths, top_k)
            
        # 混合检索或向量检索
        all_candidate_docs = []
        for expanded_query in expanded_queries:
            if ENABLE_HYBRID_SEARCH:
                # 使用混合检索
                docs = self.hybrid_retrieve(expanded_query, rerank_candidates)
            else:
                # 使用向量检索
                docs = self.retriever.invoke(expanded_query)
                if len(docs) > rerank_candidates:
                    docs = docs[:rerank_candidates]
            
            all_candidate_docs.extend(docs)
            
        # 去重（基于文档内容）
        unique_docs = []
        seen_content = set()
        for doc in all_candidate_docs:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)
                
        print(f"检索获得 {len(unique_docs)} 个候选文档")
        
        # 重排（如果重排器可用）
        if self.reranker and len(unique_docs) > top_k:
            try:
                reranked_results = self.reranker.rerank(query, unique_docs, top_k)
                final_docs = [doc for doc, score in reranked_results]
                scores = [score for doc, score in reranked_results]
                
                print(f"重排后返回 {len(final_docs)} 个文档")
                print(f"重排分数范围: {min(scores):.4f} - {max(scores):.4f}")
                
                return final_docs
            except Exception as e:
                print(f"⚠️ 重排失败: {e}，使用原始检索结果")
                return unique_docs[:top_k]
        else:
            # 不重排或候选数量不足
            return unique_docs[:top_k]
    
    def compare_retrieval_methods(self, query: str, top_k: int = 5, image_paths: List[str] = None):
        """比较不同检索方法的效果"""
        if not self.retriever:
            return {}
        
        results = {
            'query': query,
            'image_paths': image_paths
        }
        
        # 原始检索 (使用 invoke 替代 get_relevant_documents)
        original_docs = self.retriever.invoke(query)[:top_k]
        results['vector_retrieval'] = {
            'count': len(original_docs),
            'documents': [{
                'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                'metadata': getattr(doc, 'metadata', {})
            } for doc in original_docs]
        }
        
        # 混合检索（如果启用）
        if ENABLE_HYBRID_SEARCH and self.ensemble_retriever:
            hybrid_docs = self.hybrid_retrieve(query, top_k)
            results['hybrid_retrieval'] = {
                'count': len(hybrid_docs),
                'documents': [{
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': getattr(doc, 'metadata', {})
                } for doc in hybrid_docs]
            }
        
        # 查询扩展检索（如果启用）
        if ENABLE_QUERY_EXPANSION and self.query_expansion_model:
            expanded_docs = self.enhanced_retrieve(query, top_k, use_query_expansion=True)
            results['expanded_query_retrieval'] = {
                'count': len(expanded_docs),
                'documents': [{
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': getattr(doc, 'metadata', {})
                } for doc in expanded_docs]
            }
        
        # 多模态检索（如果启用且有图像）
        if ENABLE_MULTIMODAL and image_paths:
            multimodal_docs = self.multimodal_retrieve(query, image_paths, top_k)
            results['multimodal_retrieval'] = {
                'count': len(multimodal_docs),
                'documents': [{
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': getattr(doc, 'metadata', {})
                } for doc in multimodal_docs]
            }
        
        # 增强检索（带重排）
        enhanced_docs = self.enhanced_retrieve(query, top_k)
        results['enhanced_retrieval'] = {
            'count': len(enhanced_docs),
            'documents': [{
                'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                'metadata': getattr(doc, 'metadata', {})
            } for doc in enhanced_docs]
        }
        
        # 添加配置信息
        results['configuration'] = {
            'hybrid_search_enabled': ENABLE_HYBRID_SEARCH,
            'query_expansion_enabled': ENABLE_QUERY_EXPANSION,
            'multimodal_enabled': ENABLE_MULTIMODAL,
            'reranker_used': self.reranker is not None,
            'hybrid_weights': HYBRID_SEARCH_WEIGHTS if ENABLE_HYBRID_SEARCH else None,
            'multimodal_weights': MULTIMODAL_WEIGHTS if ENABLE_MULTIMODAL else None
        }
        
        return results

    def format_docs(self, docs):
        """格式化文档用于生成"""
        return "\n\n".join(doc.page_content for doc in docs)


def initialize_document_processor():
    """初始化文档处理器并设置知识库"""
    processor: DocumentProcessor = DocumentProcessor()
    vectorstore, retriever, doc_splits = processor.setup_knowledge_base()
    return processor, vectorstore, retriever, doc_splits