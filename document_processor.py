"""
文档处理和向量化模块
负责文档加载、文本分块、向量化和向量数据库初始化
"""

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    KNOWLEDGE_BASE_URLS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)
from reranker import create_reranker


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
        
        # 初始化重排器
        self.reranker = None
        self._setup_reranker()
    
    def _setup_reranker(self):
        """设置重排器"""
        try:
            # 使用混合重排器获得最佳效果
            self.reranker = create_reranker('hybrid', self.embeddings)
            print("✅ 重排器初始化成功")
        except Exception as e:
            print(f"⚠️ 重排器初始化失败: {e}")
            print("将使用基础检索，不进行重排")
    
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
    
    def enhanced_retrieve(self, query: str, top_k: int = 5, rerank_candidates: int = 20):
        """增强检索：先检索更多候选，然后重排"""
        if not self.retriever:
            print("⚠️ 检索器未初始化")
            return []
        
        # 1. 初始检索：获取更多候选文档
        initial_docs = self.retriever.get_relevant_documents(query)
        
        # 获取更多候选（如果可能）
        if hasattr(self.retriever, 'search_kwargs'):
            # 修改检索参数以获取更多结果
            original_k = self.retriever.search_kwargs.get('k', 4)
            self.retriever.search_kwargs['k'] = min(rerank_candidates, len(initial_docs))
            candidate_docs = self.retriever.get_relevant_documents(query)
            self.retriever.search_kwargs['k'] = original_k  # 恢复原设置
        else:
            candidate_docs = initial_docs
        
        print(f"初始检索获得 {len(candidate_docs)} 个候选文档")
        
        # 2. 重排（如果重排器可用）
        if self.reranker and len(candidate_docs) > top_k:
            try:
                reranked_results = self.reranker.rerank(query, candidate_docs, top_k)
                final_docs = [doc for doc, score in reranked_results]
                scores = [score for doc, score in reranked_results]
                
                print(f"重排后返回 {len(final_docs)} 个文档")
                print(f"重排分数范围: {min(scores):.4f} - {max(scores):.4f}")
                
                return final_docs
            except Exception as e:
                print(f"⚠️ 重排失败: {e}，使用原始检索结果")
                return candidate_docs[:top_k]
        else:
            # 不重排或候选数量不足
            return candidate_docs[:top_k]
    
    def compare_retrieval_methods(self, query: str, top_k: int = 5):
        """比较不同检索方法的效果"""
        if not self.retriever:
            return {}
        
        # 原始检索
        original_docs = self.retriever.get_relevant_documents(query)[:top_k]
        
        # 增强检索（带重排）
        enhanced_docs = self.enhanced_retrieve(query, top_k)
        
        return {
            'query': query,
            'original_retrieval': {
                'count': len(original_docs),
                'documents': [{
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': getattr(doc, 'metadata', {})
                } for doc in original_docs]
            },
            'enhanced_retrieval': {
                'count': len(enhanced_docs),
                'documents': [{
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': getattr(doc, 'metadata', {})
                } for doc in enhanced_docs]
            },
            'reranker_used': self.reranker is not None
        }

    def format_docs(self, docs):
        """格式化文档用于生成"""
        return "\n\n".join(doc.page_content for doc in docs)


def initialize_document_processor():
    """初始化文档处理器并设置知识库"""
    processor: DocumentProcessor = DocumentProcessor()
    vectorstore, retriever, doc_splits = processor.setup_knowledge_base()
    return processor, vectorstore, retriever, doc_splits