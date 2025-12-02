"""
文档处理和向量化模块
负责文档加载、文本分块、向量化和向量数据库初始化
"""

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader

# 尝试导入 langchain_milvus，如果失败则回退到 langchain_community 并应用补丁
try:
    from langchain_milvus import MilvusVectorStore as Milvus
    print("✅ 使用 langchain-milvus (新版)")
except ImportError:
    try:
        from langchain_community.vectorstores import Milvus
        print("⚠️ 使用 langchain_community.vectorstores.Milvus (旧版)")
        
        # Monkeypatch: 修复旧版 LangChain 对 Milvus Lite 本地文件路径的校验问题
        # 旧版 _create_connection_alias 强制要求 URI 以 http/https 开头
        def _patched_create_connection_alias(self, connection_args):
            uri = connection_args.get("uri")
            # 为本地文件生成唯一的 alias
            if uri:
                import hashlib
                return hashlib.md5(uri.encode()).hexdigest()
            return "default"
            
        # 应用补丁
        Milvus._create_connection_alias = _patched_create_connection_alias
        print("🔧 已应用 Milvus Lite 路径校验补丁")
    except ImportError:
        pass

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
    # 向量库配置
    VECTOR_STORE_TYPE,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_USER,
    MILVUS_PASSWORD,
    MILVUS_URI,
    MILVUS_INDEX_TYPE,
    MILVUS_INDEX_PARAMS,
    MILVUS_SEARCH_PARAMS,
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

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document


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
        
        return self._process_results(all_results)
    
    async def ainvoke(self, query):
        """异步执行检索并合并结果"""
        import asyncio
        
        # 并发获取各检索器的结果
        # 注意：假设所有 retriever 都支持 ainvoke
        tasks = [retriever.ainvoke(query) for retriever in self.retrievers]
        results_list = await asyncio.gather(*tasks)
        
        all_results = []
        for i, results in enumerate(results_list):
            for doc in results:
                # 添加检索器索引和权重信息
                doc.metadata["retriever_index"] = i
                doc.metadata["retriever_weight"] = self.weights[i]
                all_results.append(doc)
                
        return self._process_results(all_results)

    def _process_results(self, all_results):
        """排序和去重处理"""
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
    
    def initialize_vectorstore(self):
        """初始化向量数据库连接"""
        if self.vectorstore:
            return

        print("正在连接向量数据库...")
        
        # 强制使用 Milvus
        try:
            # 准备连接参数
            connection_args = {}
            is_local_file = False
            
            # 优先使用 URI
            if MILVUS_URI and len(MILVUS_URI.strip()) > 0:
                is_local_file = not (MILVUS_URI.startswith("http://") or MILVUS_URI.startswith("https://"))
                
                real_uri = MILVUS_URI
                if is_local_file:
                    import os
                    # Milvus Lite requires absolute path in some versions/environments
                    if not os.path.isabs(real_uri):
                        real_uri = os.path.abspath(real_uri)
                        print(f"📂 将相对路径转换为绝对路径: {real_uri}")
                    
                    # 确保父目录存在
                    parent_dir = os.path.dirname(real_uri)
                    if parent_dir and not os.path.exists(parent_dir):
                        print(f"📂 创建 Milvus 存储目录: {parent_dir}")
                        os.makedirs(parent_dir, exist_ok=True)
                
                mode_name = "Lite (Local File)" if is_local_file else "Cloud (HTTP)"
                print(f"🔄 正在连接 Milvus {mode_name} ({real_uri})...")
                connection_args["uri"] = real_uri
                
                if not is_local_file and MILVUS_PASSWORD:
                        connection_args["token"] = MILVUS_PASSWORD
            else:
                print(f"🔄 正在连接 Milvus Server ({MILVUS_HOST}:{MILVUS_PORT})...")
                connection_args = {
                    "host": MILVUS_HOST,
                    "port": MILVUS_PORT,
                    "user": MILVUS_USER,
                    "password": MILVUS_PASSWORD
                }

            # 显式建立全局连接 (修复 ConnectionNotExistException)
            try:
                from pymilvus import connections, utility
                print(f"🔌 尝试建立 pymilvus 全局连接 (Alias: default)...")
                # 移除旧连接（如果存在）以防参数变更
                if connections.has_connection("default"):
                    connections.disconnect("default")
                
                connections.connect(alias="default", **connection_args)
                print("✅ pymilvus 全局连接建立成功")
                
                # 检查集合是否存在 (提前检查，避免 LangChain 内部出错)
                if utility.has_collection(COLLECTION_NAME, using="default"):
                    print(f"✅ 集合 {COLLECTION_NAME} 已存在")
                else:
                    print(f"ℹ️ 集合 {COLLECTION_NAME} 不存在，将由 Milvus 类自动创建")
                    
            except ImportError:
                print("⚠️ 未找到 pymilvus 库，跳过显式连接")
            except Exception as e:
                print(f"⚠️ 显式连接尝试失败: {e}")
                # 继续尝试，也许 LangChain 内部能处理

            # 确定索引类型
            # Milvus Lite (本地模式) 仅支持 FLAT, IVF_FLAT, AUTOINDEX，不支持 HNSW
            final_index_type = MILVUS_INDEX_TYPE
            final_index_params = MILVUS_INDEX_PARAMS
            
            if is_local_file and MILVUS_INDEX_TYPE == "HNSW":
                print("⚠️ 检测到 Milvus Lite (本地模式)，HNSW 索引不受支持，自动切换为 AUTOINDEX")
                final_index_type = "AUTOINDEX"
                final_index_params = {} # AUTOINDEX 不需要复杂参数

            # 初始化 Milvus 连接 (不删除旧数据)
            # 注意：由于我们已经手动建立了全局连接 'default'，
            # 这里我们将 connection_args 简化为仅指向该 alias，
            # 避免 LangChain 再次尝试连接或因参数问题覆盖连接。
            self.vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"alias": "default"}, # ✅ 复用已建立的连接
                index_params={
                    "metric_type": "L2",
                    "index_type": final_index_type,
                    "params": final_index_params
                },
                search_params={
                    "metric_type": "L2", 
                    "params": MILVUS_SEARCH_PARAMS
                },
                drop_old=False,  # ✅ 持久化关键：不删除旧索引
                auto_id=True
            )
            print("✅ Milvus 向量数据库连接成功")
        except ImportError:
            print("❌ 未安装 pymilvus，请运行: pip install pymilvus")
            raise
        except Exception as e:
            print(f"❌ Milvus 连接失败: {e}")
            raise

        # 配置检索器
        retriever_kwargs = {}
        # if ENABLE_MULTIMODAL:
            # retriever_kwargs["expr"] = "data_type == 'text'"
        self.retriever = self.vectorstore.as_retriever(search_kwargs=retriever_kwargs)

    def check_existing_urls(self, urls: List[str]) -> set:
        """检查哪些URL已经存在于向量库中"""
        if not self.vectorstore:
            return set()
            
        existing = set()
        print("正在检查已存在的文档...")
        try:
            # 尝试通过检索来检查
            # 注意：这里假设 source 字段可以作为过滤条件
            for url in urls:
                # 使用 similarity_search 但带有严格过滤，且只取1条
                # 这里的 query 没关系，主要看 filter
                try:
                    # 注意：Milvus 的 expr 语法
                    expr = f'source == "{url}"'
                    res = self.vectorstore.similarity_search(
                        "test", 
                        k=1, 
                        expr=expr
                    )
                    if res:
                        existing.add(url)
                except Exception as e:
                    # 如果失败，可能是 schema 问题，尝试 metadata 字段
                    try:
                        expr = f'metadata["source"] == "{url}"'
                        res = self.vectorstore.similarity_search(
                            "test", 
                            k=1, 
                            expr=expr
                        )
                        if res:
                            existing.add(url)
                    except:
                        pass
                        
            print(f"✅ 发现 {len(existing)} 个已存在的 URL")
        except Exception as e:
            print(f"⚠️ 检查现有URL失败: {e}")
            
        return existing

    def add_documents_to_vectorstore(self, doc_splits):
        """添加文档到向量库"""
        if not doc_splits:
            return

        print(f"正在添加 {len(doc_splits)} 个文档块到向量数据库...")
        if not self.vectorstore:
            self.initialize_vectorstore()
            
        # 添加元数据
        for doc in doc_splits:
            if 'source_type' not in doc.metadata:
                source = doc.metadata.get('source', '')
                if any(fmt in source.lower() for fmt in SUPPORTED_IMAGE_FORMATS):
                    doc.metadata['data_type'] = 'image'
                else:
                    doc.metadata['data_type'] = 'text'

        self.vectorstore.add_documents(doc_splits)
        print("✅ 文档添加完成")
        
    def create_vectorstore(self, doc_splits, persist_directory=None):
        """(已弃用) 兼容旧接口，但使用新逻辑"""
        print("⚠️ create_vectorstore 已弃用，请使用 initialize_vectorstore 和 add_documents_to_vectorstore")
        self.initialize_vectorstore()
        if doc_splits:
            self.add_documents_to_vectorstore(doc_splits)
        return self.vectorstore, self.retriever

    def get_all_documents_from_vectorstore(self, limit: Optional[int] = None) -> List[Document]:
        """从已持久化的向量数据库读取所有文档内容并构造 Document 列表"""
        if not self.vectorstore:
            return []
        try:
            data = self.vectorstore._collection.get(include=["documents", "metadatas"])  # type: ignore
            docs_raw = data.get("documents") or []
            metas = data.get("metadatas") or []
            docs: List[Document] = []
            for i, content in enumerate(docs_raw):
                if content:
                    meta = metas[i] if i < len(metas) else {}
                    docs.append(Document(page_content=content, metadata=meta))
            if limit:
                return docs[:limit]
            return docs
        except Exception as e:
            print(f"⚠️ 读取向量库文档失败: {e}")
            return []
    
    def setup_knowledge_base(self, urls=None, enable_graphrag=False):
        """设置完整的知识库（加载、分割、向量化）
        
        Args:
            urls: 文档URL列表
            enable_graphrag: 是否启用GraphRAG索引
            
        Returns:
            vectorstore, retriever, doc_splits
        """
        if urls is None:
            urls = KNOWLEDGE_BASE_URLS
            
        # 1. 初始化向量库连接
        self.initialize_vectorstore()
        
        # 2. 检查已存在的 URL (去重)
        existing_urls = self.check_existing_urls(urls)
        new_urls = [url for url in urls if url not in existing_urls]
        
        doc_splits = []
        if new_urls:
            print(f"🔄 发现 {len(new_urls)} 个新 URL，开始处理...")
            docs = self.load_documents(new_urls)
            doc_splits = self.split_documents(docs)
            self.add_documents_to_vectorstore(doc_splits)
        else:
            print("✅ 所有 URL 已存在，跳过文档加载和向量化")
            
        # 3. 初始化混合检索 (BM25)
        if ENABLE_HYBRID_SEARCH:
            print("正在初始化混合检索 (BM25)...")
            try:
                bm25_docs = []
                # 如果有旧数据且这次没有加载全部数据，必须从 DB 加载所有文档以重建 BM25
                # 注意：如果只有新文档，BM25 只会包含新文档，这是不对的。
                # 只要有 existing_urls，说明库里有旧数据。
                if len(existing_urls) > 0:
                    print("🔄 正在从向量库加载所有文档以重建 BM25 索引...")
                    # 注意：这里假设内存够大
                    all_docs = self.get_all_documents_from_vectorstore()
                    bm25_docs = all_docs
                else:
                    # 全新构建
                    bm25_docs = doc_splits
                
                if bm25_docs:
                    self.bm25_retriever = BM25Retriever.from_documents(
                        bm25_docs, 
                        k=KEYWORD_SEARCH_K,
                        k1=BM25_K1,
                        b=BM25_B
                    )
                    
                    self.ensemble_retriever = CustomEnsembleRetriever(
                        retrievers=[self.retriever, self.bm25_retriever],
                        weights=[HYBRID_SEARCH_WEIGHTS["vector"], HYBRID_SEARCH_WEIGHTS["keyword"]]
                    )
                    print("✅ 混合检索初始化成功")
                else:
                    print("⚠️ 没有文档用于初始化 BM25")
            except Exception as e:
                print(f"⚠️ 混合检索初始化失败: {e}")
                self.ensemble_retriever = None
        
        # 返回 doc_splits用于GraphRAG索引 (注意：这里只返回了新增的)
        return self.vectorstore, self.retriever, doc_splits
    
    async def async_expand_query(self, query: str) -> List[str]:
        """异步扩展查询"""
        if not self.query_expansion_model:
            return [query]
            
        try:
            # 使用LLM生成扩展查询
            prompt = QUERY_EXPANSION_PROMPT.format(query=query)
            expanded_queries_text = await self.query_expansion_model.ainvoke(prompt)
            
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
            return expanded_queries[:MAX_EXPANDED_QUERIES + 1]
        except Exception as e:
            print(f"⚠️ 异步查询扩展失败: {e}")
            return [query]

    async def async_hybrid_retrieve(self, query: str, top_k: int = 5, filter_type: str = "text") -> List:
        """异步混合检索
        
        Args:
            filter_type: 数据类型过滤，"text" (默认), "image", 或 "all" (不过滤)
        """
        # 构建搜索参数
        search_kwargs = {}
        if filter_type != "all" and ENABLE_MULTIMODAL:
            search_kwargs["expr"] = f"data_type == '{filter_type}'"

        if not ENABLE_HYBRID_SEARCH or not self.ensemble_retriever:
            # 纯向量检索，直接支持 search_kwargs
            if self.vectorstore:
                return await self.vectorstore.asimilarity_search(query, k=top_k, **search_kwargs)
            return await self.retriever.ainvoke(query)
            
        try:
            # 混合检索
            # 注意：目前 CustomEnsembleRetriever 的 invoke/ainvoke 尚未透传 search_kwargs
            # 为了让混合检索也享受到过滤优化，我们需要修改 CustomEnsembleRetriever 或者在这里处理
            # 鉴于 CustomEnsembleRetriever 比较简单，我们假设它主要用于文本
            # 如果需要严格过滤，最好在 vectorstore 层面处理
            
            # 临时方案：如果是混合检索且需要过滤，我们可能需要传递给 retriever
            # 但标准 retriever 接口不支持动态传参。
            # 策略：如果 filter_type 是 text (默认)，且我们在 init 时已经设置了默认不严格过滤，
            # 这里其实无法动态改变 retriever 的行为，除非我们重新生成一个 retriever 或者修改 retriever.search_kwargs
            
            # 动态修改 retriever 的 search_kwargs (这是 LangChain retriever 的特性)
            if filter_type != "all" and ENABLE_MULTIMODAL:
                self.retriever.search_kwargs["expr"] = f"data_type == '{filter_type}'"
            else:
                self.retriever.search_kwargs.pop("expr", None)
                
            results = await self.ensemble_retriever.ainvoke(query)
            return results[:top_k]
        except Exception as e:
            print(f"⚠️ 异步混合检索失败: {e}")
            print("回退到向量检索")
            if self.vectorstore:
                return await self.vectorstore.asimilarity_search(query, k=top_k, **search_kwargs)
            return await self.retriever.ainvoke(query)

    async def async_enhanced_retrieve(self, query: str, top_k: int = 5, rerank_candidates: int = 20, 
                         image_paths: List[str] = None, use_query_expansion: bool = None):
        """异步增强检索"""
        import asyncio
        
        # 确定是否使用查询扩展
        if use_query_expansion is None:
            use_query_expansion = ENABLE_QUERY_EXPANSION
            
        # 如果启用查询扩展，生成扩展查询
        if use_query_expansion:
            expanded_queries = await self.async_expand_query(query)
            print(f"查询扩展: {len(expanded_queries)} 个查询")
        else:
            expanded_queries = [query]
            
        # 多模态检索（暂时保持同步，使用线程池）
        if image_paths and ENABLE_MULTIMODAL:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.multimodal_retrieve, query, image_paths, top_k)
            
        # 混合检索或向量检索
        all_candidate_docs = []
        
        # 决定过滤策略
        # 默认情况下，如果只是文本查询，为了性能优化，我们只检索文本数据
        # 如果提供了图像，或者用户显式要求，可以放开限制
        filter_type = "text" # 默认只搜文本，实现百万级数据的性能优化
        if image_paths:
            filter_type = "all" # 跨模态时搜所有
            
        # 构建过滤表达式 (仅用于直接调用 vectorstore 的情况，async_hybrid_retrieve 内部已处理)
        search_kwargs = {}
        if filter_type != "all" and ENABLE_MULTIMODAL:
             search_kwargs["expr"] = f"data_type == '{filter_type}'"

        async def retrieve_single(q):
            if ENABLE_HYBRID_SEARCH:
                # 使用支持动态过滤的 hybrid retrieve
                 docs = await self.async_hybrid_retrieve(q, rerank_candidates, filter_type=filter_type)
            else:
                # 使用带有过滤条件的检索
                if self.vectorstore:
                    docs = await self.vectorstore.asimilarity_search(
                        q, 
                        k=rerank_candidates,
                        **search_kwargs # 传入 expr
                    )
                else:
                    # Fallback
                    docs = await self.retriever.ainvoke(q)
                
                if len(docs) > rerank_candidates:
                    docs = docs[:rerank_candidates]
            return docs

        # 并发执行所有查询的检索
        results = await asyncio.gather(*[retrieve_single(q) for q in expanded_queries])
        
        for docs in results:
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
        # 注意：重排通常是计算密集型，建议放入线程池
        if self.reranker and len(unique_docs) > top_k:
            try:
                loop = asyncio.get_running_loop()
                # rerank 方法内部可能也比较耗时
                reranked_results = await loop.run_in_executor(
                    None, 
                    self.reranker.rerank, 
                    query, unique_docs, top_k
                )
                final_docs = [doc for doc, score in reranked_results]
                scores = [score for doc, score in reranked_results]
                
                print(f"重排后返回 {len(final_docs)} 个文档")
                print(f"重排分数范围: {min(scores):.4f} - {max(scores):.4f}")
                
                return final_docs
            except Exception as e:
                print(f"⚠️ 重排失败: {e}，使用原始检索结果")
                return unique_docs[:top_k]
        else:
            return unique_docs[:top_k]
    
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
        
        # 1. 文本查询 (Text-to-Text & Text-to-Image)
        # 如果提供了文本查询，我们希望它能检索到文本和相关图像
        # 此时不应该限制 data_type，或者应该显式包含两者
        
        # 如果没有提供图像，这可能是一个纯文本查询，但也可能想搜图
        # 这里我们让 self.retriever (或 hybrid) 负责所有模态的检索
        # (前提是它们都在同一个向量空间，CLIP 可以做到这一点)
        text_docs = []
        if query:
             text_docs = self.hybrid_retrieve(query, top_k) if ENABLE_HYBRID_SEARCH else self.retriever.invoke(query)[:top_k]
        
        # 如果没有提供图像输入，直接返回文本查询的结果
        if not image_paths:
            return text_docs
            
        try:
            # 2. 图像查询 (Image-to-Text & Image-to-Image)
            image_results = []
            for image_path in image_paths:
                # 检查文件格式
                file_ext = image_path.split('.')[-1].lower()
                if file_ext not in SUPPORTED_IMAGE_FORMATS:
                    print(f"⚠️ 不支持的图像格式: {file_ext}")
                    continue
                    
                # 编码图像
                image_embedding = self.encode_image(image_path)
                
                # 使用图像嵌入进行检索
                if self.vectorstore:
                    # 图像可以检索文本描述，也可以检索相似图像
                    # 这里我们不做限制，检索所有类型
                    img_docs = self.vectorstore.similarity_search_by_vector(
                        embedding=image_embedding,
                        k=top_k
                    )
                    image_results.extend(img_docs)
                
            # 合并文本查询结果和图像查询结果
            # 简单合并并去重
            all_docs = text_docs + image_results
            
            # 去重
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                content = doc.page_content
                if content not in seen_content:
                    seen_content.add(content)
                    unique_docs.append(doc)
            
            final_docs = unique_docs[:top_k]
            
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
    print("🚀 初始化文档处理器 (Milvus 版)...")
    processor = DocumentProcessor()
    
    # 直接设置知识库
    # Milvus 的连接和索引逻辑在 DocumentProcessor.create_vectorstore 中处理
    vectorstore, retriever, doc_splits = processor.setup_knowledge_base()
    
    return processor, vectorstore, retriever, doc_splits
