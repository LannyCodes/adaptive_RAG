"""
配置和环境设置模块
包含API密钥管理、模型配置和URL配置
"""

import getpass
import os

# 尝试加载.env文件，如果没有安装dotenv则跳过
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env文件已加载")
except ImportError:
    print("⚠️  python-dotenv未安装，将使用系统环境变量")
    print("提示：运行 'pip install python-dotenv' 来安装")


def _set_env(var: str):
    """设置环境变量，优先从.env文件读取，如果不存在则提示用户输入"""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


def setup_environment():
    """设置所有必需的环境变量"""
    _set_env("TAVILY_API_KEY")
    # 不再需要NOMIC_API_KEY，使用HuggingFace本地嵌入
    
    # 验证API密钥是否已设置
    tavily_key = os.environ.get("TAVILY_API_KEY")
    
    if tavily_key:
        if tavily_key.startswith("tvly-"):
            print(f"✅ TAVILY_API_KEY 已加载: {tavily_key[:10]}...")
        else:
            print(f"✅ TAVILY_API_KEY 已从环境变量中加载")
    else:
        print("⚠️  TAVILY_API_KEY 未找到")


LLM_BACKEND = os.environ.get("LLM_BACKEND", "tongyi")  # 原来是使用Ollama
LOCAL_LLM = os.environ.get("LOCAL_LLM", "qwen2:1.5b")

TONGYI_API_KEY = os.environ.get("TONGYI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
TONGYI_BASE_URL = os.environ.get("TONGYI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
TONGYI_MODEL = os.environ.get("TONGYI_MODEL", "qwen-plus")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# 知识库URL配置
KNOWLEDGE_BASE_URLS = [
    "https://learnprompting.org/docs/introduction",
    "https://lilianweng.github.io/posts/2017-06-21-overview/",
    "https://lilianweng.github.io/posts/2017-07-08-stock-rnn-part-1/",
    "https://lilianweng.github.io/posts/2017-07-22-stock-rnn-part-2/",
    "https://lilianweng.github.io/posts/2017-08-01-interpretation/",
    "https://lilianweng.github.io/posts/2017-08-20-gan/",
    "https://lilianweng.github.io/posts/2017-09-28-information-bottleneck/",
    "https://lilianweng.github.io/posts/2017-10-15-word-embedding/",
    "https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/",
    "https://lilianweng.github.io/posts/2017-12-15-object-recognition-part-2/",
    "https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/",
    "https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/",
    "https://lilianweng.github.io/posts/2018-02-19-rl-overview/",
    "https://lilianweng.github.io/posts/2018-04-08-policy-gradient/",
    "https://lilianweng.github.io/posts/2018-05-05-drl-implementation/",
    "https://lilianweng.github.io/posts/2018-06-24-attention/",
    "https://lilianweng.github.io/posts/2018-08-12-vae/",
    "https://lilianweng.github.io/posts/2018-10-13-flow-models/",
    "https://lilianweng.github.io/posts/2018-11-30-meta-learning/",
    "https://lilianweng.github.io/posts/2018-12-27-object-recognition-part-4/",
    "https://lilianweng.github.io/posts/2019-01-31-lm/",
    "https://lilianweng.github.io/posts/2019-03-14-overfit/",
    "https://lilianweng.github.io/posts/2019-05-05-domain-randomization/",
    "https://lilianweng.github.io/posts/2019-06-23-meta-rl/",
    "https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/",
    "https://lilianweng.github.io/posts/2019-11-10-self-supervised/",
    "https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/",
    "https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/",
    "https://lilianweng.github.io/posts/2020-06-07-exploration-drl/",
    "https://lilianweng.github.io/posts/2020-08-06-nas/",
    "https://lilianweng.github.io/posts/2020-10-29-odqa/",
    "https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/",
    "https://lilianweng.github.io/posts/2021-03-21-lm-toxicity/",
    "https://lilianweng.github.io/posts/2021-05-31-contrastive/",
    "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
    "https://lilianweng.github.io/posts/2021-09-25-train-large/",
    "https://lilianweng.github.io/posts/2021-12-05-semi-supervised/",
    "https://lilianweng.github.io/posts/2022-02-20-active-learning/",
    "https://lilianweng.github.io/posts/2022-04-15-data-gen/",
    "https://lilianweng.github.io/posts/2022-06-09-vlm/",
    "https://lilianweng.github.io/posts/2022-09-08-ntk/",
    "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/",
    "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://lilianweng.github.io/posts/2024-04-13-diffusion-models-v2/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2025-05-01-thinking/"
]

# 文档分块配置
CHUNK_SIZE = 1024  # 增加到 1024 以保留更多上下文，配合 BGE-M3 使用
CHUNK_OVERLAP = 200  # 增加重叠，防止信息截断

# 向量数据库配置
VECTOR_STORE_TYPE = "milvus"  # 强制使用 Milvus
COLLECTION_NAME = "adaptive_rag"
EMBEDDING_MODEL = "BAAI/bge-m3"  # 升级为 BGE-M3，支持 8192 长度，完美适配长 Chunk

# Milvus 配置 (仅当 VECTOR_STORE_TYPE="milvus" 时生效)
# 1. Milvus Lite (本地文件模式): 仅需设置 MILVUS_URI，无需 User/Password。适合 Kaggle/本地开发。
# 2. Zilliz Cloud (云服务): 需要设置 MILVUS_URI (https://...) 和 MILVUS_PASSWORD (API Key/Token)。需官网注册。
# 3. Milvus Server (Docker/K8s): 需要设置 HOST/PORT，可选 User/Password。
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_USER = os.environ.get("MILVUS_USER", "")      # 仅在自建 Server 开启认证或使用 Zilliz Cloud 时需要
MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD", "") # Zilliz Cloud 的 API Key 也填在这里
# Milvus Lite 配置: 如果设置了 MILVUS_URI (如 "./milvus_demo.db")，将优先使用本地文件模式
MILVUS_URI = os.environ.get("MILVUS_URI", "./milvus_rag.db")

# Milvus 性能调优 (百万级数据推荐配置)
# 索引类型: HNSW (最快/吃内存), IVF_SQ8 (省内存/速度快/轻微精度损失), IVF_FLAT (平衡)
MILVUS_INDEX_TYPE = "HNSW" 
# 索引构建参数 (M: 邻居数, efConstruction: 构建深度)
MILVUS_INDEX_PARAMS = {"M": 8, "efConstruction": 64} 
# 搜索参数 (ef: 搜索范围，值越小越快但精度越低。默认是 10，百万级建议设为 30-50)
MILVUS_SEARCH_PARAMS = {"ef": 10}

# Elasticsearch 配置 (用于大规模关键词检索)
# 替代内存版 BM25，支持百万级数据
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
ES_USER = os.environ.get("ES_USER", "")  # 如果有密码认证
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")
ES_INDEX_NAME = os.environ.get("ES_INDEX_NAME", "rag_keyword_index")
ES_VERIFY_CERTS = os.environ.get("ES_VERIFY_CERTS", "false").lower() == "true"

# 搜索配置
WEB_SEARCH_RESULTS_COUNT = 3

# GraphRAG配置
ENABLE_GRAPHRAG = os.environ.get("ENABLE_GRAPH_RAG", "true").lower() == "true"  # 默认开启
GRAPHRAG_INDEX_PATH = "./data/knowledge_graph.json"  # 图谱索引保存路径

# 确保数据目录存在
import os
os.makedirs("./data", exist_ok=True)
GRAPHRAG_COMMUNITY_ALGORITHM = os.environ.get("GRAPH_COMMUNITY_ALGORITHM", "louvain")  # 社区检测算法: louvain, greedy, label_propagation
GRAPHRAG_MAX_HOPS = 2  # 本地查询最大跳数
GRAPHRAG_TOP_K_COMMUNITIES = 5  # 全局查询使用的社区数量
GRAPHRAG_BATCH_SIZE = 10  # 实体提取批处理大小
GRAPH_ENTITY_EXTRACTION_MODEL = os.environ.get("GRAPH_ENTITY_EXTRACTION_MODEL", "llama2")
GRAPH_RELATION_EXTRACTION_MODEL = os.environ.get("GRAPH_RELATION_EXTRACTION_MODEL", "llama2")
GRAPH_COMMUNITY_DETECTION = os.environ.get("GRAPH_COMMUNITY_DETECTION", "true").lower() == "true"
GRAPH_VISUALIZATION = os.environ.get("GRAPH_VISUALIZATION", "true").lower() == "true"
GRAPH_LAYOUT = os.environ.get("GRAPH_LAYOUT", "spring")

# 混合检索策略配置
ENABLE_HYBRID_SEARCH = os.environ.get("ENABLE_HYBRID_SEARCH", "true").lower() == "true"  # 默认开启
HYBRID_SEARCH_WEIGHTS = {"vector": 0.5, "keyword": 0.5}  # 向量检索和关键词检索的权重
KEYWORD_SEARCH_K = 5  # 关键词检索返回的文档数量
BM25_K1 = float(os.environ.get("BM25_K1", "1.5"))  # BM25算法的k1参数
BM25_B = float(os.environ.get("BM25_B", "0.75"))  # BM25算法的b参数

# 查询扩展优化配置
ENABLE_QUERY_EXPANSION = os.environ.get("ENABLE_QUERY_EXPANSION", "true").lower() == "true"  # 默认开启
QUERY_EXPANSION_MODEL = LOCAL_LLM  # 复用 LOCAL_LLM (Qwen2.5-7B)，避免额外下载 Mistral
QUERY_EXPANSION_PROMPT = """请为以下查询生成3-5个相关的扩展查询，这些查询应该从不同角度探索原始查询的主题。
原始查询: {query}
扩展查询: """  # 查询扩展提示模板
MAX_EXPANDED_QUERIES = int(os.environ.get("QUERY_EXPANSION_TOP_K", "2"))  # 最多使用的扩展查询数量

# 多模态支持配置
ENABLE_MULTIMODAL = os.environ.get("ENABLE_MULTIMODAL", "true").lower() == "true"  # OCR+VLM 双通道，默认开启
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp"]  # 支持的图像格式

# OCR 配置
OCR_ENGINE = os.environ.get("OCR_ENGINE", "auto")  # auto / paddleocr / tesseract / easyocr
OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "ch")  # ch=中英文, en=英文, jpn=日文等

# VLM (视觉语言模型) 配置 — 用于描述图片内容
VLM_BACKEND = os.environ.get("VLM_BACKEND", "tongyi")  # tongyi / openai / local
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen-vl-plus")  # 通义千问视觉模型
VLM_API_KEY = os.environ.get("VLM_API_KEY", "") or TONGYI_API_KEY  # 默认复用通义 API Key
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "") or TONGYI_BASE_URL  # 默认复用通义 Base URL

# 图片存储路径
IMAGE_STORAGE_DIR = os.environ.get("IMAGE_STORAGE_DIR", "./data/images")

# 旧配置（保留兼容，不再使用 CLIP）
MULTIMODAL_IMAGE_MODEL = os.environ.get("MULTIMODAL_IMAGE_MODEL", "openai/clip-vit-base-patch32")
IMAGE_EMBEDDING_DIM = 512
MULTIMODAL_WEIGHTS = {"text": 0.7, "image": 0.3}

# 高级重排器配置
ENABLE_ADVANCED_RERANKER = os.environ.get("ENABLE_ADVANCED_RERANKER", "false").lower() == "true"  # 默认关闭
ADVANCED_RERANKER_TYPE = os.environ.get("ADVANCED_RERANKER_TYPE", "context_aware")  # context_aware 或 multi_task

# 上下文感知重排器参数
CONTEXT_AWARE_WEIGHT = float(os.environ.get("CONTEXT_AWARE_WEIGHT", "0.3"))  # 上下文权重
CONTEXT_AWARE_MODEL = os.environ.get("CONTEXT_AWARE_MODEL", "BAAI/bge-reranker-base")
CONTEXT_AWARE_MAX_LENGTH = int(os.environ.get("CONTEXT_AWARE_MAX_LENGTH", "1024"))

# 多任务重排器参数
MULTI_TASK_WEIGHTS = {
    'relevance': float(os.environ.get("MT_RELEVANCE_WEIGHT", "0.35")),
    'diversity': float(os.environ.get("MT_DIVERSITY_WEIGHT", "0.25")),
    'novelty': float(os.environ.get("MT_NOVELTY_WEIGHT", "0.15")),
    'authority': float(os.environ.get("MT_AUTHORITY_WEIGHT", "0.15")),
    'recency': float(os.environ.get("MT_RECENCY_WEIGHT", "0.10"))
}
MULTI_TASK_DIVERSITY_LAMBDA = float(os.environ.get("MT_DIVERSITY_LAMBDA", "0.5"))


def get_api_keys():
    """获取API密钥并返回字典"""
    return {
        "tavily": os.environ.get("TAVILY_API_KEY")
    }


def validate_api_keys():
    """验证API密钥是否已设置"""
    keys = get_api_keys()
    missing_keys = []
    
    if not keys["tavily"]:
        missing_keys.append("TAVILY_API_KEY")
    
    if missing_keys:
        raise ValueError(f"缺少必需的API密钥: {', '.join(missing_keys)}\n请在.env文件中设置这些密钥")
    
    return True
