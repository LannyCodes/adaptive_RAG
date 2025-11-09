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
        print("✅ TAVILY_API_KEY 已从环境变量中加载")
    else:
        print("⚠️  TAVILY_API_KEY 未找到")


# 模型配置
# Kaggle环境推荐使用较小的模型以加快下载速度
# 可选模型:
#   - "mistral" (4GB) - 质量最好，但下载慢
#   - "phi" (1.6GB) - 平衡选择，速度较快
#   - "tinyllama" (600MB) - 最快，质量稍低
#   - "qwen:0.5b" (350MB) - 极小模型，速度极快
LOCAL_LLM = "mistral"  # 在Kaggle中可改为 "phi" 或 "tinyllama"

# 知识库URL配置
KNOWLEDGE_BASE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 文档分块配置
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50  # 添加重叠以保持上下文连贯性，提升检索准确率

# 向量数据库配置
COLLECTION_NAME = "rag-chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace嵌入模型

# 搜索配置
WEB_SEARCH_RESULTS_COUNT = 3

# GraphRAG配置
ENABLE_GRAPHRAG = True  # 是否启用GraphRAG功能
GRAPHRAG_INDEX_PATH = "./data/knowledge_graph.json"  # 图谱索引保存路径

# 确保数据目录存在
import os
os.makedirs("./data", exist_ok=True)
GRAPHRAG_COMMUNITY_ALGORITHM = "louvain"  # 社区检测算法: louvain, greedy, label_propagation
GRAPHRAG_MAX_HOPS = 2  # 本地查询最大跳数
GRAPHRAG_TOP_K_COMMUNITIES = 5  # 全局查询使用的社区数量
GRAPHRAG_BATCH_SIZE = 10  # 实体提取批处理大小

# 混合检索策略配置
ENABLE_HYBRID_SEARCH = True  # 是否启用混合检索策略
HYBRID_SEARCH_WEIGHTS = {"vector": 0.7, "keyword": 0.3}  # 向量检索和关键词检索的权重
KEYWORD_SEARCH_K = 5  # 关键词检索返回的文档数量
BM25_K1 = 1.2  # BM25算法的k1参数
BM25_B = 0.75  # BM25算法的b参数

# 查询扩展优化配置
ENABLE_QUERY_EXPANSION = True  # 是否启用查询扩展
QUERY_EXPANSION_MODEL = "mistral"  # 用于查询扩展的模型
QUERY_EXPANSION_PROMPT = """请为以下查询生成3-5个相关的扩展查询，这些查询应该从不同角度探索原始查询的主题。
原始查询: {query}
扩展查询: """  # 查询扩展提示模板
MAX_EXPANDED_QUERIES = 3  # 最多使用的扩展查询数量

# 多模态支持配置
ENABLE_MULTIMODAL = True  # 是否启用多模态支持
MULTIMODAL_IMAGE_MODEL = "openai/clip-vit-base-patch32"  # 图像嵌入模型
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "gif", "bmp"]  # 支持的图像格式
IMAGE_EMBEDDING_DIM = 512  # 图像嵌入维度
MULTIMODAL_WEIGHTS = {"text": 0.7, "image": 0.3}  # 文本和图像检索的权重


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