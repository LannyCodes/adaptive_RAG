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
    # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 文档分块配置
CHUNK_SIZE = 250
CHUNK_OVERLAP = 0

# 向量数据库配置
COLLECTION_NAME = "rag-chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace嵌入模型

# 搜索配置
WEB_SEARCH_RESULTS_COUNT = 3

# GraphRAG配置
ENABLE_GRAPHRAG = True  # 是否启用GraphRAG功能
GRAPHRAG_INDEX_PATH = "./data/knowledge_graph.json"  # 图谱索引保存路径
GRAPHRAG_COMMUNITY_ALGORITHM = "louvain"  # 社区检测算法: louvain, greedy, label_propagation
GRAPHRAG_MAX_HOPS = 2  # 本地查询最大跳数
GRAPHRAG_TOP_K_COMMUNITIES = 5  # 全局查询使用的社区数量
GRAPHRAG_BATCH_SIZE = 10  # 实体提取批处理大小


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