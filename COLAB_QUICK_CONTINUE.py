"""
Colab 快速继续脚本 - 从超时处恢复
复制到 Colab 运行，会自动检测并继续处理
"""

print("🚀 GraphRAG 恢复脚本 v2.0")
print("="*60)

import sys
import os

# ==================== 1. 设置环境 ====================
print("\n1️⃣ 设置环境...")

# 设置项目路径
project_path = '/content/drive/MyDrive/adaptive_RAG'
if project_path not in sys.path:
    sys.path.insert(0, project_path)
print(f"   ✅ 项目路径: {project_path}")

# ==================== 2. 重启 Ollama ====================
print("\n2️⃣ 重启 Ollama...")

import subprocess
import time

subprocess.run(['pkill', '-9', 'ollama'], stderr=subprocess.DEVNULL)
time.sleep(2)

ollama_process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
time.sleep(5)

import requests
try:
    r = requests.get('http://localhost:11434/api/tags', timeout=5)
    print(f"   ✅ Ollama 运行正常" if r.status_code == 200 else f"   ⚠️ 状态码: {r.status_code}")
except:
    print("   ❌ Ollama 未响应")

# ==================== 3. 加载文档 ====================
print("\n3️⃣ 加载文档...")

from config import setup_environment
from document_processor import DocumentProcessor

setup_environment()

# 创建文档处理器
doc_processor = DocumentProcessor()

# 加载文档（使用默认 URLs）
vectorstore, retriever, doc_splits = doc_processor.setup_knowledge_base(
    enable_graphrag=True
)

print(f"   ✅ 已加载 {len(doc_splits)} 个文档")

# ==================== 4. 修改超时配置 ====================
print("\n4️⃣ 增加超时时间...")

entity_file = os.path.join(project_path, 'entity_extractor.py')
with open(entity_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 修改默认参数
if 'timeout: int = 60' in content:
    content = content.replace(
        'timeout: int = 60, max_retries: int = 3',
        'timeout: int = 180, max_retries: int = 5'
    )
    with open(entity_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("   ✅ 超时已改为 180 秒，重试改为 5 次")
else:
    print("   ℹ️ 已经是修改后的配置")

# 重新加载模块
import importlib
for mod in ['entity_extractor', 'graph_indexer']:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

# ==================== 5. 继续处理 ====================
print("\n5️⃣ 继续处理文档...")
print("="*60)

from graph_indexer import GraphRAGIndexer

# 配置起始位置
START_INDEX = 55  # 👈 从文档 #56 开始，修改这里可以跳过某些文档
BATCH_SIZE = 3    # 👈 批次大小，可以改为 1-5

print(f"\n   起始位置: 文档 #{START_INDEX + 1}")
print(f"   批次大小: {BATCH_SIZE}")
print(f"   待处理: {len(doc_splits) - START_INDEX} 个文档\n")

remaining_docs = doc_splits[START_INDEX:]

indexer = GraphRAGIndexer()

try:
    graph = indexer.index_documents(
        documents=remaining_docs,
        batch_size=BATCH_SIZE,
        save_path=f"{project_path}/knowledge_graph_recovered.pkl"
    )
    
    print("\n✅ 处理完成！")
    stats = graph.get_statistics()
    print(f"📊 节点: {stats['num_nodes']}, 边: {stats['num_edges']}, 社区: {stats['num_communities']}")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    print("\n建议:")
    print("   • 如果文档 #56 超时，修改 START_INDEX = 56 跳过它")
    print("   • 如果 Ollama 崩溃，重新运行此脚本")
    print("   • 减小 BATCH_SIZE 到 1 或 2")
