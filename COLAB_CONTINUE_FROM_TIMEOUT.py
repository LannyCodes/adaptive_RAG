"""
在 Colab 中从超时处继续处理的完整脚本
直接复制到 Colab 代码单元格运行
"""

print("🚀 GraphRAG 超时恢复脚本")
print("="*60)

# ==================== 步骤 0: 检查前置条件 ====================
print("\n📋 步骤 0: 检查前置条件...")

import sys
import os

# 挂载 Google Drive（如果还没有挂载）
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        print("   挂载 Google Drive...")
        drive.mount('/content/drive')
    else:
        print("   ✅ Google Drive 已挂载")
except:
    print("   ⚠️ 不在 Colab 环境中")

# 设置路径
project_path = '/content/drive/MyDrive/adaptive_RAG'
sys.path.insert(0, project_path)

print(f"   项目路径: {project_path}")

# ==================== 步骤 1: 重启 Ollama ====================
print("\n🔄 步骤 1: 重启 Ollama 服务...")

import subprocess
import time

# 杀掉旧进程
!pkill -9 ollama 2>/dev/null

time.sleep(2)

# 启动新进程
print("   启动 Ollama 服务...")
ollama_process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    preexec_fn=os.setpgrp
)

time.sleep(5)

# 验证服务
import requests
try:
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        print("   ✅ Ollama 服务运行正常")
    else:
        print(f"   ⚠️ Ollama 响应异常: {response.status_code}")
except Exception as e:
    print(f"   ❌ Ollama 服务未响应: {e}")
    print("   请检查 Ollama 是否正确安装")

# ==================== 步骤 2: 加载配置和文档 ====================
print("\n📚 步骤 2: 加载配置和文档...")

# 导入配置
from config import setup_environment

try:
    setup_environment()
    print("   ✅ 环境配置加载成功")
except Exception as e:
    print(f"   ⚠️ 环境配置警告: {e}")

# 检查是否已经有 doc_splits 变量
if 'doc_splits' in dir():
    print(f"   ✅ 检测到已有 doc_splits: {len(doc_splits)} 个文档")
    use_existing_docs = True
else:
    print("   ⚠️ 未检测到 doc_splits，需要重新加载文档")
    use_existing_docs = False

# 如果没有 doc_splits，重新加载
if not use_existing_docs:
    print("\n   正在加载文档...")
    from document_processor import DocumentProcessor
    
    doc_processor = DocumentProcessor()
    
    # 使用默认 URL 或自定义 URL
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]
    
    vectorstore, retriever, doc_splits = doc_processor.setup_knowledge_base(
        urls=urls,
        enable_graphrag=True
    )
    
    print(f"   ✅ 文档加载完成: {len(doc_splits)} 个文档片段")

# ==================== 步骤 3: 修复超时配置 ====================
print("\n⚙️ 步骤 3: 修复超时配置...")

# 方案：直接修改 entity_extractor.py 文件内容
entity_extractor_path = os.path.join(project_path, 'entity_extractor.py')

# 读取原文件
with open(entity_extractor_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 检查是否已经修改过
if 'timeout: int = 180' in content:
    print("   ✅ entity_extractor.py 已经包含超时修复")
else:
    print("   📝 修改 entity_extractor.py...")
    
    # 替换初始化方法的签名
    content = content.replace(
        'def __init__(self, timeout: int = 60, max_retries: int = 3):',
        'def __init__(self, timeout: int = 180, max_retries: int = 5):'
    )
    
    # 保存修改
    with open(entity_extractor_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ 已将默认超时时间改为 180 秒，重试次数改为 5 次")

# 重新加载模块
import importlib

if 'entity_extractor' in sys.modules:
    importlib.reload(sys.modules['entity_extractor'])
    print("   🔄 entity_extractor 模块已重新加载")

if 'graph_indexer' in sys.modules:
    importlib.reload(sys.modules['graph_indexer'])
    print("   🔄 graph_indexer 模块已重新加载")

# ==================== 步骤 4: 确定继续处理的起点 ====================
print("\n📊 步骤 4: 确定处理起点...")

# 让用户选择从哪里开始
print("\n请选择继续处理的方式:")
print("  1. 从文档 #56 重新开始（包含 #56）")
print("  2. 跳过文档 #56，从 #57 开始")
print("  3. 从头开始处理所有文档")
print("  4. 自定义起始位置")

# 默认选项（可以修改）
choice = 1  # 👈 修改这里来选择不同的选项

if choice == 1:
    start_index = 55  # 文档 #56 的索引
    print(f"\n   ✅ 选择: 从文档 #56 开始（索引 {start_index}）")
elif choice == 2:
    start_index = 56  # 跳过 #56
    print(f"\n   ✅ 选择: 跳过文档 #56，从 #57 开始（索引 {start_index}）")
elif choice == 3:
    start_index = 0
    print(f"\n   ✅ 选择: 从头开始处理所有文档")
else:
    # 自定义
    start_index = 55  # 👈 修改这里来自定义起始位置
    print(f"\n   ✅ 选择: 自定义起始位置（索引 {start_index}）")

remaining_docs = doc_splits[start_index:]
print(f"   待处理文档数: {len(remaining_docs)} 个")

# ==================== 步骤 5: 开始处理 ====================
print("\n🚀 步骤 5: 开始处理文档...")
print("="*60)

from graph_indexer import GraphRAGIndexer

# 创建索引器
indexer = GraphRAGIndexer()

# 开始索引
try:
    graph = indexer.index_documents(
        documents=remaining_docs,
        batch_size=3,  # 👈 可以调整批次大小（1-5 推荐）
        save_path=os.path.join(project_path, "knowledge_graph_recovered.pkl")
    )
    
    print("\n" + "="*60)
    print("✅ 处理完成！")
    print("="*60)
    
    # 显示统计信息
    stats = graph.get_statistics()
    print(f"\n📊 知识图谱统计:")
    print(f"   • 节点数: {stats['num_nodes']}")
    print(f"   • 边数: {stats['num_edges']}")
    print(f"   • 社区数: {stats['num_communities']}")
    print(f"   • 图密度: {stats['density']:.4f}")
    
except KeyboardInterrupt:
    print("\n⚠️ 处理被用户中断")
    print("   可以记录当前进度，稍后继续")
    
except Exception as e:
    print(f"\n❌ 处理过程中出现错误:")
    print(f"   {type(e).__name__}: {e}")
    print("\n建议:")
    print("   1. 检查上面的错误信息")
    print("   2. 如果是某个文档超时，尝试跳过它")
    print("   3. 如果是 Ollama 问题，重启服务")
    
    import traceback
    print("\n完整错误堆栈:")
    traceback.print_exc()

# ==================== 完成 ====================
print("\n" + "="*60)
print("脚本执行完成")
print("="*60)
print("\n💡 提示:")
print("   • 如果遇到超时，检查上面的错误信息")
print("   • 可以修改 choice 变量来跳过问题文档")
print("   • 可以修改 batch_size 来调整处理速度")
print("   • 图谱已保存到: knowledge_graph_recovered.pkl")
