"""
Kaggle 快速启动脚本 - 避免重复下载大模型
使用优化的小模型配置，大幅减少启动时间

使用方法:
在 Kaggle Notebook 第一个单元格运行:
    exec(open('/kaggle/working/adaptive_RAG/KAGGLE_QUICK_START.py').read())
"""

import os
import subprocess
import sys
import time

print("🚀 Kaggle 快速启动（优化版）")
print("="*70)

# ==================== 配置区域 ====================
REPO_URL = "https://github.com/LannyCodes/adaptive_RAG.git"
PROJECT_DIR = "/kaggle/working/adaptive_RAG"

# 模型选择（根据需求修改）
# "phi"       - 1.6GB, 2-3分钟下载，质量好 ⭐⭐⭐⭐ （推荐）
# "tinyllama" - 600MB, 1分钟下载，质量中等 ⭐⭐⭐
# "qwen:0.5b" - 350MB, 30秒下载，质量较低 ⭐⭐
# "mistral"   - 4GB, 5-10分钟下载，质量最好 ⭐⭐⭐⭐⭐ （慢）

PREFERRED_MODEL = "phi"  # 👈 修改这里选择模型

print(f"\n📌 配置:")
print(f"   • 仓库: {REPO_URL}")
print(f"   • 模型: {PREFERRED_MODEL}")
print()

# ==================== 步骤 1: 克隆项目 ====================
print("📦 步骤 1/6: 克隆项目...")

os.chdir('/kaggle/working')

if os.path.exists(PROJECT_DIR):
    print("   ✅ 项目已存在")
else:
    result = subprocess.run(['git', 'clone', REPO_URL], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✅ 项目克隆成功")
    else:
        print(f"   ❌ 克隆失败: {result.stderr}")
        sys.exit(1)

os.chdir(PROJECT_DIR)

# ==================== 步骤 2: 修改配置使用小模型 ====================
print("\n⚙️ 步骤 2/6: 优化模型配置...")

config_file = 'config.py'

with open(config_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换模型配置
if 'LOCAL_LLM = "mistral"' in content:
    content = content.replace(
        'LOCAL_LLM = "mistral"',
        f'LOCAL_LLM = "{PREFERRED_MODEL}"  # Kaggle优化: 使用更小的模型'
    )
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"   ✅ 已切换到 {PREFERRED_MODEL} 模型")
else:
    print(f"   ℹ️ 配置已是优化模式")

# ==================== 步骤 3: 检查并安装 Ollama ====================
print("\n🔧 步骤 3/6: 检查 Ollama...")

ollama_check = subprocess.run(['which', 'ollama'], capture_output=True)

if ollama_check.returncode == 0:
    print("   ✅ Ollama 已安装")
else:
    print("   📥 安装 Ollama...")
    subprocess.run('curl -fsSL https://ollama.com/install.sh | sh', shell=True)
    time.sleep(3)
    print("   ✅ Ollama 安装完成")

# 验证安装
version_result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
if version_result.returncode == 0:
    print(f"   📌 {version_result.stdout.strip()}")

# ==================== 步骤 4: 启动 Ollama 服务 ====================
print("\n🚀 步骤 4/6: 启动 Ollama 服务...")

# 检查是否已运行
ps_check = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True)

if ps_check.returncode == 0:
    print("   ✅ Ollama 服务已运行")
else:
    print("   🔄 启动服务...")
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(15)
    
    # 验证
    import requests
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            print("   ✅ 服务运行正常")
    except:
        print("   ⚠️ 服务验证失败，但可能仍在启动中...")

# ==================== 步骤 5: 下载优化的模型 ====================
print(f"\n📦 步骤 5/6: 下载 {PREFERRED_MODEL} 模型...")

# 检查模型是否已存在
list_result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)

if PREFERRED_MODEL in list_result.stdout:
    print(f"   ✅ {PREFERRED_MODEL} 模型已存在")
else:
    # 显示预计时间
    time_estimates = {
        "qwen:0.5b": "约30秒",
        "tinyllama": "约1分钟",
        "phi": "约2-3分钟",
        "mistral": "约5-10分钟"
    }
    
    estimated_time = time_estimates.get(PREFERRED_MODEL, "未知")
    
    print(f"   📥 开始下载（预计时间: {estimated_time}）...")
    print(f"   ⏳ 请稍候...")
    
    start_time = time.time()
    
    pull_result = subprocess.run(
        ['ollama', 'pull', PREFERRED_MODEL],
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start_time
    
    if pull_result.returncode == 0:
        print(f"   ✅ 模型下载完成（耗时: {int(elapsed)}秒）")
    else:
        print(f"   ⚠️ 下载警告: {pull_result.stderr[:200]}")

# ==================== 步骤 6: 安装 Python 依赖 ====================
print("\n📦 步骤 6/6: 安装 Python 依赖...")

subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_graphrag.txt', '-q'])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 
                'langchain', 'langchain-core', 'langchain-community', 
                'langchain-text-splitters', '-q'])

print("   ✅ 依赖安装完成")

# ==================== 设置 Python 路径 ====================
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ==================== 完成 ====================
print("\n" + "="*70)
print("✅ 环境准备完成！")
print("="*70)

print(f"\n📊 配置摘要:")
print(f"   • 工作目录: {os.getcwd()}")
print(f"   • 使用模型: {PREFERRED_MODEL}")
print(f"   • Python路径: 已添加")

# 显示模型对比
print(f"\n📌 模型选择说明:")
print("   • phi (当前) - 平衡速度和质量，推荐日常使用")
print("   • tinyllama - 最快下载，适合快速测试")
print("   • mistral - 质量最高，但下载慢（不推荐Kaggle）")

print(f"\n💡 下一步:")
print("   1. 开始 GraphRAG 索引:")
print("      from document_processor import DocumentProcessor")
print("      from graph_indexer import GraphRAGIndexer")
print("      ")
print("      doc_processor = DocumentProcessor()")
print("      vectorstore, retriever, doc_splits = doc_processor.setup_knowledge_base(enable_graphrag=True)")
print("      ")
print("      indexer = GraphRAGIndexer()")
print("      graph = indexer.index_documents(doc_splits, batch_size=3)")
print()
print("   2. 如需切换模型，修改脚本顶部的 PREFERRED_MODEL 变量")

print("\n⚠️ 提示:")
print(f"   • 当前使用 {PREFERRED_MODEL} 模型，比 Mistral 快 {2 if PREFERRED_MODEL == 'phi' else 5}x")
print("   • 会话结束后仍需重新下载（但速度已大幅提升）")
print("   • 如需最佳质量，本地开发时可用 Mistral")
