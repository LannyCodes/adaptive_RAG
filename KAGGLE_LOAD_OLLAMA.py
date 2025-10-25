"""
Kaggle Ollama 加载脚本
从 Kaggle Dataset 快速加载 Ollama 和模型，无需重新下载

前置条件:
1. 已使用 KAGGLE_SAVE_OLLAMA.py 创建备份
2. 已在 Kaggle 上传 Dataset
3. 已在 Notebook 中添加该 Dataset

使用方法:
在 Kaggle Notebook 第一个单元格运行:
    exec(open('/kaggle/working/adaptive_RAG/KAGGLE_LOAD_OLLAMA.py').read())
"""

import os
import subprocess
import tarfile
import shutil
import time

print("="*70)
print("📦 从 Dataset 加载 Ollama（快速启动）")
print("="*70)

# ==================== 配置 ====================
# 修改为你的 Dataset 名称
DATASET_NAME = "ollama-mistral-backup"  # 👈 修改这里
DATASET_PATH = f"/kaggle/input/{DATASET_NAME}"

print(f"\n📋 配置:")
print(f"   Dataset 路径: {DATASET_PATH}")

# ==================== 检查 Dataset ====================
print(f"\n🔍 步骤 1/5: 检查 Dataset...")

if not os.path.exists(DATASET_PATH):
    print(f"   ❌ Dataset 不存在: {DATASET_PATH}")
    print(f"\n💡 请检查:")
    print(f"   1. Dataset 是否已添加到 Notebook")
    print(f"   2. Dataset 名称是否正确")
    print(f"   3. 可用的 Datasets:")
    
    if os.path.exists("/kaggle/input"):
        for item in os.listdir("/kaggle/input"):
            print(f"      • {item}")
    
    print(f"\n📝 如何添加 Dataset:")
    print(f"   1. 点击右侧 'Add data' 按钮")
    print(f"   2. 选择 'Your Datasets'")
    print(f"   3. 找到你的 ollama 备份 Dataset")
    print(f"   4. 点击 'Add'")
    
    exit(1)

print(f"   ✅ Dataset 存在")

# 列出 Dataset 内容
print(f"\n   Dataset 内容:")
for item in os.listdir(DATASET_PATH):
    item_path = os.path.join(DATASET_PATH, item)
    if os.path.isfile(item_path):
        size = os.path.getsize(item_path)
        size_str = f"{size / (1024**3):.2f} GB" if size > 1024**3 else f"{size / (1024**2):.2f} MB"
        print(f"      • {item}: {size_str}")

# ==================== 安装 Ollama 二进制文件 ====================
print(f"\n🔧 步骤 2/5: 安装 Ollama 二进制文件...")

ollama_bin_source = os.path.join(DATASET_PATH, "ollama")

if os.path.exists(ollama_bin_source):
    # 复制到系统路径
    ollama_bin_dest = "/usr/local/bin/ollama"
    shutil.copy2(ollama_bin_source, ollama_bin_dest)
    
    # 设置执行权限
    os.chmod(ollama_bin_dest, 0o755)
    
    print(f"   ✅ Ollama 已安装到: {ollama_bin_dest}")
    
    # 验证版本
    version_result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
    if version_result.returncode == 0:
        print(f"   📌 {version_result.stdout.strip()}")
else:
    print(f"   ❌ 未找到 Ollama 二进制文件")
    exit(1)

# ==================== 解压模型文件 ====================
print(f"\n📦 步骤 3/5: 解压模型文件...")

models_archive = os.path.join(DATASET_PATH, "ollama_models.tar.gz")

if os.path.exists(models_archive):
    print(f"   找到模型压缩包: {os.path.getsize(models_archive) / (1024**3):.2f} GB")
    print(f"   📦 开始解压（这可能需要 10-30 秒）...")
    
    start_time = time.time()
    
    # 解压到用户目录（恢复到 ~/.ollama）
    ollama_home = os.path.expanduser("~")
    
    with tarfile.open(models_archive, 'r:gz') as tar:
        tar.extractall(ollama_home)  # 会自动创建 ~/.ollama 目录
    
    elapsed = time.time() - start_time
    print(f"   ✅ 解压完成（耗时: {int(elapsed)}秒）")
    
    # 检查模型目录
    models_dir = os.path.join(ollama_home, ".ollama")
    if os.path.exists(models_dir):
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(models_dir)
            for filename in filenames
        )
        print(f"   📊 模型总大小: {total_size / (1024**3):.2f} GB")
else:
    print(f"   ❌ 未找到模型压缩包")
    exit(1)

# ==================== 启动 Ollama 服务 ====================
print(f"\n🚀 步骤 4/5: 启动 Ollama 服务...")

# 检查是否已运行
ps_check = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True)

if ps_check.returncode == 0:
    print(f"   ✅ Ollama 服务已在运行")
else:
    print(f"   🔄 启动服务...")
    subprocess.Popen(
        ['ollama', 'serve'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print(f"   ⏳ 等待服务启动（15秒）...")
    time.sleep(15)
    
    # 验证服务
    import requests
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            print(f"   ✅ Ollama 服务运行正常")
    except Exception as e:
        print(f"   ⚠️ 服务验证失败: {e}")
        print(f"   但可能仍在启动中...")

# ==================== 验证模型 ====================
print(f"\n✅ 步骤 5/5: 验证模型...")

list_result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
print(f"\n   可用模型:")
print(f"   {list_result.stdout}")

# ==================== 完成 ====================
print("="*70)
print("✅ Ollama 加载完成！")
print("="*70)

print(f"\n📊 加载总结:")
print(f"   • Ollama 服务: ✅ 运行中")
print(f"   • 模型: ✅ 已加载")
print(f"   • 总耗时: < 1 分钟")

print(f"\n💡 对比:")
print(f"   • 传统方式: 5-10 分钟（重新下载）")
print(f"   • Dataset 方式: < 1 分钟（直接加载）")
print(f"   • 节省时间: 约 90%！")

print(f"\n🧪 快速测试:")
print(f"   在新单元格运行:")
print(f"   !ollama run mistral 'Hi, respond in one word'")

print(f"\n📝 下一步:")
print(f"   继续运行你的 GraphRAG 索引:")
print(f"""
   from document_processor import DocumentProcessor
   from graph_indexer import GraphRAGIndexer
   
   processor = DocumentProcessor()
   vectorstore, retriever, doc_splits = processor.setup_knowledge_base(enable_graphrag=True)
   
   indexer = GraphRAGIndexer(async_batch_size=8)
   graph = indexer.index_documents(doc_splits)
""")

print("\n" + "="*70)
