"""
Kaggle Ollama 保存脚本
将 Ollama 和模型保存到 Kaggle Dataset，下次直接使用

使用步骤:
1. 首次运行: 安装 Ollama 和下载模型后，运行本脚本保存
2. 后续使用: 使用 KAGGLE_LOAD_OLLAMA.py 从 Dataset 加载

注意: 需要手动创建 Kaggle Dataset 并上传
"""

import os
import subprocess
import shutil
import tarfile
import time
from pathlib import Path

print("="*70)
print("💾 Kaggle Ollama 保存工具")
print("="*70)

# ==================== 配置 ====================
OUTPUT_DIR = "/kaggle/working/ollama_backup"
MODEL_NAME = "mistral"  # 或者 "phi", "tinyllama" 等

print(f"\n📋 配置:")
print(f"   模型: {MODEL_NAME}")
print(f"   输出目录: {OUTPUT_DIR}")

# ==================== 步骤 1: 创建输出目录 ====================
print(f"\n📁 步骤 1/4: 创建备份目录...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"   ✅ 目录创建成功")

# ==================== 步骤 2: 备份 Ollama 二进制文件 ====================
print(f"\n📦 步骤 2/4: 备份 Ollama 二进制文件...")

ollama_bin = shutil.which('ollama')
if ollama_bin:
    print(f"   找到 Ollama: {ollama_bin}")
    
    # 复制二进制文件
    shutil.copy2(ollama_bin, os.path.join(OUTPUT_DIR, 'ollama'))
    print(f"   ✅ Ollama 二进制文件已备份")
else:
    print(f"   ❌ 未找到 Ollama，请先安装")
    exit(1)

# ==================== 步骤 3: 备份模型文件 ====================
print(f"\n🤖 步骤 3/4: 备份 {MODEL_NAME} 模型...")

# Ollama 模型存储位置（可能在不同位置）
possible_model_dirs = [
    os.path.expanduser("~/.ollama/models"),
    "/root/.ollama/models",
    os.path.expanduser("~/.ollama")
]

ollama_models_dir = None
for dir_path in possible_model_dirs:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # 检查是否有内容
        if os.listdir(dir_path):
            ollama_models_dir = os.path.dirname(dir_path) if dir_path.endswith('models') else dir_path
            break

if ollama_models_dir and os.path.exists(ollama_models_dir):
    print(f"   找到模型目录: {ollama_models_dir}")
    
    # 计算目录大小
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(ollama_models_dir)
        for filename in filenames
    )
    print(f"   模型总大小: {total_size / (1024**3):.2f} GB")
    
    # 创建压缩包（整个 .ollama 目录）
    models_archive = os.path.join(OUTPUT_DIR, 'ollama_models.tar.gz')
    print(f"   📦 创建压缩包（这可能需要几分钟）...")
    print(f"   正在压缩: {ollama_models_dir}")
    
    start_time = time.time()
    with tarfile.open(models_archive, 'w:gz') as tar:
        tar.add(ollama_models_dir, arcname='.ollama')
    
    elapsed = time.time() - start_time
    archive_size = os.path.getsize(models_archive) / (1024**3)
    
    print(f"   ✅ 压缩完成")
    print(f"      耗时: {int(elapsed)}秒")
    print(f"      压缩包大小: {archive_size:.2f} GB")
else:
    print(f"   ❌ 未找到模型目录")
    print(f"   请先运行: ollama pull {MODEL_NAME}")
    exit(1)

# ==================== 步骤 4: 生成说明文件 ====================
print(f"\n📝 步骤 4/4: 生成说明文件...")

readme_content = f"""# Ollama 备份包

## 内容
- `ollama`: Ollama 二进制文件
- `ollama_models.tar.gz`: 模型文件压缩包（包含 {MODEL_NAME}）

## 备份信息
- 备份时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 模型: {MODEL_NAME}
- 压缩包大小: {archive_size:.2f} GB

## 使用方法

### 1. 创建 Kaggle Dataset

1. 下载此目录中的所有文件到本地
2. 在 Kaggle 网站创建新 Dataset:
   - 访问: https://www.kaggle.com/datasets
   - 点击 "New Dataset"
   - 上传 `ollama` 和 `ollama_models.tar.gz`
   - 命名为: `ollama-{MODEL_NAME}-backup`
   - 设置为 Private
   - 点击 "Create"

### 2. 在 Notebook 中加载

在 Kaggle Notebook 中:

1. 添加 Dataset:
   - 点击右侧 "Add data" → "Your Datasets"
   - 选择你创建的 `ollama-{MODEL_NAME}-backup`

2. 运行加载脚本:
   ```python
   # 使用项目中的 KAGGLE_LOAD_OLLAMA.py
   exec(open('/kaggle/working/adaptive_RAG/KAGGLE_LOAD_OLLAMA.py').read())
   ```

### 3. 验证

```bash
# 检查 Ollama
ollama --version

# 检查模型
ollama list

# 测试运行
ollama run {MODEL_NAME} "Hello"
```

## 文件大小参考

不同模型的压缩包大小（近似值）:
- qwen:0.5b: ~350 MB
- tinyllama: ~600 MB
- phi: ~1.6 GB
- mistral: ~4 GB
- llama2:7b: ~3.8 GB

## 注意事项

1. ⚠️ Dataset 大小限制:
   - 免费用户: 每个 Dataset 最大 20GB
   - 需要确保压缩包 < 20GB

2. ⚠️ 上传时间:
   - 取决于你的网络速度
   - 4GB 文件可能需要 10-30 分钟

3. ✅ 优势:
   - 只需上传一次
   - 每次 Notebook 启动时直接加载（秒级）
   - 节省大量时间

## 故障排除

### 问题: 上传失败
解决: 检查网络连接，或分多次上传

### 问题: Dataset 过大
解决: 使用更小的模型（如 phi 或 tinyllama）

### 问题: 加载后 Ollama 无法运行
解决: 检查文件权限，运行 `chmod +x /usr/local/bin/ollama`
"""

readme_file = os.path.join(OUTPUT_DIR, 'README.md')
with open(readme_file, 'w', encoding='utf-8') as f:
    f.write(readme_content)

print(f"   ✅ 说明文件已生成")

# ==================== 生成加载脚本（供参考） ====================
loader_script = os.path.join(OUTPUT_DIR, 'load_example.py')
with open(loader_script, 'w', encoding='utf-8') as f:
    f.write(f'''"""
示例: 从 Kaggle Dataset 加载 Ollama
"""
import os
import subprocess
import tarfile
import shutil

# Dataset 路径（根据你的 Dataset 名称修改）
DATASET_PATH = "/kaggle/input/ollama-{MODEL_NAME}-backup"

print("📦 从 Dataset 加载 Ollama...")

# 1. 复制 Ollama 二进制文件
ollama_bin = os.path.join(DATASET_PATH, "ollama")
if os.path.exists(ollama_bin):
    shutil.copy2(ollama_bin, "/usr/local/bin/ollama")
    os.chmod("/usr/local/bin/ollama", 0o755)
    print("✅ Ollama 二进制文件已安装")

# 2. 解压模型文件
models_archive = os.path.join(DATASET_PATH, "ollama_models.tar.gz")
if os.path.exists(models_archive):
    print("📦 解压模型文件...")
    with tarfile.open(models_archive, 'r:gz') as tar:
        tar.extractall(os.path.expanduser("~/.ollama"))
    print("✅ 模型已解压")

# 3. 启动 Ollama 服务
print("🚀 启动 Ollama 服务...")
subprocess.Popen(['ollama', 'serve'])
import time
time.sleep(15)

print("✅ Ollama 已准备就绪!")
print("\\n验证:")
subprocess.run(['ollama', 'list'])
''')

print(f"   ✅ 示例脚本已生成")

# ==================== 显示文件列表 ====================
print(f"\n📊 备份内容:")
for item in os.listdir(OUTPUT_DIR):
    item_path = os.path.join(OUTPUT_DIR, item)
    size = os.path.getsize(item_path)
    size_str = f"{size / (1024**3):.2f} GB" if size > 1024**3 else f"{size / (1024**2):.2f} MB"
    print(f"   • {item}: {size_str}")

# ==================== 后续步骤说明 ====================
print("\n" + "="*70)
print("✅ 备份完成！")
print("="*70)

print(f"\n📋 后续步骤:")
print(f"""
1. 下载备份文件到本地:
   在 Kaggle Notebook 右侧 Output 中下载 {OUTPUT_DIR} 目录

2. 创建 Kaggle Dataset:
   • 访问: https://www.kaggle.com/datasets
   • 点击 "New Dataset"
   • 上传以下文件:
     - ollama (二进制文件)
     - ollama_models.tar.gz (模型压缩包)
   • 命名: ollama-{MODEL_NAME}-backup
   • 点击 "Create"

3. 下次使用:
   • 在 Notebook 中添加你的 Dataset
   • 运行 KAGGLE_LOAD_OLLAMA.py 脚本
   • 即可秒级加载，无需重新下载！

⏱️  时间对比:
   • 传统方式: 每次启动需要 5-10 分钟下载
   • Dataset 方式: 每次启动只需 10-20 秒加载
   • 节省时间: 每次节省 5-10 分钟！

💡 提示:
   • 上传 Dataset 是一次性工作
   • 之后每次 Notebook 启动都能快速加载
   • 强烈推荐！
""")

print("\n查看详细说明: cat {}/README.md".format(OUTPUT_DIR))
