"""
Kaggle 会话初始化脚本
解决 Stop Session 后项目丢失的问题

使用方法：
在 Kaggle Notebook 第一个单元格运行：
    exec(open('/kaggle/input/your-dataset/KAGGLE_INIT.py').read())
或者直接复制此脚本内容到第一个单元格
"""

import os
import subprocess
import sys
from pathlib import Path

print("🚀 Kaggle 会话自动初始化")
print("="*70)

# ==================== 配置区域 ====================
REPO_URL = "https://github.com/LannyCodes/adaptive_RAG.git"
PROJECT_DIR = "/kaggle/working/adaptive_RAG"
PREVIOUS_RUN_INPUT = "/kaggle/input/output"  # 👈 修改为您保存的 Dataset 名称

# ==================== 1. 检查并克隆项目 ====================
print("\n📦 步骤 1: 检查项目状态...")

if os.path.exists(PROJECT_DIR):
    print(f"   ✅ 项目已存在: {PROJECT_DIR}")
    print("   ℹ️ 如需更新代码，请运行:")
    print(f"      cd {PROJECT_DIR} && git pull origin main")
else:
    print(f"   📥 项目不存在，开始克隆...")
    
    os.chdir('/kaggle/working')
    
    result = subprocess.run(
        ['git', 'clone', REPO_URL],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"   ✅ 项目克隆成功")
    else:
        print(f"   ❌ 克隆失败:")
        print(f"      {result.stderr}")
        print("\n   💡 可能的原因:")
        print("      1. 网络问题")
        print("      2. 仓库地址错误")
        print("      3. 仓库是私有的（需要认证）")
        sys.exit(1)

# ==================== 2. 恢复之前的数据 ====================
print("\n💾 步骤 2: 检查之前的运行数据...")

if os.path.exists(PREVIOUS_RUN_INPUT):
    print(f"   ✅ 发现之前的数据: {PREVIOUS_RUN_INPUT}")
    
    # 列出可恢复的文件
    saved_files = list(Path(PREVIOUS_RUN_INPUT).glob('*'))
    
    if saved_files:
        print(f"   📂 可恢复的文件:")
        for file in saved_files[:10]:  # 只显示前10个
            print(f"      • {file.name}")
        
        # 恢复知识图谱（如果存在）
        kg_file = Path(PREVIOUS_RUN_INPUT) / 'knowledge_graph.pkl'
        if kg_file.exists():
            import shutil
            dest = Path(PROJECT_DIR) / 'knowledge_graph.pkl'
            shutil.copy2(kg_file, dest)
            print(f"   ✅ 已恢复知识图谱")
        
        print(f"\n   💡 如需恢复其他文件，使用:")
        print(f"      import shutil")
        print(f"      shutil.copy2('{PREVIOUS_RUN_INPUT}/文件名', '{PROJECT_DIR}/文件名')")
    else:
        print("   ⚠️ 数据目录为空")
else:
    print("   ℹ️ 未发现之前的运行数据（首次运行）")
    print(f"   💡 会话结束时，将 /kaggle/working 保存为 Dataset")
    print(f"      命名为: output")

# ==================== 3. 设置工作环境 ====================
print("\n⚙️ 步骤 3: 设置工作环境...")

# 进入项目目录
os.chdir(PROJECT_DIR)

# 添加到 Python 路径
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

print(f"   ✅ 当前目录: {os.getcwd()}")
print(f"   ✅ Python 路径已更新")

# ==================== 4. 显示系统信息 ====================
print("\n📊 步骤 4: 系统信息...")

# Python 版本
print(f"   • Python: {sys.version.split()[0]}")

# GPU 状态
gpu_check = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if gpu_check.returncode == 0:
    # 提取 GPU 信息
    for line in gpu_check.stdout.split('\n'):
        if 'Tesla' in line or 'P100' in line or 'T4' in line:
            print(f"   • GPU: {line.strip()}")
            break
else:
    print("   • GPU: 不可用")

# 磁盘空间
disk_check = subprocess.run(['df', '-h', '/kaggle/working'], capture_output=True, text=True)
if disk_check.returncode == 0:
    lines = disk_check.stdout.strip().split('\n')
    if len(lines) > 1:
        info = lines[1].split()
        print(f"   • 可用空间: {info[3]}")

# ==================== 5. 快速测试 ====================
print("\n🧪 步骤 5: 快速测试...")

# 检查关键文件
key_files = [
    'entity_extractor.py',
    'graph_indexer.py',
    'knowledge_graph.py',
    'config.py'
]

all_files_exist = True
for file in key_files:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} 缺失")
        all_files_exist = False

if not all_files_exist:
    print("\n   ⚠️ 部分关键文件缺失，请检查仓库")

# ==================== 完成 ====================
print("\n" + "="*70)
print("✅ 初始化完成！")
print("="*70)
