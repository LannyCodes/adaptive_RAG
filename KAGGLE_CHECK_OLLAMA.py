"""
Kaggle Ollama 备份与加载 - 快速验证脚本

这个脚本帮助你验证 Ollama 和模型的位置，确保备份方案正确

在 Kaggle Notebook 中运行此脚本，检查环境
"""

import os
import subprocess
import shutil

print("="*70)
print("🔍 Kaggle Ollama 环境检查")
print("="*70)

# ==================== 检查 Ollama 安装 ====================
print("\n📍 步骤 1: 检查 Ollama 安装位置")

ollama_bin = shutil.which('ollama')
if ollama_bin:
    print(f"   ✅ Ollama 已安装")
    print(f"   📂 位置: {ollama_bin}")
    
    # 检查文件信息
    file_size = os.path.getsize(ollama_bin) / (1024**2)
    print(f"   📊 大小: {file_size:.2f} MB")
    
    # 检查版本
    version_result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
    if version_result.returncode == 0:
        print(f"   📌 版本: {version_result.stdout.strip()}")
else:
    print("   ❌ Ollama 未安装")
    print("   💡 请先运行安装:")
    print("      !curl -fsSL https://ollama.com/install.sh | sh")

# ==================== 检查 Ollama 服务 ====================
print("\n📍 步骤 2: 检查 Ollama 服务状态")

ps_check = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True)
if ps_check.returncode == 0:
    print("   ✅ Ollama 服务正在运行")
else:
    print("   ⚠️  Ollama 服务未运行")
    print("   💡 请启动服务:")
    print("      import subprocess, time")
    print("      subprocess.Popen(['ollama', 'serve'])")
    print("      time.sleep(15)")

# ==================== 检查模型位置 ====================
print("\n📍 步骤 3: 检查模型存储位置")

possible_dirs = [
    "~/.ollama",
    "/root/.ollama",
    "~/.ollama/models",
    "/root/.ollama/models"
]

found_dirs = []
for dir_path in possible_dirs:
    expanded_path = os.path.expanduser(dir_path)
    if os.path.exists(expanded_path):
        # 计算目录大小
        total_size = 0
        file_count = 0
        
        for dirpath, dirnames, filenames in os.walk(expanded_path):
            for filename in filenames:
                fp = os.path.join(dirpath, filename)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
                    file_count += 1
        
        size_gb = total_size / (1024**3)
        print(f"\n   ✅ 找到: {expanded_path}")
        print(f"      📊 大小: {size_gb:.2f} GB")
        print(f"      📁 文件数: {file_count}")
        
        # 显示目录结构
        print(f"      📂 内容:")
        for item in os.listdir(expanded_path)[:10]:  # 只显示前10个
            item_path = os.path.join(expanded_path, item)
            if os.path.isdir(item_path):
                print(f"         • {item}/ (目录)")
            else:
                size = os.path.getsize(item_path) / (1024**2)
                print(f"         • {item} ({size:.2f} MB)")
        
        found_dirs.append((expanded_path, size_gb))

if not found_dirs:
    print("\n   ❌ 未找到模型目录")
    print("   💡 请先下载模型:")
    print("      !ollama pull mistral")

# ==================== 检查已下载的模型 ====================
print("\n📍 步骤 4: 检查已下载的模型")

if ollama_bin and ps_check.returncode == 0:
    list_result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    if list_result.returncode == 0:
        print("\n   已下载的模型:")
        print("   " + "-"*60)
        print("   " + list_result.stdout)
    else:
        print("   ⚠️  无法获取模型列表")
        print("   请确保 Ollama 服务正在运行")
else:
    print("   ⚠️  Ollama 服务未运行，无法检查模型")

# ==================== 推荐备份方案 ====================
print("\n" + "="*70)
print("📋 推荐备份方案")
print("="*70)

if found_dirs:
    # 选择最大的目录（通常是完整的 .ollama 目录）
    backup_dir = max(found_dirs, key=lambda x: x[1])[0]
    backup_size = max(found_dirs, key=lambda x: x[1])[1]
    
    print(f"\n推荐备份目录: {backup_dir}")
    print(f"预计压缩包大小: ~{backup_size:.2f} GB")
    
    print(f"\n💾 备份步骤:")
    print(f"""
1. 使用 KAGGLE_SAVE_OLLAMA.py 脚本
   exec(open('KAGGLE_SAVE_OLLAMA.py').read())

2. 脚本会自动:
   • 找到 Ollama 二进制文件: {ollama_bin if ollama_bin else '未找到'}
   • 打包模型目录: {backup_dir}
   • 生成压缩包: /kaggle/working/ollama_backup/

3. 下载并创建 Dataset:
   • 在 Notebook 右侧 Output 下载 ollama_backup 目录
   • 访问 https://www.kaggle.com/datasets 创建 Dataset
   • 上传 ollama 和 ollama_models.tar.gz

4. 后续使用:
   • 添加 Dataset 到 Notebook
   • 运行 KAGGLE_LOAD_OLLAMA.py
   • 40-50秒完成加载！
""")

    # 估算上传时间
    upload_time_min = int(backup_size * 2)  # 假设 2 分钟/GB
    upload_time_max = int(backup_size * 5)  # 假设 5 分钟/GB
    
    print(f"⏱️  预计时间:")
    print(f"   • 压缩时间: {int(backup_size * 0.5)}-{int(backup_size)} 分钟")
    print(f"   • 下载时间: {int(backup_size * 1)}-{int(backup_size * 3)} 分钟（取决于网络）")
    print(f"   • 上传时间: {upload_time_min}-{upload_time_max} 分钟（取决于网络）")
    print(f"   • 首次总计: ~{int(backup_size * 4)}-{int(backup_size * 10)} 分钟（一次性）")
    print(f"   • 后续加载: 40-50 秒（每次）")
    
else:
    print("\n⚠️  未找到模型目录，无法提供备份方案")
    print("请先安装 Ollama 并下载模型")

# ==================== 环境摘要 ====================
print("\n" + "="*70)
print("📊 环境摘要")
print("="*70)

print(f"""
Ollama 安装: {'✅ 是' if ollama_bin else '❌ 否'}
Ollama 服务: {'✅ 运行中' if ps_check.returncode == 0 else '❌ 未运行'}
模型目录: {'✅ 找到 ' + str(len(found_dirs)) + ' 个' if found_dirs else '❌ 未找到'}
已下载模型: {'✅ 有' if ollama_bin and ps_check.returncode == 0 else '⚠️  无法确认'}

准备就绪: {'✅ 可以开始备份' if (ollama_bin and found_dirs) else '❌ 请先完成安装和模型下载'}
""")

if ollama_bin and found_dirs:
    print("💡 下一步: 运行 KAGGLE_SAVE_OLLAMA.py 开始备份")
else:
    print("💡 下一步: 完成 Ollama 安装和模型下载")

print("\n" + "="*70)
