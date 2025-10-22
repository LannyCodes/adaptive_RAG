#!/usr/bin/env python3
"""
Google Colab环境下的GraphRAG完整运行脚本
解决Ollama服务启动和GraphRAG运行的问题

使用方法:
1. 在Colab中启用GPU
2. 复制此文件到Colab
3. 运行: !python colab_setup_and_run.py
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

print("="*70)
print("🚀 GraphRAG Colab 自动化部署脚本")
print("="*70)

# ============================================================
# 1️⃣ 检测Colab环境
# ============================================================
def check_colab_environment():
    """检测是否在Colab环境中"""
    try:
        import google.colab
        print("\n✅ 运行环境: Google Colab")
        return True
    except ImportError:
        print("\n⚠️  警告: 未检测到Colab环境")
        print("   本脚本为Colab优化，在其他环境可能需要调整")
        return False

# ============================================================
# 2️⃣ 安装Ollama
# ============================================================
def install_ollama():
    """在Colab中安装Ollama"""
    print("\n" + "="*70)
    print("📦 步骤1: 安装Ollama")
    print("="*70)
    
    # 检查是否已安装
    if os.path.exists("/usr/local/bin/ollama"):
        print("✅ Ollama已安装")
        return True
    
    print("\n📥 下载并安装Ollama...")
    try:
        # 下载Ollama安装脚本
        subprocess.run(
            ["curl", "-fsSL", "https://ollama.com/install.sh", "-o", "/tmp/install_ollama.sh"],
            check=True,
            capture_output=True
        )
        
        # 执行安装
        subprocess.run(
            ["sh", "/tmp/install_ollama.sh"],
            check=True,
            capture_output=True
        )
        
        print("✅ Ollama安装成功")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama安装失败: {e}")
        return False

# ============================================================
# 3️⃣ 后台启动Ollama服务
# ============================================================
def start_ollama_service():
    """在后台启动Ollama服务"""
    print("\n" + "="*70)
    print("🔧 步骤2: 启动Ollama服务")
    print("="*70)
    
    print("\n🔄 在后台启动Ollama服务...")
    
    # 方法1: 使用subprocess后台运行
    try:
        # 启动Ollama服务（后台）
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setpgrp  # 创建新的进程组
        )
        
        # 等待服务启动
        print("⏳ 等待Ollama服务启动...")
        time.sleep(5)
        
        # 检查服务是否运行
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True,
                timeout=3
            )
            
            if result.returncode == 0:
                print("✅ Ollama服务已启动 (PID: {})".format(ollama_process.pid))
                
                # 保存进程ID以便后续管理
                with open("/tmp/ollama.pid", "w") as f:
                    f.write(str(ollama_process.pid))
                
                return ollama_process
            else:
                print("⚠️  服务启动可能有问题，继续尝试...")
                
        except subprocess.TimeoutExpired:
            print("⚠️  服务检查超时，但进程已启动")
            return ollama_process
            
    except Exception as e:
        print(f"❌ 启动Ollama失败: {e}")
        return None

# ============================================================
# 4️⃣ 下载Mistral模型
# ============================================================
def pull_mistral_model():
    """下载Mistral模型"""
    print("\n" + "="*70)
    print("📥 步骤3: 下载Mistral模型")
    print("="*70)
    
    print("\n🔄 拉取mistral模型（这可能需要几分钟）...")
    
    try:
        # 检查模型是否已存在
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "mistral" in result.stdout:
            print("✅ Mistral模型已存在")
            return True
        
        # 下载模型
        print("📥 开始下载Mistral模型...")
        process = subprocess.Popen(
            ["ollama", "pull", "mistral"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # 实时显示下载进度
        for line in process.stdout:
            print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ Mistral模型下载完成")
            return True
        else:
            print("❌ 模型下载失败")
            return False
            
    except Exception as e:
        print(f"❌ 下载Mistral模型失败: {e}")
        return False

# ============================================================
# 5️⃣ 安装Python依赖
# ============================================================
def install_python_dependencies():
    """安装GraphRAG所需的Python包"""
    print("\n" + "="*70)
    print("📦 步骤4: 安装Python依赖")
    print("="*70)
    
    packages = [
        "langchain",
        "langchain-community",
        "langchain-core",
        "langgraph",
        "langchain-ollama",
        "chromadb",
        "sentence-transformers",
        "tiktoken",
        "beautifulsoup4",
        "requests",
        "tavily-python",
        "python-dotenv",
        "networkx",
        "python-louvain",
        "torch",
        "transformers"
    ]
    
    print("\n📥 安装必要的Python包...")
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"📥 安装 {package}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                check=True
            )
    
    print("\n✅ 所有依赖安装完成")

# ============================================================
# 6️⃣ 配置环境变量
# ============================================================
def setup_environment():
    """配置环境变量"""
    print("\n" + "="*70)
    print("🔑 步骤5: 配置环境变量")
    print("="*70)
    
    # 检查.env文件
    if os.path.exists(".env"):
        print("\n✅ 发现.env文件，加载配置...")
        from dotenv import load_dotenv
        load_dotenv()
    else:
        print("\n⚠️  未找到.env文件")
        
        # 交互式输入API密钥
        if "TAVILY_API_KEY" not in os.environ:
            from getpass import getpass
            api_key = getpass("请输入TAVILY_API_KEY (或按Enter跳过): ")
            if api_key:
                os.environ["TAVILY_API_KEY"] = api_key
                print("✅ TAVILY_API_KEY已设置")
            else:
                print("⚠️  跳过TAVILY_API_KEY设置（网络搜索功能将不可用）")
    
    print("\n📋 当前环境变量:")
    print(f"   TAVILY_API_KEY: {'已设置' if os.environ.get('TAVILY_API_KEY') else '未设置'}")

# ============================================================
# 7️⃣ 运行GraphRAG
# ============================================================
def run_graphrag():
    """运行GraphRAG主程序"""
    print("\n" + "="*70)
    print("🚀 步骤6: 运行GraphRAG")
    print("="*70)
    
    # 检查main_graphrag.py是否存在
    if not os.path.exists("main_graphrag.py"):
        print("\n❌ 未找到main_graphrag.py文件")
        print("   请确保已上传项目文件到Colab")
        return False
    
    print("\n🔄 启动GraphRAG索引构建...\n")
    
    try:
        # 运行GraphRAG
        result = subprocess.run(
            [sys.executable, "main_graphrag.py"],
            capture_output=False,  # 实时输出
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ GraphRAG运行成功!")
            return True
        else:
            print(f"\n❌ GraphRAG运行失败 (返回码: {result.returncode})")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        return False
    except Exception as e:
        print(f"\n❌ 运行GraphRAG时出错: {e}")
        return False

# ============================================================
# 8️⃣ 清理函数
# ============================================================
def cleanup():
    """清理后台进程"""
    print("\n" + "="*70)
    print("🧹 清理后台进程")
    print("="*70)
    
    # 停止Ollama服务
    if os.path.exists("/tmp/ollama.pid"):
        try:
            with open("/tmp/ollama.pid", "r") as f:
                pid = int(f.read().strip())
            
            os.kill(pid, signal.SIGTERM)
            print(f"✅ Ollama服务已停止 (PID: {pid})")
            os.remove("/tmp/ollama.pid")
            
        except Exception as e:
            print(f"⚠️  停止Ollama服务失败: {e}")

# ============================================================
# 主函数
# ============================================================
def main():
    """主执行流程"""
    ollama_process = None
    
    try:
        # 1. 检测环境
        is_colab = check_colab_environment()
        
        # 2. 安装Ollama
        if not install_ollama():
            print("\n❌ Ollama安装失败，无法继续")
            return
        
        # 3. 启动Ollama服务
        ollama_process = start_ollama_service()
        if not ollama_process:
            print("\n❌ Ollama服务启动失败，无法继续")
            return
        
        # 4. 下载模型
        if not pull_mistral_model():
            print("\n❌ Mistral模型下载失败，无法继续")
            return
        
        # 5. 安装Python依赖
        install_python_dependencies()
        
        # 6. 配置环境
        setup_environment()
        
        # 7. 运行GraphRAG
        success = run_graphrag()
        
        if success:
            print("\n" + "="*70)
            print("✅ 所有任务完成!")
            print("="*70)
            
            print("\n📊 生成的文件:")
            if os.path.exists("data/knowledge_graph.json"):
                print("   ✅ data/knowledge_graph.json")
                
                # 提供下载选项
                if is_colab:
                    print("\n💾 下载结果:")
                    print("   from google.colab import files")
                    print("   files.download('data/knowledge_graph.json')")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        
    except Exception as e:
        print(f"\n❌ 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理
        print("\n⚠️  注意: Ollama服务仍在后台运行")
        print("   如需停止: !pkill -f 'ollama serve'")
        print("   或运行: cleanup()")

if __name__ == "__main__":
    main()
