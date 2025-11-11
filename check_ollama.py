"""
Ollama服务快速检查工具
在启动RAG系统前检查Ollama是否运行
"""

import requests
import subprocess
import sys


def check_ollama_service():
    """检查Ollama服务是否运行"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print("✅ Ollama服务运行正常")
            
            # 显示已下载的模型
            try:
                models = response.json().get('models', [])
                if models:
                    print(f"\n📦 已下载的模型 ({len(models)}个):")
                    for model in models:
                        print(f"   - {model['name']}")
                else:
                    print("\n⚠️  没有已下载的模型")
                    print("请运行: ollama pull mistral")
            except:
                pass
            
            return True
        else:
            return False
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def start_ollama_service():
    """尝试启动Ollama服务"""
    print("🔧 正在尝试启动Ollama服务...")
    try:
        # 后台启动Ollama
        process = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 等待几秒让服务启动
        import time
        time.sleep(3)
        
        # 再次检查
        if check_ollama_service():
            print("✅ Ollama服务已成功启动")
            return True
        else:
            print("❌ Ollama服务启动失败")
            return False
    except FileNotFoundError:
        print("❌ 未找到ollama命令，请先安装Ollama")
        print("   安装命令: curl -fsSL https://ollama.com/install.sh | sh")
        return False
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("🔍 Ollama服务检查工具")
    print("=" * 60)
    
    if check_ollama_service():
        print("\n✅ 检查通过，可以启动RAG系统")
        return 0
    else:
        print("\n❌ Ollama服务未运行")
        
        # 询问是否自动启动
        print("\n是否尝试自动启动Ollama服务？")
        print("1. 是（推荐）")
        print("2. 否，手动启动")
        
        choice = input("\n请选择 [1/2]: ").strip()
        
        if choice == "1":
            if start_ollama_service():
                print("\n✅ 现在可以启动RAG系统了")
                return 0
            else:
                print("\n❌ 自动启动失败，请手动启动")
                print_manual_instructions()
                return 1
        else:
            print_manual_instructions()
            return 1


def print_manual_instructions():
    """打印手动启动说明"""
    print("\n" + "=" * 60)
    print("📖 手动启动Ollama服务")
    print("=" * 60)
    print("\n方式1: 在终端运行")
    print("  $ ollama serve")
    print("\n方式2: 在Python中运行")
    print("  import subprocess")
    print("  subprocess.Popen(['ollama', 'serve'])")
    print("\n方式3: 在Kaggle Notebook中运行")
    print("  %run KAGGLE_LOAD_OLLAMA.py")
    print("\n启动后请重新运行RAG系统")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
