"""
Kaggle/Colab 启动脚本
用于启动 FastAPI 服务器并配置 ngrok 穿透
"""

import os
import sys
import subprocess
import time
import threading

def install_ngrok():
    """安装 pyngrok"""
    print("🔧 正在安装 pyngrok...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])

def run_server():
    """在后台运行服务器"""
    print("🚀 启动 FastAPI 服务器...")
    subprocess.Popen([sys.executable, "server.py"])

def start_ngrok():
    """启动 ngrok 穿透"""
    try:
        from pyngrok import ngrok
        
        # 尝试读取 token
        token = os.environ.get("NGROK_AUTHTOKEN")
        if not token:
            print("\n⚠️  警告: 未设置 NGROK_AUTHTOKEN 环境变量")
            print("   虽然可以运行，但连接时间会受限。建议在 Secrets 中设置。")
            # 尝试从输入读取（仅在交互模式下）
            # token = input("请输入 ngrok authtoken (可选): ")
        
        if token:
            ngrok.set_auth_token(token)

        # 建立隧道
        public_url = ngrok.connect(8000).public_url
        
        print("\n" + "="*60)
        print(f"✅ 成功穿透! 公网访问地址:")
        print(f"👉 {public_url}")
        print("="*60 + "\n")
        
        # 保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("正在关闭...")
            ngrok.kill()
            
    except ImportError:
        print("❌ pyngrok 导入失败，请确保已安装")
    except Exception as e:
        print(f"❌ ngrok 启动失败: {e}")

if __name__ == "__main__":
    # 1. 安装依赖
    try:
        import uvicorn
        import fastapi
    except ImportError:
        print("🔧 安装 FastAPI 依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart"])
    
    try:
        import pyngrok
    except ImportError:
        install_ngrok()

    # 2. 启动 FastAPI
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(3)
    
    # 3. 启动 ngrok
    start_ngrok()
