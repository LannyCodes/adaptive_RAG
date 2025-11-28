"""
Kaggle/Colab 启动脚本
用于启动 FastAPI 服务器并配置 ngrok 穿透
"""

import os
import sys
import subprocess
import time
import threading
import re
import shutil

def install_ngrok():
    """安装 pyngrok"""
    print("🔧 正在安装 pyngrok...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])

def run_server():
    """在后台运行服务器"""
    print("🚀 启动 FastAPI 服务器...")
    subprocess.Popen([sys.executable, "server.py"])

def start_ngrok():
    try:
        from pyngrok import ngrok
        token = os.environ.get("NGROK_AUTHTOKEN")
        if not token:
            print("\n⚠️  警告: 未设置 NGROK_AUTHTOKEN 环境变量")
            return False
        ngrok.set_auth_token(token)
        public_url = ngrok.connect(8000).public_url
        print("\n" + "="*60)
        print("✅ 成功穿透! 公网访问地址:")
        print(f"👉 {public_url}")
        print("="*60 + "\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            ngrok.kill()
        return True
    except ImportError:
        print("❌ pyngrok 导入失败，请确保已安装")
        return False
    except Exception as e:
        print(f"❌ ngrok 启动失败: {e}")
        return False

def start_cloudflared():
    try:
        cmd = None
        if shutil.which("cloudflared"):
            cmd = ["cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"]
        else:
            try:
                __import__("cloudflared")
                cmd = [sys.executable, "-m", "cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"]
            except Exception:
                print("⚠️ 未找到 cloudflared，可通过 'pip install cloudflared' 安装，或跳过穿透")
                return
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        url = None
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if "trycloudflare.com" in line:
                m = re.search(r"https?://[\w\.-]+trycloudflare\.com[\S]*", line)
                if m:
                    url = m.group(0)
                    print("\n" + "="*60)
                    print("✅ 成功穿透! 公网访问地址:")
                    print(f"👉 {url}")
                    print("="*60 + "\n")
                    break
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            proc.terminate()
    except Exception as e:
        print(f"❌ Cloudflare Tunnel 启动失败: {e}")

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
    
    use_tunnel = os.environ.get("USE_TUNNEL", "true").lower() == "true"
    if use_tunnel:
        ok = start_ngrok()
        if not ok:
            start_cloudflared()
    else:
        print("\n" + "="*60)
        print("✅ 服务器已启动，局域网访问地址:")
        print("👉 http://127.0.0.1:8000")
        print("="*60 + "\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
