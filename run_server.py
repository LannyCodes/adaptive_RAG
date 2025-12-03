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

def install_ngrok(max_retries: int = 3):
    """安装 pyngrok 和 cloudflared（使用国内镜像，失败自动重试）"""
    print("🔧 正在安装 Web 穿透工具...")
    mirrors = [
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.aliyun.com/pypi/simple",
        None,  # 退回默认源
    ]
    for attempt in range(1, max_retries + 1):
        mirror = mirrors[min(attempt - 1, len(mirrors) - 1)]
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--default-timeout",
            "120",
        ]
        if mirror:
            cmd.extend(["-i", mirror])
        cmd.extend(["pyngrok", "cloudflared"])
        try:
            print(f"⏳ 第 {attempt} 次安装，使用源: {mirror or '默认 PyPI'}")
            subprocess.check_call(cmd)
            print("✅ 穿透工具安装完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 安装失败: {e}")
            time.sleep(5)
    print("❌ 多次尝试后仍无法安装 pyngrok/cloudflared")
    return False

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
        # 1. 检查系统路径
        if shutil.which("cloudflared"):
            cmd = ["cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"]
        # 2. 检查当前目录
        elif os.path.exists("./cloudflared"):
            cmd = ["./cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"]
            # 确保有执行权限
            try:
                os.chmod("./cloudflared", 0o755)
            except:
                pass
        else:
            # 如果找不到 cloudflared 二进制，尝试通过 pip 安装的 cloudflared 运行
            # 注意：cloudflared 的 pip 包可能不直接暴露 cloudflared 命令
            # 我们尝试直接下载二进制文件
            print("⚠️ 未找到 cloudflared 命令，尝试下载二进制文件...")
            try:
                # 这里简化处理，如果 pip 安装的模块无法直接运行，提示用户手动安装
                # 或者尝试使用 pyngrok 作为回退
                print("⚠️ 无法通过 Python 模块启动 cloudflared，将尝试仅使用 pyngrok")
                return
            except Exception:
                print("⚠️ 未找到 cloudflared，可通过 'pip install cloudflared' 安装，或跳过穿透")
                return
        
        if cmd:
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
        
        # 检查 cloudflared 是否存在，如果不存在尝试安装
    if not shutil.which("cloudflared"):
        # 尝试作为 Python 模块调用，但先不导入它来检查，而是直接看 pip list 或依赖 subprocess
        # 由于 cloudflared 库可能有导入问题，我们这里只做安装尝试，不做导入检查
        pass

    # 2. 启动 FastAPI
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # 等待服务器启动 (循环检查端口)
    print("⏳ 等待服务器启动...")
    import socket
    def wait_for_port(port, host='127.0.0.1', timeout=60):
        start_time = time.time()
        while True:
            try:
                with socket.create_connection((host, port), timeout=1):
                    print(f"✅ 服务器已在 {host}:{port} 就绪")
                    return True
            except (OSError, ConnectionRefusedError):
                if time.time() - start_time > timeout:
                    print(f"❌ 服务器启动超时 ({timeout}s)")
                    return False
                time.sleep(1)

    if not wait_for_port(8000):
        print("❌ 服务器未能成功启动，请检查日志")
        sys.exit(1)
    
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
