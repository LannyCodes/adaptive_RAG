import subprocess
import time
import sys
import os
import threading

def stream_reader(pipe, prefix):
    """实时读取并打印子进程输出"""
    try:
        with pipe:
            for line in iter(pipe.readline, ''):
                print(f"[{prefix}] {line.strip()}")
                # 检查是否有启动成功的标志
                if "Uvicorn running on" in line:
                    print("✅ 检测到服务器启动成功标志！")
    except Exception:
        pass

def run_all_in_one():
    print("🚀 开始集成运行流程...")
    
    # 1. 启动服务器 (使用 threading 而不是 shell &)
    print("   正在启动 FastAPI 服务器...")
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 启动一个线程来实时打印服务器日志，防止缓冲区满卡死
    t = threading.Thread(target=stream_reader, args=(server_process.stdout, "Server"))
    t.daemon = True
    t.start()
    
    # 2. 等待一段时间让服务器初始化
    print("⏳ 等待服务器初始化 (15秒)...")
    time.sleep(15)
    
    # 3. 检查服务器进程是否还活着
    if server_process.poll() is not None:
        print(f"❌ 服务器进程已退出 (Exit Code: {server_process.returncode})")
        print("   请检查上方的 [Server] 日志")
        return

    # 4. 启动 Cloudflare 隧道
    if not os.path.exists("./cloudflared"):
        print("⚠️ 未找到 cloudflared，正在下载...")
        subprocess.run("wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared", shell=True)

    print("\n🌐 启动 Cloudflare 隧道...")
    tunnel_process = subprocess.Popen(
        ["./cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 5. 实时读取隧道输出，提取链接
    print("🔍 寻找公网链接中...")
    try:
        while True:
            line = tunnel_process.stdout.readline()
            if not line:
                break
            
            # 打印隧道日志
            # print(f"[Tunnel] {line.strip()}")
            
            if "trycloudflare.com" in line:
                import re
                url_match = re.search(r"https?://[\w\.-]+trycloudflare\.com", line)
                if url_match:
                    print("\n" + "="*60)
                    print("🎉 成功建立隧道！")
                    print(f"👉 公网访问地址: {url_match.group(0)}")
                    print("="*60 + "\n")
                    # 找到链接后，我们不仅不退出，还要跳出读取循环进入纯等待模式
                    # 否则继续读取可能会阻塞或读到 EOF 导致退出
                    break
            
            # 检查服务器是否还在运行
            if server_process.poll() is not None:
                print("❌ 警告：服务器进程意外退出！")
                break
        
        # 循环结束后，保持主线程存活
        print("ℹ️ 服务已就绪，主线程进入保活模式 (按 Stop 停止)...")
        while True:
            # 持续监控子进程状态
            if server_process.poll() is not None:
                print("❌ 警告：服务器进程意外退出！")
                break
            if tunnel_process.poll() is not None:
                print("❌ 警告：隧道进程意外退出！")
                break
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("正在停止服务...")
        server_process.terminate()
        tunnel_process.terminate()

if __name__ == "__main__":
    run_all_in_one()
