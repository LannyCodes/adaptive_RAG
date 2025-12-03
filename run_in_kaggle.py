import subprocess
import time
import sys
import os

def run_background():
    print("🚀 正在后台启动服务...")
    
    # 1. 启动服务器 (输出重定向到 server.log)
    server_log = open("server.log", "w")
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=server_log,
        stderr=subprocess.STDOUT
    )
    print(f"✅ 服务器已启动 (PID: {server_process.pid})")
    
    # 2. 等待服务器就绪
    print("⏳ 等待服务器启动 (约10秒)...")
    time.sleep(10)
    
    # 3. 启动 Cloudflare 隧道
    if not os.path.exists("./cloudflared"):
        print("❌ 未找到 cloudflared，请先运行下载命令")
        return

    print("🌐 正在建立公网连接...")
    tunnel_process = subprocess.Popen(
        ["./cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # 4. 读取并打印访问链接
    print("🔍 正在获取访问地址...")
    found_url = False
    while True:
        line = tunnel_process.stdout.readline()
        if not line:
            break
        
        # 打印隧道日志以便调试
        # print(f"[Tunnel] {line.strip()}")
        
        if "trycloudflare.com" in line:
            import re
            url_match = re.search(r"https?://[\w\.-]+trycloudflare\.com", line)
            if url_match:
                print("\n" + "="*50)
                print("🎉 成功！请访问以下地址：")
                print(f"👉 {url_match.group(0)}")
                print("="*50 + "\n")
                found_url = True
                break
    
    if not found_url:
        print("⚠️ 未能获取到公网地址，请检查 server.log")

if __name__ == "__main__":
    run_background()
