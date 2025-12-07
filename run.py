import subprocess
import time
import os
import sys
import threading

def stream_logs(process, prefix):
    """实时读取子进程的日志并打印到标准输出"""
    for line in iter(process.stdout.readline, ''):
        print(f"[{prefix}] {line.strip()}", flush=True)

def main():
    print("🚀 Starting application via Python Runner...", flush=True)

    # 1. 设置环境变量
    # 确保使用 root 目录 (因为我们现在是用 root 运行)
    os.environ["OLLAMA_MODELS"] = "/root/.ollama/models"
    os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
    
    # 确保目录存在
    os.makedirs("/root/.ollama/models", exist_ok=True)

    # 2. 启动 Ollama
    print("🔴 Starting Ollama...", flush=True)
    # 使用 Popen 启动，将 stderr 重定向到 stdout 以便捕获
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    # 开启线程读取 Ollama 日志
    threading.Thread(target=stream_logs, args=(ollama_process, "OLLAMA"), daemon=True).start()

    # 等待 Ollama 启动
    print("⏳ Waiting for Ollama to initialize...", flush=True)
    time.sleep(5)

    # 3. 后台拉取模型 (不阻塞主线程)
    def pull_model():
        print("⬇️  Starting background model pull (qwen2:1.5b)...", flush=True)
        try:
            result = subprocess.run(
                ["ollama", "pull", "qwen2:1.5b"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Model pulled successfully!", flush=True)
            else:
                print(f"⚠️ Model pull failed: {result.stderr}", flush=True)
        except Exception as e:
            print(f"⚠️ Exception during model pull: {e}", flush=True)

    threading.Thread(target=pull_model, daemon=True).start()

    # 4. 启动 FastAPI (Uvicorn)
    print("🟢 Starting FastAPI Server...", flush=True)
    uvicorn_process = subprocess.Popen(
        ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    # 开启线程读取 Uvicorn 日志
    threading.Thread(target=stream_logs, args=(uvicorn_process, "FASTAPI"), daemon=True).start()

    # 5. 监控进程
    # 只要任何一个关键进程挂了，主程序就退出（以便 Docker 重启或报错）
    while True:
        if ollama_process.poll() is not None:
            print("❌ Ollama process exited unexpectedly!", flush=True)
            sys.exit(1)
        
        if uvicorn_process.poll() is not None:
            print("❌ Uvicorn process exited unexpectedly!", flush=True)
            sys.exit(1)
            
        time.sleep(1)

if __name__ == "__main__":
    main()
