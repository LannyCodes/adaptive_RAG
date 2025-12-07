import subprocess
import time
import os
import sys
import threading

def main():
    # 强制刷新 stdout
    print("🚀 Starting application via Python Runner (Direct Logging Mode)...", flush=True)

    # 1. 设置环境变量
    # 确保使用 root 目录
    os.environ["OLLAMA_MODELS"] = "/root/.ollama/models"
    os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
    
    # 确保目录存在
    os.makedirs("/root/.ollama/models", exist_ok=True)

    # 2. 启动 Ollama
    print("🔴 Starting Ollama...", flush=True)
    # 不使用 PIPE，直接继承父进程的 stdout/stderr，确保日志直接输出到 Docker
    ollama_process = subprocess.Popen(
        ["ollama", "serve"]
    )
    
    # 等待 Ollama 启动
    print("⏳ Waiting for Ollama to initialize (5s)...", flush=True)
    time.sleep(5)

    # 3. 后台拉取模型 (不阻塞主线程)
    def pull_model():
        print("⬇️  Starting background model pull (qwen2:1.5b)...", flush=True)
        try:
            # 直接调用，让 ollama 自己打印进度到 stdout
            subprocess.run(["ollama", "pull", "qwen2:1.5b"], check=False)
            print("✅ Model pull process finished.", flush=True)
        except Exception as e:
            print(f"⚠️ Exception during model pull: {e}", flush=True)

    threading.Thread(target=pull_model, daemon=True).start()

    # 4. 启动 FastAPI (Uvicorn)
    print("🟢 Starting FastAPI Server...", flush=True)
    # 直接继承 stdout/stderr
    uvicorn_process = subprocess.Popen(
        ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
    )

    # 5. 监控进程
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
