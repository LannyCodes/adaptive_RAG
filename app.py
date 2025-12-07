import subprocess
import time
import os
import sys
import threading

def main():
    # 强制刷新 stdout
    print("🚀 Starting application via Python Runner (Dual Logging Mode)...", flush=True)

    # 打开日志文件
    server_log = open("server.log", "w")
    
    # 重定向 stdout/stderr 到文件，同时保留 stdout (使用 tee 很难在 python 内部做，所以我们手动写)
    def log(message):
        print(message, flush=True)
        server_log.write(message + "\n")
        server_log.flush()

    log("🚀 App started. Initializing environment...")

    # 1. 设置环境变量
    # 动态获取 HOME 目录，适配 root 或 user 用户
    user_home = os.environ.get("HOME", "/root")
    ollama_models_dir = os.path.join(user_home, ".ollama/models")
    
    os.environ["OLLAMA_MODELS"] = ollama_models_dir
    os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
    
    # 确保目录存在
    os.makedirs(ollama_models_dir, exist_ok=True)

    # 2. 启动 Ollama
    log("🔴 Starting Ollama...")
    # 将 Ollama 的输出重定向到文件
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=server_log,
        stderr=server_log
    )
    
    # 等待 Ollama 启动
    log("⏳ Waiting for Ollama to initialize (5s)...")
    time.sleep(5)

    # 3. 后台拉取模型 (不阻塞主线程)
    def pull_model():
        log("⬇️  Starting background model pull (qwen2:1.5b)...")
        try:
            # 同样重定向输出
            subprocess.run(["ollama", "pull", "qwen2:1.5b"], stdout=server_log, stderr=server_log, check=False)
            log("✅ Model pull process finished.")
        except Exception as e:
            log(f"⚠️ Exception during model pull: {e}")

    threading.Thread(target=pull_model, daemon=True).start()

    # 4. 启动 FastAPI (Uvicorn)
    log("🟢 Starting FastAPI Server...")
    # Uvicorn 输出也写入日志文件
    uvicorn_process = subprocess.Popen(
        ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"],
        stdout=server_log,
        stderr=server_log
    )

    # 5. 监控进程
    while True:
        if ollama_process.poll() is not None:
            log("❌ Ollama process exited unexpectedly!")
            sys.exit(1)
        
        if uvicorn_process.poll() is not None:
            log("❌ Uvicorn process exited unexpectedly!")
            sys.exit(1)
            
        time.sleep(1)

if __name__ == "__main__":
    main()
