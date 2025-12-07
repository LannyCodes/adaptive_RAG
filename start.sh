#!/bin/bash
# 开启调试模式
set -x

# 设置环境变量 (确保与 Dockerfile 一致)
export OLLAMA_MODELS=/root/.ollama/models
export OLLAMA_HOST=127.0.0.1:11434

echo "🚀 Starting application on ModelScope (Root Mode)..."

# 启动 Ollama
echo "🔴 Starting Ollama..."
# 确保目录存在
mkdir -p $OLLAMA_MODELS
ollama serve > ollama.log 2>&1 &

echo "⏳ Waiting for Ollama to start..."
sleep 5

# 尝试拉取模型
echo "⬇️  Pulling model (qwen2:1.5b)..."
# 在后台拉取
(ollama pull qwen2:1.5b && echo "✅ Model pulled successfully") || echo "⚠️ Model pull failed" &

# 启动 FastAPI
echo "🟢 Starting FastAPI Server..."
# 使用 nohup 后台运行，并重定向日志
nohup uvicorn server:app --host 0.0.0.0 --port 7860 > server.log 2>&1 &

# 等待几秒
sleep 2

# 检查进程是否存活
if pgrep -f "uvicorn" > /dev/null; then
    echo "✅ FastAPI is running."
else
    echo "❌ FastAPI failed to start! Checking logs:"
    cat server.log
fi

# 保持容器运行，并输出日志到 stdout 供 ModelScope 收集
echo "📜 Tailing logs..."
tail -f server.log ollama.log
