#!/bin/bash
export OLLAMA_MODELS=/home/user/.ollama/models
export OLLAMA_HOST=127.0.0.1:11434

echo "🚀 Starting application on ModelScope..."

# 启动 Ollama
echo "🔴 Starting Ollama..."
ollama serve > ollama.log 2>&1 &

echo "⏳ Waiting for Ollama to start..."
sleep 5

# 尝试拉取模型
echo "⬇️  Pulling model (qwen2:1.5b)..."
# 在后台拉取，不阻塞服务启动
(ollama pull qwen2:1.5b && echo "✅ Model pulled successfully") || echo "⚠️ Model pull failed" &

# 启动 FastAPI
echo "🟢 Starting FastAPI Server..."
# 绑定 0.0.0.0:7860
uvicorn server:app --host 0.0.0.0 --port 7860
