#!/bin/bash
# 遇到错误不退出，确保能打印尽可能多的日志
set +e

echo "=================================================="
echo "🚀 ENTRYPOINT SCRIPT STARTED AT $(date)"
echo "=================================================="

# 显示环境信息
echo "📂 Current Directory: $(pwd)"
echo "👤 Current User: $(whoami)"
echo "🐍 Python Version: $(python --version)"

# 设置环境变量
export PYTHONUNBUFFERED=1
export OLLAMA_MODELS="/root/.ollama/models"

# 检查文件是否存在
if [ -f "app.py" ]; then
    echo "✅ app.py found."
else
    echo "❌ app.py NOT found in $(pwd)!"
    ls -la
fi

# 启动 Python 应用
echo "🚀 Executing app.py..."
python app.py

# 如果 python app.py 退出，显示退出码
EXIT_CODE=$?
echo "⚠️ app.py exited with code $EXIT_CODE"

# 保持容器运行一小会儿，以便查看日志（如果崩溃太快）
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Application failed. Sleeping for 60s to allow log collection..."
    sleep 60
fi
