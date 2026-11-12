#!/bin/bash
# 强制显示每一行命令，便于调试
set -x
# 遇到错误不退出，确保能打印尽可能多的日志
set +e

echo "=================================================="
echo "🚀 ENTRYPOINT SCRIPT STARTED AT $(date)"
echo "Force Log Refresh: $(date +%s)"
echo "=================================================="

# 显示详细环境信息 (调试用)
echo "📂 Current Directory: $(pwd)"
ls -la
echo "👤 Current User: $(whoami)"
id
echo "🐍 Python Version: $(python --version)"
echo "💾 Disk Usage:"
df -h


# 设置环境变量
export PYTHONUNBUFFERED=1
# 适配非 root 用户路径
export OLLAMA_MODELS="$HOME/.ollama/models"


# 检查文件是否存在
if [ -f "server.py" ]; then
    echo "✅ server.py found."
else
    echo "❌ server.py NOT found in $(pwd)!"
    ls -la
fi

# 启动 Python 应用
echo "🚀 Executing server.py..."
python server.py

# 如果 python server.py 退出，显示退出码
EXIT_CODE=$?
echo "⚠️ server.py exited with code $EXIT_CODE"

# 保持容器运行一小会儿，以便查看日志（如果崩溃太快）
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Application failed. Sleeping for 60s to allow log collection..."
    sleep 60
fi
