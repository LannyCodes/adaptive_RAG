# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置非交互式前端，防止 apt-get 卡住或报错
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# curl: 下载 Ollama
# build-essential: 编译依赖
# procps: 提供 ps 命令用于调试
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 安装 Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 复制依赖文件并安装
COPY requirements.txt .
# 稍微放宽版本限制以避免安装失败
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建启动脚本
# 优化策略：
# 1. 显式设置 OLLAMA_HOST 为本地
# 2. 增加日志输出
RUN echo '#!/bin/bash\n\
export OLLAMA_MODELS=/home/user/.ollama/models\n\
export OLLAMA_HOST=127.0.0.1:11434\n\
\n\
echo "🚀 Starting application..."\n\
\n\
# 先启动 FastAPI，确保端口被监听，防止 Space 认为启动失败\n\
# 使用 nohup 后台运行 FastAPI\n\
echo "🟢 Starting FastAPI Server..."\n\
nohup uvicorn server:app --host 0.0.0.0 --port 7860 > server.log 2>&1 &\n\
PID=$!\n\
echo "✅ FastAPI started with PID $PID"\n\
\n\
# 启动 Ollama\n\
echo "🔴 Starting Ollama..."\n\
ollama serve > ollama.log 2>&1 &\n\
\n\
# 等待一会\n\
sleep 5\n\
\n\
# 尝试拉取模型 (如果失败也不要让容器崩溃)\n\
echo "⬇️  Pulling model..."\n\
ollama pull tinyllama || echo "⚠️ Model pull failed, but continuing..."\n\
\n\
# 保持主进程运行，并监控日志\n\
tail -f server.log ollama.log\n\
' > start.sh && chmod +x start.sh

# 创建非 root 用户 (Hugging Face 安全要求)
RUN useradd -m -u 1000 user

# 确保目录存在并赋予权限
RUN mkdir -p /home/user/.ollama/models && chown -R user:user /home/user/.ollama
RUN mkdir -p /app && chown -R user:user /app

# 切换用户
USER user

# 设置环境变量
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH
ENV OLLAMA_MODELS=$HOME/.ollama/models

# 暴露端口 (Hugging Face 默认端口)
EXPOSE 7860

# 启动命令
CMD ["./start.sh"]
