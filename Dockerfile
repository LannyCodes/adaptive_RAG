# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# curl: 下载 Ollama
# build-essential: 编译某些 Python 库可能需要
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
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
# 1. 设置 OLLAMA_MODELS 环境变量到用户目录
# 2. 启动 Ollama
# 3. 后台拉取模型 (不阻塞服务器启动)
# 4. 启动 FastAPI (尽快监听端口以通过健康检查)
RUN echo '#!/bin/bash\n\
export OLLAMA_MODELS=/home/user/.ollama/models\n\
\n\
echo "🔴 Starting Ollama..."\n\
ollama serve &\n\
\n\
echo "⏳ Waiting for Ollama to start..."\n\
sleep 5\n\
\n\
echo "⬇️  Pulling model in background..."\n\
ollama pull tinyllama &\n\
\n\
echo "🟢 Starting FastAPI Server..."\n\
uvicorn server:app --host 0.0.0.0 --port 7860\n\
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
