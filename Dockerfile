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
# 1. 启动 Ollama 服务后台运行
# 2. 下载需要的模型 (这里用 tinyllama 以便快速演示，你可以改为 mistral 或 llama3)
# 3. 启动 FastAPI 应用 (Hugging Face Spaces 要求监听 7860 端口)
RUN echo '#!/bin/bash\n\
echo "🔴 Starting Ollama..."\n\
ollama serve &\n\
echo "⏳ Waiting for Ollama to start..."\n\
sleep 5\n\
echo "⬇️  Pulling model..."\n\
ollama pull tinyllama\n\
echo "🟢 Starting FastAPI Server..."\n\
uvicorn server:app --host 0.0.0.0 --port 7860\n\
' > start.sh && chmod +x start.sh

# 创建非 root 用户 (Hugging Face 安全要求)
RUN useradd -m -u 1000 user
# 给用户 Ollama 目录的权限
RUN mkdir -p /.ollama && chmod 777 /.ollama
RUN mkdir -p /app && chown -R user:user /app

# 切换用户
USER user

# 设置环境变量
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH

# 暴露端口 (Hugging Face 默认端口)
EXPOSE 7860

# 启动命令
CMD ["./start.sh"]
