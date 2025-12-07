# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置非交互式前端
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# curl: 下载 Ollama
# build-essential: 编译依赖
# procps: 进程管理
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 安装 Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 复制依赖文件并安装
COPY requirements.txt .
# 使用阿里云镜像源加速 pip 安装
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制项目文件
COPY . .

# 创建启动脚本
# 1. 显式设置 OLLAMA_HOST 为本地
# 2. 增加日志输出
# 3. 增加重试机制
RUN echo '#!/bin/bash\n\
export OLLAMA_MODELS=/home/user/.ollama/models\n\
export OLLAMA_HOST=127.0.0.1:11434\n\
\n\
echo "🚀 Starting application on ModelScope..."\n\
\n\
# 启动 Ollama\n\
echo "🔴 Starting Ollama..."\n\
ollama serve > ollama.log 2>&1 &\n\
\n\
echo "⏳ Waiting for Ollama to start..."\n\
sleep 5\n\
\n\
# 尝试拉取模型
echo "⬇️  Pulling model (qwen2:1.5b)..."
# 在后台拉取，不阻塞服务启动
(ollama pull qwen2:1.5b && echo "✅ Model pulled successfully") || echo "⚠️ Model pull failed" &\n\
\n\
# 启动 FastAPI\n\
echo "🟢 Starting FastAPI Server..."\n\
# 绑定 0.0.0.0:7860\n\
uvicorn server:app --host 0.0.0.0 --port 7860\n\
' > start.sh && chmod +x start.sh

# 创建非 root 用户
RUN useradd -m -u 1000 user
RUN mkdir -p /home/user/.ollama/models && chown -R user:user /home/user/.ollama
RUN mkdir -p /app && chown -R user:user /app

# 切换用户
USER user

# 设置环境变量
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH
ENV OLLAMA_MODELS=$HOME/.ollama/models
ENV OLLAMA_HOST=127.0.0.1:11434

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["/bin/bash", "/app/start.sh"]
