# 使用 Python 3.10 作为基础镜像 (尝试强制刷新构建缓存)
FROM python:3.10-slim

# 设置非交互式前端
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 1. 只安装最基础的系统依赖 (去掉了 curl 和 build-essential 以加快构建)
RUN apt-get update && apt-get install -y \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 2. 暂时跳过 Ollama 安装 (先验证 Python 环境)
# RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制项目文件
COPY . .

# 5. 创建极简启动脚本 (只启动 FastAPI)
RUN echo '#!/bin/bash\n\
echo "🚀 Starting FastAPI ONLY (Debug Mode)..."\n\
export DISABLE_OLLAMA=true\n\
\n\
# 启动 FastAPI\n\
# --workers 1 限制进程数，节省内存\n\
uvicorn server:app --host 0.0.0.0 --port 7860 --workers 1\n\
' > start.sh && chmod +x start.sh

# 创建非 root 用户
RUN useradd -m -u 1000 user
RUN mkdir -p /app && chown -R user:user /app

# 切换用户
USER user

ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["/app/start.sh"]
