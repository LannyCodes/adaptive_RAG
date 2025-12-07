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

# 复制启动脚本
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

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
