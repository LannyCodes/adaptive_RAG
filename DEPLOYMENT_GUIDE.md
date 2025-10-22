# Linux GPU部署指南 (RTX 4090)

## 🚀 自适应RAG系统在Linux RTX 4090环境部署

本指南将详细介绍如何在配备NVIDIA RTX 4090 GPU的Linux服务器上部署自适应RAG系统。

## 📋 环境要求

### 硬件要求
- NVIDIA RTX 4090 GPU
- 至少16GB内存（推荐32GB）
- 50GB+可用磁盘空间
- Ubuntu 20.04+ / CentOS 8+ / RHEL 8+

### 软件要求
- Linux操作系统（推荐Ubuntu 22.04 LTS）
- NVIDIA驱动程序（推荐535+）
- CUDA 12.0+
- Docker（可选但推荐）
- Python 3.8-3.11

## 🔧 步骤1：系统准备

### 1.1 更新系统
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential python3-pip python3-venv
```

### 1.2 安装NVIDIA驱动和CUDA
```bash
# 检查GPU
lspci | grep -i nvidia

# 添加NVIDIA软件源
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# 安装NVIDIA驱动和CUDA
sudo apt-get install -y nvidia-driver-535 cuda-12-2

# 重启系统
sudo reboot
```

### 1.3 验证GPU安装
```bash
# 重启后验证
nvidia-smi
nvcc --version
```

## 🐳 步骤2：Docker环境配置（推荐）

### 2.1 安装Docker
```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2.2 安装NVIDIA Container Toolkit
```bash
# 添加NVIDIA Docker源
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2.3 创建Dockerfile
```dockerfile
# 创建 Dockerfile
cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 更新系统并安装Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制项目文件
COPY requirements.txt .
COPY *.py .
COPY *.md .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 暴露端口（如果需要Web界面）
EXPOSE 8000

# 启动命令
CMD ["python3", "main.py"]
EOF
```

## 🐍 步骤3：Python环境配置（直接部署）

### 3.1 创建Python虚拟环境
```bash
# 克隆项目
git clone <your-repo-url> adaptive_rag
cd adaptive_rag

# 创建虚拟环境
python3 -m venv rag_env
source rag_env/bin/activate

# 升级pip
pip install --upgrade pip
```

### 3.2 修改requirements.txt以支持GPU
需要更新requirements.txt以优化GPU使用：

```bash
# 创建GPU优化的requirements文件
cat > requirements_gpu.txt << 'EOF'
# 核心框架
langchain>=0.1.0
langgraph>=0.0.40
langchain-community>=0.0.20
langchain-core>=0.1.0

# LLM集成
langchain-ollama>=0.1.0

# 向量数据库和嵌入（GPU优化版本）
chromadb>=0.4.0
sentence-transformers>=2.2.0
torch>=2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
torchvision>=0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118
transformers>=4.30.0
accelerate>=0.20.0

# 文档处理
tiktoken>=0.5.0
beautifulsoup4>=4.12.0
requests>=2.31.0

# 网络搜索
tavily-python>=0.3.0

# 数据处理
numpy>=1.24.0,<2.0
pandas>=2.0.0

# 工具库
python-dotenv>=1.0.0
pydantic>=2.0.0
typing-extensions>=4.0.0

# GPU加速库
cupy-cuda12x>=12.0.0
faiss-gpu>=1.7.4
EOF
```

### 3.3 安装依赖
```bash
# 安装GPU优化依赖
pip install -r requirements_gpu.txt
```

## 🛠️ 步骤4：修改配置以优化GPU使用

### 4.1 更新document_processor.py以使用GPU
需要修改嵌入模型配置：

```python
# 在document_processor.py中修改
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'},  # 使用GPU
    encode_kwargs={'normalize_embeddings': True}
)
```

### 4.2 创建GPU优化配置
```python
# 创建 gpu_config.py
cat > gpu_config.py << 'EOF'
import torch
import os

# GPU配置
if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_COUNT = torch.cuda.device_count()
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"发现 {GPU_COUNT} 个GPU: {GPU_NAME}")
    
    # 设置CUDA优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 设置GPU内存管理
    torch.cuda.empty_cache()
else:
    DEVICE = "cpu"
    print("未发现GPU，使用CPU模式")

# 优化设置
EMBEDDING_BATCH_SIZE = 32 if DEVICE == "cuda" else 8
MAX_WORKERS = 4 if DEVICE == "cuda" else 2
EOF
```

## 🤖 步骤5：安装和配置Ollama

### 5.1 安装Ollama
```bash
# 下载并安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 或者使用Docker
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### 5.2 下载模型
```bash
# 下载Mistral模型
ollama pull mistral

# 或者下载更大的模型（如果GPU内存足够）
ollama pull llama2:13b
ollama pull codellama:34b
```

### 5.3 启动Ollama服务
```bash
# 启动Ollama服务
ollama serve &

# 验证服务
curl http://localhost:11434/api/version
```

## 🔐 步骤6：环境变量配置

### 6.1 创建.env文件
```bash
cat > .env << 'EOF'
# API密钥
TAVILY_API_KEY=your_tavily_api_key_here

# GPU配置
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090架构

# 模型配置
HF_HOME=/app/models
TRANSFORMERS_CACHE=/app/models

# 性能优化
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
EOF
```

## 🚀 步骤7：部署和启动

### 7.1 使用Docker部署
```bash
# 构建镜像
docker build -t adaptive-rag:gpu .

# 运行容器
docker run -d \
  --gpus all \
  --name adaptive-rag \
  --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  adaptive-rag:gpu
```

### 7.2 直接Python部署
```bash
# 激活虚拟环境
source rag_env/bin/activate

# 启动系统
python main.py
```

## 📊 步骤8：性能监控

### 8.1 创建监控脚本
```bash
cat > monitor_gpu.py << 'EOF'
import psutil
import GPUtil
import time

def monitor_system():
    while True:
        # GPU监控
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.load*100}% | 内存: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")
        
        # CPU和内存监控
        print(f"CPU: {psutil.cpu_percent()}% | 内存: {psutil.virtual_memory().percent}%")
        print("-" * 50)
        time.sleep(5)

if __name__ == "__main__":
    monitor_system()
EOF

pip install gputil
python monitor_gpu.py
```

## 🔧 步骤9：性能优化配置

### 9.1 创建优化启动脚本
```bash
cat > start_optimized.sh << 'EOF'
#!/bin/bash

# 设置GPU优化环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# 启动系统
source rag_env/bin/activate
python main.py
EOF

chmod +x start_optimized.sh
```

### 9.2 创建系统服务
```bash
# 创建systemd服务
sudo tee /etc/systemd/system/adaptive-rag.service > /dev/null << 'EOF'
[Unit]
Description=Adaptive RAG System
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/adaptive_rag
Environment=PATH=/path/to/adaptive_rag/rag_env/bin
ExecStart=/path/to/adaptive_rag/rag_env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable adaptive-rag
sudo systemctl start adaptive-rag
```

## 🐛 步骤10：故障排除

### 10.1 常见问题

1. **CUDA内存不足**
```bash
# 减少批处理大小
export EMBEDDING_BATCH_SIZE=16
# 或者启用梯度检查点
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

2. **Ollama连接问题**
```bash
# 检查Ollama状态
sudo systemctl status ollama
# 重启Ollama
sudo systemctl restart ollama
```

3. **权限问题**
```bash
# 添加用户到docker组
sudo usermod -aG docker $USER
# 重新登录
```

### 10.2 性能调优

```bash
# GPU性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 9251,2100

# 系统优化
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 📈 预期性能

在RTX 4090环境下的预期性能：
- **文档嵌入**: ~1000 documents/second
- **查询响应**: ~2-5 seconds per query
- **GPU利用率**: 60-80%
- **内存使用**: 8-12GB GPU memory

## 🎯 验证部署

```bash
# 测试GPU可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 测试系统
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是LLM智能体？"}'
```

这个部署指南提供了完整的Linux GPU环境配置，确保您的自适应RAG系统能够充分利用RTX 4090的计算能力。