#!/bin/bash
# GPU部署脚本 - 一键部署自适应RAG系统到Linux RTX 4090环境

set -e  # 遇到错误立即退出

echo "🚀 开始部署自适应RAG系统到GPU环境..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否为root用户
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}请不要使用root用户运行此脚本${NC}"
   exit 1
fi

# 检查GPU
check_gpu() {
    echo "🔍 检查GPU环境..."
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ 发现NVIDIA GPU:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo -e "${RED}❌ 未发现NVIDIA GPU或驱动未安装${NC}"
        exit 1
    fi
}

# 检查CUDA
check_cuda() {
    echo "🔍 检查CUDA环境..."
    if command -v nvcc &> /dev/null; then
        echo -e "${GREEN}✅ CUDA版本:${NC}"
        nvcc --version | grep "release"
    else
        echo -e "${YELLOW}⚠️ CUDA未安装或未添加到PATH${NC}"
    fi
}

# 安装Docker
install_docker() {
    if ! command -v docker &> /dev/null; then
        echo "📦 安装Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        echo -e "${GREEN}✅ Docker安装完成${NC}"
    else
        echo -e "${GREEN}✅ Docker已安装${NC}"
    fi
}

# 安装NVIDIA Container Toolkit
install_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        echo "🐳 安装NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
        echo -e "${GREEN}✅ NVIDIA Container Toolkit安装完成${NC}"
    else
        echo -e "${GREEN}✅ NVIDIA Container Toolkit已配置${NC}"
    fi
}

# 创建环境配置
setup_env() {
    echo "⚙️ 配置环境变量..."
    if [ ! -f .env ]; then
        cp .env.example .env
        echo -e "${YELLOW}⚠️ 请编辑 .env 文件并设置您的API密钥${NC}"
        echo "   - TAVILY_API_KEY: 从 https://tavily.com/ 获取"
        read -p "按回车键继续..."
    fi
}

# 选择部署方式
choose_deployment() {
    echo "🎯 选择部署方式:"
    echo "1) Docker Compose部署 (推荐)"
    echo "2) 直接Python部署"
    read -p "请选择 (1-2): " choice
    
    case $choice in
        1)
            deploy_docker
            ;;
        2)
            deploy_python
            ;;
        *)
            echo -e "${RED}无效选择${NC}"
            exit 1
            ;;
    esac
}

# Docker部署
deploy_docker() {
    echo "🐳 使用Docker Compose部署..."
    
    # 安装docker-compose
    if ! command -v docker-compose &> /dev/null; then
        echo "安装Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    # 构建并启动服务
    echo "构建镜像..."
    docker-compose -f docker-compose.gpu.yml build
    
    echo "启动服务..."
    docker-compose -f docker-compose.gpu.yml up -d
    
    echo -e "${GREEN}✅ Docker部署完成!${NC}"
    echo "访问: http://localhost:8000"
    echo "监控: http://localhost:9445 (GPU监控)"
    echo "日志: docker-compose -f docker-compose.gpu.yml logs -f"
}

# Python直接部署
deploy_python() {
    echo "🐍 使用Python直接部署..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo "安装Python3..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    fi
    
    # 创建虚拟环境
    if [ ! -d "rag_env" ]; then
        echo "创建Python虚拟环境..."
        python3 -m venv rag_env
    fi
    
    # 激活虚拟环境并安装依赖
    source rag_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements_gpu.txt
    
    # 安装Ollama
    if ! command -v ollama &> /dev/null; then
        echo "安装Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
    # 启动Ollama服务
    echo "启动Ollama服务..."
    ollama serve &
    sleep 5
    
    # 下载模型
    echo "下载Mistral模型..."
    ollama pull mistral
    
    # 创建启动脚本
    cat > start_gpu.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
source rag_env/bin/activate
python main.py
EOF
    chmod +x start_gpu.sh
    
    echo -e "${GREEN}✅ Python部署完成!${NC}"
    echo "启动命令: ./start_gpu.sh"
}

# 验证部署
verify_deployment() {
    echo "🔍 验证部署..."
    sleep 10
    
    if curl -f http://localhost:8000/health 2>/dev/null; then
        echo -e "${GREEN}✅ 服务运行正常${NC}"
    else
        echo -e "${YELLOW}⚠️ 服务可能还在启动中，请稍后检查${NC}"
    fi
    
    # 显示GPU使用情况
    echo "📊 GPU状态:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
}

# 显示部署信息
show_info() {
    echo ""
    echo "🎉 部署完成!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 服务地址:"
    echo "   - 主服务: http://localhost:8000"
    echo "   - Ollama: http://localhost:11434"
    echo ""
    echo "🔧 常用命令:"
    echo "   - 查看日志: docker-compose -f docker-compose.gpu.yml logs -f"
    echo "   - 重启服务: docker-compose -f docker-compose.gpu.yml restart"
    echo "   - 停止服务: docker-compose -f docker-compose.gpu.yml down"
    echo "   - GPU监控: watch -n 1 nvidia-smi"
    echo ""
    echo "📚 文档位置:"
    echo "   - 部署指南: DEPLOYMENT_GUIDE.md"
    echo "   - 快速开始: QUICKSTART.md"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# 主函数
main() {
    echo "🤖 自适应RAG系统 GPU部署脚本"
    echo "适用于: Linux + RTX 4090"
    echo ""
    
    check_gpu
    check_cuda
    install_docker
    install_nvidia_docker
    setup_env
    choose_deployment
    verify_deployment
    show_info
}

# 运行主函数
main "$@"

reactive