#!/bin/bash
# ============================================================
# 自适应 RAG 系统 -- 一键部署脚本
# 适用：阿里云 ECS Ubuntu 22.04 / Alibaba Cloud Linux 3
# 使用：bash deploy.sh
# ============================================================

set -e

echo "============================================================"
echo "  自适应 RAG 系统 -- 一键部署"
echo "============================================================"

# ---- 1. 检查环境 ----
echo ""
echo "[1/7] 检查系统环境..."

# 检查是否 root
if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] 请使用 root 用户运行此脚本"
    exit 1
fi

# 检查 .env
if [ ! -f .env ]; then
    echo "[ERROR] 未找到 .env 文件！"
    echo ""
    echo "请先创建配置文件："
    echo "  cp .env.production .env"
    echo "  vim .env   # 填入真实 API Key"
    echo ""
    exit 1
fi

# 检查必要的环境变量
for key in LLM_BACKEND TAVILY_API_KEY; do
    if ! grep -q "^${key}=" .env; then
        echo "[ERROR] .env 中缺少必要配置: ${key}"
        exit 1
    fi
done

# 检查通义千问 Key（如果选择了 tongyi 后端）
if grep -Eq '^LLM_BACKEND="?tongyi"?' .env; then
    if ! grep -Eq '^TONGYI_API_KEY=+"?sk-' .env; then
        echo "[WARN] LLM_BACKEND=tongyi 但 TONGYI_API_KEY 未配置或格式不对"
        echo "   请在 .env 中设置: TONGYI_API_KEY=sk-xxxxxxxx"
        echo "   （加不加引号均可，如：TONGYI_API_KEY=\"sk-xxx\" 或 TONGYI_API_KEY=sk-xxx）"
        exit 1
    fi
fi

echo "[OK] 环境检查通过"

# ---- 2. 安装 Docker ----
echo ""
echo "[2/7] 安装 Docker..."

if command -v docker &> /dev/null; then
    echo "[OK] Docker 已安装: $(docker --version)"
else
    echo "正在安装 Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "[OK] Docker 安装完成: $(docker --version)"
fi

# ---- 3. 安装 Docker Compose ----
echo ""
echo "[3/7] 安装 Docker Compose..."

if docker compose version &> /dev/null; then
    echo "[OK] Docker Compose 已安装: $(docker compose version)"
else
    echo "正在安装 Docker Compose 插件..."
    apt-get update -qq
    apt-get install -y -qq docker-compose-plugin 2>/dev/null || {
        # 备选方案：手动安装
        mkdir -p /usr/local/lib/docker/cli-plugins
        curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
            -o /usr/local/lib/docker/cli-plugins/docker-compose
        chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    }
    echo "[OK] Docker Compose 安装完成"
fi

# ---- 4. 配置 Swap（2核8G 建议加 2G swap 防止 OOM）----
echo ""
echo "[4/7] 检查 Swap 配置..."

SWAP_SIZE=$(free -m | awk '/Swap:/ {print $2}')
if [ "$SWAP_SIZE" -lt 1024 ]; then
    echo "正在创建 2G Swap 文件..."
    fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    # 持久化
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' >> /etc/fstab
    fi
    echo "[OK] Swap 创建完成 (2G)"
else
    echo "[OK] Swap 已存在: ${SWAP_SIZE}MB"
fi

# ---- 5. 构建镜像 ----
echo ""
echo "[5/7] 构建 Docker 镜像（首次构建约 5-10 分钟）..."

docker compose build --no-cache 2>&1 | tail -5
echo "[OK] 镜像构建完成"

# ---- 6. 停止旧容器并启动 ----
echo ""
echo "[6/7] 启动服务..."

# 停止旧容器（如果存在）
docker compose down 2>/dev/null || true

# 启动新容器
docker compose up -d

echo "[OK] 服务启动成功"

# ---- 7. 等待服务就绪 ----
echo ""
echo "[7/7] 等待服务就绪..."

MAX_WAIT=180  # 最长等 3 分钟
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
        PUBLIC_IP=$(curl -sf http://100.100.100.200/latest/meta-data/eipv4 2>/dev/null || hostname -I | awk '{print $1}')
        echo ""
        echo "============================================================"
        echo "  部署成功！"
        echo "============================================================"
        echo ""
        echo "  访问地址: http://${PUBLIC_IP}:8000"
        echo "  API 文档: http://${PUBLIC_IP}:8000/docs"
        echo ""
        echo "  常用命令："
        echo "    查看日志:   docker compose logs -f"
        echo "    重启服务:   docker compose restart"
        echo "    停止服务:   docker compose down"
        echo "    上传文档:   docker compose exec adaptive-rag python upload_and_index.py /app/data/uploads/"
        echo ""
        echo "============================================================"
        exit 0
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  等待中... (${WAITED}s / ${MAX_WAIT}s)"
done

echo ""
echo "[WARN] 服务启动超时，查看日志排查："
echo "  docker compose logs --tail=100"
echo ""
exit 1
