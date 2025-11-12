#!/bin/bash

# 磁盘空间占用检测脚本
# 用于检测指定目录下所有文件和文件夹的大小，并按大小倒序排列

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
DIRECTORY="${1:-$HOME}"
TOP_N="${2:-100}"
SHOW_HIDDEN=true

# 帮助信息
show_help() {
    echo "用法: $0 [目录路径] [显示数量]"
    echo ""
    echo "参数:"
    echo "  目录路径    要扫描的目录 (默认: 当前用户主目录)"
    echo "  显示数量    显示前N个最大的文件 (默认: 100)"
    echo ""
    echo "选项:"
    echo "  -h, --help  显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                          # 扫描主目录，显示前100个最大文件"
    echo "  $0 ~ 50                     # 扫描主目录，显示前50个最大文件"
    echo "  $0 /Users/pro 200           # 扫描指定目录，显示前200个最大文件"
    echo "  $0 ~/Library 100            # 扫描Library目录"
    echo ""
    echo "常见大文件目录:"
    echo "  ~/Library                   # 应用程序缓存和数据"
    echo "  ~/Downloads                 # 下载文件"
    echo "  ~/Documents                 # 文档"
    echo "  ~/Desktop                   # 桌面"
    echo "  ~/Movies                    # 视频文件"
    echo "  ~/Pictures                  # 图片文件"
    echo ""
    echo "提示: 此脚本会扫描所有文件（包括隐藏文件），可能需要较长时间"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [ -z "$DIR_SET" ]; then
                DIRECTORY="$1"
                DIR_SET=true
            elif [ -z "$NUM_SET" ]; then
                TOP_N="$1"
                NUM_SET=true
            fi
            shift
            ;;
    esac
done

# 展开路径
DIRECTORY="${DIRECTORY/#\~/$HOME}"

# 检查目录是否存在
if [ ! -d "$DIRECTORY" ]; then
    echo -e "${RED}错误: 目录不存在: $DIRECTORY${NC}"
    exit 1
fi

echo -e "${GREEN}========================================================${NC}"
echo -e "${YELLOW}正在扫描目录: $DIRECTORY${NC}"
echo -e "${YELLOW}显示前 $TOP_N 个最大的文件（包括所有隐藏文件）${NC}"
echo -e "${GREEN}========================================================${NC}"
echo ""
echo -e "${BLUE}请稍候，正在扫描所有文件并计算大小...${NC}"
echo -e "${BLUE}这可能需要几分钟时间，请耐心等待...${NC}"
echo ""

# 临时文件
TEMP_FILE=$(mktemp)

# 扫描目录下的所有文件（包括隐藏文件）
# 使用 find 命令查找所有文件，然后计算每个文件的大小
find "$DIRECTORY" -type f -exec du -k {} + 2>/dev/null | sort -rn > "$TEMP_FILE"

# 打印表头
echo -e "${GREEN}========================================================${NC}"
printf "${YELLOW}%-15s %-s${NC}\n" "大小" "文件路径"
echo -e "${GREEN}========================================================${NC}"

# 计数器
count=0
total_size=0

# 读取并显示结果
while IFS=$'\t' read -r size path; do
    count=$((count + 1))
    total_size=$((total_size + size))
    
    # 转换大小为人类可读格式
    if [ $size -lt 1024 ]; then
        readable_size="${size} KB"
    elif [ $size -lt 1048576 ]; then
        readable_size=$(awk "BEGIN {printf \"%.2f MB\", $size/1024}")
    elif [ $size -lt 1073741824 ]; then
        readable_size=$(awk "BEGIN {printf \"%.2f GB\", $size/1048576}")
    else
        readable_size=$(awk "BEGIN {printf \"%.2f TB\", $size/1073741824}")
    fi
    
    # 打印结果
    printf "%-15s %s\n" "$readable_size" "$path"
    
    # 达到指定数量后退出
    if [ $count -ge $TOP_N ]; then
        break
    fi
done < "$TEMP_FILE"

# 打印总计
echo -e "${GREEN}========================================================${NC}"

# 转换总大小
if [ $total_size -lt 1024 ]; then
    total_readable="${total_size} KB"
elif [ $total_size -lt 1048576 ]; then
    total_readable=$(awk "BEGIN {printf \"%.2f MB\", $total_size/1024}")
elif [ $total_size -lt 1073741824 ]; then
    total_readable=$(awk "BEGIN {printf \"%.2f GB\", $total_size/1048576}")
else
    total_readable=$(awk "BEGIN {printf \"%.2f TB\", $total_size/1073741824}")
fi

echo -e "${YELLOW}总计 (前$count个文件): $total_readable${NC}"
echo -e "${GREEN}========================================================${NC}"

# 清理临时文件
rm -f "$TEMP_FILE"

echo ""
echo -e "${BLUE}提示: 使用 '$0 --help' 查看更多选项${NC}"
