#!/usr/bin/env bash
# Kaggle vLLM 0.7.3 环境安装/修复脚本。
#
# 作用：
#   1. 创建独立 venv（默认 /kaggle/working/vllm-env）
#   2. 按 requirements-vllm.txt 安装 vLLM 和兼容版本依赖
#   3. 校验 transformers 是否还带 all_special_tokens_extended
#
# 用法（Kaggle cell）：
#   !bash /kaggle/working/adaptive_RAG/setup_vllm_env.sh
#
# 可选环境变量：
#   VLLM_VENV      venv 路径，默认 /kaggle/working/vllm-env
#   PYTHON_BIN     创建 venv 用的 Python，默认 python
#
# 如果 venv 已损坏，先删除目录再执行：
#   rm -rf /kaggle/working/vllm-env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VLLM_VENV:-/kaggle/working/vllm-env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQ_FILE="$SCRIPT_DIR/requirements-vllm.txt"

if [ ! -f "$REQ_FILE" ]; then
    echo "[setup] 找不到 $REQ_FILE" >&2
    exit 1
fi

check_env() {
    "$VENV_DIR/bin/python" - <<'PY'
import sys

import huggingface_hub
import tokenizers
import transformers
import vllm
from transformers import PreTrainedTokenizerBase

print(
    f"  vllm={vllm.__version__} "
    f"transformers={transformers.__version__} "
    f"tokenizers={tokenizers.__version__} "
    f"huggingface_hub={huggingface_hub.__version__}"
)

ok = (
    vllm.__version__.startswith("0.7.3")
    and transformers.__version__ == "4.48.3"
    and huggingface_hub.__version__.startswith("0.27.")
    and tokenizers.__version__.startswith("0.21.")
    and hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended")
)
sys.exit(0 if ok else 1)
PY
}

if [ -x "$VENV_DIR/bin/python" ] && check_env; then
    echo "[setup] vLLM 环境已就绪，跳过安装"
    exit 0
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[setup] 创建 venv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" --system-site-packages
else
    echo "[setup] 复用已有 venv: $VENV_DIR"
fi

echo "[setup] 升级 pip ..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "[setup] 安装/修复 vLLM 依赖 ..."
"$VENV_DIR/bin/python" -m pip install --no-cache-dir -r "$REQ_FILE"

echo "[setup] 校验版本 ..."
if ! check_env; then
    echo "[setup] 版本校验失败：请检查 $REQ_FILE 是否被其他依赖覆盖" >&2
    exit 1
fi

echo "[setup] pip check（仅提示，不阻断）..."
"$VENV_DIR/bin/python" -m pip check || true

echo "[setup] 完成。启动 vLLM："
echo "  python kaggle_vllm_dual_t4.py"
