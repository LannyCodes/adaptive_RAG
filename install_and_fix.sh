#!/bin/bash
# 安装依赖和修复常见问题的脚本

echo "============================================"
echo "🚀 Adaptive RAG 安装和修复脚本"
echo "============================================"

# 1. 安装 Hugging Face CLI（如果未安装）
echo "📦 检查 Hugging Face CLI..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚙️ 安装 huggingface_hub..."
    pip install huggingface_hub
else
    echo "✅ Hugging Face CLI 已安装"
fi

# 2. 安装所有依赖
echo ""
echo "📦 安装项目依赖..."
pip install -r requirements.txt

# 3. 安装 rank_bm25（如果未安装）
echo ""
echo "📦 检查 rank_bm25..."
python -c "import rank_bm25" 2>/dev/null || {
    echo "⚙️ 安装 rank_bm25..."
    pip install rank-bm25
}

# 4. 运行 Hugging Face 配置脚本
echo ""
echo "🔧 配置 Hugging Face 访问..."
python configure_huggingface.py

# 5. 验证安装
echo ""
echo "🔍 验证安装结果..."

# 检查 rank_bm25
echo "检查 rank_bm25..."
python -c "import rank_bm25; print('✅ rank_bm25 可用')" || echo "❌ rank_bm25 安装失败"

# 检查 Hugging Face 登录
echo "检查 Hugging Face 登录状态..."
if huggingface-cli whoami &>/dev/null; then
    echo "✅ Hugging Face 已登录"
else
    echo "⚠️ Hugging Face 未登录，可能无法访问受限模型"
fi

# 检查 Vectara 模型访问
echo "检查 Vectara 模型访问..."
python -c "
try:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained('vectara/hallucination_evaluation_model')
    print('✅ Vectara 模型可访问')
except:
    print('❌ Vectara 模型不可访问，将使用 NLI 方法')
"

echo ""
echo "============================================"
echo "🎉 安装和配置完成!"
echo "============================================"
echo ""
echo "💡 现在可以运行: python setup_and_run.py"