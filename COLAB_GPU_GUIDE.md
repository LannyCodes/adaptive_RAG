# 🚀 Google Colab GPU 测试指南

## 📋 概述

我为您创建了两个文件用于在Google Colab上测试GPU性能：

1. **`colab_gpu_demo.ipynb`** - Jupyter Notebook版本（推荐）
2. **`colab_gpu_test.py`** - Python脚本版本

## 🎯 使用方法

### 方法1: 使用Notebook（推荐）

#### 步骤1: 上传到Colab

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 点击 `文件` → `上传笔记本`
3. 选择 `colab_gpu_demo.ipynb`

#### 步骤2: 启用GPU

1. 点击顶部菜单 `运行时` → `更改运行时类型`
2. 硬件加速器选择 `GPU`
3. GPU类型选择 `T4`（免费版）或 `A100`（Colab Pro）
4. 点击 `保存`

#### 步骤3: 运行测试

1. 点击 `运行时` → `全部运行`
2. 或者逐个单元格运行（Shift + Enter）

### 方法2: 使用Python脚本

#### 步骤1: 上传文件

1. 在Colab中创建新笔记本
2. 点击左侧文件夹图标
3. 上传 `colab_gpu_test.py`

#### 步骤2: 运行脚本

```python
# 在Colab单元格中运行
!python colab_gpu_test.py
```

## 📊 测试内容

### 1. GPU环境检测 ✅
- CUDA可用性检查
- GPU型号和显存信息
- nvidia-smi输出

### 2. 矩阵运算性能测试 ⚡
- CPU vs GPU 5000x5000矩阵乘法
- 预期加速比: **10-50x**

### 3. 文本嵌入性能测试 📝
- 使用sentence-transformers
- 1000个文本的嵌入生成
- CPU vs GPU对比
- 预期加速比: **5-10x**

### 4. GraphRAG组件测试 🔍
- 简化版知识图谱构建
- 实体和关系管理
- GPU加速的向量检索

### 5. 显存监控 💾
- 实时显存使用情况
- 内存分配统计

## 📈 预期结果

### Google Colab 免费版 (T4 GPU)

| 测试项目 | CPU时间 | GPU时间 | 加速比 |
|---------|---------|---------|--------|
| 矩阵运算 (5000x5000) | ~8-10秒 | ~0.3-0.5秒 | 20-30x |
| 文本嵌入 (1000文本) | ~30-40秒 | ~5-8秒 | 5-7x |
| GraphRAG索引 (100文档) | ~15分钟 | ~3-5分钟 | 3-5x |

### Google Colab Pro (A100 GPU)

| 测试项目 | CPU时间 | GPU时间 | 加速比 |
|---------|---------|---------|--------|
| 矩阵运算 | ~8秒 | ~0.2秒 | 40x |
| 文本嵌入 | ~35秒 | ~3秒 | 10-12x |
| GraphRAG索引 | ~15分钟 | ~2-3分钟 | 5-7x |

## 🔧 运行完整GraphRAG项目

如果GPU测试成功，可以在Colab上运行完整的GraphRAG项目：

### 步骤1: 上传项目文件

在Colab中创建新的单元格：

```python
# 方式1: 从Google Drive加载
from google.colab import drive
drive.mount('/content/drive')

# 复制项目文件
!cp -r /content/drive/MyDrive/adaptive_RAG /content/
%cd /content/adaptive_RAG
```

或者：

```python
# 方式2: 从GitHub克隆
!git clone YOUR_GITHUB_REPO_URL
%cd adaptive_RAG
```

### 步骤2: 安装依赖

```python
# 安装基础依赖
!pip install -q -r requirements.txt

# 安装GraphRAG依赖
!pip install -q -r requirements_graphrag.txt
```

### 步骤3: 配置API密钥

```python
import os
from getpass import getpass

# 安全输入API密钥
os.environ['TAVILY_API_KEY'] = getpass('输入 TAVILY_API_KEY: ')

# 验证
print("✅ API密钥已设置")
```

### 步骤4: 运行GraphRAG

```python
# 运行主程序
!python main_graphrag.py
```

### 步骤5: 下载结果

```python
# 下载构建好的知识图谱
from google.colab import files

# 下载图谱文件
files.download('data/knowledge_graph.json')

print("✅ 图谱已下载到本地")
```

## 💡 优化建议

### 1. 批处理大小优化

在 `config.py` 中调整：

```python
# GPU优化配置
GRAPHRAG_BATCH_SIZE = 20  # GPU可以处理更大批次
```

### 2. 使用GPU优化的模型

```python
# 使用更大的嵌入模型（GPU环境）
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### 3. 启用混合精度

```python
# 在entity_extractor.py中
import torch
torch.set_float32_matmul_precision('medium')  # 提升性能
```

## ⚠️ 注意事项

### Colab资源限制

1. **免费版限制**:
   - 连续使用时间: 最多12小时
   - GPU使用配额: 每周有限
   - 闲置超时: 90分钟自动断开

2. **建议**:
   - 定期保存进度到Google Drive
   - 使用`files.download()`下载重要结果
   - 避免长时间空闲

### 数据持久化

```python
# 定期保存到Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 保存图谱
import shutil
shutil.copy(
    'data/knowledge_graph.json',
    '/content/drive/MyDrive/graphrag_backup.json'
)
```

## 🐛 常见问题

### Q1: GPU连接失败

**A**: 检查运行时类型
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
# 如果False，重新设置运行时类型
```

### Q2: 内存不足

**A**: 减小批处理大小
```python
GRAPHRAG_BATCH_SIZE = 5  # 降低批次
```

### Q3: 会话超时

**A**: 使用Colab Pro或定期运行代码保持活跃
```python
# 在后台定期执行
import time
while True:
    print("Keep alive...")
    time.sleep(300)  # 每5分钟执行一次
```

## 📚 参考资源

- [Google Colab官方文档](https://colab.research.google.com/notebooks/intro.ipynb)
- [GPU加速指南](https://colab.research.google.com/notebooks/gpu.ipynb)
- [Colab Pro定价](https://colab.research.google.com/signup)

## 🎓 下一步学习

1. **理解GPU加速原理**: 查看测试代码中的性能对比
2. **优化GraphRAG参数**: 根据GPU性能调整配置
3. **扩展到生产环境**: 考虑使用AWS/GCP的GPU实例

---

## ✅ 总结

| 优势 | 说明 |
|------|------|
| 🆓 免费GPU | T4 GPU免费使用 |
| ⚡ 高性能 | 3-10倍加速 |
| 🔄 零配置 | 无需本地安装 |
| 💾 自动保存 | 集成Google Drive |
| 🌐 随时访问 | 仅需浏览器 |

**推荐**: 在本地CPU环境速度慢时，使用Colab GPU可以大幅提升GraphRAG索引构建速度！

---

**立即开始**: 上传 `colab_gpu_demo.ipynb` 到 [Google Colab](https://colab.research.google.com/) 并启用GPU! 🚀
