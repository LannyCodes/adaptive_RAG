# 📦 Google Colab GPU测试文件总结

## ✅ 已创建的文件

| 文件名 | 类型 | 用途 | 推荐度 |
|--------|------|------|--------|
| **colab_gpu_demo.ipynb** | Jupyter Notebook | 完整的交互式GPU测试 | ⭐⭐⭐⭐⭐ |
| **colab_quick_test.py** | Python脚本 | 一键快速GPU测试 | ⭐⭐⭐⭐⭐ |
| **colab_gpu_test.py** | Python脚本 | 模块化GPU测试工具 | ⭐⭐⭐⭐ |
| **COLAB_GPU_GUIDE.md** | 文档 | 详细使用指南 | ⭐⭐⭐⭐⭐ |

---

## 🚀 快速开始（3种方式）

### 方式1: Notebook交互式测试 ⭐推荐

**适合**: 第一次使用，想要详细了解每个步骤

```bash
# 步骤1: 上传文件
上传 colab_gpu_demo.ipynb 到 Google Colab

# 步骤2: 启用GPU
运行时 → 更改运行时类型 → GPU

# 步骤3: 运行
运行时 → 全部运行
```

**优势**:
- ✅ 可视化输出
- ✅ 分步执行，易于理解
- ✅ 支持实时修改
- ✅ Markdown说明清晰

---

### 方式2: 快速一键测试 ⭐最快

**适合**: 快速验证GPU性能

```python
# 在Colab新建笔记本，运行以下代码：

# 1. 启用GPU (运行时 → GPU)

# 2. 复制并运行
!wget https://your-repo/colab_quick_test.py
!python colab_quick_test.py

# 或直接复制代码到单元格运行
```

**优势**:
- ✅ 零配置
- ✅ 自动安装依赖
- ✅ 5分钟完成全部测试
- ✅ 一次性输出完整报告

---

### 方式3: 模块化测试工具

**适合**: 开发者深度定制

```python
# 在Colab中
!wget https://your-repo/colab_gpu_test.py
!python colab_gpu_test.py
```

**优势**:
- ✅ 代码结构清晰
- ✅ 易于扩展
- ✅ 可集成到其他项目

---

## 📊 测试内容对比

| 测试项目 | Notebook | Quick Test | GPU Test |
|---------|----------|------------|----------|
| GPU环境检测 | ✅ | ✅ | ✅ |
| 矩阵运算测试 | ✅ | ✅ | ✅ |
| 文本嵌入测试 | ✅ | ✅ | ✅ |
| GraphRAG组件 | ✅ | ❌ | ❌ |
| 显存监控 | ✅ | ✅ | ✅ |
| 性能报告 | ✅ | ✅ | ✅ |
| 交互式说明 | ✅ | ❌ | ❌ |
| nvidia-smi | ✅ | ✅ | ✅ |

---

## 🎯 使用场景推荐

### 场景1: 首次测试GPU
**推荐**: `colab_gpu_demo.ipynb`
- 详细的说明文档
- 分步执行，便于学习
- 可视化效果好

### 场景2: 快速验证性能
**推荐**: `colab_quick_test.py`
- 一键运行
- 5分钟得到结果
- 完整性能报告

### 场景3: 集成到CI/CD
**推荐**: `colab_gpu_test.py`
- 模块化设计
- 易于自动化
- 返回标准化结果

### 场景4: 学习GPU优化
**推荐**: `COLAB_GPU_GUIDE.md` + `colab_gpu_demo.ipynb`
- 理论+实践
- 详细的性能分析
- 优化建议

---

## 📈 预期性能提升

### Google Colab T4 GPU (免费版)

| 任务 | CPU | GPU | 加速比 |
|------|-----|-----|--------|
| 矩阵运算 (5000x5000) | 8秒 | 0.3秒 | **25x** |
| 文本嵌入 (1000条) | 35秒 | 6秒 | **6x** |
| GraphRAG索引 (100文档) | 15分钟 | 4分钟 | **3.8x** |

### Google Colab A100 GPU (Pro版)

| 任务 | CPU | GPU | 加速比 |
|------|-----|-----|--------|
| 矩阵运算 | 8秒 | 0.2秒 | **40x** |
| 文本嵌入 | 35秒 | 3秒 | **12x** |
| GraphRAG索引 | 15分钟 | 2.5分钟 | **6x** |

---

## 🔧 完整GraphRAG部署流程

### 步骤1: GPU性能测试
```python
# 运行quick test验证GPU
!python colab_quick_test.py
```

### 步骤2: 上传项目文件
```python
# 方式A: 从Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/adaptive_RAG /content/
%cd /content/adaptive_RAG

# 方式B: 从GitHub
!git clone YOUR_REPO_URL
%cd adaptive_RAG
```

### 步骤3: 安装依赖
```python
!pip install -q -r requirements.txt
!pip install -q -r requirements_graphrag.txt
```

### 步骤4: 配置API密钥
```python
import os
from getpass import getpass
os.environ['TAVILY_API_KEY'] = getpass('TAVILY_API_KEY: ')
```

### 步骤5: 运行GraphRAG
```python
!python main_graphrag.py
```

### 步骤6: 下载结果
```python
from google.colab import files
files.download('data/knowledge_graph.json')
```

---

## 💡 优化技巧

### 1. 批处理大小
```python
# config.py
GRAPHRAG_BATCH_SIZE = 20  # GPU环境可增大
```

### 2. 嵌入模型选择
```python
# GPU环境使用更大模型
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### 3. 混合精度训练
```python
import torch
torch.set_float32_matmul_precision('medium')
```

### 4. 数据持久化
```python
# 定期保存到Drive
import shutil
shutil.copy(
    'data/knowledge_graph.json',
    '/content/drive/MyDrive/backup.json'
)
```

---

## ⚠️ 注意事项

### Colab免费版限制
- ⏰ 连续使用: 最多12小时
- 🔄 GPU配额: 每周有限
- ⏸️ 闲置超时: 90分钟

### 建议
- 💾 定期保存进度
- ⬇️ 及时下载结果
- 🔄 使用后台任务保持活跃

---

## 📚 文件使用优先级

### 新手用户
1. 📖 先阅读 `COLAB_GPU_GUIDE.md`
2. 🚀 运行 `colab_gpu_demo.ipynb`
3. ✅ 验证性能后部署完整项目

### 高级用户
1. ⚡ 直接运行 `colab_quick_test.py`
2. 📊 查看性能报告
3. 🔧 根据需求调整配置

### 开发者
1. 🔍 研究 `colab_gpu_test.py` 源码
2. 🛠️ 根据需求定制功能
3. 🔄 集成到自动化流程

---

## 🎯 关键性能指标

### 必须达到的基准
- ✅ GPU检测: CUDA可用
- ✅ 矩阵加速: >10x
- ✅ 嵌入加速: >5x
- ✅ 显存使用: <80%

### 如果低于基准
1. 检查GPU类型 (应该是T4或A100)
2. 重启运行时
3. 检查依赖版本

---

## 📞 获取帮助

### 常见问题
- 查看 `COLAB_GPU_GUIDE.md` 的FAQ部分

### 性能问题
- 运行 `colab_quick_test.py` 获取诊断报告

### 技术支持
- 提供测试报告输出
- 说明具体错误信息

---

## ✅ 总结

| 文件 | 何时使用 |
|------|---------|
| `colab_gpu_demo.ipynb` | 首次使用、学习、演示 |
| `colab_quick_test.py` | 快速验证、CI/CD、批量测试 |
| `colab_gpu_test.py` | 深度定制、集成开发 |
| `COLAB_GPU_GUIDE.md` | 参考文档、问题排查 |

**推荐流程**: 
1. 阅读 `COLAB_GPU_GUIDE.md` (5分钟)
2. 运行 `colab_quick_test.py` (5分钟)
3. 如果性能符合预期，部署完整GraphRAG项目

**预期结果**: 
- GPU可用 ✅
- 3-6倍整体加速 ✅
- 节省10+分钟时间 ✅

---

🚀 **立即开始**: 上传任一文件到 [Google Colab](https://colab.research.google.com/) 并启用GPU!
