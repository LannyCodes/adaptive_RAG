# Kaggle Ollama 持久化方案

## 🎯 问题

在 Kaggle 上每次会话结束后：
- ❌ Ollama 安装被删除（位于 `/usr/local/bin/`）
- ❌ 模型被删除（位于 `~/.ollama/`）
- ❌ 每次重启需要 10-15 分钟重新下载

## ✅ 解决方案

将 Ollama 和模型保存到 **Kaggle Dataset**（永久存储），后续加载只需 40-50 秒。

---

## 📋 完整流程

### 阶段 1: 首次备份（一次性，约 30-60 分钟）

#### 步骤 1: 在 Kaggle Notebook 中准备

```python
# Cell 1: 克隆项目
import os
os.chdir('/kaggle/working')
!git clone https://github.com/你的用户名/adaptive_RAG.git
%cd adaptive_RAG

# Cell 2: 安装 Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# Cell 3: 启动服务
import subprocess
import time
subprocess.Popen(['ollama', 'serve'])
time.sleep(15)

# Cell 4: 下载模型
!ollama pull mistral  # 或 phi, tinyllama

# Cell 5: 验证环境（可选但推荐）
exec(open('KAGGLE_CHECK_OLLAMA.py').read())
```

#### 步骤 2: 运行备份脚本

```python
# Cell 6: 执行备份
exec(open('KAGGLE_SAVE_OLLAMA.py').read())
```

**输出示例：**
```
====================================================================
💾 Kaggle Ollama 保存工具
====================================================================

📋 配置:
   模型: mistral
   输出目录: /kaggle/working/ollama_backup

📁 步骤 1/4: 创建备份目录...
   ✅ 目录创建成功

📦 步骤 2/4: 备份 Ollama 二进制文件...
   找到 Ollama: /usr/local/bin/ollama
   ✅ Ollama 二进制文件已备份

🤖 步骤 3/4: 备份 mistral 模型...
   找到模型目录: /root/.ollama
   模型总大小: 4.12 GB
   📦 创建压缩包（这可能需要几分钟）...
   正在压缩: /root/.ollama
   ✅ 压缩完成
      耗时: 180秒
      压缩包大小: 4.10 GB

📝 步骤 4/4: 生成说明文件...
   ✅ 说明文件已生成

====================================================================
✅ 备份完成！
====================================================================
```

#### 步骤 3: 下载备份文件

1. 在 Kaggle Notebook 右侧点击 **"Output"** 标签
2. 找到 `ollama_backup` 目录
3. 点击下载（约 4GB，需要 5-15 分钟取决于网络）

#### 步骤 4: 创建 Kaggle Dataset

1. **访问 Kaggle Datasets**
   - URL: https://www.kaggle.com/datasets
   - 点击 **"New Dataset"**

2. **上传文件**
   - 拖拽或选择：
     - `ollama` (约 50MB)
     - `ollama_models.tar.gz` (约 4GB)

3. **配置 Dataset**
   - **Title**: `ollama-mistral-backup`
   - **Visibility**: Private
   - 点击 **"Create"**

4. **等待上传**
   - 约 10-30 分钟（取决于网络）

---

### 阶段 2: 后续使用（每次约 1-2 分钟）

#### 步骤 1: 新建 Notebook

1. 添加 Dataset
   - 点击右侧 **"Add data"**
   - 选择 **"Your Datasets"**
   - 搜索 `ollama-mistral-backup`
   - 点击 **"Add"**

#### 步骤 2: 克隆项目并加载 Ollama

```python
# Cell 1: 克隆项目
import os
os.chdir('/kaggle/working')
!git clone https://github.com/你的用户名/adaptive_RAG.git
%cd adaptive_RAG

# Cell 2: 加载 Ollama（快速！）
exec(open('KAGGLE_LOAD_OLLAMA.py').read())
```

**输出示例：**
```
====================================================================
📦 从 Dataset 加载 Ollama（快速启动）
====================================================================

📋 配置:
   Dataset 路径: /kaggle/input/ollama-mistral-backup

🔍 步骤 1/5: 检查 Dataset...
   ✅ Dataset 存在

🔧 步骤 2/5: 安装 Ollama 二进制文件...
   ✅ Ollama 已安装到: /usr/local/bin/ollama
   📌 ollama version 0.1.x

📦 步骤 3/5: 解压模型文件...
   找到模型压缩包: 4.10 GB
   📦 开始解压（这可能需要 10-30 秒）...
   ✅ 解压完成（耗时: 25秒）
   📊 模型总大小: 4.12 GB

🚀 步骤 4/5: 启动 Ollama 服务...
   🔄 启动服务...
   ⏳ 等待服务启动（15秒）...
   ✅ Ollama 服务运行正常

✅ 步骤 5/5: 验证模型...
   可用模型:
   NAME            ID          SIZE    MODIFIED
   mistral:latest  xxx         4.1 GB  2 minutes ago

====================================================================
✅ Ollama 加载完成！
====================================================================

📊 加载总结:
   • Ollama 服务: ✅ 运行中
   • 模型: ✅ 已加载
   • 总耗时: < 1 分钟
```

#### 步骤 3: 开始使用

```python
# Cell 3: 运行你的 GraphRAG 项目
from document_processor import DocumentProcessor
from graph_indexer import GraphRAGIndexer

processor = DocumentProcessor()
vectorstore, retriever, doc_splits = processor.setup_knowledge_base(enable_graphrag=True)

indexer = GraphRAGIndexer(async_batch_size=8)
graph = indexer.index_documents(doc_splits)
```

---

## ⏱️ 时间对比

### 传统方式（每次启动）

| 步骤 | 时间 |
|------|------|
| 下载安装脚本 | 30秒 |
| 安装 Ollama | 1分钟 |
| 下载 Mistral | 5-10分钟 |
| 启动服务 | 15秒 |
| **总计** | **10-15分钟** ❌ |

### Dataset 方式（每次启动）

| 步骤 | 时间 |
|------|------|
| 复制二进制 | 2秒 |
| 解压模型 | 25秒 |
| 启动服务 | 15秒 |
| **总计** | **40-50秒** ✅ |

### 收益分析

- **首次投入**：30-60 分钟（一次性）
- **每次节省**：10+ 分钟
- **运行 5 次回本**：5 × 10 = 50 分钟 > 30 分钟
- **运行 10 次后**：累计节省 **100+ 分钟**！

---

## 🔍 验证脚本

在备份前建议运行验证脚本，确保环境正确：

```python
# 检查 Ollama 安装和模型位置
exec(open('KAGGLE_CHECK_OLLAMA.py').read())
```

**该脚本会检查：**
- ✅ Ollama 安装位置
- ✅ Ollama 服务状态
- ✅ 模型存储目录
- ✅ 已下载的模型列表
- ✅ 推荐备份方案

---

## 📊 不同模型的对比

| 模型 | 原始大小 | 压缩后 | 下载时间 | 解压时间 | 推荐度 |
|------|----------|--------|----------|----------|--------|
| qwen:0.5b | 350MB | ~300MB | 30秒 | 5秒 | ⭐⭐ 快但质量低 |
| tinyllama | 600MB | ~550MB | 1分钟 | 8秒 | ⭐⭐⭐ 快速测试 |
| phi | 1.6GB | ~1.5GB | 2-3分钟 | 15秒 | ⭐⭐⭐⭐ **推荐** |
| mistral | 4GB | ~4GB | 5-10分钟 | 25秒 | ⭐⭐⭐⭐⭐ 质量最好 |

**建议：**
- 开发测试：使用 `phi`（平衡）
- 快速验证：使用 `tinyllama`
- 生产环境：使用 `mistral`

---

## ❓ 常见问题

### Q1: 脚本是否正确？

**A**: 是的，已修正。脚本会：
- ✅ 自动查找 Ollama 安装位置（`/usr/local/bin/ollama`）
- ✅ 自动查找模型目录（`~/.ollama` 或 `/root/.ollama`）
- ✅ 完整备份整个 `.ollama` 目录（包括 models, manifests 等）
- ✅ 正确解压到 `~/.ollama`

### Q2: Dataset 名称可以改吗？

**A**: 可以！修改 `KAGGLE_LOAD_OLLAMA.py` 中的：
```python
DATASET_NAME = "你的Dataset名称"  # 第18行
```

### Q3: 上传失败怎么办？

**A**: 可能原因：
1. 网络不稳定 → 重试或使用稳定网络
2. 文件太大 → 使用更小的模型（如 phi）
3. 浏览器问题 → 尝试更换浏览器

### Q4: 可以备份多个模型吗？

**A**: 可以！在备份前下载多个模型：
```python
!ollama pull phi
!ollama pull tinyllama
!ollama pull mistral
# 然后运行备份脚本，会一起打包
```

### Q5: Dataset 有大小限制吗？

**A**: 是的
- 免费用户：每个 Dataset ≤ 20GB
- Kaggle 专业版：更大限额

---

## 🎯 最佳实践

### ✅ 推荐做法

1. **使用较小模型**：首选 `phi`（1.6GB）
2. **验证后再备份**：运行 `KAGGLE_CHECK_OLLAMA.py`
3. **Dataset 设为 Private**：避免占用公开配额
4. **定期更新**：模型有更新时重新备份

### ⚠️ 注意事项

1. **首次上传需要时间**：计划好 30-60 分钟
2. **网络稳定性**：确保上传期间网络稳定
3. **Dataset 管理**：定期清理不用的 Datasets
4. **备份验证**：首次加载后测试模型是否正常

---

## 📝 完整示例

### 首次使用（Kaggle Notebook）

```python
# ========== Cell 1: 环境准备 ==========
import os
os.chdir('/kaggle/working')
!git clone https://github.com/你的仓库/adaptive_RAG.git
%cd adaptive_RAG

# ========== Cell 2: 安装 Ollama ==========
!curl -fsSL https://ollama.com/install.sh | sh

# ========== Cell 3: 启动服务 ==========
import subprocess, time
subprocess.Popen(['ollama', 'serve'])
time.sleep(15)

# ========== Cell 4: 下载模型 ==========
!ollama pull phi  # 推荐使用 phi

# ========== Cell 5: 验证环境 ==========
exec(open('KAGGLE_CHECK_OLLAMA.py').read())

# ========== Cell 6: 备份 ==========
exec(open('KAGGLE_SAVE_OLLAMA.py').read())

# ========== 手动操作 ==========
# 1. 在右侧 Output 下载 ollama_backup
# 2. 访问 kaggle.com/datasets 创建 Dataset
# 3. 上传 ollama 和 ollama_models.tar.gz
```

### 后续使用（每次新 Notebook）

```python
# ========== Cell 1: 克隆项目 ==========
import os
os.chdir('/kaggle/working')
!git clone https://github.com/你的仓库/adaptive_RAG.git
%cd adaptive_RAG

# ========== Cell 2: 快速加载 ==========
# 注意：需要先在右侧 Add data 添加你的 Dataset
exec(open('KAGGLE_LOAD_OLLAMA.py').read())

# ========== Cell 3: 开始工作 ==========
from graph_indexer import GraphRAGIndexer
from document_processor import DocumentProcessor

processor = DocumentProcessor()
vectorstore, retriever, doc_splits = processor.setup_knowledge_base(enable_graphrag=True)

indexer = GraphRAGIndexer(async_batch_size=8)
graph = indexer.index_documents(doc_splits)

print("✅ 一切就绪！开始使用 GraphRAG！")
```

---

## 🎉 总结

### ✅ 优势
- **大幅节省时间**：每次启动从 10-15 分钟 → 40-50 秒
- **稳定可靠**：不受网络波动影响
- **一次投入**：首次 30-60 分钟，之后永久受益
- **易于使用**：两个脚本自动化全流程

### 📈 投资回报
- 首次投入：30-60 分钟
- 每次节省：10+ 分钟
- 5 次使用后回本
- 长期收益：**节省数小时**

### 💡 强烈推荐
如果你经常使用 Kaggle 运行这个项目，**强烈建议**使用这个方案！

---

**祝使用愉快！🚀**

有问题请参考：
- 验证脚本：`KAGGLE_CHECK_OLLAMA.py`
- 备份脚本：`KAGGLE_SAVE_OLLAMA.py`
- 加载脚本：`KAGGLE_LOAD_OLLAMA.py`
