# Kaggle Ollama Dataset 保存与加载指南

## 📋 目录
1. [问题背景](#问题背景)
2. [解决方案](#解决方案)
3. [详细步骤](#详细步骤)
4. [时间对比](#时间对比)
5. [故障排除](#故障排除)

---

## 问题背景

### Kaggle 存储特性

在 Kaggle 环境中：

| 目录 | 会话结束后 | 说明 |
|------|----------|------|
| `/usr/local` | ❌ 删除 | Ollama 安装位置 |
| `/kaggle/working` | ❌ 删除 | 工作目录 |
| `/home` | ❌ 删除 | 用户目录（模型存储位置） |
| `/kaggle/input` | ✅ 保留 | **Dataset 目录（永久）** |

### 当前问题

每次启动 Kaggle Notebook 都需要：
1. 下载 Ollama 安装脚本（~100MB）
2. 安装 Ollama
3. 下载模型（Mistral 4GB，需要 5-10 分钟）

**总耗时：约 10-15 分钟**

---

## 解决方案

### 核心思路

将 Ollama 和模型**一次性**保存到 Kaggle Dataset（永久存储），后续每次启动直接加载。

### 优势

- ✅ **只需上传一次**：将 Ollama 和模型保存为 Dataset
- ✅ **秒级加载**：后续启动只需 10-30 秒
- ✅ **节省时间**：每次节省 10+ 分钟
- ✅ **稳定可靠**：不受网络影响

---

## 详细步骤

### 阶段 1: 首次备份（一次性工作）

#### 1.1 在 Kaggle Notebook 中准备环境

```python
# 1. 克隆项目
!git clone https://github.com/你的用户名/adaptive_RAG.git
%cd adaptive_RAG

# 2. 安装 Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# 3. 启动 Ollama 服务（后台运行）
import subprocess
subprocess.Popen(['ollama', 'serve'])

# 4. 等待服务启动
import time
time.sleep(15)

# 5. 下载模型
!ollama pull mistral  # 或 phi, tinyllama 等
```

#### 1.2 运行备份脚本

```python
# 执行备份脚本
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
   找到模型目录: /root/.ollama/models
   模型总大小: 4.12 GB
   📦 创建压缩包（这可能需要几分钟）...
   ✅ 压缩完成
      耗时: 180秒
      压缩包大小: 4.10 GB

📝 步骤 4/4: 生成说明文件...
   ✅ 说明文件已生成

📊 备份内容:
   • ollama: 0.05 GB
   • ollama_models.tar.gz: 4.10 GB
   • README.md: 0.00 MB

====================================================================
✅ 备份完成！
====================================================================
```

#### 1.3 下载备份文件

在 Kaggle Notebook 右侧：
1. 点击 **Output** 标签
2. 找到 `ollama_backup` 目录
3. 点击下载按钮
4. 等待下载完成（约 4GB，取决于网络速度）

#### 1.4 创建 Kaggle Dataset

1. **访问 Kaggle Datasets 页面**
   - 打开：https://www.kaggle.com/datasets
   - 点击右上角 **"New Dataset"** 按钮

2. **上传文件**
   - 将下载的两个文件拖拽上传：
     - `ollama` (二进制文件，约 50MB)
     - `ollama_models.tar.gz` (模型压缩包，约 4GB)
   
3. **配置 Dataset**
   - **Title**: `ollama-mistral-backup`（或其他名称）
   - **Subtitle**: "Ollama with Mistral model for quick loading"
   - **Visibility**: **Private**（避免占用公开配额）
   - **License**: 选择合适的开源协议

4. **创建**
   - 点击 **"Create"** 按钮
   - 等待上传完成（4GB 大约需要 10-30 分钟，取决于网络）

---

### 阶段 2: 后续使用（每次启动）

#### 2.1 添加 Dataset 到 Notebook

在 Kaggle Notebook 中：
1. 点击右侧 **"Add data"** 按钮
2. 选择 **"Your Datasets"** 标签
3. 搜索并选择你的 `ollama-mistral-backup`
4. 点击 **"Add"** 按钮

#### 2.2 克隆项目

```python
# 在第一个单元格
import os
os.chdir('/kaggle/working')

!git clone https://github.com/你的用户名/adaptive_RAG.git
%cd adaptive_RAG
```

#### 2.3 加载 Ollama

```python
# 在第二个单元格
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

   Dataset 内容:
      • ollama: 0.05 GB
      • ollama_models.tar.gz: 4.10 GB

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
   NAME        ID              SIZE    MODIFIED
   mistral:latest  xxx         4.1 GB  2 minutes ago

====================================================================
✅ Ollama 加载完成！
====================================================================

📊 加载总结:
   • Ollama 服务: ✅ 运行中
   • 模型: ✅ 已加载
   • 总耗时: < 1 分钟

💡 对比:
   • 传统方式: 5-10 分钟（重新下载）
   • Dataset 方式: < 1 分钟（直接加载）
   • 节省时间: 约 90%！
```

#### 2.4 开始使用

```python
# 在第三个单元格
from document_processor import DocumentProcessor
from graph_indexer import GraphRAGIndexer

# 加载文档
processor = DocumentProcessor()
vectorstore, retriever, doc_splits = processor.setup_knowledge_base(enable_graphrag=True)

# 使用异步索引（速度快）
indexer = GraphRAGIndexer(async_batch_size=8)
graph = indexer.index_documents(doc_splits)
```

---

## 时间对比

### 传统方式（每次启动）

| 步骤 | 耗时 |
|------|------|
| 下载安装脚本 | 30秒 |
| 安装 Ollama | 1分钟 |
| 下载 Mistral 模型 | 5-10分钟 |
| 启动服务 | 15秒 |
| **总计** | **约 10-15 分钟** |

### Dataset 方式（每次启动）

| 步骤 | 耗时 |
|------|------|
| 加载 Dataset（自动） | 0秒 |
| 复制 Ollama 二进制 | 2秒 |
| 解压模型文件 | 20-30秒 |
| 启动服务 | 15秒 |
| **总计** | **约 40-50 秒** |

### 节省时间

- ✅ 首次上传：30 分钟（一次性工作）
- ✅ 后续每次：节省 **10+ 分钟**
- ✅ 运行 10 次后：累计节省 **100+ 分钟**

---

## 不同模型的大小对比

| 模型 | 原始大小 | 压缩后大小 | 下载时间 | 解压时间 |
|------|----------|-----------|----------|----------|
| qwen:0.5b | 350MB | ~300MB | 30秒 | 5秒 |
| tinyllama | 600MB | ~550MB | 1分钟 | 8秒 |
| phi | 1.6GB | ~1.5GB | 2-3分钟 | 15秒 |
| mistral | 4GB | ~4GB | 5-10分钟 | 25秒 |
| llama2:7b | 3.8GB | ~3.8GB | 5-10分钟 | 25秒 |

### 推荐选择

- **开发测试**：phi（平衡速度和质量）
- **快速验证**：tinyllama（最快）
- **最佳质量**：mistral（如果网络好）

---

## 故障排除

### 问题 1: Dataset 不存在

**症状：**
```
❌ Dataset 不存在: /kaggle/input/ollama-mistral-backup
```

**解决方案：**
1. 检查 Dataset 是否已添加到 Notebook
2. 检查 Dataset 名称是否正确
3. 修改 `KAGGLE_LOAD_OLLAMA.py` 中的 `DATASET_NAME`

### 问题 2: 上传 Dataset 失败

**症状：**
上传时卡住或失败

**解决方案：**
1. 检查网络连接
2. 使用更小的模型（如 phi 或 tinyllama）
3. 分多次尝试上传

### 问题 3: Ollama 无法运行

**症状：**
```
ollama: command not found
```

**解决方案：**
```bash
# 检查文件权限
chmod +x /usr/local/bin/ollama

# 验证安装
ollama --version
```

### 问题 4: 模型列表为空

**症状：**
```
ollama list
# 输出为空
```

**解决方案：**
```python
# 检查模型目录
import os
models_dir = os.path.expanduser("~/.ollama/models")
print(os.listdir(models_dir))

# 重新解压模型
# 重新运行 KAGGLE_LOAD_OLLAMA.py
```

### 问题 5: Dataset 超过大小限制

**症状：**
上传时提示 Dataset 过大

**解决方案：**
1. Kaggle 免费用户每个 Dataset 限制 20GB
2. 使用更小的模型
3. 或考虑升级为 Kaggle 专业版

---

## 高级优化

### 1. 多模型备份

如果想备份多个模型：

```bash
# 修改 KAGGLE_SAVE_OLLAMA.py
# 在下载模型步骤添加：
!ollama pull phi
!ollama pull tinyllama
!ollama pull mistral

# 然后运行备份脚本
# 所有模型会一起打包
```

### 2. 使用更快的压缩

```python
# 修改压缩命令（牺牲压缩率换取速度）
# 在 KAGGLE_SAVE_OLLAMA.py 中修改：
with tarfile.open(models_archive, 'w') as tar:  # 去掉 :gz
    tar.add(ollama_models_dir, arcname='models')
```

### 3. 增量更新

如果模型有更新：
1. 在 Kaggle Notebook 中下载新模型
2. 重新运行 `KAGGLE_SAVE_OLLAMA.py`
3. 下载新的压缩包
4. 更新 Dataset（覆盖旧文件）

---

## 完整工作流示例

### 第一次使用（约 45 分钟）

```python
# === Notebook Cell 1: 准备环境 ===
!git clone https://github.com/你的用户名/adaptive_RAG.git
%cd adaptive_RAG

# === Notebook Cell 2: 安装 Ollama ===
!curl -fsSL https://ollama.com/install.sh | sh

# === Notebook Cell 3: 启动服务 ===
import subprocess, time
subprocess.Popen(['ollama', 'serve'])
time.sleep(15)

# === Notebook Cell 4: 下载模型 ===
!ollama pull mistral  # 5-10 分钟

# === Notebook Cell 5: 备份 ===
exec(open('KAGGLE_SAVE_OLLAMA.py').read())  # 3-5 分钟

# === 然后手动：===
# 1. 下载 ollama_backup 目录（5-15 分钟）
# 2. 创建 Kaggle Dataset 上传（10-30 分钟）
```

### 后续使用（约 2 分钟）

```python
# === Notebook Cell 1: 克隆项目 ===
%cd /kaggle/working
!git clone https://github.com/你的用户名/adaptive_RAG.git
%cd adaptive_RAG

# === Notebook Cell 2: 加载 Ollama ===
exec(open('KAGGLE_LOAD_OLLAMA.py').read())  # 40-50 秒

# === Notebook Cell 3: 开始工作 ===
from document_processor import DocumentProcessor
from graph_indexer import GraphRAGIndexer

processor = DocumentProcessor()
vectorstore, retriever, doc_splits = processor.setup_knowledge_base(enable_graphrag=True)

indexer = GraphRAGIndexer(async_batch_size=8)
graph = indexer.index_documents(doc_splits)
```

---

## 总结

### ✅ 优势
- 一次性上传，永久使用
- 每次启动节省 10+ 分钟
- 不受网络波动影响
- 稳定可靠

### ⚠️ 注意事项
- 首次上传需要时间和网络
- Dataset 有大小限制（20GB）
- 需要手动管理 Dataset

### 💡 建议
- **强烈推荐**用于频繁使用 Kaggle 的场景
- 选择合适大小的模型（推荐 phi）
- 保持 Dataset 为 Private 避免占用配额

---

**祝使用愉快！🎉**
