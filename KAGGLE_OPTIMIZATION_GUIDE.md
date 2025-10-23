# Kaggle 环境优化指南 - 避免重复下载模型

## 🚨 问题

每次 Kaggle 会话重启后，Ollama 模型需要重新下载，Mistral 模型约 4GB，非常耗时。

## 💡 解决方案

### 方案 1: 使用更小的模型（推荐⭐⭐⭐⭐⭐）

**最佳选择**：不需要修改代码，只需在下载模型时选择更小的版本。

#### 可选模型对比

| 模型 | 大小 | 下载时间 | 质量 | 推荐场景 |
|-----|------|---------|------|---------|
| `mistral` | ~4GB | 5-10分钟 | ⭐⭐⭐⭐⭐ | 本地开发 |
| `phi` | ~1.6GB | 2-3分钟 | ⭐⭐⭐⭐ | **Kaggle推荐** |
| `tinyllama` | ~600MB | 1分钟 | ⭐⭐⭐ | 快速测试 |
| `qwen:0.5b` | ~350MB | 30秒 | ⭐⭐ | 极速测试 |

#### 使用方法

**选项 A**: 修改 `config.py`
```python
# 在 /kaggle/working/adaptive_RAG/config.py 中
LOCAL_LLM = "phi"  # 👈 改为 phi 或 tinyllama
```

**选项 B**: 运行时覆盖（不修改代码）
```python
# 在 Kaggle Notebook 中
import os
os.environ['LOCAL_LLM_OVERRIDE'] = 'phi'

# 然后正常导入
from config import LOCAL_LLM
# LOCAL_LLM 会自动使用 'phi'
```

**选项 C**: 直接在下载时指定
```python
# 下载更小的模型
!ollama pull phi  # 代替 mistral

# 或者
!ollama pull tinyllama
```

---

### 方案 2: 持久化模型到 Kaggle Dataset（中等推荐⭐⭐⭐）

将下载好的模型保存为 Dataset，下次会话直接加载。

#### 步骤

**会话 1（首次）：**
```python
import subprocess
import shutil
import os

# 1. 下载模型
subprocess.run(['ollama', 'pull', 'phi'])

# 2. 找到模型存储位置
# Ollama 模型通常存储在 ~/.ollama/models
ollama_models = os.path.expanduser('~/.ollama/models')

# 3. 复制到工作目录（会被保存为输出）
if os.path.exists(ollama_models):
    shutil.copytree(
        ollama_models,
        '/kaggle/working/ollama_models',
        dirs_exist_ok=True
    )
    print("✅ 模型已复制到 /kaggle/working/ollama_models")
    print("📌 会话结束后，将此目录保存为 Dataset")

# 4. 会话结束时：Save Version → Save as Dataset
#    命名为: ollama-models-cache
```

**会话 2（后续）：**
```python
import shutil
import os

# 1. 从 Dataset 恢复模型
models_cache = '/kaggle/input/ollama-models-cache'

if os.path.exists(models_cache):
    print("📥 恢复 Ollama 模型...")
    
    # 创建 Ollama 模型目录
    ollama_dir = os.path.expanduser('~/.ollama/models')
    os.makedirs(ollama_dir, exist_ok=True)
    
    # 复制模型文件
    shutil.copytree(
        models_cache,
        ollama_dir,
        dirs_exist_ok=True
    )
    
    print("✅ 模型已恢复，无需重新下载！")
else:
    print("⚠️ 未找到缓存，需要重新下载")
```

**注意**：此方法有局限性，因为 Ollama 的模型存储结构复杂，可能不完全兼容。

---

### 方案 3: 使用云端 LLM API（高级方案⭐⭐⭐⭐）

完全避免本地模型，使用云端 API。

#### 可选 API

1. **OpenAI API**（需付费）
2. **Anthropic Claude API**（需付费）
3. **Hugging Face Inference API**（免费，有限额）
4. **Together AI**（免费额度）

#### 代码修改示例

修改 `entity_extractor.py`:

```python
# 原代码
from langchain_community.chat_models import ChatOllama
self.llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)

# 改为使用 OpenAI API
from langchain_openai import ChatOpenAI
self.llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # 或 gpt-4
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 或使用 Hugging Face
from langchain_community.llms import HuggingFaceHub
self.llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)
```

**优点**：
- ✅ 无需下载模型
- ✅ 速度快（云端 GPU）
- ✅ 质量好（GPT-4 等高级模型）

**缺点**：
- ❌ 需要 API Key
- ❌ 可能产生费用
- ❌ 依赖网络

---

### 方案 4: 预构建 Docker 镜像（技术方案⭐⭐）

创建包含预下载模型的 Docker 镜像。

**步骤**：
1. 本地构建包含 Ollama + 模型的 Docker 镜像
2. 推送到 Docker Hub
3. 在 Kaggle 中拉取该镜像

**局限**：Kaggle 对 Docker 支持有限。

---

## 🎯 最佳实践推荐

### 推荐组合策略

**快速开发/测试**：
```python
# 使用 phi 模型（平衡速度和质量）
LOCAL_LLM = "phi"
```

**生产环境**：
```python
# 使用云端 API（速度快、质量高）
# 在 Kaggle Secrets 中设置 OPENAI_API_KEY
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")
```

**完全离线**：
```python
# 使用 tinyllama（最快下载）
LOCAL_LLM = "tinyllama"
```

---

## 📋 Kaggle 完整工作流程（优化版）

### 单元格 1: 初始化
```python
import os, subprocess, sys

os.chdir('/kaggle/working')
if not os.path.exists('adaptive_RAG'):
    subprocess.run(['git', 'clone', 'https://github.com/LannyCodes/adaptive_RAG.git'])

os.chdir('adaptive_RAG')

# 修改配置使用更小的模型
with open('config.py', 'r') as f:
    content = f.read()

content = content.replace('LOCAL_LLM = "mistral"', 'LOCAL_LLM = "phi"')

with open('config.py', 'w') as f:
    f.write(content)

print("✅ 已切换到 phi 模型")

sys.path.insert(0, '/kaggle/working/adaptive_RAG')
```

### 单元格 2: 安装 Ollama
```python
# 安装 Ollama
subprocess.run('curl -fsSL https://ollama.com/install.sh | sh', shell=True)

# 启动服务
subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(15)
```

### 单元格 3: 下载优化的模型
```python
import time

# 使用更小的模型
print("📥 下载 phi 模型（约1.6GB，2-3分钟）...")
subprocess.run(['ollama', 'pull', 'phi'])

print("✅ 模型下载完成")
```

### 单元格 4: 安装依赖并运行
```python
!pip install -r requirements_graphrag.txt -q

# 继续您的处理...
```

---

## 🔢 时间对比

| 场景 | Mistral | Phi | TinyLlama | 云端API |
|-----|---------|-----|-----------|---------|
| **首次下载** | 5-10分钟 | 2-3分钟 | 1分钟 | 0分钟 |
| **后续会话** | 5-10分钟 | 2-3分钟 | 1分钟 | 0分钟 |
| **每周总耗时**<br>（5次会话） | 25-50分钟 | 10-15分钟 | 5分钟 | 0分钟 |

---

## 💰 成本对比

| 方案 | 时间成本 | 金钱成本 | 质量 |
|-----|---------|---------|------|
| Mistral | 高 ❌ | 免费 ✅ | 高 ✅ |
| Phi | 中 ✅ | 免费 ✅ | 中高 ✅ |
| TinyLlama | 低 ✅ | 免费 ✅ | 中 ⚠️ |
| GPT-3.5 API | 极低 ✅ | 约$0.5-2/天 ⚠️ | 极高 ✅ |

---

## 🎁 快速配置脚本

将以下代码保存为 `KAGGLE_QUICK_START.py`：

```python
"""
Kaggle 快速启动脚本 - 自动使用优化配置
"""

import os
import subprocess
import sys
import time

print("🚀 Kaggle 快速启动（优化版）")
print("="*60)

# 1. 克隆项目
os.chdir('/kaggle/working')
if not os.path.exists('adaptive_RAG'):
    subprocess.run(['git', 'clone', 'https://github.com/LannyCodes/adaptive_RAG.git'])

os.chdir('adaptive_RAG')

# 2. 自动选择模型（根据配置）
USE_SMALL_MODEL = True  # 👈 改为 False 使用 Mistral

if USE_SMALL_MODEL:
    MODEL_NAME = "phi"
    print("✅ 使用优化模型: phi (1.6GB)")
else:
    MODEL_NAME = "mistral"
    print("✅ 使用标准模型: mistral (4GB)")

# 修改配置
with open('config.py', 'r') as f:
    content = f.read()

content = content.replace(
    'LOCAL_LLM = "mistral"',
    f'LOCAL_LLM = "{MODEL_NAME}"'
)

with open('config.py', 'w') as f:
    f.write(content)

# 3. 安装 Ollama
check = subprocess.run(['which', 'ollama'], capture_output=True)
if check.returncode != 0:
    print("📥 安装 Ollama...")
    subprocess.run('curl -fsSL https://ollama.com/install.sh | sh', shell=True)

# 4. 启动服务
subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(15)

# 5. 下载模型
print(f"📦 下载 {MODEL_NAME} 模型...")
subprocess.run(['ollama', 'pull', MODEL_NAME])

# 6. 安装依赖
print("📦 安装依赖...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_graphrag.txt', '-q'])

sys.path.insert(0, '/kaggle/working/adaptive_RAG')

print("\n" + "="*60)
print("✅ 环境准备完成！")
print("="*60)
print(f"\n📌 使用模型: {MODEL_NAME}")
print("📌 现在可以运行 GraphRAG 索引了")
```

---

## 总结

**最推荐的解决方案**：

1. ⭐⭐⭐⭐⭐ **使用 Phi 模型** - 平衡了速度和质量
2. ⭐⭐⭐⭐ **使用云端 API** - 适合生产环境
3. ⭐⭐⭐ **使用 TinyLlama** - 快速测试

**实际操作**：
- 只需将 `config.py` 中的 `LOCAL_LLM = "mistral"` 改为 `LOCAL_LLM = "phi"`
- 或在 Kaggle 中运行时自动替换（见快速启动脚本）

这样每次会话只需 2-3 分钟下载模型，而不是 5-10 分钟！
