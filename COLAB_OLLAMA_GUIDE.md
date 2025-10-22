# GraphRAG Colab 完整运行指南

## 🎯 在Colab中运行Ollama的3种方法

### 方法1: 后台运行Ollama（推荐）⭐⭐⭐⭐⭐

在Colab中，您可以在单个单元格中后台启动Ollama，然后在另一个单元格运行GraphRAG。

#### 步骤1: 安装Ollama

```bash
# 单元格1: 安装Ollama
!curl -fsSL https://ollama.com/install.sh | sh
```

#### 步骤2: 后台启动Ollama服务

```python
# 单元格2: 后台启动Ollama
import subprocess
import time
import os

# 启动Ollama服务（后台）
ollama_process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    preexec_fn=os.setpgrp
)

print("⏳ 等待Ollama服务启动...")
time.sleep(5)

# 验证服务是否启动
!curl -s http://localhost:11434/api/tags | head -5

print(f"✅ Ollama服务已启动 (PID: {ollama_process.pid})")
```

#### 步骤3: 下载Mistral模型

```bash
# 单元格3: 下载模型
!ollama pull mistral
```

#### 步骤4: 安装Python依赖

```bash
# 单元格4: 安装依赖
!pip install -q langchain langchain-community langchain-core langgraph
!pip install -q chromadb sentence-transformers tiktoken
!pip install -q tavily-python python-dotenv networkx python-louvain
```

#### 步骤5: 配置API密钥

```python
# 单元格5: 配置环境
import os
from getpass import getpass

os.environ['TAVILY_API_KEY'] = getpass('输入TAVILY_API_KEY: ')
print("✅ API密钥已设置")
```

#### 步骤6: 运行GraphRAG

```python
# 单元格6: 运行GraphRAG
!python main_graphrag.py
```

#### 步骤7: 下载结果（可选）

```python
# 单元格7: 下载生成的图谱
from google.colab import files
files.download('data/knowledge_graph.json')
```

---

### 方法2: 使用tmux（高级）⭐⭐⭐⭐

```bash
# 单元格1: 安装tmux
!apt-get install -y tmux

# 单元格2: 在tmux会话中启动Ollama
!tmux new-session -d -s ollama 'ollama serve'

# 单元格3: 检查会话
!tmux ls

# 单元格4: 下载模型
!ollama pull mistral

# 单元格5: 运行GraphRAG
!python main_graphrag.py

# 单元格6: 停止tmux会话（清理）
!tmux kill-session -t ollama
```

---

### 方法3: 使用nohup（简单）⭐⭐⭐

```bash
# 单元格1: 后台启动Ollama
!nohup ollama serve > /tmp/ollama.log 2>&1 &

# 单元格2: 等待启动
import time
time.sleep(5)

# 单元格3: 检查日志
!tail -20 /tmp/ollama.log

# 单元格4: 下载模型
!ollama pull mistral

# 单元格5: 运行GraphRAG
!python main_graphrag.py

# 单元格6: 停止Ollama（清理）
!pkill -f 'ollama serve'
```

---

## 🚀 一键运行脚本（最简单）⭐⭐⭐⭐⭐

我已经为您创建了一个自动化脚本 `colab_setup_and_run.py`，它会：
1. ✅ 自动安装Ollama
2. ✅ 后台启动服务
3. ✅ 下载Mistral模型
4. ✅ 安装Python依赖
5. ✅ 配置环境变量
6. ✅ 运行GraphRAG

### 使用方法:

```bash
# 方法A: 直接运行脚本
!python colab_setup_and_run.py

# 方法B: 或者在Python中
import subprocess
subprocess.run(["python", "colab_setup_and_run.py"])
```

---

## 📊 完整的Colab Notebook示例

创建一个新的Colab笔记本，按顺序运行以下单元格：

### 单元格1: 环境准备

```python
# 检测GPU
import torch
print(f"GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
```

### 单元格2: 安装Ollama

```bash
%%bash
curl -fsSL https://ollama.com/install.sh | sh
echo "✅ Ollama安装完成"
```

### 单元格3: 后台启动Ollama

```python
import subprocess
import time
import os

print("🔄 启动Ollama服务...")

# 后台启动
process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    preexec_fn=os.setpgrp
)

# 等待启动
time.sleep(5)

# 验证
import requests
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=3)
    if response.status_code == 200:
        print(f"✅ Ollama服务运行正常 (PID: {process.pid})")
    else:
        print("⚠️  服务响应异常")
except:
    print("⚠️  无法连接服务，但进程已启动")

# 保存进程ID（重要！）
ollama_pid = process.pid
print(f"📝 保存的PID: {ollama_pid}")
```

### 单元格4: 下载模型

```bash
%%bash
echo "📥 下载Mistral模型..."
ollama pull mistral
echo "✅ 模型下载完成"
ollama list
```

### 单元格5: 上传项目文件

```python
# 方式A: 从Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制项目文件
!cp -r /content/drive/MyDrive/adaptive_RAG /content/
%cd /content/adaptive_RAG

# 方式B: 手动上传
# from google.colab import files
# uploaded = files.upload()
```

### 单元格6: 安装依赖

```bash
%%bash
pip install -q -r requirements.txt
pip install -q -r requirements_graphrag.txt
echo "✅ 依赖安装完成"
```

### 单元格7: 配置环境

```python
import os
from getpass import getpass

# 设置API密钥
if not os.path.exists('.env'):
    api_key = getpass('输入TAVILY_API_KEY: ')
    with open('.env', 'w') as f:
        f.write(f'TAVILY_API_KEY={api_key}\n')
    print("✅ .env文件已创建")
else:
    print("✅ 使用现有.env文件")
```

### 单元格8: 运行GraphRAG

```python
# 方式A: 直接运行
!python main_graphrag.py

# 方式B: 在Python中运行（可以捕获输出）
import subprocess

result = subprocess.run(
    ["python", "main_graphrag.py"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print("错误信息:")
    print(result.stderr)
```

### 单元格9: 下载结果

```python
# 下载生成的知识图谱
from google.colab import files

if os.path.exists('data/knowledge_graph.json'):
    files.download('data/knowledge_graph.json')
    print("✅ 文件已下载")
else:
    print("❌ 未找到图谱文件")

# 保存到Google Drive
import shutil
shutil.copy(
    'data/knowledge_graph.json',
    '/content/drive/MyDrive/graphrag_backup.json'
)
print("✅ 已备份到Google Drive")
```

### 单元格10: 清理（可选）

```python
# 停止Ollama服务
import os
import signal

try:
    os.kill(ollama_pid, signal.SIGTERM)
    print(f"✅ Ollama服务已停止 (PID: {ollama_pid})")
except:
    print("⚠️  停止服务失败，手动停止:")
    !pkill -f 'ollama serve'
```

---

## ⚠️ 常见问题

### Q1: Ollama服务启动后立即退出

**A**: 使用 `subprocess.Popen` 而不是 `subprocess.run`:

```python
# ❌ 错误方式
!ollama serve &  # 会立即退出

# ✅ 正确方式
import subprocess
process = subprocess.Popen(["ollama", "serve"])
```

### Q2: 连接被拒绝 (Connection refused)

**A**: 等待服务完全启动:

```python
import time
time.sleep(10)  # 增加等待时间
```

### Q3: 进程管理困难

**A**: 使用PID文件:

```python
# 保存PID
with open('/tmp/ollama.pid', 'w') as f:
    f.write(str(process.pid))

# 后续停止
with open('/tmp/ollama.pid', 'r') as f:
    pid = int(f.read())
os.kill(pid, signal.SIGTERM)
```

### Q4: 会话超时导致服务停止

**A**: 定期执行代码保持活跃:

```python
import time
while True:
    print("Keep alive...")
    time.sleep(300)  # 每5分钟
```

---

## 📚 推荐的完整流程

1. ✅ **运行自动化脚本** - `!python colab_setup_and_run.py`
2. ✅ **或按照Notebook示例** - 逐步执行每个单元格
3. ✅ **定期保存结果** - 到Google Drive

---

## 💡 最佳实践

1. **始终保存Ollama的PID**: 方便后续管理
2. **使用try-finally**: 确保清理后台进程
3. **定期备份**: 保存中间结果到Drive
4. **监控显存**: 避免OOM错误

```python
# 最佳实践示例
import subprocess
import atexit
import signal

# 启动Ollama
ollama_process = subprocess.Popen(["ollama", "serve"])

# 注册清理函数
def cleanup():
    try:
        ollama_process.terminate()
        print("✅ Ollama已停止")
    except:
        pass

atexit.register(cleanup)

# 运行您的代码
try:
    # ... 您的GraphRAG代码 ...
    pass
finally:
    cleanup()
```

---

**推荐**: 直接使用 `colab_setup_and_run.py` 脚本，它已经处理了所有这些细节！🚀
