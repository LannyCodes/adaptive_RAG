# GraphRAG 故障排除指南

## 问题：处理批次时卡住不动

### 症状
- 处理到第6个批次时，实体提取后程序卡住
- 没有错误信息，只是停止响应
- CPU/GPU使用率下降到0

### 根本原因

#### 1. **LLM超时问题** ⏱️
- **原因**: Ollama服务可能在处理复杂请求时超时
- **表现**: 请求挂起，没有响应也没有错误
- **解决方案**: 已添加timeout参数和重试机制

#### 2. **内存泄漏** 💾
- **原因**: 多次LLM调用后，Ollama可能积累内存
- **表现**: 响应变慢，最终完全停止
- **解决方案**: 
  ```bash
  # 重启Ollama服务
  pkill ollama
  ollama serve
  ```

#### 3. **连接池耗尽** 🔌
- **原因**: 太多并发请求，没有正确关闭连接
- **表现**: 新请求无法建立连接
- **解决方案**: 已添加重试延迟和异常处理

#### 4. **文档内容过长** 📄
- **原因**: 某些文档chunk可能超过LLM的上下文窗口
- **表现**: LLM静默失败
- **解决方案**: 已限制为2000字符

## 已实施的修复

### 1. 添加超时控制
```python
EntityExtractor(timeout=60, max_retries=3)
```
- 每次LLM调用最多60秒超时
- 失败后最多重试3次
- 重试间隔递增（2s, 4s, 6s）

### 2. 改进的错误处理
```python
try:
    result = extractor.extract_from_document(...)
except Exception as e:
    print(f"❌ 文档处理失败: {e}")
    extraction_results.append({"entities": [], "relations": []})
```
- 捕获所有异常
- 添加空结果而不是崩溃
- 继续处理下一个文档

### 3. 详细的进度日志
```
⚙️  === 批次 6/10 (文档 51-60) ===
🔍 文档 #51: 开始提取...
   🔄 提取实体 (尝试 1/3)... ✅ 提取到 5 个实体
   🔄 提取关系 (尝试 1/3)... ✅ 提取到 3 个关系
📊 文档 #51 完成: 5 实体, 3 关系
```

## 故障排除步骤

### 步骤 1: 检查Ollama服务状态
```bash
# 检查Ollama是否运行
ps aux | grep ollama

# 查看Ollama日志
tail -f ~/.ollama/logs/server.log

# 检查模型是否加载
ollama list
```

### 步骤 2: 检查系统资源
```bash
# 内存使用
free -h  # Linux
top      # 查看Ollama进程

# 在Colab中
!nvidia-smi  # GPU内存
!ps aux | grep ollama
```

### 步骤 3: 减小批次大小
```python
# 在 main_graphrag.py 或调用代码中
graph = indexer.index_documents(
    documents=doc_splits,
    batch_size=5,  # 从10降到5
    save_path="./knowledge_graph.pkl"
)
```

### 步骤 4: 测试单个文档
```python
# 测试提取器是否工作
from entity_extractor import EntityExtractor

extractor = EntityExtractor(timeout=30, max_retries=2)
result = extractor.extract_from_document(
    "测试文本...",
    doc_index=0
)
print(result)
```

### 步骤 5: 重启Ollama服务
```bash
# 完全重启Ollama
pkill -9 ollama
sleep 2
ollama serve &

# 等待服务启动
sleep 5

# 验证服务
curl http://localhost:11434/api/tags
```

## 性能优化建议

### 1. 调整超时参数
```python
# 对于较慢的机器或GPU
extractor = EntityExtractor(
    timeout=120,      # 增加到2分钟
    max_retries=5     # 更多重试次数
)
```

### 2. 使用更小的模型
```python
# 在 config.py 中
LOCAL_LLM = "mistral:7b"     # 默认
# 改为
LOCAL_LLM = "llama2:7b"      # 更快
# 或
LOCAL_LLM = "phi:latest"     # 最快，但质量较低
```

### 3. 增加批次间延迟
```python
# 在 graph_indexer.py 中，批次循环后添加
import time
for i in range(0, len(documents), batch_size):
    # ... 处理批次 ...
    time.sleep(2)  # 给Ollama 2秒恢复时间
```

### 4. 限制并发请求
```python
# 使用线程池控制并发
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(extract, doc) for doc in batch]
    results = [f.result() for f in futures]
```

## 在Google Colab中的特殊问题

### 问题: Colab会话超时
**解决方案**: 使用checkpoint保存进度
```python
# 每处理N个批次保存一次
if batch_num % 5 == 0:
    checkpoint = {
        'extraction_results': extraction_results,
        'processed_docs': i + len(batch)
    }
    import pickle
    with open(f'/content/drive/MyDrive/checkpoint_{batch_num}.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
```

### 问题: Ollama内存不足
**解决方案**: 在Colab中设置较小的上下文窗口
```python
# 启动Ollama时
!OLLAMA_NUM_GPU=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve > /tmp/ollama.log 2>&1 &
```

## 监控和调试

### 添加详细日志
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graphrag_debug.log'),
        logging.StreamHandler()
    ]
)
```

### 使用超时上下文管理器
```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# 使用
with timeout(60):
    result = extractor.extract_from_document(text)
```

## 常见错误信息

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Connection refused` | Ollama未运行 | `ollama serve` |
| `Timeout` | LLM响应慢 | 增加timeout参数 |
| `CUDA out of memory` | GPU内存不足 | 减小batch_size |
| `JSON decode error` | LLM输出格式错误 | 检查prompt模板 |
| 卡住无输出 | LLM挂起 | 重启Ollama，添加超时 |

## 快速修复清单

✅ **立即尝试这些步骤**:

1. **重启Ollama**
   ```bash
   pkill ollama && sleep 2 && ollama serve &
   ```

2. **减小批次大小**
   ```python
   batch_size=3  # 从10改为3
   ```

3. **增加超时时间**
   ```python
   EntityExtractor(timeout=120, max_retries=5)
   ```

4. **检查第6个文档**
   ```python
   # 单独处理第6个文档看是否有特殊问题
   doc_6 = documents[5]
   print(f"文档长度: {len(doc_6.page_content)}")
   print(f"前500字符: {doc_6.page_content[:500]}")
   ```

5. **使用检查点恢复**
   ```python
   # 从第6批次重新开始
   start_index = 50  # 跳过前5批次
   documents_remaining = documents[start_index:]
   ```

## 预防措施

1. **开始前验证环境**
   ```bash
   # 检查所有依赖
   python colab_install_deps.py
   
   # 测试Ollama
   ollama list
   ollama run mistral "Hello"
   ```

2. **使用小数据集测试**
   ```python
   # 先用5个文档测试
   test_docs = doc_splits[:5]
   graph = indexer.index_documents(test_docs, batch_size=2)
   ```

3. **监控资源使用**
   ```python
   import psutil
   print(f"内存使用: {psutil.virtual_memory().percent}%")
   ```

## 获取帮助

如果问题持续，请提供以下信息：

1. **系统信息**
   - OS版本
   - Python版本
   - Ollama版本
   - 可用内存/GPU

2. **错误日志**
   - 最后一条成功的输出
   - 完整的错误堆栈
   - Ollama日志 (`~/.ollama/logs/server.log`)

3. **复现步骤**
   - 文档数量
   - batch_size
   - 在哪个批次卡住

## 总结

**最可能的原因**: LLM调用超时或Ollama内存积累

**最快的解决方案**:
1. 重启Ollama服务
2. 减小batch_size到3-5
3. 使用更新后的带超时和重试的代码

现在的代码已经包含了所有这些保护措施，应该能够稳定运行！
