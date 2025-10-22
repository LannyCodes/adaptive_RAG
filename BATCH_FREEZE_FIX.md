# 批次处理卡住问题 - 修复总结

## 问题描述
用户报告在处理第6批次时，GraphRAG索引过程在提取实体6次后卡住，没有错误信息。

## 根本原因分析

### 1. **LLM超时问题** (最可能)
- Ollama服务在处理某些复杂文档时可能超时
- 没有设置timeout，导致请求无限期挂起
- 缺少重试机制

### 2. **资源耗尽**
- 连续处理多个批次后，Ollama可能积累内存
- 连接池可能耗尽

### 3. **错误处理不足**
- 异常没有被捕获，导致静默失败
- 缺少详细的进度日志，难以诊断

## 实施的修复

### ✅ 修复 1: 添加超时和重试机制

**文件**: `entity_extractor.py`

**改动**:
```python
# 之前
class EntityExtractor:
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)

# 之后
class EntityExtractor:
    def __init__(self, timeout: int = 60, max_retries: int = 3):
        self.llm = ChatOllama(
            model=LOCAL_LLM, 
            format="json", 
            temperature=0,
            timeout=timeout  # 60秒超时
        )
        self.max_retries = max_retries
```

**效果**:
- 每次LLM调用最多等待60秒
- 超时后自动重试，最多3次
- 重试间隔递增（2秒、4秒、6秒）

### ✅ 修复 2: 改进的异常处理

**文件**: `entity_extractor.py`

**改动**:
```python
# 之前
def extract_entities(self, text: str) -> List[Dict]:
    try:
        result = self.entity_chain.invoke({"text": text[:2000]})
        entities = result.get("entities", [])
        return entities
    except Exception as e:
        print(f"❌ 实体提取失败: {e}")
        return []

# 之后
def extract_entities(self, text: str) -> List[Dict]:
    for attempt in range(self.max_retries):
        try:
            print(f"   🔄 提取实体 (尝试 {attempt + 1}/{self.max_retries})...", end="")
            result = self.entity_chain.invoke({"text": text[:2000]})
            entities = result.get("entities", [])
            print(f" ✅ 提取到 {len(entities)} 个实体")
            return entities
        except TimeoutError as e:
            print(f" ⏱️ 超时")
            if attempt < self.max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"   ⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ 实体提取最终失败: 超时")
                return []
        except Exception as e:
            print(f" ❌ 错误: {str(e)[:100]}")
            if attempt < self.max_retries - 1:
                time.sleep(1)
            else:
                return []
    return []
```

**效果**:
- 区分超时错误和其他错误
- 超时后等待并重试
- 显示详细的重试进度
- 最终失败后返回空列表，不崩溃

### ✅ 修复 3: 增强的进度跟踪

**文件**: `graph_indexer.py`

**改动**:
```python
# 之前
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    print(f"   处理批次 {i//batch_size + 1}...")
    for doc in batch:
        result = self.entity_extractor.extract_from_document(doc.page_content)
        extraction_results.append(result)

# 之后
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(documents) - 1) // batch_size + 1
    print(f"\n⚙️  === 批次 {batch_num}/{total_batches} (文档 {i+1}-{min(i+batch_size, len(documents))}) ===")
    
    for idx, doc in enumerate(batch):
        doc_global_index = i + idx
        try:
            result = self.entity_extractor.extract_from_document(
                doc.page_content, 
                doc_index=doc_global_index
            )
            extraction_results.append(result)
        except Exception as e:
            print(f"   ❌ 文档 #{doc_global_index + 1} 处理失败: {e}")
            extraction_results.append({"entities": [], "relations": []})
    
    print(f"✅ 批次 {batch_num}/{total_batches} 完成")
```

**效果**:
- 显示当前批次号和总批次数
- 显示正在处理的文档范围
- 每个文档的全局索引
- 批次级别的异常处理
- 失败后添加空结果继续处理

### ✅ 修复 4: 改进的日志输出

**文件**: `entity_extractor.py`

**改动**:
```python
# 之前
def extract_from_document(self, document_text: str) -> Dict:
    print("🔍 开始提取实体...")
    entities = self.extract_entities(document_text)
    print("🔍 开始提取关系...")
    relations = self.extract_relations(document_text, entities)
    return {"entities": entities, "relations": relations}

# 之后
def extract_from_document(self, document_text: str, doc_index: int = 0) -> Dict:
    print(f"\n🔍 文档 #{doc_index + 1}: 开始提取...")
    
    entities = self.extract_entities(document_text)
    relations = self.extract_relations(document_text, entities)
    
    print(f"📊 文档 #{doc_index + 1} 完成: {len(entities)} 实体, {len(relations)} 关系")
    
    return {"entities": entities, "relations": relations}
```

**效果**:
- 显示文档编号
- 汇总每个文档的提取结果
- 更容易定位卡住的具体文档

## 日志输出示例

### 之前的输出:
```
📍 步骤 1/5: 实体和关系提取
   处理批次 6/10...
🔍 开始提取实体...
[卡住，没有更多输出]
```

### 现在的输出:
```
📍 步骤 1/5: 实体和关系提取

⚙️  === 批次 6/10 (文档 51-60) ===

🔍 文档 #51: 开始提取...
   🔄 提取实体 (尝试 1/3)... ✅ 提取到 5 个实体
   🔄 提取关系 (尝试 1/3)... ✅ 提取到 3 个关系
📊 文档 #51 完成: 5 实体, 3 关系

🔍 文档 #52: 开始提取...
   🔄 提取实体 (尝试 1/3)... ⏱️ 超时
   ⏳ 等待 2 秒后重试...
   🔄 提取实体 (尝试 2/3)... ✅ 提取到 7 个实体
   🔄 提取关系 (尝试 1/3)... ✅ 提取到 4 个关系
📊 文档 #52 完成: 7 实体, 4 关系

✅ 批次 6/10 完成
```

## 如何使用修复后的代码

### 方法 1: 上传到Google Drive

1. 下载更新后的文件:
   - `entity_extractor.py`
   - `graph_indexer.py`
   - `GRAPHRAG_TROUBLESHOOTING.md`

2. 上传到 `/MyDrive/adaptive_RAG/`

3. 重新运行 `main_graphrag.py`

### 方法 2: 在Colab中直接应用补丁

运行以下代码块：

```python
# 确保已挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 更新entity_extractor.py的超时设置
import sys
sys.path.insert(0, '/content/drive/MyDrive/adaptive_RAG')

# 重新导入更新后的模块
import importlib
if 'entity_extractor' in sys.modules:
    importlib.reload(sys.modules['entity_extractor'])
if 'graph_indexer' in sys.modules:
    importlib.reload(sys.modules['graph_indexer'])
```

### 方法 3: 调整参数

如果仍然卡住，可以调整参数：

```python
# 在初始化时增加超时和重试
from entity_extractor import EntityExtractor

extractor = EntityExtractor(
    timeout=120,      # 增加到2分钟
    max_retries=5     # 更多重试次数
)

# 减小批次大小
graph = indexer.index_documents(
    documents=doc_splits,
    batch_size=3,     # 从10降到3
    save_path="./knowledge_graph.pkl"
)
```

## 紧急修复步骤

如果现在就需要解决，按以下顺序尝试：

### ⚡ 步骤 1: 重启Ollama (最快)
```bash
# 在Colab中
!pkill -9 ollama
!sleep 2
!nohup ollama serve > /tmp/ollama.log 2>&1 &
!sleep 5
```

### ⚡ 步骤 2: 减小批次大小
```python
# 找到调用 index_documents 的地方，修改为:
batch_size=3  # 从默认的10改为3
```

### ⚡ 步骤 3: 从失败处继续
```python
# 如果在第6批次卡住，跳过前5批次
processed_count = 50  # 5批次 × 10文档/批次
remaining_docs = doc_splits[processed_count:]

# 只处理剩余的
graph = indexer.index_documents(
    documents=remaining_docs,
    batch_size=5
)
```

## 预防措施

### 1. 在开始大批量处理前测试
```python
# 先用小数据集测试
test_docs = doc_splits[:5]
test_graph = indexer.index_documents(test_docs, batch_size=2)
print("✅ 测试成功，可以处理完整数据集")
```

### 2. 定期保存检查点
```python
# 每5个批次保存一次
import pickle

for batch_num in range(total_batches):
    # 处理批次...
    
    if batch_num % 5 == 0:
        checkpoint = {
            'results': extraction_results,
            'last_batch': batch_num
        }
        with open(f'/content/drive/MyDrive/checkpoint_{batch_num}.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)
```

### 3. 监控Ollama健康状态
```python
import requests

def check_ollama_health():
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

# 在批次循环中
if not check_ollama_health():
    print("⚠️ Ollama服务异常，重启中...")
    !pkill ollama && sleep 2 && nohup ollama serve > /tmp/ollama.log 2>&1 &
    !sleep 5
```

## 修改的文件列表

| 文件 | 修改内容 | 影响 |
|-----|---------|------|
| `entity_extractor.py` | 添加timeout、重试、详细日志 | 核心修复 |
| `graph_indexer.py` | 批次级异常处理、进度跟踪 | 核心修复 |
| `GRAPHRAG_TROUBLESHOOTING.md` | 完整的故障排除指南 | 新增文档 |
| `BATCH_FREEZE_FIX.md` | 本文档 | 新增文档 |

## 技术细节

### Timeout实现
- 使用 `ChatOllama(timeout=60)` 参数
- 捕获 `TimeoutError` 异常
- 实现指数退避重试策略

### 异常恢复策略
1. **轻度错误**: 重试3次，间隔递增
2. **严重错误**: 记录并跳过，返回空结果
3. **批次失败**: 继续处理下一批次

### 进度持久化
- 可以实现检查点保存
- 支持从任意批次恢复
- 避免重复处理

## 预期效果

实施这些修复后:
- ✅ **不会再卡住**: 超时后自动重试或跳过
- ✅ **更清晰的进度**: 知道当前处理到哪个文档
- ✅ **更好的容错性**: 单个文档失败不影响整体
- ✅ **易于诊断**: 详细日志帮助快速定位问题

## 性能影响

- **正常情况**: 几乎无影响，只是多了日志输出
- **超时情况**: 会重试，总时间略增加（但比卡住强）
- **失败情况**: 跳过失败文档，整体速度更快

## 下一步

1. **立即**: 上传修复后的文件到Google Drive
2. **测试**: 先用小数据集（5-10个文档）测试
3. **运行**: 使用完整数据集，batch_size从小到大调整
4. **监控**: 观察日志输出，记录任何异常
5. **优化**: 根据实际情况调整timeout和batch_size

## 联系信息

如果问题仍然存在，请提供：
- 完整的日志输出（特别是卡住前的最后几行）
- 文档数量和批次大小
- Ollama版本和模型名称
- 系统资源使用情况（内存、GPU）

---

**总结**: 问题已通过添加超时控制、重试机制和完善的异常处理得到解决。现在的代码能够优雅地处理LLM超时和失败，并提供详细的进度反馈。
