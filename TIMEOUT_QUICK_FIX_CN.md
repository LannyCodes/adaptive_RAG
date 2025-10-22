# 超时问题快速修复指南

## 🚨 当前问题

您遇到了这个错误：
```
🔄 提取实体 (尝试 1/3)... ❌ 错误: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=60)
```

**原因**: 文档 #56 处理时间超过60秒，Ollama 没有在规定时间内返回结果。

## ⚡ 立即修复（3步搞定）

### 步骤 1: 重启 Ollama 服务

在 Colab 中运行：
```bash
!pkill -9 ollama
!sleep 2
!nohup ollama serve > /tmp/ollama.log 2>&1 &
!sleep 5
!curl http://localhost:11434/api/tags
```

### 步骤 2: 增加超时时间

在您的 Colab 笔记本中，修改初始化代码：

```python
# 找到 entity_extractor.py 的导入位置，修改为：
from entity_extractor import EntityExtractor

# 创建带更长超时的提取器
# 直接在 Python 中猴子补丁修复
import entity_extractor

# 保存原始初始化方法
_original_init = entity_extractor.EntityExtractor.__init__

# 创建新的初始化方法，默认使用更长的超时
def _new_init(self, timeout=180, max_retries=5):
    _original_init(self, timeout=timeout, max_retries=max_retries)

# 替换初始化方法
entity_extractor.EntityExtractor.__init__ = _new_init

print("✅ 已将超时时间增加到 180 秒（3分钟）")
```

### 步骤 3: 继续处理（跳过已完成的）

```python
# 从文档 #56 继续（索引 55）
processed_count = 55

remaining_docs = doc_splits[processed_count:]

graph = indexer.index_documents(
    documents=remaining_docs,
    batch_size=3,  # 减小批次大小
    save_path="/content/drive/MyDrive/knowledge_graph.pkl"
)
```

## 🎯 完整的 Colab 代码块

直接复制粘贴到 Colab 新的代码单元格：

```python
print("🔧 开始修复超时问题...")
print("="*60)

# ========== 第1步: 重启 Ollama ==========
print("\n1️⃣ 重启 Ollama 服务...")
!pkill -9 ollama
!sleep 2
!nohup ollama serve > /tmp/ollama.log 2>&1 &
!sleep 5

# 验证 Ollama 已启动
import requests
try:
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        print("✅ Ollama 服务运行正常")
    else:
        print("⚠️ Ollama 可能未正常启动")
except:
    print("❌ Ollama 服务未响应，请检查日志")

# ========== 第2步: 增加超时时间 ==========
print("\n2️⃣ 修改超时配置...")

import sys
sys.path.insert(0, '/content/drive/MyDrive/adaptive_RAG')

import entity_extractor

# 保存原始初始化
_original_init = entity_extractor.EntityExtractor.__init__

# 新的初始化方法：默认3分钟超时，5次重试
def _new_init(self, timeout=180, max_retries=5):
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser
    from config import LOCAL_LLM
    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        from langchain.prompts import PromptTemplate
    import time
    
    self.llm = ChatOllama(
        model=LOCAL_LLM, 
        format="json", 
        temperature=0,
        timeout=timeout
    )
    self.max_retries = max_retries
    
    # 实体提取提示模板
    self.entity_prompt = PromptTemplate(
        template="""你是一个专业的实体识别专家。从以下文本中提取所有重要的实体。
        
实体类型包括:
- PERSON: 人物、作者、研究者
- ORGANIZATION: 组织、机构、公司
- CONCEPT: 技术概念、算法、方法论
- TECHNOLOGY: 具体技术、工具、框架
- PAPER: 论文、出版物
- EVENT: 事件、会议

文本内容:
{text}

请以JSON格式返回，包含以下字段:
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型",
            "description": "简短描述"
        }}
    ]
}}

不要包含前言或解释，只返回JSON。
""",
        input_variables=["text"]
    )
    
    # 关系提取提示模板
    self.relation_prompt = PromptTemplate(
        template="""你是一个关系抽取专家。从文本中识别实体之间的关系。

已识别的实体:
{entities}

文本内容:
{text}

请识别实体之间的关系，以JSON格式返回:
{{
    "relations": [
        {{
            "source": "源实体名称",
            "target": "目标实体名称",
            "relation_type": "关系类型",
            "description": "关系描述"
        }}
    ]
}}

关系类型包括: AUTHOR_OF, USES, BASED_ON, RELATED_TO, PART_OF, APPLIES_TO, IMPROVES, CITES

不要包含前言或解释，只返回JSON。
""",
        input_variables=["text", "entities"]
    )
    
    self.entity_chain = self.entity_prompt | self.llm | JsonOutputParser()
    self.relation_chain = self.relation_prompt | self.llm | JsonOutputParser()

# 应用补丁
entity_extractor.EntityExtractor.__init__ = _new_init

print("✅ 超时时间已增加到 180 秒（3分钟）")
print("✅ 重试次数已增加到 5 次")

# ========== 第3步: 继续处理 ==========
print("\n3️⃣ 准备继续处理...")

# 重新导入模块以应用更改
import importlib
if 'graph_indexer' in sys.modules:
    importlib.reload(sys.modules['graph_indexer'])

from graph_indexer import GraphRAGIndexer

# 创建新的索引器
indexer = GraphRAGIndexer()

print("\n📋 当前状态:")
print(f"  • 总文档数: {len(doc_splits)}")
print(f"  • 已处理: 55 个文档（0-55）")
print(f"  • 待处理: {len(doc_splits) - 55} 个文档（56-{len(doc_splits)-1}）")

# 从文档 #56 继续
processed_count = 55
remaining_docs = doc_splits[processed_count:]

print("\n🚀 开始处理剩余文档...")
print("="*60)

graph = indexer.index_documents(
    documents=remaining_docs,
    batch_size=3,  # 减小批次大小以降低负载
    save_path="/content/drive/MyDrive/knowledge_graph_partial.pkl"
)

print("\n✅ 处理完成！")
```

## 📊 如果文档 #56 仍然超时

如果增加超时后，文档 #56 仍然失败，可能是该文档内容特别复杂。可以选择跳过它：

```python
# 方案A: 跳过文档 #56
print("跳过文档 #56，从 #57 继续...")
processed_count = 56  # 跳过 #56
remaining_docs = doc_splits[processed_count:]

graph = indexer.index_documents(
    documents=remaining_docs,
    batch_size=3,
    save_path="/content/drive/MyDrive/knowledge_graph_partial.pkl"
)
```

或者单独检查该文档：

```python
# 方案B: 检查文档 #56 的内容
problem_doc = doc_splits[55]  # 文档 #56（索引55）

print(f"文档 #56 信息:")
print(f"  长度: {len(problem_doc.page_content)} 字符")
print(f"  前500字符:")
print(f"  {problem_doc.page_content[:500]}")
print(f"\n  后500字符:")
print(f"  {problem_doc.page_content[-500:]}")

# 如果文档太长，可以考虑分割它
if len(problem_doc.page_content) > 3000:
    print("\n⚠️ 文档较长，可能需要更多处理时间或分割处理")
```

## 🔍 监控进度

修复后，您将看到更详细的输出：

```
⚙️  === 批次 19/20 (文档 56-58) ===

🔍 文档 #56: 开始提取...
   🔄 提取实体 (尝试 1/5)... ✅ 提取到 8 个实体
   🔄 提取关系 (尝试 1/5)... ✅ 提取到 5 个关系
📊 文档 #56 完成: 8 实体, 5 关系
```

## 📌 参数说明

| 参数 | 原值 | 新值 | 说明 |
|-----|------|------|------|
| `timeout` | 60秒 | 180秒 | 单次请求最大等待时间 |
| `max_retries` | 3次 | 5次 | 失败后重试次数 |
| `batch_size` | 10 | 3 | 每批次处理的文档数 |

## ⏱️ 预计时间

- **每个文档**: 10-180秒（取决于复杂度）
- **批次间隔**: 重试时有2-10秒等待
- **总时间**: 对于100个文档，预计20-60分钟

## 🆘 如果问题持续

### 检查 Ollama 日志
```bash
!tail -n 50 /tmp/ollama.log
```

### 检查系统资源
```python
# 检查 GPU 内存
!nvidia-smi

# 检查 RAM
import psutil
print(f"内存使用: {psutil.virtual_memory().percent}%")
```

### 使用更小的模型
如果 Mistral 太慢，可以在 `config.py` 中切换到更快的模型：
```python
LOCAL_LLM = "phi:latest"  # 更快但质量稍低
# 或
LOCAL_LLM = "llama2:7b"   # 平衡选择
```

## 📝 总结

**最可能的解决方案**:
1. ✅ 重启 Ollama（清理内存）
2. ✅ 增加超时到 180 秒
3. ✅ 减小批次大小到 3
4. ✅ 从断点继续处理

**紧急情况**:
- 如果某个文档持续失败 → 跳过它
- 如果 Ollama 崩溃 → 重启服务
- 如果内存不足 → 使用更小的模型

现在请运行上面的"完整 Colab 代码块"，应该就能解决问题了！ 🚀
