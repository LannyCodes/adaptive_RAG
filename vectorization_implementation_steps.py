"""
文字转向量的具体实现步骤（代码层面）
展示 HuggingFace Embeddings 内部的实际操作
"""

print("=" * 80)
print("文字 → 向量的具体实现步骤")
print("=" * 80)

# ============================================================================
# 准备工作：模拟完整的向量化过程
# ============================================================================
print("\n" + "=" * 80)
print("🔧 准备：安装和导入需要的库")
print("=" * 80)

print("""
需要的库：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pip install transformers torch sentence-transformers

导入：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
""")


# ============================================================================
# Step 1: 加载模型和分词器
# ============================================================================
print("\n" + "=" * 80)
print("Step 1: 加载预训练模型和分词器")
print("=" * 80)

print("""
代码：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# 1. 加载分词器（负责文字 → ID）
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载模型（负责 ID → 向量）
model = AutoModel.from_pretrained(model_name)
model.eval()  # 设置为评估模式（不训练）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这两个东西做什么？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tokenizer（分词器）：
├─ 词汇表（vocabulary）：30,000+ 个词
│  例如：{"hello": 1, "world": 2, "machine": 3456, ...}
└─ 分词规则：如何切分文字

Model（模型）：
├─ Embedding 层：词汇表 → 初始向量
│  30,000 × 384 的矩阵（每个词对应一个 384 维向量）
├─ Transformer 层：6 层 BERT encoder
│  每层都有 Self-Attention + Feed Forward
└─ 参数量：22M（2200万个数字）
""")


# ============================================================================
# Step 2: 分词（Tokenization）
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: 分词 - 文字转为 Token IDs")
print("=" * 80)

print("""
输入文本：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
text = "Machine learning is a subset of artificial intelligence"

代码：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分词并转换为模型输入格式
encoded_input = tokenizer(
    text,
    padding=True,      # 填充到相同长度
    truncation=True,   # 超长截断
    max_length=512,    # 最大长度
    return_tensors='pt' # 返回 PyTorch tensor
)

print(encoded_input)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输出（encoded_input 包含）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  'input_ids': tensor([[
      101,     # [CLS] 特殊标记
      3698,    # "machine"
      4083,    # "learning"
      2003,    # "is"
      1037,    # "a"
      2042,    # "subset"
      1997,    # "of"
      7976,    # "artificial"
      4454,    # "intelligence"
      102      # [SEP] 特殊标记
  ]]),
  
  'attention_mask': tensor([[
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1  # 所有位置都有效（1表示关注，0表示忽略）
  ]])
}

详细解释：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

input_ids:
  每个数字对应一个词
  101 = [CLS]（句子开始标记）
  3698 = "machine"
  102 = [SEP]（句子结束标记）

attention_mask:
  告诉模型哪些位置是真实内容（1），哪些是填充（0）
  例如：[1, 1, 1, 0, 0] 表示前3个是真实词，后2个是填充
""")


# ============================================================================
# Step 3: 通过 Embedding 层获取初始向量
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Token IDs → 初始向量（Embedding 层）")
print("=" * 80)

print("""
这一步发生在模型内部：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

input_ids = [101, 3698, 4083, 2003, ...]
                ↓
        Embedding 表查询
                ↓

Embedding 表（简化）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
这是一个巨大的矩阵：30,522 × 384
（30,522 是词汇表大小，384 是向量维度）

  ID    |  第1维  第2维  第3维  ...  第384维
  ─────────────────────────────────────────
  101   |  0.12  -0.34   0.56  ...   0.78   ← [CLS]
  3698  |  0.23   0.45  -0.67  ...   0.89   ← "machine"
  4083  |  0.34  -0.56   0.78  ...  -0.90   ← "learning"
  2003  |  0.45   0.67  -0.89  ...   0.12   ← "is"
  ...

查询过程（类似字典查询）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID 101  → 查表 → [0.12, -0.34, 0.56, ..., 0.78]
ID 3698 → 查表 → [0.23, 0.45, -0.67, ..., 0.89]
ID 4083 → 查表 → [0.34, -0.56, 0.78, ..., -0.90]
...

结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
token_embeddings = [
    [0.12, -0.34, 0.56, ..., 0.78],  # [CLS]
    [0.23,  0.45, -0.67, ..., 0.89],  # "machine"
    [0.34, -0.56, 0.78, ..., -0.90],  # "learning"
    [0.45,  0.67, -0.89, ..., 0.12],  # "is"
    ...
]
形状：(10, 384)  # 10 个 tokens，每个 384 维

⚠️ 注意：这些还不是最终向量！需要通过 Transformer 处理！
""")


# ============================================================================
# Step 4: Transformer 处理（核心！）
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Transformer 处理 - Self-Attention（核心步骤）")
print("=" * 80)

print("""
代码：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with torch.no_grad():  # 不计算梯度（不训练）
    outputs = model(**encoded_input)

# outputs.last_hidden_state 就是 Transformer 的输出
token_embeddings = outputs.last_hidden_state
print(token_embeddings.shape)  # torch.Size([1, 10, 384])
                               #   批次  tokens  维度

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Transformer 内部做了什么？（6 层处理）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入：初始 embeddings
  [CLS]:     [0.12, -0.34, 0.56, ...]
  machine:   [0.23,  0.45, -0.67, ...]
  learning:  [0.34, -0.56, 0.78, ...]
  is:        [0.45,  0.67, -0.89, ...]
  ...

        ↓
┌──────────────────────────────────────────────────────────┐
│ Layer 1: Self-Attention                                  │
│ ──────────────────────────────────────────────────────── │
│                                                          │
│ 每个词"看"其他所有词，更新自己的向量：                    │
│                                                          │
│ "machine" 看到 "learning" → 理解这是一个词组              │
│ "learning" 看到 "artificial" → 理解与AI相关              │
│ "is" 看到前后词 → 理解是连接词                           │
│                                                          │
│ 更新后的向量包含了上下文信息                              │
└──────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────┐
│ Layer 2: Self-Attention                                  │
│ ──────────────────────────────────────────────────────── │
│ 继续深化理解...                                          │
│ "machine learning" 作为整体理解                          │
└──────────────────────────────────────────────────────────┘
        ↓
        ... (Layer 3, 4, 5) ...
        ↓
┌──────────────────────────────────────────────────────────┐
│ Layer 6: Self-Attention (最后一层)                       │
│ ──────────────────────────────────────────────────────── │
│ 每个词的向量现在包含了：                                  │
│ - 自己的语义                                             │
│ - 上下文信息                                             │
│ - 整个句子的含义                                         │
└──────────────────────────────────────────────────────────┘
        ↓
最终输出：
  [CLS]:     [0.234,  0.567, -0.890, ...]  # 更新后，包含全句信息
  machine:   [0.345, -0.678,  0.123, ...]  # 包含 "learning" 的信息
  learning:  [0.456,  0.789, -0.234, ...]  # 包含 "machine" 的信息
  ...

形状：(1, 10, 384)
      批次 tokens 维度
""")


# ============================================================================
# Step 5: Mean Pooling - 合并成一个句子向量
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Mean Pooling - 把多个词向量合并成一个句子向量")
print("=" * 80)

print("""
问题：现在有 10 个词，每个词一个向量
     如何变成 1 个句子向量？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

代码：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def mean_pooling(token_embeddings, attention_mask):
    \"\"\"
    对所有词向量求平均（考虑 attention_mask）
    \"\"\"
    # token_embeddings: (1, 10, 384)
    # attention_mask:   (1, 10)
    
    # 扩展 mask 的维度以匹配 embeddings
    # (1, 10) → (1, 10, 1) → (1, 10, 384)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    
    # 将 embeddings 与 mask 相乘（忽略填充部分）
    # 然后对所有词求和
    sum_embeddings = torch.sum(
        token_embeddings * input_mask_expanded, 
        dim=1  # 在 token 维度求和
    )
    
    # 计算有效 token 的数量
    sum_mask = torch.clamp(
        input_mask_expanded.sum(dim=1), 
        min=1e-9  # 避免除零
    )
    
    # 求平均
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings

# 使用
sentence_embedding = mean_pooling(
    token_embeddings, 
    encoded_input['attention_mask']
)

print(sentence_embedding.shape)  # torch.Size([1, 384])
                                 #   批次  维度

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

具体计算（简化示例）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

10 个词向量，每个 384 维：
  Token 1: [0.234,  0.567, -0.890, ..., 0.123]
  Token 2: [0.345, -0.678,  0.123, ..., 0.234]
  Token 3: [0.456,  0.789, -0.234, ..., 0.345]
  ...
  Token 10: [0.567, 0.890,  0.345, ..., 0.456]

求平均（对每一维分别平均）：
  第1维: (0.234 + 0.345 + 0.456 + ... + 0.567) / 10 = 0.412
  第2维: (0.567 - 0.678 + 0.789 + ... + 0.890) / 10 = 0.523
  第3维: (-0.890 + 0.123 - 0.234 + ... + 0.345) / 10 = -0.089
  ...
  第384维: (0.123 + 0.234 + 0.345 + ... + 0.456) / 10 = 0.289

句子向量 = [0.412, 0.523, -0.089, ..., 0.289]  (384维)
""")


# ============================================================================
# Step 6: 归一化（Normalization）
# ============================================================================
print("\n" + "=" * 80)
print("Step 6: L2 归一化 - 将向量长度缩放到 1")
print("=" * 80)

print("""
代码：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import torch.nn.functional as F

# L2 归一化
sentence_embedding = F.normalize(
    sentence_embedding, 
    p=2,    # L2 范数
    dim=1   # 在特征维度归一化
)

print(sentence_embedding.shape)  # torch.Size([1, 384])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

归一化的作用：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

归一化前的向量：
  v = [0.412, 0.523, -0.089, ..., 0.289]
  长度 ||v|| = √(0.412² + 0.523² + ... + 0.289²) = 2.37

归一化后的向量：
  v_norm = v / ||v||
  v_norm = [0.412/2.37, 0.523/2.37, ..., 0.289/2.37]
         = [0.174, 0.221, -0.038, ..., 0.122]
  长度 ||v_norm|| = 1  ✓

好处：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 所有向量长度相同（都是1），方便比较
✅ 余弦相似度 = 点积（计算更快）
   cos_sim(a, b) = a·b / (||a|| × ||b||)
   如果归一化: cos_sim(a, b) = a·b  ← 简化了！

✅ 消除向量长度的影响，只关注方向
""")


# ============================================================================
# Step 7: 最终输出
# ============================================================================
print("\n" + "=" * 80)
print("Step 7: 得到最终的句子向量")
print("=" * 80)

print("""
最终结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 转换为 numpy 数组（方便使用）
final_vector = sentence_embedding.cpu().numpy()[0]

print(final_vector.shape)  # (384,)
print(final_vector[:5])    # 前5个数字
# [0.174, 0.221, -0.038, 0.095, 0.312]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这就是最终的句子向量！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入: "Machine learning is a subset of artificial intelligence"
输出: [0.174, 0.221, -0.038, ..., 0.122]  (384 个数字)

这个向量包含了：
✅ 每个词的语义
✅ 词与词之间的关系
✅ 整个句子的含义

可以用来：
✅ 计算与其他句子的相似度
✅ 存入向量数据库
✅ 进行语义搜索
""")


# ============================================================================
# 完整代码汇总
# ============================================================================
print("\n" + "=" * 80)
print("📝 完整代码汇总（实际可运行）")
print("=" * 80)

print("""
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def text_to_vector(text):
    \"\"\"
    完整的文字转向量流程
    \"\"\"
    # Step 1: 加载模型
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Step 2: 分词
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Step 3 & 4: 通过模型（Embedding + Transformer）
    with torch.no_grad():
        outputs = model(**encoded_input)
        token_embeddings = outputs.last_hidden_state
    
    # Step 5: Mean Pooling
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    sentence_embedding = sum_embeddings / sum_mask
    
    # Step 6: 归一化
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    
    # Step 7: 转为 numpy
    return sentence_embedding.cpu().numpy()[0]


# 使用示例：
text = "Machine learning is a subset of artificial intelligence"
vector = text_to_vector(text)

print(f"输入: {text}")
print(f"向量维度: {vector.shape}")  # (384,)
print(f"前10个数字: {vector[:10]}")
print(f"向量长度: {np.linalg.norm(vector)}")  # 应该是 1.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

你的项目中的简化调用：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query(text)
# ↑ 这一行内部执行了上面所有 7 个步骤！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


# ============================================================================
# 关键步骤时间分析
# ============================================================================
print("\n" + "=" * 80)
print("⏱️  各步骤耗时分析")
print("=" * 80)

print("""
假设处理一个句子（10个词）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: 加载模型           0.5-2秒   (只需一次，可复用)
Step 2: 分词               <1毫秒    (非常快)
Step 3: Embedding 查表     <1毫秒    (矩阵索引)
Step 4: Transformer 处理   10-50毫秒 (6层计算，最慢)
Step 5: Mean Pooling       <1毫秒    (简单平均)
Step 6: 归一化             <1毫秒    (简单除法)
Step 7: 转换格式           <1毫秒

总耗时: 10-50毫秒 (GPU) 或 50-200毫秒 (CPU)

批量处理（20个句子）:
  单个处理: 20 × 50ms = 1000ms
  批量处理: 100ms ← 快10倍！(GPU并行)
  
这就是为什么要批量向量化！
""")


print("\n" + "=" * 80)
print("✅ 文字转向量的实现步骤讲解完毕！")
print("=" * 80)
print("""
核心步骤回顾：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

文字
  ↓ Step 1: 加载模型
Tokenizer + Model
  ↓ Step 2: 分词
Token IDs: [101, 3698, 4083, ...]
  ↓ Step 3: Embedding 查表
初始向量: [(10, 384)]
  ↓ Step 4: Transformer 处理
更新向量: [(10, 384)]  包含上下文信息
  ↓ Step 5: Mean Pooling
句子向量: [(1, 384)]
  ↓ Step 6: 归一化
归一化向量: [(1, 384)]  长度=1
  ↓ Step 7: 输出
最终向量: [0.174, 0.221, ..., 0.122]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

现在你知道了每一步的具体操作！
""")
print()
