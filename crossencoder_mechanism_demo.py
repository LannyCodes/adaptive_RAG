"""
CrossEncoder 核心机制详解 Demo
通过具体代码演示"输入拼接"、"联合编码"、"注意力机制"等概念
"""

import numpy as np
from typing import List, Tuple


print("=" * 80)
print("CrossEncoder 核心机制详解 - 从零开始理解")
print("=" * 80)


# ============================================================================
# Part 1: 输入拼接 (Input Concatenation)
# ============================================================================
print("\n" + "=" * 80)
print("📝 Part 1: 输入拼接 (Input Concatenation)")
print("=" * 80)

query = "什么是人工智能？"
document = "人工智能是计算机科学的一个分支"

print(f"\n原始输入：")
print(f"  Query:    {query}")
print(f"  Document: {document}")

# CrossEncoder 的关键：将 Query 和 Document 拼接成一个序列
# 使用特殊标记分隔
concatenated_input = f"[CLS] {query} [SEP] {document} [SEP]"

print(f"\n拼接后的输入：")
print(f"  {concatenated_input}")
print(f"\n说明：")
print(f"  [CLS]  - 分类标记，用于提取整体表示")
print(f"  [SEP]  - 分隔符，标记 Query 和 Document 的边界")
print(f"  这样 Query 和 Document 在同一个序列中，可以互相'看到'对方")


# ============================================================================
# Part 2: 分词 (Tokenization)
# ============================================================================
print("\n" + "=" * 80)
print("🔤 Part 2: 分词 (Tokenization)")
print("=" * 80)

# 简化的分词过程（实际使用 BERT tokenizer）
def simple_tokenize(text: str) -> List[str]:
    """简化的分词函数"""
    # 实际 BERT 会将文本分解为 subword tokens
    # 这里简化为字符级别
    tokens = []
    for word in text.split():
        if word.startswith('[') and word.endswith(']'):
            tokens.append(word)  # 特殊标记
        else:
            # 简化：每个字作为一个 token
            tokens.extend(list(word))
    return tokens

tokens = simple_tokenize(concatenated_input)
print(f"\n分词结果（简化版）：")
print(f"  {tokens}")
print(f"\n每个 token 都会被转换为向量（embedding）")


# ============================================================================
# Part 3: 词向量化 (Embedding)
# ============================================================================
print("\n" + "=" * 80)
print("🎯 Part 3: 词向量化 (Embedding)")
print("=" * 80)

# 模拟：将每个 token 转换为向量
vocab_size = 100  # 词汇表大小（简化）
embedding_dim = 8  # 向量维度（实际 BERT 是 768 维）

# 创建一个简单的词嵌入矩阵
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.1

def get_embedding(token: str) -> np.ndarray:
    """获取 token 的向量表示（简化）"""
    # 实际使用预训练的 embedding
    # 这里用 hash 模拟
    idx = hash(token) % vocab_size
    return embedding_matrix[idx]

# 获取所有 token 的 embedding
token_embeddings = [get_embedding(token) for token in tokens[:10]]  # 只展示前10个

print(f"\n示例：前3个 token 的向量表示")
for i in range(min(3, len(tokens))):
    print(f"\n  Token: '{tokens[i]}'")
    print(f"  向量: {token_embeddings[i][:4]}... (只显示前4维)")
    print(f"  形状: {token_embeddings[i].shape}")


# ============================================================================
# Part 4: 自注意力机制 (Self-Attention) - 核心！
# ============================================================================
print("\n" + "=" * 80)
print("🌟 Part 4: 自注意力机制 (Self-Attention) - 核心机制！")
print("=" * 80)

print("\n自注意力让每个 token 都能'看到'所有其他 token")
print("这就是 CrossEncoder 能理解 Query-Document 关系的关键！")

# 简化的注意力计算
def simple_attention(query_vec: np.ndarray, 
                     key_vecs: List[np.ndarray], 
                     value_vecs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化的注意力机制
    
    Args:
        query_vec: 查询向量 (当前 token)
        key_vecs: 键向量列表 (所有 tokens)
        value_vecs: 值向量列表 (所有 tokens)
    
    Returns:
        output: 加权后的输出向量
        attention_weights: 注意力权重
    """
    # 1. 计算注意力分数 (Query 与每个 Key 的相似度)
    scores = []
    for key_vec in key_vecs:
        # 点积相似度
        score = np.dot(query_vec, key_vec)
        scores.append(score)
    
    # 2. Softmax 归一化 (将分数转换为概率分布)
    scores = np.array(scores)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # 3. 加权求和 (根据注意力权重聚合信息)
    output = np.zeros_like(value_vecs[0])
    for weight, value_vec in zip(attention_weights, value_vecs):
        output += weight * value_vec
    
    return output, attention_weights


# 演示：计算第一个 token 对所有 token 的注意力
print("\n演示：计算 '[CLS]' token 对所有 token 的注意力")
print("-" * 80)

if len(token_embeddings) > 0:
    current_token_vec = token_embeddings[0]  # [CLS] token
    
    # 计算注意力
    output, attention_weights = simple_attention(
        current_token_vec, 
        token_embeddings, 
        token_embeddings
    )
    
    print(f"\n注意力权重分布：")
    for i, (token, weight) in enumerate(zip(tokens[:len(attention_weights)], attention_weights)):
        bar = "█" * int(weight * 50)  # 可视化权重
        print(f"  Token {i:2d} '{token:8s}': {weight:.4f} {bar}")
    
    print(f"\n说明：")
    print(f"  - 权重越高，表示 [CLS] 对该 token 的关注度越高")
    print(f"  - 这些权重用于聚合信息，形成新的表示")
    print(f"  - 在真实 CrossEncoder 中，这个过程在多层中重复")


# ============================================================================
# Part 5: 注意力矩阵可视化
# ============================================================================
print("\n" + "=" * 80)
print("📊 Part 5: 注意力矩阵 - Query 与 Document 的交互")
print("=" * 80)

# 计算完整的注意力矩阵
def compute_attention_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """计算完整的注意力矩阵"""
    n = len(embeddings)
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        _, weights = simple_attention(embeddings[i], embeddings, embeddings)
        attention_matrix[i] = weights
    
    return attention_matrix

if len(token_embeddings) >= 5:
    attention_matrix = compute_attention_matrix(token_embeddings[:5])
    
    print("\n注意力矩阵（前5个tokens）：")
    print("     ", end="")
    for j, token in enumerate(tokens[:5]):
        print(f"{token[:4]:>6s}", end=" ")
    print()
    
    for i, token in enumerate(tokens[:5]):
        print(f"{token[:4]:>4s} ", end="")
        for j in range(5):
            # 用颜色深浅表示注意力强度
            val = attention_matrix[i, j]
            if val > 0.3:
                symbol = "█"
            elif val > 0.2:
                symbol = "▓"
            elif val > 0.1:
                symbol = "▒"
            else:
                symbol = "░"
            print(f"{symbol:>6s}", end=" ")
        print()
    
    print("\n说明：")
    print("  - 每一行表示一个 token 对所有 token 的注意力")
    print("  - █ 表示高注意力，░ 表示低注意力")
    print("  - Query 的 token 可以直接关注 Document 的 token！")
    print("  - 这就是'联合编码'的核心：Query 和 Document 互相感知")


# ============================================================================
# Part 6: 多层 Transformer 的作用
# ============================================================================
print("\n" + "=" * 80)
print("🏗️  Part 6: 多层 Transformer - 深层语义理解")
print("=" * 80)

print("\nCrossEncoder (如 BERT) 通常有 12 层 Transformer：")
print("""
Layer 1:  学习基础词汇关系
          └─ "人工" 和 "智能" 组合成 "人工智能"

Layer 2-4: 学习短语级语义
          └─ "人工智能" 与 "计算机科学" 的关系

Layer 5-8: 学习句子级语义
          └─ 理解 Query 在问"什么是"，Document 在解释"是..."

Layer 9-12: 学习深层推理
          └─ 判断 Document 是否回答了 Query
          └─ 输出最终相关性分数
""")


# ============================================================================
# Part 7: CrossEncoder vs Bi-Encoder 对比
# ============================================================================
print("\n" + "=" * 80)
print("⚖️  Part 7: CrossEncoder vs Bi-Encoder 对比")
print("=" * 80)

print("\n【Bi-Encoder (传统向量检索)】")
print("""
Query    → Encoder → Vector₁ (768维)
                        ↓
Document → Encoder → Vector₂ (768维)
                        ↓
                 Cosine Similarity
                        ↓
                    Score: 0.85

问题：
  ❌ Query 和 Document 分别编码，互不感知
  ❌ 无法捕捉细微的语义关系
  ❌ 例如："苹果手机" vs "iPhone" 可能匹配度低
""")

print("\n【CrossEncoder (深度重排)】")
print("""
[Query + Document] → Joint Encoder → Score: 8.26
         ↓
  Self-Attention 机制让 Query 的每个词
  都能看到 Document 的每个词
         ↓
  理解："苹果" = "Apple"
        "手机" = "iPhone"
        → 高度相关！

优势：
  ✅ 深层语义交互
  ✅ 理解同义词、上下位关系
  ✅ 理解否定、转折等复杂语义
  ✅ 准确率提升 15-20%
""")


# ============================================================================
# Part 8: 实际使用 CrossEncoder
# ============================================================================
print("\n" + "=" * 80)
print("💻 Part 8: 实际使用 CrossEncoder (真实代码)")
print("=" * 80)

print("\n使用 sentence-transformers 库：\n")
print("""
from sentence_transformers import CrossEncoder

# 1. 加载预训练模型
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2. 准备 Query-Document 对
pairs = [
    ["什么是人工智能？", "人工智能是计算机科学的一个分支"],
    ["什么是人工智能？", "今天天气很好"],
]

# 3. 批量打分（自动完成输入拼接、联合编码、注意力计算）
scores = model.predict(pairs)
# 输出: [8.26, -2.45]

# 4. 排序
ranked = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)
print(ranked[0])  # 最相关的文档
""")


# ============================================================================
# Part 9: 注意力机制的直观理解
# ============================================================================
print("\n" + "=" * 80)
print("🧠 Part 9: 注意力机制的直观理解")
print("=" * 80)

print("""
想象你在阅读一个问题和一篇文章：

问题："Python 是谁创建的？"
文章："Python 是由 Guido van Rossum 在 1991 年创建的编程语言"

【人类如何理解】
1. 看到问题中的"Python" → 在文章中找到对应的"Python" ✓
2. 看到问题中的"谁创建" → 在文章中找"创建"附近的人名 ✓
3. 发现"Guido van Rossum" → 这就是答案！ ✓

【CrossEncoder 的注意力机制】
1. "Python" token 关注文章中的 "Python" token (高权重)
2. "谁" token 关注文章中的人名 tokens (高权重)
3. "创建" token 关注文章中的 "创建" token (高权重)
4. 通过多层注意力，模型理解了问题和答案的对应关系
5. 输出高分数：9.2 分！

这就是为什么 CrossEncoder 比简单的向量余弦相似度准确得多！
""")


# ============================================================================
# Part 10: 总结
# ============================================================================
print("\n" + "=" * 80)
print("📚 Part 10: 核心概念总结")
print("=" * 80)

print("""
1️⃣  输入拼接 (Input Concatenation)
   ├─ 将 Query 和 Document 拼成一个序列
   └─ 格式: [CLS] Query [SEP] Document [SEP]

2️⃣  联合编码 (Joint Encoding)
   ├─ Query 和 Document 在同一个 Transformer 中处理
   └─ 不是分开编码再比较，而是一起编码！

3️⃣  自注意力机制 (Self-Attention)
   ├─ 每个 token 计算对所有其他 token 的注意力权重
   ├─ 高权重 = 强关联
   └─ Query 的词可以直接"看到"并"理解" Document 的词

4️⃣  多层堆叠 (Multi-layer)
   ├─ 12 层 Transformer 逐层提取更深层的语义
   ├─ 低层：词汇级
   ├─ 中层：短语级
   └─ 高层：句子级推理

5️⃣  输出分数 (Relevance Score)
   ├─ 最后一层的 [CLS] token 表示整体相关性
   └─ 通过全连接层输出一个分数（-10 到 10）

关键优势：
✅ 深层语义交互 - 不是简单的向量比较
✅ 理解复杂关系 - 同义词、否定、转折等
✅ 准确率更高 - 比 Bi-Encoder 提升 15-20%

代价：
⚠️  速度较慢 - 每个 Query-Doc 对都要重新计算
⚠️  不可预计算 - 无法提前为文档生成向量

最佳实践：
🎯 两阶段检索
   └─ 阶段1: Bi-Encoder 快速召回 (Top 100)
   └─ 阶段2: CrossEncoder 精准重排 (Top 10)
""")

print("\n" + "=" * 80)
print("✅ Demo 完成！现在你应该理解了 CrossEncoder 的工作原理")
print("=" * 80)
print("\n💡 提示：运行 test_crossencoder_reranking.py 查看实际效果！\n")
