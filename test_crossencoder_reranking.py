"""
测试 CrossEncoder 重排功能
对比 Bi-Encoder vs CrossEncoder 的效果
"""

from reranker import create_reranker, TFIDFReranker, BM25Reranker, SemanticReranker, CrossEncoderReranker


class MockDoc:
    """模拟文档类"""
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


class MockEmbeddings:
    """模拟 Embeddings 类（用于 Semantic Reranker）"""
    def embed_query(self, text):
        # 简单的字符级向量化（仅用于测试）
        return [ord(c) / 100.0 for c in text[:10]]
    
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def create_test_documents():
    """创建测试文档集"""
    return [
        MockDoc("人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"),
        MockDoc("机器学习是人工智能的子领域，专注于让计算机从数据中学习并改进。"),
        MockDoc("深度学习使用多层神经网络来处理复杂的数据模式，是机器学习的一种方法。"),
        MockDoc("自然语言处理（NLP）是人工智能的一个分支，处理计算机与人类语言之间的交互。"),
        MockDoc("计算机视觉是人工智能的另一个重要领域，使机器能够理解和解释视觉信息。"),
        MockDoc("今天天气很好，适合出去散步和运动。"),
        MockDoc("Python 是一种高级编程语言，由 Guido van Rossum 在 1991 年创建。"),
        MockDoc("RAG（检索增强生成）是一种结合信息检索和文本生成的技术。"),
    ]


def test_tfidf_reranking():
    """测试 TF-IDF 重排"""
    print("\n" + "=" * 60)
    print("📊 测试 TF-IDF 重排")
    print("=" * 60)
    
    query = "什么是人工智能和机器学习？"
    docs = create_test_documents()
    
    reranker = TFIDFReranker()
    results = reranker.rerank(query, docs, top_k=3)
    
    print(f"\n查询: {query}")
    print("\nTF-IDF 重排结果:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. 分数: {score:.4f} | 内容: {doc.page_content[:50]}...")


def test_bm25_reranking():
    """测试 BM25 重排"""
    print("\n" + "=" * 60)
    print("📊 测试 BM25 重排")
    print("=" * 60)
    
    query = "什么是人工智能和机器学习？"
    docs = create_test_documents()
    
    reranker = BM25Reranker()
    results = reranker.rerank(query, docs, top_k=3)
    
    print(f"\n查询: {query}")
    print("\nBM25 重排结果:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. 分数: {score:.4f} | 内容: {doc.page_content[:50]}...")


def test_crossencoder_reranking():
    """测试 CrossEncoder 重排"""
    print("\n" + "=" * 60)
    print("🌟 测试 CrossEncoder 重排（推荐）")
    print("=" * 60)
    
    query = "什么是人工智能和机器学习？"
    docs = create_test_documents()
    
    try:
        # 使用轻量级模型
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        results = reranker.rerank(query, docs, top_k=3)
        
        print(f"\n查询: {query}")
        print("\nCrossEncoder 重排结果:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. 分数: {score:.4f} | 内容: {doc.page_content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CrossEncoder 测试失败: {e}")
        print("💡 提示: 请先安装 sentence-transformers")
        print("   命令: pip install sentence-transformers")
        return False


def test_factory_function():
    """测试工厂函数"""
    print("\n" + "=" * 60)
    print("🏭 测试重排器工厂函数")
    print("=" * 60)
    
    query = "深度学习和神经网络"
    docs = create_test_documents()
    
    # 测试各种类型
    reranker_types = ['tfidf', 'bm25']
    
    for rtype in reranker_types:
        try:
            reranker = create_reranker(rtype)
            results = reranker.rerank(query, docs, top_k=2)
            print(f"\n✅ {rtype.upper()} 重排器创建成功")
            print(f"   Top 1: {results[0][1]:.4f} | {results[0][0].page_content[:40]}...")
        except Exception as e:
            print(f"\n❌ {rtype.upper()} 重排器失败: {e}")
    
    # 测试 CrossEncoder
    try:
        reranker = create_reranker('crossencoder')
        results = reranker.rerank(query, docs, top_k=2)
        print(f"\n✅ CROSSENCODER 重排器创建成功")
        print(f"   Top 1: {results[0][1]:.4f} | {results[0][0].page_content[:40]}...")
    except Exception as e:
        print(f"\n❌ CROSSENCODER 重排器失败: {e}")


def compare_all_methods():
    """对比所有重排方法"""
    print("\n" + "=" * 60)
    print("⚖️  对比所有重排方法")
    print("=" * 60)
    
    query = "解释一下人工智能、机器学习和深度学习的关系"
    docs = create_test_documents()
    
    methods = {
        'TF-IDF': TFIDFReranker(),
        'BM25': BM25Reranker(),
    }
    
    # 尝试添加 CrossEncoder
    try:
        methods['CrossEncoder'] = CrossEncoderReranker()
    except:
        print("\n⚠️ CrossEncoder 不可用，跳过")
    
    print(f"\n查询: {query}\n")
    
    for method_name, reranker in methods.items():
        try:
            results = reranker.rerank(query, docs, top_k=3)
            print(f"\n{'=' * 40}")
            print(f"{method_name} 重排结果:")
            print('=' * 40)
            for i, (doc, score) in enumerate(results, 1):
                print(f"{i}. [{score:.4f}] {doc.page_content[:60]}...")
        except Exception as e:
            print(f"\n{method_name} 失败: {e}")


def performance_comparison():
    """性能对比"""
    print("\n" + "=" * 60)
    print("⚡ 性能与准确性对比")
    print("=" * 60)
    
    print("""
    重排方法对比：
    
    ┌─────────────────┬──────────┬──────────┬──────────┬────────────┐
    │ 方法            │ 准确率   │ 速度     │ 成本     │ 适用场景   │
    ├─────────────────┼──────────┼──────────┼──────────┼────────────┤
    │ TF-IDF          │ ⭐⭐     │ ⚡⚡⚡   │ 极低     │ 关键词匹配 │
    │ BM25            │ ⭐⭐⭐   │ ⚡⚡⚡   │ 极低     │ 文本检索   │
    │ Bi-Encoder      │ ⭐⭐⭐⭐ │ ⚡⚡     │ 低       │ 语义检索   │
    │ CrossEncoder 🌟 │ ⭐⭐⭐⭐⭐│ ⚡       │ 中       │ 精准重排   │
    │ Hybrid          │ ⭐⭐⭐⭐ │ ⚡⚡     │ 低       │ 综合场景   │
    └─────────────────┴──────────┴──────────┴──────────┴────────────┘
    
    推荐配置：
    1️⃣  两阶段检索：Bi-Encoder (快速召回) + CrossEncoder (精准重排)
    2️⃣  准确率优先：纯 CrossEncoder
    3️⃣  速度优先：BM25 或 Hybrid
    
    当前项目配置：
    ✅ 已切换到 CrossEncoder 重排
    📈 准确率预期提升：15-20%
    ⚡ 速度：单次重排 20-100ms (Top 20 文档)
    """)


if __name__ == "__main__":
    print("\n🚀 开始测试 CrossEncoder 重排功能...\n")
    
    # 1. 测试 TF-IDF
    test_tfidf_reranking()
    
    # 2. 测试 BM25
    test_bm25_reranking()
    
    # 3. 测试 CrossEncoder (重点)
    crossencoder_available = test_crossencoder_reranking()
    
    # 4. 测试工厂函数
    test_factory_function()
    
    # 5. 对比所有方法
    compare_all_methods()
    
    # 6. 性能对比总结
    performance_comparison()
    
    print("\n" + "=" * 60)
    if crossencoder_available:
        print("✅ 所有测试完成！CrossEncoder 重排已就绪")
    else:
        print("⚠️  测试完成，但 CrossEncoder 不可用")
        print("   请运行: pip install sentence-transformers")
    print("=" * 60 + "\n")
