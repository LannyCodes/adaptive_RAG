#!/usr/bin/env python3
"""
重排功能测试脚本
演示不同重排策略的效果
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from document_processor import DocumentProcessor
from reranker import *
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document
import time


def create_test_documents():
    """创建测试文档"""
    return [
        Document(
            page_content="人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            metadata={"source": "ai_intro.txt", "category": "AI基础"}
        ),
        Document(
            page_content="机器学习是人工智能的一个重要子领域，通过算法让计算机从数据中学习模式和规律。",
            metadata={"source": "ml_basics.txt", "category": "机器学习"}
        ),
        Document(
            page_content="深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。",
            metadata={"source": "dl_guide.txt", "category": "深度学习"}
        ),
        Document(
            page_content="自然语言处理（NLP）是人工智能领域的一个重要分支，专注于使计算机理解和处理人类语言。",
            metadata={"source": "nlp_overview.txt", "category": "自然语言处理"}
        ),
        Document(
            page_content="计算机视觉是人工智能的另一个重要领域，使计算机能够识别和理解图像和视频内容。",
            metadata={"source": "cv_intro.txt", "category": "计算机视觉"}
        ),
        Document(
            page_content="强化学习是机器学习的一种类型，通过与环境交互来学习最优的行为策略。",
            metadata={"source": "rl_basics.txt", "category": "强化学习"}
        ),
        Document(
            page_content="今天的天气非常好，阳光明媚，适合外出游玩和运动。",
            metadata={"source": "weather.txt", "category": "天气"}
        ),
        Document(
            page_content="区块链是一种分布式账本技术，具有去中心化、不可篡改等特点。",
            metadata={"source": "blockchain.txt", "category": "区块链"}
        )
    ]


def test_reranker_comparison():
    """比较不同重排器的效果"""
    print("🔍 重排器效果比较测试")
    print("=" * 60)
    
    # 创建测试数据
    query = "什么是人工智能和机器学习？"
    documents = create_test_documents()
    
    # 创建一个简单的嵌入模型（用于测试）
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ 成功加载嵌入模型")
    except Exception as e:
        print(f"❌ 嵌入模型加载失败: {e}")
        print("将使用基础重排器进行测试")
        embeddings = None
    
    # 测试不同的重排器
    rerankers = []
    
    # TF-IDF重排器
    rerankers.append(("TF-IDF", TFIDFReranker()))
    
    # BM25重排器
    rerankers.append(("BM25", BM25Reranker()))
    
    if embeddings:
        # 语义重排器
        rerankers.append(("语义相似度", SemanticReranker(embeddings)))
        
        # 混合重排器
        rerankers.append(("混合策略", HybridReranker(embeddings)))
        
        # 多样性重排器
        rerankers.append(("多样性优化", DiversityReranker(embeddings)))
    
    # 执行测试
    for name, reranker in rerankers:
        print(f"\n📊 {name} 重排结果:")
        print("-" * 40)
        
        start_time = time.time()
        try:
            results = reranker.rerank(query, documents, top_k=5)
            end_time = time.time()
            
            print(f"⏱️ 处理时间: {(end_time - start_time)*1000:.2f}ms")
            
            for i, (doc, score) in enumerate(results, 1):
                content = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                category = doc.metadata.get('category', '未知')
                print(f"{i}. [分数: {score:.4f}] [{category}] {content}")
                
        except Exception as e:
            print(f"❌ 重排失败: {e}")


def test_reranking_with_embeddings():
    """测试带嵌入的重排功能"""
    print("\n\n🧠 嵌入模型重排测试")
    print("=" * 60)
    
    try:
        # 创建文档处理器
        processor = DocumentProcessor()
        
        # 创建测试文档
        test_docs = create_test_documents()
        
        # 测试查询
        queries = [
            "人工智能的定义是什么？",
            "机器学习和深度学习的区别",
            "自然语言处理的应用",
            "今天天气怎么样？"
        ]
        
        for query in queries:
            print(f"\n🔍 查询: {query}")
            print("-" * 30)
            
            if processor.reranker:
                # 使用重排功能
                results = processor.reranker.rerank(query, test_docs, top_k=3)
                
                for i, (doc, score) in enumerate(results, 1):
                    content = doc.page_content[:60] + "..." if len(doc.page_content) > 60 else doc.page_content
                    category = doc.metadata.get('category', '未知')
                    print(f"{i}. [分数: {score:.4f}] [{category}] {content}")
            else:
                print("❌ 重排器未初始化")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def test_performance_comparison():
    """性能对比测试"""
    print("\n\n⚡ 性能对比测试")
    print("=" * 60)
    
    documents = create_test_documents() * 10  # 增加文档数量
    query = "人工智能技术的发展趋势"
    
    # 测试不同重排器的性能
    rerankers_config = [
        ("无重排", None),
        ("TF-IDF", TFIDFReranker()),
        ("BM25", BM25Reranker())
    ]
    
    for name, reranker in rerankers_config:
        times = []
        
        # 多次测试取平均值
        for _ in range(5):
            start_time = time.time()
            
            if reranker:
                results = reranker.rerank(query, documents, top_k=5)
            else:
                # 模拟无重排的情况
                results = documents[:5]
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"{name}: 平均处理时间 {avg_time:.2f}ms (文档数: {len(documents)})")


def main():
    """主测试函数"""
    print("🚀 向量重排功能综合测试")
    print("=" * 80)
    
    try:
        # 基础重排器比较
        test_reranker_comparison()
        
        # 嵌入模型重排测试
        test_reranking_with_embeddings()
        
        # 性能对比测试
        test_performance_comparison()
        
        print("\n\n✅ 所有测试完成!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n❌ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()