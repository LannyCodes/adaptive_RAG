"""
高级重排器使用示例
演示上下文感知重排和多任务重排的使用方法
"""

from reranker import ContextAwareReranker, MultiTaskReranker, create_reranker
from langchain_core.documents import Document
import numpy as np


def test_context_aware_reranker():
    """测试上下文感知重排器"""
    print("\n" + "="*60)
    print("测试上下文感知重排器")
    print("="*60)
    
    # 模拟文档
    documents = [
        Document(
            page_content="机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
            metadata={"source": "wiki.edu", "type": "textbook"}
        ),
        Document(
            page_content="深度学习使用多层神经网络来处理复杂的数据模式。",
            metadata={"source": "arxiv.org", "type": "paper"}
        ),
        Document(
            page_content="Python是一种流行的编程语言，广泛用于数据科学。",
            metadata={"source": "python.org", "type": "documentation"}
        ),
        Document(
            page_content="自然语言处理让计算机能够理解和生成人类语言。",
            metadata={"source": "nlp-gov.org", "type": "research"}
        ),
        Document(
            page_content="计算机视觉是AI的一个重要领域，用于图像识别。",
            metadata={"source": "cv-conference.com", "type": "paper"}
        )
    ]
    
    query = "什么是机器学习？"
    
    # 创建上下文信息
    context = {
        'conversation_history': [
            {"role": "user", "content": "我对人工智能很感兴趣"},
            {"role": "assistant", "content": "AI包含机器学习、深度学习等多个领域"},
            {"role": "user", "content": "能详细介绍一下机器学习吗？"}
        ],
        'user_preferences': {
            'preferred_topics': ['机器学习', '算法', '数据科学'],
            'avoid_topics': ['数学公式', '理论证明']
        },
        'previous_documents': [
            Document(page_content="人工智能是计算机科学的一个分支，致力于创建智能机器。")
        ],
        'query_intent': {
            'type': 'conceptual',
            'difficulty': 'beginner'
        }
    }
    
    # 方法1: 直接创建
    try:
        reranker = ContextAwareReranker(
            model_name="BAAI/bge-reranker-base",
            max_length=1024,
            context_weight=0.3
        )
        
        results = reranker.rerank(
            query=query,
            documents=documents,
            top_k=3,
            context=context
        )
        
        print(f"\n查询: {query}")
        print(f"\n重排结果 (Top 3):")
        print("-" * 60)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"{rank}. 分数: {score:.4f}")
            print(f"   内容: {doc.page_content[:80]}...")
            print(f"   来源: {doc.metadata.get('source', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"⚠️ 上下文感知重排器测试失败: {e}")
        print("   这可能是因为模型未下载或GPU内存不足")


def test_multi_task_reranker():
    """测试多任务重排器"""
    print("\n" + "="*60)
    print("测试多任务重排器")
    print("="*60)
    
    # 模拟文档（带时间和权威性信息）
    from datetime import datetime, timedelta
    
    documents = [
        Document(
            page_content="2023年最新的机器学习综述论文总结了当前研究进展。",
            metadata={
                "source": "arxiv.edu",
                "author": "张三教授",
                "citations": 150,
                "timestamp": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            }
        ),
        Document(
            page_content="机器学习基础教程：从线性回归到神经网络。",
            metadata={
                "source": "university.edu",
                "author": "李四博士",
                "citations": 50,
                "timestamp": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            }
        ),
        Document(
            page_content="实用Python机器学习指南，包含大量代码示例。",
            metadata={
                "source": "github.com",
                "author": "开发团队",
                "citations": 20,
                "timestamp": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            }
        ),
        Document(
            page_content="机器学习数学基础：线性代数、概率论和优化理论。",
            metadata={
                "source": "textbook.org",
                "author": "王五教授",
                "citations": 300,
                "timestamp": (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
            }
        ),
        Document(
            page_content="工业界机器学习应用案例：推荐系统、广告排序等。",
            metadata={
                "source": "tech-company.com",
                "author": "工程师团队",
                "citations": 80,
                "timestamp": (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            }
        )
    ]
    
    query = "机器学习入门学习资料"
    
    # 方法1: 使用工厂函数
    try:
        # 需要先有embeddings_model（从document_processor获取）
        from config import EMBEDDING_MODEL
        from langchain_huggingface import HuggingFaceEmbeddings
        
        print("加载嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        reranker = create_reranker(
            'multi_task',
            embeddings_model=embeddings,
            weights={
                'relevance': 0.35,      # 相关性
                'diversity': 0.25,      # 多样性
                'novelty': 0.15,        # 新颖性
                'authority': 0.15,      # 权威性
                'recency': 0.10         # 时效性
            },
            diversity_lambda=0.5
        )
        
        results = reranker.rerank(
            query=query,
            documents=documents,
            top_k=3
        )
        
        print(f"\n查询: {query}")
        print(f"权重配置: {reranker.weights}")
        print(f"\n重排结果 (Top 3):")
        print("-" * 60)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"{rank}. 综合分数: {score:.4f}")
            print(f"   内容: {doc.page_content[:80]}...")
            print(f"   来源: {doc.metadata.get('source', 'N/A')}")
            print(f"   引用: {doc.metadata.get('citations', 0)}")
            print(f"   时间: {doc.metadata.get('timestamp', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"⚠️ 多任务重排器测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_comparison():
    """比较不同重排器的效果"""
    print("\n" + "="*60)
    print("重排器效果比较")
    print("="*60)
    
    documents = [
        Document(page_content="机器学习算法可以从数据中学习模式并做出预测。"),
        Document(page_content="深度学习是机器学习的一个子领域，使用神经网络。"),
        Document(page_content="Python编程基础：变量、循环和函数。"),
        Document(page_content="机器学习在医疗诊断中的应用案例分析。"),
        Document(page_content="数据预处理：清洗、转换和特征工程。")
    ]
    
    query = "机器学习应用案例"
    
    # 测试不同的重排器
    rerankers_to_test = [
        ('CrossEncoder (基础)', lambda: create_reranker('crossencoder', model_name='BAAI/bge-reranker-base')),
        ('上下文感知', lambda: create_reranker('context_aware', context_weight=0.3)),
    ]
    
    for name, create_fn in rerankers_to_test:
        try:
            print(f"\n{name}:")
            print("-" * 60)
            
            reranker = create_fn()
            context = {
                'conversation_history': [
                    {"role": "user", "content": "我想了解机器学习的实际应用"}
                ],
                'query_intent': {'type': 'practical'}
            }
            
            # 根据重排器类型调用
            if name == '上下文感知':
                results = reranker.rerank(query, documents, top_k=3, context=context)
            else:
                results = reranker.rerank(query, documents, top_k=3)
            
            for rank, (doc, score) in enumerate(results, 1):
                print(f"  {rank}. [{score:.4f}] {doc.page_content[:60]}...")
                
        except Exception as e:
            print(f"  ⚠️ {name} 失败: {e}")


if __name__ == "__main__":
    print("\n🚀 高级重排器使用示例")
    print("="*60)
    
    # 测试上下文感知重排
    test_context_aware_reranker()
    
    # 测试多任务重排
    test_multi_task_reranker()
    
    # 比较不同重排器
    test_comparison()
    
    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60)
