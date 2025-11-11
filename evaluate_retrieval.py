"""
自适应RAG系统检索效果评估脚本
评估不同检索策略和配置的效果
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入项目模块
from main import AdaptiveRAGSystem
from document_processor import DocumentProcessor
from retrieval_evaluation import RetrievalEvaluator, RetrievalResult, RetrievalTestSet
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document

# 导入LangChain相关模块
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    try:
        from langchain_core.retrievers import EnsembleRetriever
        from langchain_core.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
    except ImportError:
        print("Warning: Could not import advanced retriever components. Some features may be limited.")
        EnsembleRetriever = None
        ContextualCompressionRetriever = None
        LLMChainExtractor = None


class AdaptiveRAGRetriever:
    """自适应RAG系统检索器包装器"""
    
    def __init__(self, system_config: Dict[str, Any], retriever_type: str = "default"):
        """
        初始化检索器
        
        Args:
            system_config: 系统配置
            retriever_type: 检索器类型
        """
        self.system_config = system_config
        self.retriever_type = retriever_type
        self.system = None
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化RAG系统"""
        try:
            # 根据检索器类型调整配置
            config = self.system_config.copy()
            
            if self.retriever_type == "vector_only":
                config["retrieval_strategy"] = "vector"
            elif self.retriever_type == "bm25_only":
                config["retrieval_strategy"] = "bm25"
            elif self.retriever_type == "hybrid":
                config["retrieval_strategy"] = "hybrid"
            elif self.retriever_type == "graph":
                config["retrieval_strategy"] = "graph"
            elif self.retriever_type == "compression":
                config["use_compression"] = True
            elif self.retriever_type == "rerank":
                config["use_reranking"] = True
            elif self.retriever_type == "query_expansion":
                config["use_query_expansion"] = True
            
            # 创建系统实例
            self.system = AdaptiveRAGSystem(config)
            
            # 初始化文档处理器（如果需要）
            if not hasattr(self.system, 'document_processor') or self.system.document_processor is None:
                self.system.document_processor = DocumentProcessor(config)
            
        except Exception as e:
            print(f"初始化RAG系统失败: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        检索文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            检索到的文档列表
        """
        try:
            # 使用系统的检索方法
            if hasattr(self.system, 'retrieve'):
                docs = self.system.retrieve(query, top_k)
            else:
                # 如果没有直接的retrieve方法，尝试通过文档处理器检索
                if self.system.document_processor:
                    docs = self.system.document_processor.retrieve(query, top_k)
                else:
                    raise ValueError("无法找到检索方法")
            
            return docs[:top_k]
        except Exception as e:
            print(f"检索失败: {e}")
            return []


def create_evaluation_dataset(data_dir: str = "data", num_queries: int = 20) -> RetrievalTestSet:
    """
    从项目数据创建评估数据集
    
    Args:
        data_dir: 数据目录
        num_queries: 查询数量
        
    Returns:
        检索测试集
    """
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"数据目录 {data_dir} 不存在，创建示例数据集")
        from retrieval_evaluation import create_sample_test_set
        return create_sample_test_set()
    
    # 尝试从现有数据创建测试集
    try:
        # 加载文档
        documents = []
        doc_files = []
        
        # 查找所有文本文件
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.txt') or file.endswith('.md'):
                    doc_files.append(os.path.join(root, file))
        
        # 如果没有找到文档文件，创建示例数据集
        if not doc_files:
            print(f"在 {data_dir} 中未找到文档文件，创建示例数据集")
            from retrieval_evaluation import create_sample_test_set
            return create_sample_test_set()
        
        # 读取文档内容
        for i, file_path in enumerate(doc_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(Document(page_content=content, metadata={"source": file_path, "doc_id": str(i)}))
        
        # 生成查询（这里简化处理，实际应用中应该使用真实查询）
        queries = []
        qrels = {}
        
        # 从文档中提取关键句子作为查询
        for i in range(min(num_queries, len(documents))):
            doc = documents[i]
            sentences = doc.page_content.split('.')
            if sentences:
                # 取第一个非空句子作为查询
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:  # 确保查询有足够长度
                        queries.append(sentence)
                        # 假设查询与当前文档相关
                        qrels[str(i)] = {str(i): 2}  # 高度相关
                        # 可能与其他文档也相关
                        for j in range(min(3, len(documents))):
                            if j != i:
                                qrels[str(i)][str(j)] = 1  # 部分相关
                        break
        
        # 保存查询文件
        with open("eval_queries.txt", "w", encoding="utf-8") as f:
            for query in queries:
                f.write(query + "\n")
        
        # 保存文档文件
        with open("eval_documents.txt", "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.page_content + "\n")
        
        # 保存相关性标注文件
        with open("eval_qrels.csv", "w", encoding="utf-8") as f:
            for query_id, doc_relevance in qrels.items():
                for doc_id, relevance in doc_relevance.items():
                    f.write(f"{query_id},{doc_id},{relevance}\n")
        
        print(f"评估数据集已创建:")
        print(f"- 查询数量: {len(queries)}")
        print(f"- 文档数量: {len(documents)}")
        print(f"- eval_queries.txt: 查询文件")
        print(f"- eval_documents.txt: 文档文件")
        print(f"- eval_qrels.csv: 相关性标注文件")
        
        return RetrievalTestSet("eval_queries.txt", "eval_documents.txt", "eval_qrels.csv")
    
    except Exception as e:
        print(f"创建评估数据集失败: {e}")
        print("创建示例数据集")
        from retrieval_evaluation import create_sample_test_set
        return create_sample_test_set()


def evaluate_retrievers(system_config: Dict[str, Any], 
                       retriever_types: List[str],
                       test_set: RetrievalTestSet,
                       output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """
    评估多个检索器
    
    Args:
        system_config: 系统配置
        retriever_types: 检索器类型列表
        test_set: 测试集
        output_dir: 输出目录
        
    Returns:
        评估结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化评估器
    evaluator = RetrievalEvaluator()
    
    # 存储所有检索结果
    all_results = {}
    
    # 评估每个检索器
    for retriever_type in retriever_types:
        print(f"\n评估检索器: {retriever_type}")
        print("=" * 50)
        
        try:
            # 创建检索器
            retriever = AdaptiveRAGRetriever(system_config, retriever_type)
            
            # 获取检索结果
            results = test_set.get_retrieval_results(retriever)
            all_results[retriever_type] = results
            
            print(f"完成 {len(results)} 个查询的检索")
            
        except Exception as e:
            print(f"评估检索器 {retriever_type} 失败: {e}")
            continue
    
    # 比较检索器
    if len(all_results) > 1:
        print("\n比较检索器性能")
        print("=" * 50)
        metrics = evaluator.compare_retrievers(all_results)
        
        # 生成报告
        report = evaluator.generate_report(
            metrics, 
            os.path.join(output_dir, "retrieval_evaluation_report.md")
        )
        
        # 绘制比较图
        evaluator.plot_metrics_comparison(
            metrics, 
            os.path.join(output_dir, "retrieval_evaluation_comparison.png")
        )
        
        # 保存详细指标
        metrics_data = {}
        for name, metric in metrics.items():
            metrics_data[name] = {
                "precision_at_k": metric.precision_at_k,
                "recall_at_k": metric.recall_at_k,
                "f1_at_k": metric.f1_at_k,
                "map_score": metric.map_score,
                "mrr": metric.mrr,
                "ndcg_at_k": metric.ndcg_at_k,
                "coverage": metric.coverage,
                "diversity": metric.diversity,
                "novelty": metric.novelty,
                "latency": metric.latency
            }
        
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        return {
            "metrics": metrics,
            "metrics_data": metrics_data,
            "report": report,
            "results": all_results
        }
    else:
        print("只有一个检索器成功评估，跳过比较")
        return {"results": all_results}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估自适应RAG系统的检索效果")
    parser.add_argument("--config", type=str, default="config.py", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="输出目录")
    parser.add_argument("--num_queries", type=int, default=20, help="查询数量")
    parser.add_argument("--retrievers", nargs="+", 
                       default=["default", "vector_only", "bm25_only", "hybrid"],
                       help="要评估的检索器类型")
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        if args.config.endswith('.py'):
            # 动态导入Python配置文件
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", args.config)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            system_config = config_module.config
        else:
            # 加载JSON配置文件
            with open(args.config, 'r', encoding='utf-8') as f:
                system_config = json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("使用默认配置")
        system_config = {
            "model_name": "gpt-3.5-turbo",
            "vector_store": "faiss",
            "retrieval_strategy": "hybrid",
            "use_reranking": False,
            "use_compression": False,
            "use_query_expansion": False
        }
    
    # 创建评估数据集
    print("创建评估数据集")
    test_set = create_evaluation_dataset(args.data_dir, args.num_queries)
    
    # 评估检索器
    print("\n开始评估检索器")
    results = evaluate_retrievers(system_config, args.retrievers, test_set, args.output_dir)
    
    print("\n评估完成!")
    print(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()