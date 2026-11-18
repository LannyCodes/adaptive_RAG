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
from langchain_core.documents import Document

# 导入LangChain相关模块
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


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


"""

========================================================================
汇总
========================================================================
总数        : 20
成功        : 20
失败        : 0
HTTP 429    : 0
缓存命中    : 0
有指标样本  : 20
可评排序样本: 15（仅 agent_graded）
总墙钟时间  : 96.439s
平均延迟    : 66.173s
P50 / P95   : 78.816s / 94.239s
最大延迟    : 96.399s
平均 P@1    : 0.9333  (仅 agent_graded)
平均 P@3    : 0.6889  (仅 agent_graded)
平均 R@3    : 0.8256  (仅 agent_graded)
平均 MAP    : 0.9444  (仅 agent_graded)
平均 MRR    : 0.9667  (仅 agent_graded)
检索耗时均值: 2.9118s
检索次数均值: 1.45
检索文档均值: 4.3
评分相关均值: 2.7
工具调用均值: 6.55
指标来源    : {'agent_graded_fallback': 3, 'agent_graded': 15, 'unavailable': 2}
评分兜底    : 3  | 使用网络搜索: 3
幻觉检查    : 通过 0 / 未通过 1
报告文件    : /kaggle/working/adaptive_RAG/data/benchmark_concurrent_20_v3.json
========================================================================

"""
"""
{
  "run_id": "20261118-124836-079c62",
  "base_url": "http://127.0.0.1:8000",
  "concurrency": 20,
  "questions": 20,
  "summary": {
    "total": 20,
    "success": 20,
    "failed": 0,
    "http_429": 0,
    "cached": 0,
    "metrics_available": 20,
    "ranked_samples": 15,
    "wall_seconds": 96.439,
    "latency_mean": 66.173,
    "latency_p50": 78.816,
    "latency_p95": 94.239,
    "latency_max": 96.399,
    "precision_at_1": 0.9333,
    "precision_at_3": 0.6889,
    "recall_at_3": 0.8256,
    "map_score": 0.9444,
    "mrr": 0.9667,
    "retrieval_latency_mean": 2.9118,
    "retrieval_calls_mean": 1.45,
    "retrieved_docs_mean": 4.3,
    "graded_relevant_mean": 2.7,
    "tool_calls_mean": 6.55,
    "metric_source_counts": {
      "agent_graded_fallback": 3,
      "agent_graded": 15,
      "unavailable": 2
    },
    "grader_fallback": 3,
    "web_search_used": 3,
    "hallucination_passed": 0,
    "hallucination_failed": 1
  },
  "results": [
    {
      "index": 1,
      "question": "AI Agent的四个核心组成部分是什么？",
      "status": 200,
      "elapsed": 47.726783615000386,
      "cached": false,
      "answer_preview": "根据网络搜索结果，AI Agent的核心组成部分包括：\n\n1. **大语言模型（LLM）**：作为Agent的推理引擎，负责处理自然语言指令，进行逻辑推理、上下文理解以及决策生成。它决定了Agent的智商上限和指令遵循能力。\n2. **规划模块**：规划能力是Agent应对复杂任务的关键。面对一个宏大目标，规划模块会将其拆解为多个可执行的子任务。\n3. **记忆模块**：记忆模块使Agent具备保",
      "sources_count": 5,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 1.0,
        "recall_at_1": 0.2,
        "recall_at_3": 0.6,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 6.231101,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 5,
        "graded_total_docs": 5,
        "retrieval_calls": 2,
        "unique_sources": 2,
        "route": "vectorstore",
        "answerability": null,
        "web_search_calls": 1,
        "rewrite_calls": 1,
        "tool_call_count": 8,
        "grader_fallback": true,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded_fallback",
        "ground_truth": false,
        "evaluation_note": "评分兜底保留了全部文档，Precision/Recall/MAP 仅供参考"
      },
      "error": null
    },
    {
      "index": 2,
      "question": "什么是Chain-of-Thought (CoT) 提示技术？",
      "status": 200,
      "elapsed": 35.085312314000475,
      "cached": false,
      "answer_preview": "根据网络搜索结果，Chain-of-Thought (CoT) 提示技术是一种提示工程技术，它要求大型语言模型在给出最终答案之前，先生成中间步骤，以此来展示其推理过程。与基于训练的方法不同，CoT 是一种简单的提示策略，可直接应用于现有模型，而无需修改其权重或架构。这项技术通过把复杂问题分解成循序渐进的逻辑步骤，能显著提升模型在多步骤任务（如算术、常识推理和符号逻辑）中的性能。同时，它也使模型得出",
      "sources_count": 5,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 1.0,
        "recall_at_1": 0.2,
        "recall_at_3": 0.6,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 2.588099,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 5,
        "graded_total_docs": 5,
        "retrieval_calls": 1,
        "unique_sources": 2,
        "route": "vectorstore",
        "answerability": null,
        "web_search_calls": 1,
        "rewrite_calls": 0,
        "tool_call_count": 5,
        "grader_fallback": true,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded_fallback",
        "ground_truth": false,
        "evaluation_note": "评分兜底保留了全部文档，Precision/Recall/MAP 仅供参考"
      },
      "error": null
    },
    {
      "index": 3,
      "question": "大语言模型面临哪些类型的对抗攻击？",
      "status": 200,
      "elapsed": 31.32587256899933,
      "cached": false,
      "answer_preview": "Sorry, need more steps to process this request.",
      "sources_count": 1,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 0.5,
        "precision_at_5": 0.5,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 5.051421,
        "retrieved_docs_count": 2,
        "last_retrieval_docs": 2,
        "graded_relevant_count": 1,
        "graded_total_docs": 2,
        "retrieval_calls": 2,
        "unique_sources": 1,
        "route": "vectorstore",
        "answerability": "可回答性: 不足，需要更多信息",
        "web_search_calls": 0,
        "rewrite_calls": 1,
        "tool_call_count": 9,
        "grader_fallback": false,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 4,
      "question": "AI Agent中的记忆系统分为哪两种类型？",
      "status": 200,
      "elapsed": 27.393955356000333,
      "cached": false,
      "answer_preview": "根据知识库文档，AI Agent中的记忆系统分为两种类型：短时记忆和长时记忆。\n\n- 短时记忆：主要利用上下文学习来实现，使模型能够处理和学习当前任务的相关信息。\n- 长时记忆：提供给代理保留和回忆（理论上无限）信息的能力，通常通过利用外部向量存储和快速检索来实现。",
      "sources_count": 2,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 0.3333333333333333,
        "precision_at_5": 0.4,
        "recall_at_1": 0.5,
        "recall_at_3": 0.5,
        "recall_at_5": 1.0,
        "map_score": 0.75,
        "mrr": 1.0,
        "latency": 3.827115,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 2,
        "graded_total_docs": 5,
        "retrieval_calls": 1,
        "unique_sources": 3,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 6,
        "grader_fallback": false,
        "generation_latency": 4.022779,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 5,
      "question": "如何通过提示工程来引导LLM的行为？",
      "status": 200,
      "elapsed": 59.77940369600037,
      "cached": false,
      "answer_preview": "Sorry, need more steps to process this request.",
      "sources_count": 3,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 0.75,
        "recall_at_1": 0.3333333333333333,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 4.042637,
        "retrieved_docs_count": 4,
        "last_retrieval_docs": 4,
        "graded_relevant_count": 3,
        "graded_total_docs": 4,
        "retrieval_calls": 2,
        "unique_sources": 2,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 1,
        "tool_call_count": 10,
        "grader_fallback": false,
        "generation_latency": 14.482484,
        "hallucination_check": "未通过: 答案存在文档不能支撑的内容。应改写查询重新检索。",
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 6,
      "question": "对抗性攻击如何影响大语言模型的安全性？",
      "status": 200,
      "elapsed": 62.27668269800051,
      "cached": false,
      "answer_preview": "优化后的查询为：“对抗性攻击如何影响大语言模型的安全性，以及什么是对抗训练？”\n\n我们将使用这个优化后的查询再次从知识库中检索相关信息。",
      "sources_count": 3,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 0.6666666666666666,
        "precision_at_5": 0.75,
        "recall_at_1": 0.3333333333333333,
        "recall_at_3": 0.6666666666666666,
        "recall_at_5": 1.0,
        "map_score": 0.9166666666666666,
        "mrr": 1.0,
        "latency": 6.215075,
        "retrieved_docs_count": 4,
        "last_retrieval_docs": 4,
        "graded_relevant_count": 3,
        "graded_total_docs": 4,
        "retrieval_calls": 2,
        "unique_sources": 2,
        "route": "vectorstore",
        "answerability": "可回答性: 不足，需要更多信息",
        "web_search_calls": 0,
        "rewrite_calls": 1,
        "tool_call_count": 7,
        "grader_fallback": false,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 7,
      "question": "AI Agent的任务规划能力包括哪些方面？",
      "status": 200,
      "elapsed": 24.131571048000296,
      "cached": false,
      "answer_preview": "根据知识库文档，AI Agent的任务规划能力主要体现在以下几个方面：\n\n1. **规划**：AI Agent能够将任务分解成更小的子任务，并根据当前环境和历史信息做出决策。规划过程旨在优化当前的信念与长期目标之间的平衡。\n\n2. **反思机制**：通过合成记忆，形成更高层次的推理，并指导未来的行为。这涉及到将最近的观察转化为三个最相关的高层次问题，并基于这些问题生成答案。\n\n3. **工具使用*",
      "sources_count": 2,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 0.0,
        "precision_at_3": 0.6666666666666666,
        "precision_at_5": 0.4,
        "recall_at_1": 0.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "map_score": 0.5833333333333333,
        "mrr": 0.5,
        "latency": 4.336128,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 2,
        "graded_total_docs": 5,
        "retrieval_calls": 1,
        "unique_sources": 2,
        "route": "vectorstore",
        "answerability": null,
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 4,
        "grader_fallback": false,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 8,
      "question": "什么是提示工程中的上下文提示？",
      "status": 200,
      "elapsed": 37.82836311699975,
      "cached": false,
      "answer_preview": "根据知识库文档，提示工程中的上下文提示是指一种方法，通过这种方式，可以指导大规模语言模型（LLM）的行为以实现期望的结果，而无需更新模型权重。上下文提示是提示工程的一种形式，其核心目标是实现对齐和模型可引导性。在零样本和少样本学习中，上下文提示通过提供任务文本和期望结果的示例来引导模型。这些示例帮助模型更好地理解人类意图和期望的答案类型，从而提高性能。然而，这种方法会增加令牌消耗，并且当输入和输出",
      "sources_count": 1,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 0.3333333333333333,
        "precision_at_5": 0.25,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 3.179457,
        "retrieved_docs_count": 4,
        "last_retrieval_docs": 4,
        "graded_relevant_count": 1,
        "graded_total_docs": 4,
        "retrieval_calls": 1,
        "unique_sources": 3,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 6,
        "grader_fallback": false,
        "generation_latency": 7.658978,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 9,
      "question": "什么是自监督学习？",
      "status": 200,
      "elapsed": 85.30360190699957,
      "cached": false,
      "answer_preview": "根据知识库文档，自监督学习是一种利用数据本身的未标记信息进行训练的方法，其目的是利用大量未标记数据中的潜在标签，而无需手动标注。具体来说，自监督学习通过构建一个监督学习任务，仅使用数据的一部分信息来预测另一部分信息，从而从数据中获取监督信息。这样，所有的信息需求（输入和标签）都得到了满足，而不需要人工标注。\n\n自监督学习的一个重要特点是，它可以通过构造特定的预训练任务来利用数据中的未标记信息，而这",
      "sources_count": 3,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 0.75,
        "recall_at_1": 0.3333333333333333,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 2.53069,
        "retrieved_docs_count": 4,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 3,
        "graded_total_docs": 4,
        "retrieval_calls": 2,
        "unique_sources": 2,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 1,
        "tool_call_count": 9,
        "grader_fallback": false,
        "generation_latency": 12.111026,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 10,
      "question": "什么是提示工程指南？",
      "status": 200,
      "elapsed": 79.64590966600008,
      "cached": false,
      "answer_preview": "根据知识库文档，提示工程指南（Prompt Engineering Guide）是关于如何使用提示（prompts）来引导生成式AI模型的行为，以实现预期结果的一种方法，而不涉及模型权重的更新。这是一种经验科学，提示工程方法的效果在不同模型之间差异很大，因此需要大量的实验和启发式方法。\n\n提示工程的核心目标是实现对模型的对齐和引导。它主要关注自动提示设计，包括增强语言模型、检索、编程语言和外部AP",
      "sources_count": 3,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 0.6666666666666666,
        "precision_at_5": 0.75,
        "recall_at_1": 0.3333333333333333,
        "recall_at_3": 0.6666666666666666,
        "recall_at_5": 1.0,
        "map_score": 0.9166666666666666,
        "mrr": 1.0,
        "latency": 2.276746,
        "retrieved_docs_count": 4,
        "last_retrieval_docs": 4,
        "graded_relevant_count": 3,
        "graded_total_docs": 4,
        "retrieval_calls": 1,
        "unique_sources": 4,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 6,
        "grader_fallback": false,
        "generation_latency": 15.53925,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 11,
      "question": "MinMaxScaler的公式是什么？请列出公式并说明",
      "status": 200,
      "elapsed": 72.18006192200028,
      "cached": false,
      "answer_preview": "根据网络搜索结果，`MinMaxScaler` 的公式如下：\n\n\\[ X_{\\text{new}} = a + \\frac{(X - X_{\\text{min}})}{(X_{\\text{max}} - X_{\\text{min}})} (b - a) \\]\n\n其中：\n- \\( X \\) 是原始数据。\n- \\( X_{\\text{min}} \\) 是数据中的最小值。\n- \\( X_{\\text{m",
      "sources_count": 4,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 1.0,
        "recall_at_1": 0.25,
        "recall_at_3": 0.75,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 3.41381,
        "retrieved_docs_count": 4,
        "last_retrieval_docs": 4,
        "graded_relevant_count": 4,
        "graded_total_docs": 4,
        "retrieval_calls": 2,
        "unique_sources": 4,
        "route": "vectorstore",
        "answerability": null,
        "web_search_calls": 1,
        "rewrite_calls": 1,
        "tool_call_count": 7,
        "grader_fallback": true,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded_fallback",
        "ground_truth": false,
        "evaluation_note": "评分兜底保留了全部文档，Precision/Recall/MAP 仅供参考"
      },
      "error": null
    },
    {
      "index": 12,
      "question": "什么是主体资信供应商风险偏离度？",
      "status": 200,
      "elapsed": 60.119081240000014,
      "cached": false,
      "answer_preview": "根据知识库文档，主体资信供应商风险偏离度是风险预测引擎（R-Engine）实现“千人千面”差异化预警的核心特征之一。它将承运商的历史履约能力转化为动态可计算的数学指标，具体通过变异系数（CV）来衡量。变异系数的计算公式为：\n\n\\[ CV = \\frac{\\sigma}{\\mu} \\]\n\n其中，\\(\\mu\\) 代表历史时效均值，\\(\\sigma\\) 代表历史时效标准差。变异系数通过除以均值来消除量纲",
      "sources_count": 4,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 0.0,
        "precision_at_3": 0.0,
        "precision_at_5": 0.0,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "map_score": 0.0,
        "mrr": 0.0,
        "latency": 2.024326,
        "retrieved_docs_count": 4,
        "last_retrieval_docs": 4,
        "graded_relevant_count": 0,
        "graded_total_docs": 4,
        "retrieval_calls": 1,
        "unique_sources": 1,
        "route": "vectorstore",
        "answerability": null,
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 3,
        "grader_fallback": false,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "unavailable",
        "ground_truth": false,
        "evaluation_note": "本次 Agent 未完成文档评分，Precision/Recall/MAP 不可用"
      },
      "error": null
    },
    {
      "index": 13,
      "question": "如何提升LLM面对对抗性攻击的鲁棒性？",
      "status": 200,
      "elapsed": 87.78037269499964,
      "cached": false,
      "answer_preview": "根据知识库文档，提升LLM面对对抗性攻击的鲁棒性可以从以下几个方面入手：\n\n1. **更新上下文示例**：\n   - **FIFO（先进先出）**：可以替换初始的手工制作的示例，但这样可能会导致生成的内容与原始训练数据不一致。\n   - **LIFO（后进先出）**：只替换最新的成功攻击示例，但这种方法在多样性和攻击效果上受到限制。\n   - **Scoring（评分）**：通过优先队列对示例进行",
      "sources_count": 1,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 0.5,
        "precision_at_5": 0.5,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 0.911393,
        "retrieved_docs_count": 2,
        "last_retrieval_docs": 2,
        "graded_relevant_count": 1,
        "graded_total_docs": 2,
        "retrieval_calls": 1,
        "unique_sources": 1,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 6,
        "grader_fallback": false,
        "generation_latency": 17.194315,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 14,
      "question": "AI Agent的工具使用能力是如何工作的？",
      "status": 200,
      "elapsed": 80.81074114100011,
      "cached": false,
      "answer_preview": "根据知识库文档，AI Agent的工具使用能力允许代理调用外部API以获取不在模型权重中的额外信息。这特别适用于访问当前信息、执行代码以及获取难以整合到预训练模型中的专有信息源。通过利用这些外部工具，代理可以增强其解决问题的能力并更有效地执行任务。",
      "sources_count": 1,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 0.3333333333333333,
        "precision_at_5": 0.2,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 1.153977,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 1,
        "graded_total_docs": 5,
        "retrieval_calls": 2,
        "unique_sources": 2,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 1,
        "tool_call_count": 9,
        "grader_fallback": false,
        "generation_latency": 3.810802,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 15,
      "question": "谁和 Ben Bloomberg 一起开发了 Harmoniser？",
      "status": 200,
      "elapsed": 87.61594482199962,
      "cached": false,
      "answer_preview": "Sorry, need more steps to process this request.",
      "sources_count": 5,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 1.0,
        "recall_at_1": 0.2,
        "recall_at_3": 0.6,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 3.296895,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 5,
        "graded_total_docs": 5,
        "retrieval_calls": 2,
        "unique_sources": 1,
        "route": "vectorstore",
        "answerability": "可回答性: 不足，需要更多信息",
        "web_search_calls": 0,
        "rewrite_calls": 1,
        "tool_call_count": 9,
        "grader_fallback": false,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 16,
      "question": "Behind the Tech 播客都有哪些嘉宾？",
      "status": 200,
      "elapsed": 85.72380676399916,
      "cached": false,
      "answer_preview": "Sorry, need more steps to process this request.",
      "sources_count": 5,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 1.0,
        "recall_at_1": 0.2,
        "recall_at_3": 0.6,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 2.373436,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 5,
        "graded_total_docs": 5,
        "retrieval_calls": 2,
        "unique_sources": 1,
        "route": "vectorstore",
        "answerability": "可回答性: 不足，需要更多信息",
        "web_search_calls": 0,
        "rewrite_calls": 1,
        "tool_call_count": 9,
        "grader_fallback": false,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 17,
      "question": "MIT 和微软之间有什么联系？",
      "status": 200,
      "elapsed": 78.81551971099998,
      "cached": false,
      "answer_preview": "根据知识库文档，MIT 和微软之间存在以下联系：\n\n1. **微软研究部门**：微软研究是微软的科研部门，也是世界上最具影响力的计算机科学研究机构之一。它在人工智能、机器学习等领域进行深入研究。文档中提到，Peter Lee 是微软研究部门的领导者之一，而 Kevin Scott 作为首席技术官，也与 Peter Lee 有着密切的联系。\n\n2. **学术合作**：Eric Horvitz 是微软",
      "sources_count": 5,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 0.0,
        "precision_at_3": 0.0,
        "precision_at_5": 0.0,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "map_score": 0.0,
        "mrr": 0.0,
        "latency": 0.825393,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 0,
        "graded_total_docs": 5,
        "retrieval_calls": 1,
        "unique_sources": 1,
        "route": "vectorstore",
        "answerability": null,
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 3,
        "grader_fallback": false,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "unavailable",
        "ground_truth": false,
        "evaluation_note": "本次 Agent 未完成文档评分，Precision/Recall/MAP 不可用"
      },
      "error": null
    },
    {
      "index": 18,
      "question": "Kevin Scott 和 Jacob Collier 有什么关系？",
      "status": 200,
      "elapsed": 89.286785753,
      "cached": false,
      "answer_preview": "根据知识库文档，Kevin Scott 和 Jacob Collier 之间的关系主要体现在以下几个方面：\n\n1. **Kevin Scott 是 TECH_EPISODE 的主持人**，而 Jacob Collier 是该节目的嘉宾之一。他们在节目中讨论了音乐、创新和讲故事等话题。\n2. Kevin Scott 与 Jacob Collier 有过多次互动，包括在 TECH_EPISODE 的采",
      "sources_count": 4,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 0.8,
        "recall_at_1": 0.25,
        "recall_at_3": 0.75,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 1.258591,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 4,
        "graded_total_docs": 5,
        "retrieval_calls": 1,
        "unique_sources": 1,
        "route": "vectorstore",
        "answerability": null,
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 4,
        "grader_fallback": false,
        "generation_latency": 0.0,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 19,
      "question": "Kevin Scott 是谁？",
      "status": 200,
      "elapsed": 94.23867455500022,
      "cached": false,
      "answer_preview": "根据知识库文档，Kevin Scott 是一位与创业公司相关的人士，他的公司获得了风险投资并筹集了5000万美元的战略和venture资金。他还位于一个机器车间的中间位置，并且使用Python作为默认编程语言，这可能是在Google工作的经历。此外，Kevin Scott 主持了Kevin Scott Podcast Ep-59这一期节目。由于信息中没有明确提到他是哪家公司的创始人或高管，我们无法",
      "sources_count": 5,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 1.0,
        "precision_at_5": 1.0,
        "recall_at_1": 0.2,
        "recall_at_3": 0.6,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 1.854795,
        "retrieved_docs_count": 5,
        "last_retrieval_docs": 5,
        "graded_relevant_count": 5,
        "graded_total_docs": 5,
        "retrieval_calls": 1,
        "unique_sources": 1,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 5,
        "grader_fallback": false,
        "generation_latency": 2.52623,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    },
    {
      "index": 20,
      "question": "BATCH-2026-11-18-V3 的负责人和部署区域是什么？",
      "status": 200,
      "elapsed": 96.39868788600052,
      "cached": false,
      "answer_preview": "根据知识库文档，BATCH-2026-11-18-V3的负责人是Bob，部署区域是kaggle-t4x2。",
      "sources_count": 1,
      "metrics_available": true,
      "metrics": {
        "precision_at_1": 1.0,
        "precision_at_3": 0.3333333333333333,
        "precision_at_5": 0.25,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "map_score": 1.0,
        "mrr": 1.0,
        "latency": 0.844262,
        "retrieved_docs_count": 4,
        "last_retrieval_docs": 4,
        "graded_relevant_count": 1,
        "graded_total_docs": 4,
        "retrieval_calls": 1,
        "unique_sources": 3,
        "route": "vectorstore",
        "answerability": "可回答性: 足够",
        "web_search_calls": 0,
        "rewrite_calls": 0,
        "tool_call_count": 6,
        "grader_fallback": false,
        "generation_latency": 0.893342,
        "hallucination_check": null,
        "metric_source": "agent_graded",
        "ground_truth": false,
        "evaluation_note": "相关标签来自 Agent 的 grade_documents，不是人工标注 ground truth"
      },
      "error": null
    }
  ]
}

"""