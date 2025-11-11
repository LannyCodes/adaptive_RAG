"""
检索效果评估模块
提供多种评估指标和方法，用于评估RAG系统中检索结果的质量
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    query: str
    retrieved_docs: List[Document]
    relevant_docs: List[Document]  # 真实相关的文档
    retrieval_time: float
    scores: Optional[List[float]] = None  # 检索分数


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    map_score: float  # 平均精度均值
    mrr: float  # 平均倒数排名
    ndcg_at_k: Dict[int, float]
    coverage: float  # 覆盖率
    diversity: float  # 多样性
    novelty: float  # 新颖性
    latency: float  # 平均检索延迟


class RetrievalEvaluator:
    """检索效果评估器"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化评估器
        
        Args:
            embedding_model: 用于计算语义相似度的嵌入模型
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def evaluate_retrieval(self, results: List[RetrievalResult], k_values: List[int] = [1, 3, 5, 10]) -> EvaluationMetrics:
        """
        评估检索结果
        
        Args:
            results: 检索结果列表
            k_values: 要计算的k值列表
            
        Returns:
            评估指标
        """
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        ndcg_at_k = {}
        
        total_precision = {k: 0 for k in k_values}
        total_recall = {k: 0 for k in k_values}
        total_f1 = {k: 0 for k in k_values}
        total_ndcg = {k: 0 for k in k_values}
        
        all_precisions = []
        all_reciprocal_ranks = []
        all_latencies = []
        
        for result in results:
            query = result.query
            retrieved_docs = result.retrieved_docs
            relevant_docs = result.relevant_docs
            retrieval_time = result.retrieval_time
            
            all_latencies.append(retrieval_time)
            
            # 获取相关文档的ID或内容
            relevant_ids = set()
            for doc in relevant_docs:
                # 使用文档内容作为ID，实际应用中可以使用文档ID
                doc_id = doc.page_content[:50]  # 使用前50个字符作为ID
                relevant_ids.add(doc_id)
            
            # 计算每个k值的指标
            for k in k_values:
                retrieved_k = retrieved_docs[:k]
                retrieved_k_ids = set()
                
                for doc in retrieved_k:
                    doc_id = doc.page_content[:50]
                    retrieved_k_ids.add(doc_id)
                
                # 计算交集
                intersection = len(relevant_ids.intersection(retrieved_k_ids))
                
                # 计算Precision@K
                precision_k = intersection / k if k > 0 else 0
                total_precision[k] += precision_k
                
                # 计算Recall@K
                recall_k = intersection / len(relevant_ids) if len(relevant_ids) > 0 else 0
                total_recall[k] += recall_k
                
                # 计算F1@K
                if precision_k + recall_k > 0:
                    f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
                else:
                    f1_k = 0
                total_f1[k] += f1_k
                
                # 计算NDCG@K
                if result.scores:
                    # 创建相关性分数 (1表示相关，0表示不相关)
                    relevance_scores = []
                    for doc in retrieved_k:
                        doc_id = doc.page_content[:50]
                        relevance = 1 if doc_id in relevant_ids else 0
                        relevance_scores.append(relevance)
                    
                    # 理想排序 (所有相关文档排在前面)
                    ideal_relevance = sorted(relevance_scores, reverse=True)
                    
                    # 计算NDCG
                    if len(relevance_scores) > 1 and sum(ideal_relevance) > 0:
                        try:
                            ndcg_k = ndcg_score([ideal_relevance], [relevance_scores], k=k)
                            total_ndcg[k] += ndcg_k
                        except:
                            # 如果计算失败，使用简化的NDCG计算
                            dcg = 0
                            idcg = 0
                            for i, rel in enumerate(relevance_scores):
                                dcg += rel / np.log2(i + 2) if rel > 0 else 0
                            for i, rel in enumerate(ideal_relevance):
                                idcg += rel / np.log2(i + 2) if rel > 0 else 0
                            ndcg_k = dcg / idcg if idcg > 0 else 0
                            total_ndcg[k] += ndcg_k
                    else:
                        total_ndcg[k] += 1.0  # 如果没有相关文档或只有一个文档，NDCG为1
            
            # 计算平均精度 (AP)
            precisions = []
            for i, doc in enumerate(retrieved_docs):
                doc_id = doc.page_content[:50]
                if doc_id in relevant_ids:
                    precision_at_i = len(relevant_ids.intersection(set(
                        d.page_content[:50] for d in retrieved_docs[:i+1]
                    ))) / (i + 1)
                    precisions.append(precision_at_i)
            
            ap = sum(precisions) / len(relevant_ids) if precisions else 0
            all_precisions.append(ap)
            
            # 计算倒数排名 (RR)
            for i, doc in enumerate(retrieved_docs):
                doc_id = doc.page_content[:50]
                if doc_id in relevant_ids:
                    rr = 1 / (i + 1)
                    all_reciprocal_ranks.append(rr)
                    break
            else:
                all_reciprocal_ranks.append(0)
        
        # 计算平均指标
        num_results = len(results)
        for k in k_values:
            precision_at_k[k] = total_precision[k] / num_results
            recall_at_k[k] = total_recall[k] / num_results
            f1_at_k[k] = total_f1[k] / num_results
            ndcg_at_k[k] = total_ndcg[k] / num_results
        
        map_score = sum(all_precisions) / num_results if all_precisions else 0
        mrr = sum(all_reciprocal_ranks) / num_results if all_reciprocal_ranks else 0
        latency = sum(all_latencies) / num_results if all_latencies else 0
        
        # 计算覆盖率、多样性和新颖性
        coverage = self._calculate_coverage(results)
        diversity = self._calculate_diversity(results)
        novelty = self._calculate_novelty(results)
        
        return EvaluationMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            map_score=map_score,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            coverage=coverage,
            diversity=diversity,
            novelty=novelty,
            latency=latency
        )
    
    def _calculate_coverage(self, results: List[RetrievalResult]) -> float:
        """计算覆盖率 - 检索到的唯一文档数与总文档数的比例"""
        all_retrieved = set()
        all_relevant = set()
        
        for result in results:
            for doc in result.retrieved_docs:
                doc_id = doc.page_content[:50]
                all_retrieved.add(doc_id)
            
            for doc in result.relevant_docs:
                doc_id = doc.page_content[:50]
                all_relevant.add(doc_id)
        
        coverage = len(all_retrieved) / len(all_relevant) if all_relevant else 0
        return coverage
    
    def _calculate_diversity(self, results: List[RetrievalResult]) -> float:
        """计算多样性 - 检索结果之间的平均语义差异"""
        all_similarities = []
        
        for result in results:
            if len(result.retrieved_docs) < 2:
                continue
                
            # 获取文档嵌入
            doc_texts = [doc.page_content for doc in result.retrieved_docs]
            embeddings = self.embedding_model.encode(doc_texts, convert_to_tensor=True)
            
            # 计算文档之间的余弦相似度
            cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
            
            # 获取上三角矩阵（排除对角线）
            upper_triangle_indices = torch.triu_indices(len(cos_sim), len(cos_sim), offset=1)
            similarities = cos_sim[upper_triangle_indices[0], upper_triangle_indices[1]]
            
            # 多样性 = 1 - 平均相似度
            diversity = 1 - similarities.mean().item()
            all_similarities.append(diversity)
        
        return sum(all_similarities) / len(all_similarities) if all_similarities else 0
    
    def _calculate_novelty(self, results: List[RetrievalResult]) -> float:
        """计算新颖性 - 检索结果中不重复内容的比例"""
        total_docs = 0
        unique_docs = set()
        
        for result in results:
            for doc in result.retrieved_docs:
                total_docs += 1
                doc_id = doc.page_content[:50]
                unique_docs.add(doc_id)
        
        novelty = len(unique_docs) / total_docs if total_docs > 0 else 0
        return novelty
    
    def compare_retrievers(self, retriever_results: Dict[str, List[RetrievalResult]], 
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, EvaluationMetrics]:
        """
        比较多个检索器的性能
        
        Args:
            retriever_results: 检索器名称到检索结果的映射
            k_values: 要计算的k值列表
            
        Returns:
            检索器名称到评估指标的映射
        """
        metrics = {}
        
        for name, results in retriever_results.items():
            print(f"评估检索器: {name}")
            metrics[name] = self.evaluate_retrieval(results, k_values)
        
        return metrics
    
    def generate_report(self, metrics: Dict[str, EvaluationMetrics], 
                        save_path: Optional[str] = None) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 检索器名称到评估指标的映射
            save_path: 报告保存路径
            
        Returns:
            报告文本
        """
        report = []
        report.append("# 检索效果评估报告\n")
        
        # 创建比较表
        df_data = []
        for name, metric in metrics.items():
            row = {"检索器": name}
            row.update({
                f"Precision@{k}": f"{metric.precision_at_k[k]:.4f}" 
                for k in sorted(metric.precision_at_k.keys())
            })
            row.update({
                f"Recall@{k}": f"{metric.recall_at_k[k]:.4f}" 
                for k in sorted(metric.recall_at_k.keys())
            })
            row.update({
                f"F1@{k}": f"{metric.f1_at_k[k]:.4f}" 
                for k in sorted(metric.f1_at_k.keys())
            })
            row.update({
                f"NDCG@{k}": f"{metric.ndcg_at_k[k]:.4f}" 
                for k in sorted(metric.ndcg_at_k.keys())
            })
            row.update({
                "MAP": f"{metric.map_score:.4f}",
                "MRR": f"{metric.mrr:.4f}",
                "覆盖率": f"{metric.coverage:.4f}",
                "多样性": f"{metric.diversity:.4f}",
                "新颖性": f"{metric.novelty:.4f}",
                "延迟(ms)": f"{metric.latency*1000:.2f}"
            })
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        report.append("## 指标比较表\n")
        report.append(df.to_string(index=False))
        report.append("\n\n")
        
        # 添加指标解释
        report.append("## 指标解释\n")
        report.append("- **Precision@K**: 前K个结果中相关文档的比例\n")
        report.append("- **Recall@K**: 前K个结果中相关文档占所有相关文档的比例\n")
        report.append("- **F1@K**: Precision和Recall的调和平均数\n")
        report.append("- **NDCG@K**: 归一化折扣累积增益，考虑排序位置\n")
        report.append("- **MAP**: 平均精度均值，所有查询的平均精度\n")
        report.append("- **MRR**: 平均倒数排名，第一个相关文档排名的倒数平均值\n")
        report.append("- **覆盖率**: 检索到的唯一文档数与总文档数的比例\n")
        report.append("- **多样性**: 检索结果之间的平均语义差异\n")
        report.append("- **新颖性**: 检索结果中不重复内容的比例\n")
        report.append("- **延迟**: 平均检索时间\n")
        
        # 添加最佳检索器
        report.append("## 最佳检索器\n")
        
        # 找出每个指标的最佳检索器
        best_metrics = {}
        for metric_name in ["precision_at_5", "recall_at_5", "f1_at_5", "ndcg_at_5", "map_score", "mrr"]:
            best_name = max(metrics.keys(), key=lambda x: getattr(metrics[x], metric_name))
            best_metrics[metric_name] = best_name
            report.append(f"- **{metric_name}**: {best_name}\n")
        
        report_text = "".join(report)
        
        # 保存报告
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"报告已保存到: {save_path}")
        
        return report_text
    
    def plot_metrics_comparison(self, metrics: Dict[str, EvaluationMetrics], 
                              save_path: Optional[str] = None):
        """
        绘制指标比较图
        
        Args:
            metrics: 检索器名称到评估指标的映射
            save_path: 图表保存路径
        """
        # 准备数据
        retriever_names = list(metrics.keys())
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("检索器性能比较", fontsize=16)
        
        # Precision@K
        ax = axes[0, 0]
        k_values = sorted(list(metrics[retriever_names[0]].precision_at_k.keys()))
        for name in retriever_names:
            precision_values = [metrics[name].precision_at_k[k] for k in k_values]
            ax.plot(k_values, precision_values, marker='o', label=name)
        ax.set_title("Precision@K")
        ax.set_xlabel("K")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(True)
        
        # Recall@K
        ax = axes[0, 1]
        for name in retriever_names:
            recall_values = [metrics[name].recall_at_k[k] for k in k_values]
            ax.plot(k_values, recall_values, marker='o', label=name)
        ax.set_title("Recall@K")
        ax.set_xlabel("K")
        ax.set_ylabel("Recall")
        ax.legend()
        ax.grid(True)
        
        # F1@K
        ax = axes[0, 2]
        for name in retriever_names:
            f1_values = [metrics[name].f1_at_k[k] for k in k_values]
            ax.plot(k_values, f1_values, marker='o', label=name)
        ax.set_title("F1@K")
        ax.set_xlabel("K")
        ax.set_ylabel("F1")
        ax.legend()
        ax.grid(True)
        
        # NDCG@K
        ax = axes[1, 0]
        for name in retriever_names:
            ndcg_values = [metrics[name].ndcg_at_k[k] for k in k_values]
            ax.plot(k_values, ndcg_values, marker='o', label=name)
        ax.set_title("NDCG@K")
        ax.set_xlabel("K")
        ax.set_ylabel("NDCG")
        ax.legend()
        ax.grid(True)
        
        # MAP和MRR
        ax = axes[1, 1]
        map_values = [metrics[name].map_score for name in retriever_names]
        mrr_values = [metrics[name].mrr for name in retriever_names]
        x = np.arange(len(retriever_names))
        width = 0.35
        ax.bar(x - width/2, map_values, width, label='MAP')
        ax.bar(x + width/2, mrr_values, width, label='MRR')
        ax.set_title("MAP和MRR")
        ax.set_xticks(x)
        ax.set_xticklabels(retriever_names)
        ax.legend()
        ax.grid(True)
        
        # 其他指标
        ax = axes[1, 2]
        other_metrics = ['coverage', 'diversity', 'novelty']
        metric_values = {metric: [] for metric in other_metrics}
        for name in retriever_names:
            for metric in other_metrics:
                metric_values[metric].append(getattr(metrics[name], metric))
        
        x = np.arange(len(retriever_names))
        width = 0.25
        for i, metric in enumerate(other_metrics):
            ax.bar(x + i*width, metric_values[metric], width, label=metric)
        ax.set_title("其他指标")
        ax.set_xticks(x + width)
        ax.set_xticklabels(retriever_names)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()


class RetrievalTestSet:
    """检索测试集"""
    
    def __init__(self, queries_file: str, documents_file: str, qrels_file: str):
        """
        初始化测试集
        
        Args:
            queries_file: 查询文件路径，每行一个查询
            documents_file: 文档文件路径，每行一个文档
            qrels_file: 相关性标注文件路径，格式为: query_id,doc_id,relevance
        """
        self.queries = self._load_queries(queries_file)
        self.documents = self._load_documents(documents_file)
        self.qrels = self._load_qrels(qrels_file)
    
    def _load_queries(self, file_path: str) -> Dict[str, str]:
        """加载查询"""
        queries = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                queries[str(i)] = line.strip()
        return queries
    
    def _load_documents(self, file_path: str) -> Dict[str, Document]:
        """加载文档"""
        documents = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                doc = Document(page_content=line.strip(), metadata={"doc_id": str(i)})
                documents[str(i)] = doc
        return documents
    
    def _load_qrels(self, file_path: str) -> Dict[str, Dict[str, int]]:
        """加载相关性标注"""
        qrels = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    query_id, doc_id, relevance = parts[0], parts[1], int(parts[2])
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = relevance
        return qrels
    
    def get_retrieval_results(self, retriever, top_k: int = 10) -> List[RetrievalResult]:
        """
        使用指定检索器获取检索结果
        
        Args:
            retriever: 检索器，需要有一个retrieve(query, top_k)方法
            top_k: 返回的文档数量
            
        Returns:
            检索结果列表
        """
        results = []
        
        for query_id, query_text in self.queries.items():
            start_time = time.time()
            retrieved_docs = retriever.retrieve(query_text, top_k)
            retrieval_time = time.time() - start_time
            
            # 获取相关文档
            relevant_docs = []
            if query_id in self.qrels:
                for doc_id, relevance in self.qrels[query_id].items():
                    if relevance > 0 and doc_id in self.documents:
                        relevant_docs.append(self.documents[doc_id])
            
            result = RetrievalResult(
                query=query_text,
                retrieved_docs=retrieved_docs,
                relevant_docs=relevant_docs,
                retrieval_time=retrieval_time
            )
            results.append(result)
        
        return results


def create_sample_test_set():
    """创建示例测试集"""
    # 创建示例查询
    queries = [
        "什么是机器学习？",
        "深度学习和机器学习的区别是什么？",
        "如何评估机器学习模型的性能？",
        "自然语言处理有哪些应用？",
        "计算机视觉的基本任务是什么？"
    ]
    
    # 创建示例文档
    documents = [
        "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。",
        "评估机器学习模型的常用指标包括准确率、精确率、召回率和F1分数。",
        "自然语言处理是计算机科学和人工智能的一个分支，专注于计算机与人类语言之间的交互。",
        "计算机视觉是人工智能的一个领域，训练计算机解释和理解视觉世界。",
        "强化学习是机器学习的一个类型，它关注软件代理应该如何在环境中采取行动以最大化累积奖励。",
        "数据预处理是机器学习流程中的重要步骤，包括数据清洗、特征选择和特征工程。",
        "过拟合是机器学习中的一个常见问题，指模型在训练数据上表现良好但在新数据上表现不佳。",
        "卷积神经网络（CNN）是一类深度神经网络，最常用于分析视觉图像。",
        "循环神经网络（RNN）是一类人工神经网络，其中节点之间的连接形成有向图沿时间序列。"
    ]
    
    # 创建相关性标注
    qrels = {
        "0": {"0": 2, "1": 1, "6": 1, "7": 1},  # 什么是机器学习？
        "1": {"0": 1, "1": 2, "8": 1, "9": 1},  # 深度学习和机器学习的区别
        "2": {"2": 2, "7": 1},  # 如何评估机器学习模型的性能
        "3": {"3": 2, "9": 1},  # 自然语言处理的应用
        "4": {"4": 2, "8": 1}   # 计算机视觉的基本任务
    }
    
    # 保存文件
    with open("sample_queries.txt", "w", encoding="utf-8") as f:
        for query in queries:
            f.write(query + "\n")
    
    with open("sample_documents.txt", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")
    
    with open("sample_qrels.csv", "w", encoding="utf-8") as f:
        for query_id, doc_relevance in qrels.items():
            for doc_id, relevance in doc_relevance.items():
                f.write(f"{query_id},{doc_id},{relevance}\n")
    
    print("示例测试集已创建:")
    print("- sample_queries.txt: 查询文件")
    print("- sample_documents.txt: 文档文件")
    print("- sample_qrels.csv: 相关性标注文件")
    
    return RetrievalTestSet("sample_queries.txt", "sample_documents.txt", "sample_qrels.csv")


if __name__ == "__main__":
    # 创建示例测试集
    test_set = create_sample_test_set()
    
    # 创建评估器
    evaluator = RetrievalEvaluator()
    
    # 这里应该使用您的实际检索器
    # 以下是一个模拟的检索器，用于演示
    class MockRetriever:
        def __init__(self, name):
            self.name = name
        
        def retrieve(self, query, top_k=10):
            # 模拟检索结果
            import random
            all_docs = list(test_set.documents.values())
            # 模拟不同质量的检索器
            if self.name == "good":
                # 好的检索器：有更高概率返回相关文档
                relevant_docs = [doc for doc in all_docs if any(keyword in doc.page_content.lower() 
                                for keyword in query.lower().split()[:2])]
                if relevant_docs:
                    results = relevant_docs[:min(top_k//2, len(relevant_docs))]
                    results += random.sample(all_docs, min(top_k-len(results), len(all_docs)))
                else:
                    results = random.sample(all_docs, min(top_k, len(all_docs)))
            elif self.name == "medium":
                # 中等检索器
                relevant_docs = [doc for doc in all_docs if any(keyword in doc.page_content.lower() 
                                for keyword in [query.lower().split()[0]])]
                if relevant_docs:
                    results = relevant_docs[:min(top_k//3, len(relevant_docs))]
                    results += random.sample(all_docs, min(top_k-len(results), len(all_docs)))
                else:
                    results = random.sample(all_docs, min(top_k, len(all_docs)))
            else:
                # 差的检索器：随机返回
                results = random.sample(all_docs, min(top_k, len(all_docs)))
            
            return results
    
    # 创建不同质量的检索器
    good_retriever = MockRetriever("good")
    medium_retriever = MockRetriever("medium")
    poor_retriever = MockRetriever("poor")
    
    # 获取检索结果
    good_results = test_set.get_retrieval_results(good_retriever)
    medium_results = test_set.get_retrieval_results(medium_retriever)
    poor_results = test_set.get_retrieval_results(poor_retriever)
    
    # 比较检索器
    retriever_results = {
        "好的检索器": good_results,
        "中等检索器": medium_results,
        "差的检索器": poor_results
    }
    
    # 评估检索器
    metrics = evaluator.compare_retrievers(retriever_results)
    
    # 生成报告
    report = evaluator.generate_report(metrics, "retrieval_evaluation_report.md")
    print(report)
    
    # 绘制比较图
    evaluator.plot_metrics_comparison(metrics, "retrieval_evaluation_comparison.png")