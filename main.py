"""
主应用程序入口
集成所有模块，构建工作流并运行自适应RAG系统
"""

import time
from langgraph.graph import END, StateGraph, START
from pprint import pprint

from config import setup_environment, validate_api_keys, ENABLE_GRAPHRAG
from document_processor import initialize_document_processor
from routers_and_graders import initialize_graders_and_router
from workflow_nodes import WorkflowNodes, GraphState

# 添加 LangSmith 集成
from langsmith_integration import setup_langsmith
from langsmith_integration import (
    AlertLevel,
    AlertRule
)
from typing import Optional
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from knowledge_graph import initialize_knowledge_graph, initialize_community_summarizer
    from graph_retriever import initialize_graph_retriever
except ImportError:
    print("⚠️ 无法导入知识图谱模块，GraphRAG功能将不可用")
    ENABLE_GRAPHRAG = False


class AdaptiveRAGSystem:
    """自适应RAG系统主类"""
    
    def __init__(self):
        print("初始化自适应RAG系统...")
        
        # 设置 LangSmith 追踪和性能监控
        print("设置 LangSmith 追踪...")
        self.langsmith_manager = setup_langsmith(
            project_name="adaptive-rag-project",
            enable_performance_monitoring=True,
            enable_alerts=True
        )
        
        # 初始化告警回调函数
        self._setup_alert_callbacks()
        
        # 设置环境和验证API密钥
        try:
            setup_environment()
            validate_api_keys()  # 验证API密钥是否正确设置
            print("✅ API密钥验证成功")
        except ValueError as e:
            print(f"❌ {e}")
            raise
        
        from config import LLM_BACKEND
        if LLM_BACKEND == "ollama":
            print("🔍 检查 Ollama 服务状态...")
            if not self._check_ollama_service():
                print("\n" + "="*60)
                print("❌ Ollama 服务未启动！")
                print("="*60)
                print("\n请先启动 Ollama 服务：")
                print("\n方法1: 在终端运行")
                print("  $ ollama serve")
                print("\n方法2: 在 Kaggle Notebook 中运行")
                print("  import subprocess")
                print("  subprocess.Popen(['ollama', 'serve'])")
                print("\n方法3: 使用快捷脚本")
                print("  %run KAGGLE_LOAD_OLLAMA.py")
                print("="*60)
                raise ConnectionError("Ollama 服务未运行，请先启动服务")
            print("✅ Ollama 服务运行正常")
        
        # 初始化文档处理器
        print("设置文档处理器...")
        self.doc_processor, self.vectorstore, self.retriever, self.doc_splits = initialize_document_processor()
        
        # 初始化评分器和路由器
        print("初始化评分器和路由器...")
        self.graders = initialize_graders_and_router()
        
        # 初始化知识图谱 (如果启用)
        self.graph_retriever = None
        if ENABLE_GRAPHRAG:
            print("初始化 GraphRAG...")
            try:
                kg = initialize_knowledge_graph()
                # 尝试加载已有的图谱数据
                try:
                    kg.load_from_file("knowledge_graph.json")
                except FileNotFoundError:
                    print("   未找到 existing knowledge_graph.json, 将使用空图谱")
                
                self.graph_retriever = initialize_graph_retriever(kg)
                print("✅ GraphRAG 初始化成功")
            except Exception as e:
                print(f"⚠️ GraphRAG 初始化失败: {e}")
        
        # 初始化工作流节点
        print("设置工作流节点...")
        # WorkflowNodes 将在 _build_workflow 中初始化
        
        # 构建工作流
        print("构建工作流图...")
        self.app = self._build_workflow()
        
        print("✅ 自适应RAG系统初始化完成！")
    
    def _setup_alert_callbacks(self):
        """设置告警回调函数"""
        def alert_callback(rule, metric_value):
            """默认告警回调：记录到控制台"""
            print(f"\n🔔 [告警通知] {rule.name}\n"
                  f"   级别: {rule.level.value}\n"
                  f"   指标: {rule.metric_name}\n"
                  f"   当前值: {metric_value:.2f}\n"
                  f"   阈值: {rule.operator} {rule.threshold}")
        
        self.langsmith_manager.add_alert_callback(alert_callback)
    
    def _check_ollama_service(self) -> bool:
        """检查 Ollama 服务是否运行"""
        import requests
        try:
            # 尝试连接 Ollama API
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False
    
    def _build_workflow(self):
        """构建工作流图"""
        # 创建工作流节点实例，传递DocumentProcessor实例和retriever
        self.workflow_nodes = WorkflowNodes(
            doc_processor=self.doc_processor,
            graders=self.graders,
            retriever=self.retriever
        )
        
        workflow = StateGraph(GraphState)
        
        # 定义节点
        workflow.add_node("web_search", self.workflow_nodes.web_search)
        workflow.add_node("retrieve", self.workflow_nodes.retrieve)
        workflow.add_node("grade_documents", self.workflow_nodes.grade_documents)
        workflow.add_node("generate", self.workflow_nodes.generate)
        workflow.add_node("transform_query", self.workflow_nodes.transform_query)
        workflow.add_node("decompose_query", self.workflow_nodes.decompose_query)
        workflow.add_node("prepare_next_query", self.workflow_nodes.prepare_next_query)
        
        # 构建图
        workflow.add_conditional_edges(
            START,
            self.workflow_nodes.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "decompose_query", # 向量检索前先进行查询分解
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("decompose_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.workflow_nodes.decide_to_generate,
            {
                "transform_query": "transform_query",
                "prepare_next_query": "prepare_next_query",
                "generate": "generate",
                "web_search": "web_search", # 添加 web_search 作为回退选项
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("prepare_next_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.workflow_nodes.grade_generation_v_documents_and_question,
            {
                "not supported": "transform_query",  # 修复：有幻觉时重新转换查询，而不是再次生成
                "useful": END,
                "not useful": "transform_query",
            },
        )
        
        # 编译（设置递归限制以防止无限循环）
        # 获取 LangSmith 回调配置
        callback_config = self.langsmith_manager.get_callback_config()
        
        return workflow.compile(
            checkpointer=None,
            interrupt_before=None,
            interrupt_after=None,
            debug=False,
            **callback_config  # 添加 LangSmith 回调
        )
    
    async def query(self, question: str, verbose: bool = True):
        """
        处理查询 (异步版本)
        
        Args:
            question (str): 用户问题
            verbose (bool): 是否显示详细输出
            
        Returns:
            dict: 包含最终答案和评估指标的字典
        """
        import asyncio
        from datetime import datetime
        
        print(f"\n🔍 处理问题: {question}")
        print("=" * 50)
        
        # 记录查询开始时间
        query_start_time = datetime.now()
        
        inputs = {"question": question, "retry_count": 0}  # 初始化重试计数器
        final_generation = None
        retrieval_metrics = None
        routing_decision = "unknown"
        
        # 设置配置，增加递归限制
        config = {"recursion_limit": 50}  # 增加到 50，默认是 25
        
        print("\n🤖 思考过程:")
        async for output in self.app.astream(inputs, config=config):
            for key, value in output.items():
                if verbose:
                    # 简单的节点执行提示，模拟流式感
                    print(f"  ↳ 执行节点: {key}...", end="\r")
                    # 异步暂停
                    await asyncio.sleep(0.1) 
                    print(f"  ✅ 完成节点: {key}      ")
                    
                # 记录路由决策
                if key == "start":
                    routing_decision = value.get("next_node", "unknown")
                
                final_generation = value.get("generation", final_generation)
                # 保存检索评估指标
                if "retrieval_metrics" in value:
                    retrieval_metrics = value["retrieval_metrics"]
                    
                    # 使用 LangSmith 记录检索事件
                    if hasattr(self, 'langsmith_manager') and self.langsmith_manager.enable_performance_monitoring:
                        self.langsmith_manager.log_retrieval_event(
                            query=question,
                            documents_count=retrieval_metrics.get('retrieved_docs_count', 0),
                            retrieval_time=retrieval_metrics.get('latency', 0) * 1000,  # 转换为毫秒
                            top_k=3
                        )
                
                # 记录生成事件
                if key == "generate":
                    generation = value.get("generation", "")
                    if generation and hasattr(self, 'langsmith_manager') and self.langsmith_manager.enable_performance_monitoring:
                        # 估算token使用量（中文约2字符=1token，英文约4字符=1token）
                        estimated_tokens = len(generation) // 2
                        
                        self.langsmith_manager.log_generation_event(
                            prompt=question,
                            generation=generation,
                            generation_time=0,  # 生成时间已在generate节点中处理
                            tokens_used=estimated_tokens
                        )
        
        print("\n" + "=" * 50)
        print("🎯 最终答案:")
        print("-" * 30)
        
        # 模拟流式输出效果 (打字机效果)
        if final_generation:
            import sys
            for char in final_generation:
                sys.stdout.write(char)
                sys.stdout.flush()
                # 异步暂停
                await asyncio.sleep(0.01) # 控制打字速度
            print() # 换行
        else:
            print("未生成答案")
            
        print("=" * 50)
        
        # 计算总查询时间并记录到 LangSmith
        query_end_time = datetime.now()
        total_latency = (query_end_time - query_start_time).total_seconds() * 1000  # 毫秒
        
        if hasattr(self, 'langsmith_manager') and self.langsmith_manager.enable_performance_monitoring:
            self.langsmith_manager.log_query_complete(
                question=question,
                answer=final_generation or "",
                total_latency=total_latency,
                routing_decision=routing_decision,
                metrics=retrieval_metrics
            )
        
        # 返回包含答案和评估指标的字典
        return {
            "answer": final_generation,
            "retrieval_metrics": retrieval_metrics
        }
    
    def interactive_mode(self):
        """交互模式，允许用户持续提问"""
        import asyncio
        print("\n🤖 欢迎使用自适应RAG系统!")
        print("💡 输入问题开始对话，输入 'quit' 或 'exit' 退出")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n❓ 请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 感谢使用，再见!")
                    break
                
                if not question:
                    print("⚠️  请输入一个有效的问题")
                    continue
                
                # 使用 asyncio.run 执行异步查询
                result = asyncio.run(self.query(question))
                
                # 显示检索评估摘要
                if result.get("retrieval_metrics"):
                    metrics = result["retrieval_metrics"]
                    print("\n📊 检索评估摘要:")
                    print(f"   - 检索耗时: {metrics.get('latency', 0):.4f}秒")
                    print(f"   - 检索文档数: {metrics.get('retrieved_docs_count', 0)}")
                    print(f"   - Precision@3: {metrics.get('precision_at_3', 0):.4f}")
                    print(f"   - Recall@3: {metrics.get('recall_at_3', 0):.4f}")
                    print(f"   - MAP: {metrics.get('map_score', 0):.4f}")
                
            except KeyboardInterrupt:
                print("\n👋 感谢使用，再见!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()
                print("请重试或输入 'quit' 退出")


def main():
    """主函数"""
    import asyncio
    try:
        # 初始化系统
        rag_system: AdaptiveRAGSystem = AdaptiveRAGSystem()
        
        # 测试查询 - 基于Lilian Weng的三篇博客生成的10个问题
        test_questions = [
            "AI Agent的四个核心组成部分是什么？",
            "什么是Chain-of-Thought (CoT) 提示技术？",
            "大语言模型面临哪些类型的对抗攻击？",
            "AI Agent中的记忆系统分为哪两种类型？",
            "如何通过提示工程来引导LLM的行为？",
            "对抗性攻击如何影响大语言模型的安全性？",
            "AI Agent的任务规划能力包括哪些方面？",
            "什么是提示工程中的上下文提示？",
            "如何提升LLM面对对抗性攻击的鲁棒性？",
            "AI Agent的工具使用能力是如何工作的？"
        ]
        
        # 检查GPU使用情况
        print("\n🔍 检查硬件加速配置...")
        print("=" * 60)
        
        # 检查CUDA/GPU
        try:
            import torch
            print(f"PyTorch版本: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA可用: ✅")
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print(f"CUDA可用: ❌")
                print(f"使用设备: CPU")
        except ImportError:
            print("⚠️ 未安装PyTorch，无法检测GPU")
        
        # 检查向量数据库配置
        print("\n📊 向量数据库配置:")
        try:
            from config import VECTOR_STORE_TYPE, MILVUS_URI, MILVUS_HOST, MILVUS_PORT
            print(f"向量数据库类型: {VECTOR_STORE_TYPE}")
            
            if VECTOR_STORE_TYPE == "milvus":
                print(f"Milvus URI: {MILVUS_URI if MILVUS_URI else f'{MILVUS_HOST}:{MILVUS_PORT}'}")
                
                # 检查 Milvus 连接状态
                try:
                    from pymilvus import connections, utility
                    
                    # 检查是否已连接
                    if connections.has_connection("default"):
                        print(f"Milvus 连接状态: ✅ 已连接")
                        
                        # 检查集合信息
                        if utility.has_collection("rag_milvus", using="default"):
                            print(f"Milvus 集合: rag_milvus ✅")
                        else:
                            print(f"Milvus 集合: rag_milvus ❌ (不存在)")
                    else:
                        print(f"Milvus 连接状态: ⚠️ 未连接 (将在查询时连接)")
                        
                except ImportError:
                    print("⚠️ 未安装 pymilvus，无法检测 Milvus 状态")
                except Exception as e:
                    print(f"⚠️ Milvus 状态检测失败: {e}")
            else:
                print(f"⚠️ 未知的向量数据库类型: {VECTOR_STORE_TYPE}")
                
        except ImportError:
            print("⚠️ 无法导入配置，跳过向量数据库检测")
        
        print("=" * 60)
        
        # 测试异步检索性能 - 使用真正的并发执行
        print("\n🚀 开始测试异步检索性能（并发执行）")
        print("=" * 60)
        print(f"测试问题数量: {len(test_questions)}")
        print("=" * 60)
        
        import time
        start_time = time.time()
        
        # 使用 asyncio.gather 实现真正的并发执行
        async def run_concurrent_queries():
            """并发执行所有查询"""
            tasks = []
            for idx, test_question in enumerate(test_questions, 1):
                # 为每个查询创建任务
                task = asyncio.create_task(
                    rag_system.query(test_question, verbose=False)
                )
                tasks.append((idx, test_question, task))
            
            # 并发执行所有任务
            results = []
            for idx, test_question, task in tasks:
                print(f"\n{'='*60}")
                print(f"查询 {idx}/{len(test_questions)}: {test_question[:50]}...")
                print(f"{'='*60}")
                
                try:
                    result = await task
                    query_time = time.time() - start_time  # 相对于开始时间
                    results.append({
                        "question": test_question,
                        "time": query_time,
                        "metrics": result.get("retrieval_metrics")
                    })
                    print(f"✅ 完成 - 耗时: {query_time:.4f}秒")
                except Exception as e:
                    print(f"❌ 失败: {e}")
                    results.append({
                        "question": test_question,
                        "time": 0,
                        "error": str(e),
                        "metrics": None
                    })
            
            return results
        
        # 运行并发查询
        results = asyncio.run(run_concurrent_queries())
        
        total_time = time.time() - start_time
        
        # 显示性能测试摘要
        print("\n" + "=" * 60)
        print("📊 异步检索性能测试摘要（并发执行）")
        print("=" * 60)
        print(f"总查询数: {len(test_questions)}")
        print(f"总耗时: {total_time:.4f}秒")
        print(f"平均耗时: {total_time/len(test_questions):.4f}秒")
        print(f"最快查询: {min(r['time'] for r in results if r['time'] > 0):.4f}秒")
        print(f"最慢查询: {max(r['time'] for r in results if r['time'] > 0):.4f}秒")
        
        # 计算并发效率
        if len(test_questions) > 1:
            # 如果是串行执行，总时间应该是所有查询时间的总和
            serial_time = sum(r['time'] for r in results if r['time'] > 0)
            efficiency = (serial_time / total_time) * 100 if total_time > 0 else 0
            print(f"并发效率: {efficiency:.1f}% (相比串行执行)")
        print("=" * 60)
        
        # 显示每个查询的详细指标
        print("\n📋 各查询详细指标:")
        print("-" * 60)
        for idx, result in enumerate(results, 1):
            print(f"\n查询 {idx}: {result['question'][:50]}...")
            print(f"  耗时: {result['time']:.4f}秒")
            if result.get('metrics'):
                metrics = result['metrics']
                print(f"  检索文档数: {metrics.get('retrieved_docs_count', 0)}")
                print(f"  Precision@3: {metrics.get('precision_at_3', 0):.4f}")
                print(f"  Recall@3: {metrics.get('recall_at_3', 0):.4f}")
                print(f"  MAP: {metrics.get('map_score', 0):.4f}")
        
        # 生成并显示 LangSmith 性能报告
        print("\n" + "=" * 60)
        print("📈 LangSmith 性能报告")
        print("=" * 60)
        
        if hasattr(rag_system, 'langsmith_manager'):
            # 获取性能报告
            performance_report = rag_system.langsmith_manager.get_performance_report(hours=24)
            
            if "summary" in performance_report:
                summary = performance_report["summary"]
                print(f"📊 查询统计 (过去24小时):")
                print(f"   总查询数: {summary.get('total_queries', 0)}")
                print(f"   平均延迟: {summary.get('average_latency_ms', 0):.2f}ms")
                print(f"   最小延迟: {summary.get('min_latency_ms', 0):.2f}ms")
                print(f"   最大延迟: {summary.get('max_latency_ms', 0):.2f}ms")
                
                # 显示路由分布
                routing_dist = summary.get('routing_distribution', {})
                if routing_dist:
                    print(f"\n🔀 路由决策分布:")
                    for decision, count in routing_dist.items():
                        print(f"   {decision}: {count}次")
                
                # 显示最慢查询
                slowest = performance_report.get('slowest_queries', [])
                if slowest:
                    print(f"\n🐢 最慢的5个查询:")
                    for i, query in enumerate(slowest, 1):
                        print(f"   {i}. [{query['routing']}] {query['question'][:40]}... ({query['latency_ms']:.0f}ms)")
            else:
                print("   暂无查询数据")
            
            # 显示告警规则状态
            print(f"\n🔔 告警规则状态:")
            for rule in rag_system.langsmith_manager.alert_rules:
                status = "✅" if rule.enabled else "❌"
                print(f"   {status} {rule.name} ({rule.metric_name} {rule.operator} {rule.threshold})")
        else:
            print("   LangSmith 管理器未初始化")
        
        print("=" * 60)
        
        # 启动交互模式
        rag_system.interactive_mode()
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        print("请检查配置和依赖是否正确安装")


if __name__ == "__main__":
    main()