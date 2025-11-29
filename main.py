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
        
        # 设置环境和验证API密钥
        try:
            setup_environment()
            validate_api_keys()  # 验证API密钥是否正确设置
            print("✅ API密钥验证成功")
        except ValueError as e:
            print(f"❌ {e}")
            raise
        
        # 检查 Ollama 服务是否运行
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
        return workflow.compile(
            checkpointer=None,
            interrupt_before=None,
            interrupt_after=None,
            debug=False
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
        print(f"\n🔍 处理问题: {question}")
        print("=" * 50)
        
        inputs = {"question": question, "retry_count": 0}  # 初始化重试计数器
        final_generation = None
        retrieval_metrics = None
        
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
                    
                final_generation = value.get("generation", final_generation)
                # 保存检索评估指标
                if "retrieval_metrics" in value:
                    retrieval_metrics = value["retrieval_metrics"]
        
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
        
        # 测试查询
        # test_question = "AlphaCodium论文讲的是什么？"
        test_question = "LangGraph的作者目前在哪家公司工作？"
        # test_question = "解释embedding嵌入的原理，最好列举实现过程的具体步骤"
        
        # 使用 asyncio.run 执行异步查询
        result = asyncio.run(rag_system.query(test_question))
        
        # 显示测试查询的检索评估摘要
        if result.get("retrieval_metrics"):
            metrics = result["retrieval_metrics"]
            print("\n📊 测试查询检索评估摘要:")
            print(f"   - 检索耗时: {metrics.get('latency', 0):.4f}秒")
            print(f"   - 检索文档数: {metrics.get('retrieved_docs_count', 0)}")
            print(f"   - Precision@3: {metrics.get('precision_at_3', 0):.4f}")
            print(f"   - Recall@3: {metrics.get('recall_at_3', 0):.4f}")
            print(f"   - MAP: {metrics.get('map_score', 0):.4f}")
        
        # 启动交互模式
        rag_system.interactive_mode()
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        print("请检查配置和依赖是否正确安装")


if __name__ == "__main__":
    main()