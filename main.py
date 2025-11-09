"""
主应用程序入口
集成所有模块，构建工作流并运行自适应RAG系统
"""

from langgraph.graph import END, StateGraph, START
from pprint import pprint

from config import setup_environment
from document_processor import initialize_document_processor
from routers_and_graders import initialize_graders_and_router
from workflow_nodes import WorkflowNodes, GraphState


class AdaptiveRAGSystem:
    """自适应RAG系统主类"""
    
    def __init__(self):
        print("初始化自适应RAG系统...")
        
        # 设置环境和验证API密钥
        try:
            setup_environment()
       
            print("✅ API密钥验证成功")
        except ValueError as e:
            print(f"❌ {e}")
            raise
        
        # 初始化文档处理器
        print("设置文档处理器...")
        self.doc_processor, self.vectorstore, self.retriever, self.doc_splits = initialize_document_processor()
        
        # 初始化评分器和路由器
        print("初始化评分器和路由器...")
        self.graders = initialize_graders_and_router()
        
        # 初始化工作流节点
        print("设置工作流节点...")
        self.workflow_nodes = WorkflowNodes(self.retriever, self.graders)
        
        # 构建工作流
        print("构建工作流图...")
        self.app = self._build_workflow()
        
        print("✅ 自适应RAG系统初始化完成！")
    
    def _build_workflow(self):
        """构建工作流图"""
        # 创建工作流节点实例，传递DocumentProcessor实例
        self.workflow_nodes = WorkflowNodes(
            doc_processor=self.doc_processor,
            graders=self.graders
        )
        
        workflow = StateGraph(GraphState)
        
        # 定义节点
        workflow.add_node("web_search", self.workflow_nodes.web_search)
        workflow.add_node("retrieve", self.workflow_nodes.retrieve)
        workflow.add_node("grade_documents", self.workflow_nodes.grade_documents)
        workflow.add_node("generate", self.workflow_nodes.generate)
        workflow.add_node("transform_query", self.workflow_nodes.transform_query)
        
        # 构建图
        workflow.add_conditional_edges(
            START,
            self.workflow_nodes.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.workflow_nodes.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.workflow_nodes.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
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
    
    def query(self, question: str, verbose: bool = True):
        """
        处理查询
        
        Args:
            question (str): 用户问题
            verbose (bool): 是否显示详细输出
            
        Returns:
            str: 最终答案
        """
        print(f"\n🔍 处理问题: {question}")
        print("=" * 50)
        
        inputs = {"question": question}
        final_generation = None
        
        # 设置配置，增加递归限制
        config = {"recursion_limit": 50}  # 增加到 50，默认是 25
        
        for output in self.app.stream(inputs, config=config):
            for key, value in output.items():
                if verbose:
                    pprint(f"节点 '{key}':")
                    # 可选：在每个节点打印完整状态
                    # pprint(value, indent=2, width=80, depth=None)
                final_generation = value.get("generation", final_generation)
            if verbose:
                pprint("\n---\n")
        
        print("🎯 最终答案:")
        print("-" * 30)
        print(final_generation)
        print("=" * 50)
        
        return final_generation
    
    def interactive_mode(self):
        """交互模式，允许用户持续提问"""
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
                
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n👋 感谢使用，再见!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                print("请重试或输入 'quit' 退出")


def main():
    """主函数"""
    try:
        # 初始化系统
        rag_system: AdaptiveRAGSystem = AdaptiveRAGSystem()
        
        # 测试查询
        test_question = "AlphaCodium论文讲的是什么？"
        # test_question = "解释embedding嵌入的原理，最好列举实现过程的具体步骤"
        rag_system.query(test_question)
        
        # 启动交互模式
        rag_system.interactive_mode()
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查配置和依赖是否正确安装")


if __name__ == "__main__":
    main()