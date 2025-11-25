"""
GraphRAG集成示例
展示如何在自适应RAG系统中使用知识图谱功能
"""

import os
from pprint import pprint

from config import (
    setup_environment, 
    ENABLE_GRAPHRAG,
    GRAPHRAG_INDEX_PATH,
    GRAPHRAG_BATCH_SIZE
)
from document_processor import initialize_document_processor
from graph_indexer import initialize_graph_indexer
from graph_retriever import initialize_graph_retriever


class AdaptiveRAGWithGraph:
    """集成GraphRAG的自适应RAG系统"""
    
    def __init__(self, enable_graphrag=True, rebuild_index=False):
        print("🚀 初始化集成GraphRAG的自适应RAG系统...")
        print("="*60)
        
        # 设置环境
        try:
            setup_environment()
            print("✅ 环境配置完成")
        except ValueError as e:
            print(f"❌ {e}")
            raise
        
        # 初始化文档处理器
        print("\n📚 初始化文档处理器...")
        self.doc_processor, self.vectorstore, self.retriever, self.doc_splits = \
            initialize_document_processor()
        
        # GraphRAG组件
        self.enable_graphrag = enable_graphrag and ENABLE_GRAPHRAG
        self.graph_indexer = None
        self.graph_retriever = None
        self.knowledge_graph = None
        
        if self.enable_graphrag:
            self._setup_graphrag(rebuild_index)
        
        print("\n" + "="*60)
        print("✅ 系统初始化完成!")
        print("="*60)
    
    def _setup_graphrag(self, rebuild_index=False):
        """设置GraphRAG组件"""
        print("\n🔷 设置GraphRAG组件...")
        
        # 初始化索引器
        self.graph_indexer = initialize_graph_indexer()
        
        # 检查是否已有索引
        index_exists = os.path.exists(GRAPHRAG_INDEX_PATH)
        
        if index_exists and not rebuild_index:
            print(f"📂 发现现有索引: {GRAPHRAG_INDEX_PATH}")
            print("   加载现有索引...")
            self.knowledge_graph = self.graph_indexer.load_index(GRAPHRAG_INDEX_PATH)
        else:
            if rebuild_index:
                print("🔄 重新构建索引...")
            else:
                print("📝 首次构建索引...")
            
            # 构建索引
            self.knowledge_graph = self.graph_indexer.index_documents(
                documents=self.doc_splits,
                batch_size=GRAPHRAG_BATCH_SIZE,
                save_path=GRAPHRAG_INDEX_PATH
            )
        
        # 初始化检索器
        self.graph_retriever = initialize_graph_retriever(self.knowledge_graph)
        print("✅ GraphRAG组件设置完成")
    
    def query_vector_only(self, question: str) -> str:
        """仅使用向量检索"""
        print(f"\n{'='*60}")
        print(f"🔍 向量检索模式")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        docs = self.retriever.get_relevant_documents(question)
        
        print(f"\n📄 检索到 {len(docs)} 个文档片段:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"\n片段 {i}:")
            print(f"{doc.page_content[:200]}...")
        
        return self.doc_processor.format_docs(docs)
    
    def query_graph_local(self, question: str) -> str:
        """使用图谱本地查询"""
        if not self.enable_graphrag:
            return "GraphRAG未启用"
        
        print(f"\n{'='*60}")
        print(f"🔎 图谱本地查询模式")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        answer = self.graph_retriever.local_query(question)
        
        print(f"\n💡 答案:")
        print(answer)
        
        return answer
    
    def query_graph_global(self, question: str) -> str:
        """使用图谱全局查询"""
        if not self.enable_graphrag:
            return "GraphRAG未启用"
        
        print(f"\n{'='*60}")
        print(f"🌍 图谱全局查询模式")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        answer = self.graph_retriever.global_query(question)
        
        print(f"\n💡 答案:")
        print(answer)
        
        return answer
    
    def query_hybrid(self, question: str) -> dict:
        """混合查询：向量 + 图谱"""
        if not self.enable_graphrag:
            return {"error": "GraphRAG未启用"}
        
        print(f"\n{'='*60}")
        print(f"🔀 混合查询模式")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        # 向量检索
        vector_docs = self.retriever.get_relevant_documents(question)
        vector_context = self.doc_processor.format_docs(vector_docs[:3])
        
        # 图谱查询
        graph_results = self.graph_retriever.hybrid_query_with_metrics(question)
        
        result = {
            "question": question,
            "vector_retrieval": {
                "doc_count": len(vector_docs),
                "context": vector_context[:500] + "..." if len(vector_context) > 500 else vector_context
            },
            "graph_local": graph_results["local"],
            "graph_global": graph_results["global"],
            "graph_local_hallucination": graph_results.get("local_hallucination"),
            "graph_global_hallucination": graph_results.get("global_hallucination"),
            "graph_local_metrics": graph_results.get("local_metrics"),
            "graph_global_metrics": graph_results.get("global_metrics")
        }
        
        print("\n📊 结果汇总:")
        print(f"  • 向量检索: {len(vector_docs)} 个文档")
        print(f"  • 图谱本地查询完成")
        print(f"  • 图谱全局查询完成")
        
        return result
    
    def query_smart(self, question: str) -> str:
        """智能查询：自动选择最佳策略"""
        if not self.enable_graphrag:
            return self.query_vector_only(question)
        
        print(f"\n{'='*60}")
        print(f"🧠 智能查询模式")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        answer = self.graph_retriever.smart_query(question)
        
        print(f"\n💡 答案:")
        print(answer)
        
        return answer
    
    def get_graph_statistics(self):
        """获取知识图谱统计信息"""
        if not self.enable_graphrag or not self.knowledge_graph:
            print("GraphRAG未启用或图谱未构建")
            return
        
        stats = self.knowledge_graph.get_statistics()
        
        print("\n" + "="*60)
        print("📊 知识图谱统计信息")
        print("="*60)
        print(f"节点数: {stats['num_nodes']}")
        print(f"边数: {stats['num_edges']}")
        print(f"社区数: {stats['num_communities']}")
        print(f"图密度: {stats['density']:.4f}")
        print("\n实体类型分布:")
        for etype, count in stats['entity_types'].items():
            print(f"  • {etype}: {count}")
        print("="*60)
        
        return stats
    
    def interactive_mode(self):
        """交互模式"""
        print("\n" + "="*60)
        print("🤖 欢迎使用GraphRAG增强的自适应RAG系统!")
        print("="*60)
        print("\n查询模式:")
        print("  1️⃣  vector   - 仅向量检索")
        print("  2️⃣  local    - 图谱本地查询")
        print("  3️⃣  global   - 图谱全局查询")
        print("  4️⃣  hybrid   - 混合查询")
        print("  5️⃣  smart    - 智能查询（推荐）")
        print("  6️⃣  stats    - 显示图谱统计")
        print("  7️⃣  quit     - 退出")
        print("-"*60)
        
        while True:
            try:
                mode = input("\n选择模式 (1-7): ").strip()
                
                if mode in ['7', 'quit', 'exit', '退出', 'q']:
                    print("👋 感谢使用，再见!")
                    break
                
                if mode in ['6', 'stats']:
                    self.get_graph_statistics()
                    continue
                
                question = input("❓ 请输入问题: ").strip()
                
                if not question:
                    print("⚠️  请输入有效问题")
                    continue
                
                if mode in ['1', 'vector']:
                    self.query_vector_only(question)
                elif mode in ['2', 'local']:
                    self.query_graph_local(question)
                elif mode in ['3', 'global']:
                    self.query_graph_global(question)
                elif mode in ['4', 'hybrid']:
                    result = self.query_hybrid(question)
                    pprint(result)
                else:  # 默认智能模式
                    self.query_smart(question)
                
            except KeyboardInterrupt:
                print("\n👋 感谢使用，再见!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                print("请重试或输入 'quit' 退出")


def main():
    """主函数"""
    try:
        # 初始化系统（首次运行设置rebuild_index=True）
        rag_system = AdaptiveRAGWithGraph(
            enable_graphrag=True,
            rebuild_index=False  # 设为True重新构建索引
        )
        
        # 显示图谱统计
        rag_system.get_graph_statistics()
        
        # 测试查询
        print("\n" + "="*60)
        print("🧪 测试查询示例")
        print("="*60)
        
        # 示例1: 本地查询
        rag_system.query_graph_local("LLM Agent的主要组成部分是什么？")
        
        # 示例2: 全局查询  
        rag_system.query_graph_global("这些文档主要讨论了什么主题？")
        
        # 启动交互模式
        rag_system.interactive_mode()
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
