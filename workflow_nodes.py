"""
工作流节点模块
包含所有工作流节点函数和状态管理
"""

import time
from typing import List
from typing_extensions import TypedDict
try:
    from langchain_core.documents import Document
except ImportError:
    try:
    from langchain_core.documents import Document
except ImportError:
    try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from config import LOCAL_LLM, WEB_SEARCH_RESULTS_COUNT, ENABLE_HYBRID_SEARCH, ENABLE_QUERY_EXPANSION, ENABLE_MULTIMODAL
from document_processor import DocumentProcessor
from retrieval_evaluation import RetrievalEvaluator, RetrievalResult
from pprint import pprint


class GraphState(TypedDict):
    """
    表示图的状态
    
    属性:
        question: 问题
        generation: LLM生成
        documents: 文档列表
        retry_count: 重试计数器，防止无限循环
        retrieval_metrics: 检索评估指标
    """
    question: str
    generation: str
    documents: List[str]
    retry_count: int
    retrieval_metrics: dict  # 添加检索评估指标


class WorkflowNodes:
    """工作流节点类，包含所有节点函数"""
    
    def __init__(self, doc_processor, graders, retriever=None):
        self.doc_processor = doc_processor  # 接收DocumentProcessor实例
        self.retriever = retriever if retriever is not None else getattr(doc_processor, 'retriever', None)
        self.graders = graders
        
        # 初始化检索评估器
        self.retrieval_evaluator = RetrievalEvaluator()
        
        # 设置RAG链 - 使用本地提示模板
        rag_prompt_template = PromptTemplate(
            template="""你是一个问答助手。使用以下检索到的上下文来回答问题。
如果你不知道答案，就说你不知道。最多使用三句话并保持答案简洁。

问题: {question}

上下文: {context}

答案:""",
            input_variables=["question", "context"]
        )
        llm = ChatOllama(model=LOCAL_LLM, temperature=0)
        self.rag_chain = rag_prompt_template | llm | StrOutputParser()
        
        # 设置网络搜索
        self.web_search_tool = TavilySearchResults(k=WEB_SEARCH_RESULTS_COUNT)
    
    def retrieve(self, state):
        """
        检索文档
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 添加了documents键的新状态，包含检索到的文档
        """
        print("---检索---")
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        retrieval_start_time = time.time()
        
        # 使用增强检索方法，支持混合检索、查询扩展和多模态
        try:
            # 检查是否有图像路径（多模态检索）
            image_paths = state.get("image_paths", None)
            
            # 使用增强检索
            documents = self.doc_processor.enhanced_retrieve(
                question, 
                top_k=5, 
                rerank_candidates=20,
                image_paths=image_paths,
                use_query_expansion=ENABLE_QUERY_EXPANSION
            )
            
            # 记录使用的检索方法
            if ENABLE_HYBRID_SEARCH:
                print("---使用混合检索---")
            if ENABLE_QUERY_EXPANSION:
                print("---使用查询扩展---")
            if image_paths and ENABLE_MULTIMODAL:
                print("---使用多模态检索---")
                
        except Exception as e:
            print(f"⚠️ 增强检索失败: {e}，回退到基本检索")
            # 回退到基本检索
            try:
                if self.retriever is not None:
                    documents = self.retriever.invoke(question)
                elif hasattr(self.doc_processor, 'vector_retriever') and self.doc_processor.vector_retriever is not None:
                    documents = self.doc_processor.vector_retriever.invoke(question)
                    print("   使用 vector_retriever 作为备选")
                elif hasattr(self.doc_processor, 'retriever') and self.doc_processor.retriever is not None:
                    documents = self.doc_processor.retriever.invoke(question)
                    print("   使用 doc_processor.retriever 作为备选")
                else:
                    print("❌ 检索器未正确初始化，返回空文档列表")
                    documents = []
            except Exception as fallback_e:
                print(f"❌ 回退检索也失败: {fallback_e}")
                documents = []
        
        # 计算检索时间
        retrieval_time = time.time() - retrieval_start_time
        
        # 评估检索结果
        retrieval_metrics = self._evaluate_retrieval_results(question, documents, retrieval_time)
        
        return {
            "documents": documents, 
            "question": question, 
            "retry_count": retry_count,
            "retrieval_metrics": retrieval_metrics
        }
    
    def generate(self, state):
        """
        生成答案
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 添加了generation键的新状态，包含LLM生成
        """
        print("---生成---")
        question = state["question"]
        documents = state["documents"]
        
        # RAG生成
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
        """
        确定检索到的文档是否与问题相关
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 更新documents键，只包含过滤后的相关文档
        """
        print("---检查文档与问题的相关性---")
        question = state["question"]
        documents = state["documents"]
        
        # 为每个文档评分
        filtered_docs = []
        for d in documents:
            score = self.graders["document_grader"].grade(question, d.page_content)
            grade = score
            if grade == "yes":
                print("---评分：文档相关---")
                filtered_docs.append(d)
            else:
                print("---评分：文档不相关---")
                continue
        return {"documents": filtered_docs, "question": question}
    
    def transform_query(self, state):
        """
        转换查询以产生更好的问题
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 用重新表述的问题更新question键
        """
        print("---转换查询---")
        question = state["question"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0) + 1
        
        print(f"   重试次数: {retry_count}")
        
        # 重写问题
        better_question = self.graders["query_rewriter"].rewrite(question)
        return {"documents": documents, "question": better_question, "retry_count": retry_count}
    
    def web_search(self, state):
        """
        基于重新表述的问题进行网络搜索
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 用附加的网络结果更新documents键
        """
        print("---网络搜索---")
        question = state["question"]
        
        # 网络搜索
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        
        return {"documents": web_results, "question": question}
    
    def route_question(self, state):
        """
        将问题路由到网络搜索或RAG
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            str: 要调用的下一个节点
        """
        print("---路由问题---")
        question = state["question"]
        print(question)
        source = self.graders["query_router"].route(question)
        print(source)
        if source == "web_search":
            print("---将问题路由到网络搜索---")
            return "web_search"
        elif source == "vectorstore":
            print("---将问题路由到RAG---")
            return "vectorstore"
    
    def decide_to_generate(self, state):
        """
        确定是生成答案还是重新生成问题
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            str: 要调用的下一个节点的二进制决策
        """
        print("---评估已评分的文档---")
        filtered_documents = state["documents"]
        
        if not filtered_documents:
            # 所有文档都被过滤掉了
            # 我们将重新生成一个新查询
            print("---决策：所有文档都与问题不相关，转换查询---")
            return "transform_query"
        else:
            # 我们有相关文档，所以生成答案
            print("---决策：生成---")
            return "generate"
    
    def grade_generation_v_documents_and_question(self, state):
        """
        确定生成是否基于文档并回答问题
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            str: 要调用的下一个节点的决策
        """
        print("---检查幻觉---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        retry_count = state.get("retry_count", 0)
        
        # 检查是否超过最大重试次数
        MAX_RETRIES = 3
        if retry_count >= MAX_RETRIES:
            print(f"⚠️ 已达到最大重试次数 ({MAX_RETRIES})，返回当前生成结果")
            return "useful"
        
        score = self.graders["hallucination_grader"].grade(generation, documents)
        grade = score
        
        # 检查幻觉
        if grade == "yes":
            print("---决策：生成基于文档---")
            # 检查问题回答
            print("---评分生成 vs 问题---")
            score = self.graders["answer_grader"].grade(question, generation)
            grade = score
            if grade == "yes":
                print("---决策：生成解决了问题---")
                return "useful"
            else:
                print("---决策：生成没有解决问题---")
                return "not useful"
        else:
            print("---决策：生成不基于文档，重新转换查询---")
            return "not supported"


def _evaluate_retrieval_results(self, question, documents, retrieval_time):
        """
        评估检索结果的质量
        
        Args:
            question: 查询问题
            documents: 检索到的文档
            retrieval_time: 检索耗时
            
        Returns:
            dict: 评估指标
        """
        try:
            # 创建模拟的相关文档（在实际应用中，这些应该是真实的相关文档）
            # 这里我们假设前几个文档是相关的，用于演示评估功能
            relevant_docs = documents[:min(2, len(documents))] if documents else []
            
            # 创建检索结果对象
            retrieval_result = RetrievalResult(
                query=question,
                retrieved_docs=documents,
                relevant_docs=relevant_docs,
                retrieval_time=retrieval_time
            )
            
            # 评估检索结果
            metrics = self.retrieval_evaluator.evaluate_retrieval([retrieval_result], k_values=[1, 3, 5])
            
            # 提取关键指标
            result_metrics = {
                "precision_at_1": metrics.precision_at_k.get(1, 0),
                "precision_at_3": metrics.precision_at_k.get(3, 0),
                "precision_at_5": metrics.precision_at_k.get(5, 0),
                "recall_at_1": metrics.recall_at_k.get(1, 0),
                "recall_at_3": metrics.recall_at_k.get(3, 0),
                "recall_at_5": metrics.recall_at_k.get(5, 0),
                "map_score": metrics.map_score,
                "mrr": metrics.mrr,
                "latency": metrics.latency,
                "retrieved_docs_count": len(documents)
            }
            
            # 打印评估结果
            print("\n---检索评估结果---")
            print(f"检索耗时: {result_metrics['latency']:.4f}秒")
            print(f"检索文档数: {result_metrics['retrieved_docs_count']}")
            print(f"Precision@1: {result_metrics['precision_at_1']:.4f}")
            print(f"Precision@3: {result_metrics['precision_at_3']:.4f}")
            print(f"Precision@5: {result_metrics['precision_at_5']:.4f}")
            print(f"Recall@1: {result_metrics['recall_at_1']:.4f}")
            print(f"Recall@3: {result_metrics['recall_at_3']:.4f}")
            print(f"Recall@5: {result_metrics['recall_at_5']:.4f}")
            print(f"MAP: {result_metrics['map_score']:.4f}")
            print(f"MRR: {result_metrics['mrr']:.4f}")
            print("--------------------\n")
            
            return result_metrics
            
        except Exception as e:
            print(f"⚠️ 检索评估失败: {e}")
            return {
                "error": str(e),
                "latency": retrieval_time,
                "retrieved_docs_count": len(documents)
            }


def format_docs(docs):
    """格式化文档用于显示"""
    return "\n\n".join(doc.page_content for doc in docs)