"""
工作流节点模块
包含所有工作流节点函数和状态管理
"""

from typing import List
from typing_extensions import TypedDict
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
    from langchain.prompts import PromptTemplate

from config import LOCAL_LLM, WEB_SEARCH_RESULTS_COUNT
from pprint import pprint


class GraphState(TypedDict):
    """
    表示图的状态
    
    属性:
        question: 问题
        generation: LLM生成
        documents: 文档列表
    """
    question: str
    generation: str
    documents: List[str]


class WorkflowNodes:
    """工作流节点类，包含所有节点函数"""
    
    def __init__(self, retriever, graders):
        self.retriever = retriever
        self.graders = graders
        
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
        
        # 检索
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}
    
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
        
        # 重写问题
        better_question = self.graders["query_rewriter"].rewrite(question)
        return {"documents": documents, "question": better_question}
    
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
            print("---决策：生成不基于文档，重试---")
            return "not supported"


def format_docs(docs):
    """格式化文档用于显示"""
    return "\n\n".join(doc.page_content for doc in docs)