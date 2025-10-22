"""
路由器和评分器模块
包含查询路由、文档相关性评分、答案质量评分和幻觉检测
"""

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from config import LOCAL_LLM


class QueryRouter:
    """查询路由器，决定使用向量存储还是网络搜索"""
    
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
        self.prompt = PromptTemplate(
            template="""你是一个专家，负责将用户问题路由到向量存储或网络搜索。
            对于关于LLM智能体、提示工程和对抗性攻击的问题，使用向量存储。
            你不需要严格匹配问题中与这些主题相关的关键词。
            否则，使用网络搜索。根据问题给出二进制选择'web_search'或'vectorstore'。
            返回一个只包含'datasource'键的JSON，不要前言或解释。
            要路由的问题：{question}""",
            input_variables=["question"],
        )
        self.router = self.prompt | self.llm | JsonOutputParser()
    
    def route(self, question: str) -> str:
        """路由问题到相应的数据源"""
        result = self.router.invoke({"question": question})
        return result.get("datasource", "web_search")


class DocumentGrader:
    """文档相关性评分器"""
    
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
        self.prompt = PromptTemplate(
            template="""你是一个评分员，评估检索到的文档是否与用户问题相关。
            如果文档包含与用户问题相关的关键词或语义，请给出'yes'分数。
            给出二进制分数'yes'或'no'，以表明文档是否与问题相关。
            将二进制分数作为JSON提供，只包含'score'键，不要前言或解释。
            
            检索到的文档：

 {document} 


            用户问题：{question}""",
            input_variables=["question", "document"],
        )
        self.grader = self.prompt | self.llm | JsonOutputParser()
    
    def grade(self, question: str, document: str) -> str:
        """评估文档与问题的相关性"""
        result = self.grader.invoke({"question": question, "document": document})
        return result.get("score", "no")


class AnswerGrader:
    """答案质量评分器"""
    
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
        self.prompt = PromptTemplate(
            template="""你是一个评分员，评估答案是否有助于解决问题。
            这里是答案：
            \n ------- \n
            {generation} 
            \n ------- \n
            这里是问题：{question}
            给出二进制分数'yes'或'no'，表示答案是否有助于解决问题。
            将二进制分数作为JSON提供，只包含'score'键，不要前言或解释。""",
            input_variables=["generation", "question"],
        )
        self.grader = self.prompt | self.llm | JsonOutputParser()
    
    def grade(self, question: str, generation: str) -> str:
        """评估答案质量"""
        result = self.grader.invoke({"question": question, "generation": generation})
        return result.get("score", "no")


class HallucinationGrader:
    """幻觉检测器"""
    
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
        self.prompt = PromptTemplate(
            template="""你是一个评分员，评估LLM生成是否基于/支持一组检索到的事实。
            给出二进制分数'yes'或'no'。'yes'意味着答案基于/支持文档。
            将二进制分数作为JSON提供，只包含'score'键，不要前言或解释。
            
            检索到的文档：

 {documents} 


            LLM生成：{generation}""",
            input_variables=["generation", "documents"],
        )
        self.grader = self.prompt | self.llm | JsonOutputParser()
    
    def grade(self, generation: str, documents) -> str:
        """检测生成内容是否存在幻觉"""
        result = self.grader.invoke({"generation": generation, "documents": documents})
        return result.get("score", "no")


class QueryRewriter:
    """查询重写器，优化查询以获得更好的检索结果"""
    
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, temperature=0)
        self.prompt = PromptTemplate(
            template="""你是一个问题重写器，将输入问题转换为更适合向量存储检索的更好版本。
            查看初始问题并制定一个改进的问题。
            这里是初始问题：\n\n {question}。改进的问题（无前言）：\n """,
            input_variables=["question"],
        )
        self.rewriter = self.prompt | self.llm | StrOutputParser()
    
    def rewrite(self, question: str) -> str:
        """重写查询以获得更好的检索效果"""
        print(f"---原始查询: {question}---")
        rewritten_query = self.rewriter.invoke({"question": question})
        print(f"---重写查询: {rewritten_query}---")
        return rewritten_query


def initialize_graders_and_router():
    """初始化所有评分器和路由器"""
    query_router = QueryRouter()
    document_grader = DocumentGrader()
    answer_grader = AnswerGrader()
    hallucination_grader = HallucinationGrader()
    query_rewriter = QueryRewriter()
    
    return {
        "query_router": query_router,
        "document_grader": document_grader,
        "answer_grader": answer_grader,
        "hallucination_grader": hallucination_grader,
        "query_rewriter": query_rewriter
    }