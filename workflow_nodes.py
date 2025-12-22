"""
工作流节点模块
包含所有工作流节点函数和状态管理
"""

import time
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
import inspect

from config import LOCAL_LLM, WEB_SEARCH_RESULTS_COUNT, ENABLE_HYBRID_SEARCH, ENABLE_QUERY_EXPANSION, ENABLE_MULTIMODAL, EMBEDDING_MODEL
from document_processor import DocumentProcessor
from retrieval_evaluation import RetrievalEvaluator, RetrievalResult
from pprint import pprint
from routers_and_graders import create_chat_model


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
    sub_queries: List[str]  # 分解后的子问题列表
    current_query_index: int  # 当前处理的子问题索引
    original_question: str # 原始问题，用于早期终止检查


class WorkflowNodes:
    """工作流节点类，包含所有节点函数"""
    
    def __init__(self, doc_processor, graders, retriever=None):
        self.doc_processor = doc_processor  # 接收DocumentProcessor实例
        self.retriever = retriever if retriever is not None else getattr(doc_processor, 'retriever', None)
        self.graders = graders
        
        # 初始化检索评估器
        self.retrieval_evaluator = RetrievalEvaluator(embedding_model=EMBEDDING_MODEL)
        
        # 设置RAG链 - 使用本地提示模板
        rag_prompt_template = PromptTemplate(
            template="""你是一个智能问答助手。使用以下检索到的上下文来回答问题。
            
            规则：
            1. 如果你不知道答案，就说你不知道。
            2. 如果用户请求特定格式（如Markdown、列表、代码块等），请严格遵守。
            3. 如果没有特定格式要求，保持答案简洁。
            
            问题: {question}
            
            上下文: {context}
            
            答案:""",
            input_variables=["question", "context"]
        )
        llm = create_chat_model(temperature=0.0)
        self.rag_chain = rag_prompt_template | llm | StrOutputParser()
        
        # 设置网络搜索
        self.web_search_tool = TavilySearchResults(k=WEB_SEARCH_RESULTS_COUNT)
    
    def decompose_query(self, state):
        """
        将初始查询分解为子查询
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 更新sub_queries和current_query_index
        """
        print("---查询分解---")
        question = state["question"]
        
        # 使用分解器
        sub_queries = self.graders["query_decomposer"].decompose(question)
        
        # 如果分解器返回空或只有一个问题，我们仍然将其视为列表
        if not sub_queries:
            sub_queries = [question]
            
        print(f"   生成了 {len(sub_queries)} 个子查询")
        
        return {
            "sub_queries": sub_queries, 
            "current_query_index": 0,
            "question": sub_queries[0], # 将当前问题设置为第一个子查询
            "original_question": question, # 保存原始问题
            "documents": [], # 清空文档，准备开始新的检索
            "retry_count": 0
        }

    async def retrieve(self, state):
        """
        检索文档 (异步版本)
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 添加了documents键的新状态，包含检索到的文档
        """
        print("---检索---")
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        retrieval_start_time = time.time()
        
        # 记录检索开始时间
        print(f"   步骤1: 查询扩展")
        expanded_query = question
        
        # 步骤1: 查询扩展 - 增强原始查询
        try:
            # 如果启用了查询扩展，尝试扩展查询
            if ENABLE_QUERY_EXPANSION and hasattr(self, 'query_expansion_chain'):
                expanded_query_result = await self.query_expansion_chain.ainvoke(question)
                # 确保expanded_query是字符串
                if hasattr(expanded_query_result, 'content'):
                    expanded_query = expanded_query_result.content
                elif hasattr(expanded_query_result, 'page_content'):
                    expanded_query = expanded_query_result.page_content
                else:
                    expanded_query = str(expanded_query_result)
            
            print(f"   原始查询: {question}")
            print(f"   扩展查询: {expanded_query}")
        except Exception as e:
            print(f"⚠️ 查询扩展失败，使用原始查询: {e}")
            expanded_query = question
        
        documents = []
        
        # 步骤2: 主要检索逻辑
        try:
            # 检查是否有图像路径（多模态检索）
            image_paths = state.get("image_paths", None)
            
            # 使用增强检索方法，支持混合检索、查询扩展和多模态
            if hasattr(self.doc_processor, 'async_enhanced_retrieve'):
                documents = await self.doc_processor.async_enhanced_retrieve(
                    expanded_query, 
                    top_k=5, 
                    rerank_candidates=20,
                    image_paths=image_paths,
                    use_query_expansion=False  # 已经在上面处理过了
                )
                print(f"   增强检索结果: {len(documents)} 个文档")
            elif self.retriever is not None:
                # 基础检索器
                out = self.retriever.ainvoke(expanded_query)
                documents = await out if inspect.isawaitable(out) else out
                print(f"   基础检索结果: {len(documents)} 个文档")
            else:
                print("❌ 没有可用的检索器")
                documents = []
            
            # 记录使用的检索方法
            if ENABLE_HYBRID_SEARCH:
                print("---使用混合检索(向量+关键词)---")
            if image_paths and ENABLE_MULTIMODAL:
                print("---使用多模态检索---")
                
        except Exception as e:
            print(f"⚠️ 增强检索失败: {e}，回退到基本检索")
            documents = []
        
        # 步骤3: 如果检索结果不足，尝试回退检索
        MIN_DOCS_THRESHOLD = 3
        if len(documents) < MIN_DOCS_THRESHOLD:
            print(f"   步骤3: 回退检索 (当前 {len(documents)} < 阈值 {MIN_DOCS_THRESHOLD})")
            try:
                if hasattr(self.doc_processor, 'vector_retriever') and self.doc_processor.vector_retriever is not None:
                    out = self.doc_processor.vector_retriever.ainvoke(question)
                    fallback_docs = await out if inspect.isawaitable(out) else out
                    print("   使用 vector_retriever 作为备选")
                elif hasattr(self.doc_processor, 'retriever') and self.doc_processor.retriever is not None:
                    out = self.doc_processor.retriever.ainvoke(question)
                    fallback_docs = await out if inspect.isawaitable(out) else out
                    print("   使用 doc_processor.retriever 作为备选")
                else:
                    fallback_docs = []
                    print("❌ 回退检索器未正确初始化")
                
                # 合并结果并去重
                all_docs = documents + fallback_docs
                unique_docs = []
                seen_contents = set()
                for doc in all_docs:
                    content = getattr(doc, 'page_content', str(doc))
                    if content not in seen_contents:
                        unique_docs.append(doc)
                        seen_contents.add(content)
                documents = unique_docs
                print(f"   回退检索结果: {len(fallback_docs)} 个文档，合并后: {len(documents)} 个")
                
            except Exception as fallback_e:
                print(f"❌ 回退检索也失败: {fallback_e}")
        
        # === 向量多跳检索支持：合并上下文 ===
        # 如果这不是第一次检索（即重试次数 > 0 或 正在处理后续子查询），说明之前的检索结果可能不完整或问题被重写了
        # 我们应该保留之前的有价值文档，实现 "累积式上下文" (Accumulated Context)
        current_query_index = state.get("current_query_index", 0)
        if (retry_count > 0 or current_query_index > 0) and "documents" in state and state["documents"]:
            print(f"---多跳上下文合并 (轮次 {retry_count}, 子查询 {current_query_index})---")
            previous_docs = state["documents"]
            if previous_docs:
                # 简单的去重合并（基于内容）
                current_content = {getattr(d, 'page_content', str(d)) for d in documents}
                merged_count = 0
                for prev_doc in previous_docs:
                    content = getattr(prev_doc, 'page_content', str(prev_doc))
                    # 只有当内容不重复时才添加
                    if content not in current_content:
                        documents.append(prev_doc)
                        current_content.add(content)
                        merged_count += 1
                print(f"   合并了 {merged_count} 个来自上一轮/上一跳的文档，当前总文档数: {len(documents)}")
        # =================================
        
        # 计算检索时间
        retrieval_time = time.time() - retrieval_start_time
        
        print(f"   检索完成: {len(documents)} 个文档，耗时 {retrieval_time:.2f} 秒")
        
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
        original_question = state.get("original_question", question) # 优先使用原始问题
        documents = state["documents"]
        
        # RAG生成 - 使用原始问题以确保回答用户的初始意图
        # 如果用户有特定的格式要求（如Markdown），通常包含在original_question中
        generation = self.rag_chain.invoke({"context": documents, "question": original_question})
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
        
        # 提取当前上下文摘要，帮助重写器理解缺失信息
        context_summary = ""
        if documents:
            # 只提取前两个文档的摘要，避免上下文过长
            docs_content = [d.page_content for d in documents[:2]]
            context_summary = "\n---\n".join(docs_content)
            # 截断以防止过长
            if len(context_summary) > 2000:
                context_summary = context_summary[:2000] + "...(截断)"
        
        # 重写问题，传入上下文
        better_question = self.graders["query_rewriter"].rewrite(question, context=context_summary)
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
        
        try:
            # 网络搜索
            docs = self.web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = [Document(page_content=web_results)]
        except Exception as e:
            print(f"⚠️ 网络搜索失败: {e}")
            web_results = [Document(page_content="无法进行网络搜索，请根据已有知识回答。")]
        
        return {"documents": web_results, "question": question}
    
    def route_question(self, state):
        print("---路由问题---")
        question = state["question"]
        print(question)
        try:
            source = self.graders["query_router"].route(question)
        except Exception as e:
            print(f"⚠️ 查询路由失败，回退到向量检索: {e}")
            source = "vectorstore"
        print(source)
        if source == "web_search":
            print("---将问题路由到网络搜索---")
            return "web_search"
        else:
            print("---将问题路由到RAG---")
            return "vectorstore"
    
    def prepare_next_query(self, state):
        """
        准备下一个子查询：提取桥接实体并重写查询
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 更新question, current_query_index, retry_count
        """
        print("---准备下一个子查询---")
        current_query_index = state.get("current_query_index", 0)
        sub_queries = state.get("sub_queries", [])
        documents = state["documents"]
        
        # 移动到下一个索引
        next_index = current_query_index + 1
        next_query_raw = sub_queries[next_index]
        
        print(f"   原始下一个子查询: {next_query_raw}")
        
        # 提取上下文摘要用于重写（桥接实体提取）
        context_summary = ""
        if documents:
            # 使用所有相关文档作为上下文
            docs_content = [d.page_content for d in documents]
            context_summary = "\n---\n".join(docs_content)
            # 截断
            if len(context_summary) > 3000:
                context_summary = context_summary[:3000] + "...(截断)"
                
        # 使用重写器将上下文（包含桥接实体）注入到下一个查询中
        # 例如：Q1结果是"作者是J.K. Rowling"，Q2是"她出生在哪里？" -> "J.K. Rowling出生在哪里？"
        better_next_query = self.graders["query_rewriter"].rewrite(next_query_raw, context=context_summary)
        
        print(f"   优化后的下一个子查询: {better_next_query}")
        
        return {
            "question": better_next_query,
            "current_query_index": next_index,
            "retry_count": 0, # 重置重试计数
            "documents": documents # 保留文档作为上下文
        }

    def decide_to_generate(self, state):
        """
        确定是生成答案、继续下一个子查询还是重新生成问题
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            str: 要调用的下一个节点的决策
        """
        print("---评估已评分的文档---")
        filtered_documents = state["documents"]
        current_query_index = state.get("current_query_index", 0)
        sub_queries = state.get("sub_queries", [])
        original_question = state.get("original_question", "")
        retry_count = state.get("retry_count", 0)
        
        if not filtered_documents:
            # 检查是否超过最大重试次数
            if retry_count >= 3:
                print(f"⚠️ 已达到最大重试次数 ({retry_count}) 且无相关文档，回退到网络搜索")
                return "web_search"
                
            # 所有文档都被过滤掉了
            # 我们将重新生成一个新查询
            print("---决策：所有文档都与问题不相关，转换查询---")
            return "transform_query"
        else:
            # 我们有相关文档
            # 检查是否有更多子查询
            if sub_queries and current_query_index < len(sub_queries) - 1:
                # === 早期终止检查 ===
                # 检查当前累积的文档是否已经足以回答原始问题
                if original_question:
                    print("---检查是否已获取足够信息 (早期终止)---")
                    
                    # 准备文档上下文
                    docs_content = [d.page_content for d in filtered_documents]
                    context_summary = "\n---\n".join(docs_content)
                    if len(context_summary) > 5000: # 限制上下文长度
                        context_summary = context_summary[:5000]
                        
                    score = self.graders["answerability_grader"].grade(original_question, context_summary)
                    
                    if score == "yes":
                        print(f"---决策：当前信息已足够回答原始问题，跳过剩余 {len(sub_queries) - 1 - current_query_index} 个子查询---")
                        return "generate"
                    else:
                        print("---决策：信息尚不完整，继续下一个子查询---")
                
                print(f"---决策：当前子查询 ({current_query_index + 1}/{len(sub_queries)}) 完成，准备下一个---")
                return "prepare_next_query"
            else:
                # 所有子查询都完成（或没有子查询），生成答案
                print("---决策：所有子查询完成，生成---")
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
