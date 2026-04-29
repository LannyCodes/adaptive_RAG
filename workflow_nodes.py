"""
工作流节点模块
包含所有工作流节点函数和状态管理
"""

import time
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
import inspect

from config import LOCAL_LLM, WEB_SEARCH_RESULTS_COUNT, ENABLE_HYBRID_SEARCH, ENABLE_QUERY_EXPANSION, ENABLE_MULTIMODAL, EMBEDDING_MODEL, ENABLE_GRAPHRAG
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
        sub_queries: 分解后的子问题列表
        current_query_index: 当前处理的子问题索引
        original_question: 原始问题，用于早期终止检查
        kg_context: 知识图谱扩展上下文（实体邻域、关系路径、社区摘要）
    """
    question: str
    generation: str
    documents: List[str]
    retry_count: int
    retrieval_metrics: dict  # 添加检索评估指标
    sub_queries: List[str]  # 分解后的子问题列表
    current_query_index: int  # 当前处理的子问题索引
    original_question: str # 原始问题，用于早期终止检查
    kg_context: List[str]  # 知识图谱扩展上下文


class WorkflowNodes:
    """工作流节点类，包含所有节点函数"""
    
    def __init__(self, doc_processor, graders, retriever=None, graph_retriever=None):
        self.doc_processor = doc_processor  # 接收DocumentProcessor实例
        self.retriever = retriever if retriever is not None else getattr(doc_processor, 'retriever', None)
        self.graders = graders
        self.graph_retriever = graph_retriever  # GraphRAG检索器
        
        # 初始化检索评估器（复用已有的embeddings模型，避免重复加载到GPU）
        embeddings_for_eval = getattr(doc_processor, 'embeddings', None)
        self.retrieval_evaluator = RetrievalEvaluator(
            embedding_model=EMBEDDING_MODEL,
            embeddings_model=embeddings_for_eval
        )
        
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
        self.web_search_tool = TavilySearch(k=WEB_SEARCH_RESULTS_COUNT)
    
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
        检索文档 (异步版本，子查询并行检索)
        
        Args:
            state (dict): 当前图状态
            
        Returns:
            state (dict): 添加了documents键的新状态，包含检索到的文档
        """
        print("---检索---")
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        retrieval_start_time = time.time()
        
        # 读取 route_and_decompose 暂存的分解结果
        sub_queries = getattr(self, '_pending_sub_queries', None)
        original_question = getattr(self, '_pending_original_question', question)
        if sub_queries:
            print(f"   使用并行分解结果: {len(sub_queries)} 个子查询")
            # 清空暂存，避免后续 retrieve 误用
            self._pending_sub_queries = None
            self._pending_original_question = None
        else:
            sub_queries = state.get("sub_queries", None)
            original_question = state.get("original_question", question)
        
        if not sub_queries:
            sub_queries = [question]
        
        # ── 并行检索所有子查询 ──
        # 如果有多个子查询，并行检索后合并，避免串行逐个走完整循环
        if len(sub_queries) > 1 and retry_count == 0:
            print(f"   ⚡ 并行检索 {len(sub_queries)} 个子查询...")
            import asyncio
            all_documents = []
            seen_contents = set()
            
            async def retrieve_single_query(query):
                """检索单个子查询"""
                try:
                    docs = await self.doc_processor.async_enhanced_retrieve(
                        query,
                        top_k=5,
                        rerank_candidates=5,  # 子查询检索减少候选数，加速
                        image_paths=state.get("image_paths", None),
                        use_query_expansion=False
                    )
                    return docs if docs else []
                except Exception as e:
                    print(f"   ⚠️ 子查询检索失败 '{query[:30]}...': {e}")
                    return []
            
            # 并行发起所有子查询的检索
            results = await asyncio.gather(*[retrieve_single_query(q) for q in sub_queries])
            
            for docs in results:
                for doc in docs:
                    content = getattr(doc, 'page_content', str(doc))
                    if content not in seen_contents:
                        all_documents.append(doc)
                        seen_contents.add(content)
            
            print(f"   并行检索合并结果: {len(all_documents)} 个文档")
            
            # 用合并后的结果直接跳过子查询循环
            retrieval_time = time.time() - retrieval_start_time
            retrieval_metrics = self._evaluate_retrieval_results(original_question, all_documents, retrieval_time)
            
            # 知识图谱扩展（并行检索路径）
            kg_context = state.get("kg_context", [])
            if self.graph_retriever and ENABLE_GRAPHRAG:
                kg_hits = [d for d in all_documents if d.metadata.get("data_type", "").startswith("kg_")]
                if kg_hits:
                    print(f"   🔗 检测到 {len(kg_hits)} 条知识图谱命中，触发图谱扩展...")
                    try:
                        kg_context = self._expand_with_knowledge_graph(original_question, kg_hits, kg_context)
                    except Exception as e:
                        print(f"   ⚠️ 知识图谱扩展失败: {e}")
            
            return {
                "documents": all_documents,
                "question": original_question,
                "retry_count": retry_count,
                "retrieval_metrics": retrieval_metrics,
                "sub_queries": sub_queries,
                "current_query_index": len(sub_queries) - 1,  # 标记所有子查询已完成
                "original_question": original_question,
                "kg_context": kg_context,
            }
        
        # ── 单查询或重试场景 ──
        # 重试时 question 已被重写，优先使用 question 而非旧的 sub_queries
        if retry_count > 0:
            current_query = question  # transform_query 重写后的问题
            sub_queries = [question]
        else:
            current_query = sub_queries[0] if sub_queries else question
        print(f"   当前检索查询: {current_query}")
        
        # 记录检索开始时间
        print(f"   步骤1: 查询扩展")
        expanded_query = current_query
        
        # 步骤1: 查询扩展 - 增强原始查询
        try:
            # 如果启用了查询扩展，尝试扩展查询
            if ENABLE_QUERY_EXPANSION and hasattr(self, 'query_expansion_chain'):
                expanded_query_result = await self._safe_async_query_expansion_chain(self.query_expansion_chain, question)
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
                    rerank_candidates=10,  # 优化: 20→10, 减少CrossEncoder重排计算量, T4上省~2s
                    image_paths=image_paths,
                    use_query_expansion=False  # 已经在上面处理过了
                )
                print(f"   增强检索结果: {len(documents)} 个文档")
            elif self.retriever is not None:
                # 基础检索器
                try:
                    out = self.retriever.ainvoke(expanded_query)
                    # 严格检查是否为可等待对象
                    if inspect.isawaitable(out):
                        documents = await out
                        # 确保返回的是列表
                        if isinstance(documents, list):
                            pass  # 保持documents不变
                        elif documents is not None:
                            print(f"⚠️ retriever.ainvoke 返回了非列表类型: {type(documents)}")
                            documents = []
                        else:
                            documents = []
                    else:
                        # 不是可等待对象，检查是否为列表
                        if isinstance(out, list):
                            documents = out
                        elif out is not None:
                            print(f"⚠️ retriever.ainvoke 返回了非列表类型: {type(out)}")
                            documents = []
                        else:
                            documents = []
                    print(f"   基础检索结果: {len(documents)} 个文档")
                except Exception as e:
                    print(f"⚠️ 基础检索失败: {e}")
                    documents = []
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
        MIN_DOCS_THRESHOLD = 2
        if len(documents) < MIN_DOCS_THRESHOLD:
            print(f"   步骤3: 回退检索 (当前 {len(documents)} < 阈值 {MIN_DOCS_THRESHOLD})")
            try:
                # 使用安全的异步调用方法，避免 SearchResult await 问题
                fallback_docs = await self._safe_async_invoke(self.doc_processor, question)
                print("   使用 doc_processor 安全异步调用作为备选")
                
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
        
        # === 知识图谱扩展：检测 KG 命中并扩展上下文 ===
        kg_context = state.get("kg_context", [])
        if self.graph_retriever and ENABLE_GRAPHRAG:
            # 检测检索结果中是否包含知识图谱文档
            kg_hits = [d for d in documents if d.metadata.get("data_type", "").startswith("kg_")]
            if kg_hits:
                print(f"   🔗 检测到 {len(kg_hits)} 条知识图谱命中，触发图谱扩展...")
                try:
                    kg_context = self._expand_with_knowledge_graph(question, kg_hits, kg_context)
                except Exception as e:
                    print(f"   ⚠️ 知识图谱扩展失败: {e}")
        # =================================

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
        retrieval_metrics = self._evaluate_retrieval_results(current_query, documents, retrieval_time)
        
        return {
            "documents": documents, 
            "question": current_query, 
            "retry_count": retry_count,
            "retrieval_metrics": retrieval_metrics,
            "sub_queries": sub_queries,
            "current_query_index": 0,
            "original_question": original_question,
            "kg_context": kg_context,
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
        kg_context = state.get("kg_context", [])

        # 构建上下文：普通文档 + 知识图谱扩展上下文
        context = documents
        if kg_context:
            print(f"   🔗 融合 {len(kg_context)} 条知识图谱扩展上下文")
            # 将 KG 上下文转为 Document 对象加入 context
            for kc in kg_context:
                if isinstance(kc, str):
                    context.append(Document(page_content=kc, metadata={"data_type": "kg_expanded"}))
                elif isinstance(kc, Document):
                    context.append(kc)

        # RAG生成 - 使用原始问题以确保回答用户的初始意图
        # 如果用户有特定的格式要求（如Markdown），通常包含在original_question中
        # 使用线程池执行同步调用，避免阻塞事件循环
        import asyncio
        import concurrent.futures
        try:
            loop = asyncio.get_running_loop()
            # 在已有事件循环运行的线程中，使用线程池执行同步调用
            with concurrent.futures.ThreadPoolExecutor() as executor:
                generation = executor.submit(
                    self.rag_chain.invoke,
                    {"context": context, "question": original_question}
                ).result()
        except RuntimeError:
            # 没有运行中的事件循环，直接执行
            generation = self.rag_chain.invoke({"context": context, "question": original_question})
        except Exception as e:
            print(f"⚠️ 生成失败: {e}")
            generation = "生成答案时出错"
        return {"documents": documents, "question": question, "generation": generation, "kg_context": kg_context}
    
    def grade_documents(self, state):
        """
        确定检索到的文档是否与问题相关（批量评分，一次LLM调用）

        Args:
            state (dict): 当前图状态

        Returns:
            state (dict): 更新documents键，只包含过滤后的相关文档
        """
        print("---检查文档与问题的相关性---")
        question = state["question"]
        documents = state["documents"]

        if not documents:
            return {"documents": [], "question": question}

        # 批量评分：一次LLM调用评估所有文档，替代逐个调用
        try:
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import JsonOutputParser
            from routers_and_graders import create_chat_model

            # 构建批量评分提示
            docs_text = ""
            for i, doc in enumerate(documents):
                # 截取文档前500字符，避免prompt过长
                content = doc.page_content[:500]
                docs_text += f"\n---文档{i+1}---\n{content}\n"

            batch_prompt = PromptTemplate(
                template="""你是一个评分员，评估检索到的文档是否与用户问题相关。

评分标准（宽松）：
1. 只要文档与问题有哪怕一点点关联，就给出'yes'
2. 只有当文档完全不相关或主题完全相反时，才给出'no'
3. **特别注意**：不要因为术语不熟悉就判为不相关！专业术语（如金融、法律、医学术语）只要在文档上下文中有所提及，就应该判定为相关
4. 文档可能从不同角度讨论问题，部分相关也算相关

{retry_hint}
用户问题：{question}

以下是{doc_count}个文档：
{documents}

请对每个文档给出相关性评分。返回一个JSON，包含'scores'键，值为一个列表，每个元素为'yes'或'no'。
列表长度必须为{doc_count}，按文档编号顺序排列。
只返回JSON，不要前言或解释。""",
                input_variables=["question", "documents", "doc_count"],
            )

            # 重试时增加更宽松的提示
            retry_count = state.get("retry_count", 0)
            retry_hint = ""
            if retry_count > 0:
                retry_hint = "**这是重试评分，请更加宽松**：上一轮评分过于严格导致所有文档被过滤，请降低标准，只要文档可能包含有用信息就给'yes'。\n\n"

            batch_llm = create_chat_model(format="json", temperature=0.0)
            batch_chain = batch_prompt | batch_llm | JsonOutputParser()

            result = batch_chain.invoke({
                "question": question,
                "documents": docs_text,
                "doc_count": len(documents),
                "retry_hint": retry_hint,
            })

            scores = result.get("scores", [])

            # 如果返回的scores数量不匹配，回退到逐个评分
            if len(scores) != len(documents):
                print(f"⚠️ 批量评分返回数量不匹配({len(scores)} vs {len(documents)})，回退逐个评分")
                scores = None
        except Exception as e:
            print(f"⚠️ 批量评分失败: {e}，回退逐个评分")
            scores = None

        # 如果批量评分成功
        if scores is not None:
            filtered_docs = []
            rejected_docs = []  # 保留被拒绝的文档，用于兜底
            for i, (doc, score) in enumerate(zip(documents, scores)):
                if score == "yes":
                    print(f"---评分：文档{i+1}相关---")
                    filtered_docs.append(doc)
                else:
                    print(f"---评分：文档{i+1}不相关---")
                    rejected_docs.append(doc)

            # 兜底逻辑：如果所有文档都被过滤掉，保留原始文档
            # 避免"过度严格"导致进入无意义的重试循环
            if not filtered_docs and documents:
                print(f"⚠️ 所有文档都被过滤，保留原始 {len(documents)} 个文档（评分可能过于严格）")
                filtered_docs = list(documents)

            return {"documents": filtered_docs, "question": question}

        # 回退：逐个评分（原始逻辑）
        import concurrent.futures

        def grade_single(doc):
            score = self.graders["document_grader"].grade(question, doc.page_content)
            return (doc, score == "yes")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(grade_single, documents))

        filtered_docs = []
        for doc, is_relevant in results:
            if is_relevant:
                print("---评分：文档相关---")
                filtered_docs.append(doc)
            else:
                print("---评分：文档不相关---")

        # 兜底逻辑：如果所有文档都被过滤掉，保留原始文档
        if not filtered_docs and documents:
            print(f"⚠️ 所有文档都被过滤，保留原始 {len(documents)} 个文档（评分可能过于严格）")
            filtered_docs = list(documents)

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
        return {
            "documents": documents, 
            "question": better_question, 
            "retry_count": retry_count,
            # 重置子查询状态，避免循环时复用旧的子查询
            "sub_queries": [better_question],
            "current_query_index": 0,
        }
    
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
            
            # 处理不同的返回格式
            if isinstance(docs, list):
                if len(docs) > 0:
                    # 检查第一个元素的类型
                    first_doc = docs[0]
                    if isinstance(first_doc, str):
                        # 返回的是字符串列表
                        web_results = "\n".join(docs)
                    elif isinstance(first_doc, dict) and "content" in first_doc:
                        # 返回的是字典列表，提取 content 字段
                        web_results = "\n".join([d["content"] for d in docs])
                    elif hasattr(first_doc, 'page_content'):
                        # 返回的是 Document 对象列表
                        web_results = "\n".join([d.page_content for d in docs])
                    else:
                        # 未知格式，转换为字符串
                        web_results = "\n".join([str(d) for d in docs])
                else:
                    web_results = "未找到相关搜索结果。"
            else:
                # 返回的不是列表，转换为字符串
                web_results = str(docs)
            
            web_results = [Document(page_content=web_results)]
        except Exception as e:
            print(f"⚠️ 网络搜索失败: {e}")
            web_results = [Document(page_content="无法进行网络搜索，请根据已有知识回答。")]
        
        return {"documents": web_results, "question": question}
    
    async def _safe_async_invoke(self, doc_processor, query):
        """统一的异步调用方法，避免 SearchResult await 问题"""
        import inspect
        
        try:
            # 首先尝试使用异步增强检索方法
            if hasattr(doc_processor, 'async_enhanced_retrieve'):
                result = await self._safe_invoke_callable(doc_processor.async_enhanced_retrieve, query, top_k=3, use_query_expansion=False)
                if isinstance(result, list):
                    return result
                else:
                    print(f"⚠️ async_enhanced_retrieve 返回了非列表类型: {type(result)}")
                    return []
            
            # 尝试使用异步混合检索方法
            if hasattr(doc_processor, 'async_hybrid_retrieve'):
                result = await self._safe_invoke_callable(doc_processor.async_hybrid_retrieve, query, top_k=3)
                if isinstance(result, list):
                    return result
                else:
                    print(f"⚠️ async_hybrid_retrieve 返回了非列表类型: {type(result)}")
                    return []
            
            # 尝试使用向量检索
            if hasattr(doc_processor, 'vector_retriever') and doc_processor.vector_retriever is not None:
                result = await self._safe_invoke_retriever(doc_processor.vector_retriever, query)
                if isinstance(result, list):
                    return result
            
            # 尝试使用基础检索器
            if hasattr(doc_processor, 'retriever') and doc_processor.retriever is not None:
                result = await self._safe_invoke_retriever(doc_processor.retriever, query)
                if isinstance(result, list):
                    return result
            
            # 如果都没有，返回空列表
            return []
            
        except Exception as e:
            print(f"⚠️ 安全异步调用失败: {e}")
            return []

    async def _safe_invoke_callable(self, callable_obj, *args, **kwargs):
        """安全调用可调用对象，处理异步和同步两种情况"""
        import inspect
        import asyncio
        
        try:
            out = callable_obj(*args, **kwargs)
            
            # 首先检查是否为 None
            if out is None:
                return None
                
            # 检查是否为列表类型（已经是最终结果）
            if isinstance(out, list):
                return out
                
            # 检查是否为字符串类型（已经是最终结果）
            if isinstance(out, str):
                return out
                
            # 检查是否为 SearchResult 等特定对象
            if hasattr(out, 'documents') and isinstance(out.documents, list):
                # 处理 SearchResult 等类似对象
                return out.documents
                
            # 严格检查是否为真正的协程对象
            # 使用多种方法确保我们只 await 真正的协程
            is_real_coroutine = (
                inspect.iscoroutine(out) or 
                inspect.iscoroutinefunction(out) or
                (hasattr(out, '__await__') and not hasattr(out, 'documents'))
            )
            
            if is_real_coroutine:
                # 如果是真正的协程，安全地 await
                try:
                    result = await out
                    # 再次检查返回结果
                    if isinstance(result, list) or isinstance(result, str) or result is None:
                        return result
                    elif hasattr(result, 'documents') and isinstance(result.documents, list):
                        return result.documents
                    else:
                        print(f"⚠️ 协程返回了未知类型: {type(result)}")
                        return result
                except (TypeError, RuntimeError) as e:
                    print(f"⚠️ 协程 await 失败: {e}")
                    # 尝试直接使用 out（可能已经是最终结果）
                    if hasattr(out, 'documents') and isinstance(out.documents, list):
                        return out.documents
                    else:
                        return None
            elif inspect.isawaitable(out):
                # 如果被错误标记为可等待对象，但实际不是协程
                print(f"⚠️ 检测到假可等待对象: {type(out)}")
                # 尝试直接使用 out（可能已经是最终结果）
                if hasattr(out, 'documents') and isinstance(out.documents, list):
                    # 处理 SearchResult 等类似对象
                    return out.documents
                else:
                    print(f"⚠️ 假可等待对象类型未知: {type(out)}")
                    return None
            else:
                # 不是可等待对象，可能已经是最终结果
                return out
                    
        except Exception as e:
            print(f"⚠️ _safe_invoke_callable 调用失败: {e}")
            return None

    async def _safe_invoke_retriever(self, retriever, query):
        """安全调用检索器"""
        import inspect
        
        try:
            if hasattr(retriever, 'ainvoke'):
                return await self._safe_invoke_callable(retriever.ainvoke, query)
            elif hasattr(retriever, 'invoke'):
                # 同步调用转为异步执行
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, retriever.invoke, query)
            else:
                print(f"⚠️ 检索器没有 invoke 或 ainvoke 方法: {type(retriever)}")
                return []
        except Exception as e:
            print(f"⚠️ _safe_invoke_retriever 调用失败: {e}")
            return []

    async def _safe_async_query_expansion_chain(self, query_expansion_chain, question):
        """安全的异步查询扩展链调用"""
        import inspect
        import asyncio
        
        try:
            if query_expansion_chain is None:
                return ""
                
            # 尝试异步调用
            if hasattr(query_expansion_chain, 'ainvoke'):
                out = query_expansion_chain.ainvoke(question)
                
                # 首先检查是否为 None
                if out is None:
                    return ""
                    
                # 检查是否为字符串类型（已经是最终结果）
                if isinstance(out, str):
                    return out
                    
                # 严格检查是否为真正的协程对象
                # 使用多种方法确保我们只 await 真正的协程
                is_real_coroutine = (
                    inspect.iscoroutine(out) or 
                    inspect.iscoroutinefunction(out) or
                    (hasattr(out, '__await__') and not hasattr(out, 'documents'))
                )
                
                if is_real_coroutine:
                    # 如果是真正的协程，安全地 await
                    try:
                        result = await out
                        # 再次检查返回结果
                        if isinstance(result, str):
                            return result
                        elif result is None:
                            return ""
                        else:
                            print(f"⚠️ 查询扩展链返回了非字符串类型: {type(result)}")
                            return str(result) if result else ""
                    except (TypeError, RuntimeError) as te:
                        print(f"⚠️ 异步调用失败，可能是假可等待对象: {te}")
                        # 如果失败，说明 out 可能不是真正的可等待对象
                        # 尝试直接使用 out 的值
                        if isinstance(out, str):
                            return out
                        else:
                            print(f"⚠️ out 不是字符串类型: {type(out)}")
                            return ""
                elif inspect.isawaitable(out):
                    # 如果被错误标记为可等待对象，但实际不是协程
                    print(f"⚠️ 检测到假可等待对象: {type(out)}")
                    # 尝试直接使用 out（可能已经是最终结果）
                    if isinstance(out, str):
                        return out
                    else:
                        print(f"⚠️ 假可等待对象不是字符串类型: {type(out)}")
                        return ""
                else:
                    # 不是可等待对象，可能已经是最终结果
                    return out if isinstance(out, str) else ""
            # 如果没有异步方法，尝试同步调用
            elif hasattr(query_expansion_chain, 'invoke'):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, query_expansion_chain.invoke, question)
                return result if isinstance(result, str) else ""
            else:
                print("⚠️ 查询扩展链没有 invoke 或 ainvoke 方法")
                return ""
        except Exception as e:
            print(f"⚠️ 安全异步查询扩展链失败: {e}")
            return ""

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
    
    def route_and_decompose(self, state):
        """
        路由+查询分解并行执行（优化：省~3s串行等待）
        将路由决策和查询分解合并为一个节点，两者用线程池并行执行。
        如果路由结果是 vectorstore，分解结果会直接注入state。
        如果路由结果是 web_search，跳过分解。
        """
        import concurrent.futures
        
        question = state["question"]
        print("---路由+分解（并行）---")
        
        # 并行执行路由和分解
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            route_future = executor.submit(self.graders["query_router"].route, question)
            decompose_future = executor.submit(self.graders["query_decomposer"].decompose, question)
            
            # 获取路由结果
            try:
                source = route_future.result(timeout=30)
            except Exception as e:
                print(f"⚠️ 路由失败，回退到向量检索: {e}")
                source = "vectorstore"
            
            # 如果路由到 web_search，不需要分解，直接返回
            if source == "web_search":
                print("---路由决策：网络搜索（跳过分解）---")
                # 取消分解任务
                decompose_future.cancel()
                return "web_search"
            
            # 路由到 vectorstore，获取分解结果
            print("---路由决策：向量检索---")
            try:
                sub_queries = decompose_future.result(timeout=30)
            except Exception as e:
                print(f"⚠️ 查询分解失败，使用原始问题: {e}")
                sub_queries = [question]
        
        # 确保至少包含原始问题
        if not sub_queries:
            sub_queries = [question]
        
        print(f"   生成了 {len(sub_queries)} 个子查询")
        for i, sq in enumerate(sub_queries):
            print(f"   子查询{i+1}: {sq}")
        
        # 返回更新后的state + 路由结果
        # 注意：conditional_edges 的返回值是下一个节点名称
        # 但我们需要同时更新 state，所以通过返回 "vectorstore" 指向 retrieve
        # state 的更新通过这种方式：在 LangGraph 中，conditional_edges 函数
        # 只能返回节点名称，不能更新 state。所以需要另一个方案。
        # 方案：把分解结果存入实例变量，让 retrieve 节点读取
        self._pending_sub_queries = sub_queries
        self._pending_original_question = question
        
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
            if retry_count >= 2:
                print(f"⚠️ 已达到最大重试次数 ({retry_count}) 且无相关文档，回退到网络搜索")
                return "web_search"
                
            # 所有文档都被过滤掉了
            # 我们将重新生成一个新查询
            print("---决策：所有文档都与问题不相关，转换查询---")
            return "transform_query"
        else:
            # 我们有相关文档
            # 如果所有子查询已经完成（并行检索时 current_query_index 已设为末尾），直接生成
            if current_query_index >= len(sub_queries) - 1:
                print("---决策：子查询已全部完成（并行检索），生成---")
                return "generate"
            
            # 检查是否有更多子查询需要串行处理
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
        
        # 检查是否超过最大重试次数（总重试上限，防止无限循环）
        MAX_TOTAL_RETRIES = 3
        if retry_count >= MAX_TOTAL_RETRIES:
            print(f"⚠️ 已达到总重试上限 ({MAX_TOTAL_RETRIES})，返回当前生成结果（可能不完全准确）")
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
    
    def _expand_with_knowledge_graph(self, question: str, kg_hits: list, existing_context: list) -> list:
        """
        基于向量检索命中的 KG 文档，用图谱遍历扩展上下文
        
        流程:
        1. 从 kg_hits 提取种子实体名称
        2. 在 NetworkX 图谱中查找邻居实体和关系
        3. 获取相关社区摘要
        4. 去重后追加到 kg_context
        
        Args:
            question: 用户问题
            kg_hits: 检索命中的 KG 文档列表
            existing_context: 已有的 KG 上下文
            
        Returns:
            扩展后的 KG 上下文列表
        """
        kg = self.graph_retriever.kg
        context = list(existing_context)
        seen = set(existing_context)
        
        # 1. 提取种子实体
        seed_entities = set()
        for doc in kg_hits:
            data_type = doc.metadata.get("data_type", "")
            if data_type == "kg_entity":
                name = doc.metadata.get("kg_name", "")
                if name:
                    seed_entities.add(name)
            elif data_type == "kg_relationship":
                subj = doc.metadata.get("kg_subject", "")
                obj = doc.metadata.get("kg_object", "")
                if subj:
                    seed_entities.add(subj)
                if obj:
                    seed_entities.add(obj)
        
        if not seed_entities:
            return context
        
        print(f"   种子实体: {list(seed_entities)[:5]}{'...' if len(seed_entities) > 5 else ''}")
        
        # 2. 图谱遍历：获取种子实体的邻居和关系（2跳）
        expanded_entities = set(seed_entities)
        relations_found = []
        
        for entity in seed_entities:
            if entity not in kg.graph:
                continue
            
            # 1跳邻居
            neighbors = kg.get_node_neighbors(entity, depth=2) if hasattr(kg, 'get_node_neighbors') else []
            expanded_entities.update(neighbors)
            
            # 收集关系
            for neighbor in neighbors:
                if kg.graph.has_edge(entity, neighbor):
                    edge_data = kg.graph.get_edge_data(entity, neighbor)
                    if isinstance(edge_data, dict):
                        rel_type = edge_data.get('relation_type', 'RELATED')
                        desc = edge_data.get('description', '')
                        weight = edge_data.get('weight', 1.0)
                        rel_text = f"{entity} --[{rel_type}]--> {neighbor}"
                        if desc:
                            rel_text += f": {desc}"
                        if rel_text not in seen:
                            relations_found.append(rel_text)
                            seen.add(rel_text)
        
        # 3. 收集实体信息
        entity_infos = []
        for entity in list(expanded_entities)[:20]:  # 限制数量
            info = kg.get_entity_info(entity) if hasattr(kg, 'get_entity_info') else None
            if info:
                etype = info.get('type', 'UNKNOWN')
                desc = info.get('description', '')
                text = f"{entity} ({etype}): {desc}"
                if text not in seen:
                    entity_infos.append(text)
                    seen.add(text)
        
        # 4. 收集相关社区摘要
        community_texts = []
        if hasattr(kg, 'community_summaries') and kg.community_summaries:
            # 找包含种子实体的社区
            for cid, members in (kg.communities or {}).items():
                if isinstance(members, (list, set)) and any(e in members for e in seed_entities):
                    summary = kg.community_summaries.get(cid, "")
                    if summary and summary not in seen:
                        community_texts.append(f"[社区摘要 {cid}] {summary}")
                        seen.add(summary)
                        if len(community_texts) >= 3:  # 最多3个社区
                            break
        
        # 5. 组装扩展上下文
        if entity_infos:
            context.append(f"【知识图谱-扩展实体】\n" + "\n".join(entity_infos[:15]))
        if relations_found:
            context.append(f"【知识图谱-扩展关系】\n" + "\n".join(relations_found[:15]))
        if community_texts:
            context.append(f"【知识图谱-社区摘要】\n" + "\n".join(community_texts))
        
        print(f"   扩展结果: {len(entity_infos)} 实体, {len(relations_found)} 关系, {len(community_texts)} 社区")
        
        return context

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