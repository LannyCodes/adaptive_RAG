from prompt_manager import get_prompt_manager
"""
实体和关系提取模块
使用LLM从文档中提取实体、关系和属性，构建知识图谱的基础
"""

from typing import List, Dict, Tuple
import time
import asyncio
import aiohttp
import json
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser
from config import LOCAL_LLM
from routers_and_graders import create_chat_model


class EntityExtractor:
    """实体提取器 - 使用LLM从文本中提取实体（支持异步批处理）"""
    
    def __init__(self, timeout: int = 180, max_retries: int = 3, enable_async: bool = True):
        """初始化实体提取器
        
        Args:
            timeout: LLM调用超时时间（秒）- 默认180秒以应对首次模型加载
            max_retries: 失败重试次数
            enable_async: 是否启用异步处理（默认启用）
        """
        self.llm = create_chat_model(format="json", temperature=0.0, timeout=timeout)
        self.max_retries = max_retries
        self.enable_async = enable_async
        self.ollama_url = "http://localhost:11434/api/generate"
        self.timeout = timeout  # 保存超时设置供异步使用
        
        # 实体提取提示模板
        self.entity_prompt = get_prompt_manager().get_template("extract_entities")
        
        # 关系提取提示模板
        self.relation_prompt = get_prompt_manager().get_template("extract_relations")
        
        self.entity_chain = self.entity_prompt | self.llm | JsonOutputParser()
        self.relation_chain = self.relation_prompt | self.llm | JsonOutputParser()
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        从文本中提取实体（带重试机制）
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        for attempt in range(self.max_retries):
            try:
                print(f"   🔄 提取实体 (尝试 {attempt + 1}/{self.max_retries})...", end="")
                result = self.entity_chain.invoke({"text": text[:2000]})  # 限制长度
                entities = result.get("entities", [])
                print(f" ✅ 提取到 {len(entities)} 个实体")
                return entities
            except TimeoutError as e:
                print(f" ⏱️ 超时")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"   ⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ 实体提取最终失败: 超时")
                    return []
            except Exception as e:
                print(f" ❌ 错误: {str(e)[:100]}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"   ❌ 实体提取最终失败: {e}")
                    return []
        return []
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        从文本中提取实体关系（带重试机制）
        
        Args:
            text: 输入文本
            entities: 已识别的实体列表
            
        Returns:
            关系列表
        """
        if not entities:
            print("   ⚠️ 无实体，跳过关系提取")
            return []
        
        for attempt in range(self.max_retries):
            try:
                print(f"   🔄 提取关系 (尝试 {attempt + 1}/{self.max_retries})...", end="")
                entity_names = [e["name"] for e in entities]
                result = self.relation_chain.invoke({
                    "text": text[:2000],
                    "entities": ", ".join(entity_names)
                })
                relations = result.get("relations", [])
                print(f" ✅ 提取到 {len(relations)} 个关系")
                return relations
            except TimeoutError as e:
                print(f" ⏱️ 超时")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"   ⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ 关系提取最终失败: 超时")
                    return []
            except Exception as e:
                print(f" ❌ 错误: {str(e)[:100]}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"   ❌ 关系提取最终失败: {e}")
                    return []
        return []
    
    async def _async_llm_call(self, prompt: str, session: aiohttp.ClientSession, attempt: int = 0) -> Dict:
        """异步调用 Ollama API"""
        try:
            timeout = aiohttp.ClientTimeout(
                total=self.timeout,      # 总超时
                connect=30,               # 连接超时 30 秒
                sock_read=self.timeout    # 读取超时
            )
            
            async with session.post(
                self.ollama_url,
                json={
                    "model": LOCAL_LLM,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0}
                },
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return json.loads(result.get('response', '{}'))
                else:
                    raise Exception(f"API返回错误: {response.status}")
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            if attempt < self.max_retries - 1:
                wait_time = (attempt + 1) * 3
                await asyncio.sleep(wait_time)
                return await self._async_llm_call(prompt, session, attempt + 1)
            raise Exception(f"连接失败: {str(e)}")
        except Exception as e:
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2)
                return await self._async_llm_call(prompt, session, attempt + 1)
            raise
    
    async def _extract_entities_async(self, text: str, doc_index: int, session: aiohttp.ClientSession) -> List[Dict]:
        """异步提取实体"""
        prompt = self.entity_prompt.format(text=text[:2000])
        
        for attempt in range(self.max_retries):
            try:
                print(f"   [文档 #{doc_index + 1}] 🔄 提取实体 (尝试 {attempt + 1}/{self.max_retries})...", end="")
                result = await self._async_llm_call(prompt, session, attempt)
                entities = result.get("entities", [])
                print(f" ✅ {len(entities)} 个实体")
                return entities
            except Exception as e:
                print(f" ❌ {str(e)[:50]}")
                if attempt == self.max_retries - 1:
                    return []
        return []
    
    async def _extract_relations_async(self, text: str, entities: List[Dict], doc_index: int, session: aiohttp.ClientSession) -> List[Dict]:
        """异步提取关系"""
        if not entities:
            return []
        
        entity_names = [e["name"] for e in entities]
        prompt = self.relation_prompt.format(
            text=text[:2000],
            entities=", ".join(entity_names)
        )
        
        for attempt in range(self.max_retries):
            try:
                print(f"   [文档 #{doc_index + 1}] 🔄 提取关系 (尝试 {attempt + 1}/{self.max_retries})...", end="")
                result = await self._async_llm_call(prompt, session, attempt)
                relations = result.get("relations", [])
                print(f" ✅ {len(relations)} 个关系")
                return relations
            except Exception as e:
                print(f" ❌ {str(e)[:50]}")
                if attempt == self.max_retries - 1:
                    return []
        return []
    
    async def _extract_from_document_async(self, document_text: str, doc_index: int, session: aiohttp.ClientSession) -> Dict:
        """异步处理单个文档"""
        print(f"\n🔍 [文档 #{doc_index + 1}] 开始异步提取...")
        
        # 并发提取实体和关系（先实体，再关系）
        entities = await self._extract_entities_async(document_text, doc_index, session)
        relations = await self._extract_relations_async(document_text, entities, doc_index, session)
        
        print(f"📊 [文档 #{doc_index + 1}] 完成: {len(entities)} 实体, {len(relations)} 关系")
        
        return {
            "entities": entities,
            "relations": relations
        }
    
    async def extract_batch_async(self, documents: List[Tuple[str, int]]) -> List[Dict]:
        """异步批量处理多个文档
        
        Args:
            documents: 文档列表，每个元素为 (document_text, doc_index) 元组
            
        Returns:
            提取结果列表
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._extract_from_document_async(doc_text, doc_idx, session)
                for doc_text, doc_idx in documents
            ]
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"⚠️ 文档 #{documents[i][1] + 1} 处理失败: {result}")
                    processed_results.append({"entities": [], "relations": []})
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def extract_from_document(self, document_text: str, doc_index: int = 0) -> Dict:
        """
        从单个文档中提取实体和关系（同步接口，保持向后兼容）
        
        Args:
            document_text: 文档文本
            doc_index: 文档索引（用于日志）
            
        Returns:
            包含实体和关系的字典
        """
        # 同步方式调用（保持向后兼容）
        print(f"\n🔍 文档 #{doc_index + 1}: 开始提取...")
        
        entities = self.extract_entities(document_text)
        relations = self.extract_relations(document_text, entities)
        
        print(f"📊 文档 #{doc_index + 1} 完成: {len(entities)} 实体, {len(relations)} 关系")
        
        return {
            "entities": entities,
            "relations": relations
        }


class EntityDeduplicator:
    """实体去重和合并"""
    
    def __init__(self):
        self.llm = create_chat_model(format="json", temperature=0.0)
        
        self.merge_prompt = get_prompt_manager().get_template("merge_entities")
        
        self.merge_chain = self.merge_prompt | self.llm | JsonOutputParser()
    
    def _is_same_entity(self, entity1: Dict, entity2: Dict) -> bool:
        """
        使用LLM判断两个实体是否指向同一个对象
        
        Args:
            entity1: 实体1字典
            entity2: 实体2字典
            
        Returns:
            bool: 是否相同
        """
        try:
            # 准备输入
            input_data = {
                "entity1_name": entity1["name"],
                "entity1_desc": entity1.get("description", "无描述"),
                "entity2_name": entity2["name"],
                "entity2_desc": entity2.get("description", "无描述")
            }
            
            # 调用LLM
            result = self.merge_chain.invoke(input_data)
            
            # 解析结果
            return result.get("is_same", False)
        except Exception as e:
            print(f"   ⚠️ LLM判重失败 ({entity1['name']} vs {entity2['name']}): {e}")
            return False

    def deduplicate_entities(self, entities: List[Dict]) -> Dict:
        """
        去重实体列表
        
        Args:
            entities: 实体列表
            
        Returns:
            包含entities和mapping的字典
        """
        if len(entities) <= 1:
            # 返回字典格式，保持一致性
            entity_mapping = {entity["name"]: entity["name"] for entity in entities} if entities else {}
            return {
                "entities": entities,
                "mapping": entity_mapping
            }
        
        print(f"🔄 开始去重 {len(entities)} 个实体...")
        
        # 基于名称和LLM的智能去重
        unique_entities = {}
        entity_mapping = {}  # 映射别名到标准名称
        
        for entity in entities:
            name = entity["name"].lower().strip()
            
            # 查找是否有相似实体
            merged = False
            for canonical_name, canonical_entity in unique_entities.items():
                # 1. 简单的字符串匹配（作为预筛选）
                # 如果名称完全相同，或者是子串关系，则考虑合并
                is_substring = name in canonical_name or canonical_name in name
                
                if name == canonical_name:
                    # 完全匹配（忽略大小写），直接合并
                    entity_mapping[entity["name"]] = canonical_entity["name"]
                    merged = True
                    break
                elif is_substring:
                    # 子串匹配，使用LLM进行智能确认
                    # 例如："Python" 和 "Python Programming Language" -> 合并
                    # "Java" 和 "JavaScript" -> 不合并
                    if self._is_same_entity(entity, canonical_entity):
                        print(f"   ✨ 合并: {entity['name']} -> {canonical_entity['name']}")
                        entity_mapping[entity["name"]] = canonical_entity["name"]
                        merged = True
                        break
            
            if not merged:
                unique_entities[name] = entity
                entity_mapping[entity["name"]] = entity["name"]
                
        print(f"✅ 去重完成，剩余 {len(unique_entities)} 个唯一实体")
        
        return {
            "entities": list(unique_entities.values()),
            "mapping": entity_mapping
        }


def initialize_entity_extractor():
    """初始化实体提取器"""
    return EntityExtractor()
