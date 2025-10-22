"""
实体和关系提取模块
使用LLM从文档中提取实体、关系和属性，构建知识图谱的基础
"""

from typing import List, Dict, Tuple
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from config import LOCAL_LLM


class EntityExtractor:
    """实体提取器 - 使用LLM从文本中提取实体"""
    
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
        
        # 实体提取提示模板
        self.entity_prompt = PromptTemplate(
            template="""你是一个专业的实体识别专家。从以下文本中提取所有重要的实体。
            
实体类型包括:
- PERSON: 人物、作者、研究者
- ORGANIZATION: 组织、机构、公司
- CONCEPT: 技术概念、算法、方法论
- TECHNOLOGY: 具体技术、工具、框架
- PAPER: 论文、出版物
- EVENT: 事件、会议

文本内容:
{text}

请以JSON格式返回，包含以下字段:
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型",
            "description": "简短描述"
        }}
    ]
}}

不要包含前言或解释，只返回JSON。
""",
            input_variables=["text"]
        )
        
        # 关系提取提示模板
        self.relation_prompt = PromptTemplate(
            template="""你是一个关系抽取专家。从文本中识别实体之间的关系。

已识别的实体:
{entities}

文本内容:
{text}

请识别实体之间的关系，以JSON格式返回:
{{
    "relations": [
        {{
            "source": "源实体名称",
            "target": "目标实体名称",
            "relation_type": "关系类型",
            "description": "关系描述"
        }}
    ]
}}

关系类型包括: AUTHOR_OF, USES, BASED_ON, RELATED_TO, PART_OF, APPLIES_TO, IMPROVES, CITES

不要包含前言或解释，只返回JSON。
""",
            input_variables=["text", "entities"]
        )
        
        self.entity_chain = self.entity_prompt | self.llm | JsonOutputParser()
        self.relation_chain = self.relation_prompt | self.llm | JsonOutputParser()
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        try:
            result = self.entity_chain.invoke({"text": text[:2000]})  # 限制长度
            entities = result.get("entities", [])
            print(f"✅ 提取到 {len(entities)} 个实体")
            return entities
        except Exception as e:
            print(f"❌ 实体提取失败: {e}")
            return []
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        从文本中提取实体关系
        
        Args:
            text: 输入文本
            entities: 已识别的实体列表
            
        Returns:
            关系列表
        """
        try:
            entity_names = [e["name"] for e in entities]
            result = self.relation_chain.invoke({
                "text": text[:2000],
                "entities": ", ".join(entity_names)
            })
            relations = result.get("relations", [])
            print(f"✅ 提取到 {len(relations)} 个关系")
            return relations
        except Exception as e:
            print(f"❌ 关系提取失败: {e}")
            return []
    
    def extract_from_document(self, document_text: str) -> Dict:
        """
        从单个文档中提取实体和关系
        
        Args:
            document_text: 文档文本
            
        Returns:
            包含实体和关系的字典
        """
        print("🔍 开始提取实体...")
        entities = self.extract_entities(document_text)
        
        print("🔍 开始提取关系...")
        relations = self.extract_relations(document_text, entities)
        
        return {
            "entities": entities,
            "relations": relations
        }


class EntityDeduplicator:
    """实体去重和合并"""
    
    def __init__(self):
        self.llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
        
        self.merge_prompt = PromptTemplate(
            template="""判断以下两个实体是否指向同一个对象:

实体1: {entity1_name} - {entity1_desc}
实体2: {entity2_name} - {entity2_desc}

如果是同一个对象，返回:
{{
    "is_same": true,
    "canonical_name": "标准名称",
    "reason": "原因"
}}

如果不是，返回:
{{
    "is_same": false,
    "reason": "原因"
}}

只返回JSON，不要其他内容。
""",
            input_variables=["entity1_name", "entity1_desc", "entity2_name", "entity2_desc"]
        )
        
        self.merge_chain = self.merge_prompt | self.llm | JsonOutputParser()
    
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
        
        # 简单的基于名称的去重
        unique_entities = {}
        entity_mapping = {}  # 映射别名到标准名称
        
        for entity in entities:
            name = entity["name"].lower().strip()
            
            # 查找是否有相似实体
            merged = False
            for canonical_name, canonical_entity in unique_entities.items():
                # 简单的字符串匹配（可以用LLM做更智能的判断）
                if name in canonical_name or canonical_name in name:
                    entity_mapping[entity["name"]] = canonical_name
                    merged = True
                    break
            
            if not merged:
                unique_entities[name] = entity
                entity_mapping[entity["name"]] = name
        
        print(f"✅ 去重完成，剩余 {len(unique_entities)} 个唯一实体")
        
        return {
            "entities": list(unique_entities.values()),
            "mapping": entity_mapping
        }


def initialize_entity_extractor():
    """初始化实体提取器"""
    return EntityExtractor()
