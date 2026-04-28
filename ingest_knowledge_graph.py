"""
知识图谱数据导入脚本
将 source/knowledge_graph_export/ 中的知识图谱数据解析为 Document 对象并存入向量数据库

支持格式:
  - JSON-LD (.jsonld)  ← 推荐，结构化最完整
  - Turtle (.ttl)
  - N-Triples (.nt)
  - CSV 三元组 (.csv)

用法:
    # 导入 JSON-LD 文件（推荐）
    python ingest_knowledge_graph.py source/knowledge_graph_export/knowledge_graph.jsonld

    # 导入 CSV 三元组
    python ingest_knowledge_graph.py source/knowledge_graph_export/knowledge_graph_triples.csv

    # 预览模式（不写入向量库）
    python ingest_knowledge_graph.py --dry-run source/knowledge_graph_export/knowledge_graph.jsonld

    # 指定只导入实体
    python ingest_knowledge_graph.py --types entity source/knowledge_graph_export/knowledge_graph.jsonld

    # 指定导入实体和社区报告
    python ingest_knowledge_graph.py --types entity,community source/knowledge_graph_export/knowledge_graph.jsonld
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document


# ═══════════════════════════════════════════════
# JSON-LD 解析器
# ═══════════════════════════════════════════════

def _extract_name_from_id(id_uri: str) -> str:
    """从 URI 中提取实体名称，如 http://graphrag.example.org/entity/KEVIN_SCOTT -> KEVIN_SCOTT"""
    return id_uri.rstrip("/").split("/")[-1]


def _extract_type_short(type_uri: str) -> str:
    """从类型 URI 提取短名称，如 http://schema.org/PERSON -> PERSON"""
    return type_uri.rstrip("/").split("/")[-1]


def parse_jsonld(filepath: str,
                 include_types: Optional[set] = None) -> Tuple[List[Document], List[Document], List[Document]]:
    """
    解析 JSON-LD 格式的知识图谱数据

    Args:
        filepath: JSON-LD 文件路径
        include_types: 要包含的数据类型集合，None 表示全部
            可选: {"entity", "relationship", "community"}

    Returns:
        (entity_docs, relationship_docs, community_docs)
    """
    if include_types is None:
        include_types = {"entity", "relationship", "community"}

    print(f"📂 正在读取 JSON-LD 文件: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data.get("@graph", [])
    print(f"   读取到 {len(graph)} 个图节点")

    entity_docs = []
    relationship_docs = []
    community_docs = []

    entity_type_uris = {
        "http://schema.org/PERSON",
        "http://schema.org/ORGANIZATION",
        "http://schema.org/EVENT",
        "http://schema.org/GEO",
        "http://schema.org/INDUSTRY",
        "http://schema.org/PRODUCT",
        "http://schema.org/ORGANISM",
    }

    for node in graph:
        node_type = node.get("@type", "")
        node_id = node.get("@id", "")

        # ─── 实体节点 ───
        if node_type in entity_type_uris and "entity" in include_types:
            name = node.get("schema:name") or node.get("rdfs:label") or _extract_name_from_id(node_id)
            entity_type_short = _extract_type_short(node_type)
            description = node.get("schema:description", "")
            frequency = node.get("kg:frequency", 0)
            degree = node.get("kg:degree", 0.0)

            # 构造内容文本：实体名 + 类型 + 描述
            content = f"{name} ({entity_type_short}): {description}".strip()

            doc = Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "data_type": "kg_entity",
                    "kg_name": name,
                    "kg_entity_type": entity_type_short,
                    "kg_frequency": int(frequency) if frequency else 0,
                    "kg_degree": float(degree) if degree else 0.0,
                    "kg_id": node_id,
                    "file_type": "jsonld",
                }
            )
            entity_docs.append(doc)

        # ─── 关系节点 ───
        elif node_type == "rdf:Statement" and "relationship" in include_types:
            subject = node.get("rdf:subject", {})
            predicate = node.get("rdf:predicate", {})
            obj = node.get("rdf:object", {})
            description = node.get("schema:description", "")
            weight = node.get("kg:weight", 1.0)

            subject_name = _extract_name_from_id(subject.get("@id", "")) if isinstance(subject, dict) else str(subject)
            predicate_name = _extract_name_from_id(predicate.get("@id", "")) if isinstance(predicate, dict) else str(predicate)
            object_name = _extract_name_from_id(obj.get("@id", "")) if isinstance(obj, dict) else str(obj)

            # 构造内容文本：主语 - 谓语 - 宾语 + 描述
            content = f"{subject_name} --[{predicate_name}]--> {object_name}: {description}".strip()

            doc = Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "data_type": "kg_relationship",
                    "kg_subject": subject_name,
                    "kg_predicate": predicate_name,
                    "kg_object": object_name,
                    "kg_weight": float(weight) if weight else 1.0,
                    "kg_id": node_id,
                    "file_type": "jsonld",
                }
            )
            relationship_docs.append(doc)

        # ─── 社区报告节点 ───
        elif node_type == "schema:Article" and "community" in include_types:
            headline = node.get("schema:headline", "")
            summary = node.get("schema:description", "")
            full_content = node.get("schema:text", "")
            community_id = _extract_name_from_id(node_id)

            # 优先使用 full_content，其次 summary + headline
            content = full_content or f"{headline}\n\n{summary}"

            # 提取额外的社区属性
            level = node.get("kg:level", None)
            rank = node.get("kg:rank", None)
            size = node.get("kg:size", None)
            parent = node.get("kg:parent", None)

            meta = {
                "source": filepath,
                "data_type": "kg_community",
                "kg_community_id": community_id,
                "kg_headline": headline,
                "kg_level": int(level) if level is not None else None,
                "kg_rank": float(rank) if rank is not None else None,
                "kg_size": int(size) if size is not None else None,
                "kg_parent": parent,
                "kg_id": node_id,
                "file_type": "jsonld",
            }
            # 清理 None 值
            meta = {k: v for k, v in meta.items() if v is not None}

            doc = Document(
                page_content=content.strip(),
                metadata=meta,
            )
            community_docs.append(doc)

    print(f"✅ JSON-LD 解析完成:")
    print(f"   实体文档: {len(entity_docs)} 个")
    print(f"   关系文档: {len(relationship_docs)} 个")
    print(f"   社区报告文档: {len(community_docs)} 个")

    return entity_docs, relationship_docs, community_docs


# ═══════════════════════════════════════════════
# CSV 三元组解析器
# ═══════════════════════════════════════════════

def parse_csv_triples(filepath: str) -> List[Document]:
    """
    解析 CSV 格式的三元组数据 (subject, predicate, object)

    Returns:
        Document 列表，每个三元组作为一个 Document
    """
    import csv

    docs = []
    print(f"📂 正在读取 CSV 三元组文件: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过标题行

        for i, row in enumerate(reader):
            if len(row) < 3:
                continue

            subject, predicate, obj = row[0], row[1], row[2]
            subject_name = _extract_name_from_id(subject)
            predicate_name = _extract_name_from_id(predicate)

            # 如果 object 是 URI，提取名称；否则保留字面值
            if obj.startswith("http://"):
                object_name = _extract_name_from_id(obj)
            else:
                object_name = obj.strip('@"')

            content = f"{subject_name} {predicate_name} {object_name}"

            doc = Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "data_type": "kg_triple",
                    "kg_subject": subject_name,
                    "kg_predicate": predicate_name,
                    "kg_object": object_name,
                    "file_type": "csv",
                }
            )
            docs.append(doc)

    print(f"✅ CSV 解析完成: {len(docs)} 个三元组文档")
    return docs


# ═══════════════════════════════════════════════
# 文本分块（针对知识图谱数据）
# ═══════════════════════════════════════════════

class KnowledgeGraphTextSplitter:
    """
    知识图谱文档分块器

    策略:
    - 实体文档: 如果描述过长则分块，但保留实体名称和类型作为上下文前缀
    - 关系文档: 通常较短，直接保留
    - 社区报告: 按段落分块，保留标题作为上下文
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """对知识图谱文档进行分块"""
        result = []
        for doc in documents:
            data_type = doc.metadata.get("data_type", "")

            if data_type == "kg_community" and len(doc.page_content) > self.chunk_size:
                # 社区报告: 按段落分块，保留标题
                result.extend(self._split_community_report(doc))
            elif len(doc.page_content) > self.chunk_size:
                # 其他过长的文档: 简单分块
                result.extend(self._split_generic(doc))
            else:
                result.append(doc)

        return result

    def _split_community_report(self, doc: Document) -> List[Document]:
        """分块社区报告，保留标题上下文"""
        content = doc.page_content
        headline = doc.metadata.get("kg_headline", "")
        context_prefix = f"[社区报告: {headline}]\n\n" if headline else ""

        # 按二级标题分割
        sections = re.split(r'\n(?=## )', content)

        result = []
        current_chunk = context_prefix
        current_len = len(context_prefix)

        for section in sections:
            section_len = len(section)
            if current_len + section_len > self.chunk_size and current_len > len(context_prefix):
                # 保存当前块
                result.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={**doc.metadata, "chunk_index": len(result)},
                ))
                # 开始新块，保留标题上下文
                current_chunk = context_prefix + section
                current_len = len(context_prefix) + section_len
            else:
                current_chunk += "\n" + section
                current_len += section_len + 1

        # 最后一块
        if current_chunk.strip():
            result.append(Document(
                page_content=current_chunk.strip(),
                metadata={**doc.metadata, "chunk_index": len(result)},
            ))

        if len(result) > 1:
            print(f"   社区报告 '{headline}' 被分为 {len(result)} 个块")

        return result

    def _split_generic(self, doc: Document) -> List[Document]:
        """通用分块 - 纯 Python 实现，不依赖 langchain_text_splitters"""
        separators = ["\n\n", "\n", "。", "，", " ", ""]
        splits = self._recursive_split(doc.page_content, self.chunk_size, self.chunk_overlap, separators)
        return [
            Document(
                page_content=s.strip(),
                metadata={**doc.metadata, "chunk_index": i},
            )
            for i, s in enumerate(splits) if s.strip()
        ]

    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int, separators: list) -> list:
        """递归字符分割 - 仿 RecursiveCharacterTextSplitter"""
        if len(text) <= chunk_size:
            return [text]

        final_chunks = []
        # 找到当前层级的分隔符
        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []

        if not separator:
            # 无分隔符，硬切
            for i in range(0, len(text), chunk_size - chunk_overlap):
                final_chunks.append(text[i:i + chunk_size])
        else:
            # 按分隔符分割
            splits = text.split(separator)
            current_chunk = ""

            for split in splits:
                if len(split) > chunk_size and remaining_separators:
                    # 单个 split 还是太长，用下一级分隔符继续切
                    sub_chunks = self._recursive_split(split, chunk_size, chunk_overlap, remaining_separators)
                    for sc in sub_chunks:
                        if len(current_chunk) + len(separator) + len(sc) > chunk_size and current_chunk:
                            final_chunks.append(current_chunk)
                            # 重叠：保留末尾部分
                            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                            current_chunk = overlap_text + separator + sc if overlap_text else sc
                        else:
                            current_chunk = current_chunk + separator + sc if current_chunk else sc
                elif len(current_chunk) + len(separator) + len(split) > chunk_size and current_chunk:
                    final_chunks.append(current_chunk)
                    overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_chunk = overlap_text + separator + split if overlap_text else split
                else:
                    current_chunk = current_chunk + separator + split if current_chunk else split

            if current_chunk:
                final_chunks.append(current_chunk)

        return final_chunks


# ═══════════════════════════════════════════════
# 同时构建 NetworkX 知识图谱并加载到 KnowledgeGraph 对象
# ═══════════════════════════════════════════════

def build_knowledge_graph_from_jsonld(filepath: str) -> 'KnowledgeGraph':
    """
    从 JSON-LD 文件构建 KnowledgeGraph 对象

    这会同时构建:
    1. NetworkX 图 (用于图谱查询)
    2. 实体字典 (用于检索)
    3. 社区信息 (用于社区检测)

    Returns:
        KnowledgeGraph 实例
    """
    from knowledge_graph import KnowledgeGraph

    print(f"🔨 从 JSON-LD 构建知识图谱对象...")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data.get("@graph", [])
    kg = KnowledgeGraph()

    entity_type_uris = {
        "http://schema.org/PERSON",
        "http://schema.org/ORGANIZATION",
        "http://schema.org/EVENT",
        "http://schema.org/GEO",
        "http://schema.org/INDUSTRY",
        "http://schema.org/PRODUCT",
        "http://schema.org/ORGANISM",
    }

    # 第一遍: 添加所有实体
    for node in graph:
        node_type = node.get("@type", "")
        if node_type in entity_type_uris:
            name = node.get("schema:name") or node.get("rdfs:label") or _extract_name_from_id(node.get("@id", ""))
            entity_type_short = _extract_type_short(node_type)
            description = node.get("schema:description", "")
            frequency = node.get("kg:frequency", 0)
            degree = node.get("kg:degree", 0.0)

            kg.add_entity(
                name=name,
                entity_type=entity_type_short,
                description=description,
                frequency=int(frequency) if frequency else 0,
                degree=float(degree) if degree else 0.0,
            )

    # 第二遍: 添加所有关系
    for node in graph:
        node_type = node.get("@type", "")
        if node_type == "rdf:Statement":
            subject = node.get("rdf:subject", {})
            obj = node.get("rdf:object", {})
            predicate = node.get("rdf:predicate", {})
            description = node.get("schema:description", "")
            weight = node.get("kg:weight", 1.0)

            subject_name = _extract_name_from_id(subject.get("@id", "")) if isinstance(subject, dict) else ""
            object_name = _extract_name_from_id(obj.get("@id", "")) if isinstance(obj, dict) else ""
            predicate_name = _extract_name_from_id(predicate.get("@id", "")) if isinstance(predicate, dict) else ""

            if subject_name and object_name:
                kg.add_relation(
                    source=subject_name,
                    target=object_name,
                    relation_type=predicate_name,
                    description=description,
                    weight=float(weight) if weight else 1.0,
                )

    # 第三遍: 加载社区报告摘要
    for node in graph:
        node_type = node.get("@type", "")
        if node_type == "schema:Article":
            headline = node.get("schema:headline", "")
            summary = node.get("schema:description", "")
            community_id = _extract_name_from_id(node.get("@id", ""))

            # 将社区报告摘要存入 community_summaries
            try:
                cid = int(re.search(r'\d+', community_id).group()) if re.search(r'\d+', community_id) else 0
            except (ValueError, AttributeError):
                cid = hash(community_id) % 10000

            kg.community_summaries[cid] = summary or headline

    print(f"✅ 知识图谱对象构建完成:")
    stats = kg.get_statistics()
    print(f"   节点数: {stats['num_nodes']}")
    print(f"   边数: {stats['num_edges']}")
    print(f"   实体类型分布: {stats['entity_types']}")
    print(f"   社区摘要数: {len(kg.community_summaries)}")

    return kg


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="导入知识图谱数据到自适应 RAG 向量数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="知识图谱文件路径 (.jsonld 或 .csv)",
    )
    parser.add_argument(
        "--types",
        type=str,
        default="entity,relationship,community",
        help="要导入的数据类型，逗号分隔 (默认: entity,relationship,community)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="分块大小 (默认使用 config.py 中的 CHUNK_SIZE)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="分块重叠大小 (默认使用 config.py 中的 CHUNK_OVERLAP)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅解析文件，不写入向量库",
    )
    parser.add_argument(
        "--build-kg",
        action="store_true",
        help="同时构建 KnowledgeGraph 对象并保存为 knowledge_graph.json",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Milvus 集合名 (默认使用 config.py 中的 COLLECTION_NAME)",
    )
    args = parser.parse_args()

    filepath = args.filepath
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        sys.exit(1)

    # 解析数据类型
    include_types = set(args.types.split(","))
    valid_types = {"entity", "relationship", "community"}
    invalid_types = include_types - valid_types
    if invalid_types:
        print(f"⚠️ 无效的数据类型: {invalid_types}，有效类型: {valid_types}")
        include_types &= valid_types

    # ── 1. 解析文件 ─────────────────────
    print("\n" + "=" * 50)
    print("阶段 1/3: 解析知识图谱数据")
    print("=" * 50)

    ext = Path(filepath).suffix.lower()
    all_docs = []

    if ext == ".jsonld":
        entity_docs, relationship_docs, community_docs = parse_jsonld(filepath, include_types)
        all_docs = entity_docs + relationship_docs + community_docs

        if entity_docs:
            print(f"\n📊 实体统计:")
            type_counts = {}
            for doc in entity_docs:
                et = doc.metadata.get("kg_entity_type", "UNKNOWN")
                type_counts[et] = type_counts.get(et, 0) + 1
            for et, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"   {et}: {cnt} 个")

        if relationship_docs:
            total_weight = sum(d.metadata.get("kg_weight", 0) for d in relationship_docs)
            avg_weight = total_weight / len(relationship_docs) if relationship_docs else 0
            print(f"\n📊 关系统计:")
            print(f"   总关系数: {len(relationship_docs)}")
            print(f"   平均权重: {avg_weight:.2f}")

        if community_docs:
            print(f"\n📊 社区报告统计:")
            print(f"   总报告数: {len(community_docs)}")
            avg_len = sum(len(d.page_content) for d in community_docs) / len(community_docs)
            print(f"   平均长度: {avg_len:.0f} 字符")

    elif ext == ".csv":
        all_docs = parse_csv_triples(filepath)
    else:
        print(f"❌ 不支持的文件格式: {ext} (支持: .jsonld, .csv)")
        sys.exit(1)

    if not all_docs:
        print("❌ 没有解析到任何数据")
        sys.exit(1)

    print(f"\n✅ 总计解析到 {len(all_docs)} 个文档")

    # ── 2. 分块 ─────────────────────────
    from config import CHUNK_SIZE, CHUNK_OVERLAP

    chunk_size = args.chunk_size or CHUNK_SIZE
    chunk_overlap = args.chunk_overlap or CHUNK_OVERLAP

    print("\n" + "=" * 50)
    print("阶段 2/3: 知识图谱文档分块")
    print("=" * 50)

    splitter = KnowledgeGraphTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_splits = splitter.split_documents(all_docs)

    print(f"✅ 分块完成: {len(doc_splits)} 个文档块")

    if args.dry_run:
        print("\n🏁 --dry-run 模式，跳过向量库写入。前 5 个文档块预览:")
        for i, doc in enumerate(doc_splits[:5]):
            preview = doc.page_content[:200].replace("\n", "\\n")
            print(f"\n   [{i+1}] type={doc.metadata.get('data_type','?')} name={doc.metadata.get('kg_name', doc.metadata.get('kg_subject','?'))}")
            print(f"       {preview}...")

        # 也保存 knowledge_graph.json 如果需要
        if args.build_kg and ext == ".jsonld":
            print("\n🔨 构建 KnowledgeGraph 对象...")
            kg = build_knowledge_graph_from_jsonld(filepath)
            kg.save_to_file("knowledge_graph.json")
            print("✅ 已保存为 knowledge_graph.json")

        return

    # ── 3. 写入向量库 ──────────────────────
    print("\n" + "=" * 50)
    print("阶段 3/3: 写入向量数据库")
    print("=" * 50)

    from document_processor import DocumentProcessor

    print("⏳ 初始化 DocumentProcessor（加载嵌入模型，可能需要 10-30 秒）...")
    t0 = time.time()
    doc_processor = DocumentProcessor()
    print(f"✅ DocumentProcessor 初始化完成 ({time.time()-t0:.1f}s)")

    if not doc_processor.vectorstore:
        doc_processor.initialize_vectorstore()

    print(f"⏳ 正在向量化并写入 {len(doc_splits)} 个知识图谱文档块...")
    t1 = time.time()
    doc_processor.add_documents_to_vectorstore(doc_splits)
    elapsed = time.time() - t1
    print(f"✅ 写入完成! ({elapsed:.1f}s, {len(doc_splits)/elapsed:.1f} docs/s)")

    # ── 4. 构建 KnowledgeGraph 对象 ──────────
    if args.build_kg and ext == ".jsonld":
        print("\n" + "=" * 50)
        print("附加: 构建 KnowledgeGraph 对象")
        print("=" * 50)

        kg = build_knowledge_graph_from_jsonld(filepath)
        kg.save_to_file("knowledge_graph.json")
        print("✅ 已保存为 knowledge_graph.json，系统启动时可自动加载")

    # ── 5. 验证 ────────────────────────────
    print("\n" + "=" * 50)
    print("验证")
    print("=" * 50)

    try:
        # 测试检索实体
        test_queries = [
            ("KEVIN SCOTT", "kg_entity"),
            ("Behind the Tech podcast", "kg_entity"),
        ]
        for query, expected_type in test_queries:
            results = doc_processor.vectorstore.similarity_search(query, k=3)
            kg_types = [r.metadata.get("data_type", "") for r in results]
            kg_names = [r.metadata.get("kg_name", r.metadata.get("kg_subject", "")) for r in results]
            print(f"🔍 查询 '{query}': 检索到 {len(results)} 个结果")
            if results:
                print(f"   类型: {kg_types}")
                print(f"   名称: {kg_names[:3]}")
    except Exception as e:
        print(f"⚠️ 验证检索失败: {e}")

    print(f"\n🏁 全部完成! 共导入 {len(doc_splits)} 个知识图谱文档块")


if __name__ == "__main__":
    main()
