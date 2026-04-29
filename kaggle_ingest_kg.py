"""
知识图谱数据导入 Milvus Cloud (Zilliz Cloud)

在 Kaggle 上执行:
    1. clone 本项目到 Kaggle
    2. 在 Kaggle Notebook 的 Secrets 中添加 MILVUS_URI 和 MILVUS_PASSWORD
    3. 开启 Internet (Settings -> Internet -> On)
    4. 开启 GPU 加速 (Settings -> Accelerator -> GPU T4)
    5. python kaggle_ingest_kg.py

环境变量 (也可通过 Kaggle Secrets 自动设置):
    MILVUS_URI      - Zilliz Cloud URI
    MILVUS_PASSWORD - Zilliz Cloud API Key / Token
    KG_DATA_DIR     - 知识图谱数据目录 (默认自动检测: 脚本同级 source/knowledge_graph_export/)
    COLLECTION_NAME - Milvus 集合名 (默认 adaptive_rag)
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

# ═══════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════

# 尝试从 Kaggle Secrets 读取，失败则从环境变量读取
def _get_secret(key: str, default: str = "") -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key)
    except Exception:
        return os.environ.get(key, default)

MILVUS_URI = _get_secret("MILVUS_URI")
MILVUS_PASSWORD = _get_secret("MILVUS_PASSWORD")

# 自动检测项目目录: 脚本所在目录的 source/knowledge_graph_export/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_KG_DIR = os.path.join(_SCRIPT_DIR, "source", "knowledge_graph_export")
_DEFAULT_KG_OUTPUT = os.path.join(_SCRIPT_DIR, "knowledge_graph.json")

KG_DATA_DIR = os.environ.get("KG_DATA_DIR", _DEFAULT_KG_DIR)
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "adaptive_rag")
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
BATCH_SIZE = 500  # 每批写入数量

JSONLD_PATH = os.path.join(KG_DATA_DIR, "knowledge_graph.jsonld")
KG_OUTPUT_PATH = _DEFAULT_KG_OUTPUT

# Kaggle /kaggle/working 输出目录（如果存在则优先保存到那里）
if os.path.exists("/kaggle/working"):
    KG_OUTPUT_PATH = "/kaggle/working/knowledge_graph.json"

# 本地模式从 .env 读取配置
_env_file = os.path.join(_SCRIPT_DIR, ".env")
if os.path.exists(_env_file):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
        if not MILVUS_URI:
            MILVUS_URI = os.environ.get("MILVUS_URI", "")
        if not MILVUS_PASSWORD:
            MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD", "")
    except ImportError:
        # 手动解析 .env
        with open(_env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k == "MILVUS_URI" and not MILVUS_URI:
                        MILVUS_URI = v
                    elif k == "MILVUS_PASSWORD" and not MILVUS_PASSWORD:
                        MILVUS_PASSWORD = v


# ═══════════════════════════════════════════════
# 0. 安装依赖
# ═══════════════════════════════════════════════

def install_dependencies():
    """检查并安装缺失的依赖"""
    required = {
        "langchain_core": "langchain-core",
        "langchain_huggingface": "langchain-huggingface",
        "langchain_milvus": "langchain-milvus",
        "langchain_text_splitters": "langchain-text-splitters",
        "pymilvus": "pymilvus",
        "sentence_transformers": "sentence-transformers",
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"📦 安装缺失依赖: {missing}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("✅ 依赖安装完成")
    else:
        print("✅ 所有依赖已就绪")


# ═══════════════════════════════════════════════
# 1. JSON-LD 解析器
# ═══════════════════════════════════════════════

def _extract_name(id_uri: str) -> str:
    return id_uri.rstrip("/").split("/")[-1]

def _extract_type(type_uri: str) -> str:
    return type_uri.rstrip("/").split("/")[-1]

ENTITY_TYPE_URIS = {
    "http://schema.org/PERSON",
    "http://schema.org/ORGANIZATION",
    "http://schema.org/EVENT",
    "http://schema.org/GEO",
    "http://schema.org/INDUSTRY",
    "http://schema.org/PRODUCT",
    "http://schema.org/ORGANISM",
}


def parse_jsonld(filepath: str, include_types: Optional[set] = None) -> Tuple[List, List, List]:
    """解析 JSON-LD 知识图谱 -> (entity_docs, rel_docs, comm_docs)"""
    from langchain_core.documents import Document

    if include_types is None:
        include_types = {"entity", "relationship", "community"}

    print(f"📂 读取 JSON-LD: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data.get("@graph", [])
    print(f"   读取到 {len(graph)} 个图节点")

    entity_docs, rel_docs, comm_docs = [], [], []

    for node in graph:
        node_type = node.get("@type", "")
        node_id = node.get("@id", "")

        # 实体
        if node_type in ENTITY_TYPE_URIS and "entity" in include_types:
            name = node.get("schema:name") or node.get("rdfs:label") or _extract_name(node_id)
            etype = _extract_type(node_type)
            desc = node.get("schema:description", "")
            freq = node.get("kg:frequency", 0)
            deg = node.get("kg:degree", 0.0)

            content = f"{name} ({etype}): {desc}".strip()
            entity_docs.append(Document(
                page_content=content,
                metadata={
                    "source": filepath, "data_type": "kg_entity",
                    "kg_name": name, "kg_entity_type": etype,
                    "kg_frequency": int(freq) if freq else 0,
                    "kg_degree": float(deg) if deg else 0.0,
                    "file_type": "jsonld",
                }
            ))

        # 关系
        elif node_type == "rdf:Statement" and "relationship" in include_types:
            subj = node.get("rdf:subject", {})
            pred = node.get("rdf:predicate", {})
            obj = node.get("rdf:object", {})
            desc = node.get("schema:description", "")
            weight = node.get("kg:weight", 1.0)

            s_name = _extract_name(subj.get("@id", "")) if isinstance(subj, dict) else str(subj)
            p_name = _extract_name(pred.get("@id", "")) if isinstance(pred, dict) else str(pred)
            o_name = _extract_name(obj.get("@id", "")) if isinstance(obj, dict) else str(obj)

            content = f"{s_name} --[{p_name}]--> {o_name}: {desc}".strip()
            rel_docs.append(Document(
                page_content=content,
                metadata={
                    "source": filepath, "data_type": "kg_relationship",
                    "kg_subject": s_name, "kg_predicate": p_name, "kg_object": o_name,
                    "kg_weight": float(weight) if weight else 1.0,
                    "file_type": "jsonld",
                }
            ))

        # 社区报告
        elif node_type == "schema:Article" and "community" in include_types:
            headline = node.get("schema:headline", "")
            summary = node.get("schema:description", "")
            full = node.get("schema:text", "")
            cid = _extract_name(node_id)

            content = full or f"{headline}\n\n{summary}"
            meta = {
                "source": filepath, "data_type": "kg_community",
                "kg_community_id": cid, "kg_headline": headline,
                "file_type": "jsonld",
            }
            for k, v in [("kg_level", "kg:level"), ("kg_rank", "kg:rank"), ("kg_size", "kg:size")]:
                val = node.get(v)
                if val is not None:
                    meta[k] = int(val) if k != "kg_rank" else float(val)

            comm_docs.append(Document(page_content=content.strip(), metadata=meta))

    print(f"✅ 解析完成: {len(entity_docs)} 实体, {len(rel_docs)} 关系, {len(comm_docs)} 社区")
    return entity_docs, rel_docs, comm_docs


# ═══════════════════════════════════════════════
# 2. 文本分块器 (纯 Python)
# ═══════════════════════════════════════════════

class KGTextSplitter:
    """知识图谱文档分块器"""

    def __init__(self, chunk_size=1024, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs: List) -> List:
        from langchain_core.documents import Document
        result = []
        for doc in docs:
            dt = doc.metadata.get("data_type", "")
            if dt == "kg_community" and len(doc.page_content) > self.chunk_size:
                result.extend(self._split_community(doc))
            elif len(doc.page_content) > self.chunk_size:
                result.extend(self._split_generic(doc))
            else:
                result.append(doc)
        return result

    def _split_community(self, doc) -> List:
        from langchain_core.documents import Document
        content = doc.page_content
        headline = doc.metadata.get("kg_headline", "")
        prefix = f"[社区报告: {headline}]\n\n" if headline else ""
        sections = re.split(r'\n(?=## )', content)
        result, chunk, clen = [], prefix, len(prefix)
        for sec in sections:
            if clen + len(sec) > self.chunk_size and clen > len(prefix):
                result.append(Document(page_content=chunk.strip(), metadata={**doc.metadata, "chunk_index": len(result)}))
                chunk = prefix + sec
                clen = len(prefix) + len(sec)
            else:
                chunk += "\n" + sec
                clen += len(sec) + 1
        if chunk.strip():
            result.append(Document(page_content=chunk.strip(), metadata={**doc.metadata, "chunk_index": len(result)}))
        return result

    def _split_generic(self, doc) -> List:
        from langchain_core.documents import Document
        seps = ["\n\n", "\n", "。", "，", " ", ""]
        splits = self._rec_split(doc.page_content, seps)
        return [Document(page_content=s.strip(), metadata={**doc.metadata, "chunk_index": i})
                for i, s in enumerate(splits) if s.strip()]

    def _rec_split(self, text: str, seps: list) -> list:
        if len(text) <= self.chunk_size:
            return [text]
        sep = seps[0] if seps else ""
        rest = seps[1:] if len(seps) > 1 else []
        chunks, cur = [], ""
        if not sep:
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i:i + self.chunk_size])
        else:
            for part in text.split(sep):
                if len(part) > self.chunk_size and rest:
                    for sc in self._rec_split(part, rest):
                        if len(cur) + len(sep) + len(sc) > self.chunk_size and cur:
                            chunks.append(cur)
                            cur = cur[-self.chunk_overlap:] + sep + sc if self.chunk_overlap else sc
                        else:
                            cur = cur + sep + sc if cur else sc
                elif len(cur) + len(sep) + len(part) > self.chunk_size and cur:
                    chunks.append(cur)
                    cur = cur[-self.chunk_overlap:] + sep + part if self.chunk_overlap else part
                else:
                    cur = cur + sep + part if cur else part
            if cur:
                chunks.append(cur)
        return chunks


# ═══════════════════════════════════════════════
# 3. 构建 KnowledgeGraph JSON
# ═══════════════════════════════════════════════

def build_kg_json(filepath: str, output_path: str):
    """从 JSON-LD 构建 NetworkX 图谱并保存为 JSON"""
    import networkx as nx

    print("🔨 构建 NetworkX 知识图谱...")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    graph = data.get("@graph", [])

    G = nx.Graph()
    entities = {}
    community_summaries = {}

    for node in graph:
        if node.get("@type", "") in ENTITY_TYPE_URIS:
            name = node.get("schema:name") or _extract_name(node.get("@id", ""))
            etype = _extract_type(node["@type"])
            desc = node.get("schema:description", "")
            freq = node.get("kg:frequency", 0)
            deg = node.get("kg:degree", 0.0)
            G.add_node(name, type=etype, description=desc,
                       frequency=int(freq) if freq else 0,
                       degree=float(deg) if deg else 0.0)
            entities[name] = {"name": name, "type": etype, "description": desc,
                              "frequency": int(freq) if freq else 0,
                              "degree": float(deg) if deg else 0.0}

    for node in graph:
        if node.get("@type", "") == "rdf:Statement":
            subj = node.get("rdf:subject", {})
            obj = node.get("rdf:object", {})
            pred = node.get("rdf:predicate", {})
            desc = node.get("schema:description", "")
            weight = node.get("kg:weight", 1.0)
            s_name = _extract_name(subj.get("@id", "")) if isinstance(subj, dict) else ""
            o_name = _extract_name(obj.get("@id", "")) if isinstance(obj, dict) else ""
            p_name = _extract_name(pred.get("@id", "")) if isinstance(pred, dict) else ""
            if s_name in G and o_name in G:
                G.add_edge(s_name, o_name, relation_type=p_name, description=desc,
                           weight=float(weight) if weight else 1.0)

    for node in graph:
        if node.get("@type", "") == "schema:Article":
            cid = _extract_name(node.get("@id", ""))
            summary = node.get("schema:description", "")
            headline = node.get("schema:headline", "")
            try:
                cid_int = int(re.search(r'\d+', cid).group()) if re.search(r'\d+', cid) else 0
            except (ValueError, AttributeError):
                cid_int = hash(cid) % 10000
            community_summaries[cid_int] = summary or headline

    print(f"   节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
    print(f"   社区报告: {len(community_summaries)}")

    # 社区检测
    communities = []
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        print(f"   检测到 {len(communities)} 个社区")
    except Exception as e:
        print(f"   社区检测失败: {e}")

    kg_data = {
        "entities": entities,
        "communities": {str(i): list(c) for i, c in enumerate(communities)},
        "community_summaries": {str(k): v for k, v in community_summaries.items()},
        "edges": [
            {"source": u, "target": v, "data": {k: vv for k, vv in d.items()}}
            for u, v, d in G.edges(data=True)
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ 已保存 {output_path} ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("知识图谱数据导入 Milvus Cloud")
    print("=" * 60)

    # 0. 前置检查
    if not MILVUS_URI or not MILVUS_PASSWORD:
        print("❌ 缺少 MILVUS_URI 或 MILVUS_PASSWORD")
        print("   Kaggle: 在 Secrets 中添加 MILVUS_URI 和 MILVUS_PASSWORD")
        print("   本地:   在 .env 文件中设置")
        sys.exit(1)

    if not os.path.exists(JSONLD_PATH):
        print(f"❌ 数据文件不存在: {JSONLD_PATH}")
        print(f"   目录内容: {os.listdir(KG_DATA_DIR) if os.path.exists(KG_DATA_DIR) else '目录不存在'}")
        sys.exit(1)

    print(f"\n数据目录: {KG_DATA_DIR}")
    print(f"数据文件: {JSONLD_PATH}")
    print(f"Milvus URI: {MILVUS_URI[:50]}...")

    # 1. 安装依赖
    install_dependencies()

    # 2. 解析 JSON-LD
    print("\n" + "=" * 50)
    print("阶段 1/4: 解析知识图谱数据")
    print("=" * 50)

    entity_docs, rel_docs, comm_docs = parse_jsonld(JSONLD_PATH)
    all_docs = entity_docs + rel_docs + comm_docs

    print(f"\n📊 数据统计:")
    print(f"   实体: {len(entity_docs)}")
    if entity_docs:
        tc = {}
        for d in entity_docs:
            tc[d.metadata['kg_entity_type']] = tc.get(d.metadata['kg_entity_type'], 0) + 1
        for t, c in sorted(tc.items(), key=lambda x: -x[1]):
            print(f"     {t}: {c}")
    print(f"   关系: {len(rel_docs)}")
    print(f"   社区: {len(comm_docs)}")
    print(f"   总计: {len(all_docs)}")

    if not all_docs:
        print("❌ 没有解析到数据")
        sys.exit(1)

    # 3. 分块
    print("\n" + "=" * 50)
    print("阶段 2/4: 文本分块")
    print("=" * 50)

    splitter = KGTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    doc_splits = splitter.split_documents(all_docs)
    print(f"✅ 分块完成: {len(doc_splits)} 个文档块")

    # 4. 初始化嵌入模型 + Milvus
    print("\n" + "=" * 50)
    print("阶段 3/4: 初始化嵌入模型 & 写入向量库")
    print("=" * 50)

    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus
    from pymilvus import connections, utility

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    t0 = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"✅ 嵌入模型加载完成 ({time.time()-t0:.1f}s)")

    # 连接 Milvus
    print(f"\n🔌 连接 Zilliz Cloud...")
    connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_PASSWORD)
    print("✅ 已连接")

    collections = utility.list_collections()
    if COLLECTION_NAME in collections:
        from pymilvus import Collection
        col = Collection(COLLECTION_NAME)
        print(f"集合 '{COLLECTION_NAME}' 现有 {col.num_entities} 条文档 (将追加)")
    else:
        print(f"集合 '{COLLECTION_NAME}' 不存在，将自动创建")

    # 初始化向量库
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI, "token": MILVUS_PASSWORD},
        index_params={
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        },
        search_params={
            "metric_type": "L2",
            "params": {"ef": 10},
        },
        drop_old=False,  # 保留已有数据
        auto_id=True,
    )
    print("✅ Milvus 向量库初始化完成")

    # 分批写入
    total = len(doc_splits)
    t_start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = doc_splits[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        done = min(i + BATCH_SIZE, total)
        elapsed = time.time() - t_start
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed / 60 if speed > 0 else 0
        print(f"   写入 {done}/{total} ({done/total*100:.1f}%) | {speed:.1f} docs/s | ETA {eta:.1f} min")

    elapsed = time.time() - t_start
    print(f"\n✅ 向量库写入完成! {total} 条文档, 耗时 {elapsed:.1f}s ({total/elapsed:.1f} docs/s)")

    # 5. 构建 KnowledgeGraph JSON
    print("\n" + "=" * 50)
    print("阶段 4/4: 构建 KnowledgeGraph JSON")
    print("=" * 50)

    build_kg_json(JSONLD_PATH, KG_OUTPUT_PATH)

    # 6. 验证
    print("\n" + "=" * 50)
    print("验证检索效果")
    print("=" * 50)

    test_queries = ["KEVIN SCOTT", "Behind the Tech podcast", "Microsoft acquisition"]
    for q in test_queries:
        results = vectorstore.similarity_search(q, k=3)
        print(f"\n🔍 查询: '{q}'")
        for i, r in enumerate(results):
            dt = r.metadata.get("data_type", "?")
            name = r.metadata.get("kg_name", r.metadata.get("kg_subject", "?"))
            print(f"  [{i+1}] [{dt}] {name} | {r.page_content[:80]}...")

    print(f"\n🏁 全部完成!")


if __name__ == "__main__":
    main()
