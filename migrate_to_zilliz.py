"""
数据迁移脚本：从本地 Milvus Lite 迁移到 Zilliz Cloud

使用方法：
1. 设置 MILVUS_READONLY_URI 指向本地 .db 文件
2. 确保 MILVUS_URI 指向 Zilliz Cloud，MILVUS_PASSWORD 为 API Key
3. 运行：python migrate_to_zilliz.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 必须在 import langchain 之前设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from config import COLLECTION_NAME, EMBEDDING_MODEL


def get_local_documents_via_pymilvus():
    """使用 pymilvus 直接读取本地 Milvus Lite"""
    print("=" * 60)
    print("Step 1: 连接本地 Milvus Lite，读取文档...")
    print("=" * 60)

    import pymilvus
    from pymilvus import connections, Collection

    local_db_path = os.environ.get("MILVUS_READONLY_URI") or "./milvus_rag.db"
    print(f"📂 原始数据库路径: {local_db_path}")

    if not os.path.exists(local_db_path):
        print(f"❌ 本地 Milvus 文件不存在: {local_db_path}")
        return []

    # Kaggle /kaggle/input/ 是只读的，Milvus Lite 需要写 lock 文件
    # 因此需要先复制到可写目录
    import shutil
    if "/kaggle/input/" in local_db_path:
        writable_path = "/kaggle/working/milvus_rag_copy.db"
        if not os.path.exists(writable_path):
            print(f"📂 Kaggle 只读文件系统，复制到可写目录: {writable_path}")
            shutil.copy2(local_db_path, writable_path)
            print("✅ 复制完成")
        else:
            print(f"ℹ️  使用已有副本: {writable_path}")
        local_db_path = writable_path

    # 连接本地 Milvus Lite
    print(f"📂 正在连接本地 Milvus Lite...")
    connections.connect(alias="default", uri=local_db_path)
    print("✅ 本地连接成功")

    # 检查 Collection 是否存在
    from pymilvus import utility
    print(f"\n📋 查看本地数据库中的所有 Collection:")
    all_collections = utility.list_collections()
    print(f"   现有 collections: {all_collections}")

    if not utility.has_collection(COLLECTION_NAME):
        print(f"❌ Collection '{COLLECTION_NAME}' 不存在")
        print(f"   尝试查找可能的 collection 名...")
        # 尝试模糊匹配
        for name in all_collections:
            if "rag" in name.lower() or "milvus" in name.lower() or "adaptive" in name.lower():
                print(f"   找到可能的匹配: {name}")
        connections.disconnect("default")
        return []

    # 加载 Collection
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 获取 schema 信息
    schema = collection.schema
    pk_field = schema.primary_field.name
    text_field = None
    for field in schema.fields:
        if field.name == "text":
            text_field = "text"
            break
    if not text_field:
        text_field = "text"

    print(f"📋 PK field: {pk_field}, text field: {text_field}")

    # 查询所有文档
    expr = f"{pk_field} >= 0"
    try:
        res = collection.query(expr=expr, output_fields=["*"], limit=100000)
    except Exception as e:
        print(f"⚠️ 首次查询失败: {e}")
        expr = f'{pk_field} != ""'
        res = collection.query(expr=expr, output_fields=["*"], limit=100000)

    docs = []
    for item in res:
        content = item.get(text_field, "")
        metadata = {
            "source": item.get("source", ""),
            "data_type": item.get("data_type", "")
        }
        docs.append(Document(page_content=content, metadata=metadata))

    print(f"✅ 从本地 Milvus 读取了 {len(docs)} 个文档")

    # 断开连接
    connections.disconnect("default")

    return docs


def write_to_zilliz(docs):
    """将文档写入 Zilliz Cloud"""
    if not docs:
        print("⚠️ 没有文档需要写入")
        return

    print("\n" + "=" * 60)
    print("Step 2: 连接 Zilliz Cloud，写入文档...")
    print("=" * 60)

    zilliz_uri = os.environ.get("ZILLIZ_URI") or os.environ.get("MILVUS_URI")
    zilliz_password = os.environ.get("MILVUS_PASSWORD") or os.environ.get("ZILLIZ_API_KEY")

    if not zilliz_uri or not zilliz_password:
        print("❌ 未配置 Zilliz Cloud，请确保环境变量 MILVUS_URI 和 MILVUS_PASSWORD 已设置")
        return

    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"☁️  正在连接 Zilliz Cloud: {zilliz_uri}")

    zilliz_vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "uri": zilliz_uri,
            "token": zilliz_password,
        },
        index_params={"metric_type": "L2", "index_type": "AUTOINDEX", "params": {}},
        search_params={"metric_type": "L2", "params": {"ef": 10}},
        auto_id=True,
    )

    print(f"📝 正在写入 {len(docs)} 个文档到 Zilliz Cloud...")

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        zilliz_vectorstore.add_documents(batch)
        print(f"   已写入 {min(i + batch_size, len(docs))}/{len(docs)} 个文档")

    print(f"✅ 成功写入 {len(docs)} 个文档到 Zilliz Cloud!")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   访问地址: {zilliz_uri}")


def main():
    print("\n🚀 开始数据迁移：本地 Milvus Lite → Zilliz Cloud\n")

    docs = get_local_documents_via_pymilvus()

    if not docs:
        print("⚠️ 本地没有找到文档，迁移终止")
        return

    write_to_zilliz(docs)

    print("\n" + "=" * 60)
    print("🎉 迁移完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
