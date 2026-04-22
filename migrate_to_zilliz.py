"""
数据迁移脚本：从本地 Milvus Lite 迁移到 Zilliz Cloud（直接迁移向量，不重新生成）

使用方法：
1. 设置 MILVUS_READONLY_URI 指向本地 .db 文件
2. 确保 MILVUS_URI 指向 Zilliz Cloud，MILVUS_PASSWORD 为 API Key
3. 运行：python migrate_to_zilliz.py
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COLLECTION_NAME


def get_local_data():
    """使用 pymilvus 直接读取本地 Milvus Lite（包含向量）"""
    print("=" * 60)
    print("Step 1: 连接本地 Milvus Lite，读取文档和向量...")
    print("=" * 60)

    from pymilvus import connections, Collection

    local_db_path = os.environ.get("MILVUS_READONLY_URI") or "./milvus_rag.db"
    print(f"📂 原始数据库路径: {local_db_path}")

    if not os.path.exists(local_db_path):
        print(f"❌ 本地 Milvus 文件不存在: {local_db_path}")
        return [], None, None

    # Kaggle /kaggle/input/ 是只读的，复制到可写目录
    if "/kaggle/input/" in local_db_path:
        writable_path = "/kaggle/working/milvus_rag_copy.db"
        if not os.path.exists(writable_path):
            print(f"📂 Kaggle 只读，复制到可写目录: {writable_path}")
            shutil.copy2(local_db_path, writable_path)
            print("✅ 复制完成")
        else:
            print(f"ℹ️  使用已有副本: {writable_path}")
        local_db_path = writable_path

    print(f"📂 正在连接本地 Milvus Lite...")
    connections.connect(alias="default", uri=local_db_path)
    print("✅ 本地连接成功")

    # 查看所有 collection
    from pymilvus import utility
    all_collections = utility.list_collections()
    print(f"📋 本地 Collections: {all_collections}")

    local_collection_name = "rag_milvus"
    if not utility.has_collection(local_collection_name):
        print(f"❌ Collection '{local_collection_name}' 不存在")
        connections.disconnect("default")
        return [], None, None

    # 加载 Collection
    collection = Collection(local_collection_name)
    collection.load()

    # 分析 schema
    schema = collection.schema
    pk_field = schema.primary_field.name
    text_field = None
    vector_field = None
    vector_dim = None

    for field in schema.fields:
        if field.name == "text":
            text_field = field.name
        elif field.dtype.__class__.__name__ == "DataType" and field.name in ["vector", "embedding"]:
            vector_field = field.name
            # 获取维度
            if hasattr(field, "params") and "dim" in field.params:
                vector_dim = field.params["dim"]
            elif hasattr(field, "dimension"):
                vector_dim = field.dimension

    print(f"📋 PK={pk_field}, text={text_field}, vector={vector_field}, dim={vector_dim}")

    # 查询所有数据
    batch_limit = 16384
    all_res = []
    offset = 0

    print(f"📖 开始读取文档（每批 {batch_limit} 条）...")
    while True:
        try:
            res = collection.query(
                expr=f"{pk_field} >= 0",
                output_fields=["*"],
                limit=batch_limit,
                offset=offset
            )
        except Exception as e:
            print(f"⚠️ 查询失败: {e}")
            break

        if not res:
            break

        all_res.extend(res)
        print(f"   已读取 {len(all_res)} 条...")
        if len(res) < batch_limit:
            break
        offset += batch_limit

    print(f"✅ 共读取 {len(all_res)} 条数据")

    connections.disconnect("default")

    return all_res, vector_field, vector_dim


def write_to_zilliz(data, vector_field, vector_dim):
    """直接使用 pymilvus 将数据写入 Zilliz Cloud（保留原始向量）"""
    if not data:
        print("⚠️ 没有数据需要写入")
        return

    print("\n" + "=" * 60)
    print("Step 2: 连接 Zilliz Cloud，写入文档...")
    print("=" * 60)

    zilliz_uri = os.environ.get("ZILLIZ_URI") or os.environ.get("MILVUS_URI")
    zilliz_password = os.environ.get("MILVUS_PASSWORD") or os.environ.get("ZILLIZ_API_KEY")

    if not zilliz_uri or not zilliz_password:
        print("❌ 未配置 Zilliz Cloud")
        return

    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

    print(f"☁️  正在连接 Zilliz Cloud: {zilliz_uri}")
    connections.connect(alias="default", uri=zilliz_uri, token=zilliz_password)
    print("✅ Zilliz Cloud 连接成功")

    # 删除旧 collection（如果存在）
    if utility.has_collection(COLLECTION_NAME):
        print(f"🗑️  删除旧 collection: {COLLECTION_NAME}")
        utility.drop_collection(COLLECTION_NAME)

    # 创建新 collection，维度与本地一致
    print(f"📋 创建新 Collection '{COLLECTION_NAME}'（维度={vector_dim}）...")
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields=fields, description="Adaptive RAG")
    new_col = Collection(name=COLLECTION_NAME, schema=schema)

    # 创建索引
    index_params = {"metric_type": "L2", "index_type": "AUTOINDEX", "params": {}}
    new_col.create_index(field_name=vector_field, index_params=index_params)
    print("✅ Collection 创建完成")

    # 准备写入数据
    print(f"📝 正在写入 {len(data)} 条数据到 Zilliz Cloud...")

    # 分批写入
    batch_size = 50
    total_inserted = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        entities = []
        for item in batch:
            entities.append({
                "text": item.get("text", ""),
                vector_field: item.get(vector_field, []),
                "source": item.get("source", ""),
                "data_type": item.get("data_type", ""),
            })

        try:
            new_col.insert(entities)
            total_inserted += len(entities)
            print(f"   已写入 {total_inserted}/{len(data)} 条...")
        except Exception as e:
            print(f"⚠️ 写入批次 {i//batch_size} 失败: {e}")
            continue

    # 刷新
    new_col.flush()
    print(f"✅ 成功写入 {total_inserted} 条数据到 Zilliz Cloud!")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   向量维度: {vector_dim}")

    connections.disconnect("default")


def main():
    print("\n🚀 开始数据迁移：本地 Milvus Lite → Zilliz Cloud\n")

    data, vector_field, vector_dim = get_local_data()

    if not data:
        print("⚠️ 本地没有找到数据，迁移终止")
        return

    write_to_zilliz(data, vector_field, vector_dim)

    print("\n" + "=" * 60)
    print("🎉 迁移完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
