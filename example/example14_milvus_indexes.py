"""
Example 14: Milvus 索引类型演示
Milvus 支持的主流索引类型及其搜索性能

Milvus 支持的索引类型：
1. FLAT - 暴力搜索，适合小数据量，100% 召回率
2. IVF_FLAT - 倒排索引，分桶量化，适合中等数据量
3. HNSW - 图索引，高召回率，适合高质量搜索
4. AUTOINDEX - 自动索引，Milvus 自动优化参数
"""

import time
from pymilvus import MilvusClient, DataType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# ==================== 配置 ====================

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
EMBEDDING_MODEL = "/Users/jiang/models/embedding_models/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# 索引配置
INDEX_CONFIGS = {
    # FLAT: 暴力搜索，精确匹配
    "FLAT": {
        "index_type": "FLAT",
        "metric_type": "L2",
        "params": {}
    },
    # IVF_FLAT: 倒排索引，分桶暴力搜索
    "IVF_FLAT": {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nprobe": 10,"nlist": 100}
    },
    # HNSW: 图索引，高性能近似搜索
    "HNSW": {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"efConstruction": 200, "M": 16}
    },
    # AUTOINDEX: 自动优化索引
    "AUTOINDEX": {
        "index_type": "AUTOINDEX",
        "metric_type": "L2",
        "params": {}
    },
}

# ==================== 1. 准备测试数据 ====================

def generate_test_documents(num_docs: int = 100):
    """生成测试文档"""
    base_topics = [
        "银行电汇需要准备哪些信息？收款人姓名、账户号、开户行名称及SWIFT代码。",
        "第三方支付平台有哪些？PayPal、支付宝国际版、微信支付、Visa、Mastercard等。",
        "什么是SWIFT代码？SWIFT代码是银行国际代码，用于国际电汇识别银行。",
        "如何开设海外银行账户？需要准备护照、签证、居住证明等文件。",
        "电汇和票汇有什么区别？电汇是电子转账速度快，票汇是汇票形式速度慢但费用低。",
        "什么是结汇？结汇是将外汇兑换成本币的过程。",
        "个人每年购汇额度是多少？境内居民每年等值5万美元的购汇额度。",
        "西联汇款如何操作？到西联网点填写汇款单，支付汇款金额和手续费即可。",
        "什么是备用信用证？备用信用证是银行出具的支付保证函。",
        "如何查询汇款进度？可通过银行柜台、网上银行或电话查询。",
    ]

    documents = []
    for i in range(num_docs):
        base_doc = base_topics[i % len(base_topics)]
        variant = f"文档 {i + 1} - " + base_doc
        documents.append({
            "text": variant,
            "metadata": {"source": f"doc_{i + 1}", "original_topic": i % len(base_topics)}
        })
    return documents

# ==================== 2. 创建 Collection 并构建索引 ====================

def create_collection_with_index(
    client: MilvusClient,
    embeddings,
    collection_name: str,
    index_config: dict,
    documents: list,
):
    """创建指定索引类型的 Collection"""

    print(f"\n{'='*60}")
    print(f"创建 Collection: {collection_name}")
    print(f"索引类型: {index_config['index_type']}")
    print(f"{'='*60}")

    # 删除旧 Collection
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"已删除旧 Collection: {collection_name}")

    # 创建 Schema
    schema = client.create_schema(auto_id=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    # 创建索引参数
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type=index_config["index_type"],
        metric_type=index_config["metric_type"],
        params=index_config["params"]
    )

    # 创建 Collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    client.load_collection(collection_name=collection_name)
    print(f"Collection 创建成功")

    # 生成向量
    print("生成向量嵌入...")
    texts = [doc["text"] for doc in documents]
    dense_vectors = embeddings.embed_documents(texts)

    # 准备插入数据
    insert_data = []
    for i, doc in enumerate(documents):
        insert_data.append({
            "text": doc["text"],
            "dense": dense_vectors[i][:EMBEDDING_DIM],
        })

    # 插入数据
    start_time = time.time()
    client.insert(collection_name=collection_name, data=insert_data)
    client.flush(collection_name)
    add_time = time.time() - start_time

    print(f"插入 {len(insert_data)} 个文档，耗时: {add_time:.3f}s")

    return add_time

# ==================== 3. 搜索测试 ====================

def search_and_measure(client: MilvusClient, collection_name: str, embeddings, query: str, k: int = 5):
    """执行搜索并测量时间"""
    # 生成查询向量
    query_vector = embeddings.embed_query(query)[:EMBEDDING_DIM]

    # 搜索
    start_time = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=k,
        search_params={"metric_type": "L2"},
        output_fields=["text"]
    )
    search_time = time.time() - start_time

    return results, search_time

# ==================== 4. 主函数 ====================

def main():
    print("=" * 70)
    print("Milvus 索引类型演示")
    print("=" * 70)

    # 准备数据
    print("\n准备测试数据...")
    documents = generate_test_documents(100)
    print(f"生成 {len(documents)} 个测试文档")

    # 创建 embedding 模型
    print("加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 连接 Milvus
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    # 测试查询
    test_queries = [
        "银行电汇需要准备什么信息？",
        "第三方支付平台有哪些？",
        "SWIFT代码是什么？",
    ]

    # 逐个索引类型进行测试
    results_summary = {}

    for index_name, index_config in INDEX_CONFIGS.items():
        collection_name = f"index_demo_{index_name.lower()}"

        # 创建 Collection
        add_time = create_collection_with_index(
            client=client,
            embeddings=embeddings,
            collection_name=collection_name,
            index_config=index_config,
            documents=documents,
        )

        # 等待索引构建
        print("等待索引构建完成...")
        time.sleep(1)

        # 执行多次查询取平均
        search_times = []
        for query in test_queries:
            _, search_time = search_and_measure(client, collection_name, embeddings, query, k=5)
            search_times.append(search_time)

        avg_search_time = sum(search_times) / len(search_times)
        results_summary[index_name] = {
            "add_time": add_time,
            "avg_search_time": avg_search_time,
        }

        print(f"平均搜索耗时: {avg_search_time*1000:.2f}ms")

        # 示例搜索结果
        print(f"\n搜索示例 (查询: '{test_queries[0]}')")
        print("-" * 60)
        search_results, _ = search_and_measure(client, collection_name, embeddings, test_queries[0], k=3)
        for hits in search_results:
            for i, hit in enumerate(hits, 1):
                text = hit["entity"].get("text", "")[:50]
                print(f"  {i}. {text}... (distance: {hit['distance']:.4f})")

    # ==================== 5. 结果对比 ====================

    print("\n" + "=" * 70)
    print("索引性能对比")
    print("=" * 70)
    print(f"{'索引类型':<15} {'插入时间(s)':<15} {'平均搜索时间(ms)':<20}")
    print("-" * 50)
    for index_name, stats in results_summary.items():
        print(f"{index_name:<15} {stats['add_time']:<15.3f} {stats['avg_search_time']*1000:<20.2f}")

    # ==================== 6. 索引类型说明 ====================

    print("\n" + "=" * 70)
    print("索引类型说明")
    print("=" * 70)

    index_descriptions = """
    ┌──────────────┬─────────────────────────────────────────────────────────────┐
    │ 索引类型     │ 说明                                                         │
    ├──────────────┼─────────────────────────────────────────────────────────────┤
    │ FLAT         │ 暴力搜索，扫描所有向量。与目标向量计算 L2/IP 距离。           │
    │              │ 优点：100% 召回率，精确匹配                                  │
    │              │ 缺点：大数据量时速度慢，内存占用高                            │
    │              │ 适用：小数据量（<10万）需要精确结果的场景                     │
    ├──────────────┼─────────────────────────────────────────────────────────────┤
    │ IVF_FLAT     │ 倒排索引 + 分桶。将向量聚类到 N 个桶，搜索时只扫描相关桶。     │
    │              │ 优点：比 FLAT 快，可调召回率                                  │
    │              │ 缺点：需要调优 nprobe 参数                                     │
    │              │ 适用：中等数据量（10万-100万）                                │
    ├──────────────┼─────────────────────────────────────────────────────────────┤
    │ HNSW         │ 分层可导航小世界图。图索引，搜索时沿边快速定位近邻。           │
    │              │ 优点：高速，高召回率                                          │
    │              │ 缺点：内存占用高，构建慢                                       │
    │              │ 适用：高质量搜索，数据量 100万+                               │
    ├──────────────┼─────────────────────────────────────────────────────────────┤
    │ AUTOINDEX    │ Milvus 自动优化索引参数。根据数据特征自动选择最优配置。        │
    │              │ 优点：使用简单，Milvus 自动优化                               │
    │              │ 缺点：黑盒调优，不可控                                        │
    │              │ 适用：生产环境，不知道如何选择参数时                           │
    └──────────────┴─────────────────────────────────────────────────────────────┘

    重要参数说明：
    - nprobe (IVF): 搜索的桶数量，越多越精确但越慢
    - efConstruction (HNSW): 构建时的搜索范围，越大质量越高但构建越慢
    - M (HNSW): 每个节点的连接数，越大图越密，搜索越快但内存越高
    - ef (HNSW): 搜索时的搜索范围，越大越精确但越慢
    """

    print(index_descriptions)

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()