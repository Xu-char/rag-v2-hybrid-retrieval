"""
Milvus 混合搜索
BM25 + 向量检索 + 混合检索
"""

from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, LexicalHighlighter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# ==================== 配置 ====================
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "banking_rag_hybrid"
EMBEDDING_MODEL_PATH = "/Users/jiang/models/embedding_models/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# 20 条金融银行相关问答数据
TEST_DATA = [
    {"text": "银行电汇需要准备哪些信息？需要收款人姓名、账户号、开户行名称及SWIFT代码。"},
    {"text": "第三方支付平台有哪些？常见的有PayPal、支付宝国际版、微信支付、Visa、Mastercard等。"},
    {"text": "什么是SWIFT代码？SWIFT代码是银行国际代码，用于国际电汇识别银行。"},
    {"text": "如何开设海外银行账户？需要准备护照、签证、居住证明等文件。"},
    {"text": "电汇和票汇有什么区别？电汇是电子转账速度快，票汇是汇票形式速度慢但费用低。"},
    {"text": "什么是结汇？结汇是将外汇兑换成本币的过程。"},
    {"text": "个人每年购汇额度是多少？境内居民每年等值5万美元的购汇额度。"},
    {"text": "西联汇款如何操作？到西联网点填写汇款单，支付汇款金额和手续费即可。"},
    {"text": "什么是备用信用证？备用信用证是银行出具的支付保证函。"},
    {"text": "如何查询汇款进度？可通过银行柜台、网上银行或电话查询。"},
    {"text": "电汇手续费如何计算？按汇款金额比例收取，一般有最低和最高限额。"},
    {"text": "什么是票汇？票汇是银行开具汇票，由汇款人自行寄给收款人。"},
    {"text": "跨境汇款需要多长时间？通常1-5个工作日，视国家和银行而定。"},
    {"text": "什么是TT汇款？TT汇款是电汇的英文缩写，通过银行电子转账。"},
    {"text": "如何选择汇款方式？大额汇款用电汇，小额或急需用票汇。"},
    {"text": "信用卡取现手续费是多少？一般取现金额的1%-3%，最低10-20元。"},
    {"text": "什么是旅行支票？旅行支票是预付式的支票，可在银行或指定地点兑换。"},
    {"text": "外币现钞和现汇有什么区别？现钞是实际纸币，现汇是账户数字后者更划算。"},
    {"text": "什么是国际收支申报？居民个人外汇收支需按规定进行申报。"},
    {"text": "如何降低汇款手续费？选择合适银行利用汇率差或使用支付平台。"},
]


def create_collection(client: MilvusClient, name: str = COLLECTION_NAME, drop_old: bool = True):
    """创建支持混合搜索的 Collection"""
    # 删除旧 Collection
    if drop_old and client.has_collection(name):
        client.drop_collection(name)
        print(f"[Collection] 已删除旧 Collection: {name}")

    # 创建 Schema
    schema = client.create_schema(auto_id=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    # text 字段需要配置 enable_analyzer=True 和 analyzer_params 用于中文分词
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,  # Milvus 2.5 推荐更大的 max_length
        enable_analyzer=True,
        analyzer_params={"type": "chinese"}  # 使用中文分词器
    )
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 添加 BM25 函数
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    # 创建索引参数
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="AUTOINDEX",
        metric_type="IP"
    )
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )

    # 创建 Collection
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    client.load_collection(collection_name=name)  # 加载 Collection 到内存
    print(f"[Collection] 已创建并加载: {name}")
    return name


def insert_data(client: MilvusClient, collection_name: str):
    """插入测试数据"""
    # 加载 embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    # 准备数据
    data = []
    for i, item in enumerate(TEST_DATA):
        # 使用 embedding 模型生成向量
        dense_vec = embeddings.embed_query(item["text"])[:EMBEDDING_DIM]
        data.append({
            "text": item["text"],
            "dense": dense_vec,
        })

    # 插入数据
    result = client.insert(collection_name=collection_name, data=data)
    client.flush(collection_name)
    print(f"[Insert] 已插入 {len(data)} 条数据")
    return result


def search_bm25(client: MilvusClient, query: str, collection_name: str, limit: int = 5):
    """仅使用 BM25 检索"""
    res = client.search(
        collection_name=collection_name,
        data=[query],
        anns_field="sparse",
        limit=limit,
        output_fields=["text"]
    )
    print(f"\n[BM25 检索] 查询: {query}")
    for hits in res:
        for i, hit in enumerate(hits, 1):
            print(f"  {i}. {hit['entity']['text'][:60]}... (score: {hit['distance']:.4f})")
    return res


def search_bm25_with_highlight(client: MilvusClient, query: str, collection_name: str, limit: int = 5):
    """BM25 检索 + 高亮显示搜索词"""

    highlighter = LexicalHighlighter(
        pre_tags=["{"],
        post_tags=["}"],
        highlight_search_text=True
    )
    res = client.search(
        collection_name=collection_name,
        data=[query],
        anns_field="sparse",
        limit=limit,
        search_params={"metric_type": "BM25"},
        output_fields=["text"],
        highlighter=highlighter
    )
    print(f"\n[BM25 高亮检索] 查询: {query}")
    for hits in res:
        for i, hit in enumerate(hits, 1):
            original_text = hit['entity']['text']
            highlighted_text = original_text  # 默认使用原文

            # highlight 结构: {'text': {'fragments': ['...'], 'scores': []}}
            if 'highlight' in hit and hit['highlight']:
                highlight_data = hit['highlight']
                text_data = highlight_data.get('text', {})
                fragments = text_data.get('fragments', [])
                if fragments:
                    fragment = fragments[0]
                    # 提取所有高亮词项
                    import re
                    highlighted_terms = re.findall(r'\{([^}]+)\}', fragment)

                    if highlighted_terms:
                        # 找到所有匹配位置
                        all_matches = []
                        for term in highlighted_terms:
                            start = 0
                            while True:
                                pos = original_text.find(term, start)
                                if pos == -1:
                                    break
                                all_matches.append((pos, pos + len(term), term))
                                start = pos + 1

                        # 按长度从长到短排序
                        all_matches.sort(key=lambda x: x[1] - x[0], reverse=True)
                        # 跳过重叠的匹配
                        skipped = set()
                        final_matches = []
                        for start, end, term in all_matches:
                            if start not in skipped:
                                final_matches.append((start, end, term))
                                for i in range(start, end):
                                    skipped.add(i)

                        # 按位置从后往前替换
                        for start, end, term in sorted(final_matches, key=lambda x: x[0], reverse=True):
                            highlighted_text = original_text[:start] + "{" + term + "}" + original_text[end:]

            print(f"  {i}. (score: {hit['distance']:.4f})")
            print(f"     原文: {original_text}")
            print(f"     高亮: {highlighted_text}")
    return res


def analyze_text(client: MilvusClient, query: str, collection_name: str):
    """查看 Milvus 分词结果"""
    print(f"\n[分词分析] 查询: '{query}'")
    try:
        # 使用 collection 中的字段配置来分析
        result = client.run_analyzer(
            texts=[query],
            collection_name=collection_name,
            field_name="text",
            with_detail=True
        )
        print(f"  分词结果: {result}")
    except Exception as e:
        print(f"  分词失败: {e}")


def search_vector(client: MilvusClient, query: str, collection_name: str, limit: int = 5):
    """仅使用向量检索"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    query_vec = embeddings.embed_query(query)[:EMBEDDING_DIM]

    res = client.search(
        collection_name=collection_name,
        data=[query_vec],
        anns_field="dense",
        limit=limit,
        search_params={"metric_type": "IP"},
        output_fields=["text"]
    )
    print(f"\n[向量检索] 查询: {query}")
    for hits in res:
        for i, hit in enumerate(hits, 1):
            print(f"  {i}. {hit['entity']['text'][:60]}... (score: {hit['distance']:.4f})")
    return res


def search_hybrid(client: MilvusClient, query: str, collection_name: str, limit: int = 5):
    """混合检索: BM25 + 向量"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    query_vec = embeddings.embed_query(query)[:EMBEDDING_DIM]

    # 创建 BM25 搜索请求 (使用文本查询)
    search_param_sparse = {
        "data": [query],
        "anns_field": "sparse",
        "limit": limit,
        "param": {}
    }
    request_sparse = AnnSearchRequest(**search_param_sparse)

    # 创建向量搜索请求
    search_param_dense = {
        "data": [query_vec],
        "anns_field": "dense",
        "limit": limit,
        "param": {"metric_type": "IP"}
    }
    request_dense = AnnSearchRequest(**search_param_dense)

    # RRF Ranker
    ranker = Function(
        name="rrf_rerank",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 100}
    )

    # 执行混合搜索
    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=[request_sparse, request_dense],
        ranker=ranker,
        limit=limit,
        output_fields=["text"]
    )

    print(f"\n[混合检索] 查询: {query}")
    for hits in res:
        for i, hit in enumerate(hits, 1):
            print(f"  {i}. {hit.get('entity', {}).get('text', 'N/A')[:60]}... (score: {hit['distance']:.4f})")
    return res


def main():
    print("Milvus 混合搜索测试")
    print("=" * 50)

    # 连接 Milvus
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    # 创建 Collection
    collection_name = create_collection(client)

    # 插入数据
    insert_data(client, collection_name)

    # 测试查询
    test_query = "电汇需要准备什么信息？"

    # BM25 检索
    search_bm25(client, test_query, collection_name)

    # BM25 高亮检索
    analyze_text(client, test_query, collection_name)
    search_bm25_with_highlight(client, test_query, collection_name)

    # 向量检索
    search_vector(client, test_query, collection_name)

    # 混合检索
    search_hybrid(client, test_query, collection_name)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()