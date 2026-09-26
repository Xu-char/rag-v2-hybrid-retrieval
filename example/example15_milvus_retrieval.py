"""
Example 15: Milvus 检索方法
1. 基本向量检索 (similarity_search)
2. 过滤检索 (filtered search)
3. 分页检索 (pagination with offset)
4. 范围检索 (range search by distance)
5. 批量检索 (batch search multiple queries)
6. 最大边际 Relevance (MMR) 多样性检索

"""

from langchain_milvus import Milvus
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pymilvus import connections, MilvusClient

# ==================== 配置 ====================

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "retrieval_demo"
EMBEDDING_MODEL = "/Users/jiang/models/embedding_models/bge-small-zh-v1.5"

# 预连接 MilvusClient 并注册到全局 connections
client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
connections._alias_handlers[client._using] = client._handler


# ==================== 1. 准备测试数据 ====================
documents = [
    # 银行类
    Document(page_content="银行电汇需要准备哪些信息？答案：收款人姓名、账户号、开户行名称及SWIFT代码。", metadata={"category": "banking", "difficulty": "basic"}),
    Document(page_content="如何开设海外银行账户？需要准备护照、签证、居住证明等文件。", metadata={"category": "banking", "difficulty": "intermediate"}),
    Document(page_content="电汇手续费如何计算？按汇款金额比例收取，一般有最低和最高限额。", metadata={"category": "banking", "difficulty": "intermediate"}),
    Document(page_content="什么是SWIFT代码？SWIFT代码是银行国际代码，用于国际电汇识别银行。", metadata={"category": "banking", "difficulty": "basic"}),

    # 支付平台类
    Document(page_content="第三方支付平台有哪些？常见的有PayPal、支付宝国际版、微信支付、Visa、Mastercard等。", metadata={"category": "payment", "difficulty": "basic"}),
    Document(page_content="Flywire支付的优势是什么？答案：汇率透明、到账快、手续费低。", metadata={"category": "payment", "difficulty": "intermediate"}),
    Document(page_content="西联汇款如何操作？到西联网点填写汇款单，支付汇款金额和手续费即可。", metadata={"category": "payment", "difficulty": "basic"}),
    Document(page_content="信用卡支付的手续费是多少？一般为交易金额的1.5%-3%。", metadata={"category": "payment", "difficulty": "intermediate"}),

    # 外汇类
    Document(page_content="个人每年购汇额度是多少？境内居民每年等值5万美元的购汇额度。", metadata={"category": "forex", "difficulty": "basic"}),
    Document(page_content="什么是结汇？结汇是将外汇兑换成本币的过程。", metadata={"category": "forex", "difficulty": "intermediate"}),
    Document(page_content="外币现钞和现汇有什么区别？现钞是实际纸币，现汇是账户数字后者更划算。", metadata={"category": "forex", "difficulty": "intermediate"}),
    Document(page_content="什么是国际收支申报？居民个人外汇收支需按规定进行申报。", metadata={"category": "forex", "difficulty": "advanced"}),

    # 汇款类
    Document(page_content="电汇和票汇有什么区别？电汇是电子转账速度快，票汇是汇票形式速度慢但费用低。", metadata={"category": "remittance", "difficulty": "basic"}),
    Document(page_content="什么是TT汇款？TT汇款是电汇的英文缩写，通过银行电子转账。", metadata={"category": "remittance", "difficulty": "basic"}),
    Document(page_content="跨境汇款需要多长时间？通常1-5个工作日，视国家和银行而定。", metadata={"category": "remittance", "difficulty": "intermediate"}),
    Document(page_content="如何查询汇款进度？可通过银行柜台、网上银行或电话查询。", metadata={"category": "remittance", "difficulty": "basic"}),
]

print(f"准备好的文档数量: {len(documents)}")

# ==================== 2. 创建 Embedding 模型和 VectorStore ====================

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print(f"\n创建 Milvus 向量存储: {MILVUS_URI}")

vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=embeddings,
    vector_field=["dense"],
    connection_args={
        "uri": MILVUS_URI,
        "token": MILVUS_TOKEN,
    },
    collection_name=COLLECTION_NAME,
    drop_old=True,
    enable_dynamic_field=True,  # 允许动态字段存储 metadata
)

print(f"向量存储创建成功！")

# ==================== 3. 检索方法演示 ====================

print("\n" + "=" * 70)
print("Milvus 检索方法演示")
print("=" * 70)

# -------------------- 3.1 基本向量检索 --------------------
print("\n【1】基本向量检索 (similarity_search)")
print("-" * 50)

query = "电汇需要准备什么信息？"
results = vectorstore.similarity_search(query, k=3)

print(f"查询: {query}")
print(f"结果数量: {len(results)}")
for i, doc in enumerate(results, 1):
    print(f"  {i}. [{doc.metadata.get('category', 'N/A')}] {doc.page_content[:40]}...")

# -------------------- 3.2 带过滤条件的检索 --------------------
print("\n【2】过滤检索 (similarity_search with filter)")
print("-" * 50)

query = "手续费怎么计算？"
filter_expr = 'category == "payment"'

results = vectorstore.similarity_search(
    query,
    k=5,
    expr=filter_expr,
)

print(f"查询: {query}")
print(f"过滤条件: {filter_expr}")
print(f"结果数量: {len(results)}")
for i, doc in enumerate(results, 1):
    print(f"  {i}. [{doc.metadata.get('category', 'N/A')}] {doc.page_content[:40]}...")

# -------------------- 3.3 分页检索 (offset + limit) --------------------
print("\n【3】分页检索 (pagination with offset)")
print("-" * 50)

query = "汇款相关问题"
all_results = vectorstore.similarity_search(query, k=10)

print(f"查询: {query}")
print(f"总结果数: {len(all_results)}")

print("\n第 1 页 (offset=0, k=5):")
for i, doc in enumerate(all_results[:5], 1):
    print(f"  {i}. {doc.page_content[:40]}...")

print("\n第 2 页 (offset=5, k=5):")
for i, doc in enumerate(all_results[5:10], 6):
    print(f"  {i}. {doc.page_content[:40]}...")

# -------------------- 3.4 范围检索 (distance filter) --------------------
print("\n【4】范围检索 (similarity_search_with_score + 距离过滤)")
print("-" * 50)

query = "银行汇款"
results_with_score = vectorstore.similarity_search_with_score(query, k=10)

print(f"查询: {query}")
print(f"所有结果 (带相似度分数):")

for i, (doc, score) in enumerate(results_with_score, 1):
    print(f"  {i}. score={score:.4f} | {doc.page_content[:40]}...")

# -------------------- 3.5 向量检索 (similarity_search_by_vector) --------------------
print("\n【5】向量检索 (similarity_search_by_vector)")
print("-" * 50)

# 先计算查询向量，然后复用该向量进行搜索
# 适用于：已经知道查询向量（如从缓存获取）的情况
query_text = "电汇手续费"
query_vector = embeddings.embed_query(query_text)

# 使用预计算的向量进行检索（避免重复计算 embedding）
results_by_vec = vectorstore.similarity_search_by_vector(
    embedding=query_vector,
    k=3,
)

print(f"查询: {query_text}")
print(f"结果:")
for i, doc in enumerate(results_by_vec, 1):
    print(f"  {i}. {doc.page_content[:40]}...")

# -------------------- 3.6 MMR 多样性检索 --------------------
print("\n【6】MMR 多样性检索 (maximum marginal relevance)")
print("-" * 50)

query = "支付和汇款"

# MMR 通过 fetch_k 和 lambda_mult 参数控制多样性
# lambda_mult 越接近 1，越注重相关性；越接近 0，越注重多样性
results_mmr = vectorstore.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=10,       # 从 10 个候选中选取
    lambda_mult=0.5,  # 0.5 表示在相关性和多样性之间平衡
)

print(f"查询: {query}")
print(f"MMR 结果 (k=5, fetch_k=10, lambda_mult=0.5):")
for i, doc in enumerate(results_mmr, 1):
    print(f"  {i}. [{doc.metadata.get('category', 'N/A')}] {doc.page_content[:40]}...")

# 对比：纯相似度检索
results_simple = vectorstore.similarity_search(query, k=5)
print(f"\n普通检索结果 (k=5):")
for i, doc in enumerate(results_simple, 1):
    print(f"  {i}. [{doc.metadata.get('category', 'N/A')}] {doc.page_content[:40]}...")

# ==================== 4. 检索方法对比总结 ====================

print("\n" + "=" * 70)
print("检索方法总结")
print("=" * 70)

summary = """
┌─────────────────────────┬─────────────────────────────────────────────────────┐
│ 检索方法                  │ 说明                                                  │
├─────────────────────────┼─────────────────────────────────────────────────────┤
│ similarity_search        │ 基本向量检索，返回最相似的文档列表                       │
│ similarity_search_by_vec │ 通过已有向量检索，不需要再编码查询文本                    │
│ similarity_search_with_  │ 返回文档+相似度分数，可用于自定义过滤或排序              │
│   score                  │                                                      │
│ max_marginal_relevance   │ MMR 多样性检索，避免返回内容高度相似的结果               │
│                          │ 适用于需要结果多样性的场景                             │
│ filter (参数)            │ 元数据过滤，使用 Milvus 标量字段过滤条件                 │
│ offset/limit             │ 分页控制，结合 k 参数实现翻页                           │
└─────────────────────────┴─────────────────────────────────────────────────────┘

MMR (Maximum Marginal Relevance) 参数说明：
- k: 最终返回的结果数量
- fetch_k: 从向量数据库中获取的候选文档数量（应 >= k）
- lambda_mult: 多样性系数
  - 1.0: 完全注重相关性（等同于普通相似度检索）
  - 0.0: 完全注重多样性
  - 0.5: 平衡模式（推荐初始值）

常见场景选择：
- 简单问答：similarity_search(k=3)
- 结果需排序/过滤：similarity_search_with_score + 自定义处理
- 推荐系统/多样性展示：max_marginal_relevance
- 分类浏览：similarity_search + filter 按类别筛选
- 分页加载：similarity_search + Python 列表切片
"""
print(summary)

print("\n完成!")