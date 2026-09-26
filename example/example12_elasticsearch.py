"""
Example 12: Elasticsearch 混合检索示例
功能：演示 Elasticsearch 的 BM25 关键字检索 + 向量检索 + 混合检索

前置条件：
1. Elasticsearch 8.x 运行中 (localhost:9200)
2. 安装 IK Analyzer（中文分词）：./elasticsearch-plugin install https://github.com/infinilabs/analysis-ik/releases/download/v8.15.0/elasticsearch-analysis-ik-8.15.0.zip
3. 重启 ES 后使用

三种检索方式：
1. BM25 关键字检索 - 基于文本分词的相关性评分
2. 向量检索 - 基于语义 embedding 的余弦相似度
3. 混合检索 - BM25 + 向量结合 bool should 合并
"""

import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from elasticsearch import Elasticsearch

# ==================== 配置 ====================

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = "rag_es_hybrid"
EMBEDDING_MODEL = "D:\\BIGMODEL"
EMBEDDING_DIM = 1024

# ==================== 1. 准备数据 ====================

documents = [
    Document(page_content="银行电汇需要准备哪些信息？答案：收款人姓名、账户号、开户行名称及SWIFT代码。", metadata={"category": "banking"}),
    Document(page_content="第三方支付平台有哪些？常见的有PayPal、支付宝国际版、微信支付、Visa、Mastercard等。", metadata={"category": "payment"}),
    Document(page_content="信用卡支付的手续费是多少？一般为交易金额的1.5%-3%。", metadata={"category": "payment"}),
    Document(page_content="留学缴费支持哪些支付方式？答案：国际信用卡、银行电汇、第三方支付平台等。", metadata={"category": "payment"}),
    Document(page_content="Flywire支付的优势是什么？答案：汇率透明、到账快、手续费低。", metadata={"category": "payment"}),
    Document(page_content="个人每年购汇额度是多少？境内居民每年等值5万美元的购汇额度。", metadata={"category": "forex"}),
    Document(page_content="什么是结汇？结汇是将外汇兑换成本币的过程。", metadata={"category": "forex"}),
    Document(page_content="西联汇款如何操作？到西联网点填写汇款单，支付汇款金额和手续费即可。", metadata={"category": "payment"}),
    Document(page_content="电汇和票汇有什么区别？电汇是电子转账速度快，票汇是汇票形式速度慢但费用低。", metadata={"category": "banking"}),
    Document(page_content="如何查询汇款进度？可通过银行柜台、网上银行或电话查询。", metadata={"category": "banking"}),
]

print(f"准备好的文档数量: {len(documents)}")

# ==================== 2. 创建 Embedding 模型 ====================

print(f"\n加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("Embedding 模型加载完成")

# ==================== 3. 创建 Elasticsearch 混合索引 ====================

print(f"\n连接 Elasticsearch: {ES_URL}")
es = Elasticsearch(hosts=[ES_URL])

# 检查并删除旧索引
if es.indices.exists(index=ES_INDEX):
    es.indices.delete(index=ES_INDEX)
    print(f"[ES] 已删除旧索引: {ES_INDEX}")

# 创建支持混合检索的索引映射
# - text 字段：用于 BM25 关键字检索（需 IK analyzer 中文分词）
# - dense_vector 字段：用于向量检索
mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "ik_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_max_word",
                    "filter": ["lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "ik_analyzer",
                "search_analyzer": "ik_smart"
            },
            "vector": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine"
            },
            "category": {"type": "keyword"},
        }
    }
}

es.indices.create(index=ES_INDEX, body=mapping)
print(f"[ES] 索引创建成功: {ES_INDEX}")

# ==================== 4. 插入文档（text + vector）====================

print(f"\n生成向量并插入文档...")
texts = [doc.page_content for doc in documents]
vecs = embeddings.embed_documents(texts)

operations = []
for i, (text, vec) in enumerate(zip(texts, vecs)):
    operations.append({"index": {"_index": ES_INDEX}})
    operations.append({
        "text": text,
        "vector": vec[:EMBEDDING_DIM],
        "category": documents[i].metadata.get("category", ""),
    })

es.bulk(operations=operations)
es.indices.refresh(index=ES_INDEX)
print(f"[ES] 已插入 {len(texts)} 个文档")

# ==================== 5. BM25 关键字检索 ====================

print("\n" + "=" * 60)
print("【1】BM25 关键字检索")
print("=" * 60)

query_bm25 = "电汇需要准备什么"

# 查看分词结果
response = es.indices.analyze(
    index=ES_INDEX,
    body={
        "analyzer": "ik_analyzer",
        "text": "电汇需要准备什么"
    }
)
for token in response["tokens"]:
    print(token["token"])

# 检索
response = es.search(
    index=ES_INDEX,
    body={
        "query": {
            "match": {
                "text": query_bm25
            }
        },
        "size": 3
    }
)

print(f"查询: {query_bm25}")
print(f"结果数量: {response['hits']['total']['value']}")
for i, hit in enumerate(response["hits"]["hits"], 1):
    score = hit["_score"]
    text = hit["_source"]["text"][:50]
    print(f"  {i}. [score={score:.4f}] {text}...")

# ==================== 6. 向量检索 ====================

print("\n" + "=" * 60)
print("【2】向量检索 (cosine similarity)")
print("=" * 60)

query_vec = embeddings.embed_query("第三方支付平台有哪些")
query_text = "第三方支付平台有哪些"

response = es.search(
    index=ES_INDEX,
    body={
        "knn": {
            "field": "vector",
            "query_vector": query_vec[:EMBEDDING_DIM],
            "k": 3,
            "num_candidates": 10
        }
    }
)

print(f"查询: {query_text}")
for i, hit in enumerate(response["hits"]["hits"], 1):
    score = hit["_score"]
    text = hit["_source"]["text"][:50]
    print(f"  {i}. [score={score:.4f}] {text}...")

# ==================== 7. 混合检索 (BM25 + 向量 + rerank model) ====================

print("\n" + "=" * 60)
print("【3】混合检索 (BM25 + 向量 + rerank model)")
print("=" * 60)

from sentence_transformers import CrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 3
RETRIEVE_TOP_K = 10  # 先多召回一些，再重排

query_hybrid = "银行汇款手续费"
query_vector = embeddings.embed_query(query_hybrid)

# Step 1: 混合召回（BM25 + 向量，bool should 合并评分）
response = es.search(
    index=ES_INDEX,
    body={
        "query": {
            "bool": {
                "should": [
                    {"match": {"text": query_hybrid}},
                    {"knn": {"field": "vector", "query_vector": query_vector[:EMBEDDING_DIM], "k": RETRIEVE_TOP_K, "num_candidates": 20}}
                ]
            }
        },
        "size": RETRIEVE_TOP_K
    }
)

# Step 2: 提取召回文档，构建 query-doc pairs
candidate_texts = [hit["_source"]["text"] for hit in response["hits"]["hits"]]
candidate_scores = [hit["_score"] for hit in response["hits"]["hits"]]
pairs = [(query_hybrid, text) for text in candidate_texts]

print(f"查询: {query_hybrid}")
print(f"召回: {len(candidate_texts)} 条候选文档")

# Step 3: 使用 rerank 模型重排
print(f"Rerank 模型: {RERANK_MODEL}")
reranker = CrossEncoder(RERANK_MODEL)
rerank_scores = reranker.predict(pairs)

# Step 4: 按 rerank 分数排序，取 top RERANK_TOP_K
ranked = sorted(zip(candidate_texts, candidate_scores, rerank_scores), key=lambda x: x[2], reverse=True)

print(f"\nrerank 后的结果 (top {RERANK_TOP_K}):")
for i, (text, orig_score, rerank_score) in enumerate(ranked[:RERANK_TOP_K], 1):
    print(f"  {i}. [rerank={rerank_score:.4f}, orig={orig_score:.4f}] {text[:50]}...")

# ==================== 8. 带过滤的检索 ====================

print("\n" + "=" * 60)
print("【4】带元数据过滤的检索")
print("=" * 60)

query_filter = "支付"
filter_category = "payment"

response = es.search(
    index=ES_INDEX,
    body={
        "query": {
            "bool": {
                "must": [
                    {"match": {"text": query_filter}}
                ],
                "filter": [
                    {"term": {"category": filter_category}}
                ]
            }
        },
        "size": 3
    }
)

print(f"查询: {query_filter}")
print(f"过滤条件: category = {filter_category}")
for i, hit in enumerate(response["hits"]["hits"], 1):
    text = hit["_source"]["text"][:50]
    cat = hit["_source"]["category"]
    print(f"  {i}. [{cat}] {text}...")

# ==================== 9. 总结 ====================

print("\n" + "=" * 60)
print("Elasticsearch 检索方法总结")
print("=" * 60)

summary = """
┌──────────────────────┬─────────────────────────────────────────────────────────┐
│ 检索方式             │ 说明                                                      │
├──────────────────────┼─────────────────────────────────────────────────────────┤
│ match (BM25)         │ 关键字检索，使用 IK 中文分词，基于 TF-IDF 评分           │
│ knn (向量检索)        │ 语义检索，基于 dense_vector 余弦相似度                   │
│ bool should + knn   │ 混合检索，BM25 和向量结果用 should 合并（各自评分相加）   │
│ filter (term)        │ 元数据精确过滤，不参与评分                               │
│ match + filter      │ 关键字检索 + 元数据过滤组合                             │
└──────────────────────┴─────────────────────────────────────────────────────────┘

关键配置：

- text 字段 analyzer=ik_analyzer：中文分词
- vector 字段 similarity=cosine：余弦相似度

注意：
- 如果 ES 没有安装 IK Analyzer，text 字段会使用默认分词器，中文检索效果会很差
- RRF (Reciprocal Rank Fusion) 需要付费 license，basic/license-free 版本不可用
"""
print(summary)

print("完成!")