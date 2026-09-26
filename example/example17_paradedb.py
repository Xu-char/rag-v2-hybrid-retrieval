"""
Example 17: ParadeDB (pg_search + pgvector) 混合检索示例
功能：演示 PostgreSQL + ParadeDB 的 BM25 关键字检索 + 向量检索 + 混合检索

前置条件：
1. ParadeDB Docker 运行中 (localhost:5433)
   docker run --name paradedb -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres123 \\
     -e POSTGRES_DB=rag -p 5432:5432 -v paradedb_data:/var/lib/postgresql/data -d \\
     paradedb/paradedb:0.22.6-pg17

已安装扩展：
- pg_search (BM25 全文搜索)
- vector (向量搜索)

注意：ParadeDB 的 BM25 索引使用 Tantivy，默认 Raw 分词器对中文支持有限。
如需中文分词，需在创建索引时指定 tokenizer（如 jieba）。
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# ==================== 配置 ====================

PG_CONNECTION = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres123",
    "dbname": "rag",
}

EMBEDDING_MODEL = "D:\\BIGMODEL"
RERANK_MODEL = "D:\\huggingface\\hub\\models--BAAI--bge-reranker-v2-m3\\snapshots\\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
EMBEDDING_DIM = 1024

# ==================== 1. 连接数据库 ====================

def get_conn():
    return psycopg2.connect(**PG_CONNECTION)


def init_db():
    """初始化数据库：创建表和索引"""
    conn = get_conn()
    cur = conn.cursor()

    # 创建表（包含 text 和 vector 字段）
    # 先删表：CREATE TABLE IF NOT EXISTS 不会修改已存在表的列类型，
    # 若旧表仍是 VECTOR(512)，插入 1024 维向量会报维度不匹配
    cur.execute("DROP TABLE IF EXISTS documents;")

    cur.execute(f"""
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            vector VECTOR({EMBEDDING_DIM})
        );
    """)
    conn.commit()

    # 清空表数据（避免重复运行导致数据累积）
    cur.execute("TRUNCATE documents RESTART IDENTITY;")

    # 删除旧索引（如果存在）
    cur.execute("DROP INDEX IF EXISTS documents_bm25_idx;")
    cur.execute("DROP INDEX IF EXISTS documents_vector_idx;")

    # 创建 BM25 索引（使用 jieba 中文分词器）
    # key_field 必须是主键 id，text_fields 中的 content 是别名，column 指定实际列名
    cur.execute("""
        CREATE INDEX documents_bm25_idx ON documents
        USING bm25 (id, text) WITH (
            key_field='id',
            text_fields='{"content": {"tokenizer": {"type": "jieba"}, "column": "text"}}'
        );
    """)

    # 创建向量索引
    cur.execute("""
        CREATE INDEX documents_vector_idx ON documents
        USING ivfflat (vector) WITH (lists = 1);
    """)
    conn.commit()

    print("[DB] 表和索引创建成功")
    cur.close()
    conn.close()


def insert_documents(texts: list):
    """批量插入文档及其向量"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    conn = get_conn()
    cur = conn.cursor()

    # 生成向量
    vectors = embeddings.embed_documents(texts)

    # 批量插入
    data = [(text, vec[:EMBEDDING_DIM]) for text, vec in zip(texts, vectors)]
    execute_values(
        cur,
        "INSERT INTO documents (text, vector) VALUES %s",
        data,
        template="(%s, %s::vector)"
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] 已插入 {len(texts)} 个文档")


# ==================== 2. BM25 检索 ====================

def search_bm25(query: str, limit: int = 5):
    """BM25 关键字检索（使用 jieba 分词）

    ParadeDB 使用 @@@ 操作符进行 BM25 搜索。
    """
    conn = get_conn()
    cur = conn.cursor()

    # ParadeDB BM25 搜索：使用 @@@ 操作符
    cur.execute("""
        SELECT id, text
        FROM documents
        WHERE text @@@ %s
        LIMIT %s;
    """, (query, limit))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results


# ==================== 3. 向量检索 ====================

def search_vector(query: str, limit: int = 5):
    """向量相似度检索"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    query_vec = embeddings.embed_query(query)[:EMBEDDING_DIM]

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, text, (vector <=> %s::vector) AS distance
        FROM documents
        ORDER BY vector <=> %s::vector
        LIMIT %s;
    """, (query_vec, query_vec, limit))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results


# ==================== 4. 混合检索（RRF）====================

def search_hybrid_rrf(query: str, limit: int = 5, k: int = 60):
    """混合检索：BM25 + 向量，使用 Reciprocal Rank Fusion 合并

    RRF 公式: score = sum(1 / (k + rank))
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    query_vec = embeddings.embed_query(query)[:EMBEDDING_DIM]

    conn = get_conn()
    cur = conn.cursor()

    # 获取 BM25 排名（使用 ParadeDB @@@ 操作符，jieba 分词）
    cur.execute("""
        SELECT id, text
        FROM documents
        WHERE text @@@ %s
        LIMIT %s;
    """, (query, k))
    bm25_results = [(row[0], row[1]) for row in cur.fetchall()]

    # 获取向量相似度排名
    cur.execute("""
        SELECT id, text, (vector <=> %s::vector) AS vec_distance
        FROM documents
        ORDER BY vector <=> %s::vector
        LIMIT %s;
    """, (query_vec, query_vec, k))
    vector_results = [(row[0], row[1], row[2]) for row in cur.fetchall()]

    # RRF 合并
    rrf_scores = {}

    # BM25 排名（按出现顺序，BM25 不返回分数）
    for rank, (doc_id, text) in enumerate(bm25_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # 向量排名（按距离升序，距离越小排名越前）
    for rank, (doc_id, text, dist) in enumerate(vector_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # 排序取 top
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:limit]

    results = []
    for doc_id in sorted_ids:
        # 查找对应的文本
        text = next((t for (d, t) in bm25_results if d == doc_id), None)
        if text is None:
            text = next((t for (d, t, _) in vector_results if d == doc_id), "N/A")
        results.append((doc_id, text, rrf_scores[doc_id]))

    cur.close()
    conn.close()
    return results


# ==================== 5. 带 rerank 的混合检索 ====================

def search_hybrid_with_rerank(query: str, limit: int = 5, retrieve_k: int = 10):
    """混合检索 + BGE-reranker 重排

    流程：
    1. BM25 召回 retrieve_k 条
    2. 向量召回 retrieve_k 条
    3. 去重合并候选
    4. BGE-reranker 重排
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    query_vec = embeddings.embed_query(query)[:EMBEDDING_DIM]

    conn = get_conn()
    cur = conn.cursor()

    # BM25 召回（使用 ParadeDB @@@ 操作符，jieba 分词）
    cur.execute("""
        SELECT id, text
        FROM documents
        WHERE text @@@ %s
        LIMIT %s;
    """, (query, retrieve_k))
    bm25_candidates = [(row[0], row[1], 0) for row in cur.fetchall()]

    # 向量召回
    cur.execute("""
        SELECT id, text, (vector <=> %s::vector) AS vec_distance
        FROM documents
        ORDER BY vector <=> %s::vector
        LIMIT %s;
    """, (query_vec, query_vec, retrieve_k))
    vector_candidates = [(row[0], row[1], row[2]) for row in cur.fetchall()]

    # 去重合并
    seen = {}
    for doc_id, text, score in bm25_candidates:
        if doc_id not in seen:
            seen[doc_id] = ("bm25", text, score)
    for doc_id, text, score in vector_candidates:
        if doc_id not in seen:
            seen[doc_id] = ("vector", text, score)

    candidate_texts = [v[1] for v in seen.values()]
    doc_ids = list(seen.keys())

    # Rerank
    pairs = [(query, text) for text in candidate_texts]
    rerank_scores = reranker.predict(pairs)

    # 按 rerank 分数排序
    ranked = sorted(zip(doc_ids, candidate_texts, rerank_scores), key=lambda x: x[2], reverse=True)

    results = [(doc_id, text, score) for doc_id, text, score in ranked[:limit]]

    cur.close()
    conn.close()
    return results


# ==================== 主函数 ====================

def main():
    # 测试数据
    documents = [
        "银行电汇需要准备哪些信息？答案：收款人姓名、账户号、开户行名称及SWIFT代码。",
        "第三方支付平台有哪些？常见的有PayPal、支付宝国际版、微信支付、Visa、Mastercard等。",
        "信用卡支付的手续费是多少？一般为交易金额的1.5%-3%。",
        "留学缴费支持哪些支付方式？答案：国际信用卡、银行电汇、第三方支付平台等。",
        "Flywire支付的优势是什么？答案：汇率透明、到账快、手续费低。",
        "个人每年购汇额度是多少？境内居民每年等值5万美元的购汇额度。",
        "什么是结汇？结汇是将外汇兑换成本币的过程。",
        "西联汇款如何操作？到西联网点填写汇款单，支付汇款金额和手续费即可。",
        "电汇和票汇有什么区别？电汇是电子转账速度快，票汇是汇票形式速度慢但费用低。",
        "如何查询汇款进度？可通过银行柜台、网上银行或电话查询。",
    ]

    print("=" * 60)
    print("ParadeDB 混合检索演示")
    print("=" * 60)

    # 初始化
    print("\n[1] 初始化数据库...")
    init_db()

    # 插入数据
    print("\n[2] 插入文档...")
    insert_documents(documents)

    # BM25 检索
    print("\n" + "=" * 60)
    print("[3] BM25 关键字检索")
    print("=" * 60)
    query = "电汇需要"  # 使用简洁的查询词，避免"什么"等停用词问题
    results = search_bm25(query, limit=3)
    print(f"查询: {query}")
    for i, (doc_id, text) in enumerate(results, 1):
        print(f"  {i}. {text[:50]}...")

    # 向量检索
    print("\n" + "=" * 60)
    print("[4] 向量检索 (cosine similarity)")
    print("=" * 60)
    query = "第三方支付平台有哪些"
    results = search_vector(query, limit=3)
    print(f"查询: {query}")
    for i, (doc_id, text, distance) in enumerate(results, 1):
        print(f"  {i}. [dist={distance:.4f}] {text[:50]}...")

    # 混合检索 (RRF)
    print("\n" + "=" * 60)
    print("[5] 混合检索 (BM25 + 向量 + RRF)")
    print("=" * 60)
    query = "银行汇款手续费"
    results = search_hybrid_rrf(query, limit=3)
    print(f"查询: {query}")
    for i, (doc_id, text, rrf_score) in enumerate(results, 1):
        print(f"  {i}. [rrf={rrf_score:.4f}] {text[:50]}...")

    # 混合检索 + Rerank
    print("\n" + "=" * 60)
    print("[6] 混合检索 + Rerank (BGE-reranker-v2-m3)")
    print("=" * 60)
    query = "银行汇款手续费"
    results = search_hybrid_with_rerank(query, limit=3)
    print(f"查询: {query}")
    for i, (doc_id, text, rerank_score) in enumerate(results, 1):
        print(f"  {i}. [rerank={rerank_score:.4f}] {text[:50]}...")

    print("\n" + "=" * 60)
    print("配置说明")
    print("=" * 60)
    note = """
    ParadeDB BM25 使用 Tantivy + jieba 中文分词器。

    关键配置语法：
    - key_field 必须是主键（如 'id'）
    - text_fields 中定义要索引的字段及其分词器

    CREATE INDEX documents_bm25_idx ON documents
    USING bm25 (id, text) WITH (
        key_field='id',
        text_fields='{"content": {"tokenizer": {"type": "jieba"}, "column": "text"}}'
    );

    注意：
    - key_field 不能在 text_fields 中被配置
    - column 指定实际列名，content 是别名
    - 可用分词器：default, keyword, raw, jieba, chinese_compatible,
      chinese_lindera, japanese_lindera, korean_lindera, ngram 等
    """
    print(note)

    print("完成!")


if __name__ == "__main__":
    main()