"""
Example 4: FAISS 向量数据库最小运行示例
功能：向量相似度检索
"""

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==================== 1. 准备数据 ====================

# 模拟一些文档
documents = [
    Document(page_content="银行电汇需要准备哪些信息？答案：收款人姓名、账号、开户行名称及SWIFT代码。", metadata={"source": "doc1"}),
    Document(page_content="第三方支付平台有哪些？答案：Flywire、Western Union、支付宝国际版等。", metadata={"source": "doc2"}),
    Document(page_content="信用卡支付的手续费是多少？答案：一般为交易金额的1.5%-3%。", metadata={"source": "doc3"}),
    Document(page_content="留学缴费支持哪些支付方式？答案：国际信用卡、银行电汇、第三方支付平台等。", metadata={"source": "doc4"}),
    Document(page_content="Flywire支付的优势是什么？答案：汇率透明、到账快、手续费低。", metadata={"source": "doc5"}),
]

print(f"准备好的文档数量: {len(documents)}")

# ==================== 2. 创建 Embedding 模型 ====================

embeddings = HuggingFaceEmbeddings(
    model_name="D:/BIGMODEL",
    encode_kwargs={"normalize_embeddings": True},
)

# 获取向量维度
sample_vec = embeddings.embed_query("hello")
embedding_dim = len(sample_vec)
print(f"向量维度: {embedding_dim}")
print(f"样本向量: {sample_vec[:5]}...")

# ==================== 3. 构建 FAISS 索引 ====================

# 创建 L2 距离索引（也可以用 IP 内积，对归一化向量等价于余弦相似度）
# 创建 IP 内积索引（对归一化向量等价于余弦相似度）
index = faiss.IndexFlatIP(embedding_dim)

# 创建 FAISS 向量存储
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# 添加文档
vector_store.add_documents(documents=documents)

print(f"索引中的向量数量: {vector_store.index.ntotal}")

# ==================== 4. 相似度检索 ====================

query = "银行电汇需要准备什么？"
print(f"\n查询: {query}")

# 方法1：只返回文档
results = vector_store.similarity_search(query, k=3)
print(f"\n[similarity_search] 返回 top-3 文档:")
for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.page_content[:40]}...")

# 方法2：返回文档 + 分数（IP 内积，越大越相似）
results_with_scores = vector_store.similarity_search_with_score(query, k=3)
print(f"\n[similarity_search_with_score] 返回 top-3 文档 + 分数:")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"  {i}. [score={score:.4f}] {doc.page_content[:40]}...")

# 方法3：返回文档 + 分数 + 过滤（按 IP 内积）
results_filtered = vector_store.similarity_search_with_score(
    query, k=3, filter={"source": "doc1"}
)
print(f"\n[similarity_search_with_score] 过滤 top-3 文档 + 分数:")
for i, (doc, score) in enumerate(results_filtered, 1):
    print(f"  {i}. [score={score:.4f}] {doc.page_content[:40]}...")


# ==================== 5. 保存和加载索引 ====================

# save_path = "/Users/jiang/PycharmProjects/ragV2/example/faiss_indexIP"
#
# # 保存索引
# vector_store.save_local(save_path)
# print(f"\n索引已保存到: {save_path}")
#
# # 加载索引
# loaded_vector_store = FAISS.load_local(
#     save_path,
#     embeddings,
#     allow_dangerous_deserialization=True,
# )
# print(f"索引已加载，向量数量: {loaded_vector_store.index.ntotal}")
#
# # 验证加载后的检索
# results = loaded_vector_store.similarity_search(query, k=2)
# print(f"\n[加载后检索] top-2:")
# for i, doc in enumerate(results, 1):
#     print(f"  {i}. {doc.page_content[:40]}...")
#
# print("\n完成!")
