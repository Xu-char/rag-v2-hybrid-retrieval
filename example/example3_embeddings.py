"""
Example 3: HuggingFaceEmbeddings 最小运行示例
功能：将文本转换为向量（embedding）
"""

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np

# 1. 创建 embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\BIGMODEL",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # 归一化向量
)

# 2. 单个文本的 embedding
query = "银行电汇的流程是什么？"
query_vector = embeddings.embed_query(query)

print(f"查询文本: {query}")
print(f"向量维度: {len(query_vector)}")
print(f"向量（前10维）: {query_vector[:10]}")
print(f"向量范数: {np.linalg.norm(query_vector):.4f}")

# 3. 多个文档的 embedding
docs = [
    "银行电汇需要准备哪些信息？",
    "第三方支付平台有哪些？",
    "信用卡支付的手续费是多少？",
]
doc_vectors = embeddings.embed_documents(docs)

print(f"\n文档数量: {len(docs)}")
print(f"每个向量维度: {len(doc_vectors[0])}")

# 4. 计算余弦相似度
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\n查询与各文档的余弦相似度:")
for i, doc in enumerate(docs):
    sim = cosine_sim(query_vector, doc_vectors[i])
    print(f"  Doc{i + 1}: {sim:.4f} - {doc}")
