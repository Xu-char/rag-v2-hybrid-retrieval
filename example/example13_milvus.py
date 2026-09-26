"""
Example 13: Milvus 最小运行示例
功能：将文档存储到 Milvus 向量数据库
注意：连接远程 Milvus 服务器，使用 langchain-milvus
"""

from langchain_milvus import Milvus
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pymilvus import connections, MilvusClient

#  配置 

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "rag_demo222"
EMBEDDING_MODEL = "/Users/jiang/models/embedding_models/bge-small-zh-v1.5"

# 预连接 MilvusClient 并注册到全局 connections
# langchain_milvus 内部使用 ORM Collection 类，该类依赖 pymilvus.connections 全局注册表，
# 但 MilvusClient 不会自动注册自己到全局注册表
client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
connections._alias_handlers[client._using] = client._handler

documents = [
    Document(page_content="银行电汇需要准备哪些信息？答案：收款人姓名、账户号、开户行名称及SWIFT代码。", metadata={"source": "doc1"}),
    Document(page_content="第三方支付平台有哪些？答案：Flywire、Western Union、支付宝国际版等。", metadata={"source": "doc2"}),
    Document(page_content="信用卡支付的手续费是多少？答案：一般为交易金额的1.5%-3%。", metadata={"source": "doc3"}),
    Document(page_content="留学缴费支持哪些支付方式？答案：国际信用卡、银行电汇、第三方支付平台等。", metadata={"source": "doc4"}),
    Document(page_content="Flywire支付的优势是什么？答案：汇率透明、到账快、手续费低。", metadata={"source": "doc5"}),
]

print(f"准备好的文档数量: {len(documents)}")

#  2. 创建 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
)

#  3. 创建 Milvus 向量存储
print(f"\n创建 Milvus 向量存储: {MILVUS_URI}")


vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=embeddings,
    vector_field=["dense"],
    connection_args={
        "host": "localhost",
        "port": 19530,
        "token": MILVUS_TOKEN,
    },
    collection_name=COLLECTION_NAME,
    drop_old=True,
)
#  4. 添加文档
# vectorstore.add_documents(documents)
print(f"已添加 {len(documents)} 个文档到 Milvus")

#  5. 相似度检索
query = "银行电汇需要准备什么？"
print(f"\n查询: {query}")

results = vectorstore.similarity_search(query, k=3)
print(f"\n检索结果 (top-3):")
for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.page_content[:50]}...")

print("\n完成!")