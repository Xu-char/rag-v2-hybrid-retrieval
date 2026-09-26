"""
Example 10: CharacterTextSplitter 最小运行示例
功能：按段落分割 - 使用指定分隔符分割文本
"""

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# 1. 准备文档
text = """
这是第一段内容。我们讨论了银行电汇的基本信息。

这是第二段内容。我们讨论了手续费的问题。

这是第三段内容。我们讨论了支付方式的选择。

这是第四段内容。我们讨论了汇率风险规避。
"""

documents = [Document(page_content=text, metadata={"source": "example"})]

# 2. 创建分割器
splitter = CharacterTextSplitter(
    separator="\n\n",        # 分隔符：两个换行（段落）
    chunk_size=50,           # 每个 chunk 的最大字符数（设为较小值以便看到分块效果）
    chunk_overlap=10,        # 相邻 chunk 之间的重叠字符数
    length_function=len,     # 计算长度的方式
)

# 3. 分割文档
chunks = splitter.split_documents(documents)

# 4. 查看结果
print(f"分割成了 {len(chunks)} 个 chunks\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} (len={len(chunk.page_content)}) ---")
    print(f"{chunk.page_content}")
    print()