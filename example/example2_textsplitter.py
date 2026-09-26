"""
Example 2: CharacterTextSplitter 最小运行示例
功能：将长文本分割成小的 chunks
"""

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# 1. 准备文档
text = """
这是一个很长的文档。我们需要将它分割成多个小的chunks。
CharacterTextSplitter 可以帮助我们完成这个任务。
它按照指定的分隔符和chunk大小来分割文本。


这里是另一段内容。分割会在这里发生。
每一段都会成为一个独立的chunk。

最后一段内容。
"""

documents = [Document(page_content=text, metadata={"source": "example"})]

# 2. 创建分割器
splitter = CharacterTextSplitter(
    separator="\n\n",        # 分隔符：两个换行（段落）
    chunk_size=50,           # 每个 chunk 的最大字符数
    chunk_overlap=10,        # 相邻 chunk 之间的重叠字符数
    length_function=len,     # 计算长度的方式
)

# 3. 分割文档
chunks = splitter.split_documents(documents)

# 4. 查看结果
print(f"分割成了 {len(chunks)} 个 chunks\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(f"内容: {chunk.page_content}")
    print(f"元数据: {chunk.metadata}")
    print()
