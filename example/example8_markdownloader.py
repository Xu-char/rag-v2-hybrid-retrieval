"""
Example 8: UnstructuredMarkdownLoader 最小运行示例
功能：加载 Markdown 文档
"""

from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 1. 创建加载器
loader = UnstructuredMarkdownLoader(
    file_path="C:\\Users\\27889\\Desktop\\hhhh.md",
)

# 2. 加载文档
documents = loader.load()

# 3. 查看结果
print(f"加载了 {len(documents)} 个文档\n")
for i, doc in enumerate(documents):
    print(f"--- Document {i + 1} ---")
    print(f"内容（前200字符）: {doc.page_content[:200]}...")
    print(f"元数据: {doc.metadata}")
    print()
