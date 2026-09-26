"""
Example 1: TextLoader 最小运行示例
功能：加载 txt 文本文件
"""

from langchain_community.document_loaders import TextLoader

# 1. 创建加载器
loader = TextLoader(
    file_path="data/example.txt",
    encoding="utf-8"
)

# 2. 加载文档
documents = loader.load()

# 3. 查看结果
print(f"加载了 {len(documents)} 个文档")
for i, doc in enumerate(documents):
    print(f"\n--- Document {i + 1} ---")
    print(f"内容（前100字符）: {doc.page_content[:100]}...")
    print(f"元数据: {doc.metadata}")
