"""
Example 7: PyMuPDFLoader 最小运行示例
功能：加载 PDF 文档
"""

from langchain_community.document_loaders import PyMuPDFLoader

# 1. 创建加载器
loader = PyMuPDFLoader(
    file_path="C:\\Users\\27889\\Desktop\\作业存放文件夹\\徐超0825.pdf",
)

# 2. 加载文档（按页加载）
documents = loader.load()

# 3. 查看结果
print(f"加载了 {len(documents)} 页\n")
for i, doc in enumerate(documents):
    print(f"--- Page {i + 1} ---")
    print(f"页码: {doc.metadata.get('page', i)}")
    print(f"内容（前150字符）: {doc.page_content}")
    print()
