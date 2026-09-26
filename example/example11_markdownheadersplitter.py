"""
Example 11: MarkdownHeaderTextSplitter 最小运行示例
功能：按 Markdown 标题分割 - 按标题层级切分文档
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# 1. 准备 Markdown 文档
markdown_text = """# 留学缴费指南

## 银行电汇

银行电汇需要准备收款人姓名、账户号、开户行名称及SWIFT代码。手续费有两种方式：SHA和OUR。

## 第三方支付

### Flywire

Flywire支付优势：汇率透明、到账快、手续费低。

### Western Union

Western Union支持现金提取和账户转账两种方式。

## 信用卡支付

国际信用卡支付会产生1.5%-3%的手续费，以及货币转换费。

## 注意事项

留学缴费需要注意汇率波动风险，建议提前购汇或选择锁定汇率的平台。
"""

documents = [Document(page_content=markdown_text, metadata={"source": "guide.md"})]

# 2. 定义要分割的标题层级
headers_to_split_on = [
    ("#", "Header 1"),      # 一级标题
    ("##", "Header 2"),    # 二级标题
    ("###", "Header 3"),    # 三级标题
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 3. 分割文档（输入是字符串，不是 Document）
chunks = splitter.split_text(markdown_text)

# 4. 查看结果
print(f"分割成了 {len(chunks)} 个 chunks\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(f"内容:\n{chunk.page_content}")
    print(f"元数据: {chunk.metadata}")
    print()