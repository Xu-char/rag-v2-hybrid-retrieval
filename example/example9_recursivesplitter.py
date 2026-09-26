"""
Example 9: RecursiveCharacterTextSplitter 最小运行示例
功能：递归字符切分 - 按字符递归分割，保留语义完整性
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. 准备文档
text = """
银行电汇需要准备哪些信息？
答案：收款人姓名、账户号、开户行名称及SWIFT代码。

银行电汇的手续费由谁承担？
答案：手续费有两种承担方式：一是SHA（共同承担），汇款人承担国内银行手续费，收款人承担境外手续费；二是OUR（汇款人承担），所有手续费均由汇款人支付。

Flywire支付的优势是什么？
答案：Flywire支付的优势主要有：1. 汇率透明，提前锁定汇率；2. 支持多种支付渠道；3. 到账速度快，1-3个工作日；4. 全程可追踪；5. 手续费低于传统银行电汇。

留学缴费支持哪些支付方式？
答案：常见方式包括国际信用卡支付、银行电汇、第三方支付平台（如Flywire、Western Union）、部分院校支持支付宝或微信支付国际版。
"""

documents = [Document(page_content=text, metadata={"source": "example"})]

# 2. 创建分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,           # 每个 chunk 的最大字符数
    chunk_overlap=5,         # 相邻 chunk 之间的重叠字符数
    length_function=len,      # 计算长度的方式
    is_separator_regex=False, # 是否将分隔符视为正则表达式
)

# 3. 分割文档
chunks = splitter.split_documents(documents)

# 4. 查看结果
print(f"原始文本长度: {len(text)} 字符")
print(f"分割成了 {len(chunks)} 个 chunks\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} (len={len(chunk.page_content)}) ---")
    print(f"{chunk.page_content}")
    print()