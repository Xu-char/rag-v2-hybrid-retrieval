"""
Example 8: ChatOpenAI 最小运行示例
功能：调用大语言模型进行问答
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def basic_chat():
    """基础对话"""
    print("=" * 50)
    print("1. 基础对话")
    print("=" * 50)

    llm = ChatOpenAI(
        model="deepseek-chat",  # 模型名称
        base_url="https://api.deepseek.com/v1",  # API 地址
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # API Key
        temperature=0.3,  # 随机性：0-2，越高越随机
        max_tokens=2000,  # 最大输出 tokens
        timeout=30,  # 超时时间（秒）
    )

    # 单轮对话
    response = llm.invoke("你好，请介绍一下自己")
    answer = response.content if hasattr(response, "content") else response
    print(f"问题: 你好，请介绍一下自己")
    print(f"回答: {answer}\n")


def stream_chat():
    """流式输出"""
    print("=" * 50)
    print("2. 流式输出")
    print("=" * 50)

    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.3,
        max_tokens=2000,
    )

    query = "用三句话介绍自己"
    print(f"问题: {query}")
    print("回答: ", end="", flush=True)

    # 流式输出，逐块显示
    full_answer = ""
    for chunk in llm.stream(query):
        content = chunk.content if hasattr(chunk, "content") else chunk
        print(content, end="", flush=True)
        full_answer += content

    print("\n")


def chat_with_context():
    """带上下文的对话（构建 RAG 提示词）"""
    print("=" * 50)
    print("3. RAG 问答（带上下文）")
    print("=" * 50)

    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.3,
        max_tokens=2000,
    )

    # 模拟从向量数据库检索到的文档
    docs = [
        "银行电汇需要准备：收款人姓名、账户号、开户行名称及SWIFT代码",
        "银行电汇的手续费有两种承担方式：SHA（共同承担）和OUR（汇款人承担）",
        "Flywire支付的优势：汇率透明、到账快、手续费低",
    ]

    # 构建 RAG 提示词
    context = "\n".join([f"[文档{i+1}] {doc}" for i, doc in enumerate(docs)])
    prompt = f"""基于以下参考文档回答问题。如果文档中没有相关信息，请说明不知道。

参考文档：
{context}

问题：银行电汇需要准备什么？

请根据参考文档回答："""

    print(f"问题: 银行电汇需要准备什么？")
    print("回答: ", end="", flush=True)

    # 流式输出
    full_answer = ""
    for chunk in llm.stream(prompt):
        content = chunk.content if hasattr(chunk, "content") else chunk
        print(content, end="", flush=True)
        full_answer += content

    print("\n")


def chat_with_memory_storage():
    """带记忆存储的对话（LangChain 1.2 兼容）"""
    print("=" * 50)
    print("4. 带记忆存储的对话")
    print("=" * 50)

    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory

    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7,
        max_tokens=2000,
    )

    # 创建消息历史存储
    store = {}

    def get_session_history(session_id: str):
        """获取会话历史"""
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 创建带记忆的提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手，能够记住对话历史中的信息。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 创建链
    chain = prompt | llm

    # 包装成带历史的链
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 测试多轮对话（使用相同的 session_id）
    session_id = "user_001"
    conversations = [
        "你好，我叫小明",
        "我刚才介绍自己叫什么名字？",
        "我喜欢吃苹果，你喜欢什么水果？",
        "你还记得我的名字和喜好是什么吗？"
    ]

    for i, query in enumerate(conversations, 1):
        print(f"\n第{i}轮对话:")
        print(f"用户: {query}")

        # 获取带记忆的回复
        response = chain_with_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response.content}")

    print("\n")

if __name__ == "__main__":
    # basic_chat()
    # stream_chat()
    # chat_with_context()
    chat_with_memory_storage()
    print("完成!")