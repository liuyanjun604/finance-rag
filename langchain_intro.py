from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()  

# 初始化模型
llm = ChatOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.com/v1",
    model="Qwen/Qwen2.5-7B-Instruct"
)

# ── 第一步：直接发消息 ──────────────────────
print("=== 直接调用 ===")
response = llm.invoke([
    SystemMessage(content="你是一个专业的金融助手，回答简洁专业。"),
    HumanMessage(content="用一句话解释什么是股息。")
])
print(response.content)

# ── 第二步：用 PromptTemplate ───────────────
print("\n=== PromptTemplate ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的金融助手。"),
    ("user", "用简单的话解释一下什么是{concept}，举一个实际例子。")
])

chain = prompt | llm
response2 = chain.invoke({"concept": "复利"})
print(response2.content)

# ── 第三步：多轮对话记忆 ───────────────────
print("\n=== 多轮对话 ===")
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

history_store = {}

def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的金融助手。"),
    ("placeholder", "{chat_history}"),
    ("user", "{input}")
])

chain_with_history = RunnableWithMessageHistory(
    prompt_with_history | llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 第一轮
r1 = chain_with_history.invoke(
    {"input": "我叫刘彦君，我在学习金融知识。"},
    config={"configurable": {"session_id": "test"}}
)
print("第一轮:", r1.content)

# 第二轮 - 测试它是否记得
r2 = chain_with_history.invoke(
    {"input": "我叫什么名字？我在学什么？"},
    config={"configurable": {"session_id": "test"}}
)
print("第二轮:", r2.content)