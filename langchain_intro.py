from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化模型
llm = ChatOpenAI(
    api_key="sk-ecroneknxsvqdvwhdksbtfowceftjlyhflvqhvotfrrupreb",
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