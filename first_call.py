import os
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  

# 配置 API Key 和硅基流动的地址
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.com/v1"
)

# 第一次调用
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
    {
        "role": "system",
        "content": "你是一个专业的金融助手，专门帮助用户分析金融文档。回答要简洁专业，如果不确定就说不知道，不要编造数据。"
    },
    {
        "role": "user",
        "content": "什么是市盈率，怎么用它判断一只股票是否值得买？"
    }
]
)


# 打印回复
print(response.choices[0].message.content)
