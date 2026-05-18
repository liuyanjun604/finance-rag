from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from dotenv import load_dotenv
import os

# 初始化模型
load_dotenv() 
llm = ChatOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.com/v1",
    model="Qwen/Qwen2.5-7B-Instruct"
)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.com/v1",
    model="Qwen/Qwen3-Embedding-0.6B"
)

#文字转向量
texts = [
    "花旗集团2023年全年营收为781亿美元，同比增长9%。其中利息净收入为478亿美元，非利息收入为303亿美元。净利润为92亿美元，较上年下降38%，主要原因是计提了约36亿美元的法律及重组费用。截至2023年底，花旗集团总资产为2.4万亿美元，在全球160个国家和地区开展业务。"
]
vectors = embeddings.embed_documents(texts)
client = chromadb.Client()
collection = client.create_collection("finance_test")

collection.add(
    documents=texts,
    embeddings=vectors, 
    ids=[f"id{i}" for i in range(len(texts))]
)

query = "花旗2023年净利润为什么下降了？"
# 检索最相关的内容 
query_vector = embeddings.embed_query(query)
results = collection.query(
    query_embeddings=[query_vector],
)

#把检索结果塞进 prompt 
content = ""
for doc in results["documents"][0]:
    content = content + doc

#LLM 根据这段内容回答
response = llm.invoke([
    SystemMessage(content=f"""你是一个金融文档助手，只根据以下内容回答问题，不要使用其他知识：
    {content}
    如果文档中没有相关信息，请说"文档中未提及"。"""),
    HumanMessage(content=query)
])
print(response.content)

