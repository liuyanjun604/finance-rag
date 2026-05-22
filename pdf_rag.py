import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

#把文字转成向量
raw_texts = []
with open("citi-2025-annual-report.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():  # 跳过空页
            raw_texts.append({"text": text, "page": i+1})

# 切块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
texts = []
metadatas = []
for item in raw_texts:
    chunks = splitter.split_text(item["text"])
    for chunk in chunks:
        texts.append(chunk)
        metadatas.append({"page": item["page"]})

vectors = embeddings.embed_documents(texts)

#存入chroma
client = chromadb.Client()
collection = client.create_collection("annual_report")

collection.add(
    documents=texts,
    embeddings=vectors,
    metadatas=metadatas,
    ids=[f"id{i}" for i in range(len(texts))]
)

query = "Citigroup net income 2024 2025 annual results billions"
#检索相关内容
query_vector = embeddings.embed_query(query)
results = collection.query(
    query_embeddings=[query_vector],
    n_results=8
)

# 把检索结果塞进 prompt 
content = []
for doc in results["documents"]:
    content.append(doc)
# LLM 根据这段内容回答
response = llm.invoke([
    SystemMessage(content=f"""你是一个金融文档助手，只根据以下内容回答问题，不要使用其他知识：
    {content}
    如果文档中没有相关信息，请说"文档中未提及"。"""),
    HumanMessage(content=query)
])
print(response.content)
