from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import chromadb
import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage,ToolMessage
from fastapi.responses import StreamingResponse
import chromadb
from dotenv import load_dotenv
import os
from config import BASE_URL, LLM_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, N_RESULTS
from hybrid_search import hybrid_search
from agent import app as agent_app


app = FastAPI()

# 初始化：llm、embeddings、splitter、chroma client
load_dotenv() 
llm = ChatOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=BASE_URL,
    model=LLM_MODEL
)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=BASE_URL,
    model=EMBEDDING_MODEL
)


# 共享的 collection
client = chromadb.Client()
collection = client.create_collection("annual_report")

#全局变量
all_texts = []  # 存储所有文档块

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # 1. 读取上传的 PDF 文件内容
    import io
    contents = await file.read()

    # 上传的文件要先读取字节，再用 io.BytesIO 包装
    # 2. 用 PyPDF2 解析
    reader = PyPDF2.PdfReader(io.BytesIO(contents))
    raw_texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            raw_texts.append({"text": text, "page": i+1})
    # 3. 分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = []
    metadatas = []
    for item in raw_texts:
        chunks = splitter.split_text(item["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"page":item["page"]})
    global all_texts
    all_texts = texts
    # 4. 生成向量
    vectors = embeddings.embed_documents(texts)
    # 5. 存入 collection
    collection.add(
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
        ids=[f"id{i}" for i in range(len(texts))]
    )
    # 6. 返回 {"status": "ok", "chunks": 块数}
    return {"status": "ok", "chunks": len(texts)}
    pass

class AskRequest(BaseModel):
    question: str
    history: list = []  # 新增，默认空列表

# 历史记录的格式：
# [
#     {"role": "user", "content": "上一个问题"},
#     {"role": "assistant", "content": "上一个回答"}
# ]

@app.post("/ask-stream")
async def ask_stream(request: AskRequest):
    # 联合检索
    content = hybrid_search(all_texts, request.question, collection)
    # LLM 回答
    # 构建 messages
    messages = [
    SystemMessage(content=f"""你是一个金融文档助手，可以通过以下两种方式回答问题：
    1. 如果问题可以从以下文档内容中找到答案，直接根据文档回答：
    {content}
    2. 如果问题需要实时数据（如股票价格），使用工具获取。
    如果文档中没有相关信息且没有合适的工具，请说"文档中未提及"。""")
    ]        
    # 把历史对话加进去
    for msg in request.history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # 最后加上当前问题
    messages.append(HumanMessage(content=request.question))

    response = agent_app.invoke({"messages": messages})

    def generate():
        yield response["messages"][-1].content
    return StreamingResponse(generate(), media_type="text/plain")
    pass