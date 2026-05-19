from langchain_openai import OpenAIEmbeddings
import chromadb
from dotenv import load_dotenv
import os

load_dotenv() 
# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.com/v1",
    model="Qwen/Qwen3-Embedding-0.6B"
)

# ── 第一步：把文字转成向量 ─────────────────────
print("=== 文字转向量 ===")
texts = [
    "苹果公司2023年营收达到3830亿美元",
    "特斯拉是全球最大的电动车制造商",
    "巴菲特长期持有可口可乐股票",
    "市盈率是衡量股票估值的重要指标",
    "今天天气很好，适合出去散步"
]

vectors = embeddings.embed_documents(texts)
print(f"一共 {len(vectors)} 个向量")
print(f"每个向量维度：{len(vectors[0])}")

# ── 第二步：存入 Chroma ───────────────────────
print("\n=== 存入 Chroma ===")
client = chromadb.Client()
collection = client.create_collection("finance_test")

collection.add(
    documents=texts,
    embeddings=vectors,  # 手动传入向量，不让Chroma自动生成
    ids=[f"id{i}" for i in range(len(texts))]
)
print(f"已存入 {collection.count()} 条数据")

# ── 第三步：相似度检索 ────────────────────────
print("\n=== 相似度检索 ===")
query = "股票估值怎么看"
query_vector = embeddings.embed_query(query)

results = collection.query(
    query_embeddings=[query_vector],
    n_results=2
)

print(f"查询：{query}")
print("最相关的2条结果：")
for doc in results["documents"][0]:
    print(f"  → {doc}")