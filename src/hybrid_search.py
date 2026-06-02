from rank_bm25 import BM25Okapi
from config import BASE_URL, EMBEDDING_MODEL,N_RESULTS
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=BASE_URL,
    model=EMBEDDING_MODEL
)

def bm25_search(texts: list, query: str) -> list:
    # 1. 把每个文档块分词（简单按空格切）
    tokenized_corpus = [doc.split() for doc in texts]
    # 2. 建立 BM25 索引
    bm25 = BM25Okapi(tokenized_corpus)
    # 3. 查询分词
    tokenized_query = query.split()
    # 4. 获取每个文档的分数
    scores = bm25.get_scores(tokenized_query)
    # 5. 返回 N_RESULTS 个文档的索引
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N_RESULTS]
    return top_indices

def vector_search(query: str, collection) -> list:
    # 1. 把问题转成向量
    query_vector = embeddings.embed_query(query)

    # 2. 检索 collection
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=N_RESULTS
    )
    top_indices = [int(id.replace("id", "")) for id in results["ids"][0]]
    return top_indices

def hybrid_search(texts: list, query: str, collection) -> list:
    bm25_indices = bm25_search(texts, query)
    vector_indices = vector_search(query, collection)
    # 合并去重，保持顺序
    combined = list(dict.fromkeys(bm25_indices + vector_indices))
    # 返回合并后的文档内容
    return [texts[i] for i in combined]

if __name__ == "__main__":
    print("hybrid_search module ready")