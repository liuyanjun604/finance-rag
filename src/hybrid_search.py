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

def build_bm25_index(texts: list) -> BM25Okapi:
    # 把每个文档块分词（简单按空格切），建立 BM25 索引
    # 上传时构建一次，查询时复用，避免每次查询都重建
    tokenized_corpus = [doc.split() for doc in texts]
    return BM25Okapi(tokenized_corpus)

def bm25_search(bm25: BM25Okapi, query: str) -> list:
    # 1. 查询分词
    tokenized_query = query.split()
    # 2. 用预建索引获取每个文档的分数
    scores = bm25.get_scores(tokenized_query)
    # 3. 返回 N_RESULTS 个文档的索引
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

def reciprocal_rank_fusion(bm25_ranks: list, vector_ranks: list, k: int = 60) -> list:
    # RRF（Reciprocal Rank Fusion）思路：
    # 两路检索各自给出一个排好序的索引列表，越靠前的文档越相关。
    # 对每个文档，按它在每路结果中的排名 rank（从 1 开始）累加得分 1/(k + rank)，
    # 排名越靠前贡献越大；同时出现在两路里的文档会被加两次，自然得到更高的总分。
    # k 是平滑常数（经验值 60），用来削弱头部名次的极端差距，让靠后的名次也有合理权重。
    scores = {}
    for ranked in (bm25_ranks, vector_ranks):
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    # 按 RRF 得分降序返回索引
    return sorted(scores, key=lambda idx: scores[idx], reverse=True)


# 开关：True 用 RRF 融合排序，False 用旧的「合并去重保持顺序」逻辑，方便对比测试
USE_RRF = True


def hybrid_search(texts: list, query: str, collection, bm25: BM25Okapi) -> list:
    if not texts or bm25 is None:
        return []  # 没有文档直接返回空列表
    bm25_indices = bm25_search(bm25, query)
    vector_indices = vector_search(query, collection)

    if USE_RRF:
        # 用 RRF 融合两路排名，得到按相关性排序的索引
        combined = reciprocal_rank_fusion(bm25_indices, vector_indices)
    else:
        # 旧逻辑：简单合并去重，保持先后顺序（bm25 优先）
        combined = list(dict.fromkeys(bm25_indices + vector_indices))

    # 返回排序后的文档内容
    return [texts[i] for i in combined]

if __name__ == "__main__":
    print("hybrid_search module ready")