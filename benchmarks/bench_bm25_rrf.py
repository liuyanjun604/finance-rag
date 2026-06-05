"""
BM25 索引复用 vs 每次重建、以及 RRF vs 旧融合 的 benchmark。

- 语料：用项目里真实的年报 PDF，走和 /upload 一样的解析 + 分块流程。
- 函数：直接复用 src/hybrid_search.py 里的真实实现。
- 计时：time.perf_counter()，各跑 100 次取中位数。

运行：python benchmarks/bench_bm25_rrf.py
"""
import os
import statistics
import sys
import time

# 让脚本能 import 到 src/ 下的平级模块（config / hybrid_search）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chromadb
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, N_RESULTS
from hybrid_search import (
    build_bm25_index,
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    embeddings,
)

# 真实语料：项目根目录下的年报 PDF
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "citi-2025-annual-report.pdf")
# 一个典型金融查询（年报是英文，按空格分词，用英文关键词）
QUERY = "total revenue net income 2024"
RUNS = 100


def load_corpus() -> list:
    """走和 /upload 完全相同的流程：PyPDF2 解析每页 -> 分块。"""
    reader = PyPDF2.PdfReader(PDF_PATH)
    raw_texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text.strip():
            raw_texts.append(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    texts = []
    for text in raw_texts:
        texts.extend(splitter.split_text(text))
    return texts


def bench_rebuild(texts: list, query: str) -> float:
    """情况 (1)：每次查询都重新构建 BM25 索引。"""
    timings = []
    for _ in range(RUNS):
        start = time.perf_counter()
        bm25 = build_bm25_index(texts)
        bm25_search(bm25, query)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def bench_reuse(texts: list, query: str) -> float:
    """情况 (2)：复用预建好的 BM25 索引。"""
    bm25 = build_bm25_index(texts)  # 只建一次，不计入查询耗时
    timings = []
    for _ in range(RUNS):
        start = time.perf_counter()
        bm25_search(bm25, query)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def old_fusion(bm25_ranks: list, vector_ranks: list) -> list:
    """旧融合逻辑：简单合并去重，保持先后顺序（bm25 优先）。仅用于对比。"""
    return list(dict.fromkeys(bm25_ranks + vector_ranks))


def get_vector_ranks(texts: list, query: str) -> tuple:
    """跑真实向量检索：embed 全部块 -> 存入 chroma -> 查询。

    需要 embedding API；不可用时回退到一个带标注的模拟排名，保证脚本能跑完。
    返回 (vector_ranks, is_real)。
    """
    try:
        client = chromadb.Client()
        collection = client.create_collection("benchmark_rrf")
        vectors = embeddings.embed_documents(texts)
        collection.add(
            documents=texts,
            embeddings=vectors,
            ids=[f"id{i}" for i in range(len(texts))],
        )
        return vector_search(query, collection), True
    except Exception as e:
        # 没有 API key / 网络时的回退：构造一个与 bm25 部分重叠的模拟排名
        print(f"[提示] 向量检索不可用（{type(e).__name__}），改用模拟向量排名做对比演示")
        bm25 = build_bm25_index(texts)
        top = bm25_search(bm25, query)
        # 错位 + 引入新索引，模拟“两路结果不完全一致”
        simulated = top[1:] + [top[0]] + [top[0] + 1000, top[0] + 2000][:N_RESULTS - len(top)]
        return simulated[:N_RESULTS], False


def compare_fusion(texts: list, query: str):
    """对比 RRF 融合 vs 旧的合并去重融合：排序差异 + 融合本身的耗时。"""
    bm25 = build_bm25_index(texts)
    bm25_ranks = bm25_search(bm25, query)
    vector_ranks, is_real = get_vector_ranks(texts, query)

    print("\n=== RRF vs 旧融合 对比 ===")
    print(f"BM25   排序索引: {bm25_ranks}")
    print(f"向量   排序索引: {vector_ranks}" + ("" if is_real else "（模拟）"))

    rrf_ranks = reciprocal_rank_fusion(bm25_ranks, vector_ranks)
    old_ranks = old_fusion(bm25_ranks, vector_ranks)
    print(f"旧融合 排序索引: {old_ranks}")
    print(f"RRF    排序索引: {rrf_ranks}")
    print(f"两者 top-{N_RESULTS} 是否一致: {old_ranks[:N_RESULTS] == rrf_ranks[:N_RESULTS]}")

    # 融合步骤本身的耗时（均为纯内存计算，应在亚毫秒级）
    rrf_ms = statistics.median(
        [_time_call(reciprocal_rank_fusion, bm25_ranks, vector_ranks) for _ in range(RUNS)]
    ) * 1000
    old_ms = statistics.median(
        [_time_call(old_fusion, bm25_ranks, vector_ranks) for _ in range(RUNS)]
    ) * 1000
    print(f"旧融合 耗时: {old_ms:.4f} ms / 次")
    print(f"RRF    耗时: {rrf_ms:.4f} ms / 次")


def _time_call(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def main():
    texts = load_corpus()
    print(f"语料：{len(texts)} 个文档块（来自 {os.path.basename(PDF_PATH)}）")
    print(f"查询：{QUERY!r}")
    print(f"每种情况各跑 {RUNS} 次取中位数\n")

    rebuild_ms = bench_rebuild(texts, QUERY) * 1000
    reuse_ms = bench_reuse(texts, QUERY) * 1000

    print(f"(1) 每次重建索引   : {rebuild_ms:.3f} ms / 次查询")
    print(f"(2) 复用预建索引   : {reuse_ms:.3f} ms / 次查询")
    print(f"提升倍数           : {rebuild_ms / reuse_ms:.1f}x 更快")

    compare_fusion(texts, QUERY)


if __name__ == "__main__":
    main()
