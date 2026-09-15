"""
【W3·Step10】混合检索：向量召回 + BM25 召回 -> RRF 融合排序。
核心思路（面试必讲）：
  - 向量召回：语义相近但可能漏掉精确术语 → "求广"
  - BM25 召回：关键词精确匹配但不懂语义 → "求准"
  - 两路分数体系不同（余弦相似度 0~1 vs BM25 无上限），不能直接相加
    → 用 RRF（倒数排名融合）：只看【排名】不看【分数】，score = Σ 1/(60+rank)
"""
import sys

from app.config import settings
from app.rag import bm25_store, embedder, vector_store

RRF_K = 60  # RRF 平滑常数，标准取 60：防止头部排名权重过于悬殊


def retrieve(query: str, final_top_k: int | None = None) -> list[dict]:
    """混合检索主入口：两路召回 -> RRF 融合 -> top_k。
    返回 [{content, filename, page, document_id, similarity|None, rrf, sources}]"""
    vector_hits = vector_store.search(embedder.embed_query(query))
    bm25_hits = bm25_store.search(query)
    return rrf_fuse(vector_hits, bm25_hits, final_top_k or settings.RERANK_TOP_K)


def _dedup_key(hit: dict) -> str:
    """同一段文本可能被两路都召回，按内容去重（取前缀足够判重）。"""
    return hit["content"][:100]


def rrf_fuse(vector_hits: list[dict], bm25_hits: list[dict], top_k: int) -> list[dict]:
    """RRF 融合：每条的总分 = 它在各路排名中 1/(60+rank) 的累加。
    两路都命中的条目自然获得双倍分数 → 排名靠前（这正是我们要的强信号）。"""
    merged: dict[str, dict] = {}

    for rank, h in enumerate(vector_hits):
        entry = merged.setdefault(
            _dedup_key(h), {**h, "sources": set(), "rrf": 0.0}
        )
        entry["sources"].add("vector")
        entry["rrf"] += 1 / (RRF_K + rank + 1)

    for rank, h in enumerate(bm25_hits):
        entry = merged.setdefault(
            _dedup_key(h), {**h, "similarity": h.get("similarity"), "sources": set(), "rrf": 0.0}
        )
        entry["sources"].add("bm25")
        entry["rrf"] += 1 / (RRF_K + rank + 1)

    results = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
    for r in results:
        r["sources"] = sorted(r["sources"])  # set -> list，方便 JSON 序列化
    return results


if __name__ == "__main__":
    # 对比实验：python -m app.rag.retriever "你的问题"
    # 打印 纯向量 top5 vs 混合融合 top5，直观感受 BM25 的补充作用
    query = sys.argv[1] if len(sys.argv) > 1 else "测试"

    vec_only = vector_store.search(embedder.embed_query(query), top_k=5)
    hybrid = retrieve(query, final_top_k=5)

    print(f"问题: {query}")
    print("=" * 60)
    print("【纯向量 top5】")
    for i, h in enumerate(vec_only, 1):
        print(f"  [{i}] sim={h['similarity']}  {h['filename']} 第{h['page']}页  {h['content'][:60].replace(chr(10), ' ')}...")
    print("=" * 60)
    print("【混合融合 top5】(sources 标记来源: vector/bm25)")
    for i, h in enumerate(hybrid, 1):
        sim = h['similarity'] if h['similarity'] is not None else '-'
        print(f"  [{i}] rrf={h['rrf']:.4f} sim={sim} 来自={h['sources']}  {h['content'][:60].replace(chr(10), ' ')}...")
