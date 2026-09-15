"""
【W3·Step11】Rerank 精排：调硅基流动 BGE-Reranker API（cross-encoder）。
为什么需要（面试高频）：召回用 bi-encoder（向量分别编码，快但粗），
rerank 用 cross-encoder（query+chunk 一起过模型，慢但准）——所以"先广撒网、再精选"。
接口：POST {base_url}/rerank，body={"model", "query", "documents": [...], "top_n": N}
"""
import sys
import time

import httpx

from app.config import settings

MAX_RETRY = 3


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """
    输入：query + 候选 chunk 列表（混合召回的 20 条左右）
    输出：按 rerank 相关分降序的 top_k 个 chunk（附带 rerank_score 字段）
    """
    if not candidates:
        return []
    top_k = top_k or settings.RERANK_TOP_K

    url = f"{settings.SILICONFLOW_BASE_URL}/rerank"
    headers = {"Authorization": f"Bearer {settings.SILICONFLOW_API_KEY}"}
    payload = {
        "model": settings.RERANK_MODEL,
        "query": query,
        "documents": [c["content"] for c in candidates],
        "top_n": min(top_k, len(candidates)),
    }

    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  [rerank 限流 429] 等待 {wait}s 后重试...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            results = resp.json()["results"]  # [{"index": i, "relevance_score": s}]
            reranked = []
            for r in results:
                chunk = dict(candidates[r["index"]])   # 复制，不污染原对象
                chunk["rerank_score"] = round(r["relevance_score"], 4)
                reranked.append(chunk)
            return reranked   # API 已按相关分降序返回
        except httpx.HTTPError as e:
            if attempt == MAX_RETRY:
                raise RuntimeError(f"rerank 请求失败（重试 {MAX_RETRY} 次后放弃）: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    # 对比实验：python -m app.rag.reranker "你的问题"
    # 打印 RRF 融合顺序 vs rerank 精排顺序，观察 rerank 的调整
    from app.rag import retriever

    query = sys.argv[1] if len(sys.argv) > 1 else "测试"
    candidates = retriever.retrieve(query, final_top_k=20)
    reranked = rerank(query, candidates)

    print(f"问题: {query}   候选 {len(candidates)} 条")
    print("=" * 60)
    print("【RRF 融合顺序 top5】(rerank 前)")
    for i, h in enumerate(candidates[:5], 1):
        print(f"  [{i}] rrf={h['rrf']:.4f} 来自={h['sources']}  {h['content'][:50].replace(chr(10), ' ')}...")
    print("=" * 60)
    print("【rerank 精排后 top5】")
    for i, h in enumerate(reranked, 1):
        print(f"  [{i}] rerank_score={h['rerank_score']}  {h['filename']} 第{h['page']}页  {h['content'][:50].replace(chr(10), ' ')}...")
