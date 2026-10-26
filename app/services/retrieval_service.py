"""
检索编排层：把「召回 → 阈值过滤 → 精排」这三步固化成一个入口，并在此处挂缓存。

【为什么要抽这一层】
这段逻辑原本在 chat_service 里写了一份，做 Agent 时又在 tools.py 里复制了一份。
两处重复意味着：以后加个优化（比如缓存）要改两个地方，还容易改不一致。
抽出来之后，普通模式、Agent 模式、MCP Server 三条路都走这一个函数，
**加缓存也只需要加一处**。

【为什么缓存挂在"过滤 + 精排之后"而不是"召回之后"】
一次检索的成本构成：embedding API（~200ms）+ BM25（本地，~10ms）+ Rerank API（~300ms）。
最贵的是两次 API 调用。如果只缓存召回结果，Rerank 那次 API 还是省不掉。
所以缓存精排之后的最终结果，能把两次 API 调用一起省掉。

【一个容易踩的坑：必须深拷贝】
缓存里存的是 list[dict]。如果直接把这个 list 返回给调用方，
调用方一旦修改其中某个 dict（比如补充字段），改的是**缓存里的那份**，
下一次命中就会拿到被污染的数据。所以出缓存时一律 deepcopy。
"""
import copy
import logging

from app.cache import kb_cache
from app.config import settings
from app.rag import reranker, retriever

logger = logging.getLogger("app.retrieval")


def _do_search(query: str, final_top_k: int, rerank_top_k: int) -> list[dict]:
    """真正的检索（不带任何缓存）：召回 → 质量过滤 → 精排。"""
    # 1. 混合召回：向量 + BM25 各召回一批，RRF 融合（召回求广）
    hits = retriever.retrieve(query, final_top_k=final_top_k)

    # 2. 质量过滤：向量命中要求相似度达标；BM25 命中本身代表关键词真实重叠，直接保留
    hits = [
        h for h in hits
        if (h.get("similarity") is not None and h["similarity"] >= settings.SIMILARITY_THRESHOLD)
        or "bm25" in h.get("sources", [])
    ]

    # 3. 精排取 top_k；空候选时跳过，省一次 API 调用
    if hits:
        hits = reranker.rerank(query, hits, top_k=rerank_top_k)

    return hits


def search(
    query: str,
    final_top_k: int = 20,
    rerank_top_k: int | None = None,
) -> tuple[list[dict], bool]:
    """带缓存的检索入口。

    返回 (hits, cache_hit)：
      hits      —— 精排后的片段列表
      cache_hit —— 是否命中缓存（上层据此推 SSE 帧/打日志）
    """
    rerank_top_k = rerank_top_k or settings.RERANK_TOP_K

    # ---- 短路：总开关关掉时行为与加缓存前完全一致（压测对照组 / 一键回滚）----
    if not settings.CACHE_ENABLED:
        return _do_search(query, final_top_k, rerank_top_k), False

    key = kb_cache.make_key(query, scope=f"retrieval:{final_top_k}:{rerank_top_k}")
    if key is None:
        # 拿不到知识库指纹 → 不确认内容有没有变 → 禁用缓存，宁可慢不能错
        kb_cache.retrieval_cache.stats.disabled += 1
        return _do_search(query, final_top_k, rerank_top_k), False

    cached = kb_cache.retrieval_cache.get(key)
    if cached is not None:
        logger.info("[缓存命中] 检索：%s", query[:40])
        return copy.deepcopy(cached), True

    hits = _do_search(query, final_top_k, rerank_top_k)
    kb_cache.retrieval_cache.set(key, copy.deepcopy(hits))
    return hits, False
