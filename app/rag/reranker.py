"""
【W3】Rerank 精排：调硅基流动 BGE-Reranker API。
为什么需要（面试高频）：召回用 bi-encoder（向量分别编码，快但粗），
rerank 用 cross-encoder（query+chunk 一起过模型，慢但准）——所以"先广撒网、再精选"。
接口：POST {base_url}/rerank，body={"model", "query", "documents": [...], "top_n": N}
"""
import httpx

from app.config import settings


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    输入：query + 候选 chunk 列表
    输出：按相关性降序的 top RERANK_TOP_K 个 chunk（附带 score 字段）
    """
    # TODO(W3): 调 rerank API，把 relevance_score 写回 chunk["score"]，按分数取 top_k
    raise NotImplementedError
