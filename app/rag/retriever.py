"""
【W3】混合检索：向量召回 ∪ BM25 召回 -> 去重合并 -> 交给 reranker。
核心思路（面试必讲）：
  - 向量召回：语义相近但可能漏掉精确术语 → "求广"
  - BM25 召回：关键词精确匹配但不懂语义 → "求准"
  - 两路各取 top20，按内容去重合并，得到 25-30 个候选（召回阶段宁多勿漏）
  - 精排交给 reranker（cross-encoder 比 bi-encoder 更准但更慢，所以只排小集合）
"""
from app.config import settings
from app.rag import vector_store, bm25_store, embedder


def retrieve(query: str) -> list[dict]:
    """
    输入：改写后的问题
    输出：去重合并后的候选 chunk 列表（未精排）
    """
    # TODO(W3):
    # 1. query_emb = embedder.embed_query(query)
    # 2. vec_results = vector_store.search(query_emb)          # top VECTOR_TOP_K
    # 3. bm25_results = bm25_store.search(query, settings.BM25_TOP_K)
    # 4. 按 content 去重合并（保留首次出现的元数据）
    raise NotImplementedError
