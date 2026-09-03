"""
【W3】BM25 关键词检索：弥补纯向量检索对专有名词/精确匹配的短板。
依赖：rank_bm25；中文需先分词（jieba）。
注意：BM25 索引是内存结构，服务重启后需从 MySQL/Chroma 里的 chunk 重建；
     删除文档时要同步剔除对应 chunk（面试可能问"索引一致性怎么保证"）。
"""
# TODO(W3):
# import jieba
# from rank_bm25 import BM25Okapi
# 维护全局: _chunks: list[dict]（含 document_id）, _bm25: BM25Okapi | None
# 函数: add_chunks(document_id, chunks) / search(query, top_k) / delete_by_document(id) / rebuild()


def add_chunks(document_id: int, chunks: list[dict]):
    raise NotImplementedError


def search(query: str, top_k: int = 20) -> list[dict]:
    raise NotImplementedError


def delete_by_document(document_id: int):
    raise NotImplementedError
