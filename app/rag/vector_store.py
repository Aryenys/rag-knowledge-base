"""
【W1·Step5】向量库封装：Chroma 本地持久化。
设计：一个全局 collection；每条 chunk 的 id 用 f"doc{document_id}_chunk{i}" 规则生成，
     这样删除文档时可以按 where={"document_id": x} 精确删除（保证和 MySQL/BM25 一致）。
"""
import sys

import chromadb

from app.config import settings

_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)  # 数据落盘到 data/chroma/
_collection = _client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},  # 相似度度量：余弦（语义方向一致即相近）
)


def add_chunks(
    document_id: int,
    chunks: list[dict],
    embeddings: list[list[float]],
    filename: str,
):
    """把一批 chunk + 向量写入向量库，元数据带上 document_id / filename / page。"""
    ids = [f"doc{document_id}_chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            # Chroma 元数据不接受 None，页码缺失时用 -1 占位
            "page": c["page"] if c["page"] is not None else -1,
        }
        for c in chunks
    ]
    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=[c["content"] for c in chunks],  # 原文也存进 Chroma，检索时直接取回
        metadatas=metadatas,
    )
    return len(ids)


def search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    """向量召回：返回 [{"content", "filename", "page", "document_id", "similarity"}]。"""
    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k or settings.VECTOR_TOP_K,
    )
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "content": doc,
                "filename": meta["filename"],
                "page": None if meta["page"] == -1 else meta["page"],
                "document_id": meta["document_id"],
                # cosine space 下 Chroma 返回的是"距离"，相似度 = 1 - 距离
                "similarity": round(1 - dist, 4),
            }
        )
    return hits


def delete_by_document(document_id: int):
    """删除某个文档的所有 chunk（W4 删除接口用）。"""
    _collection.delete(where={"document_id": document_id})


def count() -> int:
    """库里当前 chunk 总数（观察入库是否成功用）。"""
    return _collection.count()


def get_all_chunks() -> list[dict]:
    """取出库中全部 chunk（内容+元数据）。BM25 索引以 Chroma 为单一数据源重建用。"""
    result = _collection.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        chunks.append({
            "content": doc,
            "filename": meta["filename"],
            "page": None if meta["page"] == -1 else meta["page"],
            "document_id": meta["document_id"],
        })
    return chunks


if __name__ == "__main__":
    # W1 闭环测试：
    #   入库: python -m app.rag.vector_store add data/你的文件.pdf 1
    #   检索: python -m app.rag.vector_store query "你的问题" 5
    from app.rag.loader import load_document
    from app.rag.splitter import split_documents
    from app.rag.embedder import embed_texts, embed_query

    if len(sys.argv) < 2:
        print("用法: python -m app.rag.vector_store add <文件> <doc_id>")
        print("      python -m app.rag.vector_store query <问题> [top_k]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "add":
        file_path, doc_id = sys.argv[2], int(sys.argv[3])
        filename = file_path.replace("\\", "/").split("/")[-1]
        print(f"[1/3] 解析 {filename} ...")
        pages = load_document(file_path)
        print(f"[2/3] 切分 ...")
        chunks = split_documents(pages)
        print(f"[3/3] embedding 并入库 ({len(chunks)} 个 chunk) ...")
        vectors = embed_texts([c["content"] for c in chunks])
        n = add_chunks(doc_id, chunks, vectors, filename)
        print(f"完成! 本次入库 {n} 条, 库中总计 {count()} 条")

    elif mode == "query":
        question = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        hits = search(embed_query(question), top_k=top_k)
        print(f"问题: {question}\n共 {count()} 条 chunk, 召回 top{top_k}:")
        print("=" * 60)
        for i, h in enumerate(hits, 1):
            preview = h["content"][:120].replace("\n", " ")
            print(f"[{i}] 相似度={h['similarity']}  来源={h['filename']} 第{h['page']}页")
            print(f"    {preview}...")
