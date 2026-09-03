"""
【W2/W4】文档业务编排：把 rag 层的组件串成完整入库/删除流程。
入库流程：保存文件 -> 建 MySQL 记录(processing) -> 解析 -> 切分 -> embedding
         -> 写 Chroma + BM25 -> 更新 MySQL(ready, chunk_count)
         （任何一步失败 -> 状态置 failed，并清理已写入的半成品数据）
"""
from pathlib import Path

from app.config import settings
from app.rag import loader, splitter, embedder, vector_store  # , bm25_store  # W3 启用


def ingest_document(filename: str, content: bytes, suffix: str):
    """文档入库主流程（W2 实现核心链路，W3 加 BM25，W4 加 MySQL 状态机）"""
    # TODO(W2):
    # 1. 把 content 存到 data/{document_id}_{filename}（先存临时名，拿到 id 后改名）
    # 2. pages = loader.load_document(path)
    # 3. chunks = splitter.split_documents(pages)
    # 4. embeddings = embedder.embed_texts([c["content"] for c in chunks])
    # 5. vector_store.add_chunks(document_id, chunks, embeddings, filename)
    # 6. 返回 DocumentInfo
    raise NotImplementedError


def list_documents():
    # TODO(W4): 查 MySQL documents 表
    raise NotImplementedError


def delete_document(document_id: int):
    """删除文档：⚠️ 三处保持一致——MySQL 记录、Chroma 向量、BM25 索引、磁盘文件"""
    # TODO(W4)
    raise NotImplementedError
