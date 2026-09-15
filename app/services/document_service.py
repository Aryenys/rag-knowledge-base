"""
【Step9】文档业务编排：把 rag 层组件串成完整入库/删除流程。
W4 升级点：元数据从 data/documents.json 迁移到 MySQL（接口不变，只换存储实现）

入库流程：
  落盘 -> 登记元数据(processing) -> 解析 -> 切分 -> embedding
       -> 写 Chroma -> 重建 BM25 -> 标记 ready

职责边界：本文件只做"流程编排"，数据库读写一律交给 document_repository。
"""
import time
from pathlib import Path

from app.config import settings
from app.rag import loader, splitter, embedder, vector_store, bm25_store
from app.services import document_repository

UPLOAD_DIR = Path(settings.DATA_DIR)


def ingest_document(filename: str, content: bytes, suffix: str) -> dict:
    """文档入库主流程。任一步失败抛异常，由 API 层转成 400/500 响应。"""
    # 1. 落盘：文件名加时间戳前缀，防止同名文件互相覆盖
    safe_name = f"{int(time.time() * 1000)}_{filename}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(content)

    # 2. 先登记元数据，抢占一个自增 id（此时 status=processing）
    #    这个 id 要同时写进 Chroma，是元数据与向量之间的关联键
    doc = document_repository.create_document(filename, str(file_path))
    document_id = doc["id"]

    try:
        # 3. 离线链路：解析 -> 切分 -> embedding（这一步最慢，要走 API）
        pages = loader.load_document(file_path)
        chunks = splitter.split_documents(pages)
        embeddings = embedder.embed_texts([c["content"] for c in chunks])

        # 4. 写向量库 + 重建 BM25
        vector_store.add_chunks(document_id, chunks, embeddings, filename)
        bm25_store.rebuild()

        # 5. 标记就绪，并补上真实 chunk 数
        return document_repository.mark_ready(document_id, len(chunks))

    except Exception:
        # 失败兜底：标记 failed，并清理可能已写进向量库的一半数据。
        # 没有这一步就会留下"有向量、无元数据"的孤儿（之前 Step10 的 NameError 就是这么来的）
        document_repository.mark_failed(document_id)
        vector_store.delete_by_document(document_id)
        bm25_store.rebuild()
        raise


def list_documents() -> list[dict]:
    """文档列表（名称/块数/状态/上传时间）"""
    return document_repository.list_documents()


def delete_document(document_id: int) -> bool:
    """删除文档：向量库、元数据、磁盘文件三处保持一致。返回是否找到。"""
    doc = document_repository.get_document(document_id)
    if doc is None:
        return False

    vector_store.delete_by_document(document_id)      # ① 删向量
    bm25_store.rebuild()                              # ①.5 BM25 索引同步重建
    document_repository.delete_document(document_id)  # ② 删元数据

    file_path = Path(doc["file_path"])
    if file_path.exists():
        file_path.unlink()                            # ③ 删磁盘文件
    return True
