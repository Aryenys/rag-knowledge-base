"""
【W4·Step4】文档元数据的数据访问层。

定位与 session_service 平级：只做数据库读写，不做业务。
解析、切分、embedding 那些业务流程在 document_service 里。

为什么把文档的元数据也单独抽一层？
  和 Step2 的道理一样——换数据库、写测试、加缓存时只改这一处，
  业务流程（document_service）完全不用动。
"""
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.time_utils import ensure_utc
from app.models.db_models import Document


# ---------- ORM 对象 -> dict（必须在 session 关闭前完成） ----------

def _doc_to_dict(d: Document) -> dict:
    return {
        "id": d.id,
        "filename": d.filename,
        "file_path": d.file_path,
        "chunk_count": d.chunk_count,
        "status": d.status,
        "created_at": ensure_utc(d.created_at),
    }


# ---------- 写 ----------

def create_document(filename: str, file_path: str) -> dict:
    """先落一条 status=processing 的记录，用它抢占一个自增 id。

    为什么不一气呵成地写完？
    入库要花几十秒（embedding 要调 API），这期间前端刷新列表应该能
    看到"处理中"，而不是"什么都没有"。用 JSON 文件时做不到这点，
    这正换成数据库的价值之一。
    """
    db: Session = SessionLocal()
    try:
        doc = Document(
            filename=filename,
            file_path=file_path,
            chunk_count=0,
            status="processing",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)  # 拿自增 id
        return _doc_to_dict(doc)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def mark_ready(doc_id: int, chunk_count: int) -> dict:
    """入库成功：补上 chunk 数并标记为 ready"""
    return _update(doc_id, chunk_count=chunk_count, status="ready")


def mark_failed(doc_id: int) -> dict | None:
    """入库失败：标记 failed，方便排查哪些文档没处理好"""
    return _update(doc_id, status="failed")


def _update(doc_id: int, **fields) -> dict | None:
    db: Session = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc is None:
            return None
        for k, v in fields.items():
            setattr(doc, k, v)
        db.commit()
        db.refresh(doc)
        return _doc_to_dict(doc)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_with_id(info: dict) -> dict:
    """迁移专用：保留原始 id 插入（向量是按旧 id 存的，不能变）"""
    db: Session = SessionLocal()
    try:
        doc = Document(
            id=info["id"],
            filename=info["filename"],
            file_path=info["file_path"],
            chunk_count=info["chunk_count"],
            status=info["status"],
            created_at=info["created_at"],
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return _doc_to_dict(doc)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_document(doc_id: int) -> bool:
    db: Session = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc is None:
            return False
        db.delete(doc)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------- 读 ----------

def get_document(doc_id: int) -> dict | None:
    db: Session = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        return _doc_to_dict(doc) if doc else None
    finally:
        db.close()


def list_documents() -> list[dict]:
    """文档列表，按上传时间倒序（最新上传在最上面）"""
    db: Session = SessionLocal()
    try:
        stmt = select(Document).order_by(Document.id.desc())
        return [_doc_to_dict(d) for d in db.execute(stmt).scalars().all()]
    finally:
        db.close()


def existing_ids() -> set[int]:
    """迁移用：已有 id 集合，避免重复插入主键冲突"""
    db: Session = SessionLocal()
    try:
        return set(db.execute(select(Document.id)).scalars().all())
    finally:
        db.close()
