"""
文档管理 API：上传 / 列表 / 删除。
职责边界：只做参数校验和响应组装，真正的入库流程在 services.document_service。
"""
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services import document_service
from app.models.schemas import DocumentInfo

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}
MAX_FILE_SIZE_MB = 50


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """上传文档：解析 -> 切分 -> embedding -> 写入向量库/BM25/MySQL"""
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"文件超过 {MAX_FILE_SIZE_MB}MB 限制")

    # TODO(W2): 调用 document_service.ingest_document(filename, content, suffix)
    raise HTTPException(status_code=501, detail="W2 实现：document_service.ingest_document")


@router.get("", response_model=list[DocumentInfo])
async def list_documents():
    """文档列表（名称/块数/状态/上传时间）"""
    # TODO(W4): 调用 document_service.list_documents()
    return []


@router.delete("/{document_id}")
async def delete_document(document_id: int):
    """删除文档：⚠️ MySQL、Chroma、BM25 三处都要删，保持一致性"""
    # TODO(W4): 调用 document_service.delete_document(document_id)
    raise HTTPException(status_code=501, detail="W4 实现")
