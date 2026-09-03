"""
Pydantic 请求/响应模型（API 层的"数据契约"，前后端对齐用）。
"""
from datetime import datetime
from pydantic import BaseModel


# ---------- 文档 ----------
class DocumentInfo(BaseModel):
    id: int
    filename: str
    chunk_count: int
    status: str            # processing / ready / failed
    created_at: datetime


# ---------- 问答 ----------
class ChatRequest(BaseModel):
    question: str
    session_id: int | None = None   # 为空表示新会话


class SourceChunk(BaseModel):
    """一条引用来源，前端折叠面板展示"""
    index: int           # 编号 [1] [2] ...
    filename: str
    page: int | None = None
    content: str         # 原文片段
    score: float | None = None   # rerank 分数


class SessionInfo(BaseModel):
    id: int
    title: str
    created_at: datetime


class MessageInfo(BaseModel):
    id: int
    role: str            # user / assistant
    content: str
    sources: list[SourceChunk] | None = None
    created_at: datetime
