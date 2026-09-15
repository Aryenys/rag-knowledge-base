"""
SQLAlchemy 数据库模型：三张表。
  Document    文档元数据（与向量库/BM25 通过 id 关联）
  ChatSession 会话
  ChatMessage 消息（含引用来源 JSON）
"""
from datetime import datetime

from sqlalchemy import Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.time_utils import utc_now


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)  # data/ 下的存储路径
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="processing")  # processing/ready/failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), default="新会话")  # 可用首条问题截断生成
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)  # assistant 消息的引用来源
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
