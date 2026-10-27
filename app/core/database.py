"""
MySQL 连接：engine + Session 工厂 + FastAPI 依赖注入。
用法（在 API/Service 里）：
    from app.core.database import get_db
    def endpoint(db: Session = Depends(get_db)): ...
开发期没有 MySQL 时，把 .env 的 DATABASE_URL 改成 sqlite:///./data/dev.db 即可平滑过渡。
"""
import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings
from app.models.db_models import Base

logger = logging.getLogger(__name__)

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def ensure_prompt_version_column():
    """给已有的 chat_messages 表补 prompt_version 列（幂等）。

    为什么需要它：create_all() 只负责「建新表」，对已存在的表不会补列。
    老库升级后如果不补这一列，ORM 查询会 SELECT 一个不存在的字段而全部报错。

    幂等做法：先查 information_schema / PRAGMA 看列在不在，在就直接返回，
    不在才 ALTER。所以服务重启多少次都安全。
    """
    with engine.begin() as conn:
        existing = {c["name"] for c in inspect(conn).get_columns("chat_messages")}
        if "prompt_version" in existing:
            return False                     # 已经有了，什么都不做
        conn.execute(text("ALTER TABLE chat_messages ADD COLUMN prompt_version VARCHAR(20) NULL"))
        logger.info("[db] 已为 chat_messages 补上 prompt_version 列")
        return True                          # 本次真的加了列


def init_db():
    """建表（W4 第一次跑 MySQL 相关功能时调用，或在 main.py 启动事件里调用）"""
    Base.metadata.create_all(bind=engine)
    # 老库升级：补任务4 新增的列。这里故意不吞异常——
    # 列补不上会导致所有消息查询失败，静默跳过比直接报错更难排查。
    ensure_prompt_version_column()


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
