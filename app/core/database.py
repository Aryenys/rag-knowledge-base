"""
MySQL 连接：engine + Session 工厂 + FastAPI 依赖注入。
用法（在 API/Service 里）：
    from app.core.database import get_db
    def endpoint(db: Session = Depends(get_db)): ...
开发期没有 MySQL 时，把 .env 的 DATABASE_URL 改成 sqlite:///./data/dev.db 即可平滑过渡。
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings
from app.models.db_models import Base

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """建表（W4 第一次跑 MySQL 相关功能时调用，或在 main.py 启动事件里调用）"""
    Base.metadata.create_all(bind=engine)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
