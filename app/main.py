"""
应用入口：创建 FastAPI 实例、注册路由、挂载前端静态文件。
启动方式（项目根目录）：
    uvicorn app.main:app --reload --port 8000

【W6·Step1】新增两件事：
  1. 日志：请求记录写进 logs/app.log，出事后能翻旧账
  2. 全局异常处理：未捕获的异常统一转成 JSON 响应，并记下完整 traceback
     （以前这类错误会变成纯文本 "Internal Server Error"，前端没法解析）
"""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import documents, chat
from app.core.database import init_db
from app.core.logging import get_logger, mute_noisy, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：启动时建表，关闭时收尾。

    用 lifespan 而不是已废弃的 @app.on_event("startup")——后者在 FastAPI
    新版本中会告警，lifespan 是官方推荐的现代写法（同样被 uvicorn 支持）。
    """
    init_db()   # 幂等建表：只对不存在的表动手，已有表不会被覆盖
    # 此时所有依赖模块都已导入完毕，再降噪一次才能覆盖 jieba、httpx 这类
    # "用到才导入"的第三方库 logger
    mute_noisy()
    logger.info("应用启动完成，数据库表检查通过")
    yield
    logger.info("应用正在关闭")


app = FastAPI(title="RAG Knowledge Base", version="0.1.0", lifespan=lifespan)

# 开发期允许前端跨域；生产环境由 FastAPI 托管静态文件后可收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件：记录每个请求的 方法/路径/状态码/耗时。

    慢请求（>3 秒）单独用 WARNING 标记，方便排查入库、检索这类耗时操作。
    """
    start = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        cost = time.time() - start
        msg = f"{request.method} {request.url.path} -> {status_code} ({cost:.2f}s)"
        if cost > 3:
            logger.warning(f"[慢请求] {msg}")
        else:
            logger.info(msg)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底异常处理器：任何没被业务代码捕获的异常都会走到这里。

    做两件事：
      1. logger.exception 会把完整 traceback 写进日志文件（排查靠它）
      2. 返回统一格式的 JSON，前端能稳定解析出 detail 字段
    """
    logger.exception(f"未捕获异常: {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {type(exc).__name__}"},
    )


# 注册 API 路由（统一 /api 前缀，方便前端对接）
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/api/health")
def health():
    """健康检查：docker-compose 和部署脚本都会用到"""
    return {"status": "ok"}


# 托管 AI 生成的前端（构建产物放到 frontend/ 目录下）
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if (frontend_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
