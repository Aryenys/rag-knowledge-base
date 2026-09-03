"""
应用入口：创建 FastAPI 实例、注册路由、挂载前端静态文件。
启动方式（项目根目录）：
    uvicorn app.main:app --reload --port 8000
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import documents, chat

app = FastAPI(title="RAG Knowledge Base", version="0.1.0")

# 开发期允许前端跨域；生产环境由 FastAPI 托管静态文件后可收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
