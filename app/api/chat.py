"""
问答 API：SSE 流式问答 + 会话管理。
流式接口返回 text/event-stream，前端用 EventSource/fetch-stream 消费。
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, SessionInfo, MessageInfo
from app.services import chat_service, session_service

router = APIRouter()


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    流式问答（SSE）。
    数据帧格式约定（前端按此解析）：
      - data: {"type": "token",   "content": "..."}   逐 token 推送
      - data: {"type": "sources", "content": [...]}   生成完毕后推送引用来源
      - data: {"type": "done"}                          结束帧
    """
    return StreamingResponse(
        chat_service.stream_answer(req.question, req.session_id),
        media_type="text/event-stream",  # 告诉浏览器这是 SSE 协议
        headers={
            "Cache-Control": "no-cache",   # 禁止缓存，保证实时性
            "X-Accel-Buffering": "no",     # 禁止 Nginx 等反代缓冲（防"假流式"）
        },
    )


@router.post("/sessions", response_model=SessionInfo)
async def create_session():
    """新建会话。前端点"新对话"时调用，拿返回的 id 走后续的提问请求"""
    return session_service.create_session()


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """会话列表（左侧栏），按创建时间倒序"""
    return session_service.list_sessions()


@router.get("/sessions/{session_id}/messages", response_model=list[MessageInfo])
async def get_messages(session_id: int):
    """某个会话的历史消息（回看旧会话时用）"""
    session = session_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session_service.get_messages(session_id)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: int):
    """删除会话，关联消息由数据库级联删除"""
    if not session_service.delete_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": session_id}
