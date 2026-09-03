"""
问答 API：SSE 流式问答 + 会话管理。
流式接口返回 text/event-stream，前端用 EventSource/fetch-stream 消费。
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, SessionInfo, MessageInfo

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
    # TODO(W2): 调用 chat_service.stream_answer(req)，返回 StreamingResponse
    # generator = chat_service.stream_answer(req)
    # return StreamingResponse(generator, media_type="text/event-stream")
    raise NotImplementedError("W2 实现：chat_service.stream_answer")


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """会话列表（左侧栏）"""
    # TODO(W4)
    return []


@router.get("/sessions/{session_id}/messages", response_model=list[MessageInfo])
async def get_messages(session_id: int):
    """某个会话的历史消息"""
    # TODO(W4)
    return []
