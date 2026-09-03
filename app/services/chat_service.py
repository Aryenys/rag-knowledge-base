"""
【W2/W3/W4】问答业务编排：在线 RAG 主链路 + SSE 帧组装。
链路：取历史 -> (W3)查询改写 -> 混合检索 -> rerank -> 组装 prompt -> 流式生成
     -> 推引用来源帧 -> 落库（W4）
SSE 帧格式见 api/chat.py 注释。
"""
import json

from app.rag import retriever, reranker, generator
from app.core import prompts


def stream_answer(question: str, session_id: int | None):
    """
    SSE 生成器：逐 token yield f"data: {json}\\n\\n"
    实现步骤：
      W2: 检索(纯向量) -> 生成 -> token 帧 + sources 帧 + done 帧
      W3: 换成 retriever.retrieve + reranker.rerank + generator.rewrite_query
      W4: 读写 MySQL 会话与消息
    """
    # TODO(W2)
    raise NotImplementedError
    # 伪代码参考：
    # history = load_history(session_id)                     # W4
    # standalone_q = generator.rewrite_query(question, history)  # W3
    # candidates = retriever.retrieve(standalone_q)          # W3（W2 阶段直接用 vector_store.search）
    # top_chunks = reranker.rerank(standalone_q, candidates) # W3
    # full_answer = ""
    # for token in generator.stream_generate(question, top_chunks, history):
    #     full_answer += token
    #     yield _sse({"type": "token", "content": token})
    # yield _sse({"type": "sources", "content": build_sources(top_chunks)})
    # yield _sse({"type": "done"})
    # save_messages(session_id, question, full_answer, top_chunks)  # W4


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
