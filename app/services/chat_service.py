"""
【W2·Step7】问答业务编排：在线 RAG 主链路 + SSE 帧组装。
链路：检索(纯向量) -> 流式生成 -> token 帧 + sources 帧 + done 帧
W3 升级点：检索换成 retriever.retrieve + reranker.rerank + generator.rewrite_query
W4 升级点：读写 MySQL 会话与消息
"""
import json

from app.config import settings
from app.rag import generator, retriever, reranker
from app.services import session_service


def stream_answer(question: str, session_id: int | None):
    """SSE 生成器：逐 token yield f"data: {json}\\n\\n" 格式的帧。"""
    # 0. 【W4】会话归属：前端没传 id 就新建一个（首次提问 / 点了"新对话"）
    if session_id is None:
        session_id = session_service.create_session()["id"]
    # 第一帧就把会话 id 推给前端。前端存下来、下次提问带上它 —— 多轮对话靠的就是这个
    yield _sse({"type": "session", "id": session_id})

    # 1. 从 MySQL 加载历史。
    #    ⚠️ 必须在保存本次提问【之前】加载，否则历史里会混入当前这句，
    #       改写模型就变成"把这句话补全成这句话"，指代消解直接失效。
    #    一轮对话 = user + assistant 两条，所以取 HISTORY_TURNS * 2 条。
    history = [
        # 只保留 role/content：sources 是给前端展示的引用 JSON，喂给模型纯属浪费 token
        {"role": m["role"], "content": m["content"]}
        for m in session_service.get_recent_messages(session_id, settings.HISTORY_TURNS * 2)
    ]

    # 2. 保存本次用户提问。history 为空 = 这是本会话第一条 → 用问题前 20 字当会话标题
    session_service.save_message(session_id, "user", question)
    if not history:
        session_service.update_session_title(session_id, question)

    # 3. 【Step12】查询改写：多轮对话时先做指代消解（"它" -> 具体事物）
    #    改写后的问题只用于"找资料"，生成回答仍用用户原话
    standalone_q = generator.rewrite_query(question, history)

    # 4. 检索：向量 + BM25 混合召回 20 条候选（召回求广）
    #    ⚠️ 用"改写后的问题"检索，但生成时仍用"原始问题"回答
    hits = retriever.retrieve(standalone_q, final_top_k=20)

    # 4.5 质量过滤：向量命中要求相似度达标；BM25 命中本身代表真实关键词重叠
    #     （score>0 才返回，高精度），直接保留。两路都被淘汰 → 走兜底拒答。
    hits = [
        h for h in hits
        if (h.get("similarity") is not None and h["similarity"] >= settings.SIMILARITY_THRESHOLD)
        or "bm25" in h.get("sources", [])
    ]

    # 4.6 【Step11】cross-encoder 精排取 top5（精排求准）；空候选跳过，省一次 API 调用
    if hits:
        hits = reranker.rerank(standalone_q, hits)

    # 5. 流式生成：边收 token 边推帧给前端（生成用原始问题 + 历史）
    full_answer = ""
    for token in generator.stream_generate(question, hits, history=history):
        full_answer += token  # 攒完整回答，生成结束后落库
        yield _sse({"type": "token", "content": token})

    # 6. 组装引用来源（落库和推给前端共用同一份，保证两边显示一致）
    sources = [
        {
            "index": i,
            "filename": h["filename"],
            "page": h["page"],
            "content": h["content"][:200],  # 截断预览，避免帧过大
            # 展示 rerank 相关分（cross-encoder 的权威分数）
            "score": h.get("rerank_score", h.get("similarity")),
        }
        for i, h in enumerate(hits, start=1)
    ]

    # 7. 落库助手回答（带上引用来源，回看旧会话时能还原引用面板）
    session_service.save_message(session_id, "assistant", full_answer, sources)

    # 8. 推引用来源 + 结束帧（前端惯例：回答显示完再展开引用面板）
    yield _sse({"type": "sources", "content": sources})
    yield _sse({"type": "done"})


def _sse(payload: dict) -> str:
    """把 dict 打包成一帧 SSE 数据。ensure_ascii=False：中文原样输出不转义。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
