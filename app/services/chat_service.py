"""
【W2·Step7】问答业务编排：在线 RAG 主链路 + SSE 帧组装。
链路：检索(纯向量) -> 流式生成 -> token 帧 + sources 帧 + done 帧
W3 升级点：检索换成 retriever.retrieve + reranker.rerank + generator.rewrite_query
W4 升级点：读写 MySQL 会话与消息
"""
import json

from app.cache import kb_cache
from app.config import settings
from app.core import prompts
from app.rag import generator
from app.services import retrieval_service, session_service


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

    # 4. 检索（带缓存）：召回 → 质量过滤 → 精排，三步封装在 retrieval_service 里
    #    ⚠️ 用"改写后的问题"检索，但生成时仍用"原始问题"回答
    #    命中缓存时会直接返回上次的结果，省掉 embedding 和 rerank 两次 API 调用
    hits, retrieval_cache_hit = retrieval_service.search(standalone_q, final_top_k=20)
    if retrieval_cache_hit:
        # 单独推一帧，前端据此显示"缓存命中"徽标（在回答出来之前就能看到）
        yield _sse({"type": "cache", "layer": "retrieval", "hit": True})

    # 4.7 【任务3】答案缓存：命中时连 LLM 生成都省掉，直接回放上次的结果。
    #     ⚠️ 只在「无历史」时启用：有历史时同一个问题的答案依赖上下文，
    #        不同会话里含义可能完全不同，缓存了会串台。
    #     默认关闭（CACHE_ANSWER_ENABLED=false）：生成有随机性，且回放是"一次性推完"，
    #        不是真的流式。只在"同一问题被反复问"的场景才划算，交给部署方决定。
    answer_key = None
    if settings.CACHE_ENABLED and settings.CACHE_ANSWER_ENABLED and not history:
        answer_key = kb_cache.make_key(question, scope="answer")

    if answer_key:
        cached = kb_cache.answer_cache.get(answer_key)
        if cached is not None:
            yield _sse({"type": "cache", "layer": "answer", "hit": True})
            # 版本要跟着缓存走：这条答案是当初那个版本生成的，
            # 不能因为现在换了默认版本就把功劳记到新版头上
            used_version = cached.get("prompt_version") or prompts.CURRENT_PROMPT_VERSION
            yield _sse({"type": "meta", "prompt_version": used_version})
            # 一次性推完整答案：缓存的价值就是快，没必要为了像打字机而人为 sleep 拖慢
            yield _sse({"type": "token", "content": cached["answer"]})
            session_service.save_message(
                session_id, "assistant", cached["answer"], cached["sources"],
                prompt_version=used_version,
            )
            yield _sse({"type": "sources", "content": cached["sources"]})
            yield _sse({"type": "done"})
            return

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
    # 记下生成这条答案用的 prompt 版本（任务4）：事后要排查"哪版 prompt 把答案变差了"就靠它
    session_service.save_message(
        session_id, "assistant", full_answer, sources,
        prompt_version=prompts.CURRENT_PROMPT_VERSION,
    )

    # 7.5 写答案缓存（引用来源一起存，否则命中时引用面板会是空的）
    if answer_key:
        kb_cache.answer_cache.set(answer_key, {
            "answer": full_answer,
            "sources": sources,
            "prompt_version": prompts.CURRENT_PROMPT_VERSION,
        })

    # 8. 推引用来源 + 结束帧（前端惯例：回答显示完再展开引用面板）
    yield _sse({"type": "meta", "prompt_version": prompts.CURRENT_PROMPT_VERSION})
    yield _sse({"type": "sources", "content": sources})
    yield _sse({"type": "done"})


def _sse(payload: dict) -> str:
    """把 dict 打包成一帧 SSE 数据。ensure_ascii=False：中文原样输出不转义。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
