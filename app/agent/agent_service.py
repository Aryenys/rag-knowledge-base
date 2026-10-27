"""
Agent 决策层：让模型自己决定「直接回答 / 调检索工具 / 调统计工具」。

与普通模式（chat_service.stream_answer）的区别：
  普通模式：写死的链路「检索 -> 生成」，每次提问必检索，闲聊也要付一次检索成本
  Agent 模式：把"要不要检索、检索几次"交给模型判断，可连续多轮调用工具

三个关键实现点（面试可展开讲）：
1. **流式 + tools 一次调用**：delta.content 直接转发前端（保住打字机体验），
   delta.tool_calls 分片累积。这样"模型直接回答"时不额外多发一次请求。
2. **防死循环硬闸门**：工具调用最多 AGENT_MAX_TOOL_ROUNDS 轮，
   达到上限后不再传 tools —— 没有工具可调，模型只能基于已有信息给最终答案。
3. **OpenAI 协议要求**：assistant 的 tool_calls 消息必须先回填，
   再逐条回填 role=tool 的工具结果，否则下一轮请求会被拒绝。
"""
import json

from openai import OpenAI

from app.agent import tools as agent_tools
from app.config import settings
from app.core import prompts
from app.services import session_service

# 与 generator 相同的接入方式：DeepSeek 兼容 OpenAI SDK
_client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)


def stream_agent_answer(question: str, session_id: int | None):
    """Agent 模式的 SSE 生成器。

    帧类型：session / tool / tool_result / token / sources / done
    相比普通模式新增 tool、tool_result 两种帧，用于前端展示"模型正在调用什么工具"。
    """
    # 0. 会话归属：与普通模式一致，前端没传 id 就新建，第一帧把 id 推回去
    if session_id is None:
        session_id = session_service.create_session()["id"]
    yield _sse({"type": "session", "id": session_id})

    # 1. 先加载历史，再保存本次提问（顺序不能反，否则历史里会混入当前这句）
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in session_service.get_recent_messages(session_id, settings.HISTORY_TURNS * 2)
    ]
    session_service.save_message(session_id, "user", question)
    if not history:
        session_service.update_session_title(session_id, question)

    # 2. 组装对话：Agent 系统提示里写清了"什么时候该用哪个工具"
    messages = [
        {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": question},
    ]

    full_answer = ""
    last_chunks: list[dict] = []   # 最后一次检索的片段，用于引用来源展示
    tool_round = 0

    while True:
        # 达到工具调用上限后不再提供 tools：这是防死循环的硬闸门
        allow_tools = tool_round < settings.AGENT_MAX_TOOL_ROUNDS
        kwargs = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "stream": True,
        }
        if allow_tools:
            kwargs["tools"] = agent_tools.TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"   # auto = 由模型自己决定调不调工具

        stream = _client.chat.completions.create(**kwargs)

        pending: dict[int, dict] = {}   # index -> 正在累积的一条 tool_call
        called = False
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 文本增量：直接转发，前端逐字追加
            if delta.content:
                full_answer += delta.content
                yield _sse({"type": "token", "content": delta.content})

            # 工具调用增量：OpenAI 会把 name/arguments 拆成多片推过来，必须按 index 累积拼接
            for tc in (delta.tool_calls or []):
                called = True
                slot = pending.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

        # 模型这一轮没有要工具 -> 答案已经流式输出完，结束整个循环
        if not called:
            break

        tool_round += 1
        ordered = [pending[i] for i in sorted(pending)]

        # 协议要求：assistant 的 tool_calls 消息先回填，再回填各条 role=tool 结果
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": t["id"],
                    "type": "function",
                    "function": {"name": t["name"], "arguments": t["arguments"]},
                }
                for t in ordered
            ],
        })

        for t in ordered:
            try:
                args = json.loads(t["arguments"] or "{}")
            except json.JSONDecodeError:
                # 模型偶发输出非法 JSON：给空参数让工具返回错误提示，而不是让请求崩掉
                args = {}

            # 工具调用过程推给前端：先"调用了什么"，再"结果如何"
            yield _sse({"type": "tool", "name": t["name"], "arguments": args, "round": tool_round})
            result = agent_tools.run_tool(t["name"], args)
            yield _sse({
                "type": "tool_result",
                "name": t["name"],
                "summary": result["summary"],
                "round": tool_round,
            })

            if result["chunks"]:
                # 以最后一次检索为准：这样前端引用编号 [1]..[n] 与模型在该次工具结果里看到的编号一致
                last_chunks = result["chunks"]

            messages.append({
                "role": "tool",
                "tool_call_id": t["id"],
                "content": result["for_model"],
            })
        # 继续下一轮：模型拿到工具结果后，决定"再调一次"还是"给最终答案"

    # 3. 组装引用来源（与普通模式同构，前端引用面板可直接复用）
    sources = [
        {
            "index": i,
            "filename": h["filename"],
            "page": h["page"],
            "content": h["content"][:200],
            "score": h.get("rerank_score", h.get("similarity")),
        }
        for i, h in enumerate(last_chunks, start=1)
    ]

    # 4. 落库 + 推引用 + 结束帧
    session_service.save_message(
        session_id, "assistant", full_answer, sources,
        prompt_version=prompts.CURRENT_PROMPT_VERSION,
    )
    yield _sse({"type": "meta", "prompt_version": prompts.CURRENT_PROMPT_VERSION})
    yield _sse({"type": "sources", "content": sources})
    yield _sse({"type": "done"})


def _sse(payload: dict) -> str:
    """打包一帧 SSE。ensure_ascii=False：中文原样输出不转义。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
