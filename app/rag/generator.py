"""
【W2·Step6】生成：组装 prompt -> 调 DeepSeek 流式生成。
DeepSeek 兼容 OpenAI SDK：openai.OpenAI(api_key=..., base_url=...)，stream=True。
输出是生成器（generator），由 chat_service 包成 SSE 帧推给前端。
"""
import sys

from openai import OpenAI

from app.config import settings
from app.core import prompts

_client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)


def rewrite_query(question: str, history: list[dict]) -> str:
    """【W3·Step12】查询改写：结合历史把"它怎么样"补全成独立完整的问题。
    用于检索前的 query 标准化（指代消解）。非流式，一次拿结果；temperature=0 求稳定。
    无历史时直接返回原问题（省一次 LLM 调用）。"""
    if not history:
        return question

    prompt = prompts.QUERY_REWRITE_PROMPT.format(
        history=prompts.format_history(history),
        question=question,
    )
    resp = _client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # 改写是确定性任务，不需要任何"发挥"
    )
    return resp.choices[0].message.content.strip()


def stream_generate(question: str, chunks: list[dict], history: list[dict]):
    """
    流式生成回答（生成器，逐 token yield 字符串）。
    消息结构：system(带参考资料的 RAG prompt) + 历史消息 + user(当前问题)
    """
    # 1. 组装 system prompt：有参考资料用 RAG 模板；为空（闲聊/未命中）用兜底模板
    if chunks:
        system_content = prompts.RAG_SYSTEM_PROMPT.format(
            context=prompts.build_context(chunks)
        )
    else:
        system_content = prompts.NO_CONTEXT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_content}]
    # 2. 追加最近 N 轮对话历史（W4 才有真实历史，现在传空列表即可）
    messages.extend(history)
    # 3. 用户当前问题放最后
    messages.append({"role": "user", "content": question})

    # 4. stream=True：不等全部生成完，模型每出一个 token 就推给我们一个
    stream = _client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.3,  # 知识库场景调低：让模型"守规矩"，减少自由发挥
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:  # 首尾帧的 content 可能是 None，过滤掉
            yield delta


if __name__ == "__main__":
    # W2 前的"迷你 RAG"全链路体验：
    #   python -m app.rag.generator "你的问题"
    # 流程：问题 -> 向量检索 top5 -> 组装 prompt -> DeepSeek 流式回答
    from app.rag.embedder import embed_query
    from app.rag.vector_store import search

    if len(sys.argv) < 2:
        print('用法: python -m app.rag.generator "你的问题"')
        sys.exit(1)

    question = sys.argv[1]
    hits = search(embed_query(question), top_k=5)

    print(f"问题: {question}")
    print(f"检索到 {len(hits)} 条参考资料 (top1 相似度: {hits[0]['similarity'] if hits else '无'})")
    print("=" * 60)
    for h in hits:
        print(f"  [来源] {h['filename']} 第{h['page']}页  相似度={h['similarity']}")
    print("=" * 60)
    print("回答: ", end="", flush=True)

    for token in stream_generate(question, hits, history=[]):
        print(token, end="", flush=True)  # 逐 token 打印 = 打字机效果
    print()
