"""
所有 Prompt 模板集中管理。
面试高频问题："你的 prompt 是怎么设计的？" —— 答案就在这个文件里。
设计要点：① 限定只根据资料回答（抑制幻觉）② 强制标注引用编号（可溯源）
         ③ 明确"不知道"的兜底话术 ④ 输出格式约束（分点、简洁）
"""

# ---------- 主问答 Prompt ----------
RAG_SYSTEM_PROMPT = """你是企业知识库助手。请仅根据以下【参考资料】回答用户问题。

要求：
1. 回答中引用资料时，在句末标注来源编号，如 [1]、[2]
2. 如果【参考资料】不足以回答问题，明确回复"根据现有资料无法回答该问题"，不要编造内容
3. 回答简洁、有条理，必要时分点列出

【参考资料】
{context}"""

# 单条参考资料的格式（context 由多条拼接而成）
CONTEXT_ITEM_TEMPLATE = "[{index}] (来源: {filename}{page_info})\n{content}"

# ---------- 查询改写 Prompt（解决多轮对话中的指代问题："它"、"这个"） ----------
QUERY_REWRITE_PROMPT = """根据以下对话历史，把用户的最新问题改写成一个独立、完整、无需上下文即可理解的问题。
只输出改写后的问题本身，不要输出任何其他内容。如果问题本身已经完整独立，原样输出即可。

【对话历史】
{history}

【最新问题】
{question}

改写后的问题："""


# ---------- 无检索结果兜底 Prompt（相似度过滤后 chunks 为空时用） ----------
NO_CONTEXT_SYSTEM_PROMPT = """你是企业知识库助手。当前知识库中没有检索到与用户问题相关的内容。
如果用户是在闲聊（打招呼、问候等），请友好、简短地回应。
如果用户在提问知识性问题，请明确告知"知识库中未找到相关内容"，并建议用户换一种问法或确认相关文档是否已上传。
不要编造知识库中不存在的内容。"""


def build_context(chunks: list[dict]) -> str:
    """把 rerank 后的 top-K chunk 拼成带编号的参考资料文本。
    chunks 元素格式: {"content": str, "filename": str, "page": int|None}
    """
    items = []
    for i, c in enumerate(chunks, start=1):
        page_info = f" 第{c['page']}页" if c.get("page") else ""
        items.append(
            CONTEXT_ITEM_TEMPLATE.format(
                index=i, filename=c["filename"], page_info=page_info, content=c["content"]
            )
        )
    return "\n\n".join(items)


def format_history(messages: list[dict]) -> str:
    """把最近 N 轮消息格式化为对话历史文本。
    messages 元素格式: {"role": "user"/"assistant", "content": str}
    """
    role_map = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{role_map.get(m['role'], m['role'])}: {m['content']}" for m in messages)
