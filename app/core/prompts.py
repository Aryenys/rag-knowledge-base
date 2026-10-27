"""
所有 Prompt 模板集中管理。
面试高频问题："你的 prompt 是怎么设计的？" —— 答案就在这个文件里。
设计要点：① 限定只根据资料回答（抑制幻觉）② 强制标注引用编号（可溯源）
         ③ 明确"不知道"的兜底话术 ④ 输出格式约束（分点、简洁）

【任务4 改造】
模板内容已迁到 prompt_versions/*.json，这里只做两件事：
  1. 把默认版本的模板暴露成同名常量——调用方（含 app/rag/generator.py）一行都不用改
  2. 保留 build_context / format_history 两个拼装函数

这样做的收益：切 PROMPT_VERSION 环境变量就能换整套 prompt，
而 app/rag/ 下的生成代码完全没动，默认版本（v1）的模板与改造前逐字一致。
"""
from app.core import prompt_registry

# ---------- 当前生效版本 ----------
# 注意：这里在模块加载时求值一次，所以改 PROMPT_VERSION 需要重启服务。
# 这是刻意的取舍——prompt 属于发布配置，不该在运行时被随便改，
# 否则线上会出现"同一个问题前后答案风格不一致"的诡异现象。
CURRENT_PROMPT_VERSION = prompt_registry.current_version()

# ---------- 主问答 Prompt ----------
RAG_SYSTEM_PROMPT = prompt_registry.get_template("rag_system")

# 单条参考资料的格式（context 由多条拼接而成）
CONTEXT_ITEM_TEMPLATE = prompt_registry.get_template("context_item")

# ---------- 查询改写 Prompt（解决多轮对话中的指代问题："它"、"这个"） ----------
QUERY_REWRITE_PROMPT = prompt_registry.get_template("query_rewrite")

# ---------- 无检索结果兜底 Prompt（相似度过滤后 chunks 为空时用） ----------
NO_CONTEXT_SYSTEM_PROMPT = prompt_registry.get_template("no_context")

# ---------- Agent 决策层 Prompt（任务1） ----------
# 与普通 RAG prompt 的区别：这里不再预先塞参考资料，
# 而是告诉模型"你有哪些工具、什么时候该用哪个"，把检索的决定权交给模型
AGENT_SYSTEM_PROMPT = prompt_registry.get_template("agent_system")


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
