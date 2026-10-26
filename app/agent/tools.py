"""
Agent 工具层：把"知识库检索"和"知识库统计"封装成模型可自主调用的工具。

面试必讲的两个点：
1. 工具 = 「一段 JSON Schema 描述」+「一个普通 Python 函数」。
   模型本身**不会执行任何代码**，它只是按 Schema 输出
   "我要调用哪个工具、参数是什么"这段结构化文本；
   真正执行的是我们这边的 Python 函数，执行结果再以 role=tool 消息回填给模型。
2. 检索工具内部**完整复用现有 RAG 链路**（混合召回 -> 相似度过滤 -> Rerank 精排），
   不另写一套检索逻辑，保证 Agent 模式与普通模式的答案质量一致。
"""
import json

from app.config import settings
from app.core import prompts
from app.rag import reranker, retriever
from app.services import document_repository


# ---------- 工具的 JSON Schema（OpenAI tools 参数格式，随请求发给模型） ----------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            # description 是模型判断"该不该用我"的唯一依据，要写清使用场景
            "description": (
                "检索企业知识库的正文内容，返回相关文档片段。"
                "当用户的问题需要知识库里文档的具体内容才能回答时，必须调用它，"
                "包括但不限于：某份文档讲了什么、某个技术概念是什么、某个操作怎么做、"
                "某个流程/制度/参数是什么、某段代码或命令怎么写。"
                "注意：即使问题看起来是通用常识，只要答案可能来自知识库里的文档，也应先检索。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索用的问题或关键词，必须独立完整，不要包含「它」「这个」之类的指代",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_knowledge_base_stats",
            "description": (
                "获取知识库的元信息：文档总数、切片总数、文件清单（含每个文件的切片数）。"
                "只在用户只想知道『知识库里有多少东西、装了哪些文件』这类纯数量/清单问题、"
                "而完全不需要文档正文内容时使用。"
                "如果用户想知道某份文档里讲了什么，必须用 search_knowledge_base 而不是本工具。"
            ),
            # 无参数工具也要给 parameters，且 properties 为空对象
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def search_knowledge_base(query: str) -> dict:
    """检索工具实现。

    返回三块内容，分别给三个"消费者"：
      context -> 给模型看（带编号的参考资料文本）
      chunks  -> 给前端引用面板用（结构化片段）
      summary -> 给前端工具调用过程展示用（一句话）
    """
    # 以下检索策略与 chat_service.stream_answer 完全一致：20 条候选 -> 阈值过滤 -> 精排
    hits = retriever.retrieve(query, final_top_k=20)

    # 向量命中要求相似度达标；BM25 命中代表真实关键词重叠，直接保留
    hits = [
        h for h in hits
        if (h.get("similarity") is not None and h["similarity"] >= settings.SIMILARITY_THRESHOLD)
        or "bm25" in h.get("sources", [])
    ]
    if hits:  # 空候选跳过精排，省一次 API 调用
        hits = reranker.rerank(query, hits)

    if not hits:
        return {
            "context": "知识库中没有检索到与该问题相关的内容。",
            "chunks": [],
            "summary": f"检索「{query}」：未命中相关片段",
        }
    return {
        "context": prompts.build_context(hits),
        "chunks": hits,
        "summary": f"检索「{query}」：命中 {len(hits)} 条片段",
    }


def get_knowledge_base_stats() -> dict:
    """统计工具实现：只读 MySQL 文档元数据，不碰向量库，毫秒级返回。"""
    docs = document_repository.list_documents()
    total_chunks = sum(d.get("chunk_count") or 0 for d in docs)
    files = [
        {"filename": d["filename"], "chunks": d["chunk_count"], "status": d["status"]}
        for d in docs
    ]
    return {
        "document_count": len(docs),
        "chunk_count": total_chunks,
        # 文件很多时只给前 20 个，避免文件清单撑爆模型上下文
        "files": files[:20],
        "truncated": len(files) > 20,
        "summary": f"知识库共 {len(docs)} 份文档、{total_chunks} 个切片",
    }


def run_tool(name: str, arguments: dict) -> dict:
    """统一的工具分发入口：Agent 决策层只跟这个函数打交道。

    返回 {for_model, chunks, summary}：
      for_model -> 以 role=tool 消息回填给模型的字符串
      chunks    -> 检索片段（仅检索工具非空）
      summary   -> 前端"工具调用过程"展示文案

    工具异常在这里兜底：把错误原因作为工具结果告诉模型，
    让它能换个 query 重试或如实说明，而不是让整个问答请求 500。
    """
    try:
        if name == "search_knowledge_base":
            query = (arguments or {}).get("query", "").strip()
            if not query:
                return {"for_model": "检索失败：query 参数为空", "chunks": [], "summary": "检索失败：query 为空"}
            r = search_knowledge_base(query)
            return {"for_model": r["context"], "chunks": r["chunks"], "summary": r["summary"]}

        if name == "get_knowledge_base_stats":
            r = get_knowledge_base_stats()
            payload = {k: v for k, v in r.items() if k != "summary"}
            return {
                "for_model": json.dumps(payload, ensure_ascii=False),
                "chunks": [],
                "summary": r["summary"],
            }

        # 模型偶尔会编出不存在的工具名，明确回绝而不是抛异常
        return {"for_model": f"未知工具：{name}，请只使用已提供的工具", "chunks": [], "summary": f"未知工具 {name}"}
    except Exception as e:
        return {
            "for_model": f"工具 {name} 执行失败：{type(e).__name__}: {e}",
            "chunks": [],
            "summary": f"{name} 执行失败：{type(e).__name__}",
        }
