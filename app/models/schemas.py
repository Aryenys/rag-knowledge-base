"""
Pydantic 请求/响应模型（API 层的"数据契约"，前后端对齐用）。
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------- 文档 ----------
class DocumentInfo(BaseModel):
    id: int
    filename: str
    chunk_count: int
    status: str            # processing / ready / failed
    created_at: datetime


# ---------- 问答 ----------
class ChatRequest(BaseModel):
    question: str
    session_id: int | None = None   # 为空表示新会话
    # Agent 模式开关：默认 False = 走原有"检索->生成"链路，行为完全不变；
    # 传 true 才进入 Agent 决策层，由模型自行判断是否调用工具
    agent: bool = False
    # 结构化输出开关：默认 False = 逐 token 流式吐字（体验优先）；
    # 传 true 时模型按 StructuredAnswer 的 schema 返回 JSON（可机读优先）。
    # 两者互斥：JSON 必须完整才合法，无法边生成边吐字。
    # ⚠️ agent=true 时本参数不生效（工具调用本身就是结构化协议，无需再套一层）
    structured: bool = False


# 结构化回答契约：模型按此 schema 输出，下游拿到的是可判定字段而非一整段文本。
#
# 为什么要它（普通模式做不到的事）：
#   - citations：模型说"根据资料"，但到底用了哪几条？只能从正文正则抠 [1][2]，
#     漏标/标错无法自动发现。字段化后才能校验——引用越界 = 幻觉，机器可直接拦下
#   - answered：资料不足时模型"礼貌拒答"，文本上和正常回答混在一起，
#     没法统计拒答率。有了这个字段，拒答才变成一个可观测指标
#   - confidence：模型对自己答案的把握程度，可用于决定是否转人工
#
# ⚠️ 注意：Pydantic 会把类 docstring 当作 schema 的顶层 description 塞进提示词，
#    所以这里的长说明写成注释而不是 docstring —— 给模型看的说明写在 Field 里。
class StructuredAnswer(BaseModel):
    """结构化回答（structured=true 时模型的输出契约）"""

    answer: str = Field(description="给用户的回答正文，Markdown 可用")
    citations: list[int] = Field(
        default_factory=list,
        description="本次回答实际引用到的资料编号，取值 1..N（N=提供的资料条数），未引用则为空数组",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="对答案的把握程度：high=资料明确支持；medium=资料部分支持；low=资料不足或推断成分大"
    )
    answered: bool = Field(
        description="是否真正回答了问题。资料中没有依据时填 false，此时 answer 为拒答说明"
    )
    refuse_reason: str | None = Field(
        default=None, description="answered=false 时说明缺什么资料；已回答则为 null"
    )


class SourceChunk(BaseModel):
    """一条引用来源，前端折叠面板展示"""
    index: int           # 编号 [1] [2] ...
    filename: str
    page: int | None = None
    content: str         # 原文片段
    score: float | None = None   # rerank 分数


class SessionInfo(BaseModel):
    id: int
    title: str
    created_at: datetime


class MessageInfo(BaseModel):
    id: int
    role: str            # user / assistant
    content: str
    sources: list[SourceChunk] | None = None
    created_at: datetime
