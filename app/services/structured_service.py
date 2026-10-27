"""
【任务5】结构化输出：让模型按固定 JSON schema 回答，而不是吐一整段自由文本。

## 解决什么问题

普通模式下答案是一段 Markdown 文本，下游（前端、评测、运营）拿不到机器可读的信息：

  1. 引用了哪几条资料？只能从正文里正则抠 [1][2]；模型漏标、标错编号，机器发现不了
  2. 资料不足时模型会"礼貌拒答"，但文本上和正常回答混在一起，拒答率根本没法统计
  3. 模型对答案有多大把握？无从得知，也就无法决定"要不要转人工"

结构化模式把这三点变成可判定的字段，于是：
  - 引用编号越界（比如只给了 5 条资料，模型编了个 [9]）→ 机器能直接判定为幻觉并剔除
  - answered=false → 拒答变成一个可观测指标
  - confidence=low → 可作为"建议人工复核"的触发条件

## 与流式的取舍（本设计最关键的一点）

JSON 必须完整才合法，所以结构化模式下**无法逐 token 吐字**——这是协议层面的互斥，
不是实现偷懒。因此做成按请求开关：
  - structured=false（默认）：保持流式打字机体验
  - structured=true：一次性返回，换来可机读的字段
两者由调用方按场景选，不强制二选一。前端/接口层都不用为大改。

## 为什么格式指令不放在 prompt_versions/ 里

版本文件的占位符走白名单校验（prompt_registry._ALLOWED_VARS），
而 JSON schema 本身就是满屏花括号，会被当成未知占位符直接拒绝加载。
更本质的原因是：prompt 版本管的是"回答风格"，输出格式是"接口契约"，
两者正交——换 prompt 版本不应该影响结构化字段，所以格式契约放在代码里，
跟模型定义 StructuredAnswer 同源生成，不会漂移。
"""
import json
import logging
import re

from openai import OpenAI
from pydantic import ValidationError

from app.config import settings
from app.core import prompt_registry, prompts
from app.models.schemas import StructuredAnswer

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)

# 代码块围栏：模型经常把 JSON 包在 ```json ... ``` 里
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# 格式指令。⚠️ 用 .replace 而不是 .format：schema 本身含大量花括号，
# format 会把 {"type": "object"} 当成占位符直接 KeyError
_FORMAT_INSTRUCTION = """
【输出格式要求（优先级高于上面所有要求）】
你必须且只能输出一个 JSON 对象，不要输出任何解释、前言或多余文字。
该 JSON 必须严格符合以下 JSON Schema：

{schema}

补充约束：
- citations 只能填 1 到 {max_citation} 之间的整数（对应【参考资料】的编号），
  不要重复；没有用到任何资料就填 []。绝不允许填写不存在的编号。
- 资料不足以回答时：answered 填 false，answer 写拒答说明，refuse_reason 说明缺什么资料
- 已回答时：answered 填 true，refuse_reason 填 null
- 语言与用户提问保持一致
"""

_REPAIR_HINT = (
    "你上一次的输出不是合法 JSON，错误是：{error}\n"
    "请严格按上面的 schema 重新输出，只输出 JSON 对象本身，不要加任何其它文字。"
)


def json_schema_for_prompt() -> str:
    """从 StructuredAnswer 生成给模型看的 JSON Schema。

    从模型生成而不是手写，好处是"契约"只有一份：改字段时提示词自动跟着变，
    不会出现"schema 改了、提示词忘了改"导致的默默失效。
    """
    schema = StructuredAnswer.model_json_schema()
    # title 和顶层 description 对模型没用，去掉省 token（字段级 description 要保留，
    # 那是告诉模型每个字段该怎么填的唯一说明）
    schema.pop("title", None)
    schema.pop("description", None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_system_prompt(chunks: list[dict]) -> str:
    """组装结构化模式的 system prompt = 原有 RAG prompt + 格式指令。

    复用同一个 prompt 版本（含版本继承），只在末尾追加格式约束，
    所以切 PROMPT_VERSION 时结构化模式照样跟着变。
    """
    if chunks:
        base = prompt_registry.render("rag_system", context=prompts.build_context(chunks))
    else:
        base = prompt_registry.render("no_context")
    return base + _FORMAT_INSTRUCTION.replace(
        "{schema}", json_schema_for_prompt()
    ).replace("{max_citation}", str(len(chunks)))


def parse_structured(raw: str, n_sources: int) -> tuple[dict | None, list[str]]:
    """把模型返回的文本解析成 StructuredAnswer 字典。

    设计原则：**宽松进、严格出**。模型输出不稳定是常态（加围栏、带客套话、
    编号写成字符串、漏字段），这些都尽量修；修不了的才降级，绝不抛异常。

    返回 (payload_dict | None, 警告列表)。None 表示彻底解析失败。
    """
    warnings: list[str] = []
    text = (raw or "").strip()
    if not text:
        return None, ["模型返回空内容"]

    # 1) 去掉 ```json ... ``` 围栏
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
        warnings.append("返回内容带代码块围栏，已剥离")

    # 2) 模型偶尔在 JSON 前后加客套话，截取最外层 {...}
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
            warnings.append("返回内容含非 JSON 文本，已截取最外层对象")

    # 3) 解析
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"不是合法 JSON：{e}"]

    if not isinstance(payload, dict):
        return None, [f"顶层不是对象，实际是 {type(payload).__name__}"]

    # 4) 漏字段补齐：模型偶尔只回 {"answer": "..."}，直接判失败太浪费一次调用
    for key, default in (("citations", []), ("confidence", "medium"), ("answered", True)):
        payload.setdefault(key, default)

    # 5) citations 宽松化 + 越界剔除（这一步就是"幻觉引用"的机器检测）
    raw_citations = payload.get("citations")
    if isinstance(raw_citations, (str, int)):   # 写成 "1" 或 1 而不是 [1]
        raw_citations = [raw_citations]
    if not isinstance(raw_citations, list):
        raw_citations = []

    kept: list[int] = []
    dropped: list[str] = []
    for item in raw_citations:
        try:
            n = int(item)
        except (TypeError, ValueError):
            dropped.append(repr(item))
            continue
        if 1 <= n <= n_sources:
            if n not in kept:
                kept.append(n)
        else:
            dropped.append(str(n))
    if dropped:
        warnings.append(
            f"剔除了越界/非法引用编号 {dropped}（有效范围 1..{n_sources}）——"
            f"编号超出提供的资料条数即视为幻觉引用"
        )
    payload["citations"] = sorted(kept)

    # 6) 枚举归一化
    conf = str(payload.get("confidence", "medium")).strip().lower()
    if conf not in ("high", "medium", "low"):
        warnings.append(f"confidence 取值非法（{conf!r}），回退 medium")
        conf = "medium"
    payload["confidence"] = conf
    payload["answered"] = bool(payload.get("answered", True))
    if not payload["answered"] and not payload.get("refuse_reason"):
        payload["refuse_reason"] = "（模型未说明原因）"

    # 7) 严格校验
    try:
        model = StructuredAnswer(**payload)
    except ValidationError as e:
        return None, [f"字段校验失败：{_brief_error(e)}"]

    return model.model_dump(), warnings


def _brief_error(e: ValidationError) -> str:
    """Pydantic 的报错很长，取第一条的关键信息即可。"""
    try:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", ()))
        return f"{loc}: {first.get('msg', '')}"
    except Exception:                                    # pragma: no cover
        return str(e)[:120]


def fallback_payload(raw: str, warnings: list[str]) -> dict:
    """解析彻底失败时的兜底：把原文当答案返回，并如实标注 parse 失败。

    宁可给用户一段"没解析成功"的原始文本，也不要让请求 500。
    """
    return {
        "answer": raw.strip() or "（模型未返回内容）",
        "citations": [],
        "confidence": "low",
        "answered": True,
        "refuse_reason": None,
        "parse_ok": False,
        "warnings": warnings,
    }


def generate_structured(
    question: str,
    chunks: list[dict],
    history: list[dict],
    allow_repair: bool = True,
) -> dict:
    """调 LLM 生成结构化回答。

    返回统一结构，调用方不用猜：
      {
        "ok": bool,          # 是否拿到合法结构化结果
        "data": dict,        # StructuredAnswer 字段；解析失败时是兜底 payload
        "raw": str,          # 模型原始输出（排查用）
        "warnings": [str],   # 修复/降级过程中发生的事
        "error": str | None, # 失败原因
        "repaired": bool,    # 是否靠"重试一次"才拿到合法 JSON
      }
    """
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(chunks)}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    raw = ""
    try:
        resp = _client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=0.3,
            # DeepSeek 的 JSON 模式：只保证输出是合法 JSON，不保证符合 schema，
            # 所以 schema 必须写在 prompt 里，且回来后还要自己校验（见 parse_structured）
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("[structured] LLM 调用失败：%s: %s", type(e).__name__, e)
        return {
            "ok": False, "data": fallback_payload("", [f"LLM 调用失败：{type(e).__name__}"]),
            "raw": "", "warnings": [f"LLM 调用失败：{type(e).__name__}"],
            "error": str(e), "repaired": False,
        }

    data, warnings = parse_structured(raw, len(chunks))
    repaired = False

    # 解析失败 → 带上错误原因让模型自己改一次。比直接降级更有用，只多花一次调用。
    if data is None and allow_repair:
        repaired = True
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": _REPAIR_HINT.format(error=warnings[-1])})
        try:
            resp = _client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw2 = resp.choices[0].message.content or ""
            data2, warnings2 = parse_structured(raw2, len(chunks))
            if data2 is not None:
                raw, data, warnings = raw2, data2, warnings + ["首次输出不合法，重试一次后成功"] + warnings2
            else:
                warnings = warnings + ["重试后仍不合法"] + warnings2
        except Exception as e:
            logger.warning("[structured] 修复重试失败：%s: %s", type(e).__name__, e)
            warnings.append(f"修复重试失败：{type(e).__name__}")

    if data is None:
        return {
            "ok": False, "data": fallback_payload(raw, warnings),
            "raw": raw, "warnings": warnings, "error": "解析失败，已降级为纯文本",
            "repaired": repaired,
        }

    data["parse_ok"] = True
    data["warnings"] = warnings
    return {"ok": True, "data": data, "raw": raw, "warnings": warnings, "error": None, "repaired": repaired}
