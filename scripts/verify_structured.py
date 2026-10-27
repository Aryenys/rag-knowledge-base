"""
【任务5】结构化输出的可复现验证。

分两组：
  - 第 1~2 组：纯解析逻辑，**不调 LLM**，秒级完成、零成本，随时可跑
  - 第 3~5 组：真实调用 DeepSeek（约 4 次），验证端到端真的能拿到结构化字段

用法：
  python -m scripts.verify_structured           # 全跑（含 4 次 LLM 调用，约 1 分钟）
  python -m scripts.verify_structured --quick   # 只跑不花钱的那两组
"""
import json
import sys

from app.models.schemas import StructuredAnswer
from app.services import structured_service

# 解析用例：(名称, 模型返回的原始文本, 提供的资料条数, 期望的 citations 或 None 表示应解析失败)
PARSE_CASES = [
    ("标准输出", '{"answer":"A","citations":[1,2],"confidence":"high","answered":true}', 5, [1, 2]),
    ("代码块围栏", '```json\n{"answer":"A","citations":[3],"confidence":"low","answered":true}\n```', 5, [3]),
    ("前后带客套话", '好的：{"answer":"A","citations":[1],"confidence":"medium","answered":true} 谢谢', 5, [1]),
    ("漏了字段", '{"answer":"A"}', 5, []),
    ("编号写成字符串", '{"answer":"A","citations":"2","confidence":"high","answered":true}', 5, [2]),
    ("枚举值非法", '{"answer":"A","citations":[],"confidence":"VERY","answered":true}', 5, []),
    ("引用编号越界", '{"answer":"A","citations":[1,9,"x"],"confidence":"high","answered":true}', 5, [1]),
    ("拒答", '{"answer":"无法回答","citations":[],"confidence":"low","answered":false,"refuse_reason":"缺财务文档"}', 0, []),
    ("不是 JSON", "这不是 JSON", 5, None),
    ("顶层是数组", "[1,2,3]", 5, None),
    ("空内容", "", 5, None),
]

_results: list[bool] = []


def _record(ok: bool) -> bool:
    _results.append(ok)
    return ok


def check_parse() -> bool:
    """第 1 组：解析与修复逻辑（不花钱，最关键的一组）。

    为什么值得单独测：模型输出不稳定是常态——加围栏、带客套话、编号写成字符串、
    漏字段、甚至编出不存在的引用编号。这些都必须被处理掉，而不是抛异常打挂请求。
    """
    print("=" * 60)
    print("第 1 组：解析与修复逻辑（11 个用例，不调 LLM）")
    print("=" * 60)

    all_ok = True
    for name, raw, n_sources, expect in PARSE_CASES:
        data, warnings = structured_service.parse_structured(raw, n_sources)

        if expect is None:
            ok = data is None
            detail = f"解析失败（符合预期）：{warnings[0][:40]}"
        else:
            ok = data is not None and data["citations"] == expect
            detail = f"citations={data['citations'] if data else None}（期望 {expect}）"
            if data and warnings:
                detail += f"  警告：{warnings[0][:44]}"

        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:12s} {detail}")

    print(f"\n  结果：{'PASS' if all_ok else 'FAIL'}")
    return _record(all_ok)


def check_schema_consistency() -> bool:
    """第 2 组：给模型看的 schema 与 StructuredAnswer 必须同源。

    手写 schema 最大的风险是"模型改了、提示词没改"，这种失效是静默的：
    JSON 照样合法，只是字段永远拿不到。所以这里断言两者字段集合一致。
    """
    print("\n" + "=" * 60)
    print("第 2 组：prompt 里的 schema 与 Pydantic 模型一致（不调 LLM）")
    print("=" * 60)

    schema_text = structured_service.json_schema_for_prompt()
    schema = json.loads(schema_text)
    model_fields = set(StructuredAnswer.model_fields.keys())
    schema_fields = set(schema.get("properties", {}).keys())

    ok = model_fields == schema_fields
    print(f"  模型字段：{sorted(model_fields)}")
    print(f"  schema 字段：{sorted(schema_fields)}")
    print(f"  一致：{ok}")

    # 顶层 description 是类 docstring，会白白吃掉几百 token，必须去掉
    no_desc = "description" not in schema
    print(f"  已剔除顶层 description（省 token）：{no_desc}")

    # DeepSeek 的 json_object 模式只保证"是合法 JSON"，schema 靠 prompt 传达，
    # 所以提示词里必须真的带上 schema 文本
    prompt = structured_service.build_system_prompt([])
    has_schema = "JSON Schema" in prompt and "citations" in prompt
    print(f"  system prompt 里确实带了 schema：{has_schema}")

    result = ok and no_desc and has_schema
    print(f"\n  结果：{'PASS' if result else 'FAIL'}")
    return _record(result)


def check_answered_case() -> bool:
    """第 3 组：有资料时真实生成， citations 必须落在有效范围内。"""
    print("\n" + "=" * 60)
    print("第 3 组：真实生成（有资料）→ citations 不得越界")
    print("=" * 60)

    from app.services import retrieval_service

    question = "RAG 相比纯生成模型有什么优势？用两句话回答"
    hits, _ = retrieval_service.search(question, final_top_k=20)
    print(f"  检索命中 {len(hits)} 条")

    r = structured_service.generate_structured(question, hits, [])
    d = r["data"]
    in_range = all(1 <= c <= len(hits) for c in d["citations"])

    print(f"  ok={r['ok']}  repaired={r['repaired']}")
    print(f"  citations={d['citations']}（有效范围 1..{len(hits)}）  越界检查：{in_range}")
    print(f"  confidence={d['confidence']}  answered={d['answered']}")
    print(f"  回答前 60 字：{d['answer'][:60]}...")
    if r["warnings"]:
        print(f"  警告：{r['warnings']}")

    ok = r["ok"] and in_range and d["answered"] is True
    print(f"\n  结果：{'PASS' if ok else 'FAIL'}")
    return _record(ok)


def check_refuse_case() -> bool:
    """第 4 组：没有资料时必须能识别成"拒答"，而不是编一段答案。

    这是结构化输出最实用的价值之一：拒答从"文本上和正常回答混在一起"
    变成了一个可统计的字段。
    """
    print("\n" + "=" * 60)
    print("第 4 组：真实生成（无资料）→ 必须 answered=false")
    print("=" * 60)

    r = structured_service.generate_structured(
        "我们公司 2026 年的差旅报销标准是多少？", [], []
    )
    d = r["data"]
    print(f"  ok={r['ok']}")
    print(f"  answered={d['answered']}（期望 False）")
    print(f"  refuse_reason={d['refuse_reason']}")
    print(f"  citations={d['citations']}（期望 []，没给资料就不可能有引用）")
    print(f"  回答：{d['answer'][:60]}...")

    ok = d["answered"] is False and d["citations"] == []
    print(f"\n  结果：{'PASS' if ok else 'FAIL'}")
    return _record(ok)


def check_sse_frames() -> bool:
    """第 5 组：SSE 帧序列。structured=true 推 structured 帧，false 仍推 token 帧。

    后者是回归断言：加功能不能把默认行为改坏。
    """
    print("\n" + "=" * 60)
    print("第 5 组：SSE 帧序列（结构化 vs 默认，各一次真实调用）")
    print("=" * 60)

    from app.services.chat_service import stream_answer

    results = []
    for flag, expect_type, absent_type in (
        (True, "structured", "token"),
        (False, "token", "structured"),
    ):
        frames = []
        for f in stream_answer("RAG 是什么？用一句话回答", None, structured=flag):
            frames.append(json.loads(f.replace("data: ", "").strip()))
        types = [p["type"] for p in frames]
        ok = expect_type in types and absent_type not in types
        results.append(ok)
        print(f"  structured={flag!s:5s} 帧序列={types}")
        print(f"    含 {expect_type} 帧且不含 {absent_type} 帧：{ok}")

    ok = all(results)
    print(f"\n  结果：{'PASS' if ok else 'FAIL'}")
    return _record(ok)


def main() -> None:
    quick = "--quick" in sys.argv

    # 验证过程不应往真实数据库里塞测试会话
    import app.services.session_service as session_service
    session_service.save_message = lambda *a, **k: {}
    session_service.create_session = lambda: {"id": -1}

    print("结构化输出验证" + ("（--quick：只跑不花钱的两组）" if quick else "（全量，含真实 LLM 调用）"))

    check_parse()
    check_schema_consistency()
    if not quick:
        check_answered_case()
        check_refuse_case()
        check_sse_frames()

    passed = sum(1 for x in _results if x)
    print("\n" + "=" * 60)
    print(f"汇总：{passed}/{len(_results)} 组通过")
    print("=" * 60)
    if passed != len(_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
