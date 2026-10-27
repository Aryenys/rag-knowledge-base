"""
【任务1】Agent 工具调用验证脚本。

验证"Agent 模式真的有用"，要回答三个问题，光看代码不算数：

  Q1 模型会不会选工具？—— 该检索的检索、该统计的统计、闲聊就别调
  Q2 会不会把检索质量搞差？—— Agent 检索工具和主链路必须返回一样的结果
  Q3 会不会死循环？—— 轮数闸门是不是真的卡得住

对应下面的 4 组检查。前 3 组要调 DeepSeek（花一点点钱），第 4 组只调本地检索。

用法：
  python -m scripts.verify_agent            # 全跑（约 2-3 分钟，含 4 次 LLM 调用）
  python -m scripts.verify_agent --quick    # 只跑第 2 组（不调 LLM，不花钱）

判定说明：
  工具选择用"集合相等"判定。模型偶尔会多调一次工具，这不算错，
  但如果该调的没调（比如问知识库文档数却不查库，直接编个数），就是 FAIL——
  那意味着 Agent 形同虚设。
"""
import json
import sys
import time

# ---------------------------------------------------------------------------
# 测试用例：每条都写清楚"期望什么、为什么"
# ---------------------------------------------------------------------------
CASES = [
    {
        "name": "统计类问题 → 应该调统计工具",
        "question": "知识库里一共有多少份文档？",
        "expect": {"get_knowledge_base_stats"},
        "why": "问的是元数据库里的数字，检索文档内容是查不到的，必须走 MySQL",
    },
    {
        "name": "正文类问题 → 应该调检索工具",
        "question": "RAG 相比纯生成模型有什么优势？",
        "expect": {"search_knowledge_base"},
        "why": "问的是文档里的内容，必须走向量+BM25 检索",
    },
    {
        "name": "闲聊/通用知识 → 应该直接回答，不调工具",
        "question": "用 Python 写一个快速排序函数，不要解释。",
        "expect": set(),
        "why": "知识库里没有快排，调检索纯属浪费。这正是 Agent 相对「无脑 RAG」的价值："
               "不该检索的问题省掉 1 次 embedding + 1 次 rerank",
    },
    {
        "name": "复合问题 → 应该拆成两次工具调用",
        "question": "知识库里一共有多少份文档？BGE-M3 支持的最大输入长度是多少？",
        "expect": {"get_knowledge_base_stats", "search_knowledge_base"},
        "why": "一句话里有两个子问题，且分属两类工具。能拆开说明模型真的在做决策，"
               "不是把整句丢进检索器",
    },
]


def run_agent(question: str) -> tuple[list[tuple[str, int]], str, float]:
    """跑一次 Agent 问答。

    返回（[(工具名, 第几轮)]、完整回答、耗时毫秒）。

    注意区分「轮」和「次」：模型可以在一轮里并行要求多个工具（parallel tool calling），
    这时 tool_round 只加 1，但工具被调用了多次。防死循环闸门限的是「轮」不是「次」，
    所以只统计调用次数会误判。
    """
    from app.agent.agent_service import stream_agent_answer

    calls: list[tuple[str, int]] = []
    answer_parts: list[str] = []
    t0 = time.perf_counter()

    for frame in stream_agent_answer(question, None):
        payload = json.loads(frame.replace("data: ", "").strip())
        ftype = payload.get("type")
        if ftype == "tool":
            calls.append((payload.get("name", "?"), payload.get("round", 0)))
        elif ftype == "token":
            answer_parts.append(payload.get("content", ""))

    elapsed = (time.perf_counter() - t0) * 1000
    return calls, "".join(answer_parts), elapsed


def check_tool_selection() -> int:
    """第 1 组：工具选择对不对。"""
    print("=" * 60)
    print("第 1 组：工具选择（这是 Agent 的核心能力）")
    print("=" * 60)

    passed = 0
    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {case['name']}")
        print(f"  问题：{case['question']}")
        print(f"  为什么这样期望：{case['why']}")

        calls, answer, elapsed = run_agent(case["question"])
        actual = {name for name, _ in calls}
        rounds = sorted({r for _, r in calls})
        ok = actual == case["expect"]

        print(f"  期望调用：{sorted(case['expect']) or '（不调工具）'}")
        print(f"  实际调用：{[n for n, _ in calls] or '（不调工具）'}")
        if rounds:
            print(f"  发生在第 {rounds} 轮（一轮里可以并行调多个工具）")
        print(f"  耗时：{elapsed:.0f}ms   回答长度：{len(answer)} 字")
        print(f"  结果：{'PASS' if ok else 'FAIL'}")

        if not case["expect"] and not calls:
            print("  → 未调工具，省掉 1 次 embedding + 1 次 rerank 调用")

        passed += 1 if ok else 0

    return passed


def check_no_quality_regression() -> bool:
    """第 2 组：Agent 检索工具 vs 主链路检索，结果必须一致（等价）。

    判定标准为什么不要求「逐条顺序完全一致」：
      rerank API 返回的分数存在 ±0.001 级抖动，排名靠后且分数接近的条目
      顺序会互相翻转。实测主链路自己连跑两次，第 5 条就换过一次。
      所以拿「顺序完全一致」当标准，主链路自己都过不了，这个标准没有意义。

    有意义的两个判据：
      1. 首条一致（首条是模型最依赖的那条，必须稳）
      2. 内容重合度 ≥ 4/5（允许边界条目翻转一格）
    """
    print("\n" + "=" * 60)
    print("第 2 组：检索质量不降（Agent 路径 vs 主链路路径）")
    print("=" * 60)

    from app.agent.tools import run_tool
    from app.config import settings
    from app.services import retrieval_service

    query = "RAG 相比纯生成模型有什么优势？"

    # 临时关缓存：三条检索都跑真实链路，才能比出"检索逻辑"是否一致
    old = settings.CACHE_ENABLED
    settings.CACHE_ENABLED = False
    try:
        # 对照组：主链路自己连跑两次（用来证明抖动是 rerank 固有的，不是 Agent 引入的）
        main_a, _ = retrieval_service.search(query, final_top_k=20)
        main_b, _ = retrieval_service.search(query, final_top_k=20)
        # 实验组：Agent 的检索工具
        agent_hits = run_tool("search_knowledge_base", {"query": query})["chunks"]
    finally:
        settings.CACHE_ENABLED = old

    def texts(hits):
        return [h["content"] for h in hits]

    def scores(hits):
        return [float(h.get("rerank_score") or 0) for h in hits]

    a, b, c = texts(main_a), texts(main_b), texts(agent_hits)

    print(f"\n  查询：{query}")
    print(f"\n  【对照组】主链路连跑两次，看检索本身稳不稳：")
    print(f"    首次得分：{[round(s, 4) for s in scores(main_a)]}")
    print(f"    二次得分：{[round(s, 4) for s in scores(main_b)]}")
    print(f"    完全一致：{a == b}")
    if a != b:
        print("    → 主链路自己就不完全一致：rerank 分数有微小抖动，"
              "排名靠后、分数接近的条目会翻转")
        print("    → 所以下面不能按「顺序完全一致」判定，否则这一项永远不可能通过")

    overlap = len(set(a) & set(c))
    top1_same = bool(a) and bool(c) and a[0] == c[0]
    same_len = len(a) == len(c)
    ok = top1_same and overlap >= 4 and same_len

    print(f"\n  【实验组】Agent 路径 vs 主链路：")
    print(f"    命中条数：{len(c)} vs {len(a)}（一致：{same_len}）")
    print(f"    首条一致：{top1_same}")
    print(f"    内容重合：{overlap}/{len(a)}")
    print(f"    Agent 得分：{[round(s, 4) for s in scores(agent_hits)]}")

    if ok:
        print("  → 两条路共用 retrieval_service.search，Agent 模式不引入检索质量回归")
        print("  → 残留差异来自 rerank 分数抖动，不是两条路的逻辑差异")
    else:
        print("  → 差异超出抖动范围，需要排查两条路的检索逻辑是否分叉")

    print(f"  结果：{'PASS' if ok else 'FAIL'}")
    return ok


def check_loop_guard() -> bool:
    """第 3 组：防死循环闸门。

    不用真模型验证——模型什么时候停是概率事件，测出来不稳定。
    这里伪造一个「每一次都要调工具」的模型：只要它还拿得到 tools 就永不罢休。
    如果闸门没写对，这个循环会一直转下去（跑几秒就该警觉）。
    工具名用一个不存在的名字，走 run_tool 的兜底分支，不查库不检索，零成本。
    """
    print("\n" + "=" * 60)
    print("第 3 组：防死循环闸门（伪造一个永不罢休的模型）")
    print("=" * 60)

    import app.agent.agent_service as agent_service
    from types import SimpleNamespace

    create_count = {"n": 0}

    def fake_create(**kwargs):
        """给 tools 就永远要调工具；不给 tools 就只能输出文本 → 循环结束。"""
        create_count["n"] += 1

        def gen():
            if "tools" in kwargs:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id=f"call_{create_count['n']}",
                        function=SimpleNamespace(name="__not_a_real_tool__", arguments="{}"),
                    )],
                ))])
            else:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                    content="（闸门已生效，不再提供工具）", tool_calls=None,
                ))])

        return gen()

    old_client = agent_service._client
    agent_service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    try:
        t0 = time.perf_counter()
        calls, _, _ = run_agent("（本条不真调模型）")
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        agent_service._client = old_client

    max_round = max((r for _, r in calls), default=0)
    limit = agent_service.settings.AGENT_MAX_TOOL_ROUNDS
    ok = max_round == limit and create_count["n"] == limit + 1

    print(f"\n  配置上限 AGENT_MAX_TOOL_ROUNDS = {limit}")
    print(f"  实际执行轮数：{max_round}")
    print(f"  模型被调用次数：{create_count['n']}（预期 {limit + 1} = {limit} 轮工具 + 1 次最终回答）")
    print(f"  耗时：{elapsed:.0f}ms（如果这里卡住不动，说明闸门失效了）")
    if ok:
        print("  → 闸门生效：达到上限后不再给模型传 tools，没有工具可调，只能给最终答案")
        print("  → 补充：闸门保证的是「不无限循环」，不保证答得好——"
              "被截断时答案可能不完整，这是可用性与成本的取舍")
    print(f"  结果：{'PASS' if ok else 'FAIL'}")
    return ok


def silence_db():
    """把会话落库的函数换成空操作，避免验证过程往真实数据库里塞测试会话。

    代价：history 恒为空，所以本脚本只验证单轮行为（也够用了）。
    返回 (service, 原函数元组)，用完要 restore。
    """
    import app.agent.agent_service as agent_service

    svc = agent_service.session_service
    originals = (svc.save_message, svc.get_recent_messages, svc.update_session_title)
    svc.save_message = lambda *a, **k: None
    svc.get_recent_messages = lambda *a, **k: []
    svc.update_session_title = lambda *a, **k: None
    return svc, originals


def main() -> None:
    quick = "--quick" in sys.argv

    svc, originals = silence_db()
    try:
        _run_checks(quick)
    finally:
        svc.save_message, svc.get_recent_messages, svc.update_session_title = originals


def _run_checks(quick: bool) -> None:

    print("\n【任务1】Agent 工具调用验证")
    print(f"模式：{'快速（跳过 LLM 调用）' if quick else '完整（含 4 次 LLM 调用）'}\n")

    results: list[tuple[str, bool]] = []

    if not quick:
        passed = check_tool_selection()
        results.append((f"工具选择 {passed}/{len(CASES)} 条符合预期", passed == len(CASES)))
        results.append(("防死循环闸门", check_loop_guard()))
    else:
        print("（--quick 已跳过第 1、3 组）")

    results.append(("检索质量不降", check_no_quality_regression()))

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    all_ok = all(ok for _, ok in results)
    print(f"\n总体：{'全部通过' if all_ok else '存在失败项'}")
    print("提示：工具选择有随机性（temperature=0.3），偶尔 FAIL 重跑一次看是否稳定。")


if __name__ == "__main__":
    main()
