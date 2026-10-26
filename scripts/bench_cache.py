"""
【任务3】缓存压测脚本：量化缓存到底省了多少。

怎么测才有说服力：
  同一个问题问两遍——第一遍缓存里没有（miss，真实走 API），
  第二遍缓存里有了（hit，直接返回）。两边只差一个缓存，其他条件完全一样，
  这个对比才干净。

为什么要先跑一条 warmup：
  第一次检索要加载 jieba 分词模型、初始化 Chroma 连接，这些冷启动开销
  加起来能有 8 秒。不预热的话，第一轮的数字全被冷启动污染，
  测出来的"缓存收益"会虚高到几十倍，属于自己骗自己。

用法：
  python -m scripts.bench_cache                # 默认取评测集前 30 条
  python -m scripts.bench_cache --n 10         # 只用 10 条（省额度，跑得快）
  python -m scripts.bench_cache --no-cache     # 对照组：全程关缓存（两次都真实调 API）
  python -m scripts.bench_cache --ttl 600      # 手动指定缓存 TTL（秒）

【一个真实踩过的坑：TTL 太短会让压测命中率变成 0】
串行跑 30 条真实检索要 6 分钟以上，而默认 TTL 是 300 秒——
于是第一轮写到一半，最早写入的条目就已经过期了，第二轮一个都命中不了，
报表上会显示"命中率 0%"。这不是缓存没用，是**压测的耗时超过了缓存有效期**。
所以压测默认把 TTL 放大到 3600 秒：压测要衡量的是缓存的加速能力，不是 TTL 过期行为。
真要测 TTL 过期，用 --ttl 显式指定。
"""
import json
import statistics
import sys
import time
from pathlib import Path

from app.cache import kb_cache
from app.config import settings
from app.rag import embedder, reranker, retriever
from app.services import retrieval_service

EVAL_SET_PATH = Path("data/eval_set.json")
DEFAULT_N = 30


def load_questions(n: int) -> list[str]:
    """从评测集里取问题。用真实语料而不是随便造的字符串，压测才有代表性。"""
    if not EVAL_SET_PATH.exists():
        raise SystemExit("评测集不存在，先跑: python -m scripts.build_eval_set")
    cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    # 去重：评测集里可能有重复问题（不同文档合成时撞车），重复会虚增命中率
    seen, questions = set(), []
    for c in cases:
        q = c["question"].strip()
        if q and q not in seen:
            seen.add(q)
            questions.append(q)
        if len(questions) >= n:
            break
    return questions


def measure_breakdown(question: str) -> dict:
    """拆解一次检索的时间花在哪：embedding API / rerank API 各多少毫秒。

    知道这个才能说清"缓存省的是什么"——省的是两次网络往返，不是 CPU。
    """
    t0 = time.perf_counter()
    embedder.embed_query(question)
    embed_ms = (time.perf_counter() - t0) * 1000

    candidates = retriever.retrieve(question, final_top_k=20)
    t0 = time.perf_counter()
    reranker.rerank(question, candidates)
    rerank_ms = (time.perf_counter() - t0) * 1000

    return {"embedding_ms": embed_ms, "rerank_ms": rerank_ms, "total_ms": embed_ms + rerank_ms}


def run_round(questions: list[str], label: str) -> dict:
    """跑一轮检索，返回本轮的耗时统计。"""
    latencies = []
    cache_hits = 0

    for q in questions:
        t0 = time.perf_counter()
        _, hit = retrieval_service.search(q, final_top_k=20)
        latencies.append((time.perf_counter() - t0) * 1000)
        cache_hits += 1 if hit else 0

    return {
        "label": label,
        "n": len(questions),
        "total_ms": sum(latencies),
        "avg_ms": statistics.mean(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1] if latencies else 0,
        "cache_hits": cache_hits,
        "hit_rate": cache_hits / len(questions) if questions else 0.0,
    }


def print_row(r: dict) -> None:
    print(
        f"| {r['label']:<16} | {r['n']:>3} | {r['total_ms'] / 1000:>7.1f}s "
        f"| {r['avg_ms']:>8.1f}ms | {r['p95_ms']:>8.1f}ms "
        f"| {r['cache_hits']:>3} | {r['hit_rate']:>6.1%} |"
    )


def main() -> None:
    n = DEFAULT_N
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    no_cache = "--no-cache" in sys.argv

    if no_cache:
        settings.CACHE_ENABLED = False
        print("对照组模式：CACHE_ENABLED=False，两轮都真实调用 API\n")

    # 压测默认放大 TTL（原因见文件头）。kb_cache 的实例在 import 时就按 settings 建好了，
    # 所以这里改 settings 不够，必须直接改实例的 ttl。
    ttl = 3600
    if "--ttl" in sys.argv:
        ttl = int(sys.argv[sys.argv.index("--ttl") + 1])
    kb_cache.retrieval_cache.ttl = ttl
    print(f"缓存 TTL 设为 {ttl}s（压测默认放大，避免第一轮还没跑完条目就过期）")

    questions = load_questions(n)
    print(f"压测样本：{len(questions)} 条真实问题（取自 data/eval_set.json）\n")

    # ---- warmup：把 jieba / Chroma 的冷启动开销吃掉，不然后面全是虚高数字 ----
    print("预热中（不计入统计）...")
    retrieval_service.search(questions[0], final_top_k=20)
    breakdown = measure_breakdown(questions[0])

    # ---- 第一轮：清缓存，全部 miss ----
    kb_cache.retrieval_cache.clear()
    kb_cache.retrieval_cache.stats.reset()
    print("第 1 轮：冷缓存（全部走真实检索）...")
    round1 = run_round(questions, "冷缓存(第1遍)")

    # ---- 第二轮：缓存已热，同一个问题再问一遍 ----
    print("第 2 轮：热缓存（同一个问题再问一遍）...")
    round2 = run_round(questions, "热缓存(第2遍)")

    print()
    print("=" * 92)
    print("缓存压测结果")
    print("=" * 92)
    print("| 轮次             | 条数 |   总耗时 |   平均延迟 |     P95延迟 | 命中 | 命中率 |")
    print("| :--------------- | ---: | -------: | ---------: | ----------: | ---: | -----: |")
    print_row(round1)
    print_row(round2)

    print()
    print("单条检索耗时分解（预热后实测一次）——缓存命中的就是省掉这两次网络往返：")
    print(f"  embedding API : {breakdown['embedding_ms']:.0f} ms")
    print(f"  rerank API    : {breakdown['rerank_ms']:.0f} ms")
    print(f"  合计          : {breakdown['total_ms']:.0f} ms")

    print()
    total_with_cache = round1["total_ms"] + round2["total_ms"]
    total_without_cache = round1["total_ms"] * 2  # 不开缓存时第二遍同样要真实调用
    api_calls = len(questions) * 2 - round2["cache_hits"]

    print("两轮合计对比（这才是该拿去汇报的数字）：")
    print(f"  开缓存  : {total_with_cache / 1000:.1f}s")
    print(f"  不开缓存: {total_without_cache / 1000:.1f}s")
    print(f"  节省    : {(total_without_cache - total_with_cache) / 1000:.1f}s")
    print(f"  API 调用: {api_calls} 次（不开缓存是 {len(questions) * 2} 次）")

    if round2["avg_ms"] > 0:
        # 这个倍数只能当"量级"看：分母是内存读取（约 0.1ms），
        # 本质是"网络往返 vs 内存读取"的差距，不代表系统整体快了多少倍
        print(
            f"\n单条延迟倍数：{round1['avg_ms'] / round2['avg_ms']:.0f}x "
            f"（分母是内存读取耗时，仅供量级参考，别拿这个数字当系统整体加速比）"
        )

    print()
    print("缓存统计（/api/health 里也能看到同样的数字）：")
    print(" ", kb_cache.retrieval_cache.stats.as_dict())


if __name__ == "__main__":
    main()
