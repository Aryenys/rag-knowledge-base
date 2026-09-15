"""
【W5·Step2】评估脚本：量化三种检索方案的效果，产出可贴进 README / 简历的数据。

对比方案：
  A 纯向量召回          —— W2 的基础版
  B 混合召回（RRF）      —— W3 加了 BM25
  C 混合召回 + Rerank   —— W3 加了 cross-encoder 精排

指标：
  Recall@K 命中率：正确答案是否出现在前 K 条里（有=1，无=0），最后取平均
  MRR      排名分：正确答案排第几位，取倒数（第1名=1分，第2名=0.5，第5名=0.2）

用法：
  python -m tests.evaluate           只看汇总
  python -m tests.evaluate --detail   打印每题的命中情况
"""
import json
import sys
from pathlib import Path

from app.config import settings
from app.rag import embedder, reranker, retriever, vector_store

EVAL_SET_PATH = Path("data/eval_set.json")
TOP_K = 5        # 评估"前 5 条"（和线上 RERANK_TOP_K 保持一致）
RECALL_K = 20    # 混合检索先召回 20 条，给 reranker 挑


def load_cases() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        raise SystemExit("评测集不存在，先跑: python build_eval_set.py")
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


# ---------- 指标计算 ----------

def hit_rank(hits: list[dict], keyword: str) -> int:
    """关键词第一次出现在第几条（1 起算），没命中返回 0。
    比较时统一转小写，避免 B+ Tree / b+ tree 大小写差异造成误判。"""
    kw = keyword.lower()
    for i, h in enumerate(hits, 1):
        if kw in h["content"].lower():
            return i
    return 0


def recall_at_k(rank: int, k: int) -> float:
    """命中率：答案在前 k 条内算命中"""
    return 1.0 if 0 < rank <= k else 0.0


def reciprocal_rank(rank: int) -> float:
    """倒数排名：第 1 位得 1 分，第 2 位 0.5 分，第 5 位 0.2 分，没命中 0 分"""
    return 1.0 / rank if rank > 0 else 0.0


def main():
    cases = load_cases()
    show_detail = "--detail" in sys.argv

    print(f"评测集 {len(cases)} 条    评估指标 Recall@{TOP_K} / MRR@{TOP_K}")
    print("=" * 60)

    # 三个方案的结果累计
    stats = {
        "A 纯向量": {"recall": 0.0, "mrr": 0.0, "miss": []},
        "B 混合检索": {"recall": 0.0, "mrr": 0.0, "miss": []},
        "C 混合+Rerank": {"recall": 0.0, "mrr": 0.0, "miss": []},
    }

    for idx, case in enumerate(cases, 1):
        question = case["question"]
        keyword = case["expected_keyword"]

        # A：纯向量 top5
        vec_hits = vector_store.search(embedder.embed_query(question), top_k=TOP_K)

        # B、C 共用同一批 20 条候选，这样对比才公平
        #   （只差在"是否经过 reranker 重新排序"这一点上）
        candidates = retriever.retrieve(question, final_top_k=RECALL_K)
        hybrid_hits = candidates[:TOP_K]                              # B：RRF 顺序取前 5
        reranked_hits = reranker.rerank(question, candidates, top_k=TOP_K)  # C：精排取前 5

        for name, hits in (
            ("A 纯向量", vec_hits),
            ("B 混合检索", hybrid_hits),
            ("C 混合+Rerank", reranked_hits),
        ):
            rank = hit_rank(hits, keyword)
            stats[name]["recall"] += recall_at_k(rank, TOP_K)
            stats[name]["mrr"] += reciprocal_rank(rank)
            if rank == 0:
                stats[name]["miss"].append(question)

        if show_detail:
            ra = hit_rank(vec_hits, keyword)
            rb = hit_rank(hybrid_hits, keyword)
            rc = hit_rank(reranked_hits, keyword)
            fmt = lambda r: f"第{r}位" if r else "未命中"
            print(f"{idx:2d}. {question[:34]}")
            print(f"      A={fmt(ra):<6} B={fmt(rb):<6} C={fmt(rc):<6}  关键词={keyword}")

    # ---------- 汇总输出 ----------
    n = len(cases)
    print()
    for name, s in stats.items():
        print(f"{name:<14} 命中 {n - len(s['miss']):>2}/{n}   未命中 {len(s['miss'])} 条")

    print("\n" + "=" * 60)
    print("结果（可直接贴进 README 或写进简历）")
    print("=" * 60)
    print(f"| 方案 | Recall@{TOP_K} | MRR@{TOP_K} | 命中数 |")
    print("| --- | --- | --- | --- |")
    for name, s in stats.items():
        print(f"| {name} | {s['recall'] / n:.1%} | {s['mrr'] / n:.3f} | {n - len(s['miss'])}/{n} |")

    # ---------- 三个方案全没命中的题：最值得分析 ----------
    hard = [c for c in cases if all(c["question"] in stats[name]["miss"] for name in stats)]
    if hard:
        print(f"\n三种方案全部未命中的题（{len(hard)} 条）——优先排查这些:")
        for c in hard:
            print(f"  - {c['question'][:40]}  (关键词: {c['expected_keyword']})")
        print("\n  可能原因：① 关键词本身不在原文里（评测集标注问题）")
        print("            ② 该 chunk 被切分切断了关键信息（可调 CHUNK_SIZE）")
        print("            ③ 语义确实太远，需要查询改写或更好的 embedding")


if __name__ == "__main__":
    main()
