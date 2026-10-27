"""
【任务4】Prompt 版本对比脚本：同一个问题，用两个版本的 prompt 各生成一次。

为什么要这个脚本：
  改 prompt 最常见的困境是「改完了不知道变好还是变差」。
  检索类指标（Recall/MRR）衡量的是检索，对 prompt 完全不敏感，帮不上忙；
  生成质量又没有可靠的自动指标。所以务实的做法是：
    - 用同一批检索结果喂给两个版本（控制变量：只差 prompt）
    - 把可量化的表面特征统计出来（引用标注次数、长度、是否拒答）
    - 答案并排打出来，由人来做最终判断

用法：
  python -m scripts.compare_prompts              # 默认 3 条问题，对比 v1 vs v2
  python -m scripts.compare_prompts --n 5        # 用 5 条
  python -m scripts.compare_prompts --a v1 --b v2
"""
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

from app.config import settings
from app.core import prompt_registry, prompts
from app.services import retrieval_service

CITE_RE = re.compile(r"\[(\d+)\]")
REFUSAL_HINT = "无法回答"


def load_questions(n: int) -> list[str]:
    path = Path(settings.DATA_DIR) / "eval_set.json"
    if not path.exists():
        return ["RAG 相比纯生成模型有什么优势？", "BGE-M3 支持的最大输入长度是多少？"][:n]
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data[:n]:
        q = item.get("question") or item.get("query")
        if q:
            out.append(q)
    return out


def generate(system_prompt: str, question: str) -> str:
    client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


def stats(answer: str) -> dict:
    cites = CITE_RE.findall(answer)
    return {
        "chars": len(answer),
        "citations": len(cites),
        "distinct_sources": len(set(cites)),
        "refused": REFUSAL_HINT in answer,
    }


def main() -> None:
    argv = sys.argv
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 3
    va = argv[argv.index("--a") + 1] if "--a" in argv else "v1"
    vb = argv[argv.index("--b") + 1] if "--b" in argv else "v2"

    questions = load_questions(n)
    print(f"\nPrompt 版本对比：{va} vs {vb}（{len(questions)} 条问题）")
    print("两次生成共用同一批检索结果，唯一变量是 prompt\n")

    agg = {va: {"chars": 0, "citations": 0}, vb: {"chars": 0, "citations": 0}}

    for i, q in enumerate(questions, 1):
        # 检索只做一次：两个版本吃同样的资料，这才叫控制变量
        chunks, cache_hit = retrieval_service.search(q, final_top_k=20)
        context = prompts.build_context(chunks)

        print("=" * 70)
        print(f"[{i}/{len(questions)}] {q}")
        print(f"检索命中 {len(chunks)} 条（缓存{'命中' if cache_hit else '未命中'}）")

        for v in (va, vb):
            system = prompt_registry.get_template("rag_system", version=v)
            answer = generate(system.format(context=context), q)
            s = stats(answer)
            agg[v]["chars"] += s["chars"]
            agg[v]["citations"] += s["citations"]
            print(f"\n  --- {v} ---")
            print(f"  长度 {s['chars']} 字 | 引用标注 {s['citations']} 处 "
                  f"（涉及 {s['distinct_sources']} 个来源）| 拒答：{'是' if s['refused'] else '否'}")
            print("  " + answer.replace("\n", "\n  ")[:600])
        print()

    print("=" * 70)
    print("汇总（平均值）")
    for v in (va, vb):
        c = agg[v]
        k = max(len(questions), 1)
        print(f"  {v}: 平均 {c['chars'] / k:.0f} 字，平均引用标注 {c['citations'] / k:.1f} 处")
    print("\n注意：这些只是表面特征，不代表质量好坏。")
    print("引用标注变多可能是溯源更好，也可能是啰嗦——最终要人来看上面的答案。")


if __name__ == "__main__":
    main()
