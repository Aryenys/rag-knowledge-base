"""W5 Step1：自动生成评测集。

思路（业界叫 "synthetic test set"，RAGAS 等框架也这么干）：
  从每个文档抽几个 chunk -> 让 LLM 基于这段内容编一个问题 -> 这段 chunk 就是标准答案。
  这样"问题-答案"天然对应，不需要人工标注。

产出 data/eval_set.json：
  [{"question": "...", "expected_keyword": "...", "source_doc_id": 3, "source_filename": "111.pdf"}]

用法（venv 已激活）：
    python build_eval_set.py
"""
import json
import random
import time
from pathlib import Path

from openai import OpenAI

from app.config import settings
from app.rag import vector_store

OUT_PATH = Path("data/eval_set.json")
PER_DOC = 10       # 每份文档抽几个 chunk 来出题（chunk 不够时取全部）
MIN_LEN = 120      # 太短的 chunk 信息量不足，跳过
SLEEP = 0.4        # 请求间隔，避免触发限流

_client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)

PROMPT = """下面是知识库里的一段资料。请你完成两件事：

1. 编一个"能用这段资料回答"的问题，要像真实用户会问的那样口语化自然
2. 给出一个"正确答案里必然会出现"的特征关键词

要求：
- 关键词必须是这段资料里真实出现的专有名词或术语，长度 2-10 个字
- 不要用"系统""方法""问题"这种到处都有的通用词（否则评估会误判为命中）
- 问题里不要直接照抄关键词，要让检索器真的去"找"
- 只输出 JSON，不要任何解释，格式：{{"question": "...", "keyword": "..."}}

【资料】
{content}
"""


def generate_qa(content: str) -> dict | None:
    """调 LLM 生成一个问题+关键词；失败返回 None"""
    try:
        resp = _client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": PROMPT.format(content=content[:900])}],
            temperature=0.3,   # 出题需要一点多样性，但不能跑偏
        )
    except Exception as e:
        print(f"      调用失败: {e}")
        return None

    text = resp.choices[0].message.content.strip()
    # 模型有时会用 ```json 代码块包裹，剥掉
    if "```" in text:
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]

    try:
        data = json.loads(text.strip())
    except Exception:
        print(f"      解析失败，原文: {text[:80]}")
        return None

    if not data.get("question") or not data.get("keyword"):
        return None
    return data


def main():
    chunks = vector_store.get_all_chunks()
    print(f"向量库共 {len(chunks)} 条 chunk\n")

    # 按文档分组，保证每份文档都能覆盖到（否则评测集会偏袒大文档）
    by_doc: dict[int, list] = {}
    for c in chunks:
        by_doc.setdefault(c["document_id"], []).append(c)

    random.seed(42)   # 固定随机种子，保证可复现
    results = []

    for doc_id in sorted(by_doc):
        pool = [c for c in by_doc[doc_id] if len(c["content"]) >= MIN_LEN]
        if not pool:
            print(f"  doc {doc_id}: 没有够长的 chunk，跳过")
            continue

        sampled = random.sample(pool, min(PER_DOC, len(pool)))
        fname = sampled[0]["filename"]
        print(f"  doc {doc_id} ({fname}): 抽 {len(sampled)} 条出题")

        for c in sampled:
            qa = generate_qa(c["content"])
            time.sleep(SLEEP)
            if qa is None:
                continue
            results.append({
                "question": qa["question"],
                "expected_keyword": qa["keyword"],
                "source_doc_id": doc_id,
                "source_filename": c["filename"],
            })
            print(f"      Q: {qa['question'][:36]}")
            print(f"      K: {qa['keyword']}")

    # 增量合并：保留已有题目，这样你手工删掉的低质题不会被重新生成回来
    existing = []
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    seen = {e["question"] for e in existing}
    merged = existing + [r for r in results if r["question"] not in seen]

    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n本次新出 {len(results)} 题，去重合并后评测集共 {len(merged)} 条")
    print("下一步：人工过一遍，删掉不合适的（问题太泛、关键词太通用的）")


if __name__ == "__main__":
    main()
