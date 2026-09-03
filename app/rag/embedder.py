"""
【W1·Step4】Embedding：调硅基流动 API 把文本变成向量。
接口兼容 OpenAI 格式：POST {base_url}/embeddings
要点：批量调用（省请求数）+ 失败重试（429 限流/网络抖动时指数退避）
"""
import sys
import time

import httpx

from app.config import settings

BATCH_SIZE = 32   # 每批最多 32 条（硅基流动单次上限内，太少则请求数膨胀）
MAX_RETRY = 3     # 单批最多重试 3 次


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """调一次 API，把一批文本变成向量。失败时指数退避重试。"""
    url = f"{settings.SILICONFLOW_BASE_URL}/embeddings"
    headers = {"Authorization": f"Bearer {settings.SILICONFLOW_API_KEY}"}
    payload = {"model": settings.EMBEDDING_MODEL, "input": texts}

    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:          # 限流：等待后重试
                wait = 2 ** attempt              # 指数退避：2s, 4s, 8s
                print(f"  [限流 429] 等待 {wait}s 后第 {attempt + 1} 次重试...")
                time.sleep(wait)
                continue
            resp.raise_for_status()              # 其他 4xx/5xx 直接抛异常
            data = resp.json()["data"]
            # 返回的 data 是按 index 排列的列表，每个元素 {"embedding": [...], "index": i}
            data.sort(key=lambda x: x["index"])  # 防御性排序，保证顺序与输入对应
            return [item["embedding"] for item in data]
        except httpx.HTTPError as e:
            if attempt == MAX_RETRY:
                raise RuntimeError(f"embedding 请求失败（重试 {MAX_RETRY} 次后放弃）: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError("unreachable")  # 理论上不会走到


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量 embedding：自动分批，返回与输入顺序对应的向量列表。"""
    if not texts:
        return []
    vectors = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"  embedding 第 {i // BATCH_SIZE + 1}/{total_batches} 批 ({len(batch)} 条)...")
        vectors.extend(_embed_batch(batch))
    return vectors


def embed_query(text: str) -> list[float]:
    """单条查询 embedding（问答时用）。"""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # 命令行测试：python -m app.rag.embedder
    # 用余弦相似度直观验证"语义相近 -> 向量相近"
    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b)

    s1 = "苹果是一种营养丰富的水果"
    s2 = "我最爱吃的水果是苹果"
    s3 = "操作系统中的进程调度算法"
    print(f"正在 embedding 3 个句子...")
    v1, v2, v3 = embed_texts([s1, s2, s3])
    print(f"向量维度: {len(v1)}")
    print(f"相似度(苹果句 vs 苹果句) = {cosine(v1, v2):.4f}   <- 应该较高 (>0.6)")
    print(f"相似度(苹果句 vs 操作系统) = {cosine(v1, v3):.4f}   <- 应该较低 (<0.4)")
