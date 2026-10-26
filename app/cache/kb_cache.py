"""
缓存实现：LRU（容量淘汰）+ TTL（时间淘汰）+ 命中率统计 + 知识库指纹失效。

【为什么自己写而不用 Redis】
本项目是单机部署（Docker 单容器 + 本地 Chroma），知识库规模在十万级以内，
引入 Redis 要多运维一个服务、多一次网络往返（约 0.3~1ms），
而本地内存缓存是纳秒级。所以这里选择进程内缓存。
**代价要讲清楚**：多副本部署时各副本缓存不共享，命中率会被稀释；
真到那一步再换成 Redis——接口是一样的（get/set/clear），替换成本很低。

【三个必须讲清楚的设计点】
1. **为什么用 LRU 而不是只靠 TTL**：
   TTL 管"过期"，但没管"撑爆内存"。如果用户疯狂问不同的问题，
   5 分钟内照样能塞进几万条。LRU 保证容量上限，两者是互补关系不是替代关系。
2. **为什么缓存 key 要带知识库指纹**：
   上传/删除文档后，同一个问题的正确答案变了。如果 key 里不带指纹，
   用户会一直拿到基于旧文档的检索结果——**缓存最危险的不是慢，是脏**。
   指纹变了 = key 变了 = 旧缓存自然读不到，等价于自动失效，不用人工清理。
3. **指纹查不到时怎么办**：
   MySQL 挂了就不知道知识库有没有变，此时**必须禁用缓存**（读旧数据比不缓存更糟）。
   这是降级策略：宁可慢，不能错。
"""
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import settings
from app.services import document_repository

logger = logging.getLogger("app.cache")


class CacheStats:
    """命中率统计。压测脚本和 /api/health 都靠它出数据。"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0  # 被 LRU 挤掉的条数
        self.expirations = 0  # 到 TTL 自然失效的条数
        self.disabled = 0  # 因指纹不可用而跳过缓存的次数

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """命中率：命中数 / 总查询数。分母为 0 时返回 0，避免除零。"""
        return self.hits / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "expirations": self.expirations,
            "disabled": self.disabled,
        }

    def reset(self) -> None:
        self.hits = self.misses = self.evictions = self.expirations = self.disabled = 0


class TTLCache:
    """带 TTL 的 LRU 缓存。

    底层用 OrderedDict：它是"有序字典"，把最近用过的挪到末尾（move_to_end），
    淘汰时从头部弹（last=False）——头部就是最久没用的那个，这就是 LRU 的标准写法。
    """

    def __init__(self, name: str, max_entries: int, ttl_seconds: int):
        self.name = name
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()  # FastAPI 会在线程池里跑同步生成器，必须加锁
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        """取值。三种没命中的情况：压根没有 / TTL 过期 / 缓存被禁用。"""
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.stats.misses += 1
                return None

            expire_at, value = item
            if time.monotonic() >= expire_at:
                # 惰性删除：过期了就顺手清掉，不等清理线程
                del self._data[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None

            # 命中：挪到末尾，表示"最近用过"，LRU 淘汰时最后才轮到它
            self._data.move_to_end(key)
            self.stats.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)

            # 容量超限：从头部（最久未用）开始弹
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        """清空。上传/删除文档后可以主动调一次（其实靠指纹也能自动失效，这里是双保险）。"""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# ---------- 两个缓存实例 ----------
# 检索结果缓存：省 embedding + BM25 + Rerank
retrieval_cache = TTLCache(
    name="retrieval",
    max_entries=settings.CACHE_MAX_ENTRIES,
    ttl_seconds=settings.CACHE_TTL,
)
# 答案缓存：省 LLM 生成（默认关闭）
answer_cache = TTLCache(
    name="answer",
    max_entries=settings.CACHE_MAX_ENTRIES,
    ttl_seconds=settings.CACHE_TTL,
)


# ---------- 知识库指纹 ----------

_fingerprint = {"value": None, "expire_at": 0.0}
_fingerprint_lock = threading.RLock()


def kb_fingerprint(force_refresh: bool = False) -> str | None:
    """知识库指纹：文档数 + 切片总数 + 最大文档 id。

    三者任一变化都意味着知识库内容变了（新增/删除/重新入库）。
    返回 None 表示"查不到"（MySQL 不可用），此时调用方必须禁用缓存。

    结果本地缓存 KB_FINGERPRINT_TTL 秒：
    检索一次只要几毫秒，如果每次都查库算指纹，缓存省下的时间又赔进去了。
    """
    now = time.monotonic()
    with _fingerprint_lock:
        if not force_refresh and _fingerprint["value"] and now < _fingerprint["expire_at"]:
            return _fingerprint["value"]

        try:
            docs = document_repository.list_documents()
        except Exception as e:
            # 拿不到指纹就不能确认知识库没变 → 禁用缓存，绝不能返回可能过期的结果
            logger.warning("知识库指纹获取失败，本次禁用缓存: %s", e)
            _fingerprint["value"] = None
            _fingerprint["expire_at"] = now + settings.KB_FINGERPRINT_TTL
            return None

        fingerprint = (
            f"{len(docs)}:{sum(d.get('chunk_count') or 0 for d in docs)}"
            f":{max((d['id'] for d in docs), default=0)}"
        )
        _fingerprint["value"] = fingerprint
        _fingerprint["expire_at"] = now + settings.KB_FINGERPRINT_TTL
        return fingerprint


def invalidate_on_kb_change() -> None:
    """文档增删后调用：清掉指纹的本地缓存，让它下次重新算。

    其实不清也没事（最多 10 秒后自动重算），但上传完立刻问同一个问题时，
    这 10 秒的窗口会让人以为"缓存没失效"。所以主动清一次更稳妥。
    """
    with _fingerprint_lock:
        _fingerprint["value"] = None
        _fingerprint["expire_at"] = 0.0


def normalize_query(query: str) -> str:
    """查询归一化：去首尾空白、压缩中间空白、统一小写。

    目的：让"BGE-M3 是什么"和"bge-m3   是什么 "算同一个 key，
    否则用户多打一个空格就白缓存一次。
    """
    return " ".join(query.strip().lower().split())


def make_key(query: str, scope: str = "retrieval") -> str | None:
    """生成缓存 key = 归一化问题 + 作用范围 + 知识库指纹。

    返回 None 表示指纹不可用，调用方应跳过缓存。
    """
    fingerprint = kb_fingerprint()
    if fingerprint is None:
        return None
    raw = f"{scope}|{normalize_query(query)}|kb={fingerprint}"
    # md5 只用于生成定长 key（省内存），不涉及安全，不需要 sha256
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
