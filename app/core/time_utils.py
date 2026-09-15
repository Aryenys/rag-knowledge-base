"""
【W6·Step3】统一时间处理。

━━━━━━ 一条铁律（面试时能讲出来的那种） ━━━━━━
    数据库里所有时间字段，一律存 UTC；对外输出一律带时区标记（...Z）。

为什么要这样？
  存 UTC 是因为存储层不应该关心"谁在看"——同一个字段，
  北京的用户看到 +8，纽约的用户看到 -5，但库里只有一个值。
  如果存本地时间，一旦服务器搬家到另一个时区，所有历史数据就全部错位了，
  而且没法回滚判断。

常见踩坑组合（本项目改造前就是这样）：
  1. 写入用 datetime.now()        → 取的是服务器本地时间
  2. 字段用 MySQL DATETIME        → 不保存时区信息
  3. 读出后前端直接 .slice(0,16)  → 把UTC挂钟时间当本地时间显示，差整整 8 小时
"""
from datetime import datetime, timezone


def utc_now() -> datetime:
    """当前 UTC 时间，返回 naive（不带 tzinfo）对象。

    为什么不强留 tzinfo？
    因为 MySQL 的 DATETIME 列不存时区，带 tzinfo 的 datetime 交给驱动时
    行为因驱动而异（有的报错、有的静默丢弃）。
    干脆"在入口就抹平"，配合下面的 ensure_utc() 在出口补回来，整个链路可控。
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """给从库里读出的时间补上"我是 UTC"这个标记（naive -> aware）。

    补标记有什么用？
    Pydantic 序列化 aware datetime 时会输出成 "2026-09-15T16:39:27Z"（带 Z），
    前端 new Date(这个值) 才能正确换算成浏览器本地时区。
    不补的话输出 "...T16:39:27"（无 Z），JS 会当成本地时间处理，直接差 8 小时。
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)   # 约定：库里的值都是 UTC
    return dt.astimezone(timezone.utc)
