"""
fix_legacy_time.py —— 订正 UTC 改造之前写入的历史时间数据。

背景
----
W6 Step3 之前，落库用的是 `datetime.now()`（服务器本地墙钟，本机 UTC+8）。
改造后统一存 UTC。所以老数据比新数据"快 8 小时"。

判定原理（关键）
----------------
不能靠"最大值是不是未来时间"来判断 —— 那个判据只在数据恰好落在最后 8 小时
窗口内才成立，几天前的数据根本分不出来（本次就出现过误判）。

正确做法是找**不可篡改的事实源**来对表：`documents.file_path` 指向磁盘上真实
存在的文件，而文件的修改时间（mtime）是**操作系统**写的，一定是本地墙钟。

    diff = 数据库 created_at - 磁盘 mtime
    diff ≈  0 小时  →  当时写的是本地墙钟，需要减 8 小时
    diff ≈ -8 小时  →  当时写的已是 UTC，无需改动

通用修正公式：
    delta = -(round(diff) + 本机UTC偏移小时数)

对没有磁盘对照的 chat_sessions / chat_messages，沿用 documents 得出的众数修正
（改造前它们走的是同一套 `datetime.now()` 代码路径），并在输出里显式标注这是推断。

用法
----
    python -m scripts.fix_legacy_time            # 只体检，不改数据（默认）
    python -m scripts.fix_legacy_time --apply    # 真正执行订正
"""

import os
import sys
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import text

from app.core.database import engine

TABLES = ["chat_messages", "chat_sessions", "documents"]

# 本机时区相对 UTC 的偏移小时数（东八区）
LOCAL_UTC_OFFSET_HOURS = 8

SEP = "=" * 66
SUB = "-" * 66


def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "(空)"


def line(t):
    print(SUB if t else SEP)


def calibrate_from_disk(conn):
    """
    用磁盘文件的真实修改时间校准 documents 表。
    返回 (每行 delta 列表, 众数 delta)
    """
    rows = conn.execute(
        text("SELECT id, file_path, created_at FROM documents ORDER BY id")
    ).fetchall()

    print("  逐行校准：数据库 created_at  vs  磁盘文件 mtime（OS 写入，必为本地墙钟）")
    print()
    print(f"  {'id':<4}{'数据库值':<22}{'磁盘mtime':<22}{'差值':<10}判定")
    print(f"  {'-' * 4}{'-' * 22}{'-' * 22}{'-' * 10}{'-' * 14}")

    deltas = []
    for doc_id, file_path, created_at in rows:
        # 库里存的是 Windows 路径分隔符，统一换成当前系统的分隔符
        real = str(file_path).replace("\\", os.sep)
        real_posix = str(file_path).replace("\\", "/")

        mtime = None
        for candidate in (real, real_posix):
            if os.path.exists(candidate):
                mtime = datetime.fromtimestamp(os.path.getmtime(candidate))
                break

        if mtime is None:
            print(f"  {doc_id:<4}{fmt(created_at):<22}{'(文件已不存在)':<22}"
                  f"{'-':<10}跳过")
            continue

        diff_h = (created_at - mtime).total_seconds() / 3600
        delta_h = -(round(diff_h) + LOCAL_UTC_OFFSET_HOURS)
        deltas.append(delta_h)

        verdict = f"需 {delta_h:+d}h" if delta_h else "已是 UTC"
        print(f"  {doc_id:<4}{fmt(created_at):<22}{fmt(mtime):<22}"
              f"{diff_h:+.2f}h{'':<4}{verdict}")

    if not deltas:
        return deltas, 0

    majority = Counter(deltas).most_common(1)[0][0]
    return deltas, majority


def show_table_stats(conn, phase):
    print()
    print(f"--- 各表 created_at 分布（{phase}）---")
    for t in TABLES:
        cnt, mn, mx = conn.execute(
            text(f"SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM {t}")
        ).fetchone()
        print(f"    {t:<16} 行数={cnt:<5} 最早={fmt(mn)}  最晚={fmt(mx)}")


def main():
    apply = "--apply" in sys.argv

    local_now = datetime.now()
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)

    line(1)
    print("  历史时间数据体检报告")
    line(1)
    print(f"  当前本地墙钟 : {fmt(local_now)}")
    print(f"  当前 UTC     : {fmt(utc_now)}")
    print(f"  差值         : {(local_now - utc_now).total_seconds() / 3600:.0f} 小时")
    print()

    with engine.connect() as conn:
        deltas, majority = calibrate_from_disk(conn)
        show_table_stats(conn, "订正前")

    line(0)
    if majority == 0:
        print("  结论：documents 全部已是 UTC，无需改动。")
        line(0)
        return

    print(f"  结论：这批数据是用本地墙钟写的，需要统一 {majority:+d} 小时。")
    print()
    print(f"  chat_sessions / chat_messages 没有磁盘文件可对照，")
    print(f"  但它们改造前走的是同一套 datetime.now() 代码路径，")
    print(f"  因此沿用同一个修正量 {majority:+d}h —— 这一步是推断，请留意。")
    line(0)

    if not apply:
        print("  这是试运行模式，没有修改任何数据。")
        print("  确认后加 --apply 参数再跑一次：")
        print("      python -m scripts.fix_legacy_time --apply")
        line(0)
        return

    print(f"  开始订正（每张表一个独立事务，出错自动回滚）")
    line(0)
    for t in TABLES:
        try:
            with engine.begin() as conn:
                n = conn.execute(
                    text(f"UPDATE {t} SET created_at = "
                         f"DATE_ADD(created_at, INTERVAL {majority} HOUR)")
                ).rowcount
            print(f"    OK   {t:<16} 已订正 {n} 行")
        except Exception as e:
            print(f"    FAIL {t}: {e}")
            print("         事务已回滚，该表保持原样")

    with engine.connect() as conn:
        show_table_stats(conn, "订正后")

    print()
    line(0)
    print("  复查：所有表的最晚时间都不应超过当前 UTC")
    with engine.connect() as conn:
        for t in TABLES:
            mx = conn.execute(text(f"SELECT MAX(created_at) FROM {t}")).scalar()
            ok = mx is None or mx <= utc_now
            print(f"    {t:<16} 最晚={fmt(mx)}  {'通过' if ok else '仍异常'}")

    print()
    print("  再用磁盘做一遍同样严格的复查：")
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT file_path, created_at FROM documents ORDER BY id LIMIT 1")
        ).fetchone()
        if row:
            p = str(row[0]).replace("\\", os.sep)
            p2 = str(row[0]).replace("\\", "/")
            for c in (p, p2):
                if os.path.exists(c):
                    m = datetime.fromtimestamp(os.path.getmtime(c))
                    print(f"    documents[0] 数据库={fmt(row[1])}  磁盘={fmt(m)}")
                    print(f"    差值={((row[1] - m).total_seconds() / 3600):+.2f}h"
                          f"  （应接近 {-LOCAL_UTC_OFFSET_HOURS}h，即数据库存的是 UTC）")
                    break
    line(0)


if __name__ == "__main__":
    main()
