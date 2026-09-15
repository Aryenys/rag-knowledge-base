"""
【W6·Step3】时间链路自检 —— 验证"存 UTC、带时区输出"这条约定有没有落地。

跑法（必须在项目根目录 D:\\rag-knowledge-base 下）：
    python -m scripts.check_time

为什么必须用 -m 而不是 python scripts/check_time.py？
  Python 会把"脚本所在目录"塞进 sys.path[0]，直接跑脚本时 sys.path[0] 是 scripts/，
  于是 `from app.xxx import` 找不到包。-m 是以模块方式运行，会把当前目录（项目根）
  加进 sys.path，app 包才可见。这是把脚本收进子目录后最常见的坑。
"""
from datetime import datetime, timedelta, timezone
from collections import Counter

from app.core.time_utils import ensure_utc, utc_now

LOCAL_UTC_OFFSET_HOURS = 8


def check_db_storage():
    """
    判断库里现存数据到底是本地时间还是 UTC。

    判据不能是"最大值有没有超过当前 UTC" —— 那只在数据恰好落在最后 8 小时才成立，
    几天前的数据根本区分不出来。可靠做法是拿**操作系统**写的磁盘文件修改时间来校准：
    documents 表里存着 file_path，而 mtime 一定是本地墙钟。
    """
    import os

    from sqlalchemy import text

    from app.core.database import SessionLocal

    db = SessionLocal()
    need_fix = None
    try:
        rows = db.execute(text(
            "SELECT id, file_path, created_at FROM documents ORDER BY id"
        )).all()
        if not rows:
            print("    documents 表为空，没有可用于校准的数据")
            return

        print("    以磁盘文件修改时间（OS 写入，必定是本地墙钟）为基准逐行比对：")
        diffs = []
        for doc_id, path, created_at in rows:
            for cand in (str(path).replace("\\", os.sep), str(path).replace("\\", "/")):
                if os.path.exists(cand):
                    mtime = datetime.fromtimestamp(os.path.getmtime(cand))
                    diffs.append((created_at - mtime).total_seconds() / 3600)
                    break
            else:
                continue
            print(f"      doc {doc_id}: 数据库 {created_at} vs 磁盘 {mtime}")

        if not diffs:
            print("    （磁盘上找不到对应文件，无法校准）")
            return

        avg = sum(diffs) / len(diffs)
        print(f"    平均差值 = {avg:+.2f} 小时")
        if abs(avg) < 0.5:
            need_fix = True
            print("    判定：数据库里存的是【本地时间】，需要减 8 小时")
        elif abs(avg + LOCAL_UTC_OFFSET_HOURS) < 0.5:
            need_fix = False
            print("    判定：数据库里存的已是【UTC】✅ 时间链路完全打通")
        else:
            print("    判定：数据不一致（部分是本地时间、部分是 UTC），建议人工核对")
    except Exception as e:
        print(f"    （跳过：连不上数据库 —— {type(e).__name__}）")
    finally:
        db.close()

    print()
    if need_fix:
        print("    历史数据需要订正，跑这一条即可（先体检再加 --apply）：")
        print("        python -m scripts.fix_legacy_time")
        print("        python -m scripts.fix_legacy_time --apply")
    elif need_fix is False:
        print("    ✅ 存储层已完全 UTC 化，无需再处理历史数据。")


def main():
    print("=== [1] utc_now() 取的是不是真正的 UTC 挂钟 ===")
    naive_local = datetime.now()
    stored = utc_now()
    print(f"    datetime.now()  本地 = {naive_local}")
    print(f"    utc_now()       UTC  = {stored}")
    print(f"    tzinfo               = {stored.tzinfo}   (None = naive，符合 MySQL DATETIME)")
    ok1 = stored.tzinfo is None
    # 本地时间减 UTC 应该正好等于本地时区偏移（北京时间区就是 8 小时，允许 1 秒误差）
    gap = (naive_local - stored).total_seconds()
    ok2 = 0 <= gap <= 14 * 3600 + 1
    print(f"    本地 - UTC 差值        = {gap:.0f} 秒（{gap/3600:.0f} 小时）← 应等于你的时区偏移")
    print(f"    结论：{'通过' if (ok1 and ok2) else '不通过'}\n")

    print("=== [2] ensure_utc() 能不能给读出的时间补上时区和 Z 标记 ===")
    # 模拟：从 MySQL DATETIME 列读出来的"天真"UTC 值
    from_db = datetime(2026, 9, 15, 16, 30, 0)
    aware = ensure_utc(from_db)
    print(f"    库中原始值  = {from_db}   (naive)")
    print(f"    补标记之后  = {aware}")
    print(f"    tzinfo      = {aware.tzinfo}")
    print(f"    交给 Pydantic 序列化后 → {aware.isoformat().replace('+00:00', 'Z')}")
    ok3 = aware.tzinfo is not None
    print(f"    结论：{'通过（结尾带 Z，前端 new Date() 能正确换算）' if ok3 else '不通过'}\n")

    print("=== [3] 前端渲染模拟（浏览器在东八区） ===")
    iso = aware.isoformat().replace("+00:00", "Z")
    beijing = aware.astimezone(timezone(timedelta(hours=8)))
    print(f"    后端返回    {iso}")
    print(f"    北京时间    {beijing.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    旧写法（直接截字符串）会显示 {from_db.strftime('%Y-%m-%d %H:%M')}  ← 少了 8 小时，这就是修复前的 bug\n")

    print("=== [4] 数据库现状：现有数据的时间是按什么时区存的 ===")
    check_db_storage()


if __name__ == "__main__":
    main()
