"""W4 数据库体检脚本：连接 -> 建表 -> 打印表结构与数据行数。

用法（项目根目录、venv 已激活）：
    python check_db.py

后面 W4 每一步跑一遍，就能直观看到数据有没有真的写进去。
"""
import sys
from urllib.parse import urlparse

from sqlalchemy import inspect, text

from app.config import settings
from app.core.database import engine, init_db


def masked_url(url: str) -> str:
    """把连接串里的密码打码，方便安全地把输出贴给别人排查。"""
    u = urlparse(url)
    pwd = "****" if u.password else "(空)"
    return f"{u.scheme}://{u.username}:{pwd}@{u.hostname}:{u.port or 3306}{u.path}"


def main():
    print("=" * 60)
    print("连接串:", masked_url(settings.DATABASE_URL))
    print("=" * 60)

    try:
        init_db()
        print("init_db() 执行成功（表不存在则新建，已存在则跳过）\n")
    except Exception as e:
        print(f"\ninit_db() 失败：{type(e).__name__}: {e}")
        print("\n常见原因：")
        print("  1. .env 里第 24 行的 '你的密码' 没换成真实 root 密码")
        print("  2. MySQL 服务没启动（服务里找 MySQL80 启动）")
        print("  3. rag_kb 库不存在（跑一下 python init_mysql.py）")
        print("  4. 密码含 @ # / 等特殊字符 -> 需 URL 编码，见下方说明")
        sys.exit(1)

    insp = inspect(engine)
    tables = insp.get_table_names()

    if not tables:
        print("数据库里一张表都没有，检查是否连错了库")
        sys.exit(1)

    with engine.connect() as conn:
        for t in tables:
            print(f"表 {t}")
            cols = insp.get_columns(t)
            for c in cols:
                print(f"   {c['name']:<16} {str(c['type'])}")
            count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            print(f"   -> 当前 {count} 行\n")

    print("体检通过")


if __name__ == "__main__":
    main()
