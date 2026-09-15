"""W4 Step0 验证脚本：连接 MySQL 并创建 rag_kb 数据库。

用法（项目根目录、venv 已激活）：
    python init_mysql.py

一次验证四件事：
  1. MySQL 服务是否在运行
  2. root 密码是否正确
  3. pymysql 驱动是否可用
  4. caching_sha2_password 认证是否通过（依赖 cryptography 包）
密码用 getpass 交互式输入（不回显），避免明文写进文件。
"""
import getpass
import sys

import pymysql

DB_NAME = "rag_kb"


def main():
    host = input("MySQL 地址（默认 127.0.0.1）: ").strip() or "127.0.0.1"
    port_raw = input("端口（默认 3306）: ").strip() or "3306"
    user = input("用户名（默认 root）: ").strip() or "root"
    password = getpass.getpass("密码（输入时不显示）: ")

    try:
        port = int(port_raw)
    except ValueError:
        print(f"端口必须是数字，你输入的是: {port_raw}")
        sys.exit(1)

    print(f"\n正在连接 {user}@{host}:{port} ...")
    try:
        # 不指定 database：先连上服务器本身，才能执行 CREATE DATABASE
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset="utf8mb4",
            connect_timeout=10,
        )
    except Exception as e:
        print(f"\n连接失败：{type(e).__name__}: {e}")
        print("\n常见原因：")
        print("  1. MySQL 服务没启动（开始菜单搜'服务'，找 MySQL84 启动）")
        print("  2. 密码错误")
        print("  3. 端口不是 3306")
        print("  4. 安装时选了 Skip 没配置服务")
        sys.exit(1)

    print("连接成功")
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            print(f"数据库 '{DB_NAME}' 已就绪（字符集 utf8mb4）")

            cur.execute("SELECT @@version")
            version = cur.fetchone()[0]
            print(f"MySQL 版本: {version}")

            cur.execute("SHOW DATABASES")
            dbs = [row[0] for row in cur.fetchall()]
            print(f"当前所有数据库: {', '.join(dbs)}")
        conn.commit()
    finally:
        conn.close()

    print("\n全部通过，可以进入 W4 Step1 了")


if __name__ == "__main__":
    main()
