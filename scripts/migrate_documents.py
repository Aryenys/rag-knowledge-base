"""W4 Step4 一次性数据迁移：data/documents.json -> MySQL documents 表。

用法（项目根目录、venv 已激活）：
    python migrate_documents.py

三个设计要点：
  1. 保留原始 id —— Chroma 里的向量是按 document_id 存的，id 改了元数据就和向量对不上，
     删除文档时会删错或删不掉
  2. 幂等 —— 已迁移过的 id 自动跳过，重复跑不会主键冲突
  3. 可回滚 —— 成功后把 json 改名备份，不直接删除
"""
import json
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.services import document_repository

REGISTRY_PATH = Path(settings.DATA_DIR) / "documents.json"
BACKUP_PATH = Path(settings.DATA_DIR) / "documents.json.bak"


def main():
    if not REGISTRY_PATH.exists():
        print(f"找不到 {REGISTRY_PATH}，说明已经迁移过了。")
        print("如果确实需要从备份重来，先把 documents.json.bak 改回 documents.json")
        return

    docs = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    print(f"读取到 {len(docs)} 条文档元数据")

    existing = document_repository.existing_ids()
    inserted, skipped = 0, []

    for item in docs:
        if item["id"] in existing:
            skipped.append(item["id"])
            continue
        info = {
            "id": item["id"],
            "filename": item["filename"],
            "file_path": item["file_path"],
            "chunk_count": item["chunk_count"],
            "status": item["status"],
            # json 里是 ISO 字符串，DATETIME 列需要真正的 datetime 对象
            "created_at": datetime.fromisoformat(item["created_at"]),
        }
        document_repository.create_with_id(info)
        inserted += 1

    print(f"新导入 {inserted} 条")
    if skipped:
        print(f"跳过 {len(skipped)} 条（已存在）: id={skipped}")

    rows = document_repository.list_documents()
    print(f"\n迁移后 MySQL 里共 {len(rows)} 条:")
    total_chunks = 0
    for r in rows:
        print(f"  id={r['id']:<3} {r['status']:<10} chunks={r['chunk_count']:<4} {r['filename']}")
        total_chunks += r["chunk_count"]
    print(f"  合计 {total_chunks} 个 chunk")

    REGISTRY_PATH.rename(BACKUP_PATH)
    print(f"\n原文件已备份为 {BACKUP_PATH.name}")
    print("确认网页端一切正常后，可以手动删掉这个备份")


if __name__ == "__main__":
    main()
