"""向量库查看工具：看看 Chroma 里到底存了什么。

用法（项目根目录、venv 已激活）：
    python check_vector.py

能看四件事：
  1. 向量库文件在哪、多大
  2. 总共有多少条 chunk
  3. 每个文档实际有多少条 chunk（和 MySQL 元数据对账，能查出孤儿数据）
  4. 一条向量长什么样（维度 + 前几个数字）
"""
import os
from collections import Counter
from pathlib import Path

import chromadb

from app.config import settings
from app.services import document_repository


def dir_size(path: str) -> str:
    """算目录总大小（字节 -> MB）"""
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return f"{total / 1024 / 1024:.1f} MB"


def main():
    print("=" * 66)
    print("向量库现状")
    print("=" * 66)
    print(f"存储位置 : {Path(settings.CHROMA_DIR).resolve()}")
    print(f"占用体积 : {dir_size(settings.CHROMA_DIR)}")
    print("（MySQL 里只有文件名/块数等元数据，向量本体在这里）")

    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    col = client.get_or_create_collection(
        name="knowledge_base", metadata={"hnsw:space": "cosine"}
    )

    total = col.count()
    print(f"\nchunk 总数: {total}")

    # ---------- 按文档对账 ----------
    metas = col.get(include=["metadatas"])["metadatas"]
    counter = Counter(m["document_id"] for m in metas)
    mysql_docs = {d["id"]: d for d in document_repository.list_documents()}

    # 三处对账：向量库条数 / MySQL 元数据 / 磁盘文件。
    # 删除文档后跑一遍，三列都正常就说明删干净了。
    print("\n三处对账（向量 / 元数据 / 磁盘）:")
    print("-" * 66)
    print(f"  {'doc_id':<8}{'向量':<8}{'元数据':<9}{'磁盘':<7}文件名")
    for doc_id, actual in sorted(counter.items()):
        info = mysql_docs.get(doc_id)
        if info is None:
            print(f"  {doc_id:<8}{actual:<8}{'无记录':<9}{'-':<7}⚠️ MySQL 里查不到（孤儿数据）")
            continue
        declared = info["chunk_count"]
        meta = "一致" if declared == actual else f"说{declared}条"
        disk = "在" if Path(info["file_path"]).exists() else "缺失"
        print(f"  {doc_id:<8}{actual:<8}{meta:<9}{disk:<7}{info['filename']}")

    for doc_id, info in mysql_docs.items():
        if doc_id not in counter:
            disk = "在" if Path(info["file_path"]).exists() else "缺失"
            print(f"  {doc_id:<8}{'0':<8}{'有记录':<9}{disk:<7}⚠️ 向量为空：{info['filename']}")

    # ---------- 看一条向量的样子 ----------
    sample = col.get(limit=1, include=["embeddings", "documents", "metadatas"])
    # 注意：embeddings 是 numpy 数组，不能直接用 if 判断真假（会报 ambiguous 错误）
    vecs = sample["embeddings"]
    if vecs is not None and len(vecs) > 0:
        vec = vecs[0]
        print("\n" + "=" * 66)
        print("一条向量长什么样")
        print("-" * 66)
        print(f"  维度      : {len(vec)} 个浮点数（BGE-M3 固定 1024 维）")
        print(f"  前 8 个数字: {[round(float(x), 4) for x in vec[:8]]}")
        print(f"  数组全貌  : 像这样重复 {len(vec)} 次，人眼读不出含义，但方向有意义")
        print(f"\n  对应原文  : {sample['documents'][0][:60]}...")
        print(f"  元数据    : {sample['metadatas'][0]}")

    print("\n" + "=" * 66)
    print("怎么检索？")
    print("-" * 66)
    print('  python -m app.rag.vector_store query "304 状态码是什么意思" 5')
    print("  （不需要手动看向量，让它帮你找最相似的）")


if __name__ == "__main__":
    main()
