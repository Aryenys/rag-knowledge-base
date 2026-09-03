"""
【W1·Step3】文本切分：把 loader 输出的文本块切成适合 embedding 的 chunk。
策略：LangChain 的 RecursiveCharacterTextSplitter（按段落->句子->标点递归切，尽量不切断语义）。
输入：loader 的输出 [{"text": ..., "page": ...}]
输出：list[dict]，元素 = {"content": str, "page": int|None}
"""
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def split_documents(pages: list[dict],
                    chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> list[dict]:
    """
    把每个文本块按 chunk_size 切小，相邻 chunk 保留 chunk_overlap 的重叠。
    chunk_size/overlap 默认读 config，允许传参覆盖——W5 对比实验就靠这个入口。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP,
        # 切分优先级：先尽量按段落切，段落还太长就按换行切，再不行按中文标点切……最后才硬切字符
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len,  # 按"字符数"度量长度（默认也是 len；也可换成按 token 数）
    )

    chunks = []
    for p in pages:
        for text in splitter.split_text(p["text"]):
            # 关键：切出来的每个小 chunk 继承来源页码，引用溯源才不会断链
            chunks.append({"content": text, "page": p["page"]})
    return chunks


if __name__ == "__main__":
    # 命令行测试：python -m app.rag.splitter data/你的文件.pdf [chunk_size] [chunk_overlap]
    from app.rag.loader import load_document

    if len(sys.argv) < 2:
        print("用法: python -m app.rag.splitter <文件路径> [chunk_size] [chunk_overlap]")
        sys.exit(1)

    path = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    overlap = int(sys.argv[3]) if len(sys.argv) > 3 else None

    pages = load_document(path)
    chunks = split_documents(pages, size, overlap)

    lengths = [len(c["content"]) for c in chunks]
    print(f"输入: {len(pages)} 个文本块 -> 输出: {len(chunks)} 个 chunk")
    print(f"chunk 长度: 最短 {min(lengths)} / 平均 {sum(lengths)//len(lengths)} / 最长 {max(lengths)}")
    print("=" * 50)
    for i, c in enumerate(chunks[:3]):
        print(f"--- chunk[{i}] (page={c['page']}, {len(c['content'])}字符) ---")
        print(c["content"][:150].replace("\n", " ") + "...")
    # 验证 overlap：相邻两个 chunk 的尾部/开头应有重复内容
    if len(chunks) >= 2:
        tail = chunks[0]["content"][-30:]
        print("=" * 50)
        print(f"chunk[0] 结尾 30 字: ...{tail}")
        print(f"chunk[1] 开头 30 字: {chunks[1]['content'][:30]}...")
        print("（两者应有明显重复——这就是 overlap 在起作用）")
