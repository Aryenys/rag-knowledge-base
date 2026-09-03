"""
【W1·Step2】文档解析：PDF / MD / TXT / DOCX -> 纯文本块（保留页码等元数据）。
依赖：pypdf、python-docx
输出统一格式：list[dict]，元素 = {"text": str, "page": int|None}
"""
import sys
from pathlib import Path


def load_pdf(file_path: str | Path) -> list[dict]:
    """用 pypdf 按页提取文本。每页一个元素，保留页码（引用溯源要用）。"""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):  # 页码从 1 开始，符合人类习惯
        text = (page.extract_text() or "").strip()
        if text:  # 跳过提取不出文字的页（如纯图片页、扫描件）
            pages.append({"text": text, "page": i})
    if not pages:
        raise ValueError("PDF 未提取到任何文字（可能是扫描件，需要 OCR，本项目暂不支持）")
    return pages


def load_text(file_path: str | Path) -> list[dict]:
    """MD / TXT：整体读为一段，page=None（没有页的概念）。"""
    text = Path(file_path).read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("文件内容为空")
    return [{"text": text, "page": None}]


def load_docx(file_path: str | Path) -> list[dict]:
    """用 python-docx 按段落提取，非空段落用换行拼接，整体为一段。"""
    from docx import Document

    doc = Document(str(file_path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    if not text:
        raise ValueError("DOCX 未提取到任何文字")
    return [{"text": text, "page": None}]


def load_document(file_path: str | Path) -> list[dict]:
    """统一入口：按扩展名分发到对应的解析函数。"""
    suffix = Path(file_path).suffix.lower()
    loaders = {".pdf": load_pdf, ".md": load_text, ".txt": load_text, ".docx": load_docx}
    if suffix not in loaders:
        raise ValueError(f"不支持的文件格式: {suffix}（支持 pdf/md/txt/docx）")
    return loaders[suffix](file_path)


if __name__ == "__main__":
    # 命令行测试入口：python -m app.rag.loader data/你的文件.pdf
    if len(sys.argv) < 2:
        print("用法: python -m app.rag.loader <文件路径>")
        sys.exit(1)

    path = sys.argv[1]
    result = load_document(path)
    total_chars = sum(len(p["text"]) for p in result)
    print(f"文件: {path}")
    print(f"解析出 {len(result)} 个文本块, 共 {total_chars} 字符")
    print("=" * 50)
    for p in result[:3]:  # 只打印前 3 块预览
        preview = p["text"][:200].replace("\n", " ")
        print(f"[page={p['page']}] {preview}...")
