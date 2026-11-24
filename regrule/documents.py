"""文档读取与切分（RAG 的第一步：Load → Split）。

- Load：用 pypdf / docx2txt 把 PDF / DOCX / TXT / MD 读成按页的文本；
- Split：先按段落拆成“条款”（用来做覆盖表和引用定位），
  再用 RecursiveCharacterTextSplitter 把长条款切成“片段 chunk”（用来做检索）。
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config, store
from .schemas import SourceDocument

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)


def load_file(path: Path) -> tuple[list[dict], list[str], str]:
    """返回 (pages, warnings, parser)。pages 形如 [{"page": 1, "text": "..."}]。"""
    if path.stat().st_size > config.MAX_FILE_BYTES:
        raise ValueError("文件超过 8MB。")
    suffix = path.suffix.lower()
    warnings: list[str] = []
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if len(reader.pages) > config.MAX_PDF_PAGES:
            raise ValueError("PDF 超过 80 页。")
        pages = [{"page": i + 1, "text": page.extract_text() or ""} for i, page in enumerate(reader.pages)]
        for p in pages:
            if not p["text"].strip():
                warnings.append(f"第{p['page']}页没有可选中文字，可能是空白或扫描页；本版没有 OCR，需人工核对。")
        parser = "pypdf"
    elif suffix == ".docx":
        import docx2txt

        pages = [{"page": 1, "text": docx2txt.process(str(path)) or ""}]
        warnings.append("DOCX 按段落提取，页码统一为逻辑页 1；不保证原 Word 分页与表格语义。")
        parser = "docx2txt"
    elif suffix in {".txt", ".md"}:
        pages = [{"page": 1, "text": path.read_text(encoding="utf-8-sig")}]
        parser = "text"
    else:
        raise ValueError("只支持 PDF、DOCX、TXT、MD。")
    return pages, warnings, parser


def ingest(name: str, pages: list[dict], parser: str, warnings: list[str] | None = None) -> SourceDocument:
    """切条款、切片段、写库、建索引。"""
    from .retrieval import index_chunks  # 避免循环导入

    name = (name or "").strip()
    if not name or len(name) > 200 or not pages:
        raise ValueError("文件名称或内容无效。")
    text = "\n".join(p["text"] for p in pages)
    if not text.strip() or len(text) > config.MAX_TEXT_CHARS:
        raise ValueError("需要 1 至 120000 字符的可选中文本；扫描件需要另行 OCR。")
    doc_id = store.new_id()
    clauses = []
    for p in pages:
        parts = [t.strip() for t in re.split(r"\r?\n", p["text"]) if t.strip()]
        for i, part in enumerate(parts, start=1):
            clauses.append({"id": f"{doc_id}-p{p['page']}-s{i}", "page": p["page"], "paragraph": i, "text": part})
    if len(clauses) > config.MAX_CLAUSES:
        raise ValueError("段落超过 300 段，请按章节拆分上传；不会静默截断。")
    doc = SourceDocument(
        id=doc_id, name=name, parser=parser, hash=store.sha256(text), created_at=store.now(),
        pages=pages, clauses=clauses, warnings=warnings or [],
    )
    chunks = []
    for c in doc.clauses:
        for n, piece in enumerate(SPLITTER.split_text(c.text)):
            chunks.append({
                "id": f"{c.id}-c{n}", "owner": doc_id, "kind": "document",
                "label": f"{name} · 第{c.page}页第{c.paragraph}段 · 片段{n + 1}",
                "text": piece, "clause_id": c.id,
            })
    store.put_document(doc.model_dump())
    index_chunks(chunks)
    return doc


def get(doc_id: str) -> SourceDocument:
    return SourceDocument.model_validate(store.get_document(doc_id))
