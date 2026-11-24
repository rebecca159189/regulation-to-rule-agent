import zipfile

import pytest
from pypdf import PdfWriter

from regrule import documents, fixtures

DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
{paras}
</w:body></w:document>"""


def make_docx(path, lines):
    paras = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in lines)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        z.writestr("word/document.xml", DOCX_XML.format(paras=paras))


def test_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(fixtures.example_text("gt"), encoding="utf-8")
    pages, warnings, parser = documents.load_file(p)
    doc = documents.ingest("a.txt", pages, parser, warnings)
    assert len(doc.clauses) == 3 and parser == "text"


def test_docx(tmp_path):
    p = tmp_path / "a.docx"
    make_docx(p, fixtures.example_text("gt").split("\n"))
    pages, warnings, parser = documents.load_file(p)
    doc = documents.ingest("a.docx", pages, parser, warnings)
    assert [c.text for c in doc.clauses] == fixtures.example_text("gt").split("\n")
    assert warnings and "逻辑页" in warnings[0]


def test_scanned_pdf_is_rejected(tmp_path):
    p = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    with open(p, "wb") as f:
        w.write(f)
    pages, warnings, _ = documents.load_file(p)
    assert "没有可选中文字" in warnings[0]
    with pytest.raises(ValueError, match="OCR"):
        documents.ingest("blank.pdf", pages, "pypdf", warnings)


def test_unsupported_type(tmp_path):
    p = tmp_path / "a.xlsx"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        documents.load_file(p)
