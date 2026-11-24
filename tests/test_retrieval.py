from langchain_core.embeddings import DeterministicFakeEmbedding

from regrule import documents, embeddings, fixtures, retrieval
from regrule.platform import load_profile


def test_bm25_finds_amount_clause():
    profile = load_profile()
    doc = documents.ingest("t", [{"page": 1, "text": fixtures.example_text("gt")}], "example:gt")
    hits = retrieval.search("单笔 金额 付款", "document", doc.id, profile)
    assert hits[0].clause_id == doc.clauses[1].id and hits[0].bm25_rank == 1


def test_scopes_do_not_leak():
    profile = load_profile()
    a = documents.ingest("a", [{"page": 1, "text": fixtures.example_text("gt")}], "x")
    b = documents.ingest("b", [{"page": 1, "text": fixtures.example_text("lt")}], "x")
    assert all(h.owner == a.id for h in retrieval.search("金额", "document", a.id, profile))
    assert all(h.kind == "platform" for h in retrieval.search("金额 单位", "platform", b.id, profile))


def test_clause_id_lookup():
    profile = load_profile()
    doc = documents.ingest("t", [{"page": 1, "text": fixtures.example_text("gt")}], "x")
    hits = retrieval.search(doc.clauses[2].id, "document", doc.id, profile)
    assert [h.clause_id for h in hits] == [doc.clauses[2].id]


def test_hybrid_with_vectors():
    embeddings.use_embeddings(DeterministicFakeEmbedding(size=32), "fake32")
    try:
        profile = load_profile()
        doc = documents.ingest("t", [{"page": 1, "text": fixtures.example_text("gt")}], "x")
        hits = retrieval.search("单笔 金额 付款", "document", doc.id, profile)
        assert any(h.vector_rank for h in hits) and hits[0].bm25_rank == 1
    finally:
        embeddings.use_embeddings(None)


def test_long_clause_is_chunked():
    profile = load_profile()
    long = "监管要求。" * 400  # 2000 字
    doc = documents.ingest("long", [{"page": 1, "text": long}], "x")
    hits = retrieval.search(doc.clauses[0].id, "document", doc.id, profile, k=15)
    assert len(hits) >= 3 and all(len(h.text) <= 800 for h in hits)
