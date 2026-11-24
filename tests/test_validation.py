import copy

import pytest

from regrule import documents, fixtures, retrieval
from regrule.platform import load_profile
from regrule.validation import source_comparison, to_fen, validate_draft


def setup(example_id="gt"):
    profile = load_profile()
    doc = documents.ingest("t", [{"page": 1, "text": fixtures.example_text(example_id)}], "example:" + example_id)
    evidence = {}
    for c in doc.clauses:
        for h in retrieval.search(c.id, "document", doc.id, profile):
            evidence[h.id] = h
    return doc, profile, evidence, fixtures.expected_draft(example_id, doc)


@pytest.mark.parametrize("value,unit,fen", [("3000", "yuan", 300000), ("12.34", "yuan", 1234), ("0.1", "yuan", 10),
                                            ("300000", "fen", 300000)])
def test_to_fen(value, unit, fen):
    assert to_fen(value, unit) == fen


@pytest.mark.parametrize("bad", ["1e3", "-1", "01", "1.234", ""])
def test_to_fen_rejects(bad):
    with pytest.raises(ValueError):
        to_fen(bad, "yuan")


@pytest.mark.parametrize("text,op", [("大于3000元", "gt"), ("不低于3000元", "gte"), ("大于等于3000元", "gte"),
                                     ("不超过100000元", "lte"), ("小于5000元", "lt"), ("等于3000元", "eq")])
def test_source_comparison(text, op):
    assert source_comparison(text) == op


@pytest.mark.parametrize("ex", ["gt", "gte", "lt", "lte", "eq", "decimal", "fen"])
def test_expected_drafts_pass(ex):
    doc, profile, ev, draft = setup(ex)
    _, checks = validate_draft(draft, doc, profile, ev, fixtures.DEFAULT_GOAL)
    assert checks.issues == []
    assert checks.compiled[0]["values"]["is_enabled"] == 0


def test_expression_for_gt():
    doc, profile, ev, draft = setup("gt")
    _, checks = validate_draft(draft, doc, profile, ev)
    expr = checks.compiled[0]["values"]["expression_en"]
    assert "(TRANS_AMT > 300000)" in expr and "TRANS_TIME_MS >= 1793462400000" in expr
    assert expr.count('CURRENCY == "CNY"') == 1


def test_wrong_operator_is_caught():
    doc, profile, ev, draft = setup("gt")
    bad = copy.deepcopy(draft)
    bad["rules"][0]["conditions"][1]["operator"] = "gte"   # 大于 写成 大于等于
    _, checks = validate_draft(bad, doc, profile, ev)
    codes = {i.code for i in checks.issues}
    assert {"OPERATOR_SOURCE", "BOUNDARY"} <= codes and not checks.compiled


def test_wrong_unit_is_caught():
    doc, profile, ev, draft = setup("gt")
    bad = copy.deepcopy(draft)
    bad["rules"][0]["conditions"][1]["unit"] = "fen"
    _, checks = validate_draft(bad, doc, profile, ev)
    assert "MONEY_SOURCE" in {i.code for i in checks.issues}


def test_fabricated_quote_is_caught():
    doc, profile, ev, draft = setup("gt")
    bad = copy.deepcopy(draft)
    bad["rules"][0]["conditions"][1]["evidence"][0]["quote"] = "单笔金额大于3000元"  # 原文里没有这句
    _, checks = validate_draft(bad, doc, profile, ev)
    assert "CITATION" in {i.code for i in checks.issues}


def test_missing_clause_is_caught():
    doc, profile, ev, draft = setup("gt")
    bad = copy.deepcopy(draft)
    bad["coverage"] = bad["coverage"][1:]
    _, checks = validate_draft(bad, doc, profile, ev)
    assert "MISSING_CLAUSE" in {i.code for i in checks.issues}


@pytest.mark.parametrize("ex,code", [("missing-unit", "MONEY"), ("aggregate", "UNSUPPORTED"),
                                     ("missing-version", "LIST_SOURCE")])
def test_cases_that_must_not_export(ex, code):
    doc, profile, ev, draft = setup(ex)
    _, checks = validate_draft(draft, doc, profile, ev)
    assert code in {i.code for i in checks.issues} and not checks.compiled


def test_schema_error():
    doc, profile, ev, _ = setup("gt")
    draft, checks = validate_draft({"summary": "x", "rules": "oops"}, doc, profile, ev)
    assert draft is None and checks.issues[0].code == "SCHEMA"
