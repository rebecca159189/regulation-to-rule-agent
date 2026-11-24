"""10 份人工编写的学习示例和标准答案（不是真实监管文件）。

用途：
1. 页面上一键载入，体验完整流程；
2. 演示模式下作为“预置答案”（明确标注不是模型生成）；
3. scripts/eval.py 用它评测真实模型的效果。
"""
from __future__ import annotations

from .schemas import SourceDocument

EXAMPLES = [
    {"id": "gt", "name": "01 大于3000元", "wording": "大于3000元", "operator": "gt", "value": "3000", "unit": "yuan"},
    {"id": "gte", "name": "02 不低于3000元", "wording": "不低于3000元", "operator": "gte", "value": "3000", "unit": "yuan"},
    {"id": "lt", "name": "03 小于5000元", "wording": "小于5000元", "operator": "lt", "value": "5000", "unit": "yuan"},
    {"id": "lte", "name": "04 不超过100000元", "wording": "不超过100000元", "operator": "lte", "value": "100000", "unit": "yuan"},
    {"id": "eq", "name": "05 等于3000元", "wording": "等于3000元", "operator": "eq", "value": "3000", "unit": "yuan"},
    {"id": "decimal", "name": "06 小数金额12.34元", "wording": "大于12.34元", "operator": "gt", "value": "12.34", "unit": "yuan"},
    {"id": "fen", "name": "07 原文以分计价", "wording": "大于300000分", "operator": "gt", "value": "300000", "unit": "fen"},
    {"id": "missing-unit", "name": "08 金额单位缺失", "wording": "大于3000", "operator": "gt", "value": "3000", "unit": None},
    {"id": "aggregate", "name": "09 累计窗口不支持", "wording": "近24小时累计金额大于3000元", "operator": "gt", "value": "3000", "unit": "yuan"},
    {"id": "missing-version", "name": "10 名单版本缺失", "wording": "大于3000元", "operator": "gt", "value": "3000", "unit": "yuan"},
]

# 每个示例“正确的表现”，评测脚本用
EXPECTED_BEHAVIOR = {
    "missing-unit": "clarify",      # 应该提问：单位是什么
    "missing-version": "clarify",   # 应该提问：名单版本是什么
    "aggregate": "blocked",         # 应该标记为暂不能落实
}

DEFAULT_GOAL = "提取可监控的单笔交易要求，检索字段与名单资料，并明确人工复核、报送等非规则事项。"


def example(example_id: str) -> dict:
    for item in EXAMPLES:
        if item["id"] == example_id:
            return item
    raise KeyError(f"学习示例不存在：{example_id}")


def example_text(example_id: str) -> str:
    ex = example(example_id)
    lst = "风险客户名单A" if example_id == "missing-version" else "风险客户名单A（2026-11版）"
    predicate = ex["wording"] if example_id == "aggregate" else f"单笔人民币金额{ex['wording']}"
    return (
        f"学习示例{ex['name']}，不是真实监管文件。\n"
        f"自2026年11月1日00:00（北京时间）起，手机银行转账中，{lst}客户的{predicate}且为付款时，标记为风险。\n"
        "命中记录由分析人员复核后准备报送，报送须另走审批流程。"
    )


def expected_draft(example_id: str, document: SourceDocument) -> dict:
    """标准答案。只对未改动的内置示例可用。"""
    ex = example(example_id)
    full = "\n".join(p["text"] for p in document.pages)
    if full != example_text(example_id):
        raise ValueError("只有未经改动的内置学习示例可以载入预期草稿。")
    c0, c1, c2 = document.clauses[0], document.clauses[1], document.clauses[2]
    evidence = [{"chunkId": f"{c1.id}-c0", "quote": c1.text}]
    blocked = example_id == "aggregate"
    questions = {
        "aggregate": ["需要支持24小时累计统计的模板；当前单笔模板不能表达。"],
        "missing-unit": ["请确认原文金额的单位。"],
        "missing-version": ["请确认下发名单的版本。"],
    }.get(example_id, [])
    rules = [] if blocked else [{
        "key": "REG_001",
        "name": f"名单客户金额监控 · {ex['name']}",
        "templateId": "single-transaction",
        "start": "2026-11-01T00:00:00+08:00",
        "end": None,
        "currency": "CNY",
        "scopeEvidence": evidence,
        "conditions": [
            {"fieldId": "customer", "operator": "in_list", "value": None, "unit": None,
             "listId": "risk-2026-11", "evidence": evidence},
            {"fieldId": "amount", "operator": ex["operator"], "value": ex["value"], "unit": ex["unit"],
             "listId": None, "evidence": evidence},
            {"fieldId": "direction", "operator": "eq", "value": "OUT", "unit": None,
             "listId": None, "evidence": evidence},
        ],
    }]
    return {
        "summary": "根据示例通知，将名单客户的单笔金额及付款方向映射成平台禁用规则草稿，报送流程仍由人工负责。",
        "rules": rules,
        "coverage": [
            {"clauseId": c0.id, "disposition": "not_applicable", "reason": "学习材料标识，不是监控要求。", "ruleKeys": []},
            {"clauseId": c1.id, "disposition": "blocked" if blocked else "rule",
             "reason": "原文要求24小时累计，单笔交易模板无法落实，未偷偷简化。" if blocked else "生成金额、名单、方向及生效范围条件。",
             "ruleKeys": [] if blocked else ["REG_001"]},
            {"clauseId": c2.id, "disposition": "manual", "reason": "分析人员负责复核和另行报送审批，本工具不连接报送系统。", "ruleKeys": []},
        ],
        "questions": questions,
    }
