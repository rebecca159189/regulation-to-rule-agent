"""确定性校验与编译（护栏 Guardrails）。

这里没有任何大模型调用，全是普通代码：
模型写的草稿可能出错，所以引用、比较符、金额换算、名单版本、时间、覆盖表都由程序逐项核对，
通过之后再由程序（而不是模型）拼出平台表达式。
"""
from __future__ import annotations

import json
import re
from datetime import datetime

from pydantic import ValidationError

from .schemas import BoundaryTest, Checks, Draft, Evidence, Hit, Issue, Profile, SourceDocument
from .store import sha256

OPERATORS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "=="}

# 顺序很重要：先匹配“大于等于/不低于”，再匹配“大于”
COMPARISON_WORDS = [
    ("gte", re.compile(r"大于等于|不少于|不低于|至少|at least|not less than|>=", re.I)),
    ("lte", re.compile(r"小于等于|不超过|不高于|至多|at most|not more than|<=", re.I)),
    ("gt", re.compile(r"大于|超过|高于|greater than|more than|exceeds|>", re.I)),
    ("lt", re.compile(r"小于|低于|不足|less than|below|<", re.I)),
    ("eq", re.compile(r"等于|equal to|==", re.I)),
]
UNSUPPORTED = re.compile(r"累计|近\d+小时|过去\d+小时|合计|任一条件|或者|cumulative|rolling window", re.I)
MONEY_RE = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d{1,2})?$")
TIME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):00(?:Z|[+-]\d{2}:\d{2})$")
MAX_SAFE = 2**53 - 2


def source_comparison(text: str) -> str | None:
    for op, pattern in COMPARISON_WORDS:
        if pattern.search(text):
            return op
    return None


def to_fen(value: str, unit: str) -> int:
    """精确换算成“分”。用整数运算，绝不用浮点数（0.1+0.2≠0.3 的坑）。"""
    if not MONEY_RE.match(value or ""):
        raise ValueError("金额必须为非负十进制字符串，最多两位小数，不接受科学计数或隐式舍入。")
    whole, _, frac = value.partition(".")
    if unit == "fen" and frac and int(frac):
        raise ValueError("分以下精度暂不支持，不会自动四舍五入。")
    fen = int(whole) * 100 + int(frac.ljust(2, "0")) if unit == "yuan" else int(whole)
    if fen > MAX_SAFE:
        raise ValueError("金额超出可安全表示的范围。")
    return fen


def native_money(fen: int, unit: str) -> str:
    return str(fen) if unit == "fen" else f"{fen // 100}.{fen % 100:02d}"


def valid_time(value: str | None) -> bool:
    if not value or not TIME_RE.match(value):
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def epoch_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def compare(a: int, b: int, op: str) -> bool:
    return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b, "eq": a == b}[op]


def _date_pattern(iso: str) -> re.Pattern:
    y, m, d = (int(x) for x in iso[:10].split("-"))
    return re.compile(rf"{y}(?:年|[-/])0?{m}(?:月|[-/])0?{d}(?:日)?")


def validate_draft(raw: dict, document: SourceDocument, profile: Profile,
                   evidence: dict[str, Hit], goal: str = "") -> tuple[Draft | None, Checks]:
    """返回 (解析后的草稿或 None, 校验结果)。evidence 是本任务实际检索到的全部片段。"""
    checks = Checks(coverage={"total": len(document.clauses), "assigned": 0, "blocked": 0, "manual": 0})

    def fail(code: str, path: str, message: str) -> None:
        checks.issues.append(Issue(code=code, path=path, message=message))

    # 1. 结构校验
    try:
        draft = Draft.model_validate(raw)
    except ValidationError as exc:
        for e in exc.errors()[:30]:
            fail("SCHEMA", ".".join(str(p) for p in e["loc"]), e["msg"])
        return None, checks

    checks.warnings.extend(document.warnings)
    if document.hash != sha256("\n".join(p["text"] for p in document.pages)):
        fail("SOURCE_CHANGED", "document", "原文件内容指纹不一致。")
    if profile.mode == "demo":
        checks.warnings.append("当前使用演示字段、名单和绑定ID，不是生产配置。需要用平台真实导出资料替换映射。")
    checks.warnings.append("本地结构、引文和边界检查不等于已在Java引擎加载。导出文件始终禁用，不会直接写库或报送。")

    def verify(evs: list[Evidence], path: str) -> str:
        """引文必须逐字来自本次实际检索到的本文件片段。"""
        quotes = []
        for i, e in enumerate(evs):
            hit = evidence.get(e.chunkId)
            if not hit or hit.kind != "document" or hit.owner != document.id or e.quote not in hit.text:
                fail("CITATION", f"{path}.{i}", "引文必须逐字来自本次实际检索到的监管文件片段，不能引用其他文件或平台资料代替原文。")
                continue
            quotes.append(e.quote)
        return "\n".join(quotes)

    # 2. 问题、规则编号、覆盖表
    if draft.questions:
        fail("OPEN_QUESTIONS", "questions", f"仍有{len(draft.questions)}项待确认问题。")
    keys = {r.key for r in draft.rules}
    if len(keys) != len(draft.rules):
        fail("DUPLICATE_KEY", "rules", "规则编号重复。")
    if not draft.rules:
        fail("NO_RULE", "rules", "没有可编译的监控规则。制度、报送等要求应留给人工处理，不能硬造交易规则。")

    clause_ids = {c.id for c in document.clauses}
    seen: set[str] = set()
    for cov in draft.coverage:
        if cov.clauseId not in clause_ids or cov.clauseId in seen:
            fail("COVERAGE", "coverage", "条款覆盖编号不存在或重复。")
        seen.add(cov.clauseId)
        if cov.disposition == "blocked":
            checks.coverage["blocked"] += 1
            fail("UNSUPPORTED", cov.clauseId, cov.reason)
        if cov.disposition == "manual":
            checks.coverage["manual"] += 1
        if cov.disposition == "rule" and (not cov.ruleKeys or any(k not in keys for k in cov.ruleKeys)):
            fail("COVERAGE", cov.clauseId, "监控条款没有关联有效的规则。")
        if cov.disposition != "rule" and cov.ruleKeys:
            fail("COVERAGE", cov.clauseId, "非规则条款不能声称已经由某条规则落实。")
    checks.coverage["assigned"] = sum(1 for c in document.clauses if c.id in seen)
    for c in document.clauses:
        if c.id not in seen:
            fail("MISSING_CLAUSE", c.id, "此段原文尚未说明如何处理，不能静默遗漏。")

    fields = {f.id: f for f in profile.fields}
    time_field = next((f for f in profile.fields if f.type == "timestamp_ms"), None)
    currency_field = fields.get("currency")
    b = profile.binding
    predicates: set[str] = set()

    # 3. 逐条规则
    for rule in draft.rules:
        path = f"rules.{rule.key}"
        before = len(checks.issues)
        scope_text = verify(rule.scopeEvidence, path + ".scopeEvidence")
        cond_text = "\n".join(e.quote for c in rule.conditions for e in c.evidence)
        if UNSUPPORTED.search(scope_text + cond_text):
            fail("UNSUPPORTED", path, "引用要求含累计窗口或非AND关系，当前单笔模板不能落实，需人工确认，不能直接简化。")
        if rule.templateId != "single-transaction" or rule.templateId not in {t.id for t in profile.templates}:
            fail("TEMPLATE", path, "模板不受本地适配器支持；累计窗口、OR、关联账户等不能简化成单笔比较。")
        if not (b.solutionId and b.channelId and b.transId and b.parentId) or b.ruleType is None:
            fail("BINDING", path, "决策方案、渠道、交易类型、规则组或规则类型绑定缺失。")
        if not any(n in scope_text for n in b.channelNames) or not any(n in scope_text for n in b.transactionNames):
            fail("SCOPE", path, "原文范围与选定渠道/交易类型无法核对，请补充依据或更换平台映射。")

        guards: list[str] = []
        # 生效时间
        if not rule.start or not valid_time(rule.start) or not time_field:
            fail("TIME", path + ".start", "需要明确的生效时点（带时区、分钟精度）与毫秒时间戳字段。")
        else:
            if not _date_pattern(rule.start).search(scope_text):
                fail("TIME_SOURCE", path, "生效日期没有在引用原文中找到。")
            clock = rule.start[11:16]
            if clock not in scope_text and clock.replace(":", "：") not in scope_text:
                fail("TIME_SOURCE", path, "生效时分没有在引用原文中找到，不能自行改成其他时间。")
            if "北京时间" in scope_text and not rule.start.endswith("+08:00"):
                fail("TIME_ZONE", path, "北京时间必须明确使用+08:00，不可静默改成UTC。")
            if not re.search(r"北京时间|UTC|[+-]\d{2}:\d{2}", scope_text + goal):
                fail("TIME_ZONE", path, "原文和人工目标中都没有明确时区，需要确认。")
            guards.append(f"{time_field.code} >= {epoch_ms(rule.start)}")
        if rule.end:
            if not valid_time(rule.end) or not rule.start or not valid_time(rule.start) or epoch_ms(rule.end) <= epoch_ms(rule.start):
                fail("TIME", path + ".end", "结束必须晚于开始，并带时区。")
            elif time_field:
                guards.append(f"{time_field.code} < {epoch_ms(rule.end)}")
            if not re.search(r"不含|左闭右开|exclusive", scope_text, re.I):
                fail("TIME_END", path, "结束时点是否包含当天尚不明确，不能自行选成左闭右开。")
            if valid_time(rule.end):
                if not _date_pattern(rule.end).search(scope_text):
                    fail("TIME_SOURCE", path + ".end", "结束日期没有原文依据，不能自行延长或缩短有效期。")
                end_clock = rule.end[11:16]
                if end_clock not in scope_text and end_clock.replace(":", "：") not in scope_text:
                    fail("TIME_SOURCE", path + ".end", "结束时分没有引用依据。")
                if "北京时间" in scope_text and not rule.end.endswith("+08:00"):
                    fail("TIME_ZONE", path + ".end", "结束时点也必须使用明确的北京时间+08:00。")
        elif len(re.findall(r"\d{4}(?:年|[-/])\d{1,2}(?:月|[-/])\d{1,2}", scope_text)) > 1:
            fail("TIME_END", path, "引用原文出现多个日期但没有明确结束时点，需要核对是否遗漏有效期上限。")
        # 币种
        if rule.currency:
            aliases = (currency_field.values or {}).get(rule.currency) if currency_field else None
            if not aliases or not any(a in scope_text for a in aliases):
                fail("CURRENCY", path, "币种没有平台映射或原文依据，不会自行换汇。")
            else:
                guards.append(f"{currency_field.code} == {json.dumps(rule.currency)}")

        # 条件
        expressions: list[str] = []
        for i, cond in enumerate(rule.conditions):
            at = f"{path}.conditions.{i}"
            quote = verify(cond.evidence, at + ".evidence")
            field = fields.get(cond.fieldId)
            if not field:
                fail("FIELD", at, "字段没有出现在选定版本的平台字典中。")
                continue
            op = cond.operator
            if op == "in_list":
                lst = next((l for l in profile.lists if l.id == cond.listId and l.fieldId == field.id), None)
                if not lst or field.type != "string" or cond.value is not None or cond.unit is not None:
                    fail("LIST", at, "名单绑定、版本或条件结构无效。")
                    continue
                if lst.name not in quote or lst.version not in quote:
                    fail("LIST_SOURCE", at, "引文没有同时明确名单名称与版本，不能猜测当前名单。")
                expressions.append(f"func.blacklist({json.dumps(lst.dbType)},{json.dumps(lst.type)},{field.code})")
            elif field.type == "money":
                if cond.value is None or not cond.unit or not field.unit or cond.listId is not None or not rule.currency:
                    fail("MONEY", at, "金额需要原文数值、单位、平台单位和币种，不允许缺项。")
                    continue
                try:
                    src_op = source_comparison(quote)
                    if not src_op or src_op != op:
                        fail("OPERATOR_SOURCE", at, "比较符与原文口径不一致或无法判定（大于不等于大于等于）。")
                    unit_word = "元" if cond.unit == "yuan" else "分"
                    m = re.search(rf"(?<![\d.-])(\d+(?:\.\d{{1,2}})?)\s*{unit_word}", quote.replace(",", ""))
                    if not m or to_fen(m.group(1), cond.unit) != to_fen(cond.value, cond.unit):
                        fail("MONEY_SOURCE", at, "金额或单位与引文不一致；万元、中文数词等暂需人工明确转换。")
                    fen = to_fen(cond.value, cond.unit)
                    expressions.append(f"{field.code} {OPERATORS[op]} {native_money(fen, field.unit)}")
                    if src_op:
                        for delta, label in ((-1, "下方1分"), (0, "相等"), (1, "上方1分")):
                            checks.boundaryTests.append(BoundaryTest(
                                rule=rule.key, name=f"{field.name}阈值{label}",
                                expected=compare(fen + delta, fen, src_op),
                                actual=compare(fen + delta, fen, op),
                            ))
                except (ValueError, KeyError) as exc:
                    fail("MONEY", at, str(exc))
            elif field.type == "string" and op == "eq" and cond.value is not None and cond.unit is None and cond.listId is None:
                aliases = (field.values or {}).get(cond.value)
                if not aliases or not any(a in quote for a in aliases):
                    fail("ENUM_SOURCE", at, "字符串枚举值没有字典映射或引文依据。")
                expressions.append(f"{field.code} == {json.dumps(cond.value)}")
            else:
                fail("CONDITION", at, "条件类型不受支持；不接收LLM自由编写的表达式。")

        cited = {evidence[e.chunkId].clause_id for e in [*rule.scopeEvidence, *(e for c in rule.conditions for e in c.evidence)]
                 if e.chunkId in evidence}
        covered = [c for c in draft.coverage if c.disposition == "rule" and rule.key in c.ruleKeys]
        if not covered or any(c.clauseId not in cited for c in covered):
            fail("RULE_COVERAGE", path, "规则与条款覆盖表、引文没有对应起来。")

        # 去重：模型可能把“币种”既写在 currency 属性又写成条件
        parts = list(dict.fromkeys(guards + expressions))
        expression = " && ".join(f"({p})" for p in parts)
        fp = sha256(expression)
        if fp in predicates:
            fail("DUPLICATE", path, "本批次生成了完全相同的规则条件。")
        predicates.add(fp)
        if any(re.sub(r"\s", "", r.expression) == re.sub(r"\s", "", expression) for r in profile.referenceRules):
            checks.warnings.append(f"{rule.name}与导入的已有规则表达式相同，复核时需确认是否应复用而不是重复配置。")
        related = [h for h in evidence.values() if h.kind == "existing"]
        if related:
            checks.warnings.append(f"{rule.name}检索到{len(related)}份已有规则参考；相似不等于重复或冲突，需人工核对。")

        if len(checks.issues) == before:
            checks.compiled.append({
                "key": rule.key,
                "table": "trans_rule",
                "values": {
                    "rule_name": rule.name,
                    "solution_id": b.solutionId,
                    "channel_id": b.channelId,
                    "trans_id": b.transId,
                    "parent_id": b.parentId,
                    "rule_type": b.ruleType,
                    "expression_en": expression,
                    "expression_cn": draft.summary,
                    "rule_condition_en": "true",
                    "is_enabled": 0,
                    "stat_codes": "",
                },
            })

    if any(t.actual != t.expected for t in checks.boundaryTests):
        fail("BOUNDARY", "rules", "比较符边界与引文不符。")
    checks.warnings = list(dict.fromkeys(checks.warnings))
    return draft, checks


def issues_text(checks: Checks, limit: int = 20) -> str:
    return "\n".join(f"- [{i.code}] {i.path}: {i.message}" for i in checks.issues[:limit]) or "无"
