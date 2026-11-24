"""平台资料：规则能用哪些字段、名单、模板和绑定 ID。

内置的是演示映射（通用的示例字段和表结构），不是任何生产系统的配置。
以后换成真实平台资料时，改 data/platform_profile.json 或在页面导入即可。
"""
from __future__ import annotations

import hashlib
import json

from .config import ROOT
from .schemas import Profile

PROFILE_FILE = ROOT / "data" / "platform_profile.json"

DEMO_PROFILE = {
    "id": "engine-demo",
    "name": "示例规则引擎 · 演示映射",
    "version": "1",
    "mode": "demo",
    "binding": {
        "solutionId": "1001",
        "channelId": "10",
        "transId": "20",
        "parentId": "30",
        "ruleType": 0,
        "channelNames": ["手机银行"],
        "transactionNames": ["转账"],
    },
    "fields": [
        {"id": "customer", "code": "CUST_ID", "name": "客户编号", "type": "string", "unit": None,
         "aliases": ["客户", "风险客户", "名单客户"], "values": None},
        {"id": "amount", "code": "TRANS_AMT", "name": "单笔交易金额", "type": "money", "unit": "fen",
         "aliases": ["金额", "单笔金额"], "values": None},
        {"id": "currency", "code": "CURRENCY", "name": "币种", "type": "string", "unit": None,
         "aliases": ["人民币", "CNY"], "values": {"CNY": ["人民币", "CNY"]}},
        {"id": "time", "code": "TRANS_TIME_MS", "name": "交易发生时间", "type": "timestamp_ms", "unit": None,
         "aliases": ["交易时间", "生效时间", "北京时间"], "values": None},
        {"id": "direction", "code": "PAY_DIRECTION", "name": "交易方向", "type": "string", "unit": None,
         "aliases": ["付款", "转出", "收款"], "values": {"OUT": ["付款", "转出"], "IN": ["收款", "转入"]}},
    ],
    "lists": [
        {"id": "risk-2026-11", "name": "风险客户名单A", "version": "2026-11",
         "dbType": "DEMO_RISK_202611", "type": "1", "fieldId": "customer"},
    ],
    "templates": [
        {"id": "single-transaction", "name": "单笔交易条件监控",
         "description": "支持AND组合：金额gt/gte/lt/lte/eq、字符串eq、版本化名单in_list和带时区生效期间。"
                        "不支持累计窗口、跨账户关系、OR或自动报送。"},
    ],
    "referenceRules": [],
}

RESERVED = {"func", "true", "false", "null", "undefined", "__proto__", "constructor", "prototype"}


def validate_profile(data: dict) -> Profile:
    profile = Profile.model_validate(data)
    for key in ("fields", "lists", "templates"):
        ids = [item.id for item in getattr(profile, key)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{key} 编号不能重复。")
    codes = [f.code for f in profile.fields]
    if len(set(codes)) != len(codes):
        raise ValueError("平台字段代码不能重复。")
    if any(c in RESERVED for c in codes):
        raise ValueError("字段代码不能使用保留名称。")
    for item in profile.lists:
        if not any(f.id == item.fieldId and f.type == "string" for f in profile.fields):
            raise ValueError("名单需要绑定一个字符串类型的标识字段。")
    return profile


def load_profile() -> Profile:
    if PROFILE_FILE.exists():
        return validate_profile(json.loads(PROFILE_FILE.read_text(encoding="utf-8")))
    return validate_profile(DEMO_PROFILE)


def save_profile(data: dict) -> Profile:
    profile = validate_profile(data)
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(profile.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def fingerprint(profile: Profile) -> str:
    return hashlib.sha256(json.dumps(profile.model_dump(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def platform_owner(profile: Profile) -> str:
    return f"platform-{profile.id}-{fingerprint(profile)}"


def existing_owner(profile: Profile) -> str:
    return f"existing-{fingerprint(profile)}"


def platform_facts(profile: Profile) -> list[dict]:
    """把平台资料拆成一条条“知识”，放进 platform 检索范围。"""
    dump = lambda m: json.dumps(m.model_dump(), ensure_ascii=False)
    facts = [{
        "id": "binding",
        "label": "平台绑定与适配边界",
        "text": dump(profile.binding) + f"；mode={profile.mode}。演示映射不是生产字段。导出trans_rule禁用草稿，不执行SQL，不连接生产。",
    }]
    facts += [{"id": f"field-{f.id}", "label": f"字段：{f.name}", "text": dump(f)} for f in profile.fields]
    facts += [{
        "id": f"list-{l.id}", "label": f"名单：{l.name} {l.version}",
        "text": dump(l) + "；名单条件编译为 func.blacklist(dbtype,type,field) 三参数形式；名单内容与版本需人工核对。",
    } for l in profile.lists]
    facts += [{"id": f"template-{t.id}", "label": t.name, "text": dump(t)} for t in profile.templates]
    facts.append({
        "id": "adapter", "label": "引擎配置结构依据",
        "text": "示例规则表 trans_rule 包含 solution_id、channel_id、trans_id、rule_name、"
                "expression_en、expression_cn、rule_condition_en 及 is_enabled；规则引擎解析 expression_en 执行。"
                "只编译白名单结构，不让 LLM 直接写表达式。",
    })
    owner = platform_owner(profile)
    return [{**f, "id": f"{owner}-{f['id']}", "owner": owner, "kind": "platform", "clause_id": None} for f in facts]


def reference_facts(profile: Profile) -> list[dict]:
    owner = existing_owner(profile)
    return [{
        "id": f"{owner}-ref-{hashlib.sha256(r.id.encode()).hexdigest()[:12]}",
        "owner": owner, "kind": "existing", "label": f"{r.name} · {r.id}",
        "text": json.dumps(r.model_dump(), ensure_ascii=False), "clause_id": None,
    } for r in profile.referenceRules]
