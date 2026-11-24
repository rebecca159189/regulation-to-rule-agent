"""数据结构（相当于 Java 里的 DTO + 参数校验）。

Pydantic 模型有两个用途：
1. 作为工具 submit_draft 的参数结构，LangChain 会把它转成 JSON Schema 发给大模型，
   模型就知道该按什么格式填；
2. 模型返回后用 model_validate 校验，格式不对就把错误反馈给模型重写。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Operator = Literal["gt", "gte", "lt", "lte", "eq", "in_list"]
Unit = Literal["yuan", "fen"]
Scope = Literal["document", "platform", "existing"]
Disposition = Literal["rule", "manual", "not_applicable", "blocked"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- 模型输出：规则草稿 ----------
class Evidence(Strict):
    chunkId: str = Field(min_length=1, max_length=1000, description="已检索到的本次文件片段编号，如 xxx-p1-s2-c0")
    quote: str = Field(min_length=1, max_length=1800, description="从该片段中逐字摘抄的原文，不能改写")


class Condition(Strict):
    fieldId: str = Field(min_length=1, max_length=100, description="平台字段 id，如 amount、customer、direction")
    operator: Operator = Field(description="gt 大于 / gte 大于等于 / lt 小于 / lte 小于等于 / eq 等于 / in_list 属于名单")
    value: Optional[str] = Field(default=None, max_length=200, description="金额保留原文十进制字符串（如 \"3000\"，不要换算）；字符串字段填枚举代码（如 OUT）；名单条件填 null")
    unit: Optional[Unit] = Field(default=None, description="金额的原文单位：yuan 元 / fen 分；非金额填 null")
    listId: Optional[str] = Field(default=None, max_length=100, description="in_list 时填平台名单 id，否则 null")
    evidence: list[Evidence] = Field(min_length=1, max_length=5)


class Rule(Strict):
    key: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$", description="规则编号，如 REG_001")
    name: str = Field(min_length=1, max_length=200)
    templateId: str = Field(min_length=1, max_length=100, description="平台模板 id，目前只有 single-transaction")
    start: Optional[str] = Field(default=None, max_length=60, description="生效时点，形如 2026-11-01T00:00:00+08:00")
    end: Optional[str] = Field(default=None, max_length=60, description="结束时点（左闭右开），原文没有则 null")
    currency: Optional[str] = Field(default=None, max_length=10, description="币种代码，如 CNY")
    scopeEvidence: list[Evidence] = Field(min_length=1, max_length=6, description="渠道、交易类型、生效时间、币种的原文依据")
    conditions: list[Condition] = Field(min_length=1, max_length=12)


class Coverage(Strict):
    clauseId: str = Field(min_length=1, max_length=200, description="条款编号，如 xxx-p1-s2")
    disposition: Disposition = Field(description="rule 由规则落实 / manual 人工处理 / not_applicable 非监控要求 / blocked 暂不能落实")
    reason: str = Field(min_length=1, max_length=1000)
    ruleKeys: list[str] = Field(default_factory=list, max_length=20)


class Draft(Strict):
    summary: str = Field(min_length=1, max_length=3000)
    rules: list[Rule] = Field(default_factory=list, max_length=20)
    coverage: list[Coverage] = Field(default_factory=list, max_length=300, description="文件每一段都要有一项")
    questions: list[str] = Field(default_factory=list, max_length=30, description="仍需人工确认的问题；有问题时不能导出")


# ---------- 平台资料 ----------
class Binding(Strict):
    solutionId: Optional[str] = None
    channelId: Optional[str] = None
    transId: Optional[str] = None
    parentId: Optional[str] = None
    ruleType: Optional[int] = None
    channelNames: list[str]
    transactionNames: list[str]


class PlatformField(Strict):
    id: str
    code: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
    name: str
    type: Literal["money", "string", "timestamp_ms"]
    unit: Optional[Unit] = None
    aliases: list[str] = Field(default_factory=list)
    values: Optional[dict[str, list[str]]] = None


class PlatformList(Strict):
    id: str
    name: str
    version: str
    dbType: str
    type: str
    fieldId: str


class Template(Strict):
    id: str
    name: str
    description: str


class ReferenceRule(Strict):
    id: str
    name: str
    summary: str
    expression: str


class Profile(Strict):
    id: str = Field(pattern=r"^[a-z0-9_-]{1,60}$")
    name: str
    version: str
    mode: Literal["demo", "configured"]
    binding: Binding
    fields: list[PlatformField]
    lists: list[PlatformList] = Field(default_factory=list)
    templates: list[Template]
    referenceRules: list[ReferenceRule] = Field(default_factory=list)


# ---------- 文档与检索 ----------
class Clause(BaseModel):
    id: str
    page: int
    paragraph: int
    text: str


class SourceDocument(BaseModel):
    id: str
    name: str
    parser: str
    hash: str
    created_at: str
    pages: list[dict]
    clauses: list[Clause]
    warnings: list[str] = Field(default_factory=list)


class Hit(BaseModel):
    id: str
    owner: str
    kind: Scope
    label: str
    text: str
    clause_id: Optional[str] = None
    score: float = 0.0
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None


# ---------- 校验结果 ----------
class Issue(BaseModel):
    code: str
    path: str
    message: str


class BoundaryTest(BaseModel):
    rule: str
    name: str
    expected: bool
    actual: bool


class Checks(BaseModel):
    issues: list[Issue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    compiled: list[dict] = Field(default_factory=list)
    boundaryTests: list[BoundaryTest] = Field(default_factory=list)
    coverage: dict = Field(default_factory=dict)
