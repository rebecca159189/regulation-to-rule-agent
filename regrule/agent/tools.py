"""Agent 的工具（Tools）。

大模型不能直接动数据，只能“申请调用”这里定义的函数；真正执行的是我们的代码。
每个工具返回 LangGraph 的 Command：既给模型回一条 ToolMessage，也顺便更新图的状态
（比如把检索命中的片段记进 evidence、把校验结果记进 checks）。

InjectedState / InjectedToolCallId 是“注入参数”：模型看不到也填不了，由框架自动传入。
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .. import documents, retrieval
from ..config import MAX_RETRIEVALS
from ..platform import load_profile
from ..schemas import Draft, Hit
from ..store import now
from ..validation import issues_text, validate_draft
from .prompts import format_hits


def _search(scope: str, query: str, state: dict, tool_call_id: str) -> Command:
    if len(state.get("retrievals", [])) >= MAX_RETRIEVALS:
        msg = f"本任务已达 {MAX_RETRIEVALS} 次检索上限，请根据已有证据提交草稿或提问。"
        return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id, status="error")]})
    hits = [h.model_dump() for h in retrieval.search(query, scope, state["document_id"], load_profile())]
    return Command(update={
        "evidence": {h["id"]: h for h in hits},
        "retrievals": [{"at": now(), "by": "agent", "scope": scope, "query": query, "hits": [h["id"] for h in hits]}],
        "messages": [ToolMessage(format_hits(hits), tool_call_id=tool_call_id)],
    })


@tool
def search_regulation(query: str, state: Annotated[dict, InjectedState],
                      tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """检索本次监管文件的原文片段。query 可以是关键词，也可以直接填条款编号（如 xxx-p1-s2）取整段原文。"""
    return _search("document", query, state, tool_call_id)


@tool
def search_platform(query: str, state: Annotated[dict, InjectedState],
                    tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """检索平台资料：字段字典（代码、单位、枚举值）、名单及版本、规则模板、绑定信息。"""
    return _search("platform", query, state, tool_call_id)


@tool
def search_existing_rules(query: str, state: Annotated[dict, InjectedState],
                          tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """检索已有规则（导入的参考规则和以前复核导出的规则），用于查重。已有规则不是新的监管要求。"""
    return _search("existing", query, state, tool_call_id)


@tool
def ask_human(questions: list[str], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """信息不足、不能猜的时候向人工提问（如金额单位、名单版本、时区缺失）。调用后流程暂停，等人回答。"""
    questions = [q.strip() for q in questions if q and q.strip()][:20] or ["请补充缺失信息。"]
    return Command(update={
        "questions": questions,
        "messages": [ToolMessage("已向人工提问，等待回答。", tool_call_id=tool_call_id)],
    })


HANDOFF_CODES = {"UNSUPPORTED", "OPEN_QUESTIONS", "NO_RULE"}


@tool
def handoff_to_human(reason: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """当你确认自己无法再改进草稿时（例如剩下的问题只能人工判断或需要人工改字段），把当前草稿交给人工复核。"""
    return Command(update={
        "notes": [f"模型请求转交人工：{reason[:300]}"],
        "messages": [ToolMessage("已转交人工复核。", tool_call_id=tool_call_id)],
    })


@tool
def submit_draft(draft: Draft, state: Annotated[dict, InjectedState],
                 tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """提交规则草稿。程序会立即做确定性校验（引文、比较符、金额单位、名单版本、时间、覆盖表）并返回结果。"""
    document = documents.get(state["document_id"])
    evidence = {k: Hit(**v) for k, v in state.get("evidence", {}).items()}
    _, checks = validate_draft(draft.model_dump(), document, load_profile(), evidence, state.get("goal", ""))
    remaining = state.get("max_calls", 4) - state.get("llm_calls", 0)
    ok = not checks.issues
    codes = {i.code for i in checks.issues}
    if ok:
        reply = "校验通过，已转交人工复核。"
    elif codes <= HANDOFF_CODES:
        reply = "草稿声明了暂不能落实或待确认的事项，程序校验不会放行，已转交人工复核：\n" + issues_text(checks)
    elif remaining <= 0:
        reply = "校验未通过，且调用次数已用完，转交人工处理：\n" + issues_text(checks)
    else:
        reply = (f"校验未通过（剩余 {remaining} 次调用）。请逐条修正后重新 submit_draft，"
                 "不能删掉原文要求来通过校验；如果是原文缺信息，请改用 ask_human：\n" + issues_text(checks))
    return Command(update={
        "draft": draft.model_dump(),
        "checks": checks.model_dump(),
        "draft_origin": "demo_preset" if state.get("mode") == "demo" else "llm",
        "messages": [ToolMessage(reply, tool_call_id=tool_call_id)],
    })


TOOLS = [search_regulation, search_platform, search_existing_rules, ask_human, submit_draft, handoff_to_human]
