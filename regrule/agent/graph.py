"""用 LangGraph 编排的 Agent 工作流。

                 ┌──────────────── 校验不过 / 检索完 ────────────────┐
                 ▼                                                   │
  START → prepare → agent ──(模型要调用工具)──► tools ───────────────┤
                      │                           │                   │
                      │ 预算用完                   ├─ ask_human ──► clarify（暂停等人回答）─► agent
                      ▼                           │
                     END                          └─ 校验通过/预算用完 ─► review（暂停等人复核）
                                                                          ├─ 编辑 → 重新校验 → review
                                                                          ├─ 退回给模型修改 → agent
                                                                          ├─ 驳回 → END
                                                                          └─ 批准 → export → END

关键概念：
- State（状态）：整张图共享的一份数据，每个节点读它、返回要更新的部分；
- Node（节点）：一个普通函数；Edge（边）：决定下一步去哪个节点；
- ToolNode：LangGraph 自带的“执行工具”节点；
- interrupt()：人工介入（Human-in-the-loop），图会暂停并把状态存进检查点，等页面上点了按钮再 resume；
- Checkpointer（SqliteSaver）：把每一步状态存进 SQLite，关掉页面再打开也能接着跑。
"""
from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .. import documents, retrieval, store
from ..config import EXPORT_DIR, MAX_CALLS_LIMIT
from ..llm import build_chat_model, provider_status
from ..platform import load_profile
from ..schemas import Hit
from ..validation import validate_draft
from .demo_model import ScriptedDemoModel
from .prompts import SYSTEM_PROMPT, context_message
from .tools import HANDOFF_CODES, TOOLS

def merge_dict(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


def last(a: Any, b: Any) -> Any:
    return b


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    workspace_id: str
    document_id: str
    goal: str
    mode: str                     # llm | demo
    example_id: Optional[str]
    max_calls: int
    llm_calls: int
    evidence: Annotated[dict, merge_dict]          # chunkId -> 命中片段
    retrievals: Annotated[list, operator.add]       # 检索日志
    call_log: Annotated[list, operator.add]         # 每次模型调用的记录
    notes: Annotated[list, operator.add]            # 给人看的过程提示
    draft: Annotated[Optional[dict], last]
    checks: Annotated[Optional[dict], last]
    draft_origin: Annotated[str, last]
    questions: Annotated[list, last]
    next: Annotated[str, last]
    status: str
    review: Optional[dict]
    export: Optional[dict]


# ---------------- 节点 ----------------
def prepare(state: AgentState) -> dict:
    """建立初始上下文：自动在 3 个范围各检索一次，再拼成第一条消息。"""
    doc = documents.get(state["document_id"])
    profile = load_profile()
    query = f"{state['goal']} 金额 名单 付款 单笔 生效 时间 平台 模板"[:1000]
    evidence, log = {}, []
    for scope in ("document", "platform", "existing"):
        hits = [h.model_dump() for h in retrieval.search(query, scope, doc.id, profile)]
        evidence.update({h["id"]: h for h in hits})
        log.append({"at": store.now(), "by": "prepare", "scope": scope, "query": query, "hits": [h["id"] for h in hits]})
    ctx = context_message(state["goal"], doc, profile, list(evidence.values()), state["max_calls"])
    return {
        "messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(ctx)],
        "evidence": evidence, "retrievals": log, "llm_calls": 0, "status": "running", "next": "agent",
        "notes": [f"初始检索完成：共 {len(evidence)} 个片段。"],
    }


def _model(state: AgentState, config: RunnableConfig):
    if state.get("mode") == "demo":
        return ScriptedDemoModel(example_id=state["example_id"], document_id=state["document_id"])
    # 测试时可以通过 config 注入一个假模型
    injected = (config.get("configurable") or {}).get("chat_model")
    model = injected or build_chat_model()
    from ..config import llm_settings

    choice = llm_settings().tool_choice or None
    return model.bind_tools(TOOLS, tool_choice=choice) if choice and choice != "auto" else model.bind_tools(TOOLS)


def agent(state: AgentState, config: RunnableConfig) -> dict:
    used, budget = state.get("llm_calls", 0), state.get("max_calls", 4)
    if used >= budget:
        if state.get("draft"):
            return {"next": "review", "status": "review", "notes": ["调用次数用完，转交人工复核（未自动批准）。"]}
        return {"next": "end", "status": "needs_input", "notes": ["调用次数用完，仍没有草稿。可以补充任务说明后重新运行。"]}
    started = time.time()
    entry = {"step": used + 1, "at": store.now()}
    try:
        response: AIMessage = _model(state, config).invoke(state["messages"])
    except Exception as exc:  # 网络错误、401、400 ……
        entry.update(duration_ms=int((time.time() - started) * 1000), error=f"{type(exc).__name__}: {exc}"[:500])
        return {"llm_calls": used + 1, "call_log": [entry], "next": "end", "status": "error",
                "notes": ["模型调用失败：" + entry["error"] + "（没有用预设答案代替）"]}
    usage = getattr(response, "usage_metadata", None) or {}
    entry.update(
        duration_ms=int((time.time() - started) * 1000),
        tools=[c["name"] for c in response.tool_calls],
        thought=str(response.content)[:300],
        usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")} if usage else None,
    )
    update: dict = {"messages": [response], "llm_calls": used + 1, "call_log": [entry]}
    if response.tool_calls:
        update["next"] = "tools"
    else:
        update["next"] = "agent"
        update["messages"].append(HumanMessage("你必须调用工具（检索、提问或提交草稿），不要只回复文字。"))
        update["notes"] = ["模型没有调用工具，已提醒它重新按工具方式回答。"]
    return update


def route_after_agent(state: AgentState) -> str:
    return {"tools": "tools", "review": "review", "end": END}.get(state.get("next", "agent"), "agent")


def route_after_tools(state: AgentState) -> str:
    """看模型这一轮调用了哪些工具，决定下一步。"""
    msgs = state["messages"]
    ai = next(m for m in reversed(msgs) if isinstance(m, AIMessage))
    names = [c["name"] for c in ai.tool_calls]
    if "ask_human" in names:
        return "clarify"
    if "handoff_to_human" in names:
        return "review" if state.get("draft") else "agent"
    if "submit_draft" in names:
        ids = {c["id"] for c in ai.tool_calls if c["name"] == "submit_draft"}
        failed = any(isinstance(m, ToolMessage) and m.tool_call_id in ids and m.status == "error" for m in msgs)
        remaining = state.get("max_calls", 4) - state.get("llm_calls", 0)
        codes = {i["code"] for i in (state.get("checks") or {}).get("issues", [])}
        if not failed and state.get("checks") and codes <= HANDOFF_CODES:
            return "review"  # 校验通过，或只剩“暂不能落实/待确认”这类必须人工判断的问题
        if remaining <= 0:
            return "review" if state.get("draft") else "agent"
    return "agent"


def clarify(state: AgentState) -> Command:
    """人工介入①：模型提问，图暂停，等人回答。"""
    answer = interrupt({"type": "clarify", "questions": state.get("questions", [])})
    text = (answer or {}).get("answer", "").strip() if isinstance(answer, dict) else str(answer or "").strip()
    if not text:
        return Command(goto=END, update={"status": "needs_input", "notes": ["人工没有补充信息，流程结束。"]})
    extra = min(2, MAX_CALLS_LIMIT)
    return Command(goto="agent", update={
        "messages": [HumanMessage(f"人工补充：{text}")], "questions": [], "status": "running",
        "max_calls": state.get("llm_calls", 0) + extra, "next": "agent",
        "notes": [f"人工已回答，追加 {extra} 次调用额度。"],
    })


def _digest(state: AgentState) -> str:
    doc = documents.get(state["document_id"])
    return store.sha256(json.dumps({
        "doc": doc.hash, "profile": load_profile().model_dump(), "goal": state.get("goal"),
        "draft": state.get("draft"), "checks": state.get("checks"), "origin": state.get("draft_origin"),
    }, ensure_ascii=False, sort_keys=True))


def review(state: AgentState) -> Command:
    """人工介入②：复核。可以编辑草稿、退回给模型、驳回或批准。"""
    decision = interrupt({"type": "review", "draft": state.get("draft"), "checks": state.get("checks")})
    action = decision.get("action")
    doc, profile = documents.get(state["document_id"]), load_profile()
    evidence = {k: Hit(**v) for k, v in state.get("evidence", {}).items()}

    if action == "edit":
        _, checks = validate_draft(decision["draft"], doc, profile, evidence, state.get("goal", ""))
        return Command(goto="review", update={
            "draft": decision["draft"], "checks": checks.model_dump(), "draft_origin": "manual", "status": "review",
            "notes": [f"人工修改了草稿，重新校验：{len(checks.issues)} 个错误。"],
        })
    if action == "reject":
        return Command(goto=END, update={"status": "rejected", "notes": [f"人工驳回：{decision.get('notes') or '无说明'}"]})
    if action == "revise":
        note = (decision.get("notes") or "").strip() or "请根据校验错误修订草稿。"
        return Command(goto="agent", update={
            "messages": [HumanMessage(f"人工复核意见：{note}")], "status": "running", "next": "agent",
            "max_calls": state.get("llm_calls", 0) + 2, "notes": ["人工退回给模型修改，追加 2 次调用额度。"],
        })
    if action == "approve":
        if not state.get("draft"):
            return Command(goto="review", update={"status": "review", "notes": ["没有草稿，不能批准。"]})
        _, checks = validate_draft(state["draft"], doc, profile, evidence, state.get("goal", ""))
        reviewer = (decision.get("reviewer") or "").strip()
        notes = (decision.get("notes") or "").strip()
        problem = None
        if checks.issues or not checks.compiled:
            problem = "草稿仍有校验错误，不能批准。"
        elif not reviewer:
            problem = "请填写复核人。"
        elif not decision.get("ack"):
            problem = "请勾选确认已核对原文、映射、名单版本、警告和人工事项。"
        elif checks.warnings and not notes:
            problem = "有警告时必须在复核说明里写明如何核对。"
        if problem:
            return Command(goto="review", update={"checks": checks.model_dump(), "status": "review", "notes": [problem]})
        rec = {"at": store.now(), "reviewer": reviewer, "notes": notes}
        new_state = {**state, "checks": checks.model_dump()}
        rec["digest"] = _digest(new_state)
        return Command(goto="export", update={"checks": checks.model_dump(), "review": rec, "status": "approved"})
    return Command(goto="review", update={"notes": [f"未知操作：{action}"]})


def export(state: AgentState) -> dict:
    """导出禁用状态的配置包（同一版本重复导出返回同一个包）。"""
    digest = _digest(state)
    if not state.get("review") or state["review"]["digest"] != digest:
        return {"status": "review", "notes": ["复核后内容有变化，需要重新复核。"]}
    existing = store.find_export(state["workspace_id"], digest)
    if existing:
        return {"export": existing, "status": "exported"}
    profile = load_profile()
    doc = documents.get(state["document_id"])
    bundle = {
        "id": store.new_id(), "workspaceId": state["workspace_id"], "digest": digest, "createdAt": store.now(),
        "status": "approved_disabled_draft", "platformMode": profile.mode, "profile": profile.model_dump(),
        "document": doc.model_dump(), "draft": state["draft"], "draftOrigin": state.get("draft_origin"),
        "checks": state["checks"], "review": state["review"], "retrievals": state.get("retrievals", []),
        "nativeRules": state["checks"]["compiled"],
        "notice": "这是人工复核的禁用配置草稿。未写入生产、未调用真实Java引擎、未执行名单服务、未报送。演示映射不可直接用于生产。",
    }
    store.put_export(bundle)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / f"{bundle['id']}.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    retrieval.index_approved(bundle, profile)
    return {"export": bundle, "status": "exported", "notes": ["已导出禁用配置包，并加入“已有规则”检索库。"]}


# ---------------- 组装 ----------------
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("prepare", prepare)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("clarify", clarify)
    g.add_node("review", review)
    g.add_node("export", export)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "review", "agent", END])
    g.add_conditional_edges("tools", route_after_tools, ["clarify", "review", "agent"])
    g.add_edge("export", END)
    return g.compile(checkpointer=checkpointer)


def start_input(workspace: dict, goal: str, mode: str, max_calls: int) -> dict:
    if mode == "llm" and not provider_status()["configured"]:
        raise RuntimeError(provider_status()["issue"])
    if mode == "demo" and not workspace.get("example_id"):
        raise RuntimeError("演示模式只能处理内置学习示例；处理其他文件请配置大模型。")
    return {
        "workspace_id": workspace["id"], "document_id": workspace["document_id"], "goal": goal.strip()[:4000],
        "mode": mode, "example_id": workspace.get("example_id"),
        "max_calls": max(1, min(int(max_calls), MAX_CALLS_LIMIT)),
    }
