"""Streamlit 页面。启动：streamlit run app.py

Streamlit 的运行方式：用户每点一下，整个脚本从上到下重新执行一遍，
需要跨点击保存的东西放在 st.session_state 里；Agent 的运行状态存在 LangGraph 检查点里。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from regrule import fixtures, retrieval, service, store
from regrule.config import DEFAULT_MAX_CALLS, MAX_CALLS_LIMIT
from regrule.documents import get as get_document
from regrule.embeddings import get_embeddings
from regrule.llm import provider_status
from regrule.platform import load_profile, save_profile

st.set_page_config(page_title="监管规则助手", page_icon="🛡️", layout="wide")

STATUS_TEXT = {
    "running": "运行中", "needs_input": "待补充信息", "review": "待人工复核", "approved": "已批准",
    "rejected": "已驳回", "exported": "已导出", "error": "调用出错", None: "未运行",
}
ORIGIN_TEXT = {"llm": "大模型", "demo_preset": "预置答案", "manual": "人工修改", None: "-"}


# ---------------- 侧边栏 ----------------
def sidebar() -> tuple[str, int]:
    st.sidebar.title("🛡️ 监管规则助手")
    st.sidebar.caption("LangGraph Agent + 混合检索 RAG · 本地原型")
    llm = provider_status()
    if llm["configured"]:
        st.sidebar.success(f"大模型：{llm['model']}")
    else:
        st.sidebar.warning("大模型未配置，只能用演示模式。配置方法见 README。")
    with st.sidebar:
        with st.spinner("加载嵌入模型（首次需要下载，约 1 分钟）…"):
            emb, _, emb_status = get_embeddings()
    (st.sidebar.info if emb else st.sidebar.warning)(emb_status)

    st.sidebar.divider()
    st.sidebar.subheader("任务")
    rows = store.list_workspaces()
    options = ["＋ 新建任务"] + [r["id"] for r in rows]
    labels = {r["id"]: f"{r['name']} · {r['created_at'][5:16].replace('T', ' ')}" for r in rows}
    current = st.session_state.get("ws_id")
    index = options.index(current) if current in options else 0
    choice = st.sidebar.selectbox("选择任务", options, index=index, format_func=lambda x: labels.get(x, x))
    st.session_state["ws_id"] = None if choice == "＋ 新建任务" else choice

    st.sidebar.divider()
    st.sidebar.subheader("运行设置")
    modes = ["llm", "demo"]
    mode = st.sidebar.radio(
        "Agent 模式", modes, index=0 if llm["configured"] else 1,
        format_func=lambda m: "真实大模型" if m == "llm" else "演示模式（预置答案，非模型）",
    )
    max_calls = st.sidebar.slider("调用上限（最多问模型几次）", 1, MAX_CALLS_LIMIT, DEFAULT_MAX_CALLS)
    with st.sidebar.expander("Agent 工作流图（Mermaid）"):
        st.code(service.mermaid(), language="text")
        st.caption("复制到 mermaid.live 可以看到图形。")
    return mode, max_calls


# ---------------- 新建任务 ----------------
def new_task() -> None:
    st.header("新建任务")
    t1, t2, t3 = st.tabs(["学习示例", "上传文件", "粘贴文字"])
    with t1:
        ex_id = st.selectbox("选择示例", [e["id"] for e in fixtures.EXAMPLES],
                             format_func=lambda i: fixtures.example(i)["name"])
        st.text(fixtures.example_text(ex_id))
        if st.button("载入示例", type="primary"):
            _open(service.create_from_example(ex_id))
    with t2:
        up = st.file_uploader("PDF / DOCX / TXT / MD，≤ 8MB", type=["pdf", "docx", "txt", "md"])
        if up and st.button("读取文件", type="primary"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / up.name
                path.write_bytes(up.getvalue())
                _open(service.create_from_file(up.name, path))
    with t3:
        name = st.text_input("名称", "粘贴的通知")
        text = st.text_area("通知全文", height=200)
        if st.button("读取文字", type="primary") and text.strip():
            _open(service.create_from_text(name, text))


def _open(ws: dict) -> None:
    st.session_state["ws_id"] = ws["id"]
    st.rerun()


# ---------------- 运行过程展示 ----------------
def describe(update: dict) -> str:
    node, data = next(iter(update.items()))
    if node == "__interrupt__":
        value = data[0].value
        return "⏸️ 暂停：" + ("等待人工回答问题" if value["type"] == "clarify" else "等待人工复核")
    data = data or {}
    if node == "prepare":
        return f"📚 prepare：初始检索，得到 {len(data.get('evidence', {}))} 个片段"
    if node == "agent":
        log = (data.get("call_log") or [{}])[-1]
        if log.get("error"):
            return f"❌ agent：模型调用失败 {log['error']}"
        if log.get("tools"):
            return f"🤖 agent（第 {log['step']} 次调用，{log['duration_ms']} ms）：决定调用 {', '.join(log['tools'])}"
        return "🤖 agent：" + "；".join(data.get("notes", []))
    if node == "tools":
        if "checks" in data and data["checks"]:
            n = len(data["checks"]["issues"])
            return f"🛠️ tools：提交草稿，程序校验 {'通过 ✅' if n == 0 else f'发现 {n} 个问题'}"
        if data.get("questions"):
            return "🛠️ tools：模型提问 " + "；".join(data["questions"])
        if data.get("retrievals"):
            r = data["retrievals"][-1]
            return f"🛠️ tools：检索 [{r['scope']}] “{r['query'][:30]}” → {len(r['hits'])} 条"
        return "🛠️ tools：" + "；".join(data.get("notes", []) or ["已执行"])
    return f"➡️ {node}：" + "；".join((data or {}).get("notes", []) or ["完成"])


def drive(stream) -> None:
    with st.status("Agent 运行中…", expanded=True) as box:
        try:
            for update in stream:
                box.write(describe(update))
            box.update(label="本轮结束", state="complete")
        except Exception as exc:
            box.update(label="出错", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return
    st.rerun()


# ---------------- 工作台 ----------------
def workspace(mode: str, max_calls: int) -> None:
    ws = store.get_workspace(st.session_state["ws_id"])
    doc = get_document(ws["document_id"])
    snap = service.snapshot(ws["id"])
    state = snap.values or {}
    pending = service.pending_interrupt(ws["id"])
    status = {"clarify": "needs_input", "review": "review"}.get((pending or {}).get("type"), state.get("status"))

    st.header(doc.name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("状态", STATUS_TEXT.get(status, status))
    c2.metric("模型调用", f"{state.get('llm_calls', 0)} / {state.get('max_calls', '-')}")
    c3.metric("已检索片段", len(state.get("evidence", {})))
    c4.metric("草稿来源", ORIGIN_TEXT.get(state.get("draft_origin")))
    for w in doc.warnings:
        st.warning(w)

    tabs = st.tabs(["① 原文", "② 检索调试（RAG）", "③ Agent 运行", "④ 复核与导出", "⑤ 平台资料"])
    with tabs[0]:
        st.dataframe(pd.DataFrame([{"条款编号": c.id.split("-", 1)[1], "页": c.page, "段": c.paragraph, "原文": c.text}
                                   for c in doc.clauses]), hide_index=True, width="stretch")
    with tabs[1]:
        retrieval_tab(doc)
    with tabs[2]:
        run_tab(ws, state, pending, mode, max_calls)
    with tabs[3]:
        review_tab(ws, state, pending)
    with tabs[4]:
        profile_tab()


def retrieval_tab(doc) -> None:
    st.caption("BM25 关键词检索 + 向量语义检索，用 RRF（倒数排名融合）合并。表里能看到每一路各排第几。")
    c1, c2 = st.columns([1, 3])
    scope = c1.selectbox("范围", ["document", "platform", "existing"],
                         format_func={"document": "本次文件条款", "platform": "平台资料", "existing": "已有规则"}.get)
    query = c2.text_input("检索词", "单笔 金额 付款 名单")
    if st.button("检索"):
        hits = retrieval.search(query, scope, doc.id, load_profile())
        if not hits:
            st.info("没有命中。")
        st.dataframe(pd.DataFrame([{"RRF分数": h.score, "BM25名次": h.bm25_rank, "向量名次": h.vector_rank,
                                    "标签": h.label, "内容": h.text} for h in hits]),
                     hide_index=True, width="stretch")


def run_tab(ws: dict, state: dict, pending: dict | None, mode: str, max_calls: int) -> None:
    goal = st.text_area("任务说明（会发给模型）", state.get("goal") or fixtures.DEFAULT_GOAL, height=80)
    if mode == "demo":
        st.info("演示模式：不调用大模型，按固定剧本调用工具并提交 fixtures.py 里人工写的标准答案，只用来体验流程。")
    if st.button("▶️ 从头运行 Agent", type="primary", disabled=bool(pending and pending["type"] == "clarify")):
        drive(service.run(ws["id"], goal, mode, max_calls))

    if pending and pending["type"] == "clarify":
        st.subheader("🙋 模型需要你补充信息")
        for q in pending["questions"]:
            st.markdown(f"- {q}")
        answer = st.text_area("你的回答（留空提交 = 结束流程）")
        if st.button("提交回答，继续运行"):
            drive(service.resume(ws["id"], {"answer": answer}))

    if not state:
        return
    st.subheader("过程记录")
    for note in state.get("notes", []):
        st.markdown(f"- {note}")
    calls = state.get("call_log", [])
    if calls:
        st.markdown("**每次模型调用**")
        st.dataframe(pd.DataFrame([{
            "第几次": c.get("step"), "调用的工具": ", ".join(c.get("tools") or []), "耗时ms": c.get("duration_ms"),
            "输入tokens": (c.get("usage") or {}).get("input_tokens"), "输出tokens": (c.get("usage") or {}).get("output_tokens"),
            "模型说明": c.get("thought") or c.get("error"),
        } for c in calls]), hide_index=True, width="stretch")
    with st.expander("检索记录"):
        st.dataframe(pd.DataFrame([{"发起": r["by"], "范围": r["scope"], "检索词": r["query"], "命中数": len(r["hits"])}
                                   for r in state.get("retrievals", [])]), hide_index=True, width="stretch")
    with st.expander("完整对话消息（模型实际看到的内容）"):
        for m in state.get("messages", []):
            st.markdown(f"**{m.type}**" + (f" · 工具调用：{[c['name'] for c in m.tool_calls]}" if getattr(m, "tool_calls", None) else ""))
            st.code(str(m.content)[:3000] or json.dumps([c["args"] for c in m.tool_calls], ensure_ascii=False)[:3000], language="text")


FIELD_COLS = ["fieldId", "operator", "value", "unit", "listId"]


def review_tab(ws: dict, state: dict, pending: dict | None) -> None:
    export = state.get("export")
    if export:
        st.success("已导出禁用配置包（is_enabled=0）。")
        for r in export["nativeRules"]:
            st.code(r["values"]["expression_en"], language="text", wrap_lines=True)
        st.download_button("下载 JSON 配置包", json.dumps(export, ensure_ascii=False, indent=2),
                           file_name=f"rule-export-{export['id'][:8]}.json", mime="application/json")
        return
    draft, checks = state.get("draft"), state.get("checks")
    if not draft:
        st.info("还没有草稿。先在 ③ 运行 Agent。")
        return
    in_review = bool(pending and pending["type"] == "review")
    profile = load_profile()
    if in_review and state.get("notes"):
        st.info("最新提示：" + state["notes"][-1])

    st.subheader("规则草稿")
    st.caption(draft.get("summary", ""))
    edited = json.loads(json.dumps(draft))
    for ri, rule in enumerate(edited.get("rules", [])):
        st.markdown(f"**{rule['key']} · {rule['name']}**")
        c1, c2, c3 = st.columns([2, 2, 1])
        rule["start"] = c1.text_input("生效时点", rule.get("start") or "", key=f"s{ri}") or None
        rule["end"] = c2.text_input("结束时点（左闭右开，可空）", rule.get("end") or "", key=f"e{ri}") or None
        rule["currency"] = c3.text_input("币种", rule.get("currency") or "", key=f"c{ri}") or None
        df = pd.DataFrame([{k: c.get(k) for k in FIELD_COLS} for c in rule["conditions"]], columns=FIELD_COLS)
        out = st.data_editor(df, key=f"cond{ri}", hide_index=True, width="stretch", column_config={
            "fieldId": st.column_config.SelectboxColumn("字段", options=[f.id for f in profile.fields]),
            "operator": st.column_config.SelectboxColumn("条件", options=["gt", "gte", "lt", "lte", "eq", "in_list"]),
            "value": st.column_config.TextColumn("值"),
            "unit": st.column_config.SelectboxColumn("原文单位", options=["yuan", "fen"]),
            "listId": st.column_config.SelectboxColumn("名单", options=[l.id for l in profile.lists]),
        }, disabled=not in_review)
        for ci, row in enumerate(out.to_dict("records")):
            for k in FIELD_COLS:
                v = row[k]
                rule["conditions"][ci][k] = None if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else v
        with st.expander("依据引文"):
            for c in rule["conditions"]:
                for e in c["evidence"]:
                    st.caption(f"{c['fieldId']} ← {e['quote']}")
    st.markdown("**条款覆盖表**")
    cov = pd.DataFrame(edited.get("coverage", []))
    if not cov.empty:
        cov["ruleKeys"] = cov["ruleKeys"].apply(lambda v: ",".join(v))
        cov_out = st.data_editor(cov, hide_index=True, width="stretch", disabled=not in_review, column_config={
            "clauseId": st.column_config.TextColumn("条款编号", disabled=True),
            "disposition": st.column_config.SelectboxColumn("处理方式", options=["rule", "manual", "not_applicable", "blocked"]),
            "reason": st.column_config.TextColumn("说明"),
            "ruleKeys": st.column_config.TextColumn("关联规则（逗号分隔）"),
        })
        edited["coverage"] = [{**r, "ruleKeys": [k for k in str(r["ruleKeys"]).split(",") if k]}
                              for r in cov_out.to_dict("records")]
    qs = st.text_area("待确认问题（一行一个；有问题时不能导出）", "\n".join(edited.get("questions", [])), disabled=not in_review)
    edited["questions"] = [q for q in qs.splitlines() if q.strip()]
    with st.expander("高级：直接编辑 JSON"):
        raw = st.text_area("草稿 JSON", json.dumps(edited, ensure_ascii=False, indent=2), height=300, disabled=not in_review)
    if in_review and st.button("💾 保存修改并重新校验"):
        try:
            drive(service.resume(ws["id"], {"action": "edit", "draft": json.loads(raw) if raw else edited}))
        except json.JSONDecodeError as exc:
            st.error(f"JSON 格式错误：{exc}")

    if checks:
        show_checks(checks)
    if not in_review:
        return
    st.subheader("人工复核")
    reviewer = st.text_input("复核人")
    notes = st.text_area("复核说明（有警告时必填：写明平台映射、名单版本、人工事项如何核对）")
    ack = st.checkbox("已核对全部原文、平台映射、名单版本、警告和人工事项；确认导出的是未启用草稿")
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ 批准并导出", type="primary"):
        drive(service.resume(ws["id"], {"action": "approve", "reviewer": reviewer, "notes": notes, "ack": ack}))
    if b2.button("↩️ 退回给模型修改（写在复核说明里）", disabled=state.get("mode") != "llm"):
        drive(service.resume(ws["id"], {"action": "revise", "notes": notes}))
    if b3.button("⛔ 驳回"):
        drive(service.resume(ws["id"], {"action": "reject", "notes": notes}))


def show_checks(checks: dict) -> None:
    st.subheader("程序校验结果")
    issues = checks.get("issues", [])
    if issues:
        st.error(f"{len(issues)} 个错误（有错误不能导出）")
        st.dataframe(pd.DataFrame(issues).rename(columns={"code": "错误代码", "path": "位置", "message": "说明"}),
                     hide_index=True, width="stretch")
    else:
        st.success("校验通过 ✅")
    for w in checks.get("warnings", []):
        st.warning(w)
    for r in checks.get("compiled", []):
        st.markdown(f"**编译结果 · {r['key']}**（trans_rule，is_enabled=0）")
        st.code(r["values"]["expression_en"], language="text", wrap_lines=True)
    if checks.get("boundaryTests"):
        st.markdown("**边界测试**（阈值 ±1 分，原文含义 vs 草稿比较符）")
        st.dataframe(pd.DataFrame(checks["boundaryTests"]).rename(columns={
            "rule": "规则", "name": "测试点", "expected": "按原文应命中", "actual": "按草稿会命中"}), hide_index=True, width="stretch")
    cov = checks.get("coverage", {})
    st.caption(f"条款 {cov.get('total')} 段，已说明 {cov.get('assigned')} 段，人工 {cov.get('manual')}，暂不能落实 {cov.get('blocked')}")


def profile_tab() -> None:
    profile = load_profile()
    st.caption("规则能用的字段、名单、模板和绑定 ID。当前为演示映射，不是生产配置。")
    st.json(profile.model_dump(), expanded=False)
    st.download_button("下载平台资料 JSON", json.dumps(profile.model_dump(), ensure_ascii=False, indent=2),
                       file_name="platform_profile.json", mime="application/json")
    up = st.file_uploader("导入平台资料 JSON（会替换当前资料）", type=["json"], key="profile_up")
    if up and st.button("导入"):
        try:
            save_profile(json.loads(up.getvalue().decode("utf-8")))
            st.success("已导入。新任务会使用新资料。")
        except Exception as exc:
            st.error(f"导入失败：{exc}")


mode, max_calls = sidebar()
if st.session_state.get("ws_id"):
    workspace(mode, max_calls)
else:
    new_task()
