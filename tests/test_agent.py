"""Agent 流程测试：用“假模型”代替真实大模型，验证 LangGraph 的循环、纠错、人工介入和导出。"""
import copy
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from regrule import documents, fixtures, service


def call(name, args):
    return {"name": name, "args": args, "id": "c_" + uuid.uuid4().hex[:6], "type": "tool_call"}


class ScriptModel(BaseChatModel):
    """按顺序吐出预先写好的回复；每次可以根据已有消息动态生成。"""
    steps: list

    @property
    def _llm_type(self):
        return "script"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        n = sum(isinstance(m, AIMessage) for m in messages)
        step = self.steps[min(n, len(self.steps) - 1)]
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_self_correction_then_export():
    ws = service.create_from_example("gt")
    doc = documents.get(ws["document_id"])
    good = fixtures.expected_draft("gt", doc)
    wrong = copy.deepcopy(good)
    wrong["rules"][0]["conditions"][1]["operator"] = "gte"   # 第一次故意写错

    model = ScriptModel(steps=[
        AIMessage("", tool_calls=[call("search_regulation", {"query": doc.clauses[1].id})]),
        AIMessage("", tool_calls=[call("submit_draft", {"draft": wrong})]),
        AIMessage("", tool_calls=[call("submit_draft", {"draft": good})]),
    ])
    list(service.run(ws["id"], fixtures.DEFAULT_GOAL, "llm", 4, chat_model=model))
    state = service.snapshot(ws["id"]).values
    pending = service.pending_interrupt(ws["id"])
    assert state["llm_calls"] == 3
    # 第一次提交后模型收到了具体的校验错误
    feedback = [m.content for m in state["messages"] if isinstance(m, ToolMessage) and "校验未通过" in m.content]
    assert feedback and "OPERATOR_SOURCE" in feedback[0]
    assert pending["type"] == "review" and pending["checks"]["issues"] == []

    # 复核：先不填复核人 → 被拦下
    list(service.resume(ws["id"], {"action": "approve", "reviewer": "", "notes": "x", "ack": True}, chat_model=model))
    assert service.pending_interrupt(ws["id"])["type"] == "review"
    list(service.resume(ws["id"], {"action": "approve", "reviewer": "张三", "notes": "已核对", "ack": True}, chat_model=model))
    state = service.snapshot(ws["id"]).values
    assert state["status"] == "exported"
    assert state["export"]["nativeRules"][0]["values"]["is_enabled"] == 0


def test_budget_exhausted_goes_to_human():
    ws = service.create_from_example("gt")
    doc = documents.get(ws["document_id"])
    wrong = fixtures.expected_draft("gt", doc)
    wrong["rules"][0]["conditions"][1]["unit"] = "fen"
    model = ScriptModel(steps=[AIMessage("", tool_calls=[call("submit_draft", {"draft": wrong})])])
    list(service.run(ws["id"], fixtures.DEFAULT_GOAL, "llm", 2, chat_model=model))
    state = service.snapshot(ws["id"]).values
    assert state["llm_calls"] == 2
    assert service.pending_interrupt(ws["id"])["type"] == "review"


def test_clarify_interrupt_and_resume():
    ws = service.create_from_example("missing-unit")
    model = ScriptModel(steps=[
        AIMessage("", tool_calls=[call("ask_human", {"questions": ["金额单位是什么？"]})]),
        lambda msgs: AIMessage("", tool_calls=[call("handoff_to_human", {"reason": "等人工改"})]),
    ])
    list(service.run(ws["id"], fixtures.DEFAULT_GOAL, "llm", 4, chat_model=model))
    assert service.pending_interrupt(ws["id"]) == {"type": "clarify", "questions": ["金额单位是什么？"]}
    list(service.resume(ws["id"], {"answer": "元"}, chat_model=model))
    msgs = service.snapshot(ws["id"]).values["messages"]
    assert any(getattr(m, "content", "") == "人工补充：元" for m in msgs)


def test_model_error_is_reported_not_faked():
    ws = service.create_from_example("gt")

    class Broken(ScriptModel):
        def _generate(self, *a, **k):
            raise RuntimeError("HTTP 401 Unauthorized")

    list(service.run(ws["id"], fixtures.DEFAULT_GOAL, "llm", 4, chat_model=Broken(steps=[])))
    state = service.snapshot(ws["id"]).values
    assert state["status"] == "error" and state.get("draft") is None


def test_real_chatopenai_wire_format(monkeypatch):
    """启动一个假的“DeepSeek”HTTP 服务，验证 ChatOpenAI 发出的请求格式正确、返回的工具调用能被正确处理。"""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            reply = {
                "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {
                        "name": "ask_human", "arguments": json.dumps({"questions": ["测试问题"]}, ensure_ascii=False)}}],
                }}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    try:
        ws = service.create_from_example("gt")
        list(service.run(ws["id"], fixtures.DEFAULT_GOAL, "llm", 2))
    finally:
        server.shutdown()
    req = seen[0]
    assert req["path"].endswith("/chat/completions") and req["auth"] == "Bearer sk-test"
    assert req["body"]["thinking"] == {"type": "disabled"}           # 关闭 DeepSeek 思考模式
    assert req["body"]["tool_choice"] == "required"
    assert {t["function"]["name"] for t in req["body"]["tools"]} >= {"submit_draft", "ask_human", "search_regulation"}
    draft_schema = next(t for t in req["body"]["tools"] if t["function"]["name"] == "submit_draft")["function"]["parameters"]
    assert "draft" in draft_schema["properties"] and "state" not in draft_schema["properties"]  # 注入参数不暴露给模型
    state = service.snapshot(ws["id"]).values
    assert state["call_log"][0]["usage"]["total_tokens"] == 18
    assert service.pending_interrupt(ws["id"])["questions"] == ["测试问题"]
