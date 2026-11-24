"""演示模式用的“剧本模型”——不是大模型，按固定剧本调用工具。

用途：没配置 API 时，也能把整个 LangGraph 流程（检索 → 提交草稿 → 校验 → 人工复核 → 导出）走一遍。
它只认识 10 份内置学习示例，提交的是 fixtures.py 里人工写好的标准答案，页面上会明确标注“预置答案，非模型”。
同时它也是一个很好的例子：LangChain 里任何“聊天模型”只要实现 _generate 和 bind_tools 就能接进 Agent。
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .. import documents
from ..fixtures import expected_draft


def _call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"demo_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


class ScriptedDemoModel(BaseChatModel):
    example_id: str
    document_id: str

    @property
    def _llm_type(self) -> str:
        return "scripted-demo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedDemoModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        doc = documents.get(self.document_id)
        tool_names = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        answered = any(isinstance(m, HumanMessage) and str(m.content).startswith("人工") for m in messages)
        target = doc.clauses[1].id if len(doc.clauses) > 1 else doc.clauses[0].id

        if "search_regulation" not in tool_names:
            # 第 1 步：按条款编号取第 2 段原文，确保引用的片段在证据里
            msg = AIMessage(content="先取回第2段原文作为依据。", tool_calls=[_call("search_regulation", {"query": target})])
        elif self.example_id in ("missing-unit", "missing-version") and not answered and "ask_human" not in tool_names:
            # 第 2 步（缺信息的示例）：提问而不是猜
            q = "请确认原文金额的单位。" if self.example_id == "missing-unit" else "请确认下发名单的版本。"
            msg = AIMessage(content="原文缺少关键信息，向人工提问。", tool_calls=[_call("ask_human", {"questions": [q]})])
        elif "submit_draft" in tool_names:
            # 已经提交过但没通过：演示模型不会真的修改，直接交给人工
            msg = AIMessage(content="演示模式无法根据校验结果修改草稿，交给人工。",
                            tool_calls=[_call("handoff_to_human", {"reason": "演示模式不会理解人工回答或校验错误，请在复核区手动修改。"})])
        else:
            # 第 2/3 步：提交人工写好的标准答案
            draft = expected_draft(self.example_id, doc)
            msg = AIMessage(content="提交预置答案（非模型生成）。", tool_calls=[_call("submit_draft", {"draft": draft})])
        return ChatResult(generations=[ChatGeneration(message=msg)])
