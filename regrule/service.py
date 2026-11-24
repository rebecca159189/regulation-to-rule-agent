"""服务层：页面和评测脚本都通过这里操作，不直接碰图的细节。"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import documents, fixtures, store
from .agent.graph import build_graph, start_input
from .config import CHECKPOINT_DB


@lru_cache(maxsize=1)
def graph():
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return build_graph(SqliteSaver(conn))


def _config(ws_id: str, chat_model=None) -> dict:
    cfg = {"configurable": {"thread_id": ws_id}, "recursion_limit": 60}
    if chat_model is not None:
        cfg["configurable"]["chat_model"] = chat_model
    return cfg


# ---------- 创建任务 ----------
def _workspace(doc, example_id: str | None = None) -> dict:
    ws = {"id": store.new_id(), "document_id": doc.id, "name": doc.name,
          "example_id": example_id, "created_at": store.now()}
    store.put_workspace(ws)
    return ws


def create_from_example(example_id: str) -> dict:
    ex = fixtures.example(example_id)
    doc = documents.ingest(f"学习通知 · {ex['name']}", [{"page": 1, "text": fixtures.example_text(example_id)}],
                           f"example:{example_id}")
    return _workspace(doc, example_id)


def create_from_text(name: str, text: str) -> dict:
    return _workspace(documents.ingest(name or "粘贴的文字", [{"page": 1, "text": text}], "paste"))


def create_from_file(name: str, path: Path) -> dict:
    pages, warnings, parser = documents.load_file(path)
    return _workspace(documents.ingest(name, pages, parser, warnings))


# ---------- 运行 ----------
def run(ws_id: str, goal: str, mode: str, max_calls: int, chat_model=None) -> Iterator[dict]:
    """从头运行一次，逐步产出每个节点的更新（stream_mode="updates"）。"""
    ws = store.get_workspace(ws_id)
    inputs = start_input(ws, goal, mode, max_calls)
    yield from graph().stream(inputs, _config(ws_id, chat_model), stream_mode="updates")


def resume(ws_id: str, value: dict, chat_model=None) -> Iterator[dict]:
    """人工操作之后继续运行（回答问题、复核决定）。"""
    yield from graph().stream(Command(resume=value), _config(ws_id, chat_model), stream_mode="updates")


def snapshot(ws_id: str):
    return graph().get_state(_config(ws_id))


def pending_interrupt(ws_id: str) -> dict | None:
    snap = snapshot(ws_id)
    for task in snap.tasks or ():
        for intr in task.interrupts or ():
            return intr.value
    return None


def mermaid() -> str:
    return graph().get_graph().draw_mermaid()
