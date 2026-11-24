"""SQLite 存储：文档、检索片段、任务、导出包。

用 Python 自带的 sqlite3，不需要单独装数据库服务。
LangGraph 的运行状态另存在 checkpoints.sqlite（由 SqliteSaver 管理）。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from .config import APP_DB

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(APP_DB, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks(
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL,
                label TEXT NOT NULL, text TEXT NOT NULL, clause_id TEXT);
            CREATE INDEX IF NOT EXISTS chunks_owner ON chunks(owner, kind);
            CREATE TABLE IF NOT EXISTS workspaces(
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL, name TEXT NOT NULL,
                example_id TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS exports(id TEXT PRIMARY KEY, workspace_id TEXT, digest TEXT, payload TEXT NOT NULL);
            """
        )
    return _conn


def execute(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with _lock:
        cur = db().execute(sql, params)
        rows = cur.fetchall()
        db().commit()
        return rows


def put_document(doc: dict) -> None:
    execute("INSERT OR REPLACE INTO documents VALUES(?,?)", (doc["id"], json.dumps(doc, ensure_ascii=False)))


def get_document(doc_id: str) -> dict:
    rows = execute("SELECT payload FROM documents WHERE id=?", (doc_id,))
    if not rows:
        raise KeyError("文档不存在。")
    return json.loads(rows[0]["payload"])


def put_chunks(chunks: list[dict]) -> list[dict]:
    """写入片段，返回本次新增的（已存在的跳过）。"""
    added = []
    with _lock:
        conn = db()
        for c in chunks:
            cur = conn.execute(
                "INSERT OR IGNORE INTO chunks VALUES(?,?,?,?,?,?)",
                (c["id"], c["owner"], c["kind"], c["label"], c["text"], c.get("clause_id")),
            )
            if cur.rowcount:
                added.append(c)
        conn.commit()
    return added


def chunks_for(owner: str, kind: str) -> list[dict]:
    rows = execute("SELECT * FROM chunks WHERE owner=? AND kind=? ORDER BY id", (owner, kind))
    return [dict(r) for r in rows]


def put_workspace(ws: dict) -> None:
    execute(
        "INSERT OR REPLACE INTO workspaces VALUES(?,?,?,?,?)",
        (ws["id"], ws["document_id"], ws["name"], ws.get("example_id"), ws["created_at"]),
    )


def list_workspaces(limit: int = 40) -> list[dict]:
    return [dict(r) for r in execute("SELECT * FROM workspaces ORDER BY created_at DESC LIMIT ?", (limit,))]


def get_workspace(ws_id: str) -> dict:
    rows = execute("SELECT * FROM workspaces WHERE id=?", (ws_id,))
    if not rows:
        raise KeyError("任务不存在。")
    return dict(rows[0])


def find_export(workspace_id: str, digest: str) -> dict | None:
    rows = execute("SELECT payload FROM exports WHERE workspace_id=? AND digest=?", (workspace_id, digest))
    return json.loads(rows[0]["payload"]) if rows else None


def put_export(bundle: dict) -> None:
    execute(
        "INSERT OR REPLACE INTO exports VALUES(?,?,?,?)",
        (bundle["id"], bundle["workspaceId"], bundle["digest"], json.dumps(bundle, ensure_ascii=False)),
    )
