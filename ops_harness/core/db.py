"""SQLite persistence: executions, append-only step history, messages, approval audit."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    max_steps INTEGER NOT NULL,
    deadline_at TEXT NOT NULL,
    final_answer TEXT,
    error TEXT,
    meta TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES executions(id),
    seq INTEGER NOT NULL,
    step_type TEXT NOT NULL,
    name TEXT NOT NULL,
    request_json TEXT,
    response_json TEXT,
    status TEXT NOT NULL,
    latency_ms INTEGER,
    retry_count INTEGER,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_exec ON steps(execution_id, seq);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL REFERENCES executions(id),
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_call_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_exec ON messages(execution_id, seq);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES executions(id),
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    decision TEXT,
    decided_by TEXT,
    comment TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_exec ON approvals(execution_id);
"""


def utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class Store:
    """Thread-safe SQLite store (one short-lived connection per operation, WAL mode)."""

    def __init__(self, db_path: str):
        path = Path(db_path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Lightweight in-place migration for databases created by older versions."""
        exec_cols = {row[1] for row in conn.execute("PRAGMA table_info(executions)")}
        if "meta" not in exec_cols:
            conn.execute("ALTER TABLE executions ADD COLUMN meta TEXT")

    # ------------------------------------------------------------ executions

    def create_execution(self, exec_id: str, objective: str, max_steps: int, deadline_at: str,
                         meta: str | None = None) -> None:
        now = utcnow_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO executions (id, objective, status, max_steps, deadline_at, meta, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (exec_id, objective, "PENDING", max_steps, deadline_at, meta, now, now),
            )

    def get_execution(self, exec_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM executions WHERE id = ?", (exec_id,)).fetchone()
            return dict(row) if row else None

    def update_execution(self, exec_id: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = utcnow_iso()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE executions SET {cols} WHERE id = ?", (*fields.values(), exec_id))

    def list_executions(self, limit: int = 100) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM executions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------ steps

    def add_step(
        self,
        exec_id: str,
        step_type: str,
        name: str,
        status: str,
        request_json: str | None = None,
        response_json: str | None = None,
        latency_ms: int | None = None,
        retry_count: int | None = None,
        error: str | None = None,
    ) -> str:
        import uuid

        step_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM steps WHERE execution_id = ?",
                (exec_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO steps (id, execution_id, seq, step_type, name, request_json,"
                " response_json, status, latency_ms, retry_count, error, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    step_id, exec_id, seq_row["next"], step_type, name, request_json,
                    response_json, status, latency_ms, retry_count, error, utcnow_iso(),
                ),
            )
        return step_id

    def update_step(
        self,
        step_id: str,
        status: str | None = None,
        response_json: str | None = None,
        latency_ms: int | None = None,
        retry_count: int | None = None,
        error: str | None = None,
    ) -> None:
        fields = {k: v for k, v in {
            "status": status, "response_json": response_json,
            "latency_ms": latency_ms, "retry_count": retry_count, "error": error,
        }.items() if v is not None}
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE steps SET {cols} WHERE id = ?", (*fields.values(), step_id))

    def list_steps(self, exec_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE execution_id = ? ORDER BY seq", (exec_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def count_steps(self, exec_id: str, step_type: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM steps WHERE execution_id = ? AND step_type = ?",
                (exec_id, step_type),
            ).fetchone()
            return row["n"]

    # ------------------------------------------------------------ messages

    def add_message(self, exec_id: str, role: str, content: str, tool_call_id: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM messages WHERE execution_id = ?",
                (exec_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO messages (execution_id, seq, role, content, tool_call_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (exec_id, seq_row["next"], role, content, tool_call_id, utcnow_iso()),
            )

    def get_messages(self, exec_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_call_id FROM messages WHERE execution_id = ? ORDER BY seq",
                (exec_id,),
            ).fetchall()
            out = []
            for r in rows:
                msg = {"role": r["role"], "content": r["content"]}
                if r["tool_call_id"]:
                    msg["tool_call_id"] = r["tool_call_id"]
                out.append(msg)
            return out

    # ------------------------------------------------------------ approvals

    def create_approval(self, exec_id: str, tool_name: str, arguments_json: str) -> str:
        import uuid

        approval_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO approvals (id, execution_id, tool_name, arguments_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (approval_id, exec_id, tool_name, arguments_json, utcnow_iso()),
            )
        return approval_id

    def get_approval(self, approval_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
            return dict(row) if row else None

    def update_approval(self, approval_id: str, decision: str, decided_by: str, comment: str | None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE approvals SET decision = ?, decided_by = ?, comment = ?, decided_at = ? WHERE id = ?",
                (decision, decided_by, comment, utcnow_iso(), approval_id),
            )

    def list_approvals(self, status: str | None = None, exec_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM approvals WHERE 1=1"
        params: list = []
        if status == "pending":
            query += " AND decision IS NULL"
        elif status:
            query += " AND decision = ?"
            params.append(status)
        if exec_id:
            query += " AND execution_id = ?"
            params.append(exec_id)
        query += " ORDER BY created_at DESC LIMIT 200"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
