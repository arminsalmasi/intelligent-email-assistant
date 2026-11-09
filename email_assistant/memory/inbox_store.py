"""Durable inbox state: the ranked items and their review status survive restarts.

Graph state lives in the LangGraph checkpointer; this table only keeps the
orchestrator's view (rank, triage, status, latest message, attempt) plus an
`epoch` that namespaces checkpoint keys. A new epoch (first start, or an explicit
reset) means "start from a clean slate" without deleting old checkpoints.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from email_assistant.graph.orchestrator import InboxItem


class InboxStore:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS items (thread_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            row = self._conn.execute("SELECT v FROM meta WHERE k = 'epoch'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta VALUES ('epoch', ?)", (uuid.uuid4().hex[:8],))

    @property
    def epoch(self) -> str:
        with self._lock:
            return self._conn.execute("SELECT v FROM meta WHERE k = 'epoch'").fetchone()[0]

    def save(self, item: InboxItem) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO items VALUES (?, ?)",
                (item.thread_id, item.model_dump_json()),
            )

    def load(self) -> Iterator[str]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM items").fetchall()
        return (r[0] for r in rows)

    def reset(self) -> None:
        """Forget all items and start a new checkpoint namespace."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM items")
            self._conn.execute("UPDATE meta SET v = ? WHERE k = 'epoch'", (uuid.uuid4().hex[:8],))
