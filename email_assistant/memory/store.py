"""Long-term memory: SQLite rows + embedding vectors, cosine retrieval.

Brute-force cosine is fine at mailbox scale (hundreds-thousands of rows); the
production path is pgvector / OpenSearch behind the same interface.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any

from email_assistant.clock import Clock
from email_assistant.memory.embeddings import Embedder, cosine
from email_assistant.schemas.memory import (
    SOURCE_PRIORITY,
    MemoryHit,
    MemoryKind,
    MemoryRecord,
    MemorySource,
    Sensitivity,
)


class MemoryStore:
    def __init__(self, db_path: str, embedder: Embedder, clock: Clock):
        self.embedder = embedder
        self.clock = clock
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                " id TEXT PRIMARY KEY, kind TEXT, namespace TEXT, key TEXT, content TEXT,"
                " metadata TEXT, sensitivity TEXT, source TEXT, confidence REAL,"
                " embedding TEXT, embedder TEXT, created_at TEXT, updated_at TEXT,"
                " last_used_at TEXT, use_count INTEGER, expires_at TEXT,"
                " UNIQUE(namespace, key))"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS ix_kind ON memories(kind, namespace)")

    # --------------------------------------------------------------- writes
    def put(
        self,
        kind: MemoryKind,
        namespace: str,
        key: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        sensitivity: Sensitivity = Sensitivity.INTERNAL,
        source: MemorySource = MemorySource.OBSERVED,
        confidence: float = 1.0,
        expires_at: datetime | None = None,
        embed_text: str | None = None,
    ) -> MemoryRecord | None:
        """Upsert by (namespace, key). A lower-priority source never overwrites a
        higher one (e.g. an inferred fact cannot override an explicit preference)."""
        now = self.clock.now()
        existing = self.get(namespace, key)
        if existing and SOURCE_PRIORITY[source] < SOURCE_PRIORITY[existing.source]:
            return None
        vector = self.embedder.embed([embed_text or content])[0]
        rec = MemoryRecord(
            id=existing.id if existing else str(uuid.uuid4()),
            kind=kind,
            namespace=namespace,
            key=key,
            content=content,
            metadata=metadata or {},
            sensitivity=sensitivity,
            source=source,
            confidence=confidence,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            use_count=existing.use_count if existing else 0,
            last_used_at=existing.last_used_at if existing else None,
            expires_at=expires_at,
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(namespace, key) DO UPDATE SET kind=excluded.kind,"
                " content=excluded.content, metadata=excluded.metadata,"
                " sensitivity=excluded.sensitivity, source=excluded.source,"
                " confidence=excluded.confidence, embedding=excluded.embedding,"
                " embedder=excluded.embedder, updated_at=excluded.updated_at,"
                " expires_at=excluded.expires_at",
                (
                    rec.id,
                    rec.kind.value,
                    namespace,
                    key,
                    content,
                    json.dumps(rec.metadata),
                    rec.sensitivity.value,
                    rec.source.value,
                    confidence,
                    json.dumps(vector),
                    self.embedder.name,
                    rec.created_at.isoformat(),
                    now.isoformat(),
                    rec.last_used_at.isoformat() if rec.last_used_at else None,
                    rec.use_count,
                    expires_at.isoformat() if expires_at else None,
                ),
            )
        return rec

    def touch(self, record_ids: list[str]) -> None:
        now = self.clock.now().isoformat()
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(now, rid) for rid in record_ids],
            )

    def delete(self, namespace: str, key: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE namespace = ? AND key = ?", (namespace, key)
            )
            return cur.rowcount > 0

    def purge_expired(self) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (self.clock.now().isoformat(),),
            )
            return cur.rowcount

    # ---------------------------------------------------------------- reads
    def _row(self, r: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=r["id"],
            kind=MemoryKind(r["kind"]),
            namespace=r["namespace"],
            key=r["key"],
            content=r["content"],
            metadata=json.loads(r["metadata"]),
            sensitivity=Sensitivity(r["sensitivity"]),
            source=MemorySource(r["source"]),
            confidence=r["confidence"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
            last_used_at=datetime.fromisoformat(r["last_used_at"]) if r["last_used_at"] else None,
            use_count=r["use_count"],
            expires_at=datetime.fromisoformat(r["expires_at"]) if r["expires_at"] else None,
        )

    def get(self, namespace: str, key: str) -> MemoryRecord | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM memories WHERE namespace = ? AND key = ?", (namespace, key)
            ).fetchone()
        return self._row(r) if r else None

    def list(
        self, kind: MemoryKind | None = None, namespace: str | None = None
    ) -> list[MemoryRecord]:
        q, params = "SELECT * FROM memories WHERE 1=1", []
        if kind:
            q += " AND kind = ?"
            params.append(kind.value)
        if namespace:
            q += " AND namespace = ?"
            params.append(namespace)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY updated_at DESC", params).fetchall()
        return [self._row(r) for r in rows]

    def search(
        self,
        query: str,
        k: int = 5,
        kind: MemoryKind | None = None,
        namespaces: list[str] | None = None,
        min_score: float = 0.0,
        min_confidence: float = 0.5,
    ) -> list[MemoryHit]:
        qv = self.embedder.embed([query])[0]
        q, params = (
            "SELECT * FROM memories WHERE confidence >= ? AND embedder = ?",
            [
                min_confidence,
                self.embedder.name,
            ],
        )
        if kind:
            q += " AND kind = ?"
            params.append(kind.value)
        if namespaces:
            q += f" AND namespace IN ({','.join('?' * len(namespaces))})"
            params.extend(namespaces)
        now = self.clock.now().isoformat()
        q += " AND (expires_at IS NULL OR expires_at >= ?)"
        params.append(now)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        hits = [
            MemoryHit(record=self._row(r), score=round(cosine(qv, json.loads(r["embedding"])), 4))
            for r in rows
        ]
        hits = [h for h in hits if h.score >= min_score]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
