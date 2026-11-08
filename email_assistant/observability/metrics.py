"""Telemetry store: every metric is an event row in SQLite.

Kept deliberately simple (no Prometheus in a take-home); the event schema maps
1:1 onto OpenTelemetry metrics/spans if exported later. Aggregations feed the
API `/metrics/summary` and the dashboard.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from email_assistant.observability.logging import current_context, log_event

EVENT_TYPES = {
    "node",  # node execution: name, latency_ms, ok
    "llm_call",  # model, purpose, tokens, latency, cache_hit, ok
    "llm_fallback",  # worker fell back to deterministic logic
    "embedding",  # provider, n, cache_hit
    "policy_finding",  # code, severity, action_type
    "guard",  # risk_score, quarantined
    "human_decision",  # decision, action_type
    "action",  # action_type, status
    "memory_reuse",  # mode (template|example), tokens_saved
    "error",
}


class Telemetry:
    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, type TEXT, name TEXT,"
                " value REAL, run_id TEXT, thread_id TEXT, attrs TEXT)"
            )

    def record(self, type_: str, name: str, value: float = 1.0, **attrs: Any) -> None:
        ctx = current_context()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO events (ts, type, name, value, run_id, thread_id, attrs)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    type_,
                    name,
                    float(value),
                    ctx.get("run_id"),
                    attrs.pop("thread_id", None) or ctx.get("thread_id"),
                    json.dumps(attrs, default=str),
                ),
            )
        if type_ in {"error", "llm_fallback", "policy_finding", "guard"}:
            log_event(f"{type_}.{name}", value=value, **attrs)

    def events(self, type_: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        q = "SELECT ts, type, name, value, run_id, thread_id, attrs FROM events"
        params: list[Any] = []
        if type_:
            q += " WHERE type = ?"
            params.append(type_)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [
            {
                "ts": r[0],
                "type": r[1],
                "name": r[2],
                "value": r[3],
                "run_id": r[4],
                "thread_id": r[5],
                **json.loads(r[6] or "{}"),
            }
            for r in rows
        ]

    def reset(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM events")

    def summary(self) -> dict[str, Any]:
        nodes: dict[str, list[float]] = defaultdict(list)
        for e in self.events("node"):
            nodes[e["name"]].append(e["value"])
        llm = self.events("llm_call")
        llm_live = [e for e in llm if not e.get("cache_hit")]
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t in ("human_decision", "action", "policy_finding", "llm_fallback", "error", "guard"):
            for e in self.events(t):
                counts[t][e["name"]] += 1
        reuse = self.events("memory_reuse")

        def pct(vals: list[float], p: float) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            return round(s[min(len(s) - 1, int(p * (len(s) - 1)))], 1)

        return {
            "node_latency_ms": {
                n: {"count": len(v), "p50": pct(v, 0.5), "p95": pct(v, 0.95)}
                for n, v in sorted(nodes.items())
            },
            "llm": {
                "calls": len(llm),
                "cache_hits": len(llm) - len(llm_live),
                "failures": sum(1 for e in llm if not e.get("ok", True)),
                "input_tokens": int(sum(e.get("input_tokens", 0) for e in llm)),
                "output_tokens": int(sum(e.get("output_tokens", 0) for e in llm)),
                "latency_ms_p50": pct([e["value"] for e in llm_live], 0.5),
                "latency_ms_p95": pct([e["value"] for e in llm_live], 0.95),
            },
            "fallbacks": dict(counts["llm_fallback"]),
            "human_decisions": dict(counts["human_decision"]),
            "actions": dict(counts["action"]),
            "policy_findings": dict(counts["policy_finding"]),
            "guard": dict(counts["guard"]),
            "errors": dict(counts["error"]),
            "memory_reuse": {
                "templates": sum(1 for e in reuse if e["name"] == "template"),
                "examples": sum(1 for e in reuse if e["name"] == "example"),
                "estimated_tokens_saved": int(sum(e.get("tokens_saved", 0) for e in reuse)),
            },
        }
