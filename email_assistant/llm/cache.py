"""Record/replay cache for LLM and embedding calls.

* record : read-through cache that also writes new responses
* replay : read-only; misses are reported (LLM) or raised (embeddings)
* off    : disabled

Recorded responses on the fictional fixtures are committed, so CI evaluates the
real LLM pipeline deterministically without an API key.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


def cache_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CacheMiss(KeyError):
    pass


class JsonlCache:
    """Append-only JSONL file keyed by sha256. Small, diff-friendly, no extra deps."""

    def __init__(self, path: Path, mode: str = "off"):
        self.path = path
        self.mode = mode
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        if mode != "off" and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._data[row["key"]] = row["value"]

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        if self.mode != "record":
            return
        with self._lock:
            if key in self._data:
                return
            self._data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._data)
