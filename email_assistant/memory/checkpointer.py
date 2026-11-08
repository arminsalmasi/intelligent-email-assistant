"""Short-term memory: LangGraph checkpointer with an explicit serializer allow-list.

Only our own schema types may be deserialized from checkpoints (LangGraph
blocks unregistered types in future versions; we opt in explicitly instead of
allowing everything).
"""

from __future__ import annotations

import inspect
import sqlite3
from enum import Enum

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from email_assistant.schemas import actions, email, memory, review, scheduling, state, triage

_SCHEMA_MODULES = (actions, email, memory, review, scheduling, state, triage)


def allowed_types() -> list[tuple[str, str]]:
    out = []
    for mod in _SCHEMA_MODULES:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and issubclass(obj, BaseModel | Enum):
                out.append((mod.__name__, name))
    return out


def make_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=allowed_types())


def make_checkpointer(db_path: str) -> SqliteSaver:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn, serde=make_serializer())
