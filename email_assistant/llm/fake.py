"""Scripted LLM for tests: returns pre-built objects per purpose."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from email_assistant.llm.client import ToolLoopResult

Responder = Any | Callable[[str, str], Any]


class FakeLLM:
    def __init__(self, responses: dict[str, Responder] | None = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    @property
    def available(self) -> bool:
        return True

    def _respond(self, purpose: str, system: str, user: str) -> Any:
        self.calls.append((purpose, user))
        r = self.responses.get(purpose)
        return r(system, user) if callable(r) else r

    def structured(self, schema, system, user, purpose):  # type: ignore[no-untyped-def]
        return self._respond(purpose, system, user)

    def tool_loop(self, system, user, tools, final_schema, purpose, max_steps=4):  # type: ignore[no-untyped-def]
        return self._respond(purpose, system, user), ToolLoopResult(steps=1, tool_calls=[])
