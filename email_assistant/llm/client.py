"""LLMClient: the single gateway for model calls.

Responsibilities: structured output, bounded tool loops, retries with backoff,
rate limiting, record/replay caching, and telemetry (tokens, latency, cache
hits, failures). Callers get `None` on failure and must use their deterministic
fallback - an LLM outage degrades quality, it never breaks the pipeline.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from email_assistant.llm.cache import JsonlCache, cache_key
from email_assistant.observability.metrics import Telemetry

T = TypeVar("T", bound=BaseModel)


class ToolLoopResult(BaseModel):
    steps: int
    tool_calls: list[dict[str, Any]]


class LLM(Protocol):
    """Interface consumed by workers (real client or test fake)."""

    @property
    def available(self) -> bool: ...

    def structured(self, schema: type[T], system: str, user: str, purpose: str) -> T | None: ...

    def tool_loop(
        self,
        system: str,
        user: str,
        tools: list[BaseTool],
        final_schema: type[T],
        purpose: str,
        max_steps: int = 4,
    ) -> tuple[T | None, ToolLoopResult]: ...


class LLMClient:
    def __init__(
        self,
        model: BaseChatModel | None,
        model_name: str,
        telemetry: Telemetry,
        cache: JsonlCache,
        max_retries: int = 2,
        requests_per_minute: int = 0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.model_name = model_name
        self.telemetry = telemetry
        self.cache = cache
        self.max_retries = max_retries
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._sleep = sleep
        self.cache_misses = 0

    @property
    def available(self) -> bool:
        return self.model is not None or self.cache.mode == "replay"

    # ----------------------------------------------------------------- public
    def structured(self, schema: type[T], system: str, user: str, purpose: str) -> T | None:
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        return self._structured(schema, messages, purpose)

    def tool_loop(
        self,
        system: str,
        user: str,
        tools: list[BaseTool],
        final_schema: type[T],
        purpose: str,
        max_steps: int = 4,
    ) -> tuple[T | None, ToolLoopResult]:
        by_name = {t.name: t for t in tools}
        tool_specs = [convert_to_openai_tool(t) for t in tools]
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        calls: list[dict[str, Any]] = []
        steps = 0
        for steps in range(1, max_steps + 1):
            ai = self._chat(messages, tool_specs, purpose)
            if ai is None:
                return None, ToolLoopResult(steps=steps, tool_calls=calls)
            messages.append(ai)
            if not ai.tool_calls:
                break
            for tc in ai.tool_calls:
                tool = by_name.get(tc["name"])
                try:
                    if tool is None:
                        raise KeyError(f"unknown tool {tc['name']}")
                    result = tool.invoke(tc["args"])
                except Exception as exc:  # tool errors are fed back, not raised
                    result = {"error": str(exc)}
                calls.append({"tool": tc["name"], "args": tc["args"]})
                messages.append(
                    ToolMessage(
                        content=json.dumps(result, default=str),
                        tool_call_id=tc.get("id") or tc["name"],
                        name=tc["name"],
                    )
                )
        messages.append(
            HumanMessage("Based on the tool results above, return your final decision.")
        )
        final = self._structured(final_schema, messages, f"{purpose}.final")
        return final, ToolLoopResult(steps=steps, tool_calls=calls)

    # --------------------------------------------------------------- internals
    def _throttle(self) -> None:
        if not self._min_interval:
            return
        with self._lock:
            wait = self._last_call + self._min_interval - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()

    def _call_with_retries(self, fn: Callable[[], Any], purpose: str) -> Any | None:
        for attempt in range(self.max_retries + 1):
            self._throttle()
            started = time.perf_counter()
            try:
                return fn(), (time.perf_counter() - started) * 1000
            except Exception as exc:
                self.telemetry.record(
                    "llm_call",
                    purpose,
                    (time.perf_counter() - started) * 1000,
                    model=self.model_name,
                    ok=False,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                if attempt < self.max_retries:
                    self._sleep(min(2**attempt, 16) * (0.75 + random.random() / 2))
        return None

    def _record_ok(self, purpose: str, latency_ms: float, usage: dict | None, hit: bool) -> None:
        usage = usage or {}
        self.telemetry.record(
            "llm_call",
            purpose,
            latency_ms,
            model=self.model_name,
            ok=True,
            cache_hit=hit,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    def _miss(self, purpose: str) -> None:
        self.cache_misses += 1
        self.telemetry.record("llm_fallback", purpose, reason="cache_miss_or_no_model")

    def _structured(self, schema: type[T], messages: list[BaseMessage], purpose: str) -> T | None:
        key = cache_key(
            self.model_name,
            "structured",
            schema.__name__,
            schema.model_json_schema(),
            [message_to_dict(m) for m in messages],
        )
        cached = self.cache.get(key)
        if cached is not None:
            self._record_ok(purpose, 0.0, cached.get("usage"), hit=True)
            return schema.model_validate(cached["parsed"])
        if self.model is None:
            self._miss(purpose)
            return None
        runnable = self.model.with_structured_output(schema, include_raw=True)

        def _invoke() -> tuple[T, dict]:
            out = runnable.invoke(messages)
            parsed = out.get("parsed")
            if parsed is None:
                raise ValueError(f"structured output parse failure: {out.get('parsing_error')}")
            if isinstance(parsed, dict):
                parsed = schema.model_validate(parsed)
            raw = out.get("raw")
            return parsed, dict(getattr(raw, "usage_metadata", None) or {})

        result = self._call_with_retries(_invoke, purpose)
        if result is None:
            self.telemetry.record("llm_fallback", purpose, reason="llm_error")
            return None
        (parsed, usage), latency = result
        self._record_ok(purpose, latency, usage, hit=False)
        self.cache.put(key, {"parsed": parsed.model_dump(mode="json"), "usage": usage})
        return parsed

    def _chat(
        self, messages: list[BaseMessage], tool_specs: list[dict], purpose: str
    ) -> AIMessage | None:
        key = cache_key(self.model_name, "chat", tool_specs, [message_to_dict(m) for m in messages])
        cached = self.cache.get(key)
        if cached is not None:
            self._record_ok(purpose, 0.0, cached.get("usage"), hit=True)
            return messages_from_dict([cached["message"]])[0]  # type: ignore[return-value]
        if self.model is None:
            self._miss(purpose)
            return None
        bound = self.model.bind_tools(tool_specs)
        result = self._call_with_retries(lambda: bound.invoke(messages), purpose)
        if result is None:
            self.telemetry.record("llm_fallback", purpose, reason="llm_error")
            return None
        ai, latency = result
        usage = dict(getattr(ai, "usage_metadata", None) or {})
        self._record_ok(purpose, latency, usage, hit=False)
        clean = AIMessage(content=ai.content, tool_calls=ai.tool_calls)
        self.cache.put(key, {"message": message_to_dict(clean), "usage": usage})
        return clean


class NullLLM:
    """Explicit 'no model' client: every call returns None (heuristic mode)."""

    def __init__(self, telemetry: Telemetry):
        self.telemetry = telemetry
        self.cache_misses = 0

    @property
    def available(self) -> bool:
        return False

    def structured(self, schema, system, user, purpose):  # type: ignore[no-untyped-def]
        self.telemetry.record("llm_fallback", purpose, reason="no_llm")
        return None

    def tool_loop(self, system, user, tools, final_schema, purpose, max_steps=4):  # type: ignore[no-untyped-def]
        self.telemetry.record("llm_fallback", purpose, reason="no_llm")
        return None, ToolLoopResult(steps=0, tool_calls=[])
