"""LLMClient: the single gateway for model calls.

Responsibilities: structured output, bounded tool loops, retries with backoff,
rate limiting, record/replay caching, and telemetry (tokens, latency, cache
hits, failures). Callers get `None` on failure and must use their deterministic
fallback - an LLM outage degrades quality, it never breaks the pipeline.
"""

from __future__ import annotations

import copy
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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


@dataclass
class Backend:
    """One model endpoint in the fallback chain (e.g. gemini -> bedrock -> local)."""

    name: str  # model id, also the cache namespace (e.g. "gemini-3.1-flash-lite")
    provider: str  # gemini | bedrock | local
    model: BaseChatModel | None  # None = replay-only (answers from cache)
    cache: JsonlCache
    structured_method: str | None = None  # e.g. "json_schema" for the local engine
    down: str | None = field(default=None)  # reason once the circuit breaker trips


_FATAL = (
    "resource_exhausted",
    "quota",
    "429",
    "rate limit",
    "ratelimit",
    "permission_denied",
    "unauthorized",
    "401",
    "403",
    "api key",
    "api_key",
    "credentials",
    "not_found",
    "404",
)
_TIMEOUT = ("timeout", "timed out", "deadline")
_CONNECTION = (
    "connecterror",
    "connection refused",
    "connectionerror",
    "apiconnectionerror",
    "connect error",
    "name or service not known",
)


def classify_error(exc: BaseException) -> str:
    """fatal: backend unusable for this run (quota/auth/model missing) -> trip breaker.
    timeout: slow generation, do NOT retry (it only queues more work) -> next backend.
    connection: retry with backoff, then trip.  transient: retry (5xx, parse errors)."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(k in text for k in _FATAL):
        return "fatal"
    if any(k in text for k in _TIMEOUT):
        return "timeout"
    if any(k in text for k in _CONNECTION):
        return "connection"
    return "transient"


class LLMClient:
    """Gateway over an ordered chain of backends with per-backend caches.

    Every call tries backends in order; a cache hit answers immediately. Quota,
    auth and connection failures trip a backend's circuit breaker for the rest of
    the run, so e.g. an exhausted Gemini free tier falls through to the local engine.
    """

    def __init__(
        self,
        model: BaseChatModel | None,
        model_name: str,
        telemetry: Telemetry,
        cache: JsonlCache,
        max_retries: int = 2,
        requests_per_minute: int = 0,
        sleep: Callable[[float], None] = time.sleep,
        provider: str = "gemini",
        fallbacks: list[Backend] | None = None,
        structured_method: str | None = None,
    ):
        self.backends = [
            Backend(model_name, provider, model, cache, structured_method),
            *(fallbacks or []),
        ]
        self.telemetry = telemetry
        self.max_retries = max_retries
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._sleep = sleep
        self.cache_misses = 0

    @classmethod
    def from_backends(cls, backends: list[Backend], telemetry: Telemetry, **kw: Any) -> LLMClient:
        first, *rest = backends
        return cls(
            first.model,
            first.name,
            telemetry,
            first.cache,
            provider=first.provider,
            fallbacks=rest,
            structured_method=first.structured_method,
            **kw,
        )

    @property
    def model_name(self) -> str:
        live = [b for b in self.backends if b.down is None]
        return (live or self.backends)[0].name

    @property
    def available(self) -> bool:
        return any(
            b.down is None and (b.model is not None or b.cache.mode == "replay")
            for b in self.backends
        )

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": b.name,
                "provider": b.provider,
                "live": b.model is not None,
                "cache": b.cache.mode,
                "down": b.down,
            }
            for b in self.backends
        ]

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
    def _throttle(self, backend: Backend) -> None:
        if not self._min_interval or backend.provider == "local":
            return  # the local engine queues itself; throttling is for hosted quotas
        with self._lock:
            wait = self._last_call + self._min_interval - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()

    def _call(self, backend: Backend, fn: Callable[[], Any], purpose: str) -> Any | None:
        for attempt in range(self.max_retries + 1):
            self._throttle(backend)
            started = time.perf_counter()
            try:
                return fn(), (time.perf_counter() - started) * 1000
            except Exception as exc:
                kind = classify_error(exc)
                self.telemetry.record(
                    "llm_call",
                    purpose,
                    (time.perf_counter() - started) * 1000,
                    model=backend.name,
                    ok=False,
                    attempt=attempt,
                    error=type(exc).__name__,
                    error_kind=kind,
                )
                if kind == "fatal" or (kind == "connection" and attempt == self.max_retries):
                    backend.down = f"{kind}: {type(exc).__name__}"
                    self.telemetry.record(
                        "llm_fallback",
                        purpose,
                        reason=f"backend_down:{backend.name}",
                        detail=str(exc)[:200],
                    )
                    return None
                if kind == "timeout":
                    return None
                if attempt < self.max_retries:
                    self._sleep(min(2**attempt, 16) * (0.75 + random.random() / 2))
        return None

    def _record_ok(
        self, backend: Backend, purpose: str, latency_ms: float, usage: dict | None, hit: bool
    ) -> None:
        usage = usage or {}
        self.telemetry.record(
            "llm_call",
            purpose,
            latency_ms,
            model=backend.name,
            ok=True,
            cache_hit=hit,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    def _structured(self, schema: type[T], messages: list[BaseMessage], purpose: str) -> T | None:
        msgs = [message_to_dict(m) for m in messages]
        for b in self.backends:
            if b.down:
                continue
            key = cache_key(b.name, "structured", schema.__name__, schema.model_json_schema(), msgs)
            cached = b.cache.get(key)
            if cached is not None:
                self._record_ok(b, purpose, 0.0, cached.get("usage"), hit=True)
                return schema.model_validate(cached["parsed"])
            if b.model is None:
                continue
            kwargs = {"include_raw": True}
            if b.structured_method:
                kwargs["method"] = b.structured_method
            runnable = b.model.with_structured_output(schema, **kwargs)

            def _invoke(runnable: Any = runnable) -> tuple[T, dict]:
                out = runnable.invoke(messages)
                parsed = out.get("parsed")
                if parsed is None:
                    raise ValueError(f"structured output parse failure: {out.get('parsing_error')}")
                if isinstance(parsed, dict):
                    parsed = schema.model_validate(parsed)
                raw = out.get("raw")
                return parsed, dict(getattr(raw, "usage_metadata", None) or {})

            result = self._call(b, _invoke, purpose)
            if result is None:
                continue
            (parsed, usage), latency = result
            self._record_ok(b, purpose, latency, usage, hit=False)
            b.cache.put(key, {"parsed": parsed.model_dump(mode="json"), "usage": usage})
            return parsed
        self._no_answer(purpose)
        return None

    def _chat(
        self, messages: list[BaseMessage], tool_specs: list[dict], purpose: str
    ) -> AIMessage | None:
        msgs = [message_to_dict(m) for m in messages]
        for b in self.backends:
            if b.down:
                continue
            key = cache_key(b.name, "chat", tool_specs, msgs)
            cached = b.cache.get(key)
            if cached is not None:
                self._record_ok(b, purpose, 0.0, cached.get("usage"), hit=True)
                return messages_from_dict([cached["message"]])[0]  # type: ignore[return-value]
            if b.model is None:
                continue
            # deepcopy: provider integrations normalise (mutate) tool specs in place, which
            # would change the cache key of later loop steps between live and replay.
            bound = b.model.bind_tools(copy.deepcopy(tool_specs))
            result = self._call(b, lambda bound=bound: bound.invoke(messages), purpose)
            if result is None:
                continue
            ai, latency = result
            usage = dict(getattr(ai, "usage_metadata", None) or {})
            self._record_ok(b, purpose, latency, usage, hit=False)
            clean = AIMessage(content=ai.content, tool_calls=ai.tool_calls)
            b.cache.put(key, {"message": message_to_dict(clean), "usage": usage})
            return clean
        self._no_answer(purpose)
        return None

    def _no_answer(self, purpose: str) -> None:
        if all(b.model is None for b in self.backends if not b.down):
            self.cache_misses += 1
        self.telemetry.record("llm_fallback", purpose, reason="no_backend_answered")


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
