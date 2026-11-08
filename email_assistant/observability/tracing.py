"""Node instrumentation + optional external tracing backends.

`traced_node` wraps every graph node: timing, structured log, telemetry event
and error capture (a failing worker degrades the thread, it doesn't crash the
run). External backends are opt-in via TRACING_BACKEND:
* mlflow    -> mlflow.langchain.autolog() (LangGraph spans, LLM calls, tokens)
* langsmith -> LANGSMITH_TRACING env (needs LANGSMITH_API_KEY)
"""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from langgraph.errors import GraphInterrupt

from email_assistant.config import Settings
from email_assistant.observability.logging import log_context, log_event
from email_assistant.observability.metrics import Telemetry


def traced_node(name: str, telemetry: Telemetry) -> Callable:
    def decorator(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        @functools.wraps(fn)
        def wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
            started = time.perf_counter()
            with log_context(thread_id=state.get("thread_id"), node=name):
                try:
                    update = fn(state, *args, **kwargs) or {}
                except GraphInterrupt:
                    telemetry.record(
                        "node",
                        name,
                        (time.perf_counter() - started) * 1000,
                        ok=True,
                        interrupted=True,
                    )
                    raise
                except Exception as exc:
                    ms = (time.perf_counter() - started) * 1000
                    telemetry.record("node", name, ms, ok=False)
                    telemetry.record("error", name, error=type(exc).__name__, detail=str(exc)[:300])
                    log_event("node.failed", logging.ERROR, error=repr(exc)[:300])
                    return {"errors": [f"{name}: {type(exc).__name__}: {exc}"], "completed": [name]}
                ms = (time.perf_counter() - started) * 1000
                telemetry.record("node", name, ms, ok=True)
                log_event("node.done", latency_ms=round(ms, 1))
                return update

        return wrapper

    return decorator


def setup_tracing(settings: Settings) -> str:
    if settings.tracing_backend == "mlflow":
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment)
            mlflow.langchain.autolog()
            return "mlflow"
        except Exception as exc:  # tracing must never break the app
            log_event("tracing.unavailable", logging.WARNING, backend="mlflow", error=str(exc))
            return "none"
    if settings.tracing_backend == "langsmith":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", settings.mlflow_experiment)
        return "langsmith"
    return "none"
