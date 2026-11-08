"""Structured JSON logging with correlation context (run_id, thread_id, node)."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("log_ctx", default={})

LOGGER_NAME = "email_assistant"


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **_ctx.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            data.update(fields)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = " ".join(f"{k}={v}" for k, v in _ctx.get().items())
        fields = getattr(record, "fields", {}) or {}
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        return f"{record.levelname:<7} {record.getMessage()} {ctx} {extra}".rstrip()


def setup_logging(level: str = "INFO", json_output: bool = True) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter() if json_output else PlainFormatter())
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    get_logger().log(level, event, extra={"fields": {"event": event, **fields}})


@contextmanager
def log_context(**values: Any):
    token = _ctx.set({**_ctx.get(), **values})
    try:
        yield
    finally:
        _ctx.reset(token)


def current_context() -> dict[str, Any]:
    return dict(_ctx.get())
