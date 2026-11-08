"""REST API over the assistant (used by the dashboard; OpenAPI docs at /docs)."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from email_assistant.config import get_settings
from email_assistant.graph.orchestrator import InboxItem
from email_assistant.schemas.review import ReviewRequest, ReviewResponse
from email_assistant.service import AssistantService

_lock = threading.Lock()
_svc: AssistantService | None = None


def service() -> AssistantService:
    if _svc is None:
        raise HTTPException(503, "service not ready")
    return _svc


def _build() -> AssistantService:
    return AssistantService.create(get_settings())


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _svc
    _svc = _build()
    yield


app = FastAPI(
    title="Email Assistant API",
    version="0.2.0",
    lifespan=lifespan,
    description="Agentic email assistant with human-in-the-loop approval.",
)


class ProcessResult(BaseModel):
    processed: int
    pending_reviews: int
    quarantined: int


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    s = service()
    return {
        "status": "ready",
        "llm": type(s.deps.llm).__name__,
        "llm_provider": s.settings.llm_provider,
        "embedder": s.deps.memory.embedder.name,
        "tracing": s.tracing,
        "now": s.deps.clock.now().isoformat(),
    }


@app.post("/inbox/process", response_model=ProcessResult)
def process_inbox() -> ProcessResult:
    s = service()
    with _lock:
        items = s.orchestrator.process_inbox()
    return ProcessResult(
        processed=len(items),
        pending_reviews=sum(i.status == "pending_review" for i in items),
        quarantined=sum(i.status == "quarantined" for i in items),
    )


@app.get("/inbox", response_model=list[InboxItem])
def inbox() -> list[InboxItem]:
    return service().orchestrator.ranked()


@app.get("/threads/{thread_id}")
def thread(thread_id: str) -> dict[str, Any]:
    s = service()
    t = s.deps.mail.get_thread(thread_id)
    if t is None:
        raise HTTPException(404, "thread not found")
    item = s.orchestrator.items.get(thread_id)
    return {
        "thread": t.model_dump(mode="json", by_alias=True),
        "item": item.model_dump(mode="json") if item else None,
        "audit": s.orchestrator.audit(thread_id) if item else [],
    }


@app.get("/reviews", response_model=list[ReviewRequest])
def reviews() -> list[ReviewRequest]:
    return service().orchestrator.pending_reviews()


@app.post("/reviews/{thread_id}", response_model=InboxItem)
def submit_review(thread_id: str, response: ReviewResponse) -> InboxItem:
    s = service()
    with _lock:
        try:
            return s.orchestrator.submit_review(thread_id, response)
        except KeyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(422, exc.errors()) from exc


@app.get("/calendar")
def calendar(include_cancelled: bool = True) -> list[dict[str, Any]]:
    return [
        e.model_dump(mode="json")
        for e in service().deps.calendar.list_events(include_cancelled=include_cancelled)
    ]


@app.get("/reminders")
def reminders(status: str | None = None) -> dict[str, Any]:
    s = service()
    digest = s.deps.reminders.digest(s.deps.clock.now())
    return {
        "items": [r.model_dump(mode="json") for r in s.deps.reminders.list(status=status)],
        "digest": {k: [r.id for r in v] for k, v in digest.items()},
    }


@app.post("/reminders/{reminder_id}/complete")
def complete_reminder(reminder_id: str) -> dict[str, bool]:
    s = service()
    return {"ok": s.deps.reminders.complete(reminder_id, s.deps.clock.now())}


@app.post("/reminders/{reminder_id}/snooze")
def snooze_reminder(reminder_id: str, hours: int = 24) -> dict[str, bool]:
    return {"ok": service().deps.reminders.snooze(reminder_id, hours)}


@app.get("/memories")
def memories(kind: str | None = None) -> list[dict[str, Any]]:
    from email_assistant.schemas.memory import MemoryKind

    k = MemoryKind(kind) if kind else None
    return [r.model_dump(mode="json") for r in service().deps.memory.list(kind=k)]


@app.get("/sent")
def sent() -> list[dict[str, Any]]:
    return service().deps.mail.sent_messages()


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, Any]:
    return service().deps.telemetry.summary()


@app.get("/metrics/events")
def metrics_events(type: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    return service().deps.telemetry.events(type, limit)


@app.post("/admin/reset")
def reset() -> dict[str, str]:
    """Demo helper: fresh mailbox/calendar/run. Long-term memory is kept on purpose."""
    global _svc
    with _lock:
        _svc = _build()
    return {"status": "reset"}
