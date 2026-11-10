"""Inbox orchestrator: fetch -> guard + triage all -> rank -> per-thread graph.

Owns the review queue: threads paused at the HITL interrupt wait here until a
human decision resumes them (from CLI, API or dashboard).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from email_assistant.agents.triage import TriageAgent, is_quarantined, priority_score
from email_assistant.deps import Deps
from email_assistant.memory.inbox_store import InboxStore
from email_assistant.observability.logging import log_context, log_event
from email_assistant.schemas.actions import ActionStatus, ProposedAction, apply_edits
from email_assistant.schemas.review import ReviewRequest, ReviewResponse
from email_assistant.schemas.state import GuardResult
from email_assistant.schemas.triage import ThreadSummary, TriageResult

Status = Literal["pending_review", "quarantined", "done", "archived", "error"]


class InboxItem(BaseModel):
    thread_id: str
    subject: str
    sender: str
    received_at: datetime
    score: float
    rank: int = 0
    status: Status = "done"
    triage: TriageResult
    guard: GuardResult
    summary: ThreadSummary | None = None
    actions: list[ProposedAction] = []
    errors: list[str] = []
    review: ReviewRequest | None = None
    message_id: str = ""  # latest message processed; a new message re-opens the thread
    attempt: int = 0


class ReviewRejected(ValueError):
    """The submitted review cannot be applied (stale round, unknown action, bad edit)."""


class InboxOrchestrator:
    def __init__(self, deps: Deps, graph: Any, store: InboxStore | None = None):
        self.d = deps
        self.graph = graph
        self.triage_agent = TriageAgent(deps)
        self.store = store or InboxStore(":memory:")
        self.run_id = self.store.epoch  # namespaces checkpoints; stable across restarts
        self.items: dict[str, InboxItem] = {}
        self._restore()

    def _config(self, thread_id: str) -> dict:
        attempt = self.items[thread_id].attempt if thread_id in self.items else 0
        return {"configurable": {"thread_id": f"{thread_id}@{self.run_id}#{attempt}"}}

    def _restore(self) -> None:
        """Reload items after a restart. The checkpointer is the source of truth for
        pending reviews: a review whose checkpoint is gone is re-processed, not shown."""
        for raw in self.store.load():
            item = InboxItem.model_validate_json(raw)
            self.items[item.thread_id] = item
            if item.status == "pending_review":
                if self.graph.get_state(self._config(item.thread_id)).values:
                    self._refresh(item.thread_id)
                else:
                    item.status, item.review = "error", None
                    item.errors = [*item.errors, "checkpoint missing after restart"]
                    self.store.save(item)
        if self.items:
            log_event(
                "inbox.restored",
                items=len(self.items),
                pending_reviews=sum(i.status == "pending_review" for i in self.items.values()),
            )

    def _needs_processing(self, thread_id: str, latest_id: str) -> bool:
        item = self.items.get(thread_id)
        if item is None:
            return True
        if item.status == "pending_review":
            return False  # never restart a thread a human is reviewing
        return item.status == "error" or item.message_id != latest_id

    # --------------------------------------------------------------- intake
    def process_inbox(self, limit: int | None = None) -> list[InboxItem]:
        d = self.d
        now = d.clock.now()
        with log_context(run_id=self.run_id):
            threads = [
                t for t in d.mail.list_threads() if self._needs_processing(t.id, t.latest.id)
            ]
            for t in threads:  # counterpart answered -> close awaiting-reply reminders
                if t.latest.direction == "inbound":
                    d.reminders.resolve_awaiting_reply(t.id, t.latest.sent_at)
            triaged = []
            for t in threads:
                with log_context(thread_id=t.id):
                    guard = d.detector.assess(t)
                    d.telemetry.record(
                        "guard", "quarantined" if guard.is_threat else "clean", guard.risk_score
                    )
                    tri, guard = self.triage_agent.run(t, guard)
                triaged.append((priority_score(t, tri, d, now), t, tri, guard))
            # quarantined threads belong to the security queue: below all other mail
            triaged.sort(key=lambda x: (not is_quarantined(x[3], x[2]), x[0]), reverse=True)
            if limit:
                triaged = triaged[:limit]
            log_event("inbox.ranked", threads=len(triaged))
            for rank, (score, t, tri, guard) in enumerate(triaged, 1):
                msg = t.latest_inbound or t.latest
                prev = self.items.get(t.id)
                self.items[t.id] = InboxItem(
                    message_id=t.latest.id,
                    attempt=prev.attempt + 1 if prev else 0,
                    thread_id=t.id,
                    subject=t.subject,
                    sender=msg.sender.display(),
                    received_at=msg.sent_at,
                    score=score,
                    rank=rank,
                    triage=tri,
                    guard=guard,
                )
                with log_context(thread_id=t.id):
                    self._run(
                        t.id,
                        {
                            "thread_id": t.id,
                            "thread": t,
                            "guard": guard,
                            "triage": tri,
                            "review_round": 0,
                        },
                    )
        return self.ranked()

    def _run(self, thread_id: str, payload: Any) -> None:
        item = self.items[thread_id]
        try:
            self.graph.invoke(payload, self._config(thread_id))
        except Exception as exc:  # one bad thread must not stop the inbox
            self.d.telemetry.record(
                "error", "thread", error=type(exc).__name__, detail=str(exc)[:300]
            )
            item.status, item.errors = "error", [*item.errors, f"{type(exc).__name__}: {exc}"]
            self.store.save(item)
            return
        self._refresh(thread_id)

    def _refresh(self, thread_id: str) -> None:
        item = self.items[thread_id]
        snap = self.graph.get_state(self._config(thread_id))
        values = snap.values
        item.summary = values.get("summary")
        item.actions = list(values.get("actions", {}).values())
        item.errors = values.get("errors", [])
        interrupts = [i for task in snap.tasks for i in task.interrupts]
        if interrupts:
            item.status = "pending_review"
            item.review = ReviewRequest.model_validate(interrupts[0].value)
        else:
            item.review = None
            if item.guard.is_threat:
                item.status = "quarantined"
            elif any(
                a.type.value == "archive" and a.status == ActionStatus.EXECUTED
                for a in item.actions
            ):
                item.status = "archived"
            else:
                item.status = "done"
        self.store.save(item)

    # --------------------------------------------------------------- review
    def pending_reviews(self) -> list[ReviewRequest]:
        return [i.review for i in self.ranked() if i.review is not None]

    def validate_review(self, item: InboxItem, response: ReviewResponse) -> None:
        """Reject stale, mismatched or invalid submissions BEFORE resuming the graph."""
        review = item.review
        assert review is not None
        if response.round is not None and response.round != review.round:
            raise ReviewRejected(f"stale review: round {response.round} != {review.round}")
        pending = {a.id: a for a in review.actions}
        for dec in response.decisions:
            action = pending.get(dec.action_id)
            if action is None:
                raise ReviewRejected(f"unknown or non-pending action {dec.action_id}")
            if dec.decision == "approve" and action.has_hard_violation:
                raise ReviewRejected(f"{dec.action_id} has hard policy findings; edit or reject it")
            if dec.decision == "edit":
                try:
                    apply_edits(action, dec.edits)
                except (ValueError, ValidationError) as exc:
                    raise ReviewRejected(f"invalid edit for {dec.action_id}: {exc}") from exc

    def submit_review(self, thread_id: str, response: ReviewResponse) -> InboxItem:
        item = self.items.get(thread_id)
        if item is None or item.status != "pending_review":
            raise KeyError(f"no pending review for {thread_id}")
        self.validate_review(item, response)
        with log_context(run_id=self.run_id, thread_id=thread_id):
            log_event("review.submitted", decisions=[d.decision for d in response.decisions])
            self._run(thread_id, Command(resume=response.model_dump(mode="json")))
        return self.items[thread_id]

    # ---------------------------------------------------------------- views
    def ranked(self) -> list[InboxItem]:
        return sorted(self.items.values(), key=lambda i: i.rank)

    def state(self, thread_id: str) -> dict[str, Any]:
        return self.graph.get_state(self._config(thread_id)).values

    def audit(self, thread_id: str) -> list[dict[str, Any]]:
        return self.state(thread_id).get("audit", [])
