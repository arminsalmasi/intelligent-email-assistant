"""Mock task/reminder service: deadlines, awaiting-reply follow-ups, snooze, digest."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel


class Reminder(BaseModel):
    id: str
    thread_id: str
    title: str
    due: datetime
    reminder_type: Literal["deadline", "awaiting_reply", "follow_up", "payment"]
    status: Literal["pending", "completed", "snoozed", "cancelled"] = "pending"
    notes: str = ""
    created_at: datetime
    completed_at: datetime | None = None
    completion_reason: str | None = None


class MockReminders:
    def __init__(self) -> None:
        self._items: dict[str, Reminder] = {}
        self._lock = threading.Lock()

    def create(
        self,
        idempotency_key: str,
        thread_id: str,
        title: str,
        due: datetime,
        reminder_type: str,
        notes: str,
        now: datetime,
    ) -> str:
        rid = f"rem-{idempotency_key[:10]}"
        with self._lock:
            if rid not in self._items:
                self._items[rid] = Reminder(
                    id=rid,
                    thread_id=thread_id,
                    title=title,
                    due=due,
                    reminder_type=reminder_type,
                    notes=notes,
                    created_at=now,  # type: ignore[arg-type]
                )
        return rid

    def list(self, status: str | None = None, thread_id: str | None = None) -> list[Reminder]:
        with self._lock:
            items = list(self._items.values())
        if status:
            items = [r for r in items if r.status == status]
        if thread_id:
            items = [r for r in items if r.thread_id == thread_id]
        return sorted(items, key=lambda r: r.due)

    def complete(self, reminder_id: str, now: datetime, reason: str = "manual") -> bool:
        with self._lock:
            r = self._items.get(reminder_id)
            if not r or r.status == "completed":
                return False
            self._items[reminder_id] = r.model_copy(
                update={"status": "completed", "completed_at": now, "completion_reason": reason}
            )
            return True

    def snooze(self, reminder_id: str, hours: int) -> bool:
        with self._lock:
            r = self._items.get(reminder_id)
            if not r or r.status in ("completed", "cancelled"):
                return False
            self._items[reminder_id] = r.model_copy(
                update={"status": "snoozed", "due": r.due + timedelta(hours=hours)}
            )
            return True

    def resolve_awaiting_reply(self, thread_id: str, reply_at: datetime) -> list[str]:
        """Auto-complete awaiting-reply reminders once the counterpart answers."""
        done = []
        for r in self.list(thread_id=thread_id):
            if r.reminder_type == "awaiting_reply" and r.status in ("pending", "snoozed"):
                if r.created_at <= reply_at and self.complete(r.id, reply_at, "reply_received"):
                    done.append(r.id)
        return done

    def digest(self, now: datetime, horizon_hours: int = 48) -> dict[str, list[Reminder]]:
        active = [r for r in self.list() if r.status in ("pending", "snoozed")]
        limit = now + timedelta(hours=horizon_hours)
        return {
            "overdue": [r for r in active if r.due < now],
            "due_soon": [r for r in active if now <= r.due <= limit],
            "later": [r for r in active if r.due > limit],
        }
