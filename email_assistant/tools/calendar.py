"""Mock calendar with tentative holds and a preference-aware slot finder.

The slot finder is the single source of truth for "is this time acceptable":
working hours, Friday cutoff, focus blocks, buffers between meetings, and the
attendee's own working hours (timezone overlap).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from email_assistant.clock import WEEKDAYS, at_time
from email_assistant.schemas.profile import Preferences
from email_assistant.schemas.scheduling import TimeSlot


class CalendarEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = Field(default_factory=list)
    status: Literal["confirmed", "tentative", "cancelled"] = "confirmed"
    thread_id: str | None = None
    description: str = ""


class SlotCheck(BaseModel):
    ok: bool
    reasons: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class MockCalendar:
    def __init__(self, events: list[CalendarEvent], owner_tz: str, prefs: Preferences):
        self._events = {e.id: e for e in events}
        self.tz = owner_tz
        self.prefs = prefs
        self._applied: dict[str, str] = {}  # idempotency_key -> event id
        self._lock = threading.Lock()

    @classmethod
    def from_fixture(cls, path: Path, owner_tz: str, prefs: Preferences) -> MockCalendar:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls([CalendarEvent.model_validate(e) for e in raw["events"]], owner_tz, prefs)

    # ------------------------------------------------------------------ queries
    def list_events(self, include_cancelled: bool = False) -> list[CalendarEvent]:
        with self._lock:
            evs = [e for e in self._events.values() if include_cancelled or e.status != "cancelled"]
        return sorted(evs, key=lambda e: e.start)

    def get(self, event_id: str) -> CalendarEvent | None:
        return self._events.get(event_id)

    def find_by_thread(self, thread_id: str) -> CalendarEvent | None:
        for e in self.list_events():
            if e.thread_id == thread_id and e.status == "confirmed":
                return e
        return None

    def conflicts(
        self, slot: TimeSlot, ignore: set[str] | None = None, buffer: bool = True
    ) -> list[CalendarEvent]:
        pad = timedelta(minutes=self.prefs.meeting_buffer_minutes if buffer else 0)
        padded = TimeSlot(start=slot.start - pad, end=slot.end + pad)
        return [
            e
            for e in self.list_events()
            if e.id not in (ignore or set()) and padded.overlaps(TimeSlot(start=e.start, end=e.end))
        ]

    def check_slot(
        self,
        slot: TimeSlot,
        attendee_tz: str | None = None,
        ignore: set[str] | None = None,
    ) -> SlotCheck:
        """Deterministic verifier used for both LLM and rule-based proposals."""
        reasons: list[str] = []
        tz = ZoneInfo(self.tz)
        s, e = slot.start.astimezone(tz), slot.end.astimezone(tz)
        wh = self.prefs.working_hours
        day = WEEKDAYS[s.weekday()]
        if e <= s:
            reasons.append("end_before_start")
        if day not in wh.days:
            reasons.append("non_working_day")
        if s < at_time(s.date(), wh.start, self.tz) or e > at_time(s.date(), wh.end, self.tz):
            reasons.append("outside_working_hours")
        if day == "fri" and self.prefs.friday_meeting_cutoff:
            if e > at_time(s.date(), self.prefs.friday_meeting_cutoff, self.tz):
                reasons.append("after_friday_cutoff")
        for fb in self.prefs.focus_blocks:
            if fb.day == day:
                block = TimeSlot(
                    start=at_time(s.date(), fb.start, self.tz),
                    end=at_time(s.date(), fb.end, self.tz),
                )
                if slot.overlaps(block):
                    reasons.append("focus_block")
        if attendee_tz:
            a_tz = ZoneInfo(attendee_tz)
            a_s, a_e = slot.start.astimezone(a_tz), slot.end.astimezone(a_tz)
            a_start = at_time(a_s.date(), wh.start, attendee_tz)
            a_end = at_time(a_s.date(), wh.end, attendee_tz)
            if a_s < a_start or a_e > a_end:
                reasons.append("outside_attendee_hours")
        clash = self.conflicts(slot, ignore=ignore, buffer=False)
        if clash:
            reasons.append("conflict")
        elif self.conflicts(slot, ignore=ignore, buffer=True):
            reasons.append("buffer_too_short")
        return SlotCheck(ok=not reasons, reasons=reasons, conflicts=[c.id for c in clash])

    def find_free_slots(
        self,
        after: datetime,
        duration_minutes: int,
        count: int = 3,
        attendee_tz: str | None = None,
        days_ahead: int = 10,
        step_minutes: int = 30,
        ignore: set[str] | None = None,
        window: tuple[str, str] | None = None,
        only_weekday: str | None = None,
    ) -> list[TimeSlot]:
        tz = ZoneInfo(self.tz)
        start = after.astimezone(tz)
        # round up to the next step boundary
        minute = (start.minute // step_minutes + 1) * step_minutes
        cursor = start.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
        horizon = start + timedelta(days=days_ahead)
        dur = timedelta(minutes=duration_minutes)
        out: list[TimeSlot] = []
        while cursor < horizon and len(out) < count:
            day = WEEKDAYS[cursor.weekday()]
            in_window = True
            if window:
                in_window = (
                    at_time(cursor.date(), window[0], self.tz)
                    <= cursor
                    <= at_time(cursor.date(), window[1], self.tz) - dur
                )
            if (only_weekday is None or day == only_weekday) and in_window:
                slot = TimeSlot(start=cursor, end=cursor + dur)
                if self.check_slot(slot, attendee_tz=attendee_tz, ignore=ignore).ok:
                    out.append(slot)
            cursor += timedelta(minutes=step_minutes)
        return out

    # ---------------------------------------------------------------- mutations
    def _once(self, key: str, fn) -> str:
        with self._lock:
            if key in self._applied:
                return self._applied[key]
        result = fn()
        with self._lock:
            self._applied[key] = result
        return result

    def place_hold(
        self,
        hold_id: str,
        title: str,
        slot: TimeSlot,
        attendees: list[str],
        thread_id: str,
        idempotency_key: str,
    ) -> str:
        def _do() -> str:
            with self._lock:
                self._events[hold_id] = CalendarEvent(
                    id=hold_id,
                    title=title,
                    start=slot.start,
                    end=slot.end,
                    attendees=attendees,
                    status="tentative",
                    thread_id=thread_id,
                )
            return hold_id

        return self._once(idempotency_key, _do)

    def confirm(
        self, hold_id: str, title: str, slot: TimeSlot, attendees: list[str], idempotency_key: str
    ) -> str:
        def _do() -> str:
            with self._lock:
                ev = self._events.get(hold_id)
                if ev is None:
                    raise KeyError(f"hold {hold_id} not found")
                self._events[hold_id] = ev.model_copy(
                    update={
                        "status": "confirmed",
                        "title": title,
                        "start": slot.start,
                        "end": slot.end,
                        "attendees": attendees,
                    }
                )
            return hold_id

        return self._once(idempotency_key, _do)

    def release(self, hold_id: str) -> None:
        with self._lock:
            ev = self._events.get(hold_id)
            if ev and ev.status == "tentative":
                self._events[hold_id] = ev.model_copy(update={"status": "cancelled"})

    def update(self, event_id: str, slot: TimeSlot, idempotency_key: str) -> str:
        def _do() -> str:
            with self._lock:
                ev = self._events[event_id]
                self._events[event_id] = ev.model_copy(
                    update={"start": slot.start, "end": slot.end}
                )
            return event_id

        return self._once(idempotency_key, _do)

    def cancel(self, event_id: str, idempotency_key: str) -> str:
        def _do() -> str:
            with self._lock:
                ev = self._events[event_id]
                self._events[event_id] = ev.model_copy(update={"status": "cancelled"})
            return event_id

        return self._once(idempotency_key, _do)
