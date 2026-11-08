"""Scheduling and follow-up extraction schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class MeetingRequest(BaseModel):
    """What the sender asks for, as extracted from the email (LLM or rules)."""

    kind: Literal["new", "reschedule", "cancel"] = "new"
    requested_start: str | None = Field(
        default=None,
        description="Requested start as ISO-8601 WITH the sender's UTC offset, or null "
        "if no concrete time is given. Resolve relative dates against 'now'.",
    )
    requested_timezone: str | None = Field(
        default=None, description="IANA timezone the sender expressed the time in."
    )
    preferred_window: str | None = Field(
        default=None, description="Vague preference, e.g. 'Thursday afternoon'."
    )
    duration_minutes: int = Field(default=30, ge=5, le=480)
    topic: str = ""
    source: SkipJsonSchema[str] = "llm"


class TimeSlot(BaseModel):
    start: datetime
    end: datetime

    def overlaps(self, other: TimeSlot) -> bool:
        return self.start < other.end and other.start < self.end

    def label(self, tz: str | None = None) -> str:
        from zoneinfo import ZoneInfo

        s = self.start.astimezone(ZoneInfo(tz)) if tz else self.start
        e = self.end.astimezone(ZoneInfo(tz)) if tz else self.end
        return f"{s:%a %d %b %H:%M}-{e:%H:%M} ({s.tzname()})"


class SchedulingDecision(BaseModel):
    """Output of the scheduler worker; verified deterministically before use."""

    kind: Literal["new", "reschedule", "cancel"]
    outcome: Literal["accept_requested", "propose_alternatives", "cancel", "no_slot"]
    chosen: TimeSlot | None = None
    alternatives: list[TimeSlot] = Field(default_factory=list)
    linked_event_id: str | None = None
    attendees: list[str] = Field(default_factory=list)
    title: str = ""
    rationale: str = ""
    warnings: list[str] = Field(default_factory=list)
    source: SkipJsonSchema[str] = "deterministic"


class Deadline(BaseModel):
    description: str
    due: str = Field(description="ISO-8601 datetime with offset, resolved against 'now'.")


class DeadlineExtraction(BaseModel):
    deadlines: list[Deadline] = Field(default_factory=list)
    source: SkipJsonSchema[str] = "llm"
