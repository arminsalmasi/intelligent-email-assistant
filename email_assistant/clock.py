"""Injectable clock and business-time helpers.

Every component asks the clock for "now" instead of calling datetime.now(), so
tests, the golden eval and the demo are reproducible.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def __init__(self, tz: str = "Europe/Stockholm"):
        self.tz = ZoneInfo(tz)

    def now(self) -> datetime:
        return datetime.now(self.tz)


class FixedClock:
    def __init__(self, at: datetime):
        if at.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        self._at = at

    def now(self) -> datetime:
        return self._at

    def advance(self, delta: timedelta) -> None:
        self._at = self._at + delta


def make_clock(fixed_now: str | None, tz: str) -> Clock:
    if fixed_now:
        return FixedClock(datetime.fromisoformat(fixed_now).astimezone(ZoneInfo(tz)))
    return SystemClock(tz)


def ensure_aware(dt: datetime, tz: str) -> datetime:
    """Interpret naive datetimes in the given zone; convert aware ones to it."""
    zone = ZoneInfo(tz)
    return dt.replace(tzinfo=zone) if dt.tzinfo is None else dt.astimezone(zone)


def is_business_day(d: date) -> bool:
    return d.weekday() < 5


def add_business_days(start: datetime, days: int) -> datetime:
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if is_business_day(current.date()):
            added += 1
    return current


def at_time(d: date, hhmm: str, tz: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.combine(d, time(h, m), tzinfo=ZoneInfo(tz))
