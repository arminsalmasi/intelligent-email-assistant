"""Minimal RFC 5545 iCalendar generation for meeting confirmations."""

from __future__ import annotations

from datetime import UTC, datetime


def _fmt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def build_ics(
    uid: str,
    title: str,
    start: datetime,
    end: datetime,
    organizer: str,
    attendees: list[str],
    stamp: datetime,
    method: str = "REQUEST",
    sequence: int = 0,
    description: str = "",
) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//email-assistant//EN",
        f"METHOD:{method}",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"SEQUENCE:{sequence}",
        f"DTSTAMP:{_fmt(stamp)}",
        f"DTSTART:{_fmt(start)}",
        f"DTEND:{_fmt(end)}",
        f"SUMMARY:{_escape(title)}",
        f"ORGANIZER:mailto:{organizer}",
    ]
    lines += [f"ATTENDEE;ROLE=REQ-PARTICIPANT;RSVP=TRUE:mailto:{a}" for a in attendees]
    if description:
        lines.append(f"DESCRIPTION:{_escape(description)}")
    lines += [
        "STATUS:CANCELLED" if method == "CANCEL" else "STATUS:CONFIRMED",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"
