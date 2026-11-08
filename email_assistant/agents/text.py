"""Shared deterministic text utilities (language detection, date resolution)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from email_assistant.clock import at_time

_SV = {
    "och",
    "att",
    "det",
    "som",
    "för",
    "inte",
    "med",
    "jag",
    "vi",
    "kan",
    "hej",
    "tack",
    "på",
    "är",
    "har",
    "ses",
    "nästa",
    "vecka",
    "vänliga",
    "hälsningar",
}
_ES = {
    "que",
    "los",
    "las",
    "por",
    "para",
    "una",
    "con",
    "del",
    "hola",
    "gracias",
    "saludos",
    "contrato",
    "favor",
    "podrías",
    "lunes",
    "semana",
    "oferta",
    "nuestro",
}

WEEKDAY_WORDS = {
    0: ["monday", "mon", "måndag", "lunes"],
    1: ["tuesday", "tue", "tisdag", "martes"],
    2: ["wednesday", "wed", "onsdag", "miércoles", "miercoles"],
    3: ["thursday", "thu", "torsdag", "jueves"],
    4: ["friday", "fri", "fredag", "viernes"],
    5: ["saturday", "lördag", "sábado"],
    6: ["sunday", "söndag", "domingo"],
}
_WEEKDAY_LOOKUP = {w: d for d, words in WEEKDAY_WORDS.items() for w in words}
WEEKDAY_RE = "|".join(sorted(_WEEKDAY_LOOKUP, key=len, reverse=True))
_MONTHS = [
    ["january", "januari", "enero"],
    ["february", "februari", "febrero"],
    ["march", "mars", "marzo"],
    ["april", "abril"],
    ["may", "maj", "mayo"],
    ["june", "juni", "junio"],
    ["july", "juli", "julio"],
    ["august", "augusti", "agosto"],
    ["september", "septiembre"],
    ["october", "oktober", "octubre"],
    ["november", "noviembre"],
    ["december", "diciembre"],
]
_MONTH_LOOKUP = {w: i + 1 for i, words in enumerate(_MONTHS) for w in words}
MONTH_RE = "|".join(sorted(_MONTH_LOOKUP, key=len, reverse=True))
TZ_HINTS = {
    "new york": "America/New_York",
    "eastern": "America/New_York",
    "est": "America/New_York",
    "edt": "America/New_York",
    "london": "Europe/London",
    "gmt": "Europe/London",
    "stockholm": "Europe/Stockholm",
    "cet": "Europe/Stockholm",
    "swedish time": "Europe/Stockholm",
    "lima": "America/Lima",
    "peru": "America/Lima",
    "malta": "Europe/Malta",
}


def timezone_hint(text: str) -> str | None:
    t = text.lower()
    for hint, tz in TZ_HINTS.items():
        if re.search(rf"\b{re.escape(hint)}\b", t):
            return tz
    return None


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zåäöéñáíóú]+", text.lower())
    sv = sum(w in _SV for w in words) + 2 * sum(c in text.lower() for c in "åäö")
    es = sum(w in _ES for w in words) + 2 * sum(c in text.lower() for c in "ñ¿¡")
    if max(sv, es) < 3:
        return "en"
    return "sv" if sv > es else "es"


def weekday_from_word(word: str) -> int | None:
    return _WEEKDAY_LOOKUP.get(word.lower())


def parse_time(text: str) -> tuple[int, int] | None:
    m = re.search(r"\b(?:kl\.?\s*|at\s+|a las\s+)?(\d{1,2})(?::|\.)(\d{2})\b", text)
    if m and int(m.group(1)) < 24:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text, re.IGNORECASE)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(2).lower() == "pm" else 0)
        return h, 0
    m = re.search(r"\bkl\.?\s*(\d{1,2})\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), 0
    if re.search(r"\b(eod|end of (the )?day)\b", text, re.IGNORECASE):
        return 17, 0
    return None


def resolve_day_phrase(phrase: str, now: datetime, tz: str) -> datetime | None:
    """Resolve 'Friday EOD', 'tisdag nästa vecka kl 10', '2026-11-20', 'tomorrow 12:00'."""
    p = phrase.lower()
    zone = ZoneInfo(tz)
    local_now = now.astimezone(zone)
    day: datetime | None = None
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", p)
    dm = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:de\s+)?({MONTH_RE})\b", p)
    if m:
        day = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=zone)
    elif dm:
        month = _MONTH_LOOKUP[dm.group(2)]
        year = local_now.year + (1 if month < local_now.month else 0)
        day = datetime(year, month, int(dm.group(1)), tzinfo=zone)
    elif re.search(r"\b(tomorrow|imorgon|mañana)\b", p):
        day = local_now + timedelta(days=1)
    elif re.search(r"\b(today|idag|hoy)\b", p):
        day = local_now
    else:
        m = re.search(rf"\b({WEEKDAY_RE})\b", p)
        if m:
            wd = weekday_from_word(m.group(1))
            nxt = bool(
                re.search(
                    r"(next week|nästa vecka|la próxima semana|próximo|next\s+" + m.group(1) + ")",
                    p,
                )
            )
            if wd is not None:
                days = (wd - local_now.weekday()) % 7
                if nxt:
                    days += 7
                day = local_now + timedelta(days=days)
    if day is None:
        return None
    hm = parse_time(p) or (17, 0)
    return at_time(day.date(), f"{hm[0]:02d}:{hm[1]:02d}", tz)
