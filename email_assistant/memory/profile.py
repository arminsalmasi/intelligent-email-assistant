"""Profile loading, semantic-memory seeding and protected-term extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from email_assistant.memory.store import MemoryStore
from email_assistant.schemas.memory import MemoryKind, MemorySource, Sensitivity
from email_assistant.schemas.profile import OrgFact, UserProfile

_NUMBER = re.compile(r"\d[\d,. ]{2,}\d")
_CODENAME = re.compile(r"\b(?:Project|codename)\s+([A-Z][a-z]+)")


def load_profile(path: Path) -> UserProfile:
    return UserProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def protected_terms(fact: OrgFact) -> list[str]:
    """Strings that must not leave the org if the fact is confidential/internal.

    Owner-curated terms win. Otherwise: amounts with common spellings
    (350,000 / 350 000 / 350000 / 350k) and "Project X" style codenames.
    """
    if fact.protected_terms is not None:
        return sorted(set(fact.protected_terms))
    if fact.sensitivity == Sensitivity.PUBLIC:
        return []
    terms: set[str] = set()
    for raw in _NUMBER.findall(fact.content):
        digits = re.sub(r"[^\d]", "", raw)
        if len(digits) < 4 or (len(digits) == 4 and 1900 <= int(digits) <= 2100):
            continue  # skip short numbers and years
        terms.update(
            {raw.strip(), digits, f"{int(digits):,}", f"{int(digits):,}".replace(",", " ")}
        )
        if int(digits) % 1000 == 0:
            terms.add(f"{int(digits) // 1000}k")
    terms.update(_CODENAME.findall(fact.content))
    return sorted(terms)


def seed_memory(store: MemoryStore, profile: UserProfile) -> int:
    n = 0
    prefs = profile.preferences
    pref_items = {
        "sign_off": prefs.sign_off,
        "tone": prefs.tone,
        "working_hours": f"{prefs.working_hours.start}-{prefs.working_hours.end} "
        f"{','.join(prefs.working_hours.days)} ({profile.owner.timezone})",
        "meeting_rules": (
            f"Default {prefs.default_meeting_minutes} min, {prefs.meeting_buffer_minutes} min "
            f"buffer, no meetings after {prefs.friday_meeting_cutoff} on Fridays, focus blocks: "
            + ", ".join(f"{b.day} {b.start}-{b.end}" for b in prefs.focus_blocks)
        ),
        "follow_up_policy": f"Follow up after {prefs.follow_up_business_days} business days "
        f"without reply; remind {prefs.reminder_lead_hours}h before deadlines.",
    }
    for key, value in pref_items.items():
        store.put(
            MemoryKind.SEMANTIC,
            "preferences",
            key,
            value,
            source=MemorySource.USER_EXPLICIT,
            sensitivity=Sensitivity.INTERNAL,
        )
        n += 1
    for c in profile.contacts:
        content = (
            f"{c.name} <{c.email}> - {c.role} at {c.organization} ({c.relationship})"
            f"{', VIP' if c.vip else ''}; timezone {c.timezone or 'unknown'}; language {c.language}."
        )
        store.put(
            MemoryKind.SEMANTIC,
            "contacts",
            c.email.lower(),
            content,
            metadata=c.model_dump(),
            source=MemorySource.USER_EXPLICIT,
            sensitivity=Sensitivity.INTERNAL,
        )
        n += 1
    for f in profile.facts:
        store.put(
            MemoryKind.SEMANTIC,
            "facts",
            f.key,
            f.content,
            metadata={"tags": f.tags, "protected_terms": protected_terms(f)},
            source=MemorySource.USER_EXPLICIT,
            sensitivity=f.sensitivity,
        )
        n += 1
    return n
