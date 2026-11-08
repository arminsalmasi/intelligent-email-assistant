"""Episodic memory: handled emails + the reply a human approved.

Used two ways:
* template reuse  - near-duplicate email (score >= reuse threshold, same
  category, same audience) -> adapt the approved reply deterministically, no
  LLM call. Human approval still applies. Templates containing digits are never
  reused verbatim (dates/amounts go stale).
* few-shot example - similar email (score >= example threshold) -> shown to the
  drafter as an example of the owner's style.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel

from email_assistant.memory.store import MemoryStore
from email_assistant.schemas.email import EmailThread
from email_assistant.schemas.memory import MemoryHit, MemoryKind, MemorySource, Sensitivity
from email_assistant.schemas.triage import TriageResult

EPISODE_TTL_DAYS = 90
_GREETING = re.compile(r"^(hi|hello|hej|hola|dear)\s+[^,\n]+,", re.IGNORECASE)


class EpisodeMatch(BaseModel):
    mode: Literal["template", "example"]
    hit: MemoryHit


def episode_text(thread: EmailThread, triage: TriageResult) -> str:
    msg = thread.latest_inbound or thread.latest
    return f"{triage.category.value} | {thread.subject}\n{msg.body[:1500]}"


class EpisodicMemory:
    def __init__(self, store: MemoryStore, reuse_threshold: float, example_threshold: float):
        self.store = store
        self.reuse_threshold = reuse_threshold
        self.example_threshold = example_threshold

    def record(
        self, thread: EmailThread, triage: TriageResult, reply: str, audience: str, edited: bool
    ) -> None:
        msg = thread.latest_inbound or thread.latest
        self.store.put(
            MemoryKind.EPISODIC,
            "episodes",
            thread.id,
            reply,
            metadata={
                "category": triage.category.value,
                "audience": audience,
                "sender": msg.sender.email,
                "subject": thread.subject,
                "language": triage.language,
                "edited": edited,
            },
            source=MemorySource.HITL_FEEDBACK,
            sensitivity=Sensitivity.INTERNAL if audience == "internal" else Sensitivity.PUBLIC,
            expires_at=self.store.clock.now() + timedelta(days=EPISODE_TTL_DAYS),
            embed_text=episode_text(thread, triage),
        )

    def find(self, thread: EmailThread, triage: TriageResult, audience: str) -> list[EpisodeMatch]:
        hits = self.store.search(
            episode_text(thread, triage),
            k=4,
            kind=MemoryKind.EPISODIC,
            min_score=self.example_threshold,
        )
        matches: list[EpisodeMatch] = []
        for h in hits:
            meta = h.record.metadata
            if h.record.key == thread.id:
                continue
            if meta.get("category") != triage.category.value or meta.get("audience") != audience:
                continue
            reusable = h.score >= self.reuse_threshold and not re.search(r"\d", h.record.content)
            matches.append(EpisodeMatch(mode="template" if reusable else "example", hit=h))
        return matches


def adapt_template(template: str, first_name: str) -> str:
    """Swap the greeting name; everything else stays as the human approved it."""
    if _GREETING.match(template):
        return _GREETING.sub(lambda m: f"{m.group(1)} {first_name},", template, count=1)
    return template
