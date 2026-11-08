"""Long-term memory records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryKind(StrEnum):
    SEMANTIC = "semantic"  # facts: preferences, contacts, org knowledge
    EPISODIC = "episodic"  # handled emails + the reply a human approved
    PROCEDURAL = "procedural"  # how-to: lessons distilled from human edits


class MemorySource(StrEnum):
    USER_EXPLICIT = "user_explicit"
    HITL_FEEDBACK = "hitl_feedback"
    OBSERVED = "observed"


# Higher wins when two writes target the same (namespace, key).
SOURCE_PRIORITY = {
    MemorySource.USER_EXPLICIT: 3,
    MemorySource.HITL_FEEDBACK: 2,
    MemorySource.OBSERVED: 1,
}


class Sensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"  # fine inside the org, never to external recipients
    CONFIDENTIAL = "confidential"  # never to external recipients, flagged internally


class MemoryRecord(BaseModel):
    id: str
    kind: MemoryKind
    namespace: str  # e.g. "preferences", "contacts", "facts", "episodes", "edits"
    key: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    source: MemorySource = MemorySource.OBSERVED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None = None
    use_count: int = 0
    expires_at: datetime | None = None


class MemoryHit(BaseModel):
    record: MemoryRecord
    score: float
