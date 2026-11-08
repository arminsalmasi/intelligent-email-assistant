"""LangGraph state for the per-thread graph.

Design: a TypedDict (cheap partial updates, per-field reducers) whose values
are Pydantic models validated at the boundaries that produce them. Actions use
a merge-by-id reducer so a node returning an updated action replaces it instead
of appending a duplicate.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from email_assistant.schemas.actions import ProposedAction
from email_assistant.schemas.email import EmailThread
from email_assistant.schemas.scheduling import SchedulingDecision
from email_assistant.schemas.triage import ThreadSummary, TriageResult


class GuardResult(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    is_threat: bool = False
    threat_type: str | None = None  # prompt_injection | phishing


def merge_actions(
    left: dict[str, ProposedAction] | None, right: dict[str, ProposedAction] | None
) -> dict[str, ProposedAction]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class ThreadState(TypedDict, total=False):
    thread_id: str
    thread: EmailThread
    guard: GuardResult
    triage: TriageResult
    summary: ThreadSummary
    plan: list[str]
    completed: Annotated[list[str], operator.add]
    scheduling: SchedulingDecision
    actions: Annotated[dict[str, ProposedAction], merge_actions]
    review_round: int
    audit: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
