"""Pydantic schemas: the structured contract between agents, tools and humans."""

from email_assistant.schemas.actions import (
    ActionStatus,
    ActionType,
    ProposedAction,
    RiskTier,
)
from email_assistant.schemas.email import EmailAddress, EmailMessage, EmailThread
from email_assistant.schemas.memory import MemoryKind, MemoryRecord, MemorySource, Sensitivity
from email_assistant.schemas.review import HumanDecision, ReviewRequest, ReviewResponse
from email_assistant.schemas.scheduling import MeetingRequest, SchedulingDecision, TimeSlot
from email_assistant.schemas.state import GuardResult, ThreadState
from email_assistant.schemas.triage import Category, ThreadSummary, TriageResult, Urgency

__all__ = [
    "ActionStatus",
    "ActionType",
    "Category",
    "EmailAddress",
    "EmailMessage",
    "EmailThread",
    "GuardResult",
    "HumanDecision",
    "MeetingRequest",
    "MemoryKind",
    "MemoryRecord",
    "MemorySource",
    "ProposedAction",
    "ReviewRequest",
    "ReviewResponse",
    "RiskTier",
    "SchedulingDecision",
    "Sensitivity",
    "ThreadState",
    "ThreadSummary",
    "TimeSlot",
    "TriageResult",
    "Urgency",
]
