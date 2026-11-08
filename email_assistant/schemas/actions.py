"""Typed action proposals.

Workers never perform side effects. They emit `ProposedAction`s with a typed
payload (discriminated union on `kind`). The policy engine annotates them, a
human approves/edits/rejects the approval-tier ones, and only the executor
touches the tools.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ActionType(StrEnum):
    SEND_EMAIL = "send_email"
    CALENDAR_HOLD = "calendar_hold"
    CALENDAR_CONFIRM = "calendar_confirm"
    CALENDAR_UPDATE = "calendar_update"
    CALENDAR_CANCEL = "calendar_cancel"
    CREATE_REMINDER = "create_reminder"
    APPLY_LABELS = "apply_labels"
    ARCHIVE = "archive"
    QUARANTINE = "quarantine"


class RiskTier(StrEnum):
    AUTO = "auto"  # internal, reversible: executed without approval
    APPROVAL = "approval"  # outgoing / affects other people: needs a human


# Allow-list. Anything not in this map cannot be proposed at all (no delete,
# no forward). Archive and holds are reversible.
RISK_TIERS: dict[ActionType, RiskTier] = {
    ActionType.SEND_EMAIL: RiskTier.APPROVAL,
    ActionType.CALENDAR_CONFIRM: RiskTier.APPROVAL,
    ActionType.CALENDAR_UPDATE: RiskTier.APPROVAL,
    ActionType.CALENDAR_CANCEL: RiskTier.APPROVAL,
    ActionType.CALENDAR_HOLD: RiskTier.AUTO,
    ActionType.CREATE_REMINDER: RiskTier.AUTO,
    ActionType.APPLY_LABELS: RiskTier.AUTO,
    ActionType.ARCHIVE: RiskTier.AUTO,
    ActionType.QUARANTINE: RiskTier.AUTO,
}


class ActionStatus(StrEnum):
    PROPOSED = "proposed"
    BLOCKED = "blocked"  # hard policy violation
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Attachment(BaseModel):
    filename: str
    content_type: str
    content: str


class SendEmailPayload(BaseModel):
    kind: Literal["send_email"] = "send_email"
    to: list[str]
    cc: list[str] = Field(default_factory=list)
    subject: str
    body: str
    in_reply_to: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    draft_source: str = "llm"  # llm | heuristic | memory_reuse
    used_facts: list[str] = Field(default_factory=list)  # memory keys the draft relied on
    ungrounded_claims: list[str] = Field(default_factory=list)


class CalendarHoldPayload(BaseModel):
    kind: Literal["calendar_hold"] = "calendar_hold"
    hold_id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = Field(default_factory=list)


class CalendarConfirmPayload(BaseModel):
    kind: Literal["calendar_confirm"] = "calendar_confirm"
    hold_id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = Field(default_factory=list)


class CalendarUpdatePayload(BaseModel):
    kind: Literal["calendar_update"] = "calendar_update"
    event_id: str
    start: datetime
    end: datetime


class CalendarCancelPayload(BaseModel):
    kind: Literal["calendar_cancel"] = "calendar_cancel"
    event_id: str
    reason: str = ""


class ReminderPayload(BaseModel):
    kind: Literal["create_reminder"] = "create_reminder"
    title: str
    due: datetime
    reminder_type: Literal["deadline", "awaiting_reply", "follow_up", "payment"] = "deadline"
    deadline: datetime | None = None  # the underlying deadline (due = deadline - lead time)
    notes: str = ""


class LabelsPayload(BaseModel):
    kind: Literal["apply_labels"] = "apply_labels"
    labels: list[str]
    mark_read: bool = True


class ArchivePayload(BaseModel):
    kind: Literal["archive"] = "archive"
    reason: str


class QuarantinePayload(BaseModel):
    kind: Literal["quarantine"] = "quarantine"
    reason: str
    risk_score: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)


ActionPayload = Annotated[
    SendEmailPayload
    | CalendarHoldPayload
    | CalendarConfirmPayload
    | CalendarUpdatePayload
    | CalendarCancelPayload
    | ReminderPayload
    | LabelsPayload
    | ArchivePayload
    | QuarantinePayload,
    Field(discriminator="kind"),
]
PAYLOAD_ADAPTER: TypeAdapter[ActionPayload] = TypeAdapter(ActionPayload)


class PolicyFinding(BaseModel):
    code: str
    severity: Literal["hard", "soft"]
    message: str


def stable_key(*parts: object) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ProposedAction(BaseModel):
    id: str
    thread_id: str
    payload: ActionPayload
    rationale: str = ""
    proposed_by: str = ""
    status: ActionStatus = ActionStatus.PROPOSED
    findings: list[PolicyFinding] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    idempotency_key: str = ""
    human_comment: str | None = None
    original_body: str | None = None  # draft before the human edit (feedback learning)
    error: str | None = None
    result_ref: str | None = None

    @property
    def type(self) -> ActionType:
        return ActionType(self.payload.kind)

    @property
    def risk_tier(self) -> RiskTier:
        return RISK_TIERS[self.type]

    @property
    def has_hard_violation(self) -> bool:
        return any(f.severity == "hard" for f in self.findings)

    @classmethod
    def create(
        cls,
        thread_id: str,
        payload: BaseModel,
        rationale: str,
        proposed_by: str,
        depends_on: list[str] | None = None,
    ) -> ProposedAction:
        """Deterministic id + idempotency key: re-running a worker on the same
        input yields the same action, so retries and resumes cannot duplicate
        side effects."""
        kind = payload.kind  # type: ignore[attr-defined]
        identity = payload.model_dump(
            mode="json", exclude={"body", "used_facts", "ungrounded_claims", "attachments"}
        )
        key = stable_key(thread_id, kind, identity)
        return cls(
            id=f"{kind}:{key[:10]}",
            thread_id=thread_id,
            payload=payload,  # type: ignore[arg-type]
            rationale=rationale,
            proposed_by=proposed_by,
            depends_on=depends_on or [],
            idempotency_key=key,
        )
