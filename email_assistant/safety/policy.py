"""Deterministic policy engine for proposed actions.

Runs before review AND again after human edits. Hard findings block an action
(cannot be overridden); soft findings are warnings a human may override with a
justification that is written to the audit trail.
"""

from __future__ import annotations

from datetime import datetime

from email_assistant.memory.store import MemoryStore
from email_assistant.schemas.actions import (
    RISK_TIERS,
    ActionType,
    CalendarConfirmPayload,
    CalendarHoldPayload,
    CalendarUpdatePayload,
    PolicyFinding,
    ProposedAction,
    ReminderPayload,
    SendEmailPayload,
)
from email_assistant.schemas.email import EmailThread
from email_assistant.schemas.memory import MemoryKind, Sensitivity
from email_assistant.schemas.profile import UserProfile
from email_assistant.schemas.scheduling import TimeSlot
from email_assistant.tools.calendar import MockCalendar


class PolicyEngine:
    def __init__(
        self,
        profile: UserProfile,
        memory: MemoryStore,
        calendar: MockCalendar,
        max_recipients: int = 10,
    ):
        self.profile = profile
        self.memory = memory
        self.calendar = calendar
        self.max_recipients = max_recipients

    # -------------------------------------------------------------- helpers
    def _protected(self, sensitivities: set[Sensitivity]) -> dict[str, str]:
        """protected term -> fact key, for facts at the given sensitivity levels."""
        out: dict[str, str] = {}
        for rec in self.memory.list(kind=MemoryKind.SEMANTIC, namespace="facts"):
            if rec.sensitivity in sensitivities:
                for term in rec.metadata.get("protected_terms", []):
                    out[term.lower()] = rec.key
        return out

    def _external(self, emails: list[str]) -> list[str]:
        return [e for e in emails if not self.profile.is_internal(e)]

    # ------------------------------------------------------------ evaluation
    def evaluate(self, action: ProposedAction, thread: EmailThread) -> list[PolicyFinding]:
        f: list[PolicyFinding] = []
        if action.type not in RISK_TIERS:  # defensive: allow-list
            return [
                PolicyFinding(
                    code="action_not_allowed",
                    severity="hard",
                    message=f"{action.type} is not an allowed action",
                )
            ]
        p = action.payload
        if isinstance(p, SendEmailPayload):
            f += self._email(p, thread)
        elif isinstance(p, CalendarHoldPayload | CalendarConfirmPayload | CalendarUpdatePayload):
            ignore = {getattr(p, "hold_id", None) or getattr(p, "event_id", None)} - {None}
            f += self._slot(TimeSlot(start=p.start, end=p.end), ignore)  # type: ignore[arg-type]
            attendees = getattr(p, "attendees", [])
            unknown = [
                a
                for a in attendees
                if a not in thread.participants and a != self.profile.owner.email
            ]
            if unknown:
                f.append(
                    PolicyFinding(
                        code="unknown_attendee",
                        severity="hard",
                        message=f"Attendees not in thread: {unknown}",
                    )
                )
        elif isinstance(p, ReminderPayload) and not p.title.strip():
            f.append(PolicyFinding(code="empty_title", severity="hard", message="Empty title"))
        return f

    def _email(self, p: SendEmailPayload, thread: EmailThread) -> list[PolicyFinding]:
        f: list[PolicyFinding] = []
        recipients = [r.lower() for r in p.to + p.cc]
        if not p.to:
            f.append(PolicyFinding(code="no_recipient", severity="hard", message="No recipient"))
        if len(recipients) > self.max_recipients:
            f.append(
                PolicyFinding(
                    code="too_many_recipients",
                    severity="hard",
                    message=f"{len(recipients)} recipients",
                )
            )
        if not p.body.strip():
            f.append(PolicyFinding(code="empty_body", severity="hard", message="Empty body"))
        participants = thread.participants
        new = [r for r in recipients if r not in participants]
        if new:
            f.append(
                PolicyFinding(
                    code="recipient_not_in_thread",
                    severity="hard",
                    message=f"Recipients not part of this conversation: {new}. Replies may only go "
                    "to existing participants.",
                )
            )
        body = p.body.lower() + "\n" + p.subject.lower()
        external = self._external(recipients)
        if external:
            leaks = {
                t: k
                for t, k in self._protected(
                    {Sensitivity.CONFIDENTIAL, Sensitivity.INTERNAL}
                ).items()
                if t in body
            }
            if leaks:
                f.append(
                    PolicyFinding(
                        code="confidential_leak",
                        severity="hard",
                        message=f"Draft to external recipient(s) {external} contains protected "
                        f"information from {sorted(set(leaks.values()))}: {sorted(leaks)}",
                    )
                )
        else:
            leaks = {
                t: k for t, k in self._protected({Sensitivity.CONFIDENTIAL}).items() if t in body
            }
            if leaks:
                f.append(
                    PolicyFinding(
                        code="confidential_internal",
                        severity="soft",
                        message=f"Contains confidential information {sorted(set(leaks.values()))}; "
                        "internal recipients only - confirm this is intended.",
                    )
                )
        if p.ungrounded_claims:
            f.append(
                PolicyFinding(
                    code="ungrounded_claims",
                    severity="soft",
                    message=f"Facts not found in thread or memory: {p.ungrounded_claims}",
                )
            )
        return f

    def _slot(self, slot: TimeSlot, ignore: set[str]) -> list[PolicyFinding]:
        check = self.calendar.check_slot(slot, ignore=ignore)
        f = []
        for reason in check.reasons:
            sev = "hard" if reason in ("end_before_start", "conflict") else "soft"
            f.append(
                PolicyFinding(
                    code=f"calendar_{reason}",
                    severity=sev,
                    message=f"{slot.label(self.profile.owner.timezone)}: {reason}",
                )
            )
        return f


def outbound_budget_exceeded(sent_this_run: int, limit: int) -> PolicyFinding | None:
    if sent_this_run >= limit:
        return PolicyFinding(
            code="rate_limit",
            severity="hard",
            message=f"Outbound limit of {limit} actions per run reached",
        )
    return None


def is_past(dt: datetime, now: datetime) -> bool:
    return dt < now


__all__ = ["ActionType", "PolicyEngine", "outbound_budget_exceeded"]
