"""Per-thread LangGraph: guard -> triage -> supervisor <-> workers -> policy ->
auto-execute -> [review (interrupt) -> revalidate]* -> execute -> learn.

The supervisor is a deterministic router over the structured triage result: it
builds a plan of workers and loops supervisor -> worker -> supervisor until the
plan is done. Workers only *propose* actions; side effects happen in the two
execute nodes, and approval-tier actions only after a human decision.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from email_assistant.agents.drafter import DrafterAgent
from email_assistant.agents.followup import FollowUpAgent, awaiting_reply_reminder
from email_assistant.agents.scheduler import SchedulerAgent
from email_assistant.agents.summarizer import SummarizerAgent, needs_summary
from email_assistant.agents.triage import TriageAgent, is_quarantined
from email_assistant.deps import Deps
from email_assistant.observability.tracing import traced_node
from email_assistant.schemas.actions import (
    ActionStatus,
    ActionType,
    ArchivePayload,
    LabelsPayload,
    ProposedAction,
    QuarantinePayload,
    RiskTier,
    SendEmailPayload,
    apply_edits,
)
from email_assistant.schemas.review import ReviewRequest, ReviewResponse
from email_assistant.schemas.scheduling import TimeSlot
from email_assistant.schemas.state import ThreadState
from email_assistant.schemas.triage import BULK_CATEGORIES, Category

MAX_REVIEW_ROUNDS = 3
WORKERS = ("labeler", "summarizer", "scheduler", "followup", "drafter", "archiver", "quarantine")


def _audit(event: str, **fields: Any) -> dict[str, Any]:
    return {"event": event, **fields}


class ThreadGraph:
    def __init__(self, deps: Deps):
        self.d = deps
        self.triage_agent = TriageAgent(deps)
        self.summarizer = SummarizerAgent(deps)
        self.scheduler = SchedulerAgent(deps)
        self.followup = FollowUpAgent(deps)
        self.drafter = DrafterAgent(deps)

    # ================================================================ nodes
    def guard(self, s: ThreadState) -> dict:
        if "guard" in s:
            return {}
        g = self.d.detector.assess(s["thread"])
        self.d.telemetry.record("guard", "quarantined" if g.is_threat else "clean", g.risk_score)
        return {"guard": g}

    def triage(self, s: ThreadState) -> dict:
        if "triage" in s:
            return {}
        t, g = self.triage_agent.run(s["thread"], s["guard"])
        return {"triage": t, "guard": g}

    def supervisor(self, s: ThreadState) -> dict:
        if "plan" in s:
            return {}
        t, thread = s["triage"], s["thread"]
        plan = ["labeler"]
        if is_quarantined(s["guard"], t):
            plan.append("quarantine")
        elif t.category in BULK_CATEGORIES or (
            t.category == Category.INTERNAL
            and not t.requires_reply
            and t.urgency.value == "low"
            and not t.deadline_mentioned
        ):
            plan.append("archiver")  # bulk mail and read-only FYIs
        else:
            if needs_summary(thread):
                plan.append("summarizer")
            if t.meeting_requested or t.category == Category.MEETING:
                plan.append("scheduler")
            plan.append("followup")
            if t.requires_reply and not thread.awaiting_reply:
                plan.append("drafter")
        return {
            "plan": plan,
            "audit": [
                _audit(
                    "plan",
                    plan=plan,
                    category=t.category.value,
                    urgency=t.urgency.value,
                    source=t.source,
                )
            ],
        }

    @staticmethod
    def route_supervisor(s: ThreadState) -> str:
        done = set(s.get("completed", []))
        return next((w for w in s.get("plan", []) if w not in done), "policy")

    def _actions(self, items: list[ProposedAction], worker: str) -> dict:
        return {"actions": {a.id: a for a in items}, "completed": [worker]}

    def labeler(self, s: ThreadState) -> dict:
        t = s["triage"]
        labels = [f"category/{t.category.value}", f"priority/{t.urgency.value}"]
        if t.requires_reply:
            labels.append("needs-reply")
        if s["thread"].awaiting_reply:
            labels.append("awaiting-reply")
        a = ProposedAction.create(
            s["thread_id"], LabelsPayload(labels=labels), "Triage labels", "labeler"
        )
        return self._actions([a], "labeler")

    def summarizer_node(self, s: ThreadState) -> dict:
        return {"summary": self.summarizer.run(s["thread"]), "completed": ["summarizer"]}

    def scheduler_node(self, s: ThreadState) -> dict:
        dec, acts = self.scheduler.run(s["thread"])
        out = self._actions(acts, "scheduler")
        out["scheduling"] = dec
        out["audit"] = [
            _audit(
                "scheduling",
                outcome=dec.outcome,
                kind=dec.kind,
                source=dec.source,
                warnings=dec.warnings,
            )
        ]
        return out

    def followup_node(self, s: ThreadState) -> dict:
        return self._actions(self.followup.run(s["thread"], s["triage"]), "followup")

    def drafter_node(self, s: ThreadState) -> dict:
        acts = self.drafter.run(s["thread"], s["triage"], s.get("summary"), s.get("scheduling"))
        return self._actions(acts, "drafter")

    def archiver(self, s: ThreadState) -> dict:
        a = ProposedAction.create(
            s["thread_id"],
            ArchivePayload(reason=f"Bulk mail ({s['triage'].category.value})"),
            "No action needed",
            "archiver",
        )
        return self._actions([a], "archiver")

    def quarantine(self, s: ThreadState) -> dict:
        g = s["guard"]
        a = ProposedAction.create(
            s["thread_id"],
            QuarantinePayload(
                reason=g.threat_type or "suspicious", risk_score=g.risk_score, signals=g.signals
            ),
            "Held for human security review; no automated replies",
            "quarantine",
        )
        lab = ProposedAction.create(
            s["thread_id"], LabelsPayload(labels=["quarantine"]), "Quarantine label", "quarantine"
        )
        return self._actions([a, lab], "quarantine")

    def policy(self, s: ThreadState) -> dict:
        updated, audit = {}, []
        for a in s.get("actions", {}).values():
            if a.status != ActionStatus.PROPOSED:
                continue
            findings = self.d.policy.evaluate(a, s["thread"])
            for f in findings:
                self.d.telemetry.record(
                    "policy_finding", f.code, severity=f.severity, action_type=a.type.value
                )
            if any(f.severity == "hard" for f in findings):
                status = ActionStatus.BLOCKED
            elif a.risk_tier == RiskTier.APPROVAL:
                status = ActionStatus.PENDING_APPROVAL
            else:
                status = ActionStatus.APPROVED
            updated[a.id] = a.model_copy(update={"findings": findings, "status": status})
            audit.append(
                _audit(
                    "policy", action=a.id, status=status.value, findings=[f.code for f in findings]
                )
            )
        return {"actions": updated, "audit": audit}

    def execute_auto(self, s: ThreadState) -> dict:
        return self._execute(s, tiers={RiskTier.AUTO})

    @staticmethod
    def route_pending(s: ThreadState) -> str:
        pending = any(
            a.status == ActionStatus.PENDING_APPROVAL for a in s.get("actions", {}).values()
        )
        return "review" if pending else "execute"

    def review(self, s: ThreadState) -> dict:
        pending = [a for a in s["actions"].values() if a.status == ActionStatus.PENDING_APPROVAL]
        rnd = s.get("review_round", 0) + 1
        summary = s["summary"].summary if s.get("summary") else s["triage"].reasoning or ""
        request = ReviewRequest(
            thread_id=s["thread_id"],
            subject=s["thread"].subject,
            summary=summary,
            round=rnd,
            actions=pending,
        )
        raw = interrupt(request.model_dump(mode="json"))
        response = ReviewResponse.model_validate(raw)
        decided = {dec.action_id: dec for dec in response.decisions}
        updated, audit = {}, []
        for a in pending:
            dec = decided.get(a.id)
            if dec is None:
                continue  # stays pending -> asked again
            self.d.telemetry.record("human_decision", dec.decision, action_type=a.type.value)
            if dec.decision == "approve" and a.has_hard_violation:
                # Hard findings cannot be approved away - only edited or rejected.
                audit.append(
                    _audit("approve_refused", action=a.id, findings=[f.code for f in a.findings])
                )
                continue
            if dec.decision == "approve":
                updated[a.id] = a.model_copy(
                    update={"status": ActionStatus.APPROVED, "human_comment": dec.comment}
                )
            elif dec.decision == "reject":
                updated[a.id] = a.model_copy(
                    update={"status": ActionStatus.REJECTED, "human_comment": dec.comment}
                )
            else:
                # Edits were validated against the payload schema before resuming
                # (orchestrator.validate_review); this is the same strict check.
                payload = apply_edits(a, dec.edits)
                original = None
                if isinstance(a.payload, SendEmailPayload):
                    original = a.original_body or a.payload.body  # keep the model's draft
                    if isinstance(payload, SendEmailPayload) and payload.body != a.payload.body:
                        payload = payload.model_copy(
                            update={"ungrounded_claims": [], "draft_source": "human_edit"}
                        )
                updated[a.id] = a.model_copy(
                    update={
                        "payload": payload,
                        "status": ActionStatus.EDITED,
                        "human_comment": dec.comment,
                        "original_body": original,
                    }
                )
            audit.append(
                _audit(
                    "human_decision",
                    action=a.id,
                    decision=dec.decision,
                    comment=dec.comment,
                    override=dec.override_soft_findings,
                    justification=dec.justification,
                    reviewer=response.reviewer,
                )
            )
        return {"actions": updated, "review_round": rnd, "audit": audit}

    def revalidate(self, s: ThreadState) -> dict:
        """Single gate before execution: every approval-tier action that is about to
        run (approved or edited) goes through policy again; hard findings send it
        back to review (or block it after MAX_REVIEW_ROUNDS)."""
        updated, audit = {}, []
        for a in s["actions"].values():
            if a.risk_tier != RiskTier.APPROVAL or a.status not in (
                ActionStatus.APPROVED,
                ActionStatus.EDITED,
            ):
                continue
            findings = self.d.policy.evaluate(a, s["thread"])
            if any(f.severity == "hard" for f in findings):
                over = s.get("review_round", 0) >= MAX_REVIEW_ROUNDS
                status = ActionStatus.BLOCKED if over else ActionStatus.PENDING_APPROVAL
                for f in findings:
                    self.d.telemetry.record(
                        "policy_finding",
                        f.code,
                        severity=f.severity,
                        action_type=a.type.value,
                        stage="post_edit",
                    )
                updated[a.id] = a.model_copy(update={"findings": findings, "status": status})
                audit.append(
                    _audit(
                        "revalidate",
                        action=a.id,
                        status=status.value,
                        findings=[f.code for f in findings],
                    )
                )
            else:
                updated[a.id] = a.model_copy(update={"findings": findings})
        return {"actions": updated, "audit": audit}

    def execute(self, s: ThreadState) -> dict:
        return self._execute(s, tiers={RiskTier.APPROVAL}, compensate=True)

    def learn(self, s: ThreadState) -> dict:
        d, thread, triage = self.d, s["thread"], s["triage"]
        new_actions, audit = {}, []
        for a in s.get("actions", {}).values():
            if a.type != ActionType.SEND_EMAIL or a.status != ActionStatus.EXECUTED:
                continue
            p: SendEmailPayload = a.payload  # type: ignore[assignment]
            audience = d.audience(p.to + p.cc)
            edited = a.original_body is not None and a.original_body != p.body
            d.episodic.record(thread, triage, p.body, audience, edited)
            if edited:
                learned = d.feedback.record_edit(
                    thread.id,
                    a.id,
                    triage.category.value,
                    audience,
                    a.original_body or "",
                    p.body,
                    a.human_comment,
                )
                audit.append(_audit("learned", action=a.id, items=learned))
            rem = awaiting_reply_reminder(d, thread.id, p.subject, p.to, p.body, d.clock.now())
            if rem:
                rp = rem.payload
                ref = d.reminders.create(
                    rem.idempotency_key,
                    thread.id,
                    rp.title,
                    rp.due,  # type: ignore[union-attr]
                    rp.reminder_type,
                    rp.notes,
                    d.clock.now(),
                )  # type: ignore[union-attr]
                new_actions[rem.id] = rem.model_copy(
                    update={"status": ActionStatus.EXECUTED, "result_ref": ref}
                )
        return {"actions": new_actions, "audit": audit}

    # ============================================================ execution
    def _execute(self, s: ThreadState, tiers: set[RiskTier], compensate: bool = False) -> dict:
        d = self.d
        actions = dict(s.get("actions", {}))
        updated, audit = {}, []
        sent_this_thread = 0
        if compensate:
            updated.update(self._pair_meeting_reply(actions))
            actions.update(updated)
        for a in sorted(actions.values(), key=lambda x: (len(x.depends_on), x.id)):
            if a.risk_tier not in tiers or a.status not in (
                ActionStatus.APPROVED,
                ActionStatus.EDITED,
            ):
                continue
            deps_ok = all(
                (updated.get(dep) or actions[dep]).status == ActionStatus.EXECUTED
                for dep in a.depends_on
                if dep in actions
            )
            if not deps_ok:
                updated[a.id] = a.model_copy(
                    update={"status": ActionStatus.SKIPPED, "error": "dependency not executed"}
                )
                continue
            if a.type == ActionType.SEND_EMAIL:
                sent_this_thread += 1
                if sent_this_thread > d.settings.max_outbound_per_thread:
                    updated[a.id] = a.model_copy(
                        update={"status": ActionStatus.BLOCKED, "error": "rate limit"}
                    )
                    continue
            try:
                ref = self._apply(a, s)
                updated[a.id] = a.model_copy(
                    update={"status": ActionStatus.EXECUTED, "result_ref": ref}
                )
                d.telemetry.record("action", f"{a.type.value}:executed")
                audit.append(_audit("executed", action=a.id, ref=ref))
            except Exception as exc:  # isolate failures per action
                updated[a.id] = a.model_copy(
                    update={"status": ActionStatus.FAILED, "error": str(exc)[:300]}
                )
                d.telemetry.record("action", f"{a.type.value}:failed", error=str(exc)[:200])
                d.telemetry.record("error", "execute", error=type(exc).__name__)
        if compensate:
            audit += self._compensate({**actions, **updated}, s)
        for a in actions.values():
            if a.status in (ActionStatus.REJECTED, ActionStatus.BLOCKED) and a.id not in updated:
                d.telemetry.record("action", f"{a.type.value}:{a.status.value}")
        return {"actions": updated, "audit": audit}

    @staticmethod
    def _pair_meeting_reply(actions: dict[str, ProposedAction]) -> dict[str, ProposedAction]:
        """A meeting confirmation and the reply announcing it are all-or-nothing:
        if the human rejects (or policy blocks) one, the other is skipped too."""
        dead = {ActionStatus.REJECTED, ActionStatus.BLOCKED}
        confirm = next((a for a in actions.values() if a.type == ActionType.CALENDAR_CONFIRM), None)
        email = next((a for a in actions.values() if a.type == ActionType.SEND_EMAIL), None)
        if not confirm or not email:
            return {}
        out = {}
        for x, y in ((confirm, email), (email, confirm)):
            if x.status in dead and y.status in (ActionStatus.APPROVED, ActionStatus.EDITED):
                out[y.id] = y.model_copy(
                    update={
                        "status": ActionStatus.SKIPPED,
                        "error": f"paired {x.type.value} was {x.status.value}",
                    }
                )
        return out

    def _compensate(self, actions: dict[str, ProposedAction], s: ThreadState) -> list[dict]:
        """Release tentative holds whose follow-up was rejected or blocked."""
        audit = []
        dead = {
            ActionStatus.REJECTED,
            ActionStatus.BLOCKED,
            ActionStatus.SKIPPED,
            ActionStatus.FAILED,
        }
        email_dead = any(
            a.type == ActionType.SEND_EMAIL and a.status in dead for a in actions.values()
        )
        for a in actions.values():
            if a.type != ActionType.CALENDAR_HOLD or a.status != ActionStatus.EXECUTED:
                continue
            confirm = next((c for c in actions.values() if a.id in c.depends_on), None)
            if (confirm and confirm.status in dead) or (confirm is None and email_dead):
                self.d.calendar.release(a.payload.hold_id)  # type: ignore[union-attr]
                audit.append(_audit("compensated", action=a.id, what="hold released"))
        return audit

    def _apply(self, a: ProposedAction, s: ThreadState) -> str:
        d, p, now = self.d, a.payload, self.d.clock.now()
        match a.type:
            case ActionType.SEND_EMAIL:
                owner = d.profile.owner
                from email_assistant.schemas.email import EmailAddress

                return d.mail.send(
                    a.thread_id,
                    EmailAddress(name=owner.name, email=owner.email),
                    p.to,
                    p.cc,  # type: ignore[union-attr]
                    p.subject,
                    p.body,
                    now,
                    a.idempotency_key,  # type: ignore[union-attr]
                    [att.model_dump() for att in p.attachments],
                )  # type: ignore[union-attr]
            case ActionType.CALENDAR_HOLD:
                return d.calendar.place_hold(
                    p.hold_id,
                    p.title,
                    TimeSlot(start=p.start, end=p.end),  # type: ignore[union-attr]
                    p.attendees,
                    a.thread_id,
                    a.idempotency_key,
                )  # type: ignore[union-attr]
            case ActionType.CALENDAR_CONFIRM:
                return d.calendar.confirm(
                    p.hold_id,
                    p.title,
                    TimeSlot(start=p.start, end=p.end),  # type: ignore[union-attr]
                    p.attendees,
                    a.idempotency_key,
                )  # type: ignore[union-attr]
            case ActionType.CALENDAR_UPDATE:
                return d.calendar.update(
                    p.event_id,
                    TimeSlot(start=p.start, end=p.end),  # type: ignore[union-attr]
                    a.idempotency_key,
                )
            case ActionType.CALENDAR_CANCEL:
                return d.calendar.cancel(p.event_id, a.idempotency_key)  # type: ignore[union-attr]
            case ActionType.CREATE_REMINDER:
                return d.reminders.create(
                    a.idempotency_key,
                    a.thread_id,
                    p.title,
                    p.due,  # type: ignore[union-attr]
                    p.reminder_type,
                    p.notes,
                    now,
                )  # type: ignore[union-attr]
            case ActionType.APPLY_LABELS:
                d.mail.apply_labels(a.thread_id, p.labels, p.mark_read)  # type: ignore[union-attr]
                return "labels"
            case ActionType.ARCHIVE:
                d.mail.archive(a.thread_id)
                return "archived"
            case ActionType.QUARANTINE:
                d.mail.apply_labels(a.thread_id, ["quarantine"], mark_read=False)
                return "quarantined"
        raise ValueError(f"unsupported action {a.type}")

    # ================================================================ build
    def build(self, checkpointer: BaseCheckpointSaver | None = None):
        t = self.d.telemetry
        g = StateGraph(ThreadState)
        nodes = {
            "guard": self.guard,
            "triage": self.triage,
            "supervisor": self.supervisor,
            "labeler": self.labeler,
            "summarizer": self.summarizer_node,
            "scheduler": self.scheduler_node,
            "followup": self.followup_node,
            "drafter": self.drafter_node,
            "archiver": self.archiver,
            "quarantine": self.quarantine,
            "policy": self.policy,
            "execute_auto": self.execute_auto,
            "review": self.review,
            "revalidate": self.revalidate,
            "execute": self.execute,
            "learn": self.learn,
        }
        critical = {"policy", "execute_auto", "review", "revalidate", "execute"}
        for name, fn in nodes.items():
            # Workers degrade gracefully; safety-critical nodes fail closed (the
            # exception aborts the run, nothing unvalidated can reach execute).
            g.add_node(name, traced_node(name, t, reraise=name in critical)(fn))
        g.add_edge(START, "guard")
        g.add_edge("guard", "triage")
        g.add_edge("triage", "supervisor")
        g.add_conditional_edges(
            "supervisor", self.route_supervisor, {w: w for w in (*WORKERS, "policy")}
        )
        for w in WORKERS:
            g.add_edge(w, "supervisor")
        g.add_edge("policy", "execute_auto")
        g.add_conditional_edges(
            "execute_auto", self.route_pending, {"review": "review", "execute": "execute"}
        )
        g.add_edge("review", "revalidate")
        g.add_conditional_edges(
            "revalidate", self.route_pending, {"review": "review", "execute": "execute"}
        )
        g.add_edge("execute", "learn")
        g.add_edge("learn", END)
        return g.compile(checkpointer=checkpointer)
