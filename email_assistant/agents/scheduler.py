"""Scheduler worker: the one genuinely agentic (tool-calling) worker.

Flow:
1. Extract a MeetingRequest (LLM structured output; rule-based fallback).
2. LLM tool loop over calendar tools (bounded steps) -> SchedulingDecision.
3. Deterministic verification of the decision (slot free, working hours,
   focus blocks, buffers, attendee timezone). If the LLM decision fails
   verification - or there is no LLM - a deterministic planner decides.
"LLM proposes, code verifies."
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool

from email_assistant.agents.text import parse_time, resolve_day_phrase, timezone_hint
from email_assistant.deps import Deps
from email_assistant.safety.untrusted import UNTRUSTED_NOTICE, normalise, wrap
from email_assistant.schemas.actions import (
    CalendarCancelPayload,
    CalendarConfirmPayload,
    CalendarHoldPayload,
    CalendarUpdatePayload,
    ProposedAction,
)
from email_assistant.schemas.email import EmailThread
from email_assistant.schemas.scheduling import MeetingRequest, SchedulingDecision, TimeSlot

EXTRACT_SYSTEM = f"""Extract the meeting request from the latest message of the thread.
Now is {{now}} (owner timezone {{tz}}). Resolve relative dates ("this Tuesday",
"tisdag nästa vecka") against now. Give requested_start as ISO-8601 with the UTC
offset of the timezone the sender used; set requested_timezone to that IANA zone.
kind: new | reschedule (move an existing meeting) | cancel.
{UNTRUSTED_NOTICE}"""

PLAN_SYSTEM = """You schedule meetings for {owner} (timezone {tz}). Now: {now}.
Use the tools to check the requested slot and find alternatives. Rules: respect
working hours, focus blocks, buffers and the attendee's timezone. If the
requested slot is acceptable choose it (outcome accept_requested). Otherwise
propose exactly 3 alternative slots (outcome propose_alternatives). For a
reschedule, find a slot matching the preference and set linked_event_id. For a
cancellation, set outcome cancel and linked_event_id. Never invent events."""

WINDOWS = {
    "morning": ("09:00", "12:00"),
    "afternoon": ("13:00", "17:00"),
    "förmiddag": ("09:00", "12:00"),
    "eftermiddag": ("13:00", "17:00"),
    "mañana": ("09:00", "12:00"),
    "tarde": ("13:00", "17:00"),
}
_CANCEL = r"(cancel|call off|avboka|ställa in|cancelar)"
_RESCHEDULE = r"(reschedule|move (it|our|the)|push (it|our)|flytta|instead|reprogramar|cambiar)"


class SchedulerAgent:
    def __init__(self, deps: Deps):
        self.d = deps

    # -------------------------------------------------------------- extract
    def extract(self, thread: EmailThread) -> MeetingRequest:
        now = self.d.clock.now()
        req = self.d.llm.structured(
            MeetingRequest,
            EXTRACT_SYSTEM.format(now=now.isoformat(), tz=self.d.tz),
            wrap(thread.transcript(max_messages=3)),
            purpose="scheduler.extract",
        )
        if req is not None:
            req.source = "llm"
            return req
        return self.rule_extract(thread)

    def rule_extract(self, thread: EmailThread) -> MeetingRequest:
        msg = thread.latest_inbound or thread.latest
        text = normalise(msg.body)
        low = text.lower()
        kind = (
            "cancel"
            if re.search(_CANCEL, low)
            else ("reschedule" if re.search(_RESCHEDULE, low) else "new")
        )
        contact = self.d.profile.contact(msg.sender.email)
        tz = timezone_hint(text) or (
            contact.timezone if contact and contact.timezone else self.d.tz
        )
        duration = 30
        m = re.search(r"(\d{2,3})\s*(min|minutes|minuter|minutos)", low)
        if m:
            duration = int(m.group(1))
        elif re.search(r"(an hour|1 hour|en timme|una hora)", low):
            duration = 60
        elif re.search(r"(halvtimme|half an hour|media hora)", low):
            duration = 30
        start = None
        window = next((w for w in WINDOWS if re.search(rf"\b{w}\b", low)), None)
        if kind == "new" and parse_time(low):
            resolved = resolve_day_phrase(low, self.d.clock.now(), tz)
            start = resolved.isoformat() if resolved else None
        return MeetingRequest(
            kind=kind,
            requested_start=start,
            requested_timezone=tz,
            preferred_window=window,
            duration_minutes=duration,
            topic=thread.subject,
            source="heuristic",
        )

    # ---------------------------------------------------------------- plan
    def _attendee_tz(self, thread: EmailThread, req: MeetingRequest) -> str | None:
        msg = thread.latest_inbound or thread.latest
        c = self.d.profile.contact(msg.sender.email)
        return (c.timezone if c and c.timezone else None) or req.requested_timezone

    def _tools(self, thread: EmailThread, attendee_tz: str | None) -> list[StructuredTool]:
        cal = self.d.calendar

        def check_slot(start_iso: str, duration_minutes: int = 30) -> dict:
            """Check whether a slot (ISO start with offset) is acceptable for the owner and attendee."""
            s = datetime.fromisoformat(start_iso)
            linked = cal.find_by_thread(thread.id)
            r = cal.check_slot(
                TimeSlot(start=s, end=s + timedelta(minutes=duration_minutes)),
                attendee_tz=attendee_tz,
                ignore={linked.id} if linked else None,
            )
            return r.model_dump()

        def find_free_slots(
            after_iso: str,
            duration_minutes: int = 30,
            count: int = 3,
            weekday: str | None = None,
            window_start: str | None = None,
            window_end: str | None = None,
        ) -> list[dict]:
            """Find acceptable free slots after a time. weekday: mon..fri; window: HH:MM."""
            linked = cal.find_by_thread(thread.id)
            slots = cal.find_free_slots(
                datetime.fromisoformat(after_iso),
                duration_minutes,
                count=count,
                attendee_tz=attendee_tz,
                ignore={linked.id} if linked else None,
                window=(window_start, window_end) if window_start and window_end else None,
                only_weekday=weekday,
            )
            return [s.model_dump(mode="json") for s in slots]

        def get_linked_event() -> dict | None:
            """Return the existing calendar event linked to this email thread, if any."""
            ev = cal.find_by_thread(thread.id)
            return ev.model_dump(mode="json") if ev else None

        def get_meeting_preferences() -> dict:
            """Return the owner's meeting preferences."""
            return json.loads(self.d.profile.preferences.model_dump_json())

        return [
            StructuredTool.from_function(f)
            for f in (check_slot, find_free_slots, get_linked_event, get_meeting_preferences)
        ]

    def verify(
        self,
        dec: SchedulingDecision,
        thread: EmailThread,
        req: MeetingRequest,
        attendee_tz: str | None,
    ) -> list[str]:
        """Deterministic acceptance test for a model-proposed decision."""
        try:
            return self._verify(dec, thread, attendee_tz)
        except Exception as exc:  # e.g. naive datetimes from the model
            return [f"invalid_decision:{type(exc).__name__}"]

    def _verify(
        self, dec: SchedulingDecision, thread: EmailThread, attendee_tz: str | None
    ) -> list[str]:
        problems = []
        cal = self.d.calendar
        linked = cal.find_by_thread(thread.id)
        ignore = {linked.id} if linked else None
        # Only the event linked to THIS thread may ever be moved or cancelled.
        touches_event = dec.kind in ("reschedule", "cancel") or dec.outcome == "cancel"
        if touches_event or dec.linked_event_id:
            if not linked or dec.linked_event_id != linked.id:
                problems.append("linked_event_mismatch")
        if dec.outcome == "cancel" and dec.kind != "cancel":
            problems.append("inconsistent_kind")
        if dec.outcome == "accept_requested" or (dec.kind == "reschedule" and dec.chosen):
            if not dec.chosen:
                problems.append("missing_chosen_slot")
            elif (
                dec.chosen.start < self.d.clock.now()
                or not cal.check_slot(dec.chosen, attendee_tz=attendee_tz, ignore=ignore).ok
            ):
                problems.append("chosen_slot_invalid")
        if dec.outcome == "propose_alternatives":
            if not dec.alternatives:
                problems.append("no_alternatives")
            for slot in dec.alternatives:
                if (
                    slot.start < self.d.clock.now()
                    or not cal.check_slot(slot, attendee_tz=attendee_tz, ignore=ignore).ok
                ):
                    problems.append("alternative_invalid")
                    break
        return problems

    def deterministic(
        self, thread: EmailThread, req: MeetingRequest, attendee_tz: str | None
    ) -> SchedulingDecision:
        cal, now = self.d.calendar, self.d.clock.now()
        msg = thread.latest_inbound or thread.latest
        attendees = [msg.sender.email]
        linked = cal.find_by_thread(thread.id)
        dur = req.duration_minutes
        if req.kind == "cancel":
            if not linked:
                return SchedulingDecision(
                    kind="cancel", outcome="no_slot", rationale="no linked event"
                )
            return SchedulingDecision(
                kind="cancel",
                outcome="cancel",
                linked_event_id=linked.id,
                title=linked.title,
                attendees=linked.attendees,
                rationale="sender cancelled the linked meeting",
            )
        if req.kind == "reschedule":
            if not linked:
                return SchedulingDecision(
                    kind="reschedule", outcome="no_slot", rationale="no linked event to move"
                )
            dur = int((linked.end - linked.start).total_seconds() // 60)
            pref = (req.preferred_window or "").lower()
            weekday = next(
                (d for d in ("mon", "tue", "wed", "thu", "fri") if d in (msg.body.lower()[:400])),
                None,
            )
            m = re.search(r"\b(monday|tuesday|wednesday|thursday|friday)\b", msg.body.lower())
            if m:
                weekday = m.group(1)[:3]
            win = next((v for k, v in WINDOWS.items() if k in pref or k in msg.body.lower()), None)
            slots = cal.find_free_slots(
                now,
                dur,
                count=3,
                attendee_tz=attendee_tz,
                ignore={linked.id},
                window=win,
                only_weekday=weekday,
            )
            if not slots:
                slots = cal.find_free_slots(
                    now, dur, count=3, attendee_tz=attendee_tz, ignore={linked.id}
                )
            return SchedulingDecision(
                kind="reschedule",
                outcome="accept_requested" if slots else "no_slot",
                chosen=slots[0] if slots else None,
                alternatives=slots[1:],
                linked_event_id=linked.id,
                title=linked.title,
                attendees=linked.attendees,
                rationale=f"first free slot matching '{pref or weekday or 'any'}'",
            )
        # new meeting
        title = f"{req.topic or thread.subject}".replace("Re: ", "")[:80]
        if req.requested_start:
            start = datetime.fromisoformat(req.requested_start)
            if start.tzinfo is None:
                start = start.replace(tzinfo=ZoneInfo(req.requested_timezone or self.d.tz))
            slot = TimeSlot(start=start, end=start + timedelta(minutes=dur))
            check = cal.check_slot(slot, attendee_tz=attendee_tz)
            if check.ok and start > now:
                return SchedulingDecision(
                    kind="new",
                    outcome="accept_requested",
                    chosen=slot,
                    title=title,
                    attendees=attendees,
                    rationale="requested slot is free and within hours",
                )
            after = max(now, start - timedelta(hours=1))
            reasons = check.reasons or ["in_the_past"]
        else:
            after, reasons = now, ["no_time_given"]
        alts = cal.find_free_slots(after, dur, count=3, attendee_tz=attendee_tz)
        return SchedulingDecision(
            kind="new",
            outcome="propose_alternatives" if alts else "no_slot",
            alternatives=alts,
            title=title,
            attendees=attendees,
            warnings=reasons,
            rationale=f"requested slot not usable: {', '.join(reasons)}",
        )

    def run(self, thread: EmailThread) -> tuple[SchedulingDecision, list[ProposedAction]]:
        req = self.extract(thread)
        attendee_tz = self._attendee_tz(thread, req)
        decision: SchedulingDecision | None = None
        if self.d.llm.available:
            o = self.d.profile.owner
            user = (
                f"Meeting request (extracted): {req.model_dump_json()}\n"
                f"Attendee timezone: {attendee_tz}\nThread id: {thread.id}"
            )
            decision, _trace = self.d.llm.tool_loop(
                PLAN_SYSTEM.format(owner=o.name, tz=self.d.tz, now=self.d.clock.now().isoformat()),
                user,
                self._tools(thread, attendee_tz),
                SchedulingDecision,
                purpose="scheduler.plan",
                max_steps=4,
            )
            if decision is not None:
                decision.source = "llm"
                problems = self.verify(decision, thread, req, attendee_tz)
                if problems:
                    self.d.telemetry.record(
                        "llm_fallback", "scheduler.verify", reason=",".join(problems)
                    )
                    decision = None
        if decision is None:
            decision = self.deterministic(thread, req, attendee_tz)
        return decision, self.actions(thread, decision)

    # ------------------------------------------------------------- actions
    def actions(self, thread: EmailThread, dec: SchedulingDecision) -> list[ProposedAction]:
        out: list[ProposedAction] = []
        att = dec.attendees
        if dec.outcome == "accept_requested" and dec.kind == "new" and dec.chosen:
            hold_id = f"hold-{thread.id[:3]}-{dec.chosen.start:%m%d%H%M}"
            hold = ProposedAction.create(
                thread.id,
                CalendarHoldPayload(
                    hold_id=hold_id,
                    title=dec.title,
                    start=dec.chosen.start,
                    end=dec.chosen.end,
                    attendees=att,
                ),
                "Tentative hold while the reply is reviewed",
                "scheduler",
            )
            confirm = ProposedAction.create(
                thread.id,
                CalendarConfirmPayload(
                    hold_id=hold_id,
                    title=dec.title,
                    start=dec.chosen.start,
                    end=dec.chosen.end,
                    attendees=att,
                ),
                f"Confirm meeting: {dec.rationale}",
                "scheduler",
                depends_on=[hold.id],
            )
            out += [hold, confirm]
        elif dec.outcome == "propose_alternatives":
            for s in dec.alternatives:
                out.append(
                    ProposedAction.create(
                        thread.id,
                        CalendarHoldPayload(
                            hold_id=f"hold-{thread.id[:3]}-{s.start:%m%d%H%M}",
                            title=f"HOLD: {dec.title}",
                            start=s.start,
                            end=s.end,
                            attendees=att,
                        ),
                        "Tentative hold on a proposed alternative",
                        "scheduler",
                    )
                )
        elif dec.kind == "reschedule" and dec.chosen and dec.linked_event_id:
            out.append(
                ProposedAction.create(
                    thread.id,
                    CalendarUpdatePayload(
                        event_id=dec.linked_event_id, start=dec.chosen.start, end=dec.chosen.end
                    ),
                    f"Move meeting: {dec.rationale}",
                    "scheduler",
                )
            )
        elif dec.outcome == "cancel" and dec.linked_event_id:
            out.append(
                ProposedAction.create(
                    thread.id,
                    CalendarCancelPayload(
                        event_id=dec.linked_event_id, reason="Cancelled by counterpart"
                    ),
                    "Counterpart cancelled the meeting",
                    "scheduler",
                )
            )
        return out
