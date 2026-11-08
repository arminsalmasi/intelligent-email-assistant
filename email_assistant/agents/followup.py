"""Follow-up worker: deadlines, payment due dates and awaiting-reply reminders."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from email_assistant.agents.text import MONTH_RE, WEEKDAY_RE, resolve_day_phrase
from email_assistant.clock import add_business_days
from email_assistant.deps import Deps
from email_assistant.safety.untrusted import UNTRUSTED_NOTICE, normalise, wrap
from email_assistant.schemas.actions import ProposedAction, ReminderPayload
from email_assistant.schemas.email import EmailThread
from email_assistant.schemas.scheduling import DeadlineExtraction
from email_assistant.schemas.triage import Category, TriageResult

SYSTEM = f"""Extract deadlines that the OWNER ({{owner}} <{{email}}>) must meet, from the
thread. Now: {{now}}, owner timezone {{tz}}. Resolve relative dates against now and return
ISO-8601 with offset. "EOD" means 17:00 local. Ignore deadlines that belong to other
people. Return an empty list when there is none.
{UNTRUSTED_NOTICE}"""

_DEADLINE = re.compile(
    rf"\b(by|before|until|no later than|due( date)?:?|senast|innan|antes del?|hasta el)\s+"
    rf"((?:the\s+|el\s+|next\s+|this\s+|próximo\s+)?(?:{WEEKDAY_RE}|today|tomorrow|eod|end of (?:the )?day|"
    rf"\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}\s+(?:de\s+)?(?:{MONTH_RE}))"
    rf"(?:[\s,]+(?:\d{{1,2}}\s+(?:de\s+)?(?:{MONTH_RE})|\d{{4}}-\d{{2}}-\d{{2}}))?"
    rf"(?:\s*(?:at\s+|kl\.?\s*)?\d{{1,2}}[:.]\d{{2}}|\s+eod)?)",
    re.IGNORECASE,
)


class FollowUpAgent:
    def __init__(self, deps: Deps):
        self.d = deps

    def rule_deadlines(self, thread: EmailThread) -> list[tuple[str, datetime]]:
        msg = thread.latest_inbound
        if not msg:
            return []
        out = []
        text = normalise(msg.body)
        for m in _DEADLINE.finditer(text):
            when = resolve_day_phrase(m.group(3), self.d.clock.now(), self.d.tz)
            if when and when > self.d.clock.now():
                sentence = text[max(0, m.start() - 80) : m.end()].split("\n")[-1].strip()
                out.append((sentence, when))
        return out[:2]

    def run(self, thread: EmailThread, triage: TriageResult) -> list[ProposedAction]:
        now, prefs = self.d.clock.now(), self.d.profile.preferences
        actions: list[ProposedAction] = []
        if thread.awaiting_reply:
            sent = thread.latest.sent_at
            due = add_business_days(sent, prefs.follow_up_business_days)
            to = ", ".join(a.email for a in thread.latest.to)
            actions.append(
                ProposedAction.create(
                    thread.id,
                    ReminderPayload(
                        title=f"No reply from {to}: follow up on '{thread.subject}'",
                        due=max(due, now),
                        reminder_type="awaiting_reply",
                        notes=f"Last message sent {sent:%a %d %b}; consider a polite nudge.",
                    ),
                    "Owner's question is unanswered past the follow-up window"
                    if due <= now
                    else "Track pending reply",
                    "followup",
                )
            )
            return actions
        if triage.category in (
            Category.NEWSLETTER,
            Category.MARKETING,
            Category.SPAM,
            Category.SECURITY_THREAT,
        ):
            return actions
        deadlines: list[tuple[str, datetime]] = []
        if triage.deadline_mentioned or triage.category == Category.FINANCE:
            o = self.d.profile.owner
            ext = self.d.llm.structured(
                DeadlineExtraction,
                SYSTEM.format(owner=o.name, email=o.email, now=now.isoformat(), tz=self.d.tz),
                wrap(thread.transcript(max_messages=4)),
                purpose="followup.deadlines",
            )
            if ext is not None:
                for dl in ext.deadlines:
                    try:
                        when = datetime.fromisoformat(dl.due)
                    except ValueError:
                        continue
                    if when.tzinfo and when > now:
                        deadlines.append((dl.description, when))
            else:
                deadlines = self.rule_deadlines(thread)
        for desc, when in deadlines[:2]:
            lead = timedelta(hours=prefs.reminder_lead_hours)
            rtype = "payment" if triage.category == Category.FINANCE else "deadline"
            actions.append(
                ProposedAction.create(
                    thread.id,
                    ReminderPayload(
                        title=f"{'Payment due' if rtype == 'payment' else 'Deadline'}: {thread.subject}"[
                            :120
                        ],
                        due=max(now, when - lead),
                        deadline=when,
                        reminder_type=rtype,
                        notes=desc[:300],
                    ),
                    f"Deadline {when:%a %d %b %H:%M}",
                    "followup",
                )
            )
        return actions


def awaiting_reply_reminder(
    deps: Deps, thread_id: str, subject: str, to: list[str], body: str, sent_at: datetime
) -> ProposedAction | None:
    """After we send a message that asks something, track the expected answer."""
    if "?" not in body:
        return None
    due = add_business_days(sent_at, deps.profile.preferences.follow_up_business_days)
    return ProposedAction.create(
        thread_id,
        ReminderPayload(
            title=f"Awaiting reply from {', '.join(to)}: {subject}"[:120],
            due=due,
            reminder_type="awaiting_reply",
            notes="Auto-completes when they reply.",
        ),
        "Sent message contains a question",
        "followup",
    )
