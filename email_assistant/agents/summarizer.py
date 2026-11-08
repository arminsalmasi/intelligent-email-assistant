"""Thread summarizer worker. Also the context-window manager: long threads are
represented by summary + last messages instead of the full transcript."""

from __future__ import annotations

import re

from email_assistant.deps import Deps
from email_assistant.safety.untrusted import UNTRUSTED_NOTICE, normalise, wrap
from email_assistant.schemas.email import EmailThread
from email_assistant.schemas.triage import ActionItem, ThreadSummary

SYSTEM = f"""You summarise email threads for {{owner}} <{{email}}>. Now: {{now}}.
Be factual: only state what the messages say. List action items with owner and due
date (ISO-8601 with offset) when stated, and questions still open for the owner.
Write in English regardless of the thread language.
{UNTRUSTED_NOTICE}"""

LONG_THREAD_MESSAGES = 3
LONG_THREAD_CHARS = 1500


def needs_summary(thread: EmailThread) -> bool:
    return (
        len(thread.messages) >= LONG_THREAD_MESSAGES
        or sum(len(m.body) for m in thread.messages) > LONG_THREAD_CHARS
    )


def _first_sentence(text: str) -> str:
    body = re.sub(r"^(hi|hello|hej|hola|dear)[^\n]*\n", "", text.strip(), flags=re.I)
    lines = [ln for ln in body.splitlines() if ln.strip() and not ln.lstrip().startswith(">")]
    m = re.search(r"(.+?[.!?])(\s|$)", " ".join(lines))
    return (m.group(1) if m else " ".join(lines))[:220]


def extractive_summary(thread: EmailThread, owner_email: str, owner_first: str) -> ThreadSummary:
    points = [
        f"{m.sender.name or m.sender.email}: {_first_sentence(normalise(m.body))}"
        for m in thread.messages[-5:]
    ]
    items = []
    for m in thread.messages:
        for line in normalise(m.body).splitlines():
            if (
                re.search(
                    rf"\b({re.escape(owner_first)}|you)\b.*\b(to|will|please|by)\b", line, re.I
                )
                and len(line) < 200
                and not line.lstrip().startswith(">")
            ):
                items.append(ActionItem(description=line.strip(" -*"), owner=owner_email))
    questions = [s.strip() for s in re.findall(r"[^.!?\n]*\?", normalise(thread.latest.body))][:3]
    return ThreadSummary(
        summary=f"{len(thread.messages)}-message thread '{thread.subject}'. " + points[-1],
        key_points=points,
        action_items=items[:5],
        open_questions=questions if thread.latest.direction == "inbound" else [],
        source="heuristic",
    )


class SummarizerAgent:
    def __init__(self, deps: Deps):
        self.d = deps

    def run(self, thread: EmailThread) -> ThreadSummary:
        o = self.d.profile.owner
        system = SYSTEM.format(owner=o.name, email=o.email, now=self.d.clock.now().isoformat())
        result = self.d.llm.structured(
            ThreadSummary, system, wrap(thread.transcript()), purpose="summarize"
        )
        if result is None:
            return extractive_summary(thread, o.email, o.name.split()[0])
        result.source = "llm"
        return result
