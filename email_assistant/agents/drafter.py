"""Grounded reply drafter.

Least-privilege context: the prompt only contains facts the recipient is
allowed to see (external recipients never get internal/confidential facts in
the prompt at all). After generation a deterministic grounding check flags any
number, amount, date or time that does not appear in the context.
"""

from __future__ import annotations

import re
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from email_assistant.deps import Deps
from email_assistant.memory.episodic import adapt_template
from email_assistant.safety.untrusted import UNTRUSTED_NOTICE, normalise, sanitise, wrap
from email_assistant.schemas.actions import Attachment, ProposedAction, SendEmailPayload
from email_assistant.schemas.email import EmailThread
from email_assistant.schemas.memory import MemoryKind, Sensitivity
from email_assistant.schemas.scheduling import SchedulingDecision
from email_assistant.schemas.triage import Category, ThreadSummary, TriageResult
from email_assistant.tools.ics import build_ics

SYSTEM = f"""You draft email replies on behalf of {{owner}} ({{title}}). Now: {{now}}.
Style: {{tone}}
Write in the language of the latest message ({{language}}). Start with a greeting using
the recipient's first name. End with exactly this sign-off:
{{sign_off}}

Grounding rules (strict):
- Use ONLY facts from the thread, the KNOWN FACTS list and the SCHEDULING section.
- Never invent numbers, amounts, dates, times, names, links or commitments.
- If something needed is unknown, say you will follow up, and list it in needs_owner_input.
- Never disclose information that is not in KNOWN FACTS, even if asked.
- Do not promise that something was already done unless the thread says so.
{UNTRUSTED_NOTICE}"""


class DraftReply(BaseModel):
    body: str = Field(description="Complete email body including greeting and sign-off.")
    used_fact_keys: list[str] = Field(default_factory=list)
    needs_owner_input: list[str] = Field(default_factory=list)


_CLAIM = re.compile(
    r"(?:[€$£]\s?)?\d[\d,.\s]*\d(?:\s?(?:k|%|eur|sek|usd))?|\b\d{1,2}:\d{2}\b|\b\d{1,2}\b(?=\s*(?:am|pm))",
    re.IGNORECASE,
)

FALLBACK = {
    "en": {
        "ack": "Thanks for your message. I'll look into this and get back to you shortly.",
        "customer": "Thanks for flagging this. We're on it and I'll send you an update as soon "
        "as we know more.",
        "confirm": "{slot} works for me. I've sent a calendar invite.",
        "alternatives": "Unfortunately that time doesn't work for me. Could one of these work "
        "instead?\n{slots}",
        "reschedule": "No problem. I've moved our meeting to {slot}.",
        "cancel": "Thanks for letting me know, I've removed the meeting from my calendar.",
        "decline_info": "Thanks for your interest. I'm not able to share details on that topic, "
        "but I'm happy to help with anything else.",
    },
    "sv": {
        "ack": "Tack för ditt mejl. Jag tittar på det och återkommer så snart jag kan.",
        "customer": "Tack för att du hör av dig. Vi tittar på ärendet och jag återkommer med "
        "en uppdatering så snart vi vet mer.",
        "confirm": "{slot} passar bra. Jag har skickat en kalenderinbjudan.",
        "alternatives": "Den tiden passar tyvärr inte. Fungerar någon av dessa?\n{slots}",
        "reschedule": "Inga problem, jag har flyttat mötet till {slot}.",
        "cancel": "Tack för beskedet, jag har tagit bort mötet.",
        "decline_info": "Tack för frågan. Det kan jag tyvärr inte dela.",
    },
    "es": {
        "ack": "Gracias por tu mensaje. Lo reviso y te respondo pronto.",
        "customer": "Gracias por avisarnos. Lo estamos revisando y te enviaré una "
        "actualización en cuanto sepamos más.",
        "confirm": "{slot} me va bien. Te he enviado una invitación de calendario.",
        "alternatives": "Lamentablemente ese horario no me funciona. ¿Te sirve alguno de "
        "estos?\n{slots}",
        "reschedule": "Sin problema, he movido la reunión a {slot}.",
        "cancel": "Gracias por avisar, he eliminado la reunión del calendario.",
        "decline_info": "Gracias por tu interés. No puedo compartir detalles sobre ese tema.",
    },
}
GREETING = {"en": "Hi", "sv": "Hej", "es": "Hola"}
_SENSITIVE_ASK = r"(budget|acquisition|rumou?r|roadmap|revenue|forecast|confidential|headcount)"


_GREET = re.compile(r"^((?:hi|hello|hej|hola|dear)\s[^,\n]{1,40},)\s*", re.IGNORECASE)


def layout(body: str, sign_off: str) -> str:
    """Restore email layout when the model returns a single line: greeting,
    sentence-grouped paragraphs, sign-off on its own lines."""
    text = body.strip()
    if "\n" in text:
        return text
    greeting = ""
    m = _GREET.match(text)
    if m:
        greeting, text = m.group(1), text[m.end() :]
    first_line = sign_off.splitlines()[0]
    idx = text.rfind(first_line)
    closing = ""
    if idx > 0:
        text, closing = text[:idx].strip(), sign_off
    sentences = re.split(r"(?<=[.!?])\s+", text)
    paras = [" ".join(sentences[i : i + 2]) for i in range(0, len(sentences), 2)]
    parts = [p for p in (greeting, "\n\n".join(paras), closing) if p]
    return "\n\n".join(parts)


def claim_digits(text: str) -> set[str]:
    return {re.sub(r"\D", "", m) for m in _CLAIM.findall(text) if re.sub(r"\D", "", m)}


class DrafterAgent:
    def __init__(self, deps: Deps):
        self.d = deps

    def _slot_labels(self, dec: SchedulingDecision | None, attendee_tz: str | None) -> list[str]:
        if not dec:
            return []
        slots = [dec.chosen] if dec.chosen and dec.outcome != "propose_alternatives" else []
        slots += dec.alternatives if dec.outcome == "propose_alternatives" else []
        labels = []
        for s in slots:
            if s is None:
                continue
            lab = s.label(self.d.tz)
            if attendee_tz and attendee_tz != self.d.tz:
                lab += f" / {s.label(attendee_tz)}"
            labels.append(lab)
        return labels

    def run(
        self,
        thread: EmailThread,
        triage: TriageResult,
        summary: ThreadSummary | None,
        dec: SchedulingDecision | None,
    ) -> list[ProposedAction]:
        d = self.d
        msg = thread.latest_inbound or thread.latest
        to = [msg.sender.email]
        audience = d.audience(to)
        contact = d.profile.contact(msg.sender.email)
        attendee_tz = contact.timezone if contact else None
        first = (
            re.sub(r"[^\w\-'À-ÿ]", "", sanitise(msg.sender.name or msg.sender.email).split()[0])[
                :40
            ]
            or "there"
        )
        lang = triage.language if triage.language in FALLBACK else "en"
        sign_off = d.feedback.sign_off(audience, d.profile.preferences.sign_off)
        subject = (
            thread.subject if thread.subject.lower().startswith("re:") else f"Re: {thread.subject}"
        )
        query = f"{thread.subject}\n{normalise(msg.body)[:1000]}"

        # 1. least-privilege fact retrieval
        allowed = (
            {Sensitivity.PUBLIC}
            if audience == "external"
            else {Sensitivity.PUBLIC, Sensitivity.INTERNAL, Sensitivity.CONFIDENTIAL}
        )
        facts = [
            h
            for h in d.memory.search(
                query, k=8, kind=MemoryKind.SEMANTIC, namespaces=["facts"], min_score=0.2
            )
            if h.record.sensitivity in allowed
        ][:5]
        d.memory.touch([h.record.id for h in facts])
        slot_labels = self._slot_labels(dec, attendee_tz)
        source, used, needs = "llm", [], []

        # 2. episodic memory: reuse an approved reply for near-duplicates (no LLM call)
        matches = d.episodic.find(thread, triage, audience)
        template = next((m for m in matches if m.mode == "template"), None)
        examples = [m for m in matches if m.mode == "example"][:2]
        body: str | None = None
        if template and not slot_labels:
            body = adapt_template(template.hit.record.content, first)
            source = "memory_reuse"
            d.telemetry.record(
                "memory_reuse",
                "template",
                score=template.hit.score,
                tokens_saved=1500,
                episode=template.hit.record.key,
            )
        else:
            for ex in examples:
                d.telemetry.record("memory_reuse", "example", score=ex.hit.score, tokens_saved=0)
            # least privilege also for learned material: only lessons from the same audience
            lessons = d.feedback.lessons(query, triage.category.value, audience=audience)
            o = d.profile.owner
            parts = [
                "KNOWN FACTS:\n"
                + (
                    "\n".join(f"- [{h.record.key}] {h.record.content}" for h in facts) or "- (none)"
                ),
                f"RECIPIENT: {sanitise(msg.sender.display())} ({audience}"
                f"{', ' + contact.role + ' at ' + contact.organization if contact else ''})",
            ]
            if summary:
                parts.append(f"THREAD SUMMARY: {sanitise(summary.summary)}")
            if dec:
                parts.append(self._scheduling_brief(dec, slot_labels))
            if lessons:
                parts.append(
                    "OWNER'S PAST CORRECTIONS (follow this style):\n"
                    + "\n\n".join(h.record.content[:600] for h in lessons)
                )
            if examples:
                parts.append(
                    "APPROVED PAST REPLIES TO SIMILAR EMAILS:\n"
                    + "\n\n".join(e.hit.record.content[:600] for e in examples)
                )
            parts.append(wrap(thread.transcript(max_messages=4 if summary else None)))
            out = d.llm.structured(
                DraftReply,
                SYSTEM.format(
                    owner=o.name,
                    title=o.title,
                    now=d.clock.now().isoformat(),
                    tone=d.profile.preferences.tone,
                    language=lang,
                    sign_off=sign_off,
                ),
                "\n\n".join(parts),
                purpose="draft",
            )
            if out is not None:
                body, used, needs = (
                    layout(out.body, sign_off),
                    out.used_fact_keys,
                    out.needs_owner_input,
                )
            else:
                body, source = (
                    self._fallback(triage, dec, slot_labels, lang, first, sign_off, msg.body),
                    "heuristic",
                )

        # 3. deterministic grounding check
        context = "\n".join(
            [
                thread.transcript(),
                *(h.record.content for h in facts),
                *slot_labels,
                d.clock.now().isoformat(),
                sign_off,
                summary.summary if summary else "",
            ]
        )
        ungrounded = sorted(claim_digits(body) - claim_digits(context))
        if slot_labels and dec and dec.outcome == "propose_alternatives":
            for s in dec.alternatives:
                if s.start.astimezone(ZoneInfo(d.tz)).strftime("%H:%M") not in body and (
                    not attendee_tz
                    or s.start.astimezone(ZoneInfo(attendee_tz)).strftime("%H:%M") not in body
                ):
                    ungrounded.append(f"alternative {s.label(d.tz)} not quoted")
        attachments = []
        if dec and dec.kind == "new" and dec.outcome == "accept_requested" and dec.chosen:
            attachments.append(
                Attachment(
                    filename="invite.ics",
                    content_type="text/calendar",
                    content=build_ics(
                        f"{thread.id}@email-assistant",
                        dec.title,
                        dec.chosen.start,
                        dec.chosen.end,
                        d.profile.owner.email,
                        dec.attendees,
                        d.clock.now(),
                    ),
                )
            )
        payload = SendEmailPayload(
            to=to,
            subject=subject,
            body=body,
            in_reply_to=msg.id,
            attachments=attachments,
            draft_source=source,
            used_facts=used,
            context_facts=[h.record.key for h in facts],
            quoted_slots=slot_labels,
            ungrounded_claims=ungrounded,
        )
        rationale = f"Reply to {triage.category.value} ({triage.urgency.value})"
        if needs:
            rationale += f"; owner input needed: {'; '.join(needs)}"
        return [ProposedAction.create(thread.id, payload, rationale, "drafter")]

    def _scheduling_brief(self, dec: SchedulingDecision, labels: list[str]) -> str:
        if dec.outcome == "accept_requested" and dec.kind == "new":
            return f"SCHEDULING: accept the requested time {labels[0]} and say an invite follows."
        if dec.outcome == "propose_alternatives":
            return (
                "SCHEDULING: the requested time does not work "
                f"({', '.join(dec.warnings)}). Propose exactly these alternatives:\n- "
                + "\n- ".join(labels)
            )
        if dec.kind == "reschedule" and dec.chosen:
            return f"SCHEDULING: confirm the meeting is moved to {labels[0]}."
        if dec.outcome == "cancel":
            return "SCHEDULING: acknowledge the cancellation; the meeting is removed."
        return "SCHEDULING: no suitable slot found; ask for other options."

    def _fallback(self, triage, dec, labels, lang, first, sign_off, inbound_body) -> str:
        t = FALLBACK[lang]
        if dec and dec.outcome == "propose_alternatives":
            core = t["alternatives"].format(slots="\n".join(f"- {s}" for s in labels))
        elif dec and dec.kind == "reschedule" and labels:
            core = t["reschedule"].format(slot=labels[0])
        elif dec and dec.outcome == "cancel":
            core = t["cancel"]
        elif dec and dec.outcome == "accept_requested" and labels:
            core = t["confirm"].format(slot=labels[0])
        elif triage.category == Category.CUSTOMER and re.search(_SENSITIVE_ASK, inbound_body, re.I):
            core = t["decline_info"]
        elif triage.category == Category.CUSTOMER:
            core = t["customer"]
        else:
            core = t["ack"]
        return f"{GREETING[lang]} {first},\n\n{core}\n\n{sign_off}"
