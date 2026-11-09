"""Mail provider boundary.

`MailProvider` is the seam where Gmail (users.history / drafts) or Microsoft
Graph (delta queries / createReply) adapters would plug in. Two implementations
ship: an in-memory mock seeded from fixtures, and a `.eml` directory loader.
"""

from __future__ import annotations

import html
import json
import re
import threading
from datetime import UTC, datetime
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Protocol

from email_assistant.observability.logging import log_event
from email_assistant.schemas.email import EmailAddress, EmailMessage, EmailThread


class MailProvider(Protocol):
    def list_threads(self, include_archived: bool = False) -> list[EmailThread]: ...
    def get_thread(self, thread_id: str) -> EmailThread | None: ...
    def apply_labels(self, thread_id: str, labels: list[str], mark_read: bool = True) -> None: ...
    def archive(self, thread_id: str) -> None: ...
    def send(
        self,
        thread_id: str,
        sender: EmailAddress,
        to: list[str],
        cc: list[str],
        subject: str,
        body: str,
        sent_at: datetime,
        idempotency_key: str,
        attachments: list[dict] | None = None,
    ) -> str: ...
    def sent_messages(self) -> list[dict]: ...


class MockMailProvider:
    """Thread-safe in-memory mailbox. `send` is idempotent per key."""

    def __init__(self, threads: list[EmailThread]):
        self._threads = {t.id: t.model_copy(deep=True) for t in threads}
        self._sent: dict[str, dict] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_fixture(cls, path: Path) -> MockMailProvider:
        raw = json.loads(path.read_text(encoding="utf-8"))
        threads = []
        for t in raw["threads"]:
            msgs = [
                EmailMessage.model_validate({**m, "thread_id": t["id"], "subject": t["subject"]})
                for m in t["messages"]
            ]
            threads.append(EmailThread(id=t["id"], subject=t["subject"], messages=msgs))
        return cls(threads)

    @classmethod
    def from_eml_dir(cls, directory: Path, owner_email: str | None = None) -> MockMailProvider:
        return cls(load_eml_directory(directory, owner_email))

    def list_threads(self, include_archived: bool = False) -> list[EmailThread]:
        with self._lock:
            return [
                t.model_copy(deep=True)
                for t in self._threads.values()
                if include_archived or not t.archived
            ]

    def get_thread(self, thread_id: str) -> EmailThread | None:
        with self._lock:
            t = self._threads.get(thread_id)
            return t.model_copy(deep=True) if t else None

    def apply_labels(self, thread_id: str, labels: list[str], mark_read: bool = True) -> None:
        with self._lock:
            t = self._threads[thread_id]
            t.labels = sorted(set(t.labels) | set(labels))
            if mark_read:
                for m in t.messages:
                    m.is_read = True

    def archive(self, thread_id: str) -> None:
        with self._lock:
            self._threads[thread_id].archived = True

    def send(
        self,
        thread_id: str,
        sender: EmailAddress,
        to: list[str],
        cc: list[str],
        subject: str,
        body: str,
        sent_at: datetime,
        idempotency_key: str,
        attachments: list[dict] | None = None,
    ) -> str:
        with self._lock:
            if idempotency_key in self._sent:  # retry / resume: no duplicate send
                return self._sent[idempotency_key]["message_id"]
            thread = self._threads[thread_id]
            msg_id = f"{thread_id}-sent-{len(thread.messages) + 1}"
            msg = EmailMessage(
                id=msg_id,
                thread_id=thread_id,
                sender=sender,
                to=[EmailAddress(email=e) for e in to],
                cc=[EmailAddress(email=e) for e in cc],
                sent_at=sent_at,
                subject=subject,
                body=body,
                direction="outbound",
                is_read=True,
            )
            thread.messages.append(msg)
            self._sent[idempotency_key] = {
                "message_id": msg_id,
                "thread_id": thread_id,
                "to": to,
                "cc": cc,
                "subject": subject,
                "body": body,
                "sent_at": sent_at.isoformat(),
                "attachments": [a.get("filename") for a in attachments or []],
            }
            return msg_id

    def sent_messages(self) -> list[dict]:
        with self._lock:
            return list(self._sent.values())


_REPLY_PREFIX = re.compile(r"^\s*((re|fwd?|sv|vs|aw|wg|rv)\s*:\s*)+", re.IGNORECASE)


def _html_to_text(html_text: str) -> str:
    """Tags -> text. HTML comments are kept verbatim: hidden instructions are a known
    injection channel and the guard looks for them."""
    comments: list[str] = []

    def _keep(m: re.Match) -> str:
        comments.append(m.group(0))
        return f"\x00{len(comments) - 1}\x00"

    text = re.sub(r"<!--.*?-->", _keep, html_text, flags=re.S)
    text = re.sub(r"(?is)<(script|style).*?</\1>", "", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"\x00(\d+)\x00", lambda m: comments[int(m.group(1))], text)
    return re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _addresses(values: list[str]) -> list[EmailAddress]:
    out = []
    for name, addr in getaddresses(values):
        try:
            out.append(EmailAddress(name=name or None, email=addr))
        except ValueError:
            continue  # undisclosed-recipients:; and similar
    return out


def load_eml_directory(directory: Path, owner_email: str | None = None) -> list[EmailThread]:
    """Parse RFC 5322 `.eml` files into threads.

    Threading follows `References`/`In-Reply-To` to the root `Message-ID` (what Gmail
    and Outlook do), falling back to the subject without Re:/Fwd:/SV: prefixes.
    Messages sent by `owner_email` are outbound. HTML-only bodies are reduced to text
    with HTML comments kept, so hidden instructions still reach the injection guard.
    Files without a sender or date are skipped and logged, not fatal.
    """
    parsed = []
    for path in sorted(directory.glob("*.eml")):
        with path.open("rb") as fh:
            msg = BytesParser(policy=policy.default).parse(fh)
        senders = _addresses([str(msg.get("from", ""))])
        if not senders or not msg.get("date"):
            log_event("mail.eml_skipped", file=path.name, reason="missing From or Date")
            continue
        sent_at = parsedate_to_datetime(str(msg["date"]))
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=UTC)
        part = msg.get_body(preferencelist=("plain", "html"))
        body = part.get_content() if part else ""
        if part is not None and part.get_content_type() == "text/html":
            body = _html_to_text(body)
        refs = str(msg.get("references", "")).split() + str(msg.get("in-reply-to", "")).split()
        parsed.append((path, msg, senders[0], sent_at, body.strip(), refs))

    root_of: dict[str, str] = {}  # Message-ID -> thread key
    subject_key: dict[str, str] = {}
    groups: dict[str, list[EmailMessage]] = {}
    subjects: dict[str, str] = {}
    for path, msg, sender, sent_at, body, refs in sorted(parsed, key=lambda r: r[3]):
        subject = str(msg.get("subject", "")).strip() or "(no subject)"
        norm = _REPLY_PREFIX.sub("", subject).strip() or "(no subject)"
        mid = str(msg.get("message-id", "")).strip() or f"<{path.stem}>"
        key = next((root_of[r] for r in refs if r in root_of), None)
        key = key or subject_key.get(norm.lower())
        if key is None:
            slug = re.sub(r"[^a-z0-9]+", "-", norm.lower()).strip("-")[:40] or "thread"
            key = f"eml-{slug}"
            while key in groups:
                key += "-x"
            groups[key], subjects[key] = [], norm
        root_of[mid] = key
        subject_key.setdefault(norm.lower(), key)
        outbound = owner_email is not None and sender.email == owner_email.lower()
        groups[key].append(
            EmailMessage(
                id=f"{key}-m{len(groups[key]) + 1}",
                thread_id=key,
                sender=sender,
                to=_addresses(msg.get_all("to", [])),
                cc=_addresses(msg.get_all("cc", [])),
                sent_at=sent_at,
                subject=subject,
                body=body,
                direction="outbound" if outbound else "inbound",
                is_read=outbound,
            )
        )
    return [EmailThread(id=k, subject=subjects[k], messages=m) for k, m in groups.items()]
