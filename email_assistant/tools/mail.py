"""Mail provider boundary.

`MailProvider` is the seam where Gmail (users.history / drafts) or Microsoft
Graph (delta queries / createReply) adapters would plug in. Two implementations
ship: an in-memory mock seeded from fixtures, and a `.eml` directory loader.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Protocol

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


def load_eml_directory(directory: Path) -> list[EmailThread]:
    """Parse RFC 822 `.eml` files into threads (grouped by subject without Re:/Fwd:)."""
    groups: dict[str, list[EmailMessage]] = {}
    subjects: dict[str, str] = {}
    for i, path in enumerate(sorted(directory.glob("*.eml"))):
        msg = BytesParser(policy=policy.default).parse(path.open("rb"))
        subject = str(msg.get("subject", "(no subject)"))
        norm = subject
        for prefix in ("re:", "fwd:", "fw:", "sv:", "vs:"):
            while norm.lower().startswith(prefix):
                norm = norm[len(prefix) :].strip()
        thread_id = "eml-" + "".join(c if c.isalnum() else "-" for c in norm.lower())[:40]
        body_part = msg.get_body(preferencelist=("plain", "html"))
        body = body_part.get_content() if body_part else ""
        (from_name, from_addr), *_ = getaddresses([str(msg.get("from", ""))]) or [("", "")]
        to = [
            EmailAddress(name=n or None, email=a)
            for n, a in getaddresses(msg.get_all("to", []))
            if a
        ]
        cc = [
            EmailAddress(name=n or None, email=a)
            for n, a in getaddresses(msg.get_all("cc", []))
            if a
        ]
        groups.setdefault(thread_id, []).append(
            EmailMessage(
                id=f"{thread_id}-m{i + 1}",
                thread_id=thread_id,
                sender=EmailAddress(name=from_name or None, email=from_addr),
                to=to,
                cc=cc,
                sent_at=parsedate_to_datetime(str(msg["date"])),
                subject=subject,
                body=body,
            )
        )
        subjects.setdefault(thread_id, norm)
    return [
        EmailThread(id=tid, subject=subjects[tid], messages=sorted(msgs, key=lambda m: m.sent_at))
        for tid, msgs in groups.items()
    ]
