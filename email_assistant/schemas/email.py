"""Email domain models (validated at the mail-provider boundary)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmailAddress(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = None
    email: str

    @field_validator("email")
    @classmethod
    def _normalise(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError(f"invalid email address: {v!r}")
        return v

    @property
    def domain(self) -> str:
        return self.email.rsplit("@", 1)[1]

    def display(self) -> str:
        return f"{self.name} <{self.email}>" if self.name else self.email


class EmailMessage(BaseModel):
    id: str
    thread_id: str
    sender: EmailAddress = Field(alias="from")
    to: list[EmailAddress] = Field(default_factory=list)
    cc: list[EmailAddress] = Field(default_factory=list)
    sent_at: datetime
    subject: str = ""
    body: str
    direction: Literal["inbound", "outbound"] = "inbound"
    is_read: bool = False

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("sent_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("sent_at must be timezone-aware")
        return v


class EmailThread(BaseModel):
    id: str
    subject: str
    messages: list[EmailMessage]
    labels: list[str] = Field(default_factory=list)
    archived: bool = False

    @property
    def latest(self) -> EmailMessage:
        return self.messages[-1]

    @property
    def latest_inbound(self) -> EmailMessage | None:
        return next((m for m in reversed(self.messages) if m.direction == "inbound"), None)

    @property
    def awaiting_reply(self) -> bool:
        """True when our own message is the last one in the thread."""
        return self.latest.direction == "outbound"

    @property
    def participants(self) -> set[str]:
        out: set[str] = set()
        for m in self.messages:
            out.add(m.sender.email)
            out.update(a.email for a in m.to + m.cc)
        return out

    def transcript(self, max_messages: int | None = None) -> str:
        msgs = self.messages[-max_messages:] if max_messages else self.messages
        parts = []
        for m in msgs:
            parts.append(
                f"[{m.sent_at.isoformat()}] {m.direction.upper()} from {m.sender.display()}\n"
                f"To: {', '.join(a.email for a in m.to)}"
                + (f" | Cc: {', '.join(a.email for a in m.cc)}" if m.cc else "")
                + f"\n{m.body.strip()}"
            )
        return "\n\n---\n\n".join(parts)
