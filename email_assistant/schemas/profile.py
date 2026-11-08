"""Owner profile: preferences, contacts and organisational facts (seed for semantic memory)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from email_assistant.schemas.memory import Sensitivity


class Owner(BaseModel):
    name: str
    email: str
    title: str = ""
    timezone: str = "Europe/Stockholm"


class WorkingHours(BaseModel):
    start: str = "09:00"
    end: str = "18:00"
    days: list[str] = Field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])


class FocusBlock(BaseModel):
    day: str
    start: str
    end: str


class Preferences(BaseModel):
    sign_off: str = "Best,\nAlex"
    tone: str = "Direct, warm, concise."
    working_hours: WorkingHours = Field(default_factory=WorkingHours)
    friday_meeting_cutoff: str | None = "16:00"
    meeting_buffer_minutes: int = 15
    default_meeting_minutes: int = 30
    focus_blocks: list[FocusBlock] = Field(default_factory=list)
    follow_up_business_days: int = 3
    reminder_lead_hours: int = 24


class Contact(BaseModel):
    email: str
    name: str
    organization: str = ""
    role: str = ""
    relationship: str = "external"
    vip: bool = False
    timezone: str | None = None
    language: str = "en"


class OrgFact(BaseModel):
    key: str
    content: str
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    tags: list[str] = Field(default_factory=list)
    # Data-classification terms curated by the owner; auto-derived if absent.
    protected_terms: list[str] | None = None


class UserProfile(BaseModel):
    owner: Owner
    org_domains: list[str]
    preferences: Preferences = Field(default_factory=Preferences)
    contacts: list[Contact] = Field(default_factory=list)
    facts: list[OrgFact] = Field(default_factory=list)

    def contact(self, email: str) -> Contact | None:
        email = email.lower()
        return next((c for c in self.contacts if c.email.lower() == email), None)

    def is_internal(self, email: str) -> bool:
        return email.lower().rsplit("@", 1)[-1] in {d.lower() for d in self.org_domains}
