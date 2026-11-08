"""Dependency container handed to workers and graph nodes (built in service.py)."""

from __future__ import annotations

from dataclasses import dataclass

from email_assistant.clock import Clock
from email_assistant.config import Settings
from email_assistant.llm.client import LLM
from email_assistant.memory.episodic import EpisodicMemory
from email_assistant.memory.feedback import FeedbackMemory
from email_assistant.memory.store import MemoryStore
from email_assistant.observability.metrics import Telemetry
from email_assistant.safety.injection import ThreatDetector
from email_assistant.safety.policy import PolicyEngine
from email_assistant.schemas.profile import UserProfile
from email_assistant.tools.calendar import MockCalendar
from email_assistant.tools.mail import MailProvider
from email_assistant.tools.reminders import MockReminders


@dataclass
class Deps:
    settings: Settings
    clock: Clock
    profile: UserProfile
    telemetry: Telemetry
    llm: LLM
    memory: MemoryStore
    episodic: EpisodicMemory
    feedback: FeedbackMemory
    mail: MailProvider
    calendar: MockCalendar
    reminders: MockReminders
    detector: ThreatDetector
    policy: PolicyEngine

    @property
    def tz(self) -> str:
        return self.profile.owner.timezone

    def audience(self, emails: list[str]) -> str:
        return "internal" if all(self.profile.is_internal(e) for e in emails) else "external"
