from __future__ import annotations

import pytest

from email_assistant.config import Settings
from email_assistant.service import AssistantService


def offline_settings(**overrides) -> Settings:
    return Settings(
        llm_provider="none",
        embedding_provider="hashing",
        log_level="WARNING",
        gemini_api_key=None,
        **overrides,
    )


@pytest.fixture
def svc() -> AssistantService:
    return AssistantService.create(offline_settings(), ephemeral=True)


@pytest.fixture
def processed(svc: AssistantService) -> AssistantService:
    svc.orchestrator.process_inbox()
    return svc


def thread_id(svc: AssistantService, prefix: str) -> str:
    return next(t for t in svc.orchestrator.items if t.startswith(prefix))
