"""Composition root: settings -> clock -> tools -> memory -> LLM -> graph -> orchestrator.

The API, CLI, notebook, eval runner and tests all build the system through
`AssistantService.create`, so there is exactly one wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from email_assistant.clock import Clock, make_clock
from email_assistant.config import Settings, get_settings
from email_assistant.deps import Deps
from email_assistant.graph.orchestrator import InboxOrchestrator
from email_assistant.graph.thread_graph import ThreadGraph
from email_assistant.llm.cache import JsonlCache
from email_assistant.llm.client import LLM, LLMClient, NullLLM
from email_assistant.llm.factory import LLMUnavailable, get_chat_model, model_name
from email_assistant.memory.checkpointer import make_checkpointer
from email_assistant.memory.embeddings import (
    CachedEmbedder,
    Embedder,
    GeminiEmbedder,
    HashingEmbedder,
)
from email_assistant.memory.episodic import EpisodicMemory
from email_assistant.memory.feedback import FeedbackMemory
from email_assistant.memory.profile import load_profile, seed_memory
from email_assistant.memory.store import MemoryStore
from email_assistant.observability.logging import log_event, setup_logging
from email_assistant.observability.metrics import Telemetry
from email_assistant.observability.tracing import setup_tracing
from email_assistant.safety.injection import ThreatDetector
from email_assistant.safety.policy import PolicyEngine
from email_assistant.tools.calendar import MockCalendar
from email_assistant.tools.mail import MailProvider, MockMailProvider
from email_assistant.tools.reminders import MockReminders


def build_embedder(settings: Settings, telemetry: Telemetry) -> Embedder:
    if settings.embedding_provider == "gemini":
        inner: Embedder = GeminiEmbedder(
            settings.gemini_embedding_model, settings.embedding_dim, settings.gemini_api_key
        )
    else:
        inner = HashingEmbedder()
    cache = JsonlCache(
        settings.embedding_cache_dir / f"{inner.name}.jsonl", settings.embedding_cache_mode
    )
    return CachedEmbedder(inner, cache, telemetry)


def build_llm(settings: Settings, telemetry: Telemetry) -> LLM:
    if settings.llm_provider == "none":
        return NullLLM(telemetry)
    name = model_name(settings)
    cache = JsonlCache(settings.llm_cache_dir / f"{name}.jsonl", settings.llm_cache_mode)
    try:
        model = get_chat_model(settings)
    except LLMUnavailable as exc:
        if settings.llm_cache_mode != "replay":
            log_event("llm.unavailable", reason=str(exc))
            return NullLLM(telemetry)
        model = None  # replay-only: cached responses, misses fall back
    return LLMClient(
        model, name, telemetry, cache, settings.llm_max_retries, settings.llm_requests_per_minute
    )


@dataclass
class AssistantService:
    settings: Settings
    deps: Deps
    graph: object
    orchestrator: InboxOrchestrator
    tracing: str

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        *,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        clock: Clock | None = None,
        mail: MailProvider | None = None,
        ephemeral: bool = False,
    ) -> AssistantService:
        s = settings or get_settings()
        setup_logging(s.log_level, s.log_json)
        tracing = setup_tracing(s)
        db = (lambda _: ":memory:") if ephemeral else s.db_path
        telemetry = Telemetry(db(s.telemetry_db))
        clock = clock or make_clock(s.fixed_now, "Europe/Stockholm")
        profile = load_profile(Path(s.fixtures_dir) / "profile.json")
        embedder = embedder or build_embedder(s, telemetry)
        store = MemoryStore(db(s.memory_db), embedder, clock)
        seed_memory(store, profile)
        calendar = MockCalendar.from_fixture(
            Path(s.fixtures_dir) / "calendar.json", profile.owner.timezone, profile.preferences
        )
        deps = Deps(
            settings=s,
            clock=clock,
            profile=profile,
            telemetry=telemetry,
            llm=llm or build_llm(s, telemetry),
            memory=store,
            episodic=EpisodicMemory(
                store, s.episodic_reuse_threshold, s.episodic_example_threshold
            ),
            feedback=FeedbackMemory(store),
            mail=mail or MockMailProvider.from_fixture(Path(s.fixtures_dir) / "mailbox.json"),
            calendar=calendar,
            reminders=MockReminders(),
            detector=ThreatDetector(profile.org_domains, s.injection_quarantine_threshold),
            policy=PolicyEngine(profile, store, calendar),
        )
        graph = ThreadGraph(deps).build(make_checkpointer(db(s.checkpoint_db)))
        log_event(
            "service.ready", llm=type(deps.llm).__name__, embedder=embedder.name, tracing=tracing
        )
        return cls(
            settings=s,
            deps=deps,
            graph=graph,
            orchestrator=InboxOrchestrator(deps, graph),
            tracing=tracing,
        )
