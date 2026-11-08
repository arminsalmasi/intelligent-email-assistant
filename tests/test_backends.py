"""Backend selection: explicit choice vs the auto fallback chain."""

from email_assistant.llm import factory
from tests.conftest import offline_settings


class _Model:
    pass


def _patch(monkeypatch, available: set[str]):
    monkeypatch.setattr(
        factory,
        "check_available",
        lambda s, p: None if p in available else f"{p} unavailable",
    )
    monkeypatch.setattr(factory, "get_chat_model", lambda s, p=None: _Model())


def test_auto_orders_gemini_bedrock_local(monkeypatch):
    _patch(monkeypatch, {"gemini", "bedrock", "local"})
    names = [b.provider for b in factory.resolve_backends(offline_settings(llm_provider="auto"))]
    assert names == ["gemini", "bedrock", "local"]


def test_auto_falls_back_to_local_when_hosted_unavailable(monkeypatch):
    _patch(monkeypatch, {"local"})
    backends = factory.resolve_backends(offline_settings(llm_provider="auto"))
    assert [(b.provider, b.name) for b in backends] == [("local", "assistant:32k")]
    assert backends[0].structured_method == "json_schema"  # native Ollama `format`


def test_explicit_choice_uses_only_that_backend(monkeypatch):
    _patch(monkeypatch, {"gemini", "local"})
    backends = factory.resolve_backends(offline_settings(llm_provider="local"))
    assert [b.provider for b in backends] == ["local"]


def test_nothing_available_means_rule_based(monkeypatch):
    _patch(monkeypatch, set())
    assert factory.resolve_backends(offline_settings(llm_provider="auto")) == []


def test_cache_file_names_are_filesystem_safe():
    s = offline_settings()
    assert factory.cache_file(s, "assistant:32k").name == "assistant_32k.jsonl"
