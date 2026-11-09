"""Chat-model factory: the only place that knows about providers.

`resolve_backends` turns LLM_PROVIDER into an ordered list of usable backends:
explicit choice -> exactly that provider; "auto" -> every available provider in
the order gemini -> bedrock -> local (self-hosted engine). The LLMClient falls
through this list at runtime when a backend hits quota/auth/connection errors.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import httpx
from langchain_core.language_models.chat_models import BaseChatModel

from email_assistant.config import Settings
from email_assistant.llm.cache import JsonlCache
from email_assistant.llm.client import Backend
from email_assistant.observability.logging import log_event

AUTO_ORDER = ("gemini", "bedrock", "local")


class LLMUnavailable(RuntimeError):
    pass


def model_name(settings: Settings, provider: str | None = None) -> str:
    provider = provider or settings.llm_provider
    return {
        "gemini": settings.gemini_model,
        "bedrock": settings.bedrock_model_id,
        "local": settings.assistant_tag,
    }.get(provider, "none")


def cache_file(settings: Settings, name: str) -> Path:
    return settings.llm_cache_dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', name)}.jsonl"


def gemini_key(settings: Settings) -> str | None:
    return settings.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def check_available(settings: Settings, provider: str) -> str | None:
    """None if the provider can be used, else a human-readable reason."""
    if provider == "gemini":
        return None if gemini_key(settings) else "GEMINI_API_KEY not set"
    if provider == "bedrock":
        try:
            import boto3
            import langchain_aws  # noqa: F401
        except ImportError:
            return "langchain-aws not installed (pip install '.[bedrock]')"
        import boto3

        if boto3.Session().get_credentials() is None:
            return "no AWS credentials"
        return None
    if provider == "local":
        base = settings.engine_base_url.rstrip("/")
        try:
            tags = httpx.get(f"{base}/api/tags", timeout=3.0).json().get("models", [])
        except Exception as exc:  # unreachable engine / tunnel down
            return f"engine not reachable at {base} ({type(exc).__name__})"
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        if settings.assistant_tag not in names:
            return f"tag {settings.assistant_tag} not found on engine"
        return None
    return f"unknown provider {provider}"


def get_chat_model(settings: Settings, provider: str | None = None) -> BaseChatModel:
    provider = provider or settings.llm_provider
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = gemini_key(settings)
        if not key:
            raise LLMUnavailable("GEMINI_API_KEY is not set")
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # retries are handled (and measured) by LLMClient
            thinking_budget=0,
            max_output_tokens=2048,
        )
    if provider == "bedrock":
        # Implemented against the same interface but NOT tested (no Bedrock access).
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as exc:  # optional extra
            raise LLMUnavailable("pip install '.[bedrock]' to use Bedrock") from exc
        return ChatBedrockConverse(
            model=settings.bedrock_model_id,
            region_name=settings.aws_region,
            temperature=settings.llm_temperature,
            max_tokens=2048,
        )
    if provider == "local":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise LLMUnavailable("pip install '.[local]' to use the local engine") from exc
        # No num_ctx / keep_alive: the tag pins context size and the daemon keeps
        # models loaded; overriding either would force a reload on the engine.
        return ChatOllama(
            model=settings.assistant_tag,
            base_url=settings.engine_base_url.rstrip("/"),
            temperature=max(settings.llm_temperature, 0.1),
            client_kwargs={"timeout": httpx.Timeout(settings.local_timeout_s, connect=5.0)},
            validate_model_on_init=False,
        )
    raise LLMUnavailable(f"LLM provider is '{provider}'")


def resolve_backends(settings: Settings) -> list[Backend]:
    """Ordered backends for this run. Replay mode needs no live model: the backend
    answers from its recorded cache only."""
    wanted = AUTO_ORDER if settings.llm_provider == "auto" else (settings.llm_provider,)
    if settings.llm_provider == "none":
        return []
    backends: list[Backend] = []
    for provider in wanted:
        name = model_name(settings, provider)
        cache = JsonlCache(cache_file(settings, name), settings.llm_cache_mode)
        method = "json_schema" if provider == "local" else None
        if settings.llm_cache_mode == "replay":
            # Replay is read-only and deterministic: never call a live model, and use a
            # single recording (no silent fall-through to another model's answers).
            if len(cache) or provider == wanted[-1]:
                # an empty last candidate is kept so misses are counted, not hidden
                backends.append(Backend(name, provider, None, cache, method))
                break
            continue
        reason = check_available(settings, provider)
        if reason:
            log_event("llm.backend_unavailable", provider=provider, reason=reason)
            continue
        try:
            model = get_chat_model(settings, provider)
        except LLMUnavailable as exc:
            log_event("llm.backend_unavailable", provider=provider, reason=str(exc))
            continue
        backends.append(Backend(name, provider, model, cache, method))
    log_event("llm.backends", order=[f"{b.provider}:{b.name}" for b in backends])
    return backends
