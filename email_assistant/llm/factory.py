"""Chat-model factory: the only place that knows about providers."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from email_assistant.config import Settings


class LLMUnavailable(RuntimeError):
    pass


def model_name(settings: Settings) -> str:
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    if settings.llm_provider == "bedrock":
        return settings.bedrock_model_id
    return "none"


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
    if settings.llm_provider == "bedrock":
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
    raise LLMUnavailable("LLM provider is 'none'")
