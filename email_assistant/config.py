"""Runtime configuration (environment-driven, validated by pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -----------------------------------------------------------------
    # auto    : chain of every available backend, in order gemini -> bedrock ->
    #           local; quota/auth/connection failures fall through at runtime
    # gemini  : Google Gemini via langchain-google-genai
    # bedrock : AWS Bedrock via langchain-aws (Claude Haiku 4.5, EU inference profile)
    # local   : self-hosted Ollama engine (e.g. Qwen3 "assistant:32k" on a Mac Studio)
    # none    : deterministic heuristics only (offline / CI)
    llm_provider: Literal["auto", "gemini", "bedrock", "local", "none"] = "auto"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    bedrock_model_id: str = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"  # EU inference profile
    aws_region: str = "eu-central-1"
    # Bedrock API key (bearer token). Alternative to IAM credentials; boto3 reads it
    # from the environment, so it is exported there when the model is built.
    aws_bearer_token_bedrock: SecretStr | None = None
    llm_temperature: float = 0.0
    # Local engine (Ollama). Never set num_ctx: the context size lives in the tag.
    engine_base_url: str = "http://127.0.0.1:11434"
    assistant_tag: str = "assistant:32k"
    local_timeout_s: float = 300.0  # first token on long prompts can take tens of seconds
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 4
    # LLM response cache: off | replay (read only) | record (read + write)
    llm_cache_mode: Literal["off", "replay", "record"] = "off"
    llm_cache_dir: Path = ROOT_DIR / "evals" / "llm_cache"
    llm_requests_per_minute: int = 0  # 0 = unlimited; throttle for free tiers

    # --- Embeddings (memory retrieval) ---------------------------------------
    # auto   : gemini if a key is configured, otherwise hashing
    # gemini : semantic embeddings via API
    # hashing: deterministic lexical embeddings, offline
    # (The local engine only serves chat tags; loading an embedding model there
    #  would evict one of the two pinned models, so it is not used for embeddings.)
    embedding_provider: Literal["auto", "gemini", "hashing"] = "auto"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 768
    # off | replay | record. In replay mode a cache miss raises - vector spaces
    # from different embedders must never be mixed.
    embedding_cache_mode: Literal["off", "replay", "record"] = "off"
    embedding_cache_dir: Path = ROOT_DIR / "evals" / "embed_cache"

    # --- Data / persistence --------------------------------------------------
    fixtures_dir: Path = ROOT_DIR / "fixtures"
    mail_eml_dir: Path | None = None  # load the inbox from .eml files instead of the fixture
    data_dir: Path = ROOT_DIR / "data"
    memory_db: str = "memory.db"
    checkpoint_db: str = "checkpoints.db"
    inbox_db: str = "inbox.db"
    telemetry_db: str = "telemetry.db"

    # --- Clock ---------------------------------------------------------------
    # Fixed clock makes the demo/eval reproducible. Empty = real wall clock.
    fixed_now: str | None = "2026-11-09T09:00:00+01:00"

    # --- Memory --------------------------------------------------------------
    episodic_reuse_threshold: float = 0.92  # reuse an approved reply as template
    episodic_example_threshold: float = 0.55  # use as few-shot example

    # --- Safety --------------------------------------------------------------
    max_outbound_per_thread: int = 3
    injection_quarantine_threshold: float = 0.5

    # --- Observability -------------------------------------------------------
    tracing_backend: Literal["none", "mlflow", "langsmith"] = "none"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "email-assistant"
    log_level: str = "INFO"
    log_json: bool = True

    # --- API -----------------------------------------------------------------
    api_key: str | None = None  # if set, required as X-API-Key on every API call

    def db_path(self, name: str) -> str:
        if name == ":memory:":
            return name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return str(self.data_dir / name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
