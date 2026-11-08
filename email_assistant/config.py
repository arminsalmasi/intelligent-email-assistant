"""Runtime configuration (environment-driven, validated by pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -----------------------------------------------------------------
    # gemini  : Google Gemini via langchain-google-genai (tested path)
    # bedrock : AWS Bedrock via langchain-aws (implemented, NOT tested - no access)
    # none    : deterministic heuristics only (offline / CI)
    llm_provider: Literal["gemini", "bedrock", "none"] = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    bedrock_model_id: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    aws_region: str = "eu-central-1"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 4
    # LLM response cache: off | replay (read only) | record (read + write)
    llm_cache_mode: Literal["off", "replay", "record"] = "off"
    llm_cache_dir: Path = ROOT_DIR / "evals" / "llm_cache"
    llm_requests_per_minute: int = 0  # 0 = unlimited; throttle for free tiers

    # --- Embeddings (memory retrieval) ---------------------------------------
    # gemini : semantic embeddings via API (default)
    # hashing: deterministic lexical embeddings, offline
    embedding_provider: Literal["gemini", "hashing"] = "gemini"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 768
    # off | replay | record. In replay mode a cache miss raises - vector spaces
    # from different embedders must never be mixed.
    embedding_cache_mode: Literal["off", "replay", "record"] = "off"
    embedding_cache_dir: Path = ROOT_DIR / "evals" / "embed_cache"

    # --- Data / persistence --------------------------------------------------
    fixtures_dir: Path = ROOT_DIR / "fixtures"
    data_dir: Path = ROOT_DIR / "data"
    memory_db: str = "memory.db"
    checkpoint_db: str = "checkpoints.db"
    telemetry_db: str = "telemetry.db"

    # --- Clock ---------------------------------------------------------------
    # Fixed clock makes the demo/eval reproducible. Empty = real wall clock.
    fixed_now: str | None = "2026-11-09T09:00:00+01:00"

    # --- Memory --------------------------------------------------------------
    episodic_reuse_threshold: float = 0.92  # reuse an approved reply as template
    episodic_example_threshold: float = 0.55  # use as few-shot example

    # --- Safety --------------------------------------------------------------
    max_outbound_per_run: int = 25
    injection_quarantine_threshold: float = 0.5

    # --- Observability -------------------------------------------------------
    tracing_backend: Literal["none", "mlflow", "langsmith"] = "none"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "email-assistant"
    log_level: str = "INFO"
    log_json: bool = True

    def db_path(self, name: str) -> str:
        if name == ":memory:":
            return name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return str(self.data_dir / name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
