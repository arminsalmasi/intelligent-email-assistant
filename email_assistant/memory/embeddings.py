"""Embedders for memory retrieval.

* GeminiEmbedder  - semantic embeddings (default)
* HashingEmbedder - deterministic lexical vectors (offline / no key)

Both go through `CachedEmbedder`. In replay mode a miss raises: vectors from
different embedders live in different spaces and must never be mixed.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
from typing import Protocol

from email_assistant.llm.cache import CacheMiss, JsonlCache, cache_key
from email_assistant.observability.metrics import Telemetry

_TOKEN = re.compile(r"[a-zåäöéñáíóúü0-9]+", re.IGNORECASE)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class HashingEmbedder:
    """Unigram + bigram feature hashing with log-scaled counts, L2-normalised."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> list[float]:
        tokens = [t.lower() for t in _TOKEN.findall(text) if len(t) > 2]
        feats = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return normalise([math.copysign(math.log1p(abs(x)), x) for x in v])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class GeminiEmbedder:
    def __init__(self, model: str, dim: int, api_key: str | None = None):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = (
            GoogleGenerativeAIEmbeddings(model=model, google_api_key=key) if key else None
        )
        self.dim = dim
        self.name = f"{model.split('/')[-1]}-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            raise RuntimeError("GEMINI_API_KEY is not set for Gemini embeddings")
        vecs = self._client.embed_documents(
            texts, task_type="SEMANTIC_SIMILARITY", output_dimensionality=self.dim
        )
        return [normalise(list(v)) for v in vecs]


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: JsonlCache, telemetry: Telemetry):
        self.inner = inner
        self.cache = cache
        self.telemetry = telemetry
        self.name = inner.name
        self.dim = inner.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [cache_key(self.name, t) for t in texts]
        out: list[list[float] | None] = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            if self.cache.mode == "replay":
                raise CacheMiss(f"{len(missing)} embedding(s) not in replay cache ({self.name})")
            fresh = self.inner.embed([texts[i] for i in missing])
            for i, v in zip(missing, fresh, strict=True):
                out[i] = v
                self.cache.put(keys[i], v)
        self.telemetry.record(
            "embedding", self.name, len(texts), cache_hits=len(texts) - len(missing)
        )
        return [v for v in out if v is not None]
