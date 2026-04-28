"""Sentence-embedding wrapper.

We support three backends:

* ``sentence_transformers`` (default) — BGE-M3, Qwen3-Embedding-0.6B, etc.
* ``openai`` / ``openai_compat`` — text-embedding-3-small / large (1536 / 3072 dim)
  or any OpenAI-compatible ``/embeddings`` endpoint.
* ``mock`` — deterministic hash → fixed-dim vector. No deps, no network.

The ``Embedder`` interface is intentionally minimal::

    e = build_embedder(cfg)
    vec = e.encode_one("hello world")             # list[float]
    mat = e.encode_many(["a", "b"])               # list[list[float]]
    e.dim                                          # int
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rm.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Base + factory                                                              #
# --------------------------------------------------------------------------- #

class Embedder:
    """Abstract base — subclasses implement ``_encode``."""

    dim: int = 0
    model_name: str = "abstract"

    def encode_one(self, text: str) -> list[float]:
        return self.encode_many([text])[0]

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(list(texts))

    def _encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Sentence-Transformers                                                       #
# --------------------------------------------------------------------------- #

class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "auto",
        cache_dir: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.model_name = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
        logger.info(f"Loading SentenceTransformer({model}) on {device}")
        self._st = SentenceTransformer(model, device=device, cache_folder=cache_dir)
        self.dim = int(self._st.get_sentence_embedding_dimension() or 0)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embs = self._st.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [list(map(float, v)) for v in embs]


# --------------------------------------------------------------------------- #
# OpenAI-compat                                                               #
# --------------------------------------------------------------------------- #

class OpenAICompatEmbedder(Embedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        self.model_name = model
        self.batch_size = batch_size
        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)
        # Probe dimension with a tiny call.
        probe = self._client.embeddings.create(model=model, input=["dim probe"])
        self.dim = len(probe.data[0].embedding)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            resp = self._client.embeddings.create(model=self.model_name, input=chunk)
            out.extend(list(d.embedding) for d in resp.data)
        return out


# --------------------------------------------------------------------------- #
# Mock — for unit tests                                                       #
# --------------------------------------------------------------------------- #

class MockEmbedder(Embedder):
    """Deterministic hash-based embedder. Same string ⇒ same vector, always."""

    def __init__(self, dim: int = 64, normalize: bool = True) -> None:
        self.dim = dim
        self.model_name = "mock"
        self.normalize = normalize

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        # blake2b's digest_size is capped at 64 bytes; chain SHA256 (32 bytes
        # each) to support any vector dimension.
        needed = self.dim * 2
        buf = bytearray()
        chunk = 0
        text_bytes = text.encode("utf-8")
        while len(buf) < needed:
            buf += hashlib.sha256(chunk.to_bytes(4, "big") + text_bytes).digest()
            chunk += 1
        h = bytes(buf[:needed])
        # Map each pair of bytes to a [-1, 1) float.
        vec = [
            ((h[2 * i] << 8) | h[2 * i + 1]) / 65535.0 * 2.0 - 1.0
            for i in range(self.dim)
        ]
        if self.normalize:
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / norm for x in vec]
        return vec


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class EmbedderSpec:
    backend: str = "sentence_transformers"
    model: str = "BAAI/bge-m3"
    device: str = "auto"
    cache_dir: str | None = None
    batch_size: int = 32
    base_url: str | None = None
    api_key: str | None = None
    dim: int = 64                # only used for mock


def build_embedder(cfg: dict[str, Any] | Any) -> Embedder:
    cfg = dict(cfg) if not isinstance(cfg, dict) else cfg
    backend = cfg.get("backend", "sentence_transformers")
    if backend == "mock":
        return MockEmbedder(dim=int(cfg.get("dim", 64)))
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model=cfg.get("model", "BAAI/bge-m3"),
            device=cfg.get("device", "auto"),
            cache_dir=cfg.get("cache_dir"),
            batch_size=int(cfg.get("batch_size", 32)),
        )
    if backend in {"openai", "openai_compat"}:
        return OpenAICompatEmbedder(
            model=cfg.get("model", "text-embedding-3-small"),
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
            batch_size=int(cfg.get("batch_size", 64)),
        )
    raise ValueError(f"Unknown embedder backend: {backend}")


# --------------------------------------------------------------------------- #
# Tiny utilities                                                              #
# --------------------------------------------------------------------------- #

def cosine(u: Iterable[float], v: Iterable[float]) -> float:
    u = list(u)
    v = list(v)
    if not u or not v:
        return 0.0
    dot = sum(a * b for a, b in zip(u, v, strict=False))
    nu = math.sqrt(sum(a * a for a in u)) or 1.0
    nv = math.sqrt(sum(b * b for b in v)) or 1.0
    return dot / (nu * nv)
