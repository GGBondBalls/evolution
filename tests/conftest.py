"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from rm.llm.client import MockLLMClient
from rm.llm.embed import MockEmbedder
from rm.memory.store import MemoryStore


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient(default="Thought: think.\nAction: look")


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=64)


@pytest.fixture
def store(mock_embedder) -> MemoryStore:
    s = MemoryStore(
        sqlite_path=":memory:",
        qdrant_url=None,
        collection_prefix="rmtest",
        vector_size=mock_embedder.dim,
        distance="Cosine",
    )
    yield s
    s.close()
