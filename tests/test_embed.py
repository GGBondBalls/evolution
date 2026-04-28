"""Mock embedder determinism + dim."""

from __future__ import annotations

import math

from rm.llm.embed import MockEmbedder, build_embedder, cosine


def test_mock_embedder_deterministic():
    e = MockEmbedder(dim=32)
    v1 = e.encode_one("hello world")
    v2 = e.encode_one("hello world")
    assert v1 == v2
    assert len(v1) == 32
    # Normalised → unit norm.
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-6)


def test_mock_embedder_different_inputs_differ():
    e = MockEmbedder(dim=32)
    v1 = e.encode_one("hello")
    v2 = e.encode_one("world")
    # Cosine similarity should be far from 1 for different inputs.
    assert cosine(v1, v2) < 0.9


def test_factory_builds_mock():
    e = build_embedder({"backend": "mock", "dim": 16})
    assert e.dim == 16
    assert len(e.encode_one("x")) == 16


def test_encode_many_batch():
    e = MockEmbedder(dim=8)
    batch = e.encode_many(["a", "b", "c"])
    assert len(batch) == 3 and all(len(v) == 8 for v in batch)


def test_encode_empty():
    e = MockEmbedder(dim=8)
    assert e.encode_many([]) == []
