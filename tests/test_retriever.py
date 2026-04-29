"""MemoryRetriever — query embedding + formatting."""

from __future__ import annotations

from rm.memory.retriever import MemoryRetriever, RetrievalConfig
from rm.memory.schemas import Episode, Pattern, Principle


def test_retrieve_empty_store_returns_empty_context(store, mock_embedder):
    r = MemoryRetriever(store, mock_embedder)
    ctx = r.retrieve("anything")
    assert ctx.is_empty()


def test_retrieve_finds_pattern(store, mock_embedder):
    p = Pattern(condition="open drawer", action_template="open <obj>",
                expected_effect="opens", alpha=8, beta=2,
                embedding=mock_embedder.encode_one("open drawer"))
    store.write_pattern(p)
    r = MemoryRetriever(store, mock_embedder, RetrievalConfig(min_pattern_conf=0.0))
    ctx = r.retrieve("open the drawer")
    assert any(pp.pattern_id == p.pattern_id for pp in ctx.patterns)


def test_format_for_prompt_contains_layers(store, mock_embedder):
    pr = Principle(statement="Be careful", alpha=5, beta=1,
                   embedding=mock_embedder.encode_one("careful"))
    p = Pattern(condition="open drawer", action_template="open <obj>",
                expected_effect="opens", alpha=8, beta=2,
                embedding=mock_embedder.encode_one("open drawer"))
    ep = Episode(trajectory_id="t", start_step=0, end_step=1,
                 sub_goal="open", summary="open drawer", outcome="success",
                 embedding=mock_embedder.encode_one("open drawer"))
    store.write_principle(pr)
    store.write_pattern(p)
    store.write_episode(ep)
    r = MemoryRetriever(store, mock_embedder,
                         RetrievalConfig(min_principle_conf=0.0,
                                         min_pattern_conf=0.0,
                                         confidence_threshold_for_episode_drilldown=1.1))
    ctx = r.retrieve("open drawer")
    text = r.format_for_prompt(ctx)
    assert "Memory Context" in text
    assert "Be careful" in text
    assert "open drawer" in text or "open <obj>" in text


def test_format_empty_context_returns_empty_string(mock_embedder):
    from rm.memory.schemas import MemoryContext

    r = MemoryRetriever.__new__(MemoryRetriever)
    r.cfg = RetrievalConfig()
    r.embedder = mock_embedder
    assert r.format_for_prompt(MemoryContext()) == ""


def test_dim_mismatch_returns_empty(mock_embedder):
    """If retriever's embedder dim ≠ store's vector_size, we don't crash, just empty."""
    from rm.llm.embed import MockEmbedder
    from rm.memory.store import MemoryStore

    s = MemoryStore(sqlite_path=":memory:", qdrant_url=None,
                     collection_prefix="rmtest_dim", vector_size=mock_embedder.dim)
    wrong_dim = MockEmbedder(dim=mock_embedder.dim * 2)
    r = MemoryRetriever(s, wrong_dim)
    ctx = r.retrieve("anything")
    assert ctx.is_empty()
    s.close()
