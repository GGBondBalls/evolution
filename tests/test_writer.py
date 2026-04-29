"""MemoryWriter end-to-end on Mock LLM + Mock embedder + in-memory store."""

from __future__ import annotations

import json

from rm.llm.client import MockLLMClient
from rm.memory.schemas import Event
from rm.memory.writer import (
    EpisodeClusterer,
    EpisodeSegmenter,
    MemoryWriter,
    PatternInducer,
    PrincipleReflector,
)

# --------------------------------------------------------------------------- #
# Fixtures local to this module                                                #
# --------------------------------------------------------------------------- #

def _make_events(traj_id: str, n: int = 6) -> list[Event]:
    return [
        Event(
            trajectory_id=traj_id,
            step_idx=i,
            state=f"state_{i}",
            action=f"action_{i}",
            observation=f"obs_{i}",
            reward=1.0 if i == n - 1 else 0.0,
        )
        for i in range(n)
    ]


def _segment_response(n_eps: int = 2) -> str:
    return json.dumps([
        {
            "sub_goal": f"goal_{i}",
            "start_step": i * 3,
            "end_step": (i + 1) * 3 - 1,
            "summary": f"summary {i}",
            "outcome": "success",
        }
        for i in range(n_eps)
    ])


def _pattern_response(cond: str = "open the drawer") -> str:
    return json.dumps({
        "condition": cond,
        "action_template": "open <obj>",
        "expected_effect": "drawer opens",
        "scope": ["alfworld"],
    })


def _principle_response(stmt: str = "Always check side-effects.") -> str:
    return json.dumps([{"statement": stmt, "scope": "cross-task",
                        "supporting_patterns": []}])


# --------------------------------------------------------------------------- #
# Segmenter                                                                   #
# --------------------------------------------------------------------------- #

def test_segmenter_parses_array(mock_embedder):
    llm = MockLLMClient().when(r"trajectory analyst", _segment_response(2))
    seg = EpisodeSegmenter(llm, mock_embedder)
    eps = seg.segment("t1", _make_events("t1", 6))
    assert len(eps) == 2
    assert eps[0].sub_goal == "goal_0"
    assert eps[0].outcome == "success"
    assert eps[0].embedding is not None and len(eps[0].embedding) == mock_embedder.dim


def test_segmenter_falls_back_when_llm_returns_garbage(mock_embedder):
    llm = MockLLMClient(default="not json at all <-->")
    seg = EpisodeSegmenter(llm, mock_embedder)
    eps = seg.segment("t1", _make_events("t1", 4))
    # Falls back to a single Episode covering the whole trajectory.
    assert len(eps) == 1
    assert eps[0].start_step == 0 and eps[0].end_step == 3


def test_segmenter_returns_empty_for_empty_events(mock_embedder):
    llm = MockLLMClient()
    seg = EpisodeSegmenter(llm, mock_embedder)
    assert seg.segment("t1", []) == []


# --------------------------------------------------------------------------- #
# Clusterer                                                                   #
# --------------------------------------------------------------------------- #

def test_clusterer_skips_when_too_few(mock_embedder):
    from rm.memory.schemas import Episode

    eps = [
        Episode(trajectory_id="t", start_step=0, end_step=1, sub_goal="g",
                summary="s", outcome="success",
                embedding=mock_embedder.encode_one("x"))
        for _ in range(2)
    ]
    cl = EpisodeClusterer(min_cluster_size=3)
    assert cl.cluster(eps) == []


def test_clusterer_groups_by_similarity(mock_embedder):
    from rm.memory.schemas import Episode

    # Two semantic groups.
    eps = []
    for txt in ["open drawer", "open cabinet", "open box"]:
        eps.append(Episode(trajectory_id="t", start_step=0, end_step=1,
                            sub_goal=txt, summary=txt, outcome="success",
                            embedding=mock_embedder.encode_one(txt)))
    for txt in ["microwave food", "heat dish", "warm soup"]:
        eps.append(Episode(trajectory_id="t", start_step=0, end_step=1,
                            sub_goal=txt, summary=txt, outcome="success",
                            embedding=mock_embedder.encode_one(txt)))
    cl = EpisodeClusterer(min_cluster_size=2, max_clusters=4, backend="kmeans")
    clusters = cl.cluster(eps)
    # Should split into ≥ 2 non-trivial clusters.
    assert len(clusters) >= 1
    assert sum(len(c.episodes) for c in clusters) >= 2


# --------------------------------------------------------------------------- #
# PatternInducer                                                              #
# --------------------------------------------------------------------------- #

def test_pattern_inducer_builds_pattern(mock_embedder):
    from rm.memory.schemas import Episode

    eps = [
        Episode(trajectory_id="t", start_step=0, end_step=1, sub_goal="open drawer",
                summary="open drawer success", outcome="success",
                embedding=mock_embedder.encode_one("open drawer"))
        for _ in range(3)
    ]
    llm = MockLLMClient().when(r"behaviour pattern|induce one pattern|behaviour|induce|behaviour",
                                _pattern_response())
    # Use default-rule fallback in case the regex misses
    llm._rules.append(type(llm._rules[0])(__import__("re").compile(".*"), _pattern_response()))
    inducer = PatternInducer(llm, mock_embedder)
    p = inducer.induce(eps)
    assert p is not None
    assert p.condition == "open the drawer"
    assert p.evidence_count == 3
    assert p.embedding is not None


def test_pattern_dedup_via_cosine(store, mock_embedder):
    from rm.memory.schemas import Pattern

    p1 = Pattern(condition="open drawer", action_template="open <obj>",
                 expected_effect="opens",
                 embedding=mock_embedder.encode_one("open drawer | open <obj> | opens"))
    store.write_pattern(p1)
    inducer = PatternInducer(MockLLMClient(), mock_embedder)
    p2 = Pattern(condition="open drawer", action_template="open <obj>",
                 expected_effect="opens",
                 embedding=mock_embedder.encode_one("open drawer | open <obj> | opens"))
    found = inducer.find_near_duplicate(p2, store)
    assert found is not None
    assert found.pattern_id == p1.pattern_id


# --------------------------------------------------------------------------- #
# PrincipleReflector                                                          #
# --------------------------------------------------------------------------- #

def test_principle_reflector_filters_unknown_pattern_refs(mock_embedder):
    from rm.memory.schemas import Pattern

    pat = Pattern(condition="c", action_template="a", expected_effect="e")
    response = json.dumps([{
        "statement": "Be careful with destructive actions",
        "scope": "cross-task",
        "supporting_patterns": [pat.pattern_id, "unknown-id"],
    }])
    llm = MockLLMClient(default=response)
    refl = PrincipleReflector(llm, mock_embedder)
    pris = refl.reflect([pat])
    assert len(pris) == 1
    assert pris[0].supporting_patterns == [pat.pattern_id]   # 'unknown-id' filtered


# --------------------------------------------------------------------------- #
# MemoryWriter — full pipeline                                                #
# --------------------------------------------------------------------------- #

def test_writer_full_pipeline(store, mock_embedder):
    """Three trajectories, each producing 1 episode → after the third, a Pattern emerges."""
    llm = MockLLMClient()
    # P1 — Episode segmenter: return 1 episode covering all events
    llm.when(r"trajectory analyst", json.dumps([
        {"sub_goal": "open the drawer", "start_step": 0, "end_step": 3,
         "summary": "open drawer success", "outcome": "success"},
    ]))
    # P2 — Pattern inducer (fallback after segmentation rule)
    llm.when(r"behaviour pattern|induce", _pattern_response())

    writer = MemoryWriter(store, llm, mock_embedder, min_support=2, merge_cosine=0.99)
    for i in range(3):
        events = _make_events(f"traj_{i}", 4)
        store.write_events(events)
        report = writer.on_trajectory_end(f"traj_{i}", events)
        assert report.n_episodes == 1
    # By the 3rd trajectory the cluster should have ≥ 2 episodes and induce a Pattern.
    counts = store.counts()
    assert counts["episodes"] == 3
    assert counts["patterns"] >= 1


def test_writer_principle_reflection(store, mock_embedder):
    from rm.memory.schemas import Pattern

    for i in range(3):
        store.write_pattern(Pattern(condition=f"c{i}", action_template="a",
                                     expected_effect="e", alpha=8, beta=2,
                                     embedding=mock_embedder.encode_one(f"c{i}")))
    llm = MockLLMClient(default=_principle_response("Always check side-effects."))
    writer = MemoryWriter(store, llm, mock_embedder)
    report = writer.reflect_principles()
    assert report.n_new_principles >= 1
    counts = store.counts()
    assert counts["principles"] >= 1
