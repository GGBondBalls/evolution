"""Memory store: write/read round-trip + ANN retrieval (in-memory Qdrant)."""

from __future__ import annotations

from rm.memory.schemas import Episode, Event, Pattern, Principle, RetrievalQuery, UpdateRecord

# --------------------------------------------------------------------------- #
# Round-trip                                                                  #
# --------------------------------------------------------------------------- #

def test_event_roundtrip(store, mock_embedder):
    ev = Event(trajectory_id="t1", step_idx=0, state="s", action="a", observation="o",
               embedding=mock_embedder.encode_one("s a o"))
    store.write_event(ev)
    out = store.get_events("t1")
    assert len(out) == 1
    assert out[0].event_id == ev.event_id


def test_episode_roundtrip(store, mock_embedder):
    ep = Episode(trajectory_id="t1", start_step=0, end_step=2,
                 sub_goal="g", summary="s", outcome="success",
                 embedding=mock_embedder.encode_one("g s"))
    store.write_episode(ep)
    out = store.get_episodes("t1")
    assert len(out) == 1
    assert out[0].episode_id == ep.episode_id
    assert out[0].outcome == "success"


def test_pattern_roundtrip_keeps_alpha_beta(store, mock_embedder):
    p = Pattern(condition="c", action_template="a", expected_effect="e",
                alpha=4.0, beta=2.0, evidence_count=5,
                embedding=mock_embedder.encode_one("c a e"))
    store.write_pattern(p)
    p2 = store.get_pattern(p.pattern_id)
    assert p2 is not None
    assert p2.alpha == 4.0 and p2.beta == 2.0
    assert p2.evidence_count == 5
    assert abs(p2.confidence - (4.0 / 6.0)) < 1e-6


def test_principle_roundtrip(store, mock_embedder):
    pr = Principle(statement="Be careful with destructive actions",
                   embedding=mock_embedder.encode_one("destructive"))
    store.write_principle(pr)
    pr2 = store.get_principle(pr.principle_id)
    assert pr2 is not None
    assert pr2.statement.startswith("Be careful")


def test_update_record_persisted(store):
    rec = UpdateRecord(pattern_id="p1", episode_id="e1", surprise=0.3,
                       delta_alpha=0.7, delta_beta=0.3)
    store.record_update(rec)
    counts = store.counts()
    assert counts["updates"] == 1


# --------------------------------------------------------------------------- #
# Vector retrieval                                                            #
# --------------------------------------------------------------------------- #

def test_query_patterns_by_vector(store, mock_embedder):
    # Two patterns with different embeddings; query close to one of them.
    p_close = Pattern(condition="open the drawer", action_template="open <obj>",
                      expected_effect="drawer opens", alpha=8, beta=2,
                      embedding=mock_embedder.encode_one("open drawer"))
    p_far = Pattern(condition="microwave food", action_template="heat <obj>",
                    expected_effect="food becomes hot", alpha=8, beta=2,
                    embedding=mock_embedder.encode_one("microwave food"))
    store.write_pattern(p_close)
    store.write_pattern(p_far)
    q = mock_embedder.encode_one("open drawer")
    rq = RetrievalQuery(query_text="open drawer", k_pattern=2,
                         min_principle_conf=0.0, min_pattern_conf=0.0)
    ctx = store.retrieve(q, rq)
    assert any(p.pattern_id == p_close.pattern_id for p in ctx.patterns)
    # Closest should rank first.
    assert ctx.patterns[0].pattern_id == p_close.pattern_id


def test_retrieval_drills_down_to_episodes_when_patterns_weak(store, mock_embedder):
    # Low-confidence pattern → retrieval should also pull episodes.
    weak = Pattern(condition="x", action_template="y", expected_effect="z",
                   alpha=1.0, beta=4.0,    # confidence = 0.2
                   embedding=mock_embedder.encode_one("weak"))
    store.write_pattern(weak)
    ep = Episode(trajectory_id="t1", start_step=0, end_step=2,
                 sub_goal="g", summary="s", outcome="success",
                 embedding=mock_embedder.encode_one("weak"))
    store.write_episode(ep)
    q = mock_embedder.encode_one("weak")
    rq = RetrievalQuery(query_text="weak", k_pattern=1, k_episode=2,
                         min_principle_conf=0.0, min_pattern_conf=0.0)
    ctx = store.retrieve(q, rq, confidence_threshold_for_episode_drilldown=0.5)
    assert len(ctx.episodes) >= 1


# --------------------------------------------------------------------------- #
# Counts / housekeeping                                                       #
# --------------------------------------------------------------------------- #

def test_counts_reflect_writes(store):
    assert store.counts() == {"events": 0, "episodes": 0, "patterns": 0,
                              "principles": 0, "updates": 0}
    store.write_event(Event(trajectory_id="t", step_idx=0, state="s",
                            action="a", observation="o"))
    assert store.counts()["events"] == 1


def test_all_patterns_returns_everything(store):
    for i in range(3):
        store.write_pattern(Pattern(condition=f"c{i}", action_template="a",
                                    expected_effect="e"))
    assert len(store.all_patterns()) == 3
