"""Schema instantiation, validation, computed fields."""

from __future__ import annotations

import math

import pytest

from rm.memory.schemas import (
    Episode,
    Event,
    MemoryContext,
    MemoryLayer,
    Pattern,
    Principle,
    RetrievalQuery,
    SurpriseSignal,
    UpdateRecord,
)


def test_event_basic():
    e = Event(
        trajectory_id="t1",
        step_idx=0,
        state="  s  ",
        action="  a  ",
        observation="  o  ",
    )
    assert e.event_id  # uuid4 default
    assert e.state == "s" and e.action == "a" and e.observation == "o"
    assert e.reward is None
    assert e.embedding is None


def test_episode_validates_step_order():
    Episode(trajectory_id="t1", start_step=0, end_step=5,
            sub_goal="x", summary="y", outcome="success")
    with pytest.raises(Exception):
        Episode(trajectory_id="t1", start_step=10, end_step=5,
                sub_goal="x", summary="y", outcome="success")


def test_pattern_confidence_and_uncertainty():
    p = Pattern(condition="c", action_template="a", expected_effect="e")
    assert p.confidence == pytest.approx(0.5)            # Beta(1,1)
    assert p.uncertainty == pytest.approx(1.0 / 12.0)
    p.alpha = 9.0
    p.beta = 1.0
    assert p.confidence == pytest.approx(0.9)
    assert p.uncertainty < 0.05


def test_principle_confidence():
    pr = Principle(statement="Always check reversibility")
    assert math.isclose(pr.confidence, 0.5)


def test_memory_context_helpers():
    ctx = MemoryContext()
    assert ctx.is_empty()
    assert ctx.total_items() == 0
    ctx.principles.append(Principle(statement="x"))
    assert not ctx.is_empty()
    assert ctx.total_items() == 1


def test_retrieval_query_defaults():
    q = RetrievalQuery(query_text="foo")
    assert q.k_principle == 3 and q.k_pattern == 5 and q.k_episode == 3
    assert 0 < q.min_pattern_conf < 1


def test_surprise_signal_bounds():
    SurpriseSignal(pattern_id="p", episode_id="e", score=0.0,
                   predicted="x", actual="y", backend="llm_judge")
    SurpriseSignal(pattern_id="p", episode_id="e", score=1.0,
                   predicted="x", actual="y", backend="embed_delta")
    with pytest.raises(Exception):
        SurpriseSignal(pattern_id="p", episode_id="e", score=1.5,
                       predicted="x", actual="y", backend="llm_judge")


def test_update_record_optional_revision():
    r = UpdateRecord(pattern_id="p", episode_id="e", surprise=0.3,
                     delta_alpha=0.7, delta_beta=0.3)
    assert not r.triggered_revision
    assert r.new_pattern_id is None


def test_layer_enum_string_values():
    assert MemoryLayer.EVENT.value == "event"
    assert MemoryLayer.EPISODE.value == "episode"
    assert MemoryLayer.PATTERN.value == "pattern"
    assert MemoryLayer.PRINCIPLE.value == "principle"
