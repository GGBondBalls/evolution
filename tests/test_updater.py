"""Surprise + Bayesian update + Pattern revision."""

from __future__ import annotations

import json

from rm.llm.client import MockLLMClient
from rm.memory.schemas import Episode, Pattern, SurpriseSignal
from rm.memory.updater import (
    BayesianConfig,
    BayesianUpdater,
    MemoryUpdater,
    PatternRevisor,
    SurpriseEngine,
    _judge_text_to_score,
)

# --------------------------------------------------------------------------- #
# _judge_text_to_score                                                         #
# --------------------------------------------------------------------------- #

def test_judge_text_to_score_maps_1_to_5():
    assert _judge_text_to_score("1") == 0.0
    assert _judge_text_to_score("5") == 1.0
    assert _judge_text_to_score("3") == 0.5
    assert _judge_text_to_score("garbage") == 0.5             # uninformative midpoint
    assert _judge_text_to_score("Score: 4 (high)") == 0.75
    assert _judge_text_to_score("99") == 0.5                  # no 1-5 digit → midpoint
    assert _judge_text_to_score("9 then 2 then 8") == 0.25    # picks the 2


# --------------------------------------------------------------------------- #
# SurpriseEngine — backends                                                   #
# --------------------------------------------------------------------------- #

def test_surprise_llm_judge(mock_embedder):
    """Mock LLM returns '5' meaning total refute → score = 1.0."""
    llm = MockLLMClient(default="(some predicted text)")
    # Differentiate P4 (predict) vs P5 (judge) by unique substring of each template.
    llm.when(r"diverges from the prediction", "5")
    llm.when(r"Predict:", "predicted text")

    pat = Pattern(condition="c", action_template="a", expected_effect="opens drawer")
    ep = Episode(trajectory_id="t", start_step=0, end_step=1, sub_goal="x",
                 summary="actual outcome", outcome="success")
    eng = SurpriseEngine(llm, mock_embedder, backend="llm_judge")
    sig = eng.compute(pat, ep)
    assert sig.score == 1.0
    assert sig.backend == "llm_judge"


def test_surprise_embed_delta_close_match(mock_embedder):
    pat = Pattern(condition="c", action_template="a", expected_effect="open drawer")
    ep = Episode(trajectory_id="t", start_step=0, end_step=1, sub_goal="x",
                 summary="open drawer", outcome="success")
    eng = SurpriseEngine(MockLLMClient(), mock_embedder, backend="embed_delta")
    sig = eng.compute(pat, ep)
    # Same text → cosine ≈ 1 → surprise ≈ 0
    assert sig.score < 0.05
    assert sig.backend == "embed_delta"


def test_surprise_embed_delta_far_apart(mock_embedder):
    pat = Pattern(condition="c", action_template="a", expected_effect="open drawer")
    ep = Episode(trajectory_id="t", start_step=0, end_step=1, sub_goal="x",
                 summary="microwave food entirely unrelated", outcome="failure")
    eng = SurpriseEngine(MockLLMClient(), mock_embedder, backend="embed_delta")
    sig = eng.compute(pat, ep)
    assert sig.score > 0.0


def test_surprise_logprob_falls_back_gracefully(mock_embedder):
    pat = Pattern(condition="c", action_template="a", expected_effect="x")
    ep = Episode(trajectory_id="t", start_step=0, end_step=1, sub_goal="x",
                 summary="y", outcome="partial")
    eng = SurpriseEngine(MockLLMClient(), mock_embedder, backend="logprob")
    sig = eng.compute(pat, ep)
    assert 0.0 <= sig.score <= 1.0


# --------------------------------------------------------------------------- #
# BayesianUpdater                                                              #
# --------------------------------------------------------------------------- #

def test_bayes_strong_support_increments_alpha(store):
    pat = Pattern(condition="c", action_template="a", expected_effect="e")
    store.write_pattern(pat)
    upd = BayesianUpdater(store, BayesianConfig(tau_low=0.2, tau_high=0.7))
    sig = SurpriseSignal(pattern_id=pat.pattern_id, episode_id="e1",
                         score=0.05, predicted="p", actual="a", backend="llm_judge")
    rec = upd.update(pat, sig)
    assert rec.delta_alpha == 1.0 and rec.delta_beta == 0.0
    assert pat.alpha == 2.0 and pat.beta == 1.0
    assert "e1" in pat.support_episodes


def test_bayes_strong_refute_increments_beta(store):
    pat = Pattern(condition="c", action_template="a", expected_effect="e")
    store.write_pattern(pat)
    upd = BayesianUpdater(store, BayesianConfig(tau_low=0.2, tau_high=0.7))
    sig = SurpriseSignal(pattern_id=pat.pattern_id, episode_id="e1",
                         score=0.85, predicted="p", actual="a", backend="llm_judge")
    rec = upd.update(pat, sig)
    assert rec.delta_alpha == 0.0 and rec.delta_beta == 1.0
    assert pat.beta == 2.0
    assert "e1" in pat.refute_episodes


def test_bayes_soft_update_in_middle_band(store):
    pat = Pattern(condition="c", action_template="a", expected_effect="e")
    store.write_pattern(pat)
    upd = BayesianUpdater(store, BayesianConfig(tau_low=0.2, tau_high=0.7))
    sig = SurpriseSignal(pattern_id=pat.pattern_id, episode_id="e1",
                         score=0.5, predicted="p", actual="a", backend="llm_judge")
    rec = upd.update(pat, sig)
    assert 0.0 < rec.delta_alpha < 1.0 and 0.0 < rec.delta_beta < 1.0
    assert abs(rec.delta_alpha + rec.delta_beta - 1.0) < 1e-9


def test_bayes_caps_alpha_beta(store):
    pat = Pattern(condition="c", action_template="a", expected_effect="e",
                  alpha=199.5, beta=199.5)
    store.write_pattern(pat)
    upd = BayesianUpdater(store, BayesianConfig(alpha_max=200.0, beta_max=200.0))
    upd.update(pat, SurpriseSignal(pattern_id=pat.pattern_id, episode_id="e1",
                                    score=0.05, predicted="p", actual="a",
                                    backend="llm_judge"))
    assert pat.alpha == 200.0


def test_bayes_needs_revision_after_window_of_refutes(store):
    pat = Pattern(condition="c", action_template="a", expected_effect="e")
    store.write_pattern(pat)
    upd = BayesianUpdater(
        store, BayesianConfig(tau_high=0.7, rewrite_threshold=3, rewrite_window=5)
    )
    for i in range(3):
        upd.update(pat, SurpriseSignal(
            pattern_id=pat.pattern_id, episode_id=f"e{i}",
            score=0.9, predicted="p", actual="a", backend="llm_judge",
        ))
    needs, refute_eids = upd.needs_revision(pat)
    assert needs is True
    assert len(refute_eids) == 3


# --------------------------------------------------------------------------- #
# PatternRevisor                                                              #
# --------------------------------------------------------------------------- #

def test_revisor_refine_returns_one_pattern(mock_embedder):
    response = json.dumps({
        "decision": "refine",
        "rationale": "Old action wrong in this sub-condition.",
        "patterns": [{
            "condition": "drawer is locked",
            "action_template": "unlock <obj> first",
            "expected_effect": "drawer unlocks",
            "scope": ["alfworld"],
        }],
    })
    llm = MockLLMClient(default=response)
    pat = Pattern(condition="open drawer", action_template="open <obj>",
                  expected_effect="opens")
    eps = [Episode(trajectory_id="t", start_step=0, end_step=1, sub_goal="x",
                   summary="locked drawer", outcome="failure") for _ in range(3)]
    rev = PatternRevisor(llm, mock_embedder)
    res = rev.revise(pat, eps)
    assert res.decision == "refine"
    assert len(res.new_patterns) == 1
    assert res.new_patterns[0].parent_pattern_id == pat.pattern_id
    assert res.new_patterns[0].version == pat.version + 1


def test_revisor_split_returns_multiple(mock_embedder):
    response = json.dumps({
        "decision": "split",
        "rationale": "Two distinct sub-conditions.",
        "patterns": [
            {"condition": "x", "action_template": "a1", "expected_effect": "e",
             "scope": []},
            {"condition": "y", "action_template": "a2", "expected_effect": "e",
             "scope": []},
        ],
    })
    llm = MockLLMClient(default=response)
    pat = Pattern(condition="c", action_template="a", expected_effect="e")
    rev = PatternRevisor(llm, mock_embedder)
    res = rev.revise(pat, [])
    assert res.decision == "split"
    assert len(res.new_patterns) == 2


def test_revisor_discard(mock_embedder):
    response = json.dumps({"decision": "discard", "rationale": "irreparable",
                           "patterns": []})
    llm = MockLLMClient(default=response)
    rev = PatternRevisor(llm, mock_embedder)
    pat = Pattern(condition="c", action_template="a", expected_effect="e")
    res = rev.revise(pat, [])
    assert res.decision == "discard"
    assert res.new_patterns == []


# --------------------------------------------------------------------------- #
# MemoryUpdater orchestrator                                                   #
# --------------------------------------------------------------------------- #

def test_memory_updater_processes_multiple_episodes(store, mock_embedder):
    pat = Pattern(condition="open drawer", action_template="open <obj>",
                  expected_effect="opens drawer",
                  embedding=mock_embedder.encode_one("c"))
    store.write_pattern(pat)
    eps = [
        Episode(trajectory_id="t", start_step=0, end_step=1,
                sub_goal="g", summary="opens drawer", outcome="success",
                embedding=mock_embedder.encode_one("opens drawer")),
        Episode(trajectory_id="t", start_step=2, end_step=3,
                sub_goal="g", summary="completely different", outcome="failure",
                embedding=mock_embedder.encode_one("completely different")),
    ]
    for e in eps:
        store.write_episode(e)
    upd = MemoryUpdater(store, MockLLMClient(), mock_embedder, backend="embed_delta")
    report = upd.process_episode(eps[0], retrieved_patterns=[pat])
    assert report.n_signals == 1
    # Episode 0 is similar to pattern.expected_effect → strong support, alpha ↑
    refreshed = store.get_pattern(pat.pattern_id)
    assert refreshed is not None
    assert refreshed.alpha > 1.0
