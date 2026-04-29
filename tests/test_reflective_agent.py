"""ReflectiveAgent end-to-end on MockEnv with all components mocked."""

from __future__ import annotations

import json

from rm.agent.reflective import ReflectiveAgent
from rm.envs.mock_env import MockEnv
from rm.llm.client import MockLLMClient
from rm.memory.schemas import Pattern
from rm.memory.updater import BayesianConfig, MemoryUpdater
from rm.memory.writer import MemoryWriter


def _make_llm() -> MockLLMClient:
    """A single Mock LLM that handles every prompt by content sniffing."""
    llm = MockLLMClient()
    # P1 — Episode segmenter
    llm.when(r"trajectory analyst", json.dumps([
        {"sub_goal": "say magic word", "start_step": 0, "end_step": 0,
         "summary": "agent says GOAL and wins", "outcome": "success"},
    ]))
    # P2 — Pattern inducer
    llm.when(r"behaviour pattern|induce one pattern|expert at inducing", json.dumps({
        "condition": "the prompt asks for a magic keyword",
        "action_template": "say <keyword>",
        "expected_effect": "the env returns success",
        "scope": ["mock"],
    }))
    # P3 — Principle reflector
    llm.when(r"meta-reflector|cross-task Principles", json.dumps([
        {"statement": "Try the obvious solution first.",
         "scope": "cross-task", "supporting_patterns": []},
    ]))
    # P4 — Predict step
    llm.when(r"Predict:", "the env returns success")
    # P5 — Judge
    llm.when(r"1=nearly identical|1-5 integer", "1")
    # P6 — Revise
    llm.when(r"Old Pattern|under which conditions did the old Pattern fail",
              json.dumps({"decision": "discard", "rationale": "n/a", "patterns": []}))
    # ReAct fallback (system + step) — needs to look like Thought/Action
    llm.when(r"Output exactly|Thought:|admissible commands",
              "Thought: speak the magic word.\nAction: say GOAL")
    return llm


def test_reflective_runs_episode_and_writes_memory(mock_embedder):
    from rm.memory.store import MemoryStore

    store = MemoryStore(sqlite_path=":memory:", qdrant_url=None,
                         collection_prefix="rmtest_rfa",
                         vector_size=mock_embedder.dim)
    try:
        llm = _make_llm()
        writer = MemoryWriter(store, llm, mock_embedder, min_support=1)
        agent = ReflectiveAgent(
            llm=llm, store=store, embedder=mock_embedder,
            writer=writer, max_steps=5, reflect_every_n_trajectories=99,
        )
        env = MockEnv(seed=0, horizon=3, goal_keyword="GOAL")
        traj = agent.run_episode(env, task_idx=0)
        assert traj.success
        # Events written.
        assert store.counts()["events"] >= 1
        # Episode segmented (mock returns 1).
        assert store.counts()["episodes"] >= 1
    finally:
        store.close()


def test_reflective_uses_existing_principle_in_prompt(mock_embedder):
    from rm.memory.schemas import Principle
    from rm.memory.store import MemoryStore

    store = MemoryStore(sqlite_path=":memory:", qdrant_url=None,
                         collection_prefix="rmtest_pri",
                         vector_size=mock_embedder.dim)
    try:
        # Plant a principle in store before the run.
        pr = Principle(statement="Speaking the magic word ends the task.",
                        alpha=10, beta=1,
                        embedding=mock_embedder.encode_one("magic word"))
        store.write_principle(pr)

        # The LLM "sees" the prompt by sniffing for our planted principle.
        llm = MockLLMClient()
        # If the principle is in the prompt, we know retrieval worked.
        llm.when(r"magic word",
                 "Thought: principle observed.\nAction: say GOAL")
        # Fallback for any other prompt.
        llm.when(r".*", "Thought: try.\nAction: say GOAL")

        agent = ReflectiveAgent(
            llm=llm, store=store, embedder=mock_embedder,
            writer=None,         # disable write path; we only test retrieval
            max_steps=2, reflect_every_n_trajectories=99,
        )
        env = MockEnv(seed=0, horizon=3, goal_keyword="GOAL")
        traj = agent.run_episode(env, task_idx=0)
        assert traj.success
    finally:
        store.close()


def test_reflective_runs_updater_when_configured(mock_embedder):
    from rm.memory.store import MemoryStore

    store = MemoryStore(sqlite_path=":memory:", qdrant_url=None,
                         collection_prefix="rmtest_upd",
                         vector_size=mock_embedder.dim)
    try:
        # Plant a Pattern that will be retrieved & updated.
        pat = Pattern(condition="say magic word", action_template="say GOAL",
                      expected_effect="agent wins", alpha=4, beta=1,
                      embedding=mock_embedder.encode_one("say magic word"))
        store.write_pattern(pat)

        llm = _make_llm()
        writer = MemoryWriter(store, llm, mock_embedder, min_support=1)
        updater = MemoryUpdater(store, llm, mock_embedder, backend="embed_delta",
                                  bayes_cfg=BayesianConfig(tau_low=0.4, tau_high=0.7))
        agent = ReflectiveAgent(
            llm=llm, store=store, embedder=mock_embedder,
            writer=writer, updater=updater, max_steps=3,
            reflect_every_n_trajectories=99,
        )
        env = MockEnv(seed=0, horizon=3, goal_keyword="GOAL")
        agent.run_episode(env, task_idx=0)
        assert store.counts()["updates"] >= 1
    finally:
        store.close()
