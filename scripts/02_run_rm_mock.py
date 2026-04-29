"""End-to-end RM run on MockEnv with Mock LLM + Mock embedder.

This is the Round-2 end-to-end demo: ReflectiveAgent retrieves memory,
the writer segments → clusters → induces Patterns, the updater computes
surprise + Bayesian updates, and the runner aggregates metrics.

Everything is in-memory and deterministic — useful as a regression /
"is the RM pipeline alive?" check.

Run::

    python scripts/02_run_rm_mock.py --n_tasks 8 --max_steps 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rm.agent.reflective import ReflectiveAgent
from rm.envs.mock_env import MockEnv
from rm.eval.metrics import compute_metrics
from rm.llm.client import MockLLMClient
from rm.llm.embed import MockEmbedder
from rm.memory.store import MemoryStore
from rm.memory.updater import BayesianConfig, MemoryUpdater
from rm.memory.writer import MemoryWriter
from rm.utils.logging import get_logger, setup_logging
from rm.utils.seeding import set_seed

logger = get_logger(__name__)


def _make_llm() -> MockLLMClient:
    """One Mock LLM that handles every prompt the RM pipeline can issue.

    Routing is by content sniffing — the order matters (most-specific first).
    """
    llm = MockLLMClient()
    # P1 — Episode segmentation
    llm.when(r"trajectory analyst", json.dumps([{
        "sub_goal": "say the magic word",
        "start_step": 0, "end_step": 0,
        "summary": "agent says GOAL and wins immediately",
        "outcome": "success",
    }]))
    # P2 — Pattern induction
    llm.when(r"behaviour pattern|expert at inducing|Induce ONE Pattern", json.dumps({
        "condition": "the prompt asks for a magic keyword",
        "action_template": "say <keyword>",
        "expected_effect": "the env returns success",
        "scope": ["mock"],
    }))
    # P3 — Principle reflection
    llm.when(r"meta-reflector|cross-task Principles", json.dumps([{
        "statement": "When in doubt, try the obvious solution first.",
        "scope": "cross-task",
        "supporting_patterns": [],
    }]))
    # P5 — divergence judge (must come BEFORE P4 because it has more specific markers)
    llm.when(r"diverges from the prediction", "1")
    # P4 — predict
    llm.when(r"Predict:", "the env returns success")
    # P6 — revise
    llm.when(r"Old Pattern|under which conditions did the old Pattern fail",
              json.dumps({"decision": "discard", "rationale": "n/a", "patterns": []}))
    # ReAct fallback
    llm.when(r"Output exactly|admissible commands|Thought:",
              "Thought: speak the magic word.\nAction: say GOAL")
    # Catch-all
    llm.when(r".*", "Thought: think.\nAction: say GOAL")
    return llm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reflect_every", type=int, default=4)
    ap.add_argument("--bayes_tau_low", type=float, default=0.20)
    ap.add_argument("--bayes_tau_high", type=float, default=0.70)
    ap.add_argument("--surprise_backend", default="embed_delta",
                    choices=["llm_judge", "embed_delta", "logprob"])
    args = ap.parse_args()

    setup_logging(level="INFO")
    set_seed(args.seed)

    embedder = MockEmbedder(dim=64)
    store = MemoryStore(
        sqlite_path=":memory:",
        qdrant_url=None,
        collection_prefix="rm_demo",
        vector_size=embedder.dim,
    )
    llm = _make_llm()
    writer = MemoryWriter(store, llm, embedder, min_support=2, merge_cosine=0.99)
    updater = MemoryUpdater(
        store, llm, embedder,
        backend=args.surprise_backend,
        bayes_cfg=BayesianConfig(
            tau_low=args.bayes_tau_low,
            tau_high=args.bayes_tau_high,
            rewrite_threshold=3,
            rewrite_window=5,
        ),
    )
    agent = ReflectiveAgent(
        llm=llm, store=store, embedder=embedder,
        writer=writer, updater=updater,
        max_steps=args.max_steps,
        reflect_every_n_trajectories=args.reflect_every,
    )
    env = MockEnv(seed=args.seed, horizon=args.max_steps, goal_keyword="GOAL")

    # ----------------------------------------------------------------- #
    # Run                                                               #
    # ----------------------------------------------------------------- #
    trajectories = []
    for i in range(args.n_tasks):
        traj = agent.run_episode(env, task_idx=i)
        trajectories.append(traj)
        cnt = store.counts()
        logger.info(
            f"[task {i:>3}] success={traj.success} steps={traj.n_steps} "
            f"|M|: events={cnt['events']} eps={cnt['episodes']} "
            f"pat={cnt['patterns']} pri={cnt['principles']} upd={cnt['updates']}"
        )

    metrics = compute_metrics(
        trajectories, llm_usage=llm.usage, memory_counts=store.counts()
    )
    logger.info("=" * 70)
    logger.info(
        f"Summary: SR={metrics.success_rate:.3f} "
        f"({metrics.n_success}/{metrics.n_tasks})  "
        f"avg_steps={metrics.avg_steps:.2f}  "
        f"tokens={metrics.total_tokens}  "
        f"|M|={metrics.memory_size}"
    )
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
