"""Aggregate metrics over a batch of trajectories.

Supports both the headline indicators in roadmap §4.6 and per-trial
breakdowns for the multi-trial Reflexion-style runs.

Bootstrap CI is computed via simple resampling — no scipy dependency.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from rm.agent.base import Trajectory


@dataclass
class EvalMetrics:
    n_tasks: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    success_rate_ci: tuple[float, float] = (0.0, 0.0)
    avg_steps: float = 0.0
    std_steps: float = 0.0
    avg_reward: float = 0.0
    avg_tokens: float = 0.0          # per task — only valid if usage attached
    total_tokens: int = 0
    per_trial_sr: list[float] = field(default_factory=list)
    memory_size: dict[str, int] = field(default_factory=dict)  # store.counts() snapshot

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    trajectories: list[Trajectory],
    *,
    llm_usage: Any | None = None,
    memory_counts: dict[str, int] | None = None,
    bootstrap: int = 1000,
    seed: int = 42,
) -> EvalMetrics:
    """Aggregate the standard metrics. Pass ``llm_usage=client.usage`` for tokens."""
    if not trajectories:
        return EvalMetrics()
    n = len(trajectories)
    successes = [int(t.success) for t in trajectories]
    steps = [t.n_steps for t in trajectories]
    rewards = [t.total_reward for t in trajectories]
    n_success = sum(successes)
    sr = n_success / n
    sr_lo, sr_hi = _bootstrap_ci(successes, n_iter=bootstrap, seed=seed)

    # Per-trial SR — only meaningful if Trajectory.info["trial"] is set
    per_trial = _per_trial_sr(trajectories)

    total_tokens = (
        int(getattr(llm_usage, "total", 0)) if llm_usage is not None else 0
    )
    avg_tokens = total_tokens / n if n else 0.0

    return EvalMetrics(
        n_tasks=n,
        n_success=n_success,
        success_rate=sr,
        success_rate_ci=(sr_lo, sr_hi),
        avg_steps=statistics.fmean(steps),
        std_steps=statistics.pstdev(steps) if len(steps) > 1 else 0.0,
        avg_reward=statistics.fmean(rewards),
        avg_tokens=avg_tokens,
        total_tokens=total_tokens,
        per_trial_sr=per_trial,
        memory_size=dict(memory_counts) if memory_counts else {},
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _bootstrap_ci(
    successes: list[int], *, n_iter: int = 1000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    if not successes:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(successes)
    means: list[float] = []
    for _ in range(n_iter):
        sample = [successes[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_iter * (alpha / 2))]
    hi = means[int(n_iter * (1 - alpha / 2)) - 1]
    return float(lo), float(hi)


def _per_trial_sr(trajectories: list[Trajectory]) -> list[float]:
    """Group trajectories by ``info['trial']`` and return SR per trial."""
    bins: dict[int, list[int]] = {}
    for t in trajectories:
        trial = int(t.info.get("trial", 0))
        bins.setdefault(trial, []).append(int(t.success))
    if len(bins) <= 1:
        return []
    return [sum(v) / len(v) for _, v in sorted(bins.items())]


__all__ = ["compute_metrics", "EvalMetrics"]
