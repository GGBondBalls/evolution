"""eval/metrics + eval/runner."""

from __future__ import annotations

import json
from pathlib import Path

from rm.agent.base import RandomAgent, Trajectory
from rm.envs.mock_env import MockEnv
from rm.eval.metrics import EvalMetrics, _bootstrap_ci, _per_trial_sr, compute_metrics
from rm.eval.runner import Runner, RunnerConfig

# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

def test_compute_metrics_empty_returns_zeros():
    m = compute_metrics([])
    assert isinstance(m, EvalMetrics)
    assert m.n_tasks == 0 and m.success_rate == 0.0


def test_compute_metrics_basic():
    trajs = [
        Trajectory(trajectory_id=f"t{i}", task_description="x",
                   success=(i % 2 == 0), n_steps=i + 1, total_reward=float(i))
        for i in range(4)
    ]
    m = compute_metrics(trajs, bootstrap=200)
    assert m.n_tasks == 4 and m.n_success == 2
    assert m.success_rate == 0.5
    assert m.avg_steps == (1 + 2 + 3 + 4) / 4
    # CI must contain the point estimate.
    lo, hi = m.success_rate_ci
    assert lo <= 0.5 <= hi


def test_per_trial_sr_uses_info_trial():
    trajs = []
    for trial in range(3):
        trajs.extend([
            Trajectory(trajectory_id=f"t{trial}_{i}", task_description="",
                       success=(i + trial) % 2 == 0, n_steps=1, total_reward=0,
                       info={"trial": trial})
            for i in range(4)
        ])
    out = _per_trial_sr(trajs)
    assert len(out) == 3
    assert all(0.0 <= s <= 1.0 for s in out)


def test_bootstrap_ci_single_value():
    lo, hi = _bootstrap_ci([1, 1, 1], n_iter=100)
    assert lo == 1.0 and hi == 1.0


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def test_runner_writes_jsonl_and_metrics(tmp_path: Path):
    cfg = RunnerConfig(
        n_tasks=3, n_seeds=1, n_trials_per_task=1,
        output_dir=str(tmp_path / "run1"), log_to_wandb=False,
    )
    runner = Runner(cfg)

    def env_factory(seed: int) -> MockEnv:
        return MockEnv(seed=seed, horizon=5, goal_keyword="GOAL")

    def agent_factory(seed: int) -> RandomAgent:
        return RandomAgent(max_steps=5, seed=seed)

    metrics = runner.run(env_factory, agent_factory)
    assert metrics["n_tasks"] == 3

    traj_file = Path(cfg.output_dir) / "trajectories.jsonl"
    metrics_file = Path(cfg.output_dir) / "metrics.json"
    assert traj_file.exists() and metrics_file.exists()
    lines = traj_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = json.loads(lines[0])
    assert "trajectory_id" in parsed and "steps" in parsed


def test_runner_multi_seed_multi_trial(tmp_path: Path):
    cfg = RunnerConfig(
        n_tasks=2, n_seeds=2, n_trials_per_task=2,
        output_dir=str(tmp_path / "run2"), log_to_wandb=False,
    )
    runner = Runner(cfg)
    metrics = runner.run(
        env_factory=lambda s: MockEnv(seed=s, horizon=4, goal_keyword="GOAL"),
        agent_factory=lambda s: RandomAgent(max_steps=4, seed=s),
    )
    assert metrics["n_tasks"] == 2 * 2 * 2
    # per_trial_sr should have 2 entries (one per trial level).
    assert len(metrics["per_trial_sr"]) == 2
