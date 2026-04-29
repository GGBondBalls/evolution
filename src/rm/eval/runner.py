"""Multi-task multi-seed experiment driver.

Usage::

    from rm.eval.runner import Runner, RunnerConfig
    runner = Runner(RunnerConfig(n_tasks=20, n_seeds=3, n_trials_per_task=1,
                                  output_dir="runs/E1.01"))
    runner.run(env_factory=..., agent_factory=...)

The agent / env factories are callables taking a single integer ``seed``;
this lets us re-seed both deterministically and re-build the agent's
memory store between seeds.

Outputs (per ``output_dir``):
* ``trajectories.jsonl`` — one Trajectory per line (compact JSON).
* ``metrics.json``       — aggregated EvalMetrics from compute_metrics().
* ``config.json``        — verbatim copy of the RunnerConfig.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rm.agent.base import AgentBase, Trajectory
from rm.envs.base import EnvBase
from rm.eval.metrics import compute_metrics
from rm.utils.logging import get_logger
from rm.utils.seeding import set_seed

logger = get_logger(__name__)

EnvFactory = Callable[[int], EnvBase]
AgentFactory = Callable[[int], AgentBase]


@dataclass
class RunnerConfig:
    n_tasks: int = 5
    n_seeds: int = 1
    n_trials_per_task: int = 1
    output_dir: str = "runs/_tmp"
    log_to_wandb: bool = False
    wandb_project: str = "rm-eval"
    wandb_run_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Runner:
    def __init__(self, cfg: RunnerConfig) -> None:
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._wandb = self._init_wandb() if cfg.log_to_wandb else None

    # ------------------------------------------------------------------ #

    def run(
        self,
        env_factory: EnvFactory,
        agent_factory: AgentFactory,
        *,
        llm_usage: Any | None = None,
        memory_counts_provider: Callable[[], dict[str, int]] | None = None,
    ) -> dict[str, Any]:
        """Run the cross-product seeds × tasks × trials and write outputs."""
        cfg = self.cfg
        all_trajs: list[Trajectory] = []
        traj_path = self.out / "trajectories.jsonl"
        with traj_path.open("w", encoding="utf-8") as f_out:
            for seed_idx in range(cfg.n_seeds):
                seed = 42 + seed_idx
                set_seed(seed)
                env = env_factory(seed)
                agent = agent_factory(seed)
                logger.info(
                    f"runner: seed={seed} env={env.name} agent={agent.name} "
                    f"tasks={cfg.n_tasks} trials={cfg.n_trials_per_task}"
                )
                try:
                    for task_idx in range(cfg.n_tasks):
                        for trial in range(cfg.n_trials_per_task):
                            t0 = time.time()
                            traj = agent.run_episode(env, task_idx=task_idx)
                            traj.info["seed"] = seed
                            traj.info["task_idx"] = task_idx
                            traj.info["trial"] = trial
                            traj.info["wallclock_sec"] = round(time.time() - t0, 3)
                            all_trajs.append(traj)
                            f_out.write(json.dumps(_traj_to_dict(traj),
                                                   default=str, ensure_ascii=False))
                            f_out.write("\n")
                            f_out.flush()
                            logger.info(
                                f"  [seed={seed} task={task_idx} trial={trial}] "
                                f"success={traj.success} steps={traj.n_steps} "
                                f"reward={traj.total_reward:.2f}"
                            )
                finally:
                    env.close()

        memory_counts = memory_counts_provider() if memory_counts_provider else None
        metrics = compute_metrics(
            all_trajs, llm_usage=llm_usage, memory_counts=memory_counts
        )
        # Persist.
        (self.out / "metrics.json").write_text(
            json.dumps(metrics.to_dict(), indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.out / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            f"runner: SR={metrics.success_rate:.3f} ({metrics.n_success}/{metrics.n_tasks}) "
            f"avg_steps={metrics.avg_steps:.2f} tokens={metrics.total_tokens}"
        )
        if self._wandb is not None:
            self._wandb.log(metrics.to_dict())
            self._wandb.finish()
        return metrics.to_dict()

    # ------------------------------------------------------------------ #

    def _init_wandb(self) -> Any | None:
        try:
            import wandb  # type: ignore[import-not-found]

            return wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config=asdict(self.cfg),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"wandb disabled ({exc}); continuing without")
            return None


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _traj_to_dict(t: Trajectory) -> dict[str, Any]:
    return {
        "trajectory_id": t.trajectory_id,
        "task_description": t.task_description,
        "success": t.success,
        "total_reward": t.total_reward,
        "n_steps": t.n_steps,
        "elapsed_sec": t.elapsed_sec,
        "info": t.info,
        "steps": [
            {
                "i": s.step_idx,
                "thought": s.thought,
                "action": s.action,
                "obs": s.observation[:1500],
                "reward": s.reward,
                "done": s.done,
            }
            for s in t.steps
        ],
    }


__all__ = ["Runner", "RunnerConfig"]
