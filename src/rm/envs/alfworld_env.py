"""ALFWorld text-game wrapper.

We import ``alfworld`` lazily inside the constructor — the package depends on
``textworld`` whose binary wheels are Linux-only. On Windows hosts the import
will fail with a clear hint to use WSL.

This wrapper presents a single-task, single-batch interface compatible with
``EnvBase``. We follow the conventions used by ``noahshinn/reflexion`` so
metrics are directly comparable.
"""

from __future__ import annotations

import sys
from typing import Any

from rm.envs.base import EnvBase, EnvStep
from rm.utils.logging import get_logger

logger = get_logger(__name__)


class ALFWorldEnv(EnvBase):
    name = "alfworld"

    def __init__(self, cfg: dict[str, Any]) -> None:
        if sys.platform.startswith("win"):
            logger.warning(
                "ALFWorld depends on textworld (Linux-only). On Windows, "
                "consider running this code under WSL2 or a Linux container."
            )
        try:
            import alfworld.agents.environment as environment  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "alfworld is not installed. Install with `pip install -e \".[envs]\"` "
                "and run `alfworld-download` afterwards. On Windows, use WSL."
            ) from exc

        self.cfg = dict(cfg)
        self.split = self.cfg.get("split", "eval_out_of_distribution")
        self.max_steps = int(self.cfg.get("max_steps", 50))
        self.batch_size = 1   # we collapse to single-batch
        self._env_type = self.cfg.get("type", "AlfredTWEnv")

        # ALFWorld expects a Hydra-style config dict. We pass through the
        # user's cfg verbatim — see ``configs/env/alfworld.yaml``.
        env_cls = getattr(environment, self._env_type)
        raw_env = env_cls(self.cfg, train_eval=self.split)
        self._env = raw_env.init_env(batch_size=self.batch_size)

        self._step_idx = 0
        self._done = True
        self._task_text = ""
        self._admissible: list[str] = []

    # ------------------------------------------------------------------ #
    # EnvBase interface                                                   #
    # ------------------------------------------------------------------ #

    def reset(self, task_idx: int | None = None) -> EnvStep:
        # ALFWorld picks tasks itself; we ignore task_idx for the moment.
        obs, info = self._env.reset()
        self._step_idx = 0
        self._done = False
        self._admissible = list(info.get("admissible_commands", [[]])[0])
        # Task statement appears in the first observation; cache it.
        first_obs = obs[0] if isinstance(obs, list) else obs
        self._task_text = _extract_task(first_obs)
        return EnvStep(
            observation=first_obs,
            reward=0.0,
            done=False,
            admissible_actions=self._admissible,
            info={"won": False, "task": self._task_text, **info},
        )

    def step(self, action: str) -> EnvStep:
        if self._done:
            raise RuntimeError("Env is done; call reset() first.")
        self._step_idx += 1
        obs, score, done, info = self._env.step([action])
        new_obs = obs[0] if isinstance(obs, list) else obs
        rew = float(score[0] if isinstance(score, list) else score)
        is_done = bool(done[0] if isinstance(done, list) else done)
        won = bool(info.get("won", [False])[0]) if isinstance(info.get("won"), list) else bool(info.get("won", False))
        self._admissible = list(info.get("admissible_commands", [[]])[0])
        if self._step_idx >= self.max_steps:
            is_done = True
        self._done = is_done
        return EnvStep(
            observation=new_obs,
            reward=rew,
            done=is_done,
            success=won,
            admissible_actions=self._admissible,
            info={"step": self._step_idx, **info},
        )

    @property
    def task_description(self) -> str:
        return self._task_text

    @property
    def num_tasks(self) -> int:
        return getattr(self._env, "num_games", 134)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _extract_task(obs: str) -> str:
    """Pull the 'Your task is to ...' line from ALFWorld's intro observation."""
    for line in obs.splitlines():
        line = line.strip()
        if line.lower().startswith("your task"):
            return line
    return obs.split("\n", 1)[0]
