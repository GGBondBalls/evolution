"""Common interface for ALFWorld / ScienceWorld / WebShop / Mock environments.

Why we wrap them:

* Each upstream env has its own batch / Gym-ish API quirk. We collapse them
  into a single (single-batch) ``step`` / ``reset`` returning ``EnvStep``.
* The Agent code never touches the upstream packages — only ``EnvBase``.
* ``build_env`` does **lazy** imports so a fresh Windows install can run unit
  tests without ``alfworld`` / ``textworld``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnvStep:
    observation: str
    reward: float
    done: bool
    success: bool = False
    admissible_actions: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


class EnvBase:
    """Single-task, single-batch interface."""

    name: str = "base"

    def reset(self, task_idx: int | None = None) -> EnvStep:
        raise NotImplementedError

    def step(self, action: str) -> EnvStep:
        raise NotImplementedError

    @property
    def task_description(self) -> str:
        return ""

    @property
    def num_tasks(self) -> int:
        return 1

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def build_env(cfg: dict[str, Any] | Any) -> EnvBase:
    """Construct an env from its config dict.

    Lazy-imports the heavy upstream package only when the corresponding
    environment is requested.
    """
    cfg = dict(cfg) if not isinstance(cfg, dict) else cfg
    name = (cfg.get("name") or "mock").lower()
    if name == "mock":
        from rm.envs.mock_env import MockEnv

        return MockEnv(
            seed=int(cfg.get("seed", 42)),
            horizon=int(cfg.get("horizon", 10)),
            goal_keyword=str(cfg.get("goal_keyword", "GOAL")),
        )
    if name == "alfworld":
        from rm.envs.alfworld_env import ALFWorldEnv

        return ALFWorldEnv(cfg)
    if name == "scienceworld":
        from rm.envs.scienceworld_env import ScienceWorldEnv

        return ScienceWorldEnv(cfg)
    if name == "webshop":
        from rm.envs.webshop_env import WebShopEnv

        return WebShopEnv(cfg)
    raise ValueError(f"Unknown env name: {name}")
