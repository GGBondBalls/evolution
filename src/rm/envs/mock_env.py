"""A toy text environment for unit tests.

The agent must emit the action ``say {goal}`` (or any action containing the
``goal_keyword``) before ``horizon`` steps. Useful for testing the agent loop,
the memory plumbing, and the smoke scripts on machines without ALFWorld.
"""

from __future__ import annotations

import random

from rm.envs.base import EnvBase, EnvStep


class MockEnv(EnvBase):
    name = "mock"

    def __init__(self, seed: int = 42, horizon: int = 10, goal_keyword: str = "GOAL") -> None:
        self.seed = seed
        self.horizon = horizon
        self.goal_keyword = goal_keyword
        self._rng = random.Random(seed)
        self._step_idx = 0
        self._done = True
        self._task_text = (
            f"Task: emit an action that contains the keyword '{goal_keyword}' "
            f"within {horizon} steps."
        )

    def reset(self, task_idx: int | None = None) -> EnvStep:
        self._rng = random.Random((task_idx or 0) + self.seed)
        self._step_idx = 0
        self._done = False
        obs = (
            f"You are in a featureless room. {self._task_text} "
            "Available actions: ['look', 'say hello', 'say GOAL', 'wait']."
        )
        return EnvStep(
            observation=obs,
            reward=0.0,
            done=False,
            admissible_actions=["look", "say hello", "say GOAL", "wait"],
            info={"task_idx": task_idx or 0},
        )

    def step(self, action: str) -> EnvStep:
        if self._done:
            raise RuntimeError("Env is done; call reset() first.")
        self._step_idx += 1
        action_norm = action.strip().lower()
        success = self.goal_keyword.lower() in action_norm
        reached_horizon = self._step_idx >= self.horizon
        self._done = success or reached_horizon
        return EnvStep(
            observation=("OK." if not success else f"You said the magic word: {self.goal_keyword}"),
            reward=1.0 if success else 0.0,
            done=self._done,
            success=success,
            admissible_actions=["look", "say hello", "say GOAL", "wait"],
            info={"step": self._step_idx},
        )

    @property
    def task_description(self) -> str:
        return self._task_text

    @property
    def num_tasks(self) -> int:
        return 100
