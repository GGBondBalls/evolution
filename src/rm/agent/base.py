"""Common scaffolding for all agents.

The contract is intentionally tiny:

* ``Trajectory`` is the run record (steps + outcome metrics).
* ``AgentBase.run_episode(env, task_idx)`` is the only method an experiment
  driver needs.
* Concrete agents pick their own action by overriding ``act``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from rm.envs.base import EnvBase, EnvStep
from rm.memory.schemas import Event
from rm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StepRecord:
    step_idx: int
    state: str
    thought: str
    action: str
    observation: str
    reward: float
    done: bool


@dataclass
class Trajectory:
    trajectory_id: str
    task_description: str
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0
    n_steps: int = 0
    elapsed_sec: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def to_events(self) -> list[Event]:
        events: list[Event] = []
        for s in self.steps:
            events.append(
                Event(
                    trajectory_id=self.trajectory_id,
                    step_idx=s.step_idx,
                    state=s.state,
                    action=s.action,
                    observation=s.observation,
                    reward=s.reward,
                )
            )
        return events


# --------------------------------------------------------------------------- #
# Base                                                                        #
# --------------------------------------------------------------------------- #

class AgentBase:
    """Override ``act`` and ``on_episode_end`` (optional) in subclasses."""

    name: str = "base"

    def __init__(self, max_steps: int = 50) -> None:
        self.max_steps = max_steps

    # ------------------------------------------------------------------ #

    def act(self, step: EnvStep, traj: Trajectory) -> tuple[str, str]:
        """Return ``(thought, action)``."""
        raise NotImplementedError

    def on_episode_start(self, env: EnvBase, task_idx: int | None) -> None:
        pass

    def on_episode_end(self, traj: Trajectory) -> None:
        pass

    # ------------------------------------------------------------------ #

    def run_episode(self, env: EnvBase, task_idx: int | None = None) -> Trajectory:
        traj_id = str(uuid.uuid4())
        traj = Trajectory(trajectory_id=traj_id, task_description="")
        self.on_episode_start(env, task_idx)
        t0 = time.time()
        step = env.reset(task_idx=task_idx)
        traj.task_description = env.task_description or step.info.get("task", "")
        prev_obs = step.observation
        for i in range(self.max_steps):
            thought, action = self.act(step, traj)
            new_step = env.step(action)
            rec = StepRecord(
                step_idx=i,
                state=prev_obs,
                thought=thought,
                action=action,
                observation=new_step.observation,
                reward=new_step.reward,
                done=new_step.done,
            )
            traj.steps.append(rec)
            traj.total_reward += new_step.reward
            traj.n_steps += 1
            if new_step.done:
                traj.success = bool(new_step.success or new_step.reward > 0)
                step = new_step
                break
            prev_obs = new_step.observation
            step = new_step
        traj.elapsed_sec = time.time() - t0
        self.on_episode_end(traj)
        return traj


# --------------------------------------------------------------------------- #
# Trivial agent — picks first admissible action (smoke test)                  #
# --------------------------------------------------------------------------- #

class RandomAgent(AgentBase):
    name = "random"

    def __init__(self, max_steps: int = 50, seed: int = 42) -> None:
        super().__init__(max_steps=max_steps)
        import random

        self._rng = random.Random(seed)

    def act(self, step: EnvStep, traj: Trajectory) -> tuple[str, str]:
        actions = step.admissible_actions or ["wait"]
        action = self._rng.choice(actions)
        return ("(random policy)", action)


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def build_agent(cfg: dict[str, Any] | Any, *, llm=None, memory=None) -> AgentBase:
    cfg = dict(cfg) if not isinstance(cfg, dict) else cfg
    name = (cfg.get("name") or "react").lower()
    max_steps = int(cfg.get("max_steps", 50))
    if name in {"random", "rand"}:
        return RandomAgent(max_steps=max_steps, seed=int(cfg.get("seed", 42)))
    if name in {"react", "no_mem"}:
        from rm.agent.react import ReActAgent

        return ReActAgent(
            llm=llm,
            max_steps=max_steps,
            prompt_version=cfg.get("prompt_version", "v1"),
            think_temperature=float(cfg.get("think_temperature", 0.7)),
            act_temperature=float(cfg.get("act_temperature", 0.0)),
        )
    if name == "rm":
        # Reflective agent — to be implemented in W3+. For now alias ReAct.
        from rm.agent.react import ReActAgent

        logger.info("RM agent not yet implemented; falling back to ReActAgent.")
        return ReActAgent(
            llm=llm,
            max_steps=max_steps,
            prompt_version=cfg.get("prompt_version", "v1"),
            think_temperature=float(cfg.get("think_temperature", 0.7)),
            act_temperature=float(cfg.get("act_temperature", 0.0)),
            memory=memory,
        )
    raise ValueError(f"Unknown agent: {name}")
