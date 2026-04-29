"""RM-aware Agent.

This is the Round-2 deliverable that wires the four-layer memory into the
ReAct loop:

* On every step, retrieve relevant Principles + Patterns + Episodes by query
  embedding, format them as a memory block, and inject the block into the user
  message of the ReAct prompt.
* At the end of every trajectory, write the events, segment them into Episodes
  via the LLM, cluster recent Episodes, induce new Patterns, and (optionally)
  reflect on Patterns to produce Principles.
* Per-Episode, optionally run the Predictive-Surprise + Bayesian update against
  the Patterns retrieved at that step.

Anything that depends on the memory subsystem is **optional**: pass
``memory_retriever=None`` to get a plain ReAct agent that uses the RM system
prompt but no memory I/O. This is useful for ablations.
"""

from __future__ import annotations

from rm.agent.base import Trajectory
from rm.agent.react import ReActAgent
from rm.envs.base import EnvStep
from rm.llm.client import LLMClient, MockLLMClient
from rm.llm.embed import Embedder
from rm.llm.prompts import load_prompt
from rm.memory.retriever import MemoryRetriever
from rm.memory.schemas import Pattern
from rm.memory.store import MemoryStore
from rm.memory.updater import MemoryUpdater
from rm.memory.writer import MemoryWriter, WriterReport
from rm.utils.logging import get_logger

logger = get_logger(__name__)


class ReflectiveAgent(ReActAgent):
    """ReAct + Reflective Memory."""

    name = "rm"

    def __init__(
        self,
        llm: LLMClient | MockLLMClient,
        store: MemoryStore,
        embedder: Embedder,
        *,
        retriever: MemoryRetriever | None = None,
        writer: MemoryWriter | None = None,
        updater: MemoryUpdater | None = None,
        reflect_every_n_trajectories: int = 5,
        max_steps: int = 50,
        prompt_version: str = "v1",
        think_temperature: float = 0.7,
        act_temperature: float = 0.0,
    ) -> None:
        super().__init__(
            llm=llm,
            max_steps=max_steps,
            prompt_version=prompt_version,
            think_temperature=think_temperature,
            act_temperature=act_temperature,
            memory=store,
        )
        # Override the system prompt to the RM-aware one.
        self.system_template = load_prompt("rm_system", version=prompt_version)
        self.store = store
        self.embedder = embedder
        self.retriever = retriever or MemoryRetriever(store=store, embedder=embedder)
        self.writer = writer
        self.updater = updater
        self.reflect_every_n_trajectories = reflect_every_n_trajectories
        # Per-episode caches.
        self._task_cache: str = ""
        self._patterns_used_this_step: list[Pattern] = []
        self._n_trajectories_seen: int = 0

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def on_episode_start(self, env, task_idx) -> None:  # type: ignore[override]
        self._task_cache = env.task_description or ""
        self._patterns_used_this_step = []

    def on_episode_end(self, traj: Trajectory) -> None:  # type: ignore[override]
        events = traj.to_events()
        if not events:
            return
        # Persist raw events.
        self.store.write_events(events)

        if self.writer is not None:
            try:
                report: WriterReport = self.writer.on_trajectory_end(
                    trajectory_id=traj.trajectory_id, events=events
                )
                logger.info(
                    f"reflective: traj={traj.trajectory_id} "
                    f"episodes={report.n_episodes} new_patterns={report.n_new_patterns} "
                    f"merged={report.n_merged_patterns}"
                )
                # Optionally run updater on new Episodes.
                if self.updater is not None:
                    self._run_updater_on_traj(traj.trajectory_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"reflective.writer failed: {exc}")

        # Periodic Principle reflection.
        self._n_trajectories_seen += 1
        if (
            self.writer is not None
            and self._n_trajectories_seen % self.reflect_every_n_trajectories == 0
        ):
            try:
                self.writer.reflect_principles()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"reflective.reflect_principles failed: {exc}")

    # ------------------------------------------------------------------ #
    # Memory injection                                                   #
    # ------------------------------------------------------------------ #

    def _memory_block(self, traj: Trajectory) -> str:
        if self.retriever is None:
            return ""
        last_obs = traj.steps[-1].observation if traj.steps else ""
        query = f"{self._task_cache} | {last_obs}".strip(" |")[:400]
        if not query:
            return ""
        try:
            ctx = self.retriever.retrieve(query=query)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"retriever failed: {exc}")
            return ""
        # Stash retrieved Patterns for the updater to use after segmentation.
        self._patterns_used_this_step = list(ctx.patterns)
        return self.retriever.format_for_prompt(ctx)

    # ------------------------------------------------------------------ #
    # Updater plumbing                                                   #
    # ------------------------------------------------------------------ #

    def _run_updater_on_traj(self, trajectory_id: str) -> None:
        """Compute surprise + Bayesian update for each Episode in this trajectory.

        Strategy: for each Episode, we re-retrieve top-K relevant Patterns
        based on the Episode's sub_goal (so the update is anchored to the
        Pattern population the Episode would have matched at retrieval time).
        """
        if self.updater is None:
            return
        episodes = self.store.get_episodes(trajectory_id)
        for ep in episodes:
            query = f"{ep.sub_goal} | {ep.summary}"[:400]
            try:
                ctx = self.retriever.retrieve(query=query)
            except Exception:  # noqa: BLE001
                continue
            if not ctx.patterns:
                continue
            self.updater.process_episode(ep, retrieved_patterns=ctx.patterns)

    # ------------------------------------------------------------------ #
    # Override act so we surface the memory block via the parent helper. #
    # ------------------------------------------------------------------ #

    def act(self, step: EnvStep, traj: Trajectory) -> tuple[str, str]:  # noqa: D401
        # The parent class already calls ``self._memory_block(traj)`` inside
        # its prompt construction, so no further override is needed.
        return super().act(step, traj)


__all__ = ["ReflectiveAgent"]
