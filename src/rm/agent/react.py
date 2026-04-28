"""ReAct agent — Thought→Action loop, vanilla.

This is the lower-bound baseline. The Reflective agent (W3+) will subclass
this and inject memory retrieval into the prompt.
"""

from __future__ import annotations

import re
from typing import Any

from rm.agent.base import AgentBase, Trajectory
from rm.envs.base import EnvStep
from rm.llm.client import ChatMessage, LLMClient, MockLLMClient
from rm.llm.prompts import load_prompt
from rm.utils.logging import get_logger

logger = get_logger(__name__)

_THOUGHT_RE = re.compile(r"thought\s*:\s*(.+?)(?=\n\s*action\s*:|$)", re.I | re.S)
_ACTION_RE = re.compile(r"action\s*:\s*(.+?)\s*$", re.I | re.S)


class ReActAgent(AgentBase):
    """Vanilla ReAct. ``llm=None`` falls back to a MockLLMClient (for smoke tests)."""

    name = "react"

    def __init__(
        self,
        llm: LLMClient | MockLLMClient | None = None,
        max_steps: int = 50,
        prompt_version: str = "v1",
        think_temperature: float = 0.7,
        act_temperature: float = 0.0,
        memory: Any = None,                # MemoryStore — wired by RM agent later
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.llm = llm or MockLLMClient(default="Thought: explore.\nAction: look")
        self.system_template = load_prompt("react_system", version=prompt_version)
        self.step_template = load_prompt("react_step", version=prompt_version)
        self.think_temperature = think_temperature
        self.act_temperature = act_temperature
        self.memory = memory

    # ------------------------------------------------------------------ #

    def act(self, step: EnvStep, traj: Trajectory) -> tuple[str, str]:
        system = self.system_template.format(task_description=traj.task_description)
        user = self.step_template.format(
            memory_block=self._memory_block(traj),
            trajectory=self._fmt_trajectory(traj),
            observation=step.observation[:1500],     # token guard
            admissible=", ".join(step.admissible_actions or []) or "(no admissible list)",
        )
        msgs = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]
        result = self.llm.chat(msgs, temperature=self.think_temperature)
        thought, action = _parse_thought_action(result.text, fallback=step.admissible_actions)
        return thought, action

    # ------------------------------------------------------------------ #

    def _memory_block(self, traj: Trajectory) -> str:
        # Plain ReAct has no memory; the RM agent overrides this.
        return ""

    def _fmt_trajectory(self, traj: Trajectory) -> str:
        if not traj.steps:
            return "(empty)"
        lines = []
        for s in traj.steps[-10:]:    # keep last 10 for prompt budget
            lines.append(
                f"[step {s.step_idx}] thought={s.thought[:80]} | action={s.action[:80]} "
                f"| obs={s.observation[:120]}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #

def _parse_thought_action(text: str, fallback: list[str] | None = None) -> tuple[str, str]:
    """Extract ``Thought:`` / ``Action:`` from the LLM output. Robust to extras."""
    text = text.strip()
    thought_m = _THOUGHT_RE.search(text)
    action_m = _ACTION_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    action = action_m.group(1).strip() if action_m else ""
    if not action:
        # Last-resort: take the first non-empty line as the action.
        for line in text.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("thought"):
                action = line
                break
    if not action and fallback:
        action = fallback[0]
    if not thought:
        thought = "(no thought parsed)"
    # Strip outer quotes if any.
    action = action.strip("`'\" ")
    return thought, action
