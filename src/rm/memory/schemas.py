"""Pydantic schemas for the four memory layers.

These mirror §2.2 / Appendix C of ``docs/RM_design_and_roadmap.md``. Keep this
module **dependency-free** apart from pydantic so that it can be imported
anywhere (tests, store, writer, retriever) without dragging in heavy deps.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _uuid() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryLayer(str, Enum):
    """Hierarchical layer label — used by the retriever and the eval code."""

    EVENT = "event"
    EPISODE = "episode"
    PATTERN = "pattern"
    PRINCIPLE = "principle"


class _ORMBase(BaseModel):
    """Common pydantic config — frozen=False so we can mutate (alpha/beta) in place."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ser_json_timedelta="iso8601",
        validate_assignment=True,
        extra="ignore",
    )


# --------------------------------------------------------------------------- #
# Layer 1 — Event                                                             #
# --------------------------------------------------------------------------- #

class Event(_ORMBase):
    """Raw (state, action, observation) triple — one per environment step."""

    event_id: str = Field(default_factory=_uuid)
    trajectory_id: str
    step_idx: int = Field(ge=0)
    state: str
    action: str
    observation: str
    reward: float | None = None
    ts: datetime = Field(default_factory=_now)
    embedding: list[float] | None = None

    @field_validator("state", "action", "observation")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


# --------------------------------------------------------------------------- #
# Layer 2 — Episode                                                           #
# --------------------------------------------------------------------------- #

class Episode(_ORMBase):
    """A semantically coherent slice of a trajectory (the unit of Pattern induction)."""

    episode_id: str = Field(default_factory=_uuid)
    trajectory_id: str
    start_step: int = Field(ge=0)
    end_step: int = Field(ge=0)
    sub_goal: str
    summary: str
    outcome: Literal["success", "partial", "failure"]
    key_steps: list[str] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_now)
    embedding: list[float] | None = None

    @field_validator("end_step")
    @classmethod
    def _end_after_start(cls, v: int, info) -> int:
        start = info.data.get("start_step")
        if start is not None and v < start:
            raise ValueError(f"end_step ({v}) must be >= start_step ({start})")
        return v


# --------------------------------------------------------------------------- #
# Layer 3 — Pattern (★ core)                                                  #
# --------------------------------------------------------------------------- #

class Pattern(_ORMBase):
    """A reusable behaviour pattern with Bayesian Beta(alpha, beta) confidence."""

    pattern_id: str = Field(default_factory=_uuid)
    condition: str
    action_template: str
    expected_effect: str
    scope: list[str] = Field(default_factory=list)
    support_episodes: list[str] = Field(default_factory=list)
    refute_episodes: list[str] = Field(default_factory=list)
    alpha: float = Field(default=1.0, gt=0.0)
    beta: float = Field(default=1.0, gt=0.0)
    evidence_count: int = Field(default=0, ge=0)
    last_updated: datetime = Field(default_factory=_now)
    last_used: datetime = Field(default_factory=_now)
    use_count: int = Field(default=0, ge=0)
    version: int = Field(default=1, ge=1)
    parent_pattern_id: str | None = None  # set when this Pattern is a revision of another
    embedding: list[float] | None = None

    # ------------------------------------------------------------------ #
    # Derived quantities                                                  #
    # ------------------------------------------------------------------ #

    @computed_field  # type: ignore[misc]
    @property
    def confidence(self) -> float:
        """Posterior mean of Beta(alpha, beta) — the believed success rate."""
        return self.alpha / (self.alpha + self.beta)

    @computed_field  # type: ignore[misc]
    @property
    def uncertainty(self) -> float:
        """Posterior variance of Beta(alpha, beta) — used for retrieval gating."""
        a, b = self.alpha, self.beta
        return (a * b) / (((a + b) ** 2) * (a + b + 1))


# --------------------------------------------------------------------------- #
# Layer 4 — Principle                                                         #
# --------------------------------------------------------------------------- #

class Principle(_ORMBase):
    """Cross-task cognitive bias prepended to the system prompt."""

    principle_id: str = Field(default_factory=_uuid)
    statement: str
    scope: str = "cross-task"
    supporting_patterns: list[str] = Field(default_factory=list)
    contradiction_log: list[dict[str, Any]] = Field(default_factory=list)
    alpha: float = Field(default=1.0, gt=0.0)
    beta: float = Field(default=1.0, gt=0.0)
    last_updated: datetime = Field(default_factory=_now)
    last_used: datetime = Field(default_factory=_now)
    use_count: int = Field(default=0, ge=0)
    embedding: list[float] | None = None

    @computed_field  # type: ignore[misc]
    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)


# --------------------------------------------------------------------------- #
# Retrieval / update auxiliaries                                              #
# --------------------------------------------------------------------------- #

class RetrievalQuery(_ORMBase):
    """Input to ``MemoryStore.retrieve``."""

    query_text: str
    scope_tags: list[str] = Field(default_factory=list)
    k_principle: int = 3
    k_pattern: int = 5
    k_episode: int = 3
    min_principle_conf: float = 0.6
    min_pattern_conf: float = 0.5


class MemoryContext(_ORMBase):
    """Output of retrieval — what the agent will see in its prompt."""

    principles: list[Principle] = Field(default_factory=list)
    patterns: list[Pattern] = Field(default_factory=list)
    episodes: list[Episode] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.principles or self.patterns or self.episodes)

    def total_items(self) -> int:
        return len(self.principles) + len(self.patterns) + len(self.episodes)


class SurpriseSignal(_ORMBase):
    """Result of comparing a Pattern's prediction with the actual outcome (§2.5.1)."""

    pattern_id: str
    episode_id: str
    score: float = Field(ge=0.0, le=1.0)        # 0 = perfectly predicted, 1 = totally refuted
    predicted: str
    actual: str
    backend: Literal["llm_judge", "embed_delta", "logprob"]
    ts: datetime = Field(default_factory=_now)


class UpdateRecord(_ORMBase):
    """Audit-trail entry for every Bayesian update (used by §4.5 metrics: RL/BRA/SMP)."""

    update_id: str = Field(default_factory=_uuid)
    pattern_id: str
    episode_id: str
    surprise: float = Field(ge=0.0, le=1.0)
    delta_alpha: float
    delta_beta: float
    triggered_revision: bool = False
    new_pattern_id: str | None = None
    ts: datetime = Field(default_factory=_now)
