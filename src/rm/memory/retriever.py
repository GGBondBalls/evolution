"""High-level retrieval: text query → MemoryContext → prompt-ready string.

The store already does the ANN heavy-lifting in ``MemoryStore.retrieve``. This
module adds:

* automatic query embedding (so callers pass a text query, not a vector);
* a ``format_for_prompt`` that converts the retrieved layers into a single
  string the agent will paste at the top of its user message;
* a soft token budget per layer (Principle / Pattern / Episode) — see
  ``configs/base.yaml`` ``rm.retrieval.token_budget_per_layer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rm.llm.client import LLMClient, MockLLMClient
from rm.llm.embed import Embedder
from rm.memory.schemas import (
    Episode,
    MemoryContext,
    Pattern,
    Principle,
    RetrievalQuery,
)
from rm.memory.store import MemoryStore
from rm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalConfig:
    k_principle: int = 3
    k_pattern: int = 5
    k_episode: int = 3
    min_principle_conf: float = 0.6
    min_pattern_conf: float = 0.5
    confidence_threshold_for_episode_drilldown: float = 0.5
    token_budget_per_layer: int = 1500          # rough char budget x 4


class MemoryRetriever:
    """Owns query-embedding + retrieval + text formatting."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        cfg: RetrievalConfig | None = None,
        token_counter: LLMClient | MockLLMClient | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg or RetrievalConfig()
        self.token_counter = token_counter

    # ------------------------------------------------------------------ #
    # Retrieval                                                          #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        scope_tags: list[str] | None = None,
    ) -> MemoryContext:
        if not query.strip():
            return MemoryContext()
        if self.embedder.dim != self.store.vector_size:
            logger.warning(
                f"Embedder dim {self.embedder.dim} != store dim {self.store.vector_size}; "
                f"retrieval will fail. Pass a matching embedder."
            )
            return MemoryContext()
        q_vec = self.embedder.encode_one(query)
        rq = RetrievalQuery(
            query_text=query,
            scope_tags=scope_tags or [],
            k_principle=self.cfg.k_principle,
            k_pattern=self.cfg.k_pattern,
            k_episode=self.cfg.k_episode,
            min_principle_conf=self.cfg.min_principle_conf,
            min_pattern_conf=self.cfg.min_pattern_conf,
        )
        return self.store.retrieve(
            q_vec, rq,
            confidence_threshold_for_episode_drilldown=self.cfg.confidence_threshold_for_episode_drilldown,
        )

    # ------------------------------------------------------------------ #
    # Formatting                                                         #
    # ------------------------------------------------------------------ #

    def format_for_prompt(self, ctx: MemoryContext) -> str:
        if ctx.is_empty():
            return ""
        parts: list[str] = ["[Memory Context]"]
        if ctx.principles:
            parts.append("Principles:")
            parts.extend(self._fmt_layer(ctx.principles, kind="principle"))
        if ctx.patterns:
            parts.append("Patterns:")
            parts.extend(self._fmt_layer(ctx.patterns, kind="pattern"))
        if ctx.episodes:
            parts.append("Past episodes:")
            parts.extend(self._fmt_layer(ctx.episodes, kind="episode"))
        return "\n".join(parts) + "\n\n"

    # ------------------------------------------------------------------ #

    def _fmt_layer(self, items: list[Any], *, kind: str) -> list[str]:
        budget = self.cfg.token_budget_per_layer * 4   # rough 4 chars / token
        used = 0
        lines: list[str] = []
        for it in items:
            line = self._fmt_one(it, kind=kind)
            if used + len(line) > budget and lines:
                lines.append("  … (truncated)")
                break
            lines.append(line)
            used += len(line) + 1
        return lines

    @staticmethod
    def _fmt_one(it: Any, *, kind: str) -> str:
        if kind == "principle":
            assert isinstance(it, Principle)
            return f"  • [{it.confidence:.2f}] {it.statement}"
        if kind == "pattern":
            assert isinstance(it, Pattern)
            return (
                f"  • [{it.confidence:.2f}] WHEN {it.condition} → "
                f"DO {it.action_template} (expected: {it.expected_effect})"
            )
        if kind == "episode":
            assert isinstance(it, Episode)
            return (
                f"  • [{it.outcome}] sub_goal={it.sub_goal!r} "
                f"summary={it.summary[:200]!r}"
            )
        raise ValueError(kind)


__all__ = ["MemoryRetriever", "RetrievalConfig"]
