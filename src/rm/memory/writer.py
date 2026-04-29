"""Bottom-up abstraction: Event → Episode → Pattern → Principle.

This module wires the four-layer write path described in §2.3 of the design doc:

    on_step(event)                       → store.write_event
    on_trajectory_end(traj)              → segment(events) → write_episode
                                          → cluster(recent_episodes)
                                          → induce_pattern(cluster) → upsert
    periodic_principle_reflection()      → analyse_patterns()
                                          → induce_principles()       → write

Each LLM-driven stage is wrapped in defensive parsing — a malformed reply
must NEVER bring down the agent loop. Failures degrade gracefully:

* ``segment`` — falls back to a single Episode covering the whole trajectory.
* ``induce_pattern`` — drops the cluster without writing.
* ``induce_principles`` — returns an empty list.

All three stages emit structured loguru events so the trace ends up in
``runs/<ts>/coding.log``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rm.llm.client import LLMClient, MockLLMClient
from rm.llm.embed import Embedder
from rm.llm.prompts import load_prompt
from rm.memory.schemas import Episode, Event, MemoryLayer, Pattern, Principle
from rm.memory.store import MemoryStore
from rm.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Episode segmentation (P1)                                                   #
# --------------------------------------------------------------------------- #

class EpisodeSegmenter:
    """Calls P1 on a trajectory's events to produce a list of Episodes."""

    def __init__(
        self,
        llm: LLMClient | MockLLMClient,
        embedder: Embedder,
        prompt_version: str = "v1",
        max_trace_chars: int = 6000,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.prompt = load_prompt("P1_segment", version=prompt_version)
        self.max_trace_chars = max_trace_chars

    def segment(self, trajectory_id: str, events: list[Event]) -> list[Episode]:
        if not events:
            return []
        trace = _format_trace(events, max_chars=self.max_trace_chars)
        msg = self.prompt.format(trace=trace)
        try:
            raw = self.llm.chat_json(
                [{"role": "user", "content": msg}], temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"segment: LLM failed ({exc}); using single-episode fallback")
            return [self._whole_traj_fallback(trajectory_id, events)]

        items = raw if isinstance(raw, list) else raw.get("episodes", [])
        if not items:
            logger.info("segment: LLM returned empty list; using fallback")
            return [self._whole_traj_fallback(trajectory_id, events)]

        episodes: list[Episode] = []
        max_step = events[-1].step_idx
        for item in items:
            try:
                ep = Episode(
                    trajectory_id=trajectory_id,
                    start_step=int(max(0, item["start_step"])),
                    end_step=int(min(max_step, item["end_step"])),
                    sub_goal=str(item.get("sub_goal", "")).strip()[:200],
                    summary=str(item.get("summary", "")).strip()[:1000],
                    outcome=_normalise_outcome(item.get("outcome", "partial")),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"segment: dropping bad episode item {item!r}: {exc}")
                continue
            ep.embedding = self.embedder.encode_one(f"{ep.sub_goal} | {ep.summary}")
            episodes.append(ep)
        if not episodes:
            return [self._whole_traj_fallback(trajectory_id, events)]
        return episodes

    def _whole_traj_fallback(self, traj_id: str, events: list[Event]) -> Episode:
        last = events[-1]
        outcome = "success" if (last.reward or 0.0) > 0.0 else "partial"
        summary = " | ".join(f"{e.action} → {e.observation[:40]}" for e in events[:6])
        sub_goal = events[0].state[:140] if events else "(unknown)"
        ep = Episode(
            trajectory_id=traj_id,
            start_step=events[0].step_idx,
            end_step=last.step_idx,
            sub_goal=sub_goal,
            summary=summary or "(no summary)",
            outcome=outcome,
        )
        ep.embedding = self.embedder.encode_one(f"{ep.sub_goal} | {ep.summary}")
        return ep


# --------------------------------------------------------------------------- #
# Episode clustering                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class _Cluster:
    label: int
    episodes: list[Episode]


class EpisodeClusterer:
    """Group episodes by embedding similarity. Default: KMeans + silhouette.

    HDBSCAN is preferred per the design doc but is an optional dep
    (``[cluster]`` extra) — fall back to KMeans automatically.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        max_clusters: int = 8,
        backend: str = "auto",        # auto | kmeans | hdbscan
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.backend = backend

    def cluster(self, episodes: list[Episode]) -> list[_Cluster]:
        eps = [e for e in episodes if e.embedding]
        if len(eps) < self.min_cluster_size:
            return []

        labels = self._fit_predict(eps)
        if labels is None:
            return []

        groups: dict[int, list[Episode]] = {}
        for ep, lab in zip(eps, labels, strict=True):
            if lab < 0:                         # HDBSCAN noise label
                continue
            groups.setdefault(int(lab), []).append(ep)
        return [
            _Cluster(label=lab, episodes=members)
            for lab, members in groups.items()
            if len(members) >= self.min_cluster_size
        ]

    # ------------------------------------------------------------------ #

    def _fit_predict(self, eps: list[Episode]) -> list[int] | None:
        backend = self._select_backend()
        vectors = [e.embedding for e in eps]
        if backend == "hdbscan":
            try:
                import hdbscan  # type: ignore[import-untyped]

                clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
                return list(clusterer.fit_predict(vectors))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"hdbscan failed ({exc}); falling back to kmeans")
                backend = "kmeans"
        if backend == "kmeans":
            return self._kmeans(vectors)
        return None

    def _select_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:
            import hdbscan  # type: ignore[import-untyped]  # noqa: F401

            return "hdbscan"
        except ImportError:
            return "kmeans"

    def _kmeans(self, vectors: list[list[float]]) -> list[int] | None:
        import warnings

        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.metrics import silhouette_score
        except ImportError:
            logger.warning("sklearn not available; skipping clustering")
            return None
        X = np.asarray(vectors, dtype="float32")
        n = X.shape[0]
        # Pick best k by silhouette in [2 .. min(max_clusters, n-1)].
        best_score, best_labels = -1.0, [0] * n
        for k in range(2, min(self.max_clusters, n - 1) + 1):
            try:
                with warnings.catch_warnings():
                    # Duplicate inputs trigger ConvergenceWarning; we handle it.
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
                if len(set(km.labels_)) < 2:
                    continue
                score = silhouette_score(X, km.labels_)
            except Exception:  # noqa: BLE001
                continue
            if score > best_score:
                best_score, best_labels = float(score), list(map(int, km.labels_))
        return best_labels


# --------------------------------------------------------------------------- #
# Pattern induction (P2)                                                      #
# --------------------------------------------------------------------------- #

class PatternInducer:
    """Calls P2 on a cluster of similar Episodes to produce a Pattern."""

    def __init__(
        self,
        llm: LLMClient | MockLLMClient,
        embedder: Embedder,
        prompt_version: str = "v1",
        merge_cosine: float = 0.90,
        max_episodes_in_prompt: int = 6,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.prompt = load_prompt("P2_pattern", version=prompt_version)
        self.merge_cosine = merge_cosine
        self.max_episodes_in_prompt = max_episodes_in_prompt

    def induce(self, episodes: list[Episode]) -> Pattern | None:
        if not episodes:
            return None
        body = self._format_episodes(episodes)
        msg = self.prompt.format(k=len(episodes), episodes=body)
        try:
            raw = self.llm.chat_json(
                [{"role": "user", "content": msg}], temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"induce_pattern: LLM failed ({exc}); skipping cluster")
            return None
        item = raw[0] if isinstance(raw, list) and raw else raw
        if not isinstance(item, dict):
            logger.warning(f"induce_pattern: unexpected JSON shape {type(item).__name__}")
            return None
        try:
            pattern = Pattern(
                condition=str(item.get("condition", "")).strip()[:300],
                action_template=str(item.get("action_template", "")).strip()[:200],
                expected_effect=str(item.get("expected_effect", "")).strip()[:300],
                scope=[str(s) for s in (item.get("scope") or [])][:5],
                support_episodes=[e.episode_id for e in episodes],
                evidence_count=len(episodes),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"induce_pattern: bad fields {item!r}: {exc}")
            return None
        if not (pattern.condition and pattern.action_template):
            return None
        pattern.embedding = self.embedder.encode_one(
            f"{pattern.condition} | {pattern.action_template} | {pattern.expected_effect}"
        )
        return pattern

    def find_near_duplicate(
        self, candidate: Pattern, store: MemoryStore
    ) -> Pattern | None:
        """Return an existing Pattern whose Qdrant cosine ≥ ``merge_cosine``."""
        if candidate.embedding is None:
            return None
        hits = store.query_vectors(MemoryLayer.PATTERN, candidate.embedding, top_k=1)
        if not hits:
            return None
        pid, score = hits[0]
        if score >= self.merge_cosine:
            return store.get_pattern(pid)
        return None

    def merge(self, existing: Pattern, candidate: Pattern, store: MemoryStore) -> Pattern:
        """Soft-merge: bump alpha (more support), append support_episodes, keep ID."""
        existing.alpha += float(candidate.evidence_count)
        existing.evidence_count += candidate.evidence_count
        seen = set(existing.support_episodes)
        existing.support_episodes.extend(
            eid for eid in candidate.support_episodes if eid not in seen
        )
        existing.last_updated = candidate.last_updated
        store.upsert_pattern(existing)
        return existing

    # ------------------------------------------------------------------ #

    def _format_episodes(self, eps: list[Episode]) -> str:
        eps = eps[: self.max_episodes_in_prompt]
        lines = []
        for i, e in enumerate(eps):
            lines.append(
                f"[{i}] sub_goal={e.sub_goal!r} outcome={e.outcome} "
                f"summary={e.summary[:200]!r}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Principle reflection (P3)                                                   #
# --------------------------------------------------------------------------- #

class PrincipleReflector:
    """Calls P3 on a batch of recent Patterns to induce Principles."""

    def __init__(
        self,
        llm: LLMClient | MockLLMClient,
        embedder: Embedder,
        prompt_version: str = "v1",
        max_patterns_in_prompt: int = 30,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.prompt = load_prompt("P3_principle", version=prompt_version)
        self.max_patterns_in_prompt = max_patterns_in_prompt

    def reflect(self, patterns: list[Pattern]) -> list[Principle]:
        if not patterns:
            return []
        body = self._format_patterns(patterns)
        msg = self.prompt.format(patterns=body)
        try:
            raw = self.llm.chat_json(
                [{"role": "user", "content": msg}], temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"reflect_principles: LLM failed ({exc})")
            return []
        items = raw if isinstance(raw, list) else raw.get("principles", [])
        principles: list[Principle] = []
        valid_pids = {p.pattern_id for p in patterns}
        for item in items:
            if not isinstance(item, dict):
                continue
            stmt = str(item.get("statement", "")).strip()
            if not stmt:
                continue
            supporting = [
                str(pid) for pid in (item.get("supporting_patterns") or [])
                if pid in valid_pids
            ][:5]
            pr = Principle(
                statement=stmt[:400],
                scope=str(item.get("scope", "cross-task")).strip()[:50],
                supporting_patterns=supporting,
            )
            pr.embedding = self.embedder.encode_one(pr.statement)
            principles.append(pr)
        return principles

    def _format_patterns(self, patterns: list[Pattern]) -> str:
        # Show highest-confidence first, then most-recent.
        sorted_p = sorted(
            patterns, key=lambda p: (p.confidence, p.last_updated), reverse=True
        )[: self.max_patterns_in_prompt]
        lines = []
        for p in sorted_p:
            lines.append(
                f"[{p.pattern_id[:8]}] conf={p.confidence:.2f} "
                f"cond={p.condition[:120]!r} action={p.action_template[:80]!r}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class WriterReport:
    """Summary of what the writer did. Used by tests / eval."""

    n_episodes: int = 0
    n_clusters: int = 0
    n_new_patterns: int = 0
    n_merged_patterns: int = 0
    n_new_principles: int = 0


class MemoryWriter:
    """Owns the full bottom-up pipeline."""

    def __init__(
        self,
        store: MemoryStore,
        llm: LLMClient | MockLLMClient,
        embedder: Embedder,
        *,
        min_support: int = 3,
        merge_cosine: float = 0.90,
        prompt_version: str = "v1",
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.segmenter = EpisodeSegmenter(llm, embedder, prompt_version=prompt_version)
        self.clusterer = EpisodeClusterer(min_cluster_size=min_support)
        self.inducer = PatternInducer(
            llm, embedder, prompt_version=prompt_version, merge_cosine=merge_cosine
        )
        self.reflector = PrincipleReflector(llm, embedder, prompt_version=prompt_version)

    # ------------------------------------------------------------------ #
    # Per-trajectory                                                      #
    # ------------------------------------------------------------------ #

    def on_trajectory_end(
        self, trajectory_id: str, events: Iterable[Event] | None = None
    ) -> WriterReport:
        report = WriterReport()
        events = list(events) if events is not None else self.store.get_events(trajectory_id)
        if not events:
            return report

        # 1. Segment.
        episodes = self.segmenter.segment(trajectory_id, events)
        for ep in episodes:
            self.store.write_episode(ep)
        report.n_episodes = len(episodes)
        logger.info(f"writer: trajectory_id={trajectory_id} → {len(episodes)} episodes")

        # 2. Cluster recent episodes (across trajectories) and try Pattern induction.
        recent = self.store.get_episodes()
        # SQLite doesn't store embeddings — hydrate them from Qdrant before clustering.
        if recent:
            ep_ids = [e.episode_id for e in recent]
            id2vec = self.store.fetch_vectors(MemoryLayer.EPISODE, ep_ids)
            for e in recent:
                e.embedding = id2vec.get(e.episode_id)
        clusters = self.clusterer.cluster(recent)
        report.n_clusters = len(clusters)
        for cl in clusters:
            new_p = self.inducer.induce(cl.episodes)
            if new_p is None:
                continue
            existing = self.inducer.find_near_duplicate(new_p, self.store)
            if existing is not None:
                self.inducer.merge(existing, new_p, self.store)
                report.n_merged_patterns += 1
            else:
                self.store.write_pattern(new_p)
                report.n_new_patterns += 1
        logger.info(
            f"writer: clusters={len(clusters)} new_patterns={report.n_new_patterns} "
            f"merged={report.n_merged_patterns}"
        )
        return report

    # ------------------------------------------------------------------ #
    # Periodic                                                            #
    # ------------------------------------------------------------------ #

    def reflect_principles(self, top_n_patterns: int = 200) -> WriterReport:
        report = WriterReport()
        patterns = self.store.all_patterns()
        if not patterns:
            return report
        # Most-recent first; cap to keep prompt budget.
        patterns = sorted(patterns, key=lambda p: p.last_updated, reverse=True)[
            :top_n_patterns
        ]
        new_principles = self.reflector.reflect(patterns)
        for pr in new_principles:
            self.store.write_principle(pr)
        report.n_new_principles = len(new_principles)
        logger.info(f"writer: reflect → {len(new_principles)} new principles")
        return report


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _format_trace(events: list[Event], *, max_chars: int) -> str:
    lines = []
    used = 0
    for e in events:
        line = (
            f"[step {e.step_idx}] state={e.state[:160]!r} "
            f"action={e.action[:120]!r} obs={e.observation[:160]!r} "
            f"reward={e.reward}"
        )
        used += len(line) + 1
        if used > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(line)
    return "\n".join(lines)


def _normalise_outcome(s: Any) -> str:
    s = str(s).strip().lower()
    if s in {"success", "succeeded", "ok", "win"}:
        return "success"
    if s in {"failure", "failed", "fail", "lost"}:
        return "failure"
    return "partial"


__all__ = [
    "EpisodeSegmenter",
    "EpisodeClusterer",
    "PatternInducer",
    "PrincipleReflector",
    "MemoryWriter",
    "WriterReport",
]


# Re-export for convenience (some tests grep for ``json`` usage).
_ = json
