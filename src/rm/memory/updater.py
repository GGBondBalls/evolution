"""Predictive-Surprise + Bayesian update + Pattern revision.

Implements §2.5 of the design doc::

    surprise(pattern, episode) ∈ [0, 1]
        ├─ < tau_low   →  α += 1   (strong support)
        ├─ > tau_high  →  β += 1   (strong refute)
        └─ in between  →  α += 1−s, β += s (soft)

    if recent strong refutes ≥ rewrite_thresh in last rewrite_window updates:
        revise(pattern, refute_episodes) → new pattern(s) (parent_id set)

Three Surprise backends — selected by config:

* ``llm_judge`` (default): P4 predicts → P5 scores 1-5 → normalise to [0, 1]
* ``embed_delta``: cosine_distance(embed(expected_effect), embed(actual))
* ``logprob``: not implemented in Round 2; the engine falls back to ``embed_delta``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from rm.llm.client import LLMClient, MockLLMClient
from rm.llm.embed import Embedder, cosine
from rm.llm.prompts import load_prompt
from rm.memory.schemas import Episode, Pattern, SurpriseSignal, UpdateRecord
from rm.memory.store import MemoryStore, _UpdateRow
from rm.utils.logging import get_logger

logger = get_logger(__name__)

SurpriseBackend = Literal["llm_judge", "embed_delta", "logprob"]


# --------------------------------------------------------------------------- #
# Predictive-Surprise                                                          #
# --------------------------------------------------------------------------- #

class SurpriseEngine:
    """Compares Pattern's prediction with what actually happened in the Episode."""

    def __init__(
        self,
        llm: LLMClient | MockLLMClient,
        embedder: Embedder,
        backend: SurpriseBackend = "llm_judge",
        prompt_version: str = "v1",
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.backend = backend
        self.predict_prompt = load_prompt("P4_predict", version=prompt_version)
        self.judge_prompt = load_prompt("P5_judge", version=prompt_version)

    # ------------------------------------------------------------------ #

    def compute(
        self, pattern: Pattern, episode: Episode, *, prefix: str | None = None
    ) -> SurpriseSignal:
        actual = self._actual_text(episode)
        prefix = prefix or episode.sub_goal or episode.summary[:200]
        try:
            if self.backend == "llm_judge":
                return self._llm_judge(pattern, episode, prefix=prefix, actual=actual)
            if self.backend == "embed_delta":
                return self._embed_delta(pattern, episode, prefix=prefix, actual=actual)
            if self.backend == "logprob":
                logger.info("logprob backend not implemented in Round 2; using embed_delta")
                return self._embed_delta(pattern, episode, prefix=prefix, actual=actual)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"surprise: {self.backend} failed ({exc}); embed_delta fallback")
            try:
                return self._embed_delta(pattern, episode, prefix=prefix, actual=actual)
            except Exception:
                pass
        # Last resort: maximally uninformative signal.
        return SurpriseSignal(
            pattern_id=pattern.pattern_id,
            episode_id=episode.episode_id,
            score=0.5,
            predicted="(unavailable)",
            actual=actual,
            backend=self.backend,
        )

    # ------------------------------------------------------------------ #
    # Backend: LLM-as-judge                                               #
    # ------------------------------------------------------------------ #

    def _llm_judge(
        self, pattern: Pattern, episode: Episode, *, prefix: str, actual: str
    ) -> SurpriseSignal:
        msg = self.predict_prompt.format(
            condition=pattern.condition,
            action_template=pattern.action_template,
            expected_effect=pattern.expected_effect,
            prefix=prefix,
        )
        predicted = self.llm.chat([{"role": "user", "content": msg}], temperature=0.0).text
        judge_msg = self.judge_prompt.format(predicted=predicted, actual=actual)
        score_text = self.llm.chat(
            [{"role": "user", "content": judge_msg}], temperature=0.0, max_tokens=4
        ).text
        score = _judge_text_to_score(score_text)
        return SurpriseSignal(
            pattern_id=pattern.pattern_id,
            episode_id=episode.episode_id,
            score=score,
            predicted=predicted[:500],
            actual=actual[:500],
            backend="llm_judge",
        )

    # ------------------------------------------------------------------ #
    # Backend: embedding distance                                         #
    # ------------------------------------------------------------------ #

    def _embed_delta(
        self, pattern: Pattern, episode: Episode, *, prefix: str, actual: str
    ) -> SurpriseSignal:
        predicted = pattern.expected_effect
        v_pred = self.embedder.encode_one(predicted)
        v_actual = self.embedder.encode_one(actual)
        sim = cosine(v_pred, v_actual)
        # Cosine ∈ [-1, 1] (typical for sentence embeddings ≥ 0). Map to surprise:
        # similarity 1 → surprise 0; similarity 0 → surprise ~0.5; -1 → 1.
        score = max(0.0, min(1.0, (1.0 - sim) / 2.0))
        return SurpriseSignal(
            pattern_id=pattern.pattern_id,
            episode_id=episode.episode_id,
            score=score,
            predicted=predicted[:500],
            actual=actual[:500],
            backend="embed_delta",
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _actual_text(episode: Episode) -> str:
        # ``summary`` is the LLM's own description of what happened; preferred.
        if episode.summary:
            return episode.summary
        return f"sub_goal={episode.sub_goal} outcome={episode.outcome}"


# --------------------------------------------------------------------------- #
# Bayesian update                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class BayesianConfig:
    tau_low: float = 0.20
    tau_high: float = 0.70
    alpha_max: float = 200.0          # cap to avoid runaway growth (§3.3 (A3))
    beta_max: float = 200.0
    rewrite_threshold: int = 5        # ≥ this many strong refutes in last window
    rewrite_window: int = 5


class BayesianUpdater:
    """Apply a SurpriseSignal to a Pattern: update α/β and persist."""

    def __init__(
        self,
        store: MemoryStore,
        cfg: BayesianConfig | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg or BayesianConfig()

    def update(self, pattern: Pattern, signal: SurpriseSignal) -> UpdateRecord:
        s = signal.score
        delta_alpha = 0.0
        delta_beta = 0.0
        if s < self.cfg.tau_low:
            delta_alpha = 1.0
            if signal.episode_id not in pattern.support_episodes:
                pattern.support_episodes.append(signal.episode_id)
        elif s > self.cfg.tau_high:
            delta_beta = 1.0
            if signal.episode_id not in pattern.refute_episodes:
                pattern.refute_episodes.append(signal.episode_id)
        else:
            delta_alpha = 1.0 - s
            delta_beta = s

        pattern.alpha = min(pattern.alpha + delta_alpha, self.cfg.alpha_max)
        pattern.beta = min(pattern.beta + delta_beta, self.cfg.beta_max)
        pattern.evidence_count += 1
        pattern.last_updated = signal.ts
        self.store.upsert_pattern(pattern)

        rec = UpdateRecord(
            pattern_id=pattern.pattern_id,
            episode_id=signal.episode_id,
            surprise=s,
            delta_alpha=delta_alpha,
            delta_beta=delta_beta,
            triggered_revision=False,
        )
        self.store.record_update(rec)
        return rec

    # ------------------------------------------------------------------ #
    # Decide whether a revision is warranted                              #
    # ------------------------------------------------------------------ #

    def needs_revision(self, pattern: Pattern) -> tuple[bool, list[str]]:
        """Inspect last ``rewrite_window`` updates for this Pattern."""
        with Session(self.store._engine) as s:
            rows = s.execute(
                select(_UpdateRow)
                .where(_UpdateRow.pattern_id == pattern.pattern_id)
                .order_by(_UpdateRow.ts.desc())
                .limit(self.cfg.rewrite_window)
            ).scalars().all()
        refute_eids = [
            r.episode_id for r in rows if r.surprise > self.cfg.tau_high
        ]
        return (len(refute_eids) >= self.cfg.rewrite_threshold, refute_eids)


# --------------------------------------------------------------------------- #
# Pattern revision (P6)                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class RevisionResult:
    decision: Literal["refine", "split", "discard"]
    rationale: str
    new_patterns: list[Pattern] = field(default_factory=list)


class PatternRevisor:
    """Calls P6 with refuting Episodes to produce a refined / split / discarded Pattern."""

    def __init__(
        self,
        llm: LLMClient | MockLLMClient,
        embedder: Embedder,
        prompt_version: str = "v1",
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.prompt = load_prompt("P6_revise", version=prompt_version)

    def revise(
        self,
        pattern: Pattern,
        refute_episodes: list[Episode],
    ) -> RevisionResult:
        body = "\n".join(
            f"[{i}] sub_goal={e.sub_goal!r} outcome={e.outcome} summary={e.summary[:200]!r}"
            for i, e in enumerate(refute_episodes)
        ) or "(none)"
        msg = self.prompt.format(
            pattern_id=pattern.pattern_id,
            condition=pattern.condition,
            action_template=pattern.action_template,
            expected_effect=pattern.expected_effect,
            confidence=pattern.confidence,
            evidence_count=pattern.evidence_count,
            refute_episodes=body,
        )
        try:
            raw = self.llm.chat_json([{"role": "user", "content": msg}], temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"revise: LLM failed ({exc}); decision=discard")
            return RevisionResult(decision="discard", rationale=str(exc))

        decision = str(raw.get("decision", "discard")).strip().lower()
        if decision not in {"refine", "split", "discard"}:
            decision = "discard"
        rationale = str(raw.get("rationale", ""))[:300]
        items = raw.get("patterns") or []
        new_patterns: list[Pattern] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cond = str(item.get("condition", "")).strip()
            act = str(item.get("action_template", "")).strip()
            if not (cond and act):
                continue
            np_ = Pattern(
                condition=cond[:300],
                action_template=act[:200],
                expected_effect=str(item.get("expected_effect", "")).strip()[:300],
                scope=[str(s) for s in (item.get("scope") or [])][:5],
                version=pattern.version + 1,
                parent_pattern_id=pattern.pattern_id,
            )
            np_.embedding = self.embedder.encode_one(
                f"{np_.condition} | {np_.action_template} | {np_.expected_effect}"
            )
            new_patterns.append(np_)
        return RevisionResult(decision=decision, rationale=rationale, new_patterns=new_patterns)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class UpdaterReport:
    n_signals: int = 0
    n_strong_support: int = 0
    n_strong_refute: int = 0
    n_revisions: int = 0
    n_new_patterns_from_revision: int = 0


class MemoryUpdater:
    """Glue: surprise → bayes update → conditional revision."""

    def __init__(
        self,
        store: MemoryStore,
        llm: LLMClient | MockLLMClient,
        embedder: Embedder,
        *,
        backend: SurpriseBackend = "llm_judge",
        bayes_cfg: BayesianConfig | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.engine = SurpriseEngine(llm, embedder, backend=backend, prompt_version=prompt_version)
        self.bayes = BayesianUpdater(store, cfg=bayes_cfg)
        self.revisor = PatternRevisor(llm, embedder, prompt_version=prompt_version)

    def process_episode(
        self, episode: Episode, retrieved_patterns: list[Pattern]
    ) -> UpdaterReport:
        """Apply surprise + bayesian update to every retrieved Pattern wrt this Episode."""
        report = UpdaterReport()
        for pat in retrieved_patterns:
            signal = self.engine.compute(pat, episode)
            rec = self.bayes.update(pat, signal)
            report.n_signals += 1
            if rec.delta_alpha >= 1.0 and rec.delta_beta == 0.0:
                report.n_strong_support += 1
            elif rec.delta_beta >= 1.0 and rec.delta_alpha == 0.0:
                report.n_strong_refute += 1
            # Trigger revision?
            needs, refute_eids = self.bayes.needs_revision(pat)
            if needs:
                refute_eps = [
                    e for e in self.store.get_episodes() if e.episode_id in set(refute_eids)
                ]
                rev = self.revisor.revise(pat, refute_eps)
                if rev.decision == "discard":
                    # Bury the old pattern by lifting beta to its cap.
                    pat.beta = min(self.bayes.cfg.beta_max, pat.beta * 2.0 + 1.0)
                    self.store.upsert_pattern(pat)
                    report.n_revisions += 1
                else:
                    for new_p in rev.new_patterns:
                        # New revisions inherit parent_pattern_id; α/β reset (= default 1,1)
                        self.store.write_pattern(new_p)
                        report.n_new_patterns_from_revision += 1
                    # Old pattern keeps living but with halved confidence weight.
                    pat.beta = min(self.bayes.cfg.beta_max, pat.beta + 1.0)
                    self.store.upsert_pattern(pat)
                    report.n_revisions += 1
                logger.info(
                    f"updater: revised pattern {pat.pattern_id[:8]} "
                    f"decision={rev.decision} new={len(rev.new_patterns)}"
                )
        return report


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

_DIGIT_RE = re.compile(r"[1-5]")


def _judge_text_to_score(text: str) -> float:
    """P5 returns 1-5; map to surprise score in [0, 1]."""
    text = text.strip()
    m = _DIGIT_RE.search(text)
    if not m:
        return 0.5  # uninformative
    digit = int(m.group(0))
    digit = max(1, min(5, digit))
    # 1 → 0.0 (no surprise) ... 5 → 1.0 (max surprise)
    return (digit - 1) / 4.0


__all__ = [
    "SurpriseEngine",
    "SurpriseBackend",
    "BayesianConfig",
    "BayesianUpdater",
    "PatternRevisor",
    "RevisionResult",
    "MemoryUpdater",
    "UpdaterReport",
]
