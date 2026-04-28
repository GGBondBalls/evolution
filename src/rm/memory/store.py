"""Memory storage backend.

Two physical stores, one logical store:

* **SQLite** (via SQLAlchemy 2.0) — owns all relational data, including the
  text fields of Events / Episodes / Patterns / Principles. JSON-encoded
  columns hold list/dict fields. We deliberately do NOT store embeddings in
  SQLite — they live in Qdrant.
* **Qdrant** — owns the embedding vectors and serves the ANN queries.
  ``url=None`` falls back to the in-memory client (``:memory:``), which is
  handy for tests and Windows users without Docker.

The two are kept consistent by ``MemoryStore``: every write goes through both,
every retrieval does ANN in Qdrant, then hydrates the records from SQLite.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from rm.memory.schemas import (
    Episode,
    Event,
    MemoryContext,
    MemoryLayer,
    Pattern,
    Principle,
    RetrievalQuery,
    UpdateRecord,
)
from rm.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# SQLAlchemy ORM                                                              #
# --------------------------------------------------------------------------- #

class _Base(DeclarativeBase):
    pass


class _EventRow(_Base):
    __tablename__ = "events"
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    trajectory_id: Mapped[str] = mapped_column(String, index=True)
    step_idx: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    observation: Mapped[str] = mapped_column(Text)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime)


class _EpisodeRow(_Base):
    __tablename__ = "episodes"
    episode_id: Mapped[str] = mapped_column(String, primary_key=True)
    trajectory_id: Mapped[str] = mapped_column(String, index=True)
    start_step: Mapped[int] = mapped_column(Integer)
    end_step: Mapped[int] = mapped_column(Integer)
    sub_goal: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(String)
    key_steps_json: Mapped[str] = mapped_column(Text, default="[]")
    ts: Mapped[datetime] = mapped_column(DateTime)


class _PatternRow(_Base):
    __tablename__ = "patterns"
    pattern_id: Mapped[str] = mapped_column(String, primary_key=True)
    condition: Mapped[str] = mapped_column(Text)
    action_template: Mapped[str] = mapped_column(Text)
    expected_effect: Mapped[str] = mapped_column(Text)
    scope_json: Mapped[str] = mapped_column(Text, default="[]")
    support_episodes_json: Mapped[str] = mapped_column(Text, default="[]")
    refute_episodes_json: Mapped[str] = mapped_column(Text, default="[]")
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime)
    last_used: Mapped[datetime] = mapped_column(DateTime)
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_pattern_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("patterns.pattern_id"), nullable=True
    )


class _PrincipleRow(_Base):
    __tablename__ = "principles"
    principle_id: Mapped[str] = mapped_column(String, primary_key=True)
    statement: Mapped[str] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(String, default="cross-task")
    supporting_patterns_json: Mapped[str] = mapped_column(Text, default="[]")
    contradiction_log_json: Mapped[str] = mapped_column(Text, default="[]")
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)
    last_updated: Mapped[datetime] = mapped_column(DateTime)
    last_used: Mapped[datetime] = mapped_column(DateTime)
    use_count: Mapped[int] = mapped_column(Integer, default=0)


class _UpdateRow(_Base):
    __tablename__ = "update_records"
    update_id: Mapped[str] = mapped_column(String, primary_key=True)
    pattern_id: Mapped[str] = mapped_column(String, index=True)
    episode_id: Mapped[str] = mapped_column(String)
    surprise: Mapped[float] = mapped_column(Float)
    delta_alpha: Mapped[float] = mapped_column(Float)
    delta_beta: Mapped[float] = mapped_column(Float)
    triggered_revision: Mapped[bool] = mapped_column(Integer, default=0)
    new_pattern_id: Mapped[str | None] = mapped_column(String, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime)


# --------------------------------------------------------------------------- #
# JSON helpers                                                                #
# --------------------------------------------------------------------------- #

def _dumps(v: Any) -> str:
    return json.dumps(v, default=str, ensure_ascii=False)


def _loads(s: str | None, default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return default


# --------------------------------------------------------------------------- #
# Memory store                                                                #
# --------------------------------------------------------------------------- #

class MemoryStore:
    """Single facade for both relational and vector storage."""

    LAYER_COLLECTIONS = {
        MemoryLayer.EVENT: "events",
        MemoryLayer.EPISODE: "episodes",
        MemoryLayer.PATTERN: "patterns",
        MemoryLayer.PRINCIPLE: "principles",
    }

    def __init__(
        self,
        sqlite_path: str | Path | None = None,
        qdrant_url: str | None = None,
        collection_prefix: str = "rm",
        vector_size: int = 1024,
        distance: str = "Cosine",
    ) -> None:
        self.sqlite_path = sqlite_path
        self.qdrant_url = qdrant_url
        self.collection_prefix = collection_prefix
        self.vector_size = vector_size
        self.distance = distance
        self._lock = threading.RLock()

        # SQLAlchemy engine.
        if sqlite_path is None or str(sqlite_path) == ":memory:":
            url = "sqlite:///:memory:"
        else:
            sqlite_path = Path(sqlite_path)
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{sqlite_path.as_posix()}"
        self._engine = create_engine(url, future=True, echo=False)
        _Base.metadata.create_all(self._engine)

        # Qdrant client — in-memory if no URL given.
        self._qdrant = self._build_qdrant()
        self._ensure_collections()

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def _build_qdrant(self):
        from qdrant_client import QdrantClient

        if not self.qdrant_url:
            logger.info("Qdrant: using in-memory client (no docker)")
            return QdrantClient(location=":memory:")
        logger.info(f"Qdrant: connecting to {self.qdrant_url}")
        return QdrantClient(url=self.qdrant_url)

    def _ensure_collections(self) -> None:
        from qdrant_client.http import models as qm

        for _layer, suffix in self.LAYER_COLLECTIONS.items():
            name = f"{self.collection_prefix}_{suffix}"
            collections = {c.name for c in self._qdrant.get_collections().collections}
            if name in collections:
                continue
            self._qdrant.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(
                    size=self.vector_size,
                    distance=getattr(qm.Distance, self.distance.upper(), qm.Distance.COSINE),
                ),
            )

    def _coll(self, layer: MemoryLayer) -> str:
        return f"{self.collection_prefix}_{self.LAYER_COLLECTIONS[layer]}"

    # ------------------------------------------------------------------ #
    # Writes                                                              #
    # ------------------------------------------------------------------ #

    def write_event(self, ev: Event) -> None:
        with self._lock, Session(self._engine) as s:
            s.merge(_EventRow(
                event_id=ev.event_id,
                trajectory_id=ev.trajectory_id,
                step_idx=ev.step_idx,
                state=ev.state,
                action=ev.action,
                observation=ev.observation,
                reward=ev.reward,
                ts=ev.ts,
            ))
            s.commit()
        if ev.embedding:
            self._upsert_vector(MemoryLayer.EVENT, ev.event_id, ev.embedding,
                                payload={"trajectory_id": ev.trajectory_id, "step_idx": ev.step_idx})

    def write_events(self, events: Iterable[Event]) -> None:
        events = list(events)
        if not events:
            return
        with self._lock, Session(self._engine) as s:
            for ev in events:
                s.merge(_EventRow(
                    event_id=ev.event_id,
                    trajectory_id=ev.trajectory_id,
                    step_idx=ev.step_idx,
                    state=ev.state,
                    action=ev.action,
                    observation=ev.observation,
                    reward=ev.reward,
                    ts=ev.ts,
                ))
            s.commit()
        embedded = [(e.event_id, e.embedding,
                     {"trajectory_id": e.trajectory_id, "step_idx": e.step_idx})
                    for e in events if e.embedding]
        if embedded:
            self._upsert_vectors(MemoryLayer.EVENT, embedded)

    def write_episode(self, ep: Episode) -> None:
        with self._lock, Session(self._engine) as s:
            s.merge(_EpisodeRow(
                episode_id=ep.episode_id,
                trajectory_id=ep.trajectory_id,
                start_step=ep.start_step,
                end_step=ep.end_step,
                sub_goal=ep.sub_goal,
                summary=ep.summary,
                outcome=ep.outcome,
                key_steps_json=_dumps(ep.key_steps),
                ts=ep.ts,
            ))
            s.commit()
        if ep.embedding:
            self._upsert_vector(MemoryLayer.EPISODE, ep.episode_id, ep.embedding,
                                payload={"trajectory_id": ep.trajectory_id, "outcome": ep.outcome})

    def write_pattern(self, p: Pattern) -> None:
        with self._lock, Session(self._engine) as s:
            s.merge(_PatternRow(
                pattern_id=p.pattern_id,
                condition=p.condition,
                action_template=p.action_template,
                expected_effect=p.expected_effect,
                scope_json=_dumps(p.scope),
                support_episodes_json=_dumps(p.support_episodes),
                refute_episodes_json=_dumps(p.refute_episodes),
                alpha=p.alpha,
                beta=p.beta,
                evidence_count=p.evidence_count,
                last_updated=p.last_updated,
                last_used=p.last_used,
                use_count=p.use_count,
                version=p.version,
                parent_pattern_id=p.parent_pattern_id,
            ))
            s.commit()
        if p.embedding:
            self._upsert_vector(MemoryLayer.PATTERN, p.pattern_id, p.embedding,
                                payload={"scope": p.scope, "version": p.version})

    def upsert_pattern(self, p: Pattern) -> None:
        """Alias of write_pattern — explicit for readers of the updater code."""
        self.write_pattern(p)

    def write_principle(self, pr: Principle) -> None:
        with self._lock, Session(self._engine) as s:
            s.merge(_PrincipleRow(
                principle_id=pr.principle_id,
                statement=pr.statement,
                scope=pr.scope,
                supporting_patterns_json=_dumps(pr.supporting_patterns),
                contradiction_log_json=_dumps(pr.contradiction_log),
                alpha=pr.alpha,
                beta=pr.beta,
                last_updated=pr.last_updated,
                last_used=pr.last_used,
                use_count=pr.use_count,
            ))
            s.commit()
        if pr.embedding:
            self._upsert_vector(MemoryLayer.PRINCIPLE, pr.principle_id, pr.embedding,
                                payload={"scope": pr.scope})

    def record_update(self, rec: UpdateRecord) -> None:
        with self._lock, Session(self._engine) as s:
            s.merge(_UpdateRow(
                update_id=rec.update_id,
                pattern_id=rec.pattern_id,
                episode_id=rec.episode_id,
                surprise=rec.surprise,
                delta_alpha=rec.delta_alpha,
                delta_beta=rec.delta_beta,
                triggered_revision=int(rec.triggered_revision),
                new_pattern_id=rec.new_pattern_id,
                ts=rec.ts,
            ))
            s.commit()

    # ------------------------------------------------------------------ #
    # Reads                                                               #
    # ------------------------------------------------------------------ #

    def get_events(self, trajectory_id: str) -> list[Event]:
        with Session(self._engine) as s:
            rows = s.execute(
                select(_EventRow)
                .where(_EventRow.trajectory_id == trajectory_id)
                .order_by(_EventRow.step_idx)
            ).scalars().all()
            return [_event_from_row(r) for r in rows]

    def get_episodes(self, trajectory_id: str | None = None) -> list[Episode]:
        with Session(self._engine) as s:
            stmt = select(_EpisodeRow)
            if trajectory_id is not None:
                stmt = stmt.where(_EpisodeRow.trajectory_id == trajectory_id)
            stmt = stmt.order_by(_EpisodeRow.ts)
            rows = s.execute(stmt).scalars().all()
            return [_episode_from_row(r) for r in rows]

    def get_pattern(self, pattern_id: str) -> Pattern | None:
        with Session(self._engine) as s:
            row = s.get(_PatternRow, pattern_id)
            return _pattern_from_row(row) if row else None

    def get_principle(self, principle_id: str) -> Principle | None:
        with Session(self._engine) as s:
            row = s.get(_PrincipleRow, principle_id)
            return _principle_from_row(row) if row else None

    def all_patterns(self) -> list[Pattern]:
        with Session(self._engine) as s:
            rows = s.execute(select(_PatternRow)).scalars().all()
            return [_pattern_from_row(r) for r in rows]

    def all_principles(self) -> list[Principle]:
        with Session(self._engine) as s:
            rows = s.execute(select(_PrincipleRow)).scalars().all()
            return [_principle_from_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        from sqlalchemy import func

        with Session(self._engine) as s:
            return {
                "events": s.scalar(select(func.count()).select_from(_EventRow)) or 0,
                "episodes": s.scalar(select(func.count()).select_from(_EpisodeRow)) or 0,
                "patterns": s.scalar(select(func.count()).select_from(_PatternRow)) or 0,
                "principles": s.scalar(select(func.count()).select_from(_PrincipleRow)) or 0,
                "updates": s.scalar(select(func.count()).select_from(_UpdateRow)) or 0,
            }

    # ------------------------------------------------------------------ #
    # Vector queries                                                      #
    # ------------------------------------------------------------------ #

    def _upsert_vector(
        self,
        layer: MemoryLayer,
        item_id: str,
        vec: Sequence[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        from qdrant_client.http import models as qm

        if len(vec) != self.vector_size:
            raise ValueError(
                f"Vector dim mismatch: expected {self.vector_size}, got {len(vec)} "
                f"(item_id={item_id})"
            )
        self._qdrant.upsert(
            collection_name=self._coll(layer),
            points=[qm.PointStruct(
                id=_point_id(item_id),
                vector=list(vec),
                payload={"item_id": item_id, **(payload or {})},
            )],
        )

    def _upsert_vectors(
        self,
        layer: MemoryLayer,
        items: list[tuple[str, Sequence[float], dict[str, Any]]],
    ) -> None:
        from qdrant_client.http import models as qm

        points = []
        for item_id, vec, payload in items:
            if len(vec) != self.vector_size:
                raise ValueError(
                    f"Vector dim mismatch: expected {self.vector_size}, got {len(vec)} "
                    f"(item_id={item_id})"
                )
            points.append(qm.PointStruct(
                id=_point_id(item_id),
                vector=list(vec),
                payload={"item_id": item_id, **payload},
            ))
        if points:
            self._qdrant.upsert(collection_name=self._coll(layer), points=points)

    def query_vectors(
        self,
        layer: MemoryLayer,
        vec: Sequence[float],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Returns ``[(item_id, score), ...]`` sorted by descending score."""
        if len(vec) != self.vector_size:
            raise ValueError(
                f"Query vector dim {len(vec)} != collection dim {self.vector_size}"
            )
        try:
            res = self._qdrant.query_points(
                collection_name=self._coll(layer),
                query=list(vec),
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            # qdrant-client < 1.10 fallback
            res = self._qdrant.search(
                collection_name=self._coll(layer),
                query_vector=list(vec),
                limit=top_k,
                with_payload=True,
            )
        out: list[tuple[str, float]] = []
        for hit in res:
            payload = getattr(hit, "payload", {}) or {}
            item_id = payload.get("item_id")
            if item_id:
                out.append((item_id, float(hit.score)))
        return out

    # ------------------------------------------------------------------ #
    # High-level retrieval (§2.4)                                         #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query_vec: Sequence[float],
        rq: RetrievalQuery,
        confidence_threshold_for_episode_drilldown: float = 0.5,
    ) -> MemoryContext:
        # Layer 1: Principles
        prin_hits = self.query_vectors(MemoryLayer.PRINCIPLE, query_vec, top_k=rq.k_principle)
        principles = []
        for pid, _ in prin_hits:
            pr = self.get_principle(pid)
            if pr and pr.confidence >= rq.min_principle_conf:
                principles.append(pr)

        # Layer 2: Patterns
        pat_hits = self.query_vectors(MemoryLayer.PATTERN, query_vec, top_k=rq.k_pattern)
        patterns = []
        for pid, _ in pat_hits:
            pat = self.get_pattern(pid)
            if not pat:
                continue
            if pat.confidence < rq.min_pattern_conf:
                continue
            if rq.scope_tags and pat.scope and not set(pat.scope) & set(rq.scope_tags):
                continue
            patterns.append(pat)

        # Layer 3: Episodes (only if patterns are weak)
        episodes: list[Episode] = []
        if not patterns or _avg_confidence(patterns) < confidence_threshold_for_episode_drilldown:
            ep_hits = self.query_vectors(MemoryLayer.EPISODE, query_vec, top_k=rq.k_episode)
            ep_ids = {pid for pid, _ in ep_hits}
            with Session(self._engine) as s:
                rows = s.execute(
                    select(_EpisodeRow).where(_EpisodeRow.episode_id.in_(list(ep_ids)))
                ).scalars().all()
                episodes = [_episode_from_row(r) for r in rows]

        return MemoryContext(principles=principles, patterns=patterns, episodes=episodes)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        import contextlib

        with self._lock:
            with contextlib.suppress(Exception):
                self._qdrant.close()
            self._engine.dispose()


# --------------------------------------------------------------------------- #
# Row → schema converters                                                     #
# --------------------------------------------------------------------------- #

def _event_from_row(r: _EventRow) -> Event:
    return Event(
        event_id=r.event_id,
        trajectory_id=r.trajectory_id,
        step_idx=r.step_idx,
        state=r.state,
        action=r.action,
        observation=r.observation,
        reward=r.reward,
        ts=r.ts.replace(tzinfo=timezone.utc) if r.ts.tzinfo is None else r.ts,
    )


def _episode_from_row(r: _EpisodeRow) -> Episode:
    return Episode(
        episode_id=r.episode_id,
        trajectory_id=r.trajectory_id,
        start_step=r.start_step,
        end_step=r.end_step,
        sub_goal=r.sub_goal,
        summary=r.summary,
        outcome=r.outcome,  # type: ignore[arg-type]
        key_steps=_loads(r.key_steps_json, []),
        ts=r.ts.replace(tzinfo=timezone.utc) if r.ts.tzinfo is None else r.ts,
    )


def _pattern_from_row(r: _PatternRow) -> Pattern:
    last_updated = r.last_updated.replace(tzinfo=timezone.utc) if r.last_updated.tzinfo is None else r.last_updated
    last_used = r.last_used.replace(tzinfo=timezone.utc) if r.last_used.tzinfo is None else r.last_used
    return Pattern(
        pattern_id=r.pattern_id,
        condition=r.condition,
        action_template=r.action_template,
        expected_effect=r.expected_effect,
        scope=_loads(r.scope_json, []),
        support_episodes=_loads(r.support_episodes_json, []),
        refute_episodes=_loads(r.refute_episodes_json, []),
        alpha=r.alpha,
        beta=r.beta,
        evidence_count=r.evidence_count,
        last_updated=last_updated,
        last_used=last_used,
        use_count=r.use_count,
        version=r.version,
        parent_pattern_id=r.parent_pattern_id,
    )


def _principle_from_row(r: _PrincipleRow) -> Principle:
    last_updated = r.last_updated.replace(tzinfo=timezone.utc) if r.last_updated.tzinfo is None else r.last_updated
    last_used = r.last_used.replace(tzinfo=timezone.utc) if r.last_used.tzinfo is None else r.last_used
    return Principle(
        principle_id=r.principle_id,
        statement=r.statement,
        scope=r.scope,
        supporting_patterns=_loads(r.supporting_patterns_json, []),
        contradiction_log=_loads(r.contradiction_log_json, []),
        alpha=r.alpha,
        beta=r.beta,
        last_updated=last_updated,
        last_used=last_used,
        use_count=r.use_count,
    )


def _avg_confidence(patterns: list[Pattern]) -> float:
    if not patterns:
        return 0.0
    return sum(p.confidence for p in patterns) / len(patterns)


def _point_id(item_id: str) -> str | int:
    """Qdrant only accepts int or UUID-string point IDs.

    We hash arbitrary strings to a deterministic UUID5 so the same business id
    always maps to the same Qdrant id.
    """
    try:
        uuid.UUID(item_id)
        return item_id
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_OID, item_id))
