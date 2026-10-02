"""Persistence for profiles, predictions, scores and per-session fact history.

There is one `Repository` protocol and one SQLAlchemy Core implementation. The same SQL runs
on SQLite (local, tests) and Postgres (Neon in deployment), and `DATABASE_URL` selects
between them. Nothing is ever written to local disk in deployment, because Streamlit
Cloud storage is ephemeral.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from core.log import get_logger
from core.models import Level, Prediction, Profile, Score

log = get_logger(__name__)


class RepositoryError(Exception):
    """Base class for expected, user-facing persistence errors."""


class NicknameTakenError(RepositoryError):
    pass


class ProfileNotFoundError(RepositoryError):
    pass


class PredictionLockedError(RepositoryError):
    pass


class Repository(Protocol):
    def create_profile(self, profile: Profile) -> Profile: ...
    def get_profile(self, profile_id: str) -> Profile | None: ...
    def get_profile_by_nickname(self, nickname: str) -> Profile | None: ...
    def update_profile(
        self,
        profile_id: str,
        *,
        level: Level | None = None,
        favourite_team: str | None = None,
        favourite_drivers: tuple[str, ...] | None = None,
    ) -> Profile: ...
    def save_prediction(self, prediction: Prediction) -> Prediction: ...
    def get_prediction(self, profile_id: str, race_id: str) -> Prediction | None: ...
    def lock_prediction(self, profile_id: str, race_id: str, lap: int) -> Prediction | None: ...
    def save_score(self, score: Score) -> Score: ...
    def get_score(self, profile_id: str, race_id: str) -> Score | None: ...
    def mark_fact_seen(self, session_id: str, fact_id: str) -> bool: ...
    def seen_facts(self, session_id: str) -> set[str]: ...


metadata = MetaData()

profiles = Table(
    "profiles",
    metadata,
    Column("profile_id", String(32), primary_key=True),
    Column("nickname", String(24), nullable=False),
    Column("nickname_key", String(24), nullable=False, unique=True),
    Column("favourite_team", String(64), nullable=False),
    Column("favourite_drivers", JSON, nullable=False),
    Column("level", String(8), nullable=False),
    Column("quiz_score", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

predictions = Table(
    "predictions",
    metadata,
    Column("profile_id", String(32), primary_key=True),
    Column("race_id", String(32), primary_key=True),
    Column("p1", String(3), nullable=False),
    Column("p2", String(3), nullable=False),
    Column("p3", String(3), nullable=False),
    Column("pole_to_win", Boolean, nullable=True),
    Column("first_dnf", String(3), nullable=True),
    Column("fastest_lap", String(3), nullable=True),
    Column("locked_at_lap", Integer, nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

scores = Table(
    "scores",
    metadata,
    Column("profile_id", String(32), primary_key=True),
    Column("race_id", String(32), primary_key=True),
    Column("points", Integer, nullable=False),
    Column("breakdown", JSON, nullable=False),
    Column("scored_at", DateTime(timezone=True), nullable=False),
)

facts_seen = Table(
    "session_facts_seen",
    metadata,
    Column("session_id", String(64), nullable=False),
    Column("fact_id", String(128), nullable=False),
    Column("seen_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("session_id", "fact_id", name="uq_session_fact"),
)


def make_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        if url.database in (None, "", ":memory:"):
            # One shared connection so every session sees the same in-memory DB.
            return create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(database_url, connect_args={"check_same_thread": False})
    # Neon scales to zero; pre-ping transparently replaces connections it dropped.
    return create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def _aware(dt: datetime) -> datetime:
    # SQLite drops tzinfo; everything we store is UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class SqlRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        metadata.create_all(engine)

    @classmethod
    def from_url(cls, database_url: str) -> SqlRepository:
        return cls(make_engine(database_url))

    # --- profiles ---
    @staticmethod
    def _profile(row: Any) -> Profile:
        return Profile(
            profile_id=row.profile_id,
            nickname=row.nickname,
            favourite_team=row.favourite_team,
            favourite_drivers=tuple(row.favourite_drivers),
            level=row.level,
            quiz_score=row.quiz_score,
            created_at=_aware(row.created_at),
        )

    def create_profile(self, profile: Profile) -> Profile:
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(profiles).values(
                        profile_id=profile.profile_id,
                        nickname=profile.nickname,
                        nickname_key=profile.nickname.casefold(),
                        favourite_team=profile.favourite_team,
                        favourite_drivers=list(profile.favourite_drivers),
                        level=profile.level,
                        quiz_score=profile.quiz_score,
                        created_at=profile.created_at,
                    )
                )
        except IntegrityError as exc:
            raise NicknameTakenError(f"nickname {profile.nickname!r} is taken") from exc
        log.info("profile.created", extra={"fields": {"profile_id": profile.profile_id}})
        return profile

    def get_profile(self, profile_id: str) -> Profile | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(profiles).where(profiles.c.profile_id == profile_id)).first()
        return self._profile(row) if row else None

    def get_profile_by_nickname(self, nickname: str) -> Profile | None:
        key = nickname.strip().casefold()
        with self.engine.connect() as conn:
            row = conn.execute(select(profiles).where(profiles.c.nickname_key == key)).first()
        return self._profile(row) if row else None

    def update_profile(
        self,
        profile_id: str,
        *,
        level: Level | None = None,
        favourite_team: str | None = None,
        favourite_drivers: tuple[str, ...] | None = None,
    ) -> Profile:
        current = self.get_profile(profile_id)
        if current is None:
            raise ProfileNotFoundError(profile_id)
        changes: dict[str, Any] = {}
        if level is not None:
            changes["level"] = level
        if favourite_team is not None:
            changes["favourite_team"] = favourite_team
        if favourite_drivers is not None:
            changes["favourite_drivers"] = favourite_drivers
        updated = Profile.model_validate({**current.model_dump(), **changes})  # re-validate
        with self.engine.begin() as conn:
            conn.execute(
                update(profiles)
                .where(profiles.c.profile_id == profile_id)
                .values(
                    level=updated.level,
                    favourite_team=updated.favourite_team,
                    favourite_drivers=list(updated.favourite_drivers),
                )
            )
        return updated

    # --- predictions ---
    @staticmethod
    def _prediction(row: Any) -> Prediction:
        return Prediction(
            profile_id=row.profile_id,
            race_id=row.race_id,
            p1=row.p1,
            p2=row.p2,
            p3=row.p3,
            pole_to_win=row.pole_to_win,
            first_dnf=row.first_dnf,
            fastest_lap=row.fastest_lap,
            locked_at_lap=row.locked_at_lap,
            updated_at=_aware(row.updated_at),
        )

    def save_prediction(self, prediction: Prediction) -> Prediction:
        """Insert or replace a prediction. Refuses once the stored prediction is locked."""
        key = (predictions.c.profile_id == prediction.profile_id) & (
            predictions.c.race_id == prediction.race_id
        )
        values = prediction.model_dump(exclude={"profile_id", "race_id"})
        with self.engine.begin() as conn:
            existing = conn.execute(select(predictions.c.locked_at_lap).where(key)).first()
            if existing is None:
                conn.execute(
                    insert(predictions).values(
                        profile_id=prediction.profile_id, race_id=prediction.race_id, **values
                    )
                )
            elif existing.locked_at_lap is not None:
                raise PredictionLockedError(
                    f"prediction for {prediction.race_id} locked at lap {existing.locked_at_lap}"
                )
            else:
                conn.execute(update(predictions).where(key).values(**values))
        return prediction

    def get_prediction(self, profile_id: str, race_id: str) -> Prediction | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(predictions).where(
                    (predictions.c.profile_id == profile_id) & (predictions.c.race_id == race_id)
                )
            ).first()
        return self._prediction(row) if row else None

    def lock_prediction(self, profile_id: str, race_id: str, lap: int) -> Prediction | None:
        """Lock at lights out. Idempotent: an existing lock keeps its original lap."""
        key = (
            (predictions.c.profile_id == profile_id)
            & (predictions.c.race_id == race_id)
            & predictions.c.locked_at_lap.is_(None)
        )
        with self.engine.begin() as conn:
            conn.execute(update(predictions).where(key).values(locked_at_lap=lap))
        return self.get_prediction(profile_id, race_id)

    # --- scores ---
    def save_score(self, score: Score) -> Score:
        key = (scores.c.profile_id == score.profile_id) & (scores.c.race_id == score.race_id)
        values = {
            "points": score.points,
            "breakdown": score.breakdown,
            "scored_at": score.scored_at,
        }
        with self.engine.begin() as conn:
            if conn.execute(select(scores.c.points).where(key)).first() is None:
                conn.execute(
                    insert(scores).values(
                        profile_id=score.profile_id, race_id=score.race_id, **values
                    )
                )
            else:
                conn.execute(update(scores).where(key).values(**values))
        return score

    def get_score(self, profile_id: str, race_id: str) -> Score | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(scores).where(
                    (scores.c.profile_id == profile_id) & (scores.c.race_id == race_id)
                )
            ).first()
        if row is None:
            return None
        return Score(
            profile_id=row.profile_id,
            race_id=row.race_id,
            points=row.points,
            breakdown=dict(row.breakdown),
            scored_at=_aware(row.scored_at),
        )

    # --- facts shown per session (no repeats) ---
    def mark_fact_seen(self, session_id: str, fact_id: str) -> bool:
        """Record a fact as shown. Returns False if it was already shown in this session."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(facts_seen).values(
                        session_id=session_id, fact_id=fact_id, seen_at=datetime.now(UTC)
                    )
                )
        except IntegrityError:
            return False
        return True

    def seen_facts(self, session_id: str) -> set[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(facts_seen.c.fact_id).where(facts_seen.c.session_id == session_id)
            )
            return {r.fact_id for r in rows}

    # --- test helper ---
    def _truncate_all(self) -> None:
        with self.engine.begin() as conn:
            for table in reversed(metadata.sorted_tables):
                conn.execute(delete(table))
