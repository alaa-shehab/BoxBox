"""Domain models shared across core/.

Race-state models (RaceState, DriverState, Event, ...) are added with the RaceFeed in
Phase 2. This module currently holds the persisted fan-side entities.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Level = Literal["rookie", "fan", "expert"]

_DRIVER_CODE = re.compile(r"^[A-Z]{3}$")
_NICKNAME = re.compile(r"^[\w .\-]{2,24}$")


def utcnow() -> datetime:
    return datetime.now(UTC)


def _driver_code(code: str) -> str:
    code = code.strip().upper()
    if not _DRIVER_CODE.match(code):
        raise ValueError(f"driver code must be 3 letters, got {code!r}")
    return code


class Profile(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str = Field(default_factory=lambda: uuid4().hex)
    nickname: str
    favourite_team: str
    favourite_drivers: tuple[str, ...] = ()
    level: Level = "fan"
    quiz_score: int | None = Field(default=None, ge=0, le=3)
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("nickname")
    @classmethod
    def _nickname(cls, v: str) -> str:
        v = v.strip()
        if not _NICKNAME.match(v):
            raise ValueError("nickname must be 2-24 letters, digits, spaces, '.', '_' or '-'")
        return v

    @field_validator("favourite_team")
    @classmethod
    def _team(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("favourite_team is required")
        return v

    @field_validator("favourite_drivers")
    @classmethod
    def _drivers(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        codes = tuple(dict.fromkeys(_driver_code(c) for c in v))
        if len(codes) > 2:
            raise ValueError("at most 2 favourite drivers")
        return codes


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    race_id: str
    p1: str
    p2: str
    p3: str
    pole_to_win: bool | None = None
    first_dnf: str | None = None
    fastest_lap: str | None = None
    locked_at_lap: int | None = Field(default=None, ge=0)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("p1", "p2", "p3")
    @classmethod
    def _podium_code(cls, v: str) -> str:
        return _driver_code(v)

    @field_validator("first_dnf", "fastest_lap")
    @classmethod
    def _optional_code(cls, v: str | None) -> str | None:
        return None if v is None else _driver_code(v)

    @model_validator(mode="after")
    def _distinct_podium(self) -> Prediction:
        if len({self.p1, self.p2, self.p3}) != 3:
            raise ValueError("podium picks must be three different drivers")
        return self

    @property
    def podium(self) -> tuple[str, str, str]:
        return (self.p1, self.p2, self.p3)

    @property
    def is_locked(self) -> bool:
        return self.locked_at_lap is not None


class Score(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    race_id: str
    points: int
    breakdown: dict[str, int] = Field(default_factory=dict)
    scored_at: datetime = Field(default_factory=utcnow)
