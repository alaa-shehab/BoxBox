"""Domain models shared across core/.

Fan-side entities (profiles, predictions, scores) and the race state every RaceFeed emits.
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


# --------------------------------------------------------------------------------------
# Race state: emitted by every RaceFeed, one tick per lap. Lap 0 is the starting grid.
# --------------------------------------------------------------------------------------

Compound = Literal["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]
TrackFlag = Literal["GREEN", "YELLOW", "VSC", "SC", "RED"]
DriverStatus = Literal["running", "finished", "dnf", "dns"]

COMPOUNDS: tuple[Compound, ...] = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")
FLAG_SEVERITY: dict[TrackFlag, int] = {"GREEN": 0, "YELLOW": 1, "VSC": 2, "SC": 3, "RED": 4}


class DriverInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    number: str
    name: str
    team: str
    team_colour: str = "888888"


class ClassifiedResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    position: int | None  # final order including non-classified cars
    classified: str  # "1".."20", or "R" retired, "D" disqualified, "W" withdrawn/DNS, ...
    status: str  # FastF1 status text, e.g. "Finished", "+1 Lap", "Retired"
    grid: int | None
    points: float
    laps: int


class RaceMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    race_id: str  # "{season}_{round:02d}"
    season: int
    round: int
    event_name: str
    country: str
    circuit_key: str  # stable slug of the circuit location, e.g. "sao_paulo"
    total_laps: int  # laps actually completed by the winner
    scheduled_laps: int | None = None
    is_wet: bool = False
    drivers: dict[str, DriverInfo]
    grid: dict[str, int]  # code -> grid slot (pit-lane starts placed at the back)
    results: list[ClassifiedResult]
    source: str = "FastF1 (F1 live timing archive)"


class DriverState(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    team: str
    position: int
    status: DriverStatus = "running"
    laps_completed: int = 0
    gap_to_leader_s: float | None = None
    interval_s: float | None = None
    laps_down: int = 0
    last_lap_s: float | None = None
    best_lap_s: float | None = None
    compound: Compound = "UNKNOWN"
    tyre_age: int = 0
    stint: int = 1
    pit_count: int = 0
    pitted_this_lap: bool = False


class RaceControlMsg(BaseModel):
    model_config = ConfigDict(frozen=True)

    lap: int
    category: str
    message: str
    flag: str | None = None
    drivers: tuple[str, ...] = ()


class Weather(BaseModel):
    model_config = ConfigDict(frozen=True)

    air_temp_c: float | None = None
    track_temp_c: float | None = None
    humidity_pct: float | None = None
    rainfall: bool = False


class FastestLap(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    lap: int
    time_s: float


class RaceState(BaseModel):
    model_config = ConfigDict(frozen=True)

    race_id: str
    lap: int
    total_laps: int
    flag: TrackFlag = "GREEN"  # flag shown when the leader completed this lap
    flags_this_lap: tuple[TrackFlag, ...] = ()  # every flag shown during the lap, in order
    drivers: tuple[DriverState, ...]
    weather: Weather = Weather()
    new_messages: tuple[RaceControlMsg, ...] = ()
    fastest_lap: FastestLap | None = None
    session_time_s: float | None = None  # leader's lap-end time, seconds from session start

    @property
    def is_final(self) -> bool:
        return self.lap >= self.total_laps

    def driver(self, code: str) -> DriverState | None:
        return next((d for d in self.drivers if d.code == code), None)

    def order(self) -> list[str]:
        return [d.code for d in sorted(self.drivers, key=lambda d: d.position)]


class Citation(BaseModel):
    """A source shown next to an answer: a regulation article, a URL, or a dataset."""

    model_config = ConfigDict(frozen=True)

    source_type: Literal["fia_regulation", "wikipedia", "jolpica", "fastf1", "simulator"]
    label: str  # human-readable, e.g. "2025 F1 Sporting Regulations, Art. 55.1, p. 64"
    url: str | None = None
    article: str | None = None
    page: int | None = None
