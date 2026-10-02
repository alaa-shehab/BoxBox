"""Race events detected from consecutive RaceState ticks."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EventType = Literal[
    "overtake",
    "pit_stop",
    "undercut",
    "overcut",
    "safety_car",
    "vsc",
    "red_flag",
    "penalty",
    "race_control",
    "fastest_lap",
    "tyre_cliff",
    "dnf",
    "weather_change",
]

# Events whose explanation should cite the FIA regulations.
RULE_EVENTS: frozenset[EventType] = frozenset(
    {"safety_car", "vsc", "red_flag", "penalty", "race_control"}
)

PayloadValue = str | int | float | bool | None


class Event(BaseModel):
    """A fan-independent fact about the race. Every number in `payload` comes from data."""

    model_config = ConfigDict(frozen=True)

    id: str
    race_id: str
    lap: int
    type: EventType
    drivers: tuple[str, ...] = ()
    teams: tuple[str, ...] = ()
    payload: dict[str, PayloadValue] = Field(default_factory=dict)
    summary: str  # deterministic one-liner; also the fallback when no LLM is available
    base_importance: float = Field(ge=0.0, le=1.0)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)  # after the fan's favourites boost
    involves_favourite: bool = False

    @model_validator(mode="before")
    @classmethod
    def _default_importance(cls, data: Any) -> Any:
        if isinstance(data, dict) and "importance" not in data and "base_importance" in data:
            data = {**data, "importance": data["base_importance"]}
        return data

    @property
    def needs_regs(self) -> bool:
        return self.type in RULE_EVENTS
