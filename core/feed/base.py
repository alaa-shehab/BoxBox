"""The single interface every race feature consumes.

A RaceFeed emits one RaceState per lap (lap 0 = starting grid). Consumers keep a cursor
and call `advance()` to receive the ticks they haven't seen yet. ReplayFeed (v1) and
LiveFeed (v2) both implement this, and no feature talks to a data source directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import RaceMeta, RaceState


class RaceFeed(ABC):
    meta: RaceMeta

    def __init__(self, meta: RaceMeta) -> None:
        self.meta = meta
        self._cursor = 0

    @abstractmethod
    def available_lap(self) -> int:
        """The highest lap for which a state can be produced right now."""

    @abstractmethod
    def state(self, lap: int) -> RaceState:
        """The race state at the end of `lap` (0 = grid). Raises IndexError if unavailable."""

    @property
    def current_lap(self) -> int:
        return self._cursor

    @property
    def finished(self) -> bool:
        return self._cursor >= self.meta.total_laps

    def current_state(self) -> RaceState:
        return self.state(self._cursor)

    def advance(self, to_lap: int) -> list[RaceState]:
        """Move the cursor forward to `to_lap` (clipped) and return every new tick."""
        target = min(to_lap, self.available_lap(), self.meta.total_laps)
        if target <= self._cursor:
            return []
        ticks = [self.state(lap) for lap in range(self._cursor + 1, target + 1)]
        self._cursor = target
        return ticks

    def next(self) -> RaceState | None:
        ticks = self.advance(self._cursor + 1)
        return ticks[0] if ticks else None

    def seek(self, lap: int) -> RaceState:
        """Jump to a lap without emitting intermediate ticks (consumers must resync)."""
        lap = max(0, min(lap, self.available_lap(), self.meta.total_laps))
        self._cursor = lap
        return self.state(lap)
