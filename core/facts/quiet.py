"""Quiet-period detection: N laps with no significant event for this fan -> show a fact."""

from __future__ import annotations

from collections.abc import Iterable

from core.events.types import Event


class QuietPeriodDetector:
    """Feed it each lap's personalised events. `update` returns True once per quiet spell.

    "Significant" means importance >= `threshold` after the fan's favourites boost, so a
    midfield overtake doesn't end the quiet spell, but anything involving the fan's
    favourites usually does.
    """

    def __init__(self, laps: int = 5, threshold: float = 0.5) -> None:
        if laps < 1:
            raise ValueError("laps must be >= 1")
        self.laps = laps
        self.threshold = threshold
        self.quiet_laps = 0
        self._last_lap: int | None = None

    def reset(self) -> None:
        self.quiet_laps = 0
        self._last_lap = None

    def update(self, lap: int, events: Iterable[Event]) -> bool:
        if self._last_lap is not None and lap <= self._last_lap:
            self.reset()  # seek backwards: start counting again
        self._last_lap = lap
        if lap < 1:
            return False
        if any(e.importance >= self.threshold for e in events):
            self.quiet_laps = 0
            return False
        self.quiet_laps += 1
        if self.quiet_laps >= self.laps:
            self.quiet_laps = 0
            return True
        return False
