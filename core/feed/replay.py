"""ReplayFeed: rebuilds a past race lap by lap from committed or cached ReplayData."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from core.feed.base import RaceFeed
from core.feed.clock import ReplayClock, Speed
from core.feed.data import ReplayData
from core.feed.states import build_states
from core.models import RaceState


class ReplayFeed(RaceFeed):
    def __init__(self, data: ReplayData, states: Sequence[RaceState] | None = None) -> None:
        super().__init__(data.meta)
        self.data = data
        self._states = list(states) if states is not None else build_states(data)
        if len(self._states) != self.meta.total_laps + 1:
            raise ValueError("states must cover laps 0..total_laps")

    def available_lap(self) -> int:
        return self.meta.total_laps

    def state(self, lap: int) -> RaceState:
        if not 0 <= lap <= self.meta.total_laps:
            raise IndexError(f"lap {lap} outside 0..{self.meta.total_laps}")
        return self._states[lap]

    def lap_durations(self) -> list[float]:
        """Real seconds the leader took for each lap (index 0 = lap 1)."""
        times = [s.session_time_s or 0.0 for s in self._states]
        return [max(0.0, b - a) for a, b in pairwise(times)]

    def make_clock(self, speed: Speed = 5) -> ReplayClock:
        return ReplayClock(self.lap_durations(), speed=speed)
