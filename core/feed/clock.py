"""Maps wall-clock time to replay laps at 1x, 5x, 20x, or manual stepping.

The clock is pure: callers pass `now` (e.g. `time.monotonic()`), which keeps it fully
testable and lets the UI poll it from a `st.fragment(run_every=...)` without threads.

Each lap lasts as long as the leader actually took for it, divided by the speed. Long
neutralisations (red flags) are capped at `max_lap_s` of race time, so a 26-minute
suspension doesn't leave the replay idle.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from itertools import accumulate
from typing import Literal

Speed = Literal[1, 5, 20, "step"]
SPEEDS: tuple[Speed, ...] = (1, 5, 20, "step")


class ReplayClock:
    def __init__(
        self, lap_durations_s: Sequence[float], speed: Speed = 5, max_lap_s: float = 180.0
    ) -> None:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        durations = [min(max(d, 1.0), max_lap_s) for d in lap_durations_s]
        self._ends = list(accumulate(durations))  # race time at the end of each lap
        self.total_laps = len(durations)
        self.speed: Speed = speed
        self._race_time = 0.0  # race seconds elapsed at the anchor
        self._anchor: float | None = None  # wall time when last (re)started; None = paused

    @property
    def running(self) -> bool:
        return self._anchor is not None

    def _race_time_at(self, now: float) -> float:
        if self._anchor is None or self.speed == "step":
            return self._race_time
        return self._race_time + (now - self._anchor) * self.speed

    def lap_at(self, now: float) -> int:
        t = self._race_time_at(now)
        return min(bisect.bisect_right(self._ends, t + 1e-9), self.total_laps)

    def start(self, now: float) -> None:
        if self.speed != "step" and self._anchor is None:
            self._anchor = now

    def pause(self, now: float) -> None:
        self._race_time = self._race_time_at(now)
        self._anchor = None

    def set_speed(self, speed: Speed, now: float) -> None:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        was_running = self.running and self.speed != "step"
        self._race_time = self._race_time_at(now)
        self.speed = speed
        self._anchor = now if was_running and speed != "step" else None

    def step(self, now: float, laps: int = 1) -> int:
        """Manual stepping: pause, then jump to the end of the lap `laps` ahead."""
        target = self.lap_at(now) + laps
        self.pause(now)
        self.seek(target, now)
        return self.lap_at(now)

    def seek(self, lap: int, now: float) -> None:
        """Jump to the end of `lap`, keeping the running/paused state."""
        lap = max(0, min(lap, self.total_laps))
        self._race_time = 0.0 if lap == 0 else self._ends[lap - 1]
        if self._anchor is not None:
            self._anchor = now

    def seconds_to_next_lap(self, now: float) -> float | None:
        """Wall seconds until the next tick, or None if paused, stepping or finished."""
        lap = self.lap_at(now)
        if not self.running or self.speed == "step" or lap >= self.total_laps:
            return None
        return (self._ends[lap] - self._race_time_at(now)) / float(self.speed)
