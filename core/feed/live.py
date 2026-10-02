"""LiveFeed: OpenF1 real-time feed. This is v2.

OpenF1's real-time data needs a paid sponsor account (see PLAN.md §0.4), which is out of
scope under the free-only rule. The class exists to show where it plugs in: it implements
RaceFeed exactly as ReplayFeed does, with `available_lap()` growing as laps complete.
"""

from __future__ import annotations

from core.feed.base import RaceFeed
from core.models import RaceState


class LiveFeed(RaceFeed):  # pragma: no cover - v2
    def available_lap(self) -> int:
        raise NotImplementedError("LiveFeed is planned for v2 (OpenF1 real-time is paid)")

    def state(self, lap: int) -> RaceState:
        raise NotImplementedError("LiveFeed is planned for v2 (OpenF1 real-time is paid)")
