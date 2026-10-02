"""Where replays come from: committed showcase races first, then an on-demand cache.

Showcase races are committed to `data/replays/`, so the demo never waits on FastF1.
Any other race is built on demand with FastF1 into the cache dir. That's ephemeral on
Streamlit Cloud, which is fine because it is only a cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.config import get_settings
from core.feed.builder import build_from_fastf1, race_id
from core.feed.data import ReplayData
from core.feed.replay import ReplayFeed
from core.feed.states import build_states
from core.log import get_logger
from core.models import RaceMeta, RaceState

log = get_logger(__name__)

# (season, round): chosen so that every event type fires at least once.
SHOWCASE_RACES: tuple[tuple[int, int], ...] = (
    (2024, 21),  # São Paulo: rain, red flag, safety cars, a win from P17
    (2024, 16),  # Monza: one-stop vs two-stop strategy battle
    (2025, 8),  # Monaco: mandatory two-stop rule, track position
    (2025, 12),  # Silverstone: changing weather, safety car
    (2023, 21),  # Las Vegas: safety car, penalty, recovery drive
)


class ReplayCatalog:
    def __init__(self, committed_dir: Path, cache_dir: Path, fastf1_cache_dir: Path) -> None:
        self.committed_dir = committed_dir
        self.cache_dir = cache_dir
        self.fastf1_cache_dir = fastf1_cache_dir

    @classmethod
    def default(cls) -> ReplayCatalog:
        s = get_settings()
        return cls(s.replay_dir, s.data_dir / "cache" / "replays", s.fastf1_cache_dir)

    def _dir(self, rid: str) -> Path | None:
        for base in (self.committed_dir, self.cache_dir):
            if (base / rid / "meta.json").exists():
                return base / rid
        return None

    def available(self) -> list[RaceMeta]:
        metas: dict[str, RaceMeta] = {}
        for base in (self.cache_dir, self.committed_dir):  # committed wins on conflicts
            if base.exists():
                for meta_file in sorted(base.glob("*/meta.json")):
                    meta = RaceMeta.model_validate_json(meta_file.read_text())
                    metas[meta.race_id] = meta
        return sorted(metas.values(), key=lambda m: (m.season, m.round), reverse=True)

    def has(self, rid: str) -> bool:
        return self._dir(rid) is not None

    def load_data(self, rid: str) -> ReplayData:
        directory = self._dir(rid)
        if directory is None:
            raise KeyError(f"no replay for {rid}")
        return _load_cached(directory)[0]

    def feed(self, rid: str) -> ReplayFeed:
        """A fresh feed (own cursor) over shared, cached replay data and states."""
        directory = self._dir(rid)
        if directory is None:
            raise KeyError(f"no replay for {rid}")
        data, states = _load_cached(directory)
        return ReplayFeed(data, states)

    def ensure(self, season: int, round_number: int) -> str:
        """Make a race available, building it with FastF1 if needed. Needs network."""
        rid = race_id(season, round_number)
        if not self.has(rid):
            build_from_fastf1(season, round_number, self.fastf1_cache_dir).save(
                self.cache_dir / rid
            )
        return rid


@lru_cache(maxsize=8)
def _load_cached(directory: Path) -> tuple[ReplayData, tuple[RaceState, ...]]:
    """States are immutable, so every session over the same race shares them."""
    log.info("replay.load", extra={"fields": {"dir": str(directory)}})
    data = ReplayData.load(directory)
    return data, tuple(build_states(data))
