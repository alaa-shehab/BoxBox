"""The committed showcase replays must load fast and reproduce the official results."""

from __future__ import annotations

import os
import time

import pytest

from core.feed.builder import race_id
from core.feed.catalog import SHOWCASE_RACES, ReplayCatalog
from core.feed.validate import validate_replay

CATALOG = ReplayCatalog.default()
SHOWCASE_IDS = [race_id(s, r) for s, r in SHOWCASE_RACES]


def test_all_showcase_races_committed() -> None:
    assert {m.race_id for m in CATALOG.available()} >= set(SHOWCASE_IDS)


@pytest.mark.parametrize("rid", SHOWCASE_IDS)
def test_replay_matches_official_result(rid: str) -> None:
    feed = CATALOG.feed(rid)
    assert validate_replay(feed) == []
    final = feed.state(feed.meta.total_laps)
    winner = next(r for r in feed.meta.results if r.classified == "1")
    assert final.drivers[0].code == winner.code


@pytest.mark.parametrize("rid", SHOWCASE_IDS)
def test_replay_is_compact(rid: str) -> None:
    directory = CATALOG.committed_dir / rid
    assert sum(f.stat().st_size for f in directory.iterdir()) < 100 * 1024


def test_cold_load_is_fast() -> None:
    from core.feed.catalog import _load_cached

    _load_cached.cache_clear()
    start = time.perf_counter()
    for rid in SHOWCASE_IDS:
        CATALOG.feed(rid)
    assert time.perf_counter() - start < 5.0


def test_sao_paulo_2024_story() -> None:
    """Spot-check real events in the 2024 São Paulo GP replay."""
    feed = CATALOG.feed("2024_21")
    flags = {f for lap in range(1, 70) for f in feed.state(lap).flags_this_lap}
    assert {"VSC", "SC", "RED"} <= flags
    assert feed.state(1).weather.rainfall
    assert feed.state(1).drivers[0].compound == "INTERMEDIATE"
    assert feed.meta.grid["VER"] == 17
    assert feed.state(0).driver("ALB").status == "dns"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("RUN_NETWORK_TESTS"), reason="set RUN_NETWORK_TESTS=1")
def test_rebuild_from_fastf1_matches_committed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from core.config import get_settings
    from core.feed.builder import build_from_fastf1

    fresh = build_from_fastf1(2024, 21, get_settings().fastf1_cache_dir)
    assert fresh.meta == CATALOG.load_data("2024_21").meta
