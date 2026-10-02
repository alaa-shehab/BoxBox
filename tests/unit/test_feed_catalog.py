from __future__ import annotations

from pathlib import Path

import pytest

from core.feed import catalog as catalog_mod
from core.feed.catalog import ReplayCatalog
from core.feed.data import ReplayData


@pytest.fixture
def cat(tmp_path: Path) -> ReplayCatalog:
    catalog_mod._load_cached.cache_clear()
    return ReplayCatalog(tmp_path / "committed", tmp_path / "cache", tmp_path / "ff1")


def test_empty(cat: ReplayCatalog) -> None:
    assert cat.available() == []
    assert not cat.has("2099_01")
    with pytest.raises(KeyError):
        cat.feed("2099_01")
    with pytest.raises(KeyError):
        cat.load_data("2099_01")


def test_committed_race_loads(cat: ReplayCatalog, fake_replay: ReplayData) -> None:
    fake_replay.save(cat.committed_dir / "2099_01")
    assert [m.race_id for m in cat.available()] == ["2099_01"]
    a, b = cat.feed("2099_01"), cat.feed("2099_01")
    a.advance(3)
    assert b.current_lap == 0  # independent cursors
    assert a.state(6) is b.state(6)  # shared immutable states


def test_committed_wins_over_cache(cat: ReplayCatalog, fake_replay: ReplayData) -> None:
    fake_replay.save(cat.committed_dir / "2099_01")
    renamed = fake_replay.meta.model_copy(update={"event_name": "Cached copy"})
    ReplayData(
        renamed,
        fake_replay.laps,
        fake_replay.messages,
        fake_replay.track_status,
        fake_replay.weather,
    ).save(cat.cache_dir / "2099_01")
    assert cat.available()[0].event_name == "Test Grand Prix"
    assert cat.load_data("2099_01").meta.event_name == "Test Grand Prix"


def test_ensure_builds_only_when_missing(
    cat: ReplayCatalog, fake_replay: ReplayData, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_build(season: int, rnd: int, cache_dir: Path) -> ReplayData:
        calls.append((season, rnd))
        return fake_replay

    monkeypatch.setattr(catalog_mod, "build_from_fastf1", fake_build)
    assert cat.ensure(2099, 1) == "2099_01"
    assert cat.ensure(2099, 1) == "2099_01"
    assert calls == [(2099, 1)]
    assert (cat.cache_dir / "2099_01" / "laps.parquet").exists()


def test_default_uses_settings() -> None:
    cat = ReplayCatalog.default()
    assert cat.committed_dir.name == "replays"
