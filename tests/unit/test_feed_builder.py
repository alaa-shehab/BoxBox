from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.feed.builder import from_session, race_id, slugify
from core.feed.data import ReplayData


def test_meta(fake_replay: ReplayData) -> None:
    m = fake_replay.meta
    assert m.race_id == "2099_01" == race_id(2099, 1)
    assert m.circuit_key == "sao_paulo"
    assert m.total_laps == 6 and m.scheduled_laps == 6
    assert m.is_wet is True
    assert set(m.drivers) == {"AAA", "BBB", "CCC", "DDD", "EEE"}
    assert m.drivers["BBB"].number == "2"
    # Pit-lane start (grid 0) goes to the back; everyone else keeps grid order.
    assert m.grid == {"AAA": 1, "BBB": 2, "CCC": 3, "EEE": 4, "DDD": 5}
    by_code = {r.code: r for r in m.results}
    assert by_code["DDD"].classified == "3" and by_code["DDD"].laps == 5
    assert by_code["CCC"].classified == "R"


def test_laps_normalised(fake_replay: ReplayData) -> None:
    laps = fake_replay.laps
    assert len(laps) == 19  # the junk row without a lap number is dropped
    assert laps["time_s"].dtype == "float64"
    ccc2 = laps[(laps.driver == "CCC") & (laps.lap == 2)].iloc[0]
    assert ccc2["compound"] == "UNKNOWN"  # unrecognised compound
    aaa4 = laps[(laps.driver == "AAA") & (laps.lap == 4)].iloc[0]
    assert bool(aaa4["pit_in"]) and aaa4["lap_time_s"] == pytest.approx(90.5)


def test_messages_extract_drivers(fake_replay: ReplayData) -> None:
    msgs = fake_replay.messages
    assert msgs.loc[1, "drivers"] == "BBB"
    assert msgs.loc[3, "drivers"] == "DDD"  # racing number + text mention, de-duplicated
    assert msgs.loc[3, "flag"] == "BLUE"
    assert pd.isna(msgs.loc[2, "flag"])


def test_track_status_mapping(fake_replay: ReplayData) -> None:
    assert list(fake_replay.track_status["flag"]) == ["GREEN", "SC", "GREEN"]  # "3" dropped


def test_save_load_roundtrip(fake_replay: ReplayData, tmp_path: Path) -> None:
    loaded = ReplayData.load(fake_replay.save(tmp_path / "r"))
    assert loaded.meta == fake_replay.meta
    for name in ("laps", "messages", "track_status", "weather"):
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(fake_replay, name))


def test_conform_rejects_missing_columns(fake_replay: ReplayData) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        ReplayData(
            meta=fake_replay.meta,
            laps=fake_replay.laps.drop(columns=["stint"]),
            messages=fake_replay.messages,
            track_status=fake_replay.track_status,
            weather=fake_replay.weather,
        )


def test_slugify() -> None:
    assert slugify("Yas Marina Circuit") == "yas_marina_circuit"
    assert slugify("Monte-Carlo") == "monte_carlo"


def test_from_session_tolerates_missing_total_laps() -> None:
    from tests.fixtures.fake_session import FakeSession

    session = FakeSession()
    session.total_laps = None  # type: ignore[assignment]
    assert from_session(session, 2099, 1).meta.scheduled_laps is None
