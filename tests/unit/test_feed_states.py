from __future__ import annotations

import pytest

from core.feed.builder import from_session
from core.feed.replay import ReplayFeed
from core.feed.validate import validate_replay
from tests.fixtures.fake_session import FakeSession


def test_grid_state(fake_feed: ReplayFeed) -> None:
    s = fake_feed.state(0)
    assert s.order() == ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert s.driver("EEE").status == "dns"
    assert s.driver("AAA").gap_to_leader_s is None
    assert s.driver("AAA").tyre_age == 0
    assert s.driver("CCC").tyre_age == 2  # started on a used set
    assert s.weather.rainfall is False


def test_lap1_order_gaps_intervals(fake_feed: ReplayFeed) -> None:
    s = fake_feed.state(1)
    assert s.order() == ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert [d.gap_to_leader_s for d in s.drivers[:4]] == [0.0, 1.0, 2.0, 20.0]
    assert [d.interval_s for d in s.drivers[:4]] == [None, 1.0, 1.0, 18.0]
    assert s.driver("AAA").last_lap_s == pytest.approx(90.0)


def test_overtake_and_retirement_lap3(fake_feed: ReplayFeed) -> None:
    s = fake_feed.state(3)
    assert s.order()[:3] == ["BBB", "AAA", "DDD"]
    assert s.driver("AAA").gap_to_leader_s == pytest.approx(0.8)
    ccc = s.driver("CCC")
    assert ccc.status == "dnf" and ccc.position == 4 and ccc.laps_completed == 2
    assert ccc.gap_to_leader_s is None
    assert s.driver("DDD").laps_down == 1  # crossed after the leader completed lap 4


def test_pit_stop_and_new_stint(fake_feed: ReplayFeed) -> None:
    lap4 = fake_feed.state(4).driver("AAA")
    assert lap4.pitted_this_lap and lap4.pit_count == 1 and lap4.compound == "MEDIUM"
    lap5 = fake_feed.state(5).driver("AAA")
    assert not lap5.pitted_this_lap
    assert (lap5.compound, lap5.tyre_age, lap5.stint, lap5.pit_count) == ("HARD", 1, 2, 1)
    assert lap5.gap_to_leader_s == pytest.approx(23.0)


def test_flags_within_and_at_end_of_lap(fake_feed: ReplayFeed) -> None:
    assert fake_feed.state(3).flags_this_lap == ("GREEN",)
    lap4 = fake_feed.state(4)
    assert lap4.flag == "SC" and lap4.flags_this_lap == ("GREEN", "SC")
    lap5 = fake_feed.state(5)
    assert lap5.flag == "GREEN" and lap5.flags_this_lap == ("SC", "GREEN")


def test_weather_changes(fake_feed: ReplayFeed) -> None:
    assert fake_feed.state(4).weather.rainfall is False
    lap5 = fake_feed.state(5).weather
    assert lap5.rainfall is True and lap5.humidity_pct == pytest.approx(90.0)


def test_messages_bucketed_by_lap(fake_feed: ReplayFeed) -> None:
    assert [m.message for m in fake_feed.state(4).new_messages] == [
        "SAFETY CAR DEPLOYED",
        "WAVED BLUE FLAG FOR CAR 4 (DDD)",
    ]
    assert fake_feed.state(3).new_messages[0].drivers == ("BBB",)
    assert fake_feed.state(5).new_messages == ()


def test_fastest_lap_ignores_deleted_laps(fake_feed: ReplayFeed) -> None:
    assert fake_feed.state(0).fastest_lap is None
    fl2 = fake_feed.state(2).fastest_lap
    assert fl2 is not None and (fl2.code, fl2.lap, fl2.time_s) == ("AAA", 1, pytest.approx(90.0))
    fl3 = fake_feed.state(3).fastest_lap
    assert fl3 is not None and (fl3.code, fl3.lap) == ("BBB", 3)
    assert fake_feed.state(3).driver("BBB").best_lap_s == pytest.approx(89.7)


def test_final_lap_statuses(fake_feed: ReplayFeed) -> None:
    s = fake_feed.state(6)
    assert s.is_final
    assert s.order() == ["BBB", "AAA", "DDD", "CCC", "EEE"]
    assert [d.status for d in s.drivers] == ["finished", "finished", "finished", "dnf", "dns"]
    assert s.driver("DDD").laps_down == 1 and s.driver("DDD").laps_completed == 5


def test_lapped_car_still_running_before_flag(fake_feed: ReplayFeed) -> None:
    ddd = fake_feed.state(5).driver("DDD")
    assert ddd.status == "running" and ddd.laps_down == 1


def test_final_tick_uses_official_classification() -> None:
    """A post-race time penalty can swap cars that crossed the line in the other order."""
    session = FakeSession()
    res = session.results.copy()
    res.loc[res.Abbreviation == "BBB", ["Position", "ClassifiedPosition"]] = [2.0, "2"]
    res.loc[res.Abbreviation == "AAA", ["Position", "ClassifiedPosition"]] = [1.0, "1"]
    session.results = res
    feed = ReplayFeed(from_session(session, 2099, 1))
    assert feed.state(5).order()[:2] == ["BBB", "AAA"]  # on track
    final = feed.state(6)
    assert final.order()[:2] == ["AAA", "BBB"]  # official
    assert final.driver("AAA").gap_to_leader_s == 0.0
    assert final.driver("BBB").gap_to_leader_s is None  # beat AAA on track; no valid gap
    assert final.driver("BBB").interval_s is None
    assert validate_replay(feed) == []


def test_positions_are_contiguous_every_lap(fake_feed: ReplayFeed) -> None:
    assert validate_replay(fake_feed) == []
    for lap in range(7):
        assert sorted(d.position for d in fake_feed.state(lap).drivers) == [1, 2, 3, 4, 5]


def test_validate_reports_wrong_order(fake_replay) -> None:  # type: ignore[no-untyped-def]
    meta = fake_replay.meta
    swapped = [
        r.model_copy(update={"classified": {"BBB": "2", "AAA": "1"}.get(r.code, r.classified)})
        for r in meta.results
    ]
    feed = ReplayFeed(fake_replay)
    feed.meta = meta.model_copy(update={"results": swapped})
    assert validate_replay(feed)[0].startswith("final order differs")
