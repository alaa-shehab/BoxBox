"""Event detection on the fake race and on the real 2024 São Paulo GP."""

from __future__ import annotations

from core.events.detector import EventDetector
from core.feed.builder import race_id
from core.feed.catalog import SHOWCASE_RACES, ReplayCatalog
from core.feed.replay import ReplayFeed


def _all(feed: ReplayFeed):  # type: ignore[no-untyped-def]
    return EventDetector(feed.meta).run(feed.state(lap) for lap in range(feed.meta.total_laps + 1))


def test_fake_race_story(fake_feed: ReplayFeed) -> None:
    got = {(e.lap, e.type, e.drivers) for e in _all(fake_feed)}
    assert (3, "overtake", ("BBB", "AAA")) in got
    assert (3, "dnf", ("CCC",)) in got
    assert (3, "penalty", ("BBB",)) in got
    assert (5, "pit_stop", ("AAA",)) in got  # entered on lap 4 (SC), reported on out-lap
    assert (4, "safety_car", ()) in got and (5, "safety_car", ()) in got
    assert (6, "weather_change", ()) in got  # rain from lap 5, confirmed on lap 6


def test_sao_paulo_2024_key_moments() -> None:
    feed = ReplayCatalog.default().feed("2024_21")
    events = _all(feed)
    got = {(e.type, e.drivers) for e in events}
    assert ("overtake", ("VER", "OCO")) in got  # the decisive pass for the lead
    assert ("penalty", ("BEA",)) in got and ("penalty", ("PIA",)) in got
    assert {("dnf", ("COL",)), ("dnf", ("SAI",)), ("dnf", ("HUL",))} <= got
    types = {e.type for e in events}
    assert {"safety_car", "vsc", "red_flag", "weather_change", "pit_stop"} <= types
    lead = next(e for e in events if e.type == "overtake" and e.drivers == ("VER", "OCO"))
    assert lead.lap == 43 and lead.payload["for_lead"] is True
    # Red-flag tyre changes are not reported as pit stops.
    assert not [e for e in events if e.type == "pit_stop" and e.lap == 33]


def test_every_showcase_race_produces_a_sane_event_mix() -> None:
    catalog = ReplayCatalog.default()
    for season, rnd in SHOWCASE_RACES:  # committed races only, not a local cache
        rid = race_id(season, rnd)
        events = _all(catalog.feed(rid))
        assert len({e.id for e in events}) == len(events), rid
        assert all(0 <= e.base_importance <= 1 for e in events)
        assert any(e.type == "pit_stop" for e in events), rid
