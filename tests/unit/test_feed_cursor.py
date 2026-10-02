from __future__ import annotations

import pytest

from core.feed.replay import ReplayFeed


def test_advance_emits_each_tick_once(fake_feed: ReplayFeed) -> None:
    assert fake_feed.current_lap == 0
    assert [s.lap for s in fake_feed.advance(3)] == [1, 2, 3]
    assert fake_feed.advance(3) == []
    assert fake_feed.advance(2) == []  # never goes backwards
    assert [s.lap for s in fake_feed.advance(99)] == [4, 5, 6]  # clipped
    assert fake_feed.finished


def test_next_and_seek(fake_feed: ReplayFeed) -> None:
    first = fake_feed.next()
    assert first is not None and first.lap == 1
    assert fake_feed.seek(5).lap == 5
    assert fake_feed.current_state().lap == 5
    last = fake_feed.next()
    assert last is not None and last.lap == 6
    assert fake_feed.next() is None
    assert fake_feed.seek(-3).lap == 0
    assert fake_feed.seek(100).lap == 6


def test_state_out_of_range(fake_feed: ReplayFeed) -> None:
    with pytest.raises(IndexError):
        fake_feed.state(7)


def test_states_must_cover_all_laps(fake_replay) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        ReplayFeed(fake_replay, states=[])


def test_lap_durations(fake_feed: ReplayFeed) -> None:
    assert fake_feed.lap_durations() == pytest.approx([90.0, 90.2, 90.0, 89.8, 92.0, 90.0])
