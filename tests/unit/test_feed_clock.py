from __future__ import annotations

import pytest

from core.feed.clock import ReplayClock

LAPS = [90.0, 90.0, 90.0, 90.0]


def test_paused_until_started() -> None:
    c = ReplayClock(LAPS, speed=5)
    assert not c.running
    assert c.lap_at(1_000.0) == 0
    assert c.seconds_to_next_lap(0.0) is None


@pytest.mark.parametrize(("speed", "wall_per_lap"), [(1, 90.0), (5, 18.0), (20, 4.5)])
def test_speeds(speed: int, wall_per_lap: float) -> None:
    c = ReplayClock(LAPS, speed=speed)  # type: ignore[arg-type]
    c.start(100.0)
    assert c.lap_at(100.0 + wall_per_lap - 0.01) == 0
    assert c.lap_at(100.0 + wall_per_lap) == 1
    assert c.lap_at(100.0 + 2.5 * wall_per_lap) == 2
    assert c.lap_at(1e9) == 4  # clamped at the flag
    assert c.seconds_to_next_lap(100.0) == pytest.approx(wall_per_lap)


def test_pause_and_resume() -> None:
    c = ReplayClock(LAPS, speed=5)
    c.start(0.0)
    c.pause(20.0)  # 100 race-seconds in -> lap 1 done
    assert c.lap_at(500.0) == 1
    c.start(500.0)
    assert c.lap_at(516.0) == 2  # 100 + 16*5 = 180


def test_speed_change_keeps_position() -> None:
    c = ReplayClock(LAPS, speed=1)
    c.start(0.0)
    c.set_speed(20, 100.0)  # 100 race-seconds at 1x
    assert c.lap_at(100.0) == 1
    assert c.lap_at(104.0) == 2  # +80 race-seconds -> 180
    c.set_speed("step", 104.0)
    assert c.lap_at(10_000.0) == 2  # stepping does not advance with time


def test_step_mode() -> None:
    c = ReplayClock(LAPS, speed="step")
    c.start(0.0)  # no-op in step mode
    assert not c.running
    assert c.step(0.0) == 1
    assert c.step(0.0, laps=2) == 3
    assert c.step(0.0, laps=5) == 4


def test_step_pauses_a_running_clock() -> None:
    c = ReplayClock(LAPS, speed=5)
    c.start(0.0)
    assert c.step(20.0) == 2  # was in lap 1 (100 race-s), jumps to end of lap 2
    assert not c.running
    assert c.lap_at(1_000.0) == 2


def test_seek_keeps_running_state() -> None:
    c = ReplayClock(LAPS, speed=5)
    c.start(0.0)
    c.seek(3, 50.0)
    assert c.lap_at(50.0) == 3
    assert c.lap_at(68.0) == 4
    c.pause(68.0)
    c.seek(1, 70.0)
    assert c.lap_at(1_000.0) == 1


def test_red_flag_is_capped() -> None:
    c = ReplayClock([90.0, 1500.0, 90.0], speed=1, max_lap_s=180.0)
    c.start(0.0)
    assert c.lap_at(90.0 + 180.0) == 2


def test_invalid_speed() -> None:
    with pytest.raises(ValueError):
        ReplayClock(LAPS, speed=3)  # type: ignore[arg-type]
    c = ReplayClock(LAPS)
    with pytest.raises(ValueError):
        c.set_speed(2, 0.0)  # type: ignore[arg-type]
