from __future__ import annotations

import time

import numpy as np
import pytest

from core.sim.model import _hazard, simulate
from core.sim.params import Scenario, SimDriver, SimInput, TrackParams

QUIET = TrackParams(
    "test", pit_loss_s=22.0, sc_prob=0.0, vsc_prob=0.0, dnf_per_car_lap=0.0, pass_prob=0.35
)


def drv(
    code: str,
    pos: int,
    gap: float,
    pace: float = 0.0,
    compound: str = "HARD",
    age: int = 5,
    used: tuple[str, ...] = ("MEDIUM", "HARD"),
) -> SimDriver:
    return SimDriver(
        code=code,
        position=pos,
        gap_s=gap,
        compound=compound,
        tyre_age=age,
        compounds_used=frozenset(used),
        pace_s=pace,
    )


def inp(
    *drivers: SimDriver, lap: int = 40, total: int = 50, track: TrackParams = QUIET, **kw: object
) -> SimInput:
    return SimInput(
        race_id="t",
        lap=lap,
        total_laps=total,
        drivers=drivers,
        track=track,
        deg={"HARD": 0.0, "MEDIUM": 0.0, "SOFT": 0.0},
        **kw,
    )  # type: ignore[arg-type]


def test_deterministic_for_a_seed() -> None:
    i = inp(drv("A", 1, 0.0), drv("B", 2, 1.0), drv("C", 3, 2.0))
    a, b = simulate(i, 500, seed=3), simulate(i, 500, seed=3)
    assert np.array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions, simulate(i, 500, seed=4).positions)


def test_much_faster_leader_always_wins() -> None:
    r = simulate(inp(drv("A", 1, 0.0, pace=-2.0), drv("B", 2, 1.0)), 500)
    assert r.p_win == {"A": 1.0, "B": 0.0}


def test_big_lead_holds_at_the_end() -> None:
    # 2 laps left, 100 s ahead, 1 s/lap slower: cannot be caught.
    r = simulate(inp(drv("A", 1, 0.0, pace=1.0), drv("B", 2, 100.0), lap=48), 500)
    assert r.p_win["A"] == 1.0


def test_overtaking_difficulty_controls_passes() -> None:
    pair = (drv("A", 1, 0.0), drv("B", 2, 0.5, pace=-0.6))
    no_pass = simulate(inp(*pair, track=TrackParams("t", 22, 0, 0, 0, pass_prob=0.0)), 500)
    assert no_pass.p_win["A"] == 1.0  # B is faster but can never get by
    easy = simulate(inp(*pair, track=TrackParams("t", 22, 0, 0, 0, pass_prob=1.0)), 500)
    assert easy.p_win["B"] > 0.9


def test_forced_dnf_scenario() -> None:
    i = inp(drv("A", 1, 0.0, pace=-1.0), drv("B", 2, 5.0), drv("C", 3, 10.0))
    r = simulate(i, 300, scenario=Scenario("dnf", dnf=frozenset({"A"})))
    assert r.p_win["A"] == 0.0 and (r.positions[:, 0] == 3).all()
    assert r.scenario == "dnf"


def test_two_compound_rule_forces_a_stop() -> None:
    # A leads by 5 s but has only used HARD: it must stop (22 s) before the flag. Even on
    # fresh softs (1.0 s/lap faster than B's hards) it gains at most 9 s over the 9 laps
    # left, so it rejoins ~17 s behind and B wins.
    a = drv("A", 1, 0.0, used=("HARD",), age=40)
    b = drv("B", 2, 5.0)
    r = simulate(inp(a, b, lap=40, total=50), 500)
    assert r.p_win["B"] > 0.95
    # In a wet race the rule doesn't apply.
    wet = simulate(inp(a, b, lap=40, total=50, is_wet=True), 500)
    assert wet.p_win["A"] > 0.85  # no forced stop: A stays a clear favourite


def test_forced_safety_car_bunches_the_field() -> None:
    a = drv("A", 1, 0.0)
    b = drv("B", 2, 15.0, pace=-0.8)  # faster but 15 s behind with 10 laps left
    base = simulate(inp(a, b, track=TrackParams("t", 22, 0, 0, 0, pass_prob=0.5)), 1000)
    sc = simulate(
        inp(a, b, track=TrackParams("t", 22, 0, 0, 0, pass_prob=0.5)),
        1000,
        scenario=Scenario("sc", sc_in_laps=(41, 42)),
    )
    assert base.p_win["B"] < 0.05
    assert sc.p_win["B"] > base.p_win["B"] + 0.2


def test_forced_pit_now_costs_the_pit_loss() -> None:
    # 5 laps left: A stops now (-22 s) and rejoins 10 s behind B; fresh softs gain ~1 s/lap
    # over 4 laps, so A finishes ~6 s behind, which is several standard deviations.
    i = inp(drv("A", 1, 0.0), drv("B", 2, 12.0), lap=45, total=50)
    r = simulate(i, 300, scenario=Scenario("pit", pit_now=frozenset({"A"})))
    assert r.p_win["B"] > 0.97
    assert simulate(i, 300).p_win["A"] == 1.0  # without the stop A just wins


def test_probabilities_are_consistent() -> None:
    i = inp(
        *[drv(c, n, n * 1.5) for n, c in enumerate("ABCDE", start=1)],
        track=TrackParams("t", 22, 0.5, 0.3, 0.001, 0.35),
    )
    r = simulate(i, 2000)
    assert sum(r.p_win.values()) == pytest.approx(1.0)
    assert sum(r.p_top3.values()) == pytest.approx(3.0)
    exact = r.p_exact_podium("A", "B", "C")
    assert 0 < exact <= r.p_podium_any_order(("A", "B", "C")) <= r.p_top3["A"]
    assert r.p_exact_podium("A", "B", "ZZZ") == 0.0
    assert sorted(set(r.positions.flatten().tolist())) == [1, 2, 3, 4, 5]
    assert r.expected_position["A"] < r.expected_position["E"]


def test_2000_sims_of_a_full_race_in_under_a_second() -> None:
    drivers = [
        drv(f"D{i:02d}", i, i * 1.2, pace=0.05 * i, compound="MEDIUM", age=1, used=("MEDIUM",))
        for i in range(1, 21)
    ]
    i = inp(*drivers, lap=0, total=70, track=TrackParams("t", 22, 0.5, 0.4, 0.0008, 0.35))
    start = time.perf_counter()
    r = simulate(i, 2000)
    assert time.perf_counter() - start < 1.0
    assert r.n_sims == 2000 and r.positions.shape == (2000, 20)


def test_hazard() -> None:
    h = _hazard(0.5, 50)
    assert 1 - (1 - h) ** 50 == pytest.approx(0.5)
    assert _hazard(0.0, 50) == 0.0


def test_remaining_stops_is_cost_minimising() -> None:
    from core.sim.model import _remaining_stops

    # 57 laps, deg 0.07+evo: 0 stops = 113.7 s, 1 stop = 56.9 + 36 = 92.9, 2 = 37.9 + 72.
    assert _remaining_stops(np.array([0.07]), np.array([57.0]), np.array([36.0]))[0] == 1
    # Monaco-like: low deg, long horizon, expensive track position -> stay out.
    assert _remaining_stops(np.array([0.02]), np.array([77.0]), np.array([36.0]))[0] == 0
    # Very high degradation -> two stops.
    assert _remaining_stops(np.array([0.3]), np.array([60.0]), np.array([30.0]))[0] == 2


def test_low_deg_hard_to_pass_race_has_no_extra_stops() -> None:
    track = TrackParams(
        "monaco", pit_loss_s=20, sc_prob=0, vsc_prob=0, dnf_per_car_lap=0, pass_prob=0.05
    )
    cars = [
        drv(c, i, i * 2.0, compound="HARD", age=30, used=("MEDIUM", "HARD"))
        for i, c in enumerate("ABCD", start=1)
    ]
    i = SimInput(
        race_id="t", lap=30, total_laps=78, drivers=tuple(cars), track=track, deg={"HARD": 0.01}
    )
    r = simulate(i, 300)
    assert r.stops is not None and r.stops.mean() < 0.05


def test_equal_pace_cars_do_not_swap_on_noise() -> None:
    # Same systematic pace, 0.2 s apart, passing "easy": lap noise alone must not
    # produce passes.
    from dataclasses import replace

    easy = TrackParams("t", 22, 0, 0, 0, pass_prob=1.0)
    a, b = (replace(d, pace_sd_s=0.0) for d in (drv("A", 1, 0.0), drv("B", 2, 0.2)))
    r = simulate(inp(a, b, track=easy), 500)
    assert r.p_win["A"] == 1.0
