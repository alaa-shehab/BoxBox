from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.feed.data import ReplayData
from core.feed.replay import ReplayFeed
from core.sim.degradation import (
    FUEL_EFFECT_S_PER_KG,
    FUEL_START_KG,
    DegFit,
    blend,
    fit_degradation,
    fuel_corrected,
)
from core.sim.history import RaceSummary, deg_prior, summarise, track_params
from core.sim.inputs import build_sim_input
from core.sim.params import TrackParams


def synthetic(fake_replay: ReplayData, slope: float = 0.08) -> ReplayData:
    """Two drivers x two 15-lap stints whose times follow exactly 90 + slope*age + fuel."""
    total, rows = 30, []
    for drv, base in (("AAA", 90.0), ("BBB", 90.5)):
        t = 1000.0
        for lap in range(1, total + 1):
            stint = 1 if lap <= 15 else 2
            age = lap if stint == 1 else lap - 15
            fuel = FUEL_EFFECT_S_PER_KG * FUEL_START_KG * (1 - (lap - 1) / total)
            lap_t = base + slope * age + fuel
            t += lap_t
            rows.append(
                {
                    "driver": drv,
                    "lap": lap,
                    "time_s": t,
                    "lap_time_s": lap_t,
                    "position": 1.0,
                    "compound": "MEDIUM" if stint == 1 else "HARD",
                    "tyre_life": float(age),
                    "stint": float(stint),
                    "pit_in": lap == 15,
                    "pit_out": lap == 16,
                    "deleted": False,
                }
            )
    meta = fake_replay.meta.model_copy(update={"total_laps": total})
    green = pd.DataFrame({"time_s": [0.0], "flag": ["GREEN"]})
    dry = pd.DataFrame(
        {
            "time_s": [0.0],
            "air_temp_c": [20.0],
            "track_temp_c": [30.0],
            "humidity_pct": [50.0],
            "rainfall": [False],
        }
    )
    return ReplayData(meta, pd.DataFrame(rows), fake_replay.messages.iloc[:0], green, dry)


def test_fuel_correction() -> None:
    out = fuel_corrected(np.array([93.0, 90.0]), np.array([1, 51]), 51)
    assert out[0] == pytest.approx(93.0 - 3.0)
    assert out[1] == pytest.approx(90.0 - 0.03 * 100 * (1 - 50 / 51))


def test_degradation_recovers_known_slope(fake_replay: ReplayData) -> None:
    fit = fit_degradation(synthetic(fake_replay, slope=0.08))
    assert fit.by_compound["MEDIUM"].slope_s_per_lap == pytest.approx(0.08, abs=1e-3)
    assert fit.by_compound["HARD"].slope_s_per_lap == pytest.approx(0.08, abs=1e-3)
    assert fit.by_compound["MEDIUM"].n_stints == 2
    assert fit.slope("SOFT") == 0.10  # no data: documented default


def test_degfit_clamps_and_roundtrips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from core.sim.degradation import CompoundDeg

    fit = DegFit("x", 2024, {"INTERMEDIATE": CompoundDeg(-0.05, 100, 5, 0.01)})
    assert fit.slope("INTERMEDIATE") == 0.0  # drying track -> never negative wear
    fit.save(tmp_path / "f.json")
    assert DegFit.load(tmp_path / "f.json") == fit


def test_blend() -> None:
    assert blend(0.1, None, 0) == 0.1
    assert blend(0.1, 0.2, 40, prior_weight=40) == pytest.approx(0.15)


def test_build_sim_input_from_feed(fake_feed: ReplayFeed) -> None:
    states = [fake_feed.state(lap) for lap in range(5)]
    inp = build_sim_input(fake_feed.meta, states, TrackParams("sao_paulo"))
    codes = [d.code for d in inp.drivers]
    assert codes == ["BBB", "AAA", "DDD"]  # CCC retired, EEE never started
    assert inp.lap == 4 and inp.neutralised == "SC" and inp.is_wet is False
    ddd = next(d for d in inp.drivers if d.code == "DDD")
    assert ddd.gap_s > 90.0  # one lap down -> includes a reference lap
    aaa = next(d for d in inp.drivers if d.code == "AAA")
    assert aaa.compounds_used == frozenset({"MEDIUM"})
    grid = build_sim_input(fake_feed.meta, states[:1], TrackParams("x"))
    assert [d.gap_s for d in grid.drivers] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError):
        build_sim_input(fake_feed.meta, [], TrackParams("x"))


def _row(
    circuit: str,
    season: int,
    sc: bool,
    passes: int = 30,
    wet: bool = False,
    deg: dict | None = None,
) -> RaceSummary:
    return RaceSummary(
        race_id=f"{season}_{circuit}",
        circuit_key=circuit,
        season=season,
        round=1,
        total_laps=50,
        is_wet=wet,
        had_sc=sc,
        had_vsc=False,
        had_red=False,
        pit_loss_s=20.0 if circuit == "a" else 25.0,
        passes=passes,
        dnfs=1,
        car_laps=1000,
        median_lap_s=90.0,
        deg=deg or {"MEDIUM": 0.05},
    )


def test_track_params_shrink_and_never_use_the_future() -> None:
    rows = [
        _row("a", 2021, True),
        _row("a", 2022, True),
        _row("b", 2021, False),
        _row("b", 2022, False),
        _row("a", 2024, False, deg={"MEDIUM": 0.3}),
    ]
    p = track_params("a", 2023, rows)
    # global SC rate 0.5, local [1, 1]; shrunk with k=4: (4*0.5 + 2) / 6
    assert p.sc_prob == pytest.approx(round((4 * 0.5 + 2) / 6, 3))
    assert p.pit_loss_s == 20.0 and p.n_races == 2
    assert p.dnf_per_car_lap == pytest.approx(4 / 4000)
    assert deg_prior("a", 2023, rows)["MEDIUM"] == 0.05  # the 2024 row is ignored
    assert track_params("zzz", 2021, rows) == TrackParams("zzz")  # no earlier seasons


def test_hard_to_pass_tracks_get_lower_pass_prob() -> None:
    rows = [_row("monaco", 2021, False, passes=2), _row("monza", 2021, False, passes=60)]
    assert (
        track_params("monaco", 2022, rows).pass_prob < track_params("monza", 2022, rows).pass_prob
    )


def test_summarise_fake_race(fake_replay: ReplayData) -> None:
    s = summarise(fake_replay)
    assert (s.had_sc, s.had_vsc, s.had_red) == (True, False, False)
    assert s.dnfs == 1 and s.passes == 1 and s.is_wet
