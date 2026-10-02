from __future__ import annotations

from typing import Any

import pytest

from core.events.detector import DetectorConfig, EventDetector
from core.events.types import Event
from core.models import (
    DriverInfo,
    DriverState,
    FastestLap,
    RaceControlMsg,
    RaceMeta,
    RaceState,
    Weather,
)

CODES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
META = RaceMeta(
    race_id="2099_01",
    season=2099,
    round=1,
    event_name="Test GP",
    country="Nowhere",
    circuit_key="test",
    total_laps=50,
    drivers={
        c: DriverInfo(code=c, number=str(i), name=c, team=f"Team {c}")
        for i, c in enumerate(CODES, start=1)
    },
    grid={c: i for i, c in enumerate(CODES, start=1)},
    results=[],
)


def drv(code: str, pos: int, **kw: Any) -> DriverState:
    base: dict[str, Any] = {
        "code": code,
        "team": f"Team {code}",
        "position": pos,
        "gap_to_leader_s": float(pos - 1) * 2.0,
        "interval_s": None if pos == 1 else 2.0,
        "compound": "MEDIUM",
        "tyre_age": 10,
        "last_lap_s": 90.0,
    }
    return DriverState(**{**base, **kw})


def state(lap: int, drivers: list[DriverState], **kw: Any) -> RaceState:
    flags = kw.pop("flags", ("GREEN",))
    return RaceState(
        race_id="2099_01",
        lap=lap,
        total_laps=50,
        drivers=tuple(drivers),
        flag=kw.pop("flag", flags[-1]),
        flags_this_lap=flags,
        **kw,
    )


def order(lap: int, codes: list[str], **kw: Any) -> RaceState:
    overrides: dict[str, dict[str, Any]] = kw.pop("overrides", {})
    return state(lap, [drv(c, i, **overrides.get(c, {})) for i, c in enumerate(codes, 1)], **kw)


def run(*states: RaceState, config: DetectorConfig | None = None) -> list[Event]:
    return EventDetector(META, config).run(states)


def of_type(events: list[Event], type_: str) -> list[Event]:
    return [e for e in events if e.type == type_]


# ---------------------------------------------------------------- basics
def test_first_tick_only_primes() -> None:
    assert EventDetector(META).process(order(10, CODES)) == []


def test_going_backwards_resets() -> None:
    det = EventDetector(META)
    det.process(order(10, CODES))
    det.process(order(11, CODES))
    assert det.process(order(5, ["BBB", "AAA", *CODES[2:]])) == []  # seek back: no events


def test_event_ids_are_deterministic_and_unique() -> None:
    seq = [order(10, CODES), order(11, ["BBB", "AAA", "DDD", "CCC", "EEE", "FFF"])]
    a, b = run(*seq), run(*seq)
    assert [e.id for e in a] == [e.id for e in b]
    assert len({e.id for e in a}) == len(a) == 2
    assert all(e.summary for e in a)


# ------------------------------------------------------------- overtakes
def test_overtake_for_the_lead() -> None:
    (e,) = run(order(10, CODES), order(11, ["BBB", "AAA", *CODES[2:]]))
    assert e.type == "overtake" and e.drivers == ("BBB", "AAA")
    assert e.teams == ("Team BBB", "Team AAA")
    assert e.payload["for_lead"] is True and e.payload["to_position"] == 1
    assert e.base_importance == pytest.approx(0.7)
    assert "takes the lead" in e.summary


def test_midfield_overtake_is_less_important() -> None:
    (e,) = run(order(10, CODES), order(11, ["AAA", "BBB", "CCC", "DDD", "FFF", "EEE"]))
    assert e.drivers == ("FFF", "EEE") and e.base_importance == pytest.approx(0.45)


def test_lap1_moves_are_overtakes_with_lower_importance() -> None:
    (e,) = run(order(0, CODES), order(1, ["AAA", "BBB", "CCC", "DDD", "FFF", "EEE"]))
    assert e.payload["start"] is True and e.base_importance == pytest.approx(0.3)


def test_position_change_from_pit_stop_is_not_an_overtake() -> None:
    before = order(10, CODES, overrides={"AAA": {"pitted_this_lap": True}})
    after = order(11, ["BBB", "AAA", *CODES[2:]], overrides={"AAA": {"pit_count": 1}})
    assert of_type(run(before, after), "overtake") == []


@pytest.mark.parametrize("flag", ["SC", "VSC", "RED"])
def test_no_overtakes_under_neutralisation(flag: str) -> None:
    after = order(11, ["BBB", "AAA", *CODES[2:]], flags=("GREEN", flag))
    assert of_type(run(order(10, CODES), after), "overtake") == []


def test_lapped_cars_are_ignored() -> None:
    before = order(10, CODES, overrides={"FFF": {"laps_down": 1}, "EEE": {"laps_down": 1}})
    after = order(
        11,
        ["AAA", "BBB", "CCC", "DDD", "FFF", "EEE"],
        overrides={"FFF": {"laps_down": 1}, "EEE": {"laps_down": 1}},
    )
    assert run(before, after) == []


# ------------------------------------------------------------- pit stops
def test_pit_stop_reported_on_out_lap() -> None:
    lap10 = order(10, CODES, overrides={"BBB": {"pitted_this_lap": True, "tyre_age": 22}})
    lap11 = order(
        11,
        ["AAA", "CCC", "DDD", "BBB", "EEE", "FFF"],
        overrides={"BBB": {"pit_count": 1, "compound": "HARD", "tyre_age": 1}},
    )
    (e,) = of_type(run(order(9, CODES), lap10, lap11), "pit_stop")
    assert e.lap == 11 and e.drivers == ("BBB",)
    assert e.payload == {
        "stop_number": 1,
        "from_compound": "MEDIUM",
        "to_compound": "HARD",
        "old_tyre_age": 22,
        "position_before": 2,
        "position_after": 4,
        "under": None,
    }


def test_pit_stop_under_safety_car_names_the_most_severe_flag() -> None:
    lap10 = order(10, CODES, flags=("VSC", "SC"))
    lap11 = order(11, CODES, flags=("SC",), overrides={"CCC": {"pit_count": 1}})
    (e,) = of_type(run(order(9, CODES), lap10, lap11), "pit_stop")
    assert e.payload["under"] == "SC" and "under SC" in e.summary
    assert e.base_importance == pytest.approx(0.35 + 0.2 + 0.1)


def test_red_flag_tyre_changes_are_not_pit_stops() -> None:
    lap10 = order(10, CODES, flags=("SC", "RED"))
    lap11 = order(11, CODES, flags=("RED", "GREEN"), overrides={c: {"pit_count": 1} for c in CODES})
    assert of_type(run(order(9, CODES), lap10, lap11), "pit_stop") == []


# ----------------------------------------------------- undercut/overcut
def _stop(lap: int, codes: list[str], pitted: str, done: dict[str, int], **kw: Any) -> RaceState:
    ov: dict[str, dict[str, Any]] = {c: {"pit_count": n} for c, n in done.items()}
    ov.setdefault(pitted, {})["pitted_this_lap"] = True
    return order(lap, codes, overrides=ov, **kw)


def test_undercut_success() -> None:
    # BBB (P2, 1.5s behind AAA) pits on lap 20; AAA pits on lap 22; BBB comes out ahead.
    o = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    seq = [
        order(19, o),
        _stop(20, o, "BBB", {}, overrides=None)
        if False
        else order(20, o, overrides={"BBB": {"pitted_this_lap": True, "interval_s": 1.5}}),
        order(21, ["AAA", "CCC", "BBB", "DDD", "EEE", "FFF"], overrides={"BBB": {"pit_count": 1}}),
        order(
            22,
            ["AAA", "CCC", "BBB", "DDD", "EEE", "FFF"],
            overrides={"BBB": {"pit_count": 1}, "AAA": {"pitted_this_lap": True}},
        ),
        order(
            23,
            ["CCC", "BBB", "AAA", "DDD", "EEE", "FFF"],
            overrides={"BBB": {"pit_count": 1}, "AAA": {"pit_count": 1}},
        ),
    ]
    (e,) = of_type(run(*seq), "undercut")
    assert e.lap == 23 and e.drivers == ("BBB", "AAA") and e.payload["success"] is True
    assert e.payload["first_stop_lap"] == 20 and e.payload["second_stop_lap"] == 22
    assert e.payload["interval_before_s"] == pytest.approx(1.5)


def test_undercut_failure_and_overcut() -> None:
    o = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    base = [
        order(19, o),
        order(20, o, overrides={"BBB": {"pitted_this_lap": True, "interval_s": 1.5}}),
        order(21, ["AAA", "CCC", "BBB", "DDD", "EEE", "FFF"], overrides={"BBB": {"pit_count": 1}}),
        order(
            22,
            ["AAA", "CCC", "BBB", "DDD", "EEE", "FFF"],
            overrides={"BBB": {"pit_count": 1}, "AAA": {"pitted_this_lap": True}},
        ),
    ]
    stays_ahead = order(
        23,
        ["CCC", "AAA", "BBB", "DDD", "EEE", "FFF"],
        overrides={"BBB": {"pit_count": 1}, "AAA": {"pit_count": 1}},
    )
    (e,) = of_type(run(*base, stays_ahead), "undercut")
    assert e.payload["success"] is False and "fails" in e.summary

    # Mirror: AAA (ahead) stops first and BBB, staying out, jumps it -> overcut.
    o2 = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    seq = [
        order(19, o2),
        order(20, o2, overrides={"AAA": {"pitted_this_lap": True}, "BBB": {"interval_s": 1.0}}),
        order(21, ["BBB", "CCC", "AAA", "DDD", "EEE", "FFF"], overrides={"AAA": {"pit_count": 1}}),
        order(
            22,
            ["BBB", "CCC", "AAA", "DDD", "EEE", "FFF"],
            overrides={"AAA": {"pit_count": 1}, "BBB": {"pitted_this_lap": True}},
        ),
        order(
            23,
            ["CCC", "BBB", "AAA", "DDD", "EEE", "FFF"],
            overrides={"AAA": {"pit_count": 1}, "BBB": {"pit_count": 1}},
        ),
    ]
    (o,) = of_type(run(*seq), "overcut")
    assert o.drivers == ("BBB", "AAA") and o.payload["success"] is True


def test_no_battle_when_cars_are_far_apart() -> None:
    o = CODES
    seq = [
        order(19, o),
        order(
            20,
            o,
            overrides={
                "BBB": {"pitted_this_lap": True, "interval_s": 8.0},
                "CCC": {"interval_s": 8.0},
            },
        ),
        order(21, o, overrides={"BBB": {"pit_count": 1}}),
        order(22, o, overrides={"BBB": {"pit_count": 1}, "AAA": {"pitted_this_lap": True}}),
        order(23, o, overrides={"BBB": {"pit_count": 1}, "AAA": {"pit_count": 1}}),
    ]
    assert of_type(run(*seq), "undercut") == of_type(run(*seq), "overcut") == []


def test_battle_dissolves_if_rival_stops_under_safety_car() -> None:
    o = CODES
    seq = [
        order(19, o),
        order(20, o, overrides={"BBB": {"pitted_this_lap": True, "interval_s": 1.5}}),
        order(21, o, overrides={"BBB": {"pit_count": 1}}),
        order(
            22,
            o,
            flags=("SC",),
            overrides={"BBB": {"pit_count": 1}, "AAA": {"pitted_this_lap": True}},
        ),
        order(
            23,
            ["BBB", "AAA", *CODES[2:]],
            flags=("SC",),
            overrides={"BBB": {"pit_count": 1}, "AAA": {"pit_count": 1}},
        ),
    ]
    assert of_type(run(*seq), "undercut") == []


def test_battle_expires_when_rival_never_stops() -> None:
    o = CODES
    seq = [
        order(19, o),
        order(20, o, overrides={"BBB": {"pitted_this_lap": True, "interval_s": 1.5}}),
    ]
    seq += [order(lap, o, overrides={"BBB": {"pit_count": 1}}) for lap in range(21, 35)]
    seq += [
        order(
            35,
            ["BBB", "AAA", *CODES[2:]],
            overrides={"BBB": {"pit_count": 1}, "AAA": {"pit_count": 1}},
        )
    ]
    assert of_type(run(*seq), "undercut") == []


# ------------------------------------------------------------------ flags
def test_safety_car_deployed_and_ended() -> None:
    events = run(
        order(10, CODES),
        order(11, CODES, flags=("GREEN", "SC")),
        order(12, CODES, flags=("SC",)),
        order(13, CODES, flags=("SC", "GREEN")),
    )
    sc = of_type(events, "safety_car")
    assert [(e.lap, e.payload["phase"]) for e in sc] == [(11, "deployed"), (13, "ended")]
    assert sc[0].needs_regs and sc[0].base_importance == pytest.approx(0.8)


def test_vsc_deployed_and_ended_within_one_lap() -> None:
    events = run(order(10, CODES), order(11, CODES, flags=("GREEN", "VSC", "GREEN")))
    assert [e.payload["phase"] for e in of_type(events, "vsc")] == ["deployed", "ended"]


def test_red_flag_and_resumption() -> None:
    events = run(
        order(10, CODES),
        order(11, CODES, flags=("GREEN", "RED")),
        order(12, CODES, flags=("RED", "GREEN")),
    )
    red = of_type(events, "red_flag")
    assert [(e.lap, e.payload["phase"]) for e in red] == [(11, "deployed"), (12, "ended")]
    assert "Race resumes" in red[1].summary


# --------------------------------------------------- race control messages
def _msg(text: str, drivers: tuple[str, ...] = (), flag: str | None = None) -> RaceControlMsg:
    return RaceControlMsg(lap=11, category="Other", message=text, drivers=drivers, flag=flag)


def _with_msgs(*msgs: RaceControlMsg) -> list[Event]:
    return run(order(10, CODES), order(11, CODES, new_messages=msgs))


def test_time_penalty_parsed() -> None:
    (e,) = _with_msgs(
        _msg("FIA STEWARDS: 10 SECOND TIME PENALTY FOR CAR 2 (BBB) - CAUSING A COLLISION", ("BBB",))
    )
    assert e.type == "penalty" and e.drivers == ("BBB",) and e.needs_regs
    assert e.payload["kind"] == "time" and e.payload["seconds"] == 10
    assert e.payload["reason"] == "CAUSING A COLLISION"
    assert e.base_importance == pytest.approx(0.75)  # P2 is in the top 5


@pytest.mark.parametrize(
    ("text", "kind", "grid"),
    [
        (
            "FIA STEWARDS: DRIVE THROUGH PENALTY FOR CAR 6 (FFF) - SPEEDING IN THE PIT LANE",
            "drive_through",
            None,
        ),
        ("FIA STEWARDS: 10 SECOND STOP/GO PENALTY FOR CAR 6 (FFF)", "stop_go", None),
        ("FIA STEWARDS: 5 PLACE GRID PENALTY FOR CAR 6 (FFF) AT NEXT EVENT", "grid", 5),
    ],
)
def test_other_penalty_kinds(text: str, kind: str, grid: int | None) -> None:
    (e,) = _with_msgs(_msg(text, ("FFF",)))
    assert e.payload["kind"] == kind and e.payload["grid_places"] == grid
    if kind == "stop_go":
        assert e.payload["seconds"] == 10
    assert e.base_importance == pytest.approx(0.6)  # P6 is outside the top 5


def test_served_and_cleared_penalties() -> None:
    events = _with_msgs(
        _msg("FIA STEWARDS: PENALTY SERVED - 5 SECOND TIME PENALTY FOR CAR 2 (BBB)", ("BBB",)),
        _msg(
            "FIA STEWARDS: INCIDENT INVOLVING CAR 2 (BBB) REVIEWED NO FURTHER INVESTIGATION",
            ("BBB",),
        ),
    )
    assert [(e.type, e.payload["kind"]) for e in events] == [("race_control", "cleared")]


def test_investigations_and_flags() -> None:
    events = _with_msgs(
        _msg("CAR 3 (CCC) NOTED - TRACK LIMITS", ("CCC",)),
        _msg("FIA STEWARDS: TURN 1 INCIDENT INVOLVING CAR 3 (CCC) UNDER INVESTIGATION", ("CCC",)),
        _msg(
            "BLACK AND WHITE FLAG FOR CAR 3 (CCC) - TRACK LIMITS", ("CCC",), flag="BLACK AND WHITE"
        ),
        _msg("WAVED BLUE FLAG FOR CAR 6 (FFF)", ("FFF",), flag="BLUE"),  # ignored
    )
    assert [e.payload["kind"] for e in events] == ["noted", "investigation", "black_and_white"]
    assert len({e.id for e in events}) == 3


# ------------------------------------------------------------ fastest lap
def _fl(lap: int, code: str, t: float, total: int = 50, **kw: Any) -> RaceState:
    s = order(lap, CODES, fastest_lap=FastestLap(code=code, lap=lap, time_s=t), **kw)
    return s.model_copy(update={"total_laps": total})


def test_fastest_lap_rules() -> None:
    events = run(
        _fl(1, "AAA", 92.0),
        _fl(2, "BBB", 91.0),  # too early
        _fl(3, "CCC", 90.5),  # holder change -> event
        _fl(4, "CCC", 90.2),  # same holder, early -> no event
        _fl(45, "CCC", 90.1),  # late but tiny gain -> no event
        _fl(46, "CCC", 89.7),  # late and >= 0.3s -> event
        _fl(47, "DDD", 89.6),  # holder change -> event
    )
    fl = of_type(events, "fastest_lap")
    assert [(e.lap, e.drivers[0]) for e in fl] == [(3, "CCC"), (46, "CCC"), (47, "DDD")]
    assert fl[0].payload["previous_holder"] == "BBB"
    assert fl[0].payload["improvement_s"] == pytest.approx(0.5)
    assert fl[2].base_importance == pytest.approx(0.45)


# -------------------------------------------------------------- tyre cliff
def _laps(lap: int, times: dict[str, float], **ov: dict[str, Any]) -> RaceState:
    base = {"CCC": {"interval_s": 5.0}}  # CCC runs in free air unless told otherwise
    overrides = {
        c: {"last_lap_s": t, "tyre_age": lap, **base.get(c, {}), **ov.get(c, {})}
        for c, t in times.items()
    }
    return order(lap, CODES, overrides=overrides)


def _field(ccc: float, field: float = 90.0) -> dict[str, float]:
    return {c: (ccc if c == "CCC" else field) for c in CODES}


def test_tyre_cliff_detected_once_per_stint() -> None:
    seq = [_laps(lap, _field(90.0)) for lap in range(5, 11)]
    seq += [_laps(lap, _field(91.6)) for lap in range(11, 16)]
    (e,) = of_type(run(*seq), "tyre_cliff")
    assert e.drivers == ("CCC",) and e.lap == 13
    assert e.payload["loss_vs_field_s"] == pytest.approx(1.6)
    assert e.payload["tyre_age"] == 13


def test_field_wide_slowdown_is_not_a_cliff() -> None:
    seq = [_laps(lap, _field(90.0)) for lap in range(5, 11)]
    seq += [_laps(lap, _field(92.0, field=92.0)) for lap in range(11, 16)]
    assert of_type(run(*seq), "tyre_cliff") == []


def test_cliff_needs_sustained_loss_and_ignores_incidents() -> None:
    one_slow = [_laps(lap, _field(90.0)) for lap in range(5, 11)]
    one_slow += [_laps(11, _field(92.0)), _laps(12, _field(90.0)), _laps(13, _field(90.0))]
    assert of_type(run(*one_slow), "tyre_cliff") == []
    incident = [_laps(lap, _field(90.0)) for lap in range(5, 11)]
    incident += [_laps(lap, _field(99.0)) for lap in range(11, 15)]
    assert of_type(run(*incident), "tyre_cliff") == []


def test_cliff_ignores_traffic_and_young_tyres() -> None:
    traffic = [_laps(lap, _field(90.0)) for lap in range(5, 11)]
    traffic += [_laps(lap, _field(91.6), CCC={"interval_s": 0.6}) for lap in range(11, 16)]
    assert of_type(run(*traffic), "tyre_cliff") == []
    cfg = DetectorConfig(cliff_min_tyre_age=30)
    seq = [_laps(lap, _field(90.0)) for lap in range(5, 11)]
    seq += [_laps(lap, _field(91.6)) for lap in range(11, 16)]
    assert of_type(run(*seq, config=cfg), "tyre_cliff") == []


# --------------------------------------------------------------------- dnf
def test_dnf() -> None:
    after = order(
        11,
        ["AAA", "BBB", "DDD", "EEE", "FFF", "CCC"],
        overrides={"CCC": {"status": "dnf", "laps_completed": 10}},
    )
    (e,) = run(order(10, CODES), after)
    assert e.type == "dnf" and e.drivers == ("CCC",)
    assert e.payload == {"position_before": 3, "laps_completed": 10}
    assert e.base_importance == pytest.approx(0.75)


# ----------------------------------------------------------------- weather
def _wx(lap: int, rain: bool) -> RaceState:
    return order(lap, CODES, weather=Weather(rainfall=rain, track_temp_c=25.0))


def test_weather_change_is_debounced() -> None:
    flicker = run(_wx(1, False), _wx(2, True), _wx(3, False), _wx(4, False))
    assert of_type(flicker, "weather_change") == []
    events = run(
        _wx(1, False), _wx(2, True), _wx(3, True), _wx(4, True), _wx(5, False), _wx(6, False)
    )
    w = of_type(events, "weather_change")
    assert [(e.lap, e.payload["rainfall"]) for e in w] == [(3, True), (6, False)]
    assert w[0].base_importance == pytest.approx(0.6)
