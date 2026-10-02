from __future__ import annotations

from typing import Any

from core.facts.stats import (
    RaceContext,
    career_win_facts,
    circuit_facts,
    standings_facts,
    teammate_facts,
)
from core.profiles.teams import team_key


def res(
    driver: str, pos: str, team: str = "Ferrari", given: str = "A", family: str | None = None
) -> dict[str, Any]:
    return {
        "positionText": pos,
        "position": pos if pos.isdigit() else "",
        "Driver": {"driverId": driver, "givenName": given, "familyName": family or driver},
        "Constructor": {"name": team},
    }


def race(season: int, rnd: int, *results: dict[str, Any]) -> dict[str, Any]:
    return {"season": str(season), "round": str(rnd), "Results": list(results)}


CTX = RaceContext(
    race_id="2024_21",
    season=2024,
    round=21,
    circuit_id="interlagos",
    circuit_name="Interlagos",
    entrants={
        "AAA": ("aaa", "Anna A", "Ferrari"),
        "BBB": ("bbb", "Ben B", "Ferrari"),
        "CCC": ("ccc", "Cleo C", "Toro Rosso"),
        "NEW": ("new", "Nia New", "Williams"),
    },
)

CIRCUIT = [
    race(1950, 9, res("old", "1", "Alfa Romeo", "Nino", "Old")),
    race(2008, 18, res("aaa", "1", "Ferrari"), res("bbb", "2"), res("ccc", "R", "Toro Rosso")),
    race(2015, 19, res("ccc", "1", "Toro Rosso", "Cleo", "C"), res("aaa", "3"), res("bbb", "5")),
    race(2023, 20, res("aaa", "1", "Ferrari"), res("bbb", "4"), res("ccc", "7", "Toro Rosso")),
    # The replayed race itself and later races must never count:
    race(2024, 21, res("bbb", "1"), res("new", "2", "Williams")),
    race(2025, 21, res("new", "1", "Williams")),
]


def by(facts: list, kind: str, subject: str):  # type: ignore[no-untyped-def]
    return next(f for f in facts if f.kind == kind and f.subject == subject)


def test_circuit_record_per_driver_excludes_this_and_future_races() -> None:
    facts = circuit_facts(CTX, CIRCUIT, "https://api.example/circuit")
    aaa = by(facts, "circuit_wins", "AAA")
    assert aaa.values == {"wins": 2, "starts": 3, "years": "2008, 2023"}
    assert aaa.text == "Anna A has won 2 times at Interlagos (2008, 2023) in 3 starts there."
    bbb = by(facts, "circuit_podiums", "BBB")
    assert bbb.values == {"podiums": 1, "starts": 3, "best": 2}  # 2024 win not counted
    assert by(facts, "circuit_debut", "NEW").values == {"starts": 0}
    assert all(f.source.url == "https://api.example/circuit" for f in facts)


def test_no_podium_fact() -> None:
    ctx = RaceContext("r", 2024, 21, "x", "X", {"BBB": ("bbb", "Ben B", "Ferrari")})
    circuit = [race(2020, 1, res("bbb", "6")), race(2021, 1, res("bbb", "R"))]
    fact = circuit_facts(ctx, circuit, "u")[0]
    assert fact.kind == "circuit_no_podium"
    assert fact.text == "Ben B has never finished on the podium at X in 2 starts; best finish P6."


def test_team_wins_respect_lineage_eras() -> None:
    facts = circuit_facts(CTX, CIRCUIT, "u")
    ferrari = by(facts, "team_circuit_wins", "ferrari")
    assert ferrari.values == {"wins": 2, "last_win": 2023}
    rb = by(facts, "team_circuit_wins", team_key("Toro Rosso"))
    assert rb.text.endswith("most recently in 2015.")
    # 1950 "Alfa Romeo" is not Sauber's 2019-23 Alfa Romeo: no sauber fact at all.
    assert not [f for f in facts if f.subject == "sauber"]


def test_circuit_trivia() -> None:
    facts = circuit_facts(CTX, CIRCUIT, "u")
    first = by(facts, "circuit_first_race", "interlagos")
    assert first.values == {"first_year": 1950, "races": 4}
    assert "5th championship race" in first.text
    most = by(facts, "circuit_most_wins", "interlagos")
    assert most.text == "The most successful driver at Interlagos is A aaa with 2 wins."


def test_inaugural_race() -> None:
    fact = next(f for f in circuit_facts(CTX, [], "u") if f.kind == "circuit_first_race")
    assert fact.text == "This is the first World Championship race ever held at Interlagos."


def test_career_wins_before_race() -> None:
    wins = {"aaa": [race(2023, 5), race(2024, 21), race(2024, 22)], "new": []}
    facts = {f.subject: f for f in career_win_facts(CTX, wins, {"aaa": "u1", "new": "u2"})}
    assert facts["AAA"].values == {"wins": 1}
    assert facts["AAA"].text == "Anna A goes into this race with 1 career Grand Prix win."
    assert facts["NEW"].text == "Nia New is still chasing a first Grand Prix win."


def test_standings_going_into_race() -> None:
    standings = [
        {"position": "2", "points": "315", "Driver": {"driverId": "aaa"}},
        {"position": "-", "points": "0", "Driver": {"driverId": "bbb"}},
    ]
    (fact,) = standings_facts(CTX, standings, "u")
    assert fact.text == "Anna A arrives 2nd in the 2024 championship on 315 points."
    first_round = RaceContext("r", 2024, 1, "x", "X", CTX.entrants)
    assert standings_facts(first_round, standings, "u") == []


def test_teammate_h2h_counts_only_races_both_finished() -> None:
    season = [
        race(2024, 1, res("aaa", "1"), res("bbb", "3")),
        race(2024, 2, res("aaa", "R"), res("bbb", "2")),  # AAA DNF: excluded
        race(2024, 3, res("aaa", "5"), res("bbb", "4")),
        race(2024, 21, res("bbb", "1"), res("aaa", "2")),  # the replayed race: excluded
    ]
    facts = {f.subject: f for f in teammate_facts(CTX, season, "u")}
    assert facts["AAA"].values == {"ahead": 1, "races": 2}
    assert facts["BBB"].text == (
        "Ben B has finished ahead of teammate Anna A in 1 of 2 races "
        "this season where both were classified."
    )
    assert "CCC" not in facts  # no teammate in the entry list
