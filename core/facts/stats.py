"""Computed stat facts. Pure functions over Jolpica race rows, as of BEFORE a given race.

Everything is filtered to races strictly before (season, round), so replaying a past race
never leaks its own result into the facts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.facts.models import Fact
from core.models import Citation
from core.profiles.teams import TEAM_LINEAGES, display_name, team_key

Race = dict[str, Any]


@dataclass(frozen=True)
class RaceContext:
    race_id: str
    season: int
    round: int
    circuit_id: str
    circuit_name: str
    # driver code -> (Ergast driverId, full name, constructor name)
    entrants: dict[str, tuple[str, str, str]]


def _before(races: Iterable[Race], season: int, round_number: int) -> list[Race]:
    return [r for r in races if (int(r["season"]), int(r["round"])) < (season, round_number)]


def _pos(result: dict[str, Any]) -> int | None:
    text = result.get("positionText", "")
    return int(text) if str(text).isdigit() else None


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _years(seasons: list[int]) -> str:
    return ", ".join(str(y) for y in sorted(seasons))


def _cite(label: str, url: str) -> Citation:
    return Citation(source_type="jolpica", label=f"Jolpica F1 API: {label}", url=url)


def _fact(
    ctx: RaceContext,
    kind: str,
    subject_type: str,
    subject: str,
    text: str,
    values: dict[str, Any],
    source: Citation,
    relevance: float,
) -> Fact:
    return Fact(
        id=f"{ctx.race_id}:{kind}:{subject}",
        race_id=ctx.race_id,
        kind=kind,  # type: ignore[arg-type]
        subject_type=subject_type,
        subject=subject,
        text=text,  # type: ignore[arg-type]
        values=values,
        source=source,
        relevance=relevance,
    )


def circuit_facts(ctx: RaceContext, circuit_races: list[Race], url: str) -> list[Fact]:
    """Per-driver circuit record, team wins here, and circuit trivia."""
    past = _before(circuit_races, ctx.season, ctx.round)
    src = _cite(f"all results at {ctx.circuit_name}", url)
    facts: list[Fact] = []
    by_driver: dict[str, list[tuple[int, int | None]]] = {}
    winners: Counter[str] = Counter()
    winner_names: dict[str, str] = {}
    team_wins: Counter[str] = Counter()
    team_win_years: dict[str, list[int]] = {}
    team_win_names: dict[str, set[str]] = {}
    for race in past:
        season = int(race["season"])
        for res in race.get("Results", []):
            drv = res["Driver"]["driverId"]
            by_driver.setdefault(drv, []).append((season, _pos(res)))
            if res.get("position") == "1":
                winners[drv] += 1
                winner_names[drv] = f"{res['Driver']['givenName']} {res['Driver']['familyName']}"
                key = team_key(res["Constructor"]["name"], season)  # era-aware
                team_wins[key] += 1
                team_win_years.setdefault(key, []).append(season)
                team_win_names.setdefault(key, set()).add(res["Constructor"]["name"])

    for code, (driver_id, name, _team) in ctx.entrants.items():
        history = by_driver.get(driver_id, [])
        starts = len(history)
        positions = [p for _, p in history if p is not None]
        wins = [s for s, p in history if p == 1]
        podiums = [p for p in positions if p <= 3]
        if starts == 0:
            facts.append(
                _fact(
                    ctx,
                    "circuit_debut",
                    "driver",
                    code,
                    f"This is {name}'s first World Championship race at {ctx.circuit_name}.",
                    {"starts": 0},
                    src,
                    0.9,
                )
            )
            continue
        best = min(positions) if positions else None
        if wins:
            facts.append(
                _fact(
                    ctx,
                    "circuit_wins",
                    "driver",
                    code,
                    f"{name} has won {len(wins)} time{'s' * (len(wins) > 1)} at "
                    f"{ctx.circuit_name} ({_years(wins)}) in {starts} starts there.",
                    {"wins": len(wins), "starts": starts, "years": _years(wins)},
                    src,
                    1.0,
                )
            )
        elif podiums:
            facts.append(
                _fact(
                    ctx,
                    "circuit_podiums",
                    "driver",
                    code,
                    f"{name} has {len(podiums)} podium finish{'es' * (len(podiums) > 1)} at "
                    f"{ctx.circuit_name} in {starts} starts, but has never won there; best "
                    f"finish P{best}.",
                    {"podiums": len(podiums), "starts": starts, "best": best or 0},
                    src,
                    0.95,
                )
            )
        else:
            best_txt = f"; best finish P{best}" if best else ""
            facts.append(
                _fact(
                    ctx,
                    "circuit_no_podium",
                    "driver",
                    code,
                    f"{name} has never finished on the podium at {ctx.circuit_name} in "
                    f"{starts} starts{best_txt}.",
                    {"starts": starts, "best": best or 0},
                    src,
                    0.9,
                )
            )

    for key in sorted({team_key(t) for _, _, t in ctx.entrants.values()}):
        if key not in TEAM_LINEAGES or not team_wins.get(key):
            continue
        n = team_wins[key]
        current = {t for _, _, t in ctx.entrants.values() if team_key(t) == key}
        earlier = sorted(team_win_names[key] - current)
        as_names = f" (including wins as {', '.join(earlier)})" if earlier else ""
        facts.append(
            _fact(
                ctx,
                "team_circuit_wins",
                "team",
                key,
                f"{display_name(key)} has won {n} time{'s' * (n > 1)} at {ctx.circuit_name}"
                f"{as_names}, most recently in {max(team_win_years[key])}.",
                {"wins": n, "last_win": max(team_win_years[key])},
                src,
                0.85,
            )
        )

    if not past:
        facts.append(
            _fact(
                ctx,
                "circuit_first_race",
                "circuit",
                ctx.circuit_id,
                f"This is the first World Championship race ever held at {ctx.circuit_name}.",
                {"races": 0},
                src,
                0.8,
            )
        )
    else:
        first = min(int(r["season"]) for r in past)
        facts.append(
            _fact(
                ctx,
                "circuit_first_race",
                "circuit",
                ctx.circuit_id,
                f"{ctx.circuit_name} first hosted a World Championship race in {first}; this is "
                f"its {_ordinal(len(past) + 1)} championship race.",
                {"first_year": first, "races": len(past)},
                src,
                0.6,
            )
        )
    if winners:
        top, n = winners.most_common(1)[0]
        tied = [d for d, c in winners.items() if c == n]
        if len(tied) == 1:
            facts.append(
                _fact(
                    ctx,
                    "circuit_most_wins",
                    "circuit",
                    ctx.circuit_id,
                    f"The most successful driver at {ctx.circuit_name} is {winner_names[top]} "
                    f"with {n} wins.",
                    {"wins": n},
                    src,
                    0.6,
                )
            )
    return facts


def career_win_facts(
    ctx: RaceContext, wins_by_driver: dict[str, list[Race]], urls: dict[str, str]
) -> list[Fact]:
    facts = []
    for code, (driver_id, name, _team) in ctx.entrants.items():
        if driver_id not in wins_by_driver:
            continue
        n = len(_before(wins_by_driver[driver_id], ctx.season, ctx.round))
        src = _cite(f"race wins of {name}", urls[driver_id])
        if n == 0:
            text = f"{name} is still chasing a first Grand Prix win."
        else:
            text = f"{name} goes into this race with {n} career Grand Prix win{'s' * (n > 1)}."
        facts.append(_fact(ctx, "career_wins", "driver", code, text, {"wins": n}, src, 0.6))
    return facts


def standings_facts(ctx: RaceContext, standings: list[dict[str, Any]], url: str) -> list[Fact]:
    """Championship position going into this race (standings after the previous round)."""
    if ctx.round <= 1 or not standings:
        return []
    by_id = {s["Driver"]["driverId"]: s for s in standings}
    src = _cite(f"{ctx.season} driver standings after round {ctx.round - 1}", url)
    facts = []
    for code, (driver_id, name, _team) in ctx.entrants.items():
        s = by_id.get(driver_id)
        if s is None or not str(s.get("position", "")).isdigit():
            continue
        pos, pts = int(s["position"]), float(s["points"])
        pts_txt = f"{pts:g}"
        facts.append(
            _fact(
                ctx,
                "championship_position",
                "driver",
                code,
                f"{name} arrives {_ordinal(pos)} in the {ctx.season} championship on "
                f"{pts_txt} points.",
                {"position": pos, "points": pts},
                src,
                0.7,
            )
        )
    return facts


def teammate_facts(ctx: RaceContext, season_races: list[Race], url: str) -> list[Fact]:
    """Head-to-head with the current teammate in earlier races this season (both classified)."""
    past = _before(season_races, ctx.season, ctx.round)
    by_team: dict[str, list[str]] = {}
    for code, (_id, _name, team) in ctx.entrants.items():
        by_team.setdefault(team_key(team), []).append(code)
    src = _cite(f"{ctx.season} race results", url)
    facts = []
    for codes in by_team.values():
        if len(codes) != 2:
            continue
        a, b = codes
        ida, idb = ctx.entrants[a][0], ctx.entrants[b][0]
        ahead = {a: 0, b: 0}
        both = 0
        for race in past:
            pos = {r["Driver"]["driverId"]: _pos(r) for r in race.get("Results", [])}
            pa, pb = pos.get(ida), pos.get(idb)
            if pa is None or pb is None:
                continue
            both += 1
            ahead[a if pa < pb else b] += 1
        if both == 0:
            continue
        for me, mate in ((a, b), (b, a)):
            name, mate_name = ctx.entrants[me][1], ctx.entrants[mate][1]
            facts.append(
                _fact(
                    ctx,
                    "teammate_h2h",
                    "driver",
                    me,
                    f"{name} has finished ahead of teammate {mate_name} in {ahead[me]} of "
                    f"{both} races this season where both were classified.",
                    {"ahead": ahead[me], "races": both},
                    src,
                    0.8,
                )
            )
    return facts
