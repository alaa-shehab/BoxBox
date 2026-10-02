"""Team identity across seasons and rebrands.

A fan supports a team, not one season's entry name: "AlphaTauri" (2023), "RB" (2024) and
"Racing Bulls" (2025) are the same Faenza team. All favourite-team matching goes through
`team_key`, so favourites keep working on any replayed season.

Historical names only count for a lineage within the years that team actually used them.
The 1950 "Alfa Romeo" works team is not Sauber, and the 1977-85 "Renault" works team is
not today's Alpine. `team_key(name, season)` respects those eras; without a season it
maps by name alone, which is right for the 2018+ entry lists the replays use.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from core.models import DriverInfo

ANY = (0, 9999)

# canonical key -> display name, and {entry name: (first season, last season)}
TEAM_LINEAGES: dict[str, tuple[str, dict[str, tuple[int, int]]]] = {
    "red_bull": (
        "Red Bull Racing",
        {
            "Red Bull Racing": (2005, 9999),
            "Red Bull": (2005, 9999),
            "Oracle Red Bull Racing": (2005, 9999),
        },
    ),
    "ferrari": ("Ferrari", {"Ferrari": ANY, "Scuderia Ferrari": ANY}),
    "mercedes": ("Mercedes", {"Mercedes": (2010, 9999), "Mercedes-AMG Petronas": (2010, 9999)}),
    "mclaren": ("McLaren", {"McLaren": ANY}),
    "aston_martin": (
        "Aston Martin",
        {"Aston Martin": (2021, 9999), "Racing Point": (2019, 2020), "Force India": (2008, 2018)},
    ),
    "alpine": (
        "Alpine",
        {"Alpine": (2021, 9999), "Alpine F1 Team": (2021, 9999), "Renault": (2016, 2020)},
    ),
    "williams": ("Williams", {"Williams": (1977, 9999)}),
    "racing_bulls": (
        "Racing Bulls",
        {
            "Racing Bulls": (2025, 9999),
            "RB": (2024, 2024),
            "RB F1 Team": (2024, 2024),
            "Visa Cash App RB": (2024, 2024),
            "AlphaTauri": (2020, 2023),
            "Toro Rosso": (2006, 2019),
        },
    ),
    "haas": ("Haas F1 Team", {"Haas F1 Team": (2016, 9999), "Haas": (2016, 9999)}),
    "sauber": (
        "Kick Sauber",
        {
            "Kick Sauber": (2024, 2025),
            "Sauber": (1993, 2025),
            "Alfa Romeo": (2019, 2023),
            "Alfa Romeo Racing": (2019, 2023),
            "BMW Sauber": (2006, 2010),
            "Audi": (2026, 9999),
        },
    ),
    "cadillac": ("Cadillac", {"Cadillac": (2026, 9999)}),
}


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


_ALIASES: dict[str, list[tuple[str, int, int]]] = {}
for _key, (_display, _names) in TEAM_LINEAGES.items():
    for _name, (_from, _to) in _names.items():
        _ALIASES.setdefault(_compact(_name), []).append((_key, _from, _to))


def team_key(name: str, season: int | None = None) -> str:
    """Canonical lineage key for a team name (within its era if `season` is given).

    Unknown names, or names used outside a lineage's era, fall back to a plain slug, so
    they never match a current team."""
    compact = _compact(name)
    for key, first, last in _ALIASES.get(compact, []):
        if season is None or first <= season <= last:
            return key
    return compact if season is None else f"{compact}@{season}"


def display_name(key_or_name: str) -> str:
    key = team_key(key_or_name)
    return TEAM_LINEAGES[key][0] if key in TEAM_LINEAGES else key_or_name


def same_team(a: str, b: str) -> bool:
    return bool(a) and bool(b) and team_key(a) == team_key(b)


def roster(drivers: Mapping[str, DriverInfo]) -> dict[str, list[DriverInfo]]:
    """Team display name -> drivers for one race's entry list (for onboarding choices)."""
    out: dict[str, list[DriverInfo]] = {}
    for d in sorted(drivers.values(), key=lambda d: (display_name(d.team), d.code)):
        out.setdefault(display_name(d.team), []).append(d)
    return out
