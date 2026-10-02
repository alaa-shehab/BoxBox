"""Team identity across seasons and rebrands.

A fan supports a team, not one season's entry name: "AlphaTauri" (2023), "RB" (2024) and
"Racing Bulls" (2025) are the same Faenza team. All favourite-team matching goes through
`team_key`, so favourites keep working on any replayed season.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from core.models import DriverInfo

# canonical key -> display name, with every known entry name for that team lineage.
TEAM_LINEAGES: dict[str, tuple[str, tuple[str, ...]]] = {
    "red_bull": ("Red Bull Racing", ("Red Bull Racing", "Red Bull", "Oracle Red Bull Racing")),
    "ferrari": ("Ferrari", ("Ferrari", "Scuderia Ferrari")),
    "mercedes": ("Mercedes", ("Mercedes", "Mercedes-AMG Petronas")),
    "mclaren": ("McLaren", ("McLaren",)),
    "aston_martin": ("Aston Martin", ("Aston Martin", "Racing Point", "Force India")),
    "alpine": ("Alpine", ("Alpine", "Renault")),
    "williams": ("Williams", ("Williams",)),
    "racing_bulls": (
        "Racing Bulls",
        ("Racing Bulls", "RB", "Visa Cash App RB", "AlphaTauri", "Toro Rosso"),
    ),
    "haas": ("Haas F1 Team", ("Haas F1 Team", "Haas")),
    "sauber": ("Kick Sauber", ("Kick Sauber", "Sauber", "Alfa Romeo", "Alfa Romeo Racing", "Audi")),
    "cadillac": ("Cadillac", ("Cadillac",)),
}

_ALIASES = {
    re.sub(r"[^a-z0-9]", "", name.casefold()): key
    for key, (_, names) in TEAM_LINEAGES.items()
    for name in names
}


def team_key(name: str) -> str:
    """Canonical lineage key for a team name; unknown names fall back to a slug."""
    compact = re.sub(r"[^a-z0-9]", "", name.casefold())
    return _ALIASES.get(compact, compact)


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
