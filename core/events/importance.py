"""Per-fan importance: boost events that involve the fan's favourite drivers or team."""

from __future__ import annotations

from collections.abc import Iterable

from core.events.types import Event

DRIVER_BOOST = 1.75
TEAM_BOOST = 1.4


def personalise(
    events: Iterable[Event], favourite_team: str | None, favourite_drivers: Iterable[str] = ()
) -> list[Event]:
    """Return copies with `importance` boosted for favourites, most important first."""
    drivers = {d.upper() for d in favourite_drivers}
    team = (favourite_team or "").casefold()
    out = []
    for e in events:
        if drivers & set(e.drivers):
            factor = DRIVER_BOOST
        elif team and any(t.casefold() == team for t in e.teams):
            factor = TEAM_BOOST
        else:
            factor = 1.0
        out.append(
            e.model_copy(
                update={
                    "importance": round(min(1.0, e.base_importance * factor), 3),
                    "involves_favourite": factor > 1.0,
                }
            )
        )
    return sorted(out, key=lambda e: (-e.importance, e.lap, e.id))
