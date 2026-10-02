from __future__ import annotations

from core.events.importance import personalise
from core.events.types import Event


def ev(id_: str, imp: float, drivers: tuple[str, ...] = (), teams: tuple[str, ...] = ()) -> Event:
    return Event(
        id=id_,
        race_id="r",
        lap=1,
        type="overtake",
        drivers=drivers,
        teams=teams,
        summary="x",
        base_importance=imp,
    )


def test_driver_boost_beats_team_boost() -> None:
    events = [
        ev("plain", 0.5, ("AAA",), ("Red",)),
        ev("team", 0.4, ("BBB",), ("Williams",)),
        ev("driver", 0.4, ("ALB",), ("Williams",)),
    ]
    out = personalise(events, "williams", ["alb"])
    assert [e.id for e in out] == ["driver", "team", "plain"]
    by_id = {e.id: e for e in out}
    assert by_id["driver"].importance == 0.7 and by_id["driver"].involves_favourite
    assert by_id["team"].importance == 0.56
    assert by_id["plain"].importance == 0.5 and not by_id["plain"].involves_favourite


def test_boost_is_capped_and_inputs_unchanged() -> None:
    e = ev("x", 0.9, ("ALB",))
    (out,) = personalise([e], None, ["ALB"])
    assert out.importance == 1.0
    assert e.importance == 0.9  # original untouched


def test_no_favourites() -> None:
    (out,) = personalise([ev("x", 0.3, ("AAA",), ("T",))], None)
    assert out.importance == 0.3 and not out.involves_favourite
