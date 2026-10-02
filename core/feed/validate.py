"""Sanity checks that a replay reproduces the official result."""

from __future__ import annotations

from core.feed.base import RaceFeed


def validate_replay(feed: RaceFeed) -> list[str]:
    """Return a list of problems (empty means the replay is consistent)."""
    meta = feed.meta
    problems: list[str] = []
    final = feed.state(meta.total_laps)

    finishers = [r for r in meta.results if r.classified.isdigit()]
    official = [r.code for r in sorted(finishers, key=lambda r: int(r.classified))]
    replay = [d.code for d in final.drivers if d.status == "finished"][: len(official)]
    if replay != official:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(replay, official, strict=False)) if a != b), None
        )
        problems.append(
            f"final order differs from classification at P{(first_diff or 0) + 1}: "
            f"replay={replay} official={official}"
        )

    for lap in range(meta.total_laps + 1):
        positions = sorted(d.position for d in feed.state(lap).drivers)
        if positions != list(range(1, len(meta.drivers) + 1)):
            problems.append(f"lap {lap}: positions are not 1..{len(meta.drivers)}")
            break
    return problems
