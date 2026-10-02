"""Build compact replay data for the showcase races (or any races) from FastF1.

    python scripts/build_replays.py                  # all showcase races -> data/replays/
    python scripts/build_replays.py --race 2024:21   # one race

Each build is validated: the replay's final order must match the official classification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import get_settings
from core.feed.builder import build_from_fastf1
from core.feed.catalog import SHOWCASE_RACES
from core.feed.replay import ReplayFeed
from core.feed.validate import validate_replay
from core.log import configure_logging, get_logger

log = get_logger("build_replays")


def _parse(value: str) -> tuple[int, int]:
    season, rnd = value.split(":")
    return int(season), int(rnd)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race", type=_parse, action="append", help="SEASON:ROUND, repeatable")
    parser.add_argument("--out", type=Path, default=settings.replay_dir)
    args = parser.parse_args(argv)
    configure_logging(settings.log_level, settings.log_json)

    failures = 0
    for season, rnd in args.race or SHOWCASE_RACES:
        data = build_from_fastf1(season, rnd, settings.fastf1_cache_dir)
        problems = validate_replay(ReplayFeed(data))
        out = data.save(args.out / data.meta.race_id)
        size_kb = sum(f.stat().st_size for f in out.iterdir()) / 1024
        print(
            f"{data.meta.race_id} {data.meta.event_name}: {data.meta.total_laps} laps, "
            f"{size_kb:.0f} KB -> {out}"
        )
        for p in problems:
            print(f"  ! {p}")
        failures += bool(problems)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
