"""Fit tyre degradation for every cached/committed race and write the per-track cache.

    python scripts/fit_degradation.py    # -> data/sim/degradation/{circuit}_{season}_{round}.json

Manual overrides live in data/sim/degradation/overrides.json: {"circuit_key": {"SOFT": 0.12}}
(applied on top of fitted priors at load time; see core/sim/store.py).
"""

from __future__ import annotations

import sys

from core.config import REPO_ROOT
from core.feed.catalog import ReplayCatalog
from core.sim.degradation import fit_degradation

OUT = REPO_ROOT / "data" / "sim" / "degradation"


def main() -> int:
    catalog = ReplayCatalog.default()
    OUT.mkdir(parents=True, exist_ok=True)
    for meta in sorted(catalog.available(), key=lambda m: (m.season, m.round)):
        fit = fit_degradation(catalog.load_data(meta.race_id))
        fit.save(OUT / f"{meta.circuit_key}_{meta.season}_{meta.round:02d}.json")
        summary = ", ".join(f"{c} {d.slope_s_per_lap:+.3f}" for c, d in fit.by_compound.items())
        print(f"{meta.race_id} {meta.circuit_key:22} {summary or '(no clean stints)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
