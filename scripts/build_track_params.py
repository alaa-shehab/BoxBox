"""Summarise every cached/committed race into data/sim/history.json.

    python scripts/build_track_params.py

Each row records SC/VSC/red flag, pit loss, on-track passes, DNFs and the degradation
fit for one race. Track parameters for any race are derived from rows of EARLIER seasons
only (core.sim.history.track_params), so the simulator never sees the future.
"""

from __future__ import annotations

import sys

from core.config import REPO_ROOT
from core.feed.catalog import ReplayCatalog
from core.sim.history import save_history, summarise, track_params

OUT = REPO_ROOT / "data" / "sim" / "history.json"


def main() -> int:
    catalog = ReplayCatalog.default()
    rows = []
    for meta in sorted(catalog.available(), key=lambda m: (m.season, m.round)):
        rows.append(summarise(catalog.load_data(meta.race_id)))
        r = rows[-1]
        print(
            f"{r.race_id} {r.circuit_key:22} sc={r.had_sc:d} vsc={r.had_vsc:d} "
            f"pit_loss={r.pit_loss_s} passes={r.passes} dnfs={r.dnfs}"
        )
    save_history(rows, OUT)
    print(f"wrote {len(rows)} races -> {OUT}")
    for key, season in {(r.circuit_key, 2025) for r in rows[-3:]}:
        print(key, season, track_params(key, season, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
