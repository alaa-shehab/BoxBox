"""Backtest the race simulator: calibration and Brier score against baselines.

    python scripts/backtest_sim.py           # -> evals/sim_backtest.md

Protocol (no peeking):
  * Track parameters and degradation priors come only from seasons before each race.
  * Two knobs (pace uncertainty, grid-pace prior) are tuned on the TUNE season; results
    are reported on the held-out TEST season.
  * At checkpoints (lap 0 = grid, 25%, 50%, 75% of race distance) the simulator predicts
    P(top 3) and P(win) for every running car, scored against the final classification.
  * Baselines: "order holds" (P = 1 if currently in the top 3 / leading, else 0) and a
    historical lookup P(top 3 | current position, race fraction) learned from earlier
    seasons, with Laplace smoothing. That second one is a strong, honest baseline.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import product

from core.config import REPO_ROOT
from core.feed.catalog import ReplayCatalog
from core.models import RaceMeta, RaceState
from core.sim.history import deg_prior, load_history, track_params
from core.sim.inputs import build_sim_input
from core.sim.model import simulate

OUT = REPO_ROOT / "evals" / "sim_backtest.md"
FRACTIONS = (0.0, 0.25, 0.5, 0.75)
EPS = 1e-4


@dataclass
class Row:
    race_id: str
    fraction: float
    code: str
    position: int
    p_top3: float
    p_win: float
    y_top3: int
    y_win: int


def brier(ps: list[float], ys: list[int]) -> float:
    return statistics.fmean((p - y) ** 2 for p, y in zip(ps, ys, strict=True))


def logloss(ps: list[float], ys: list[int]) -> float:
    return -statistics.fmean(
        math.log(min(max(p, EPS), 1 - EPS)) if y else math.log(1 - min(max(p, EPS), 1 - EPS))
        for p, y in zip(ps, ys, strict=True)
    )


def checkpoint_laps(meta: RaceMeta) -> list[tuple[float, int]]:
    return [(f, round(f * meta.total_laps)) for f in FRACTIONS]


def actual(meta: RaceMeta) -> tuple[set[str], str | None]:
    top3 = {r.code for r in meta.results if r.classified.isdigit() and int(r.classified) <= 3}
    win = next((r.code for r in meta.results if r.classified == "1"), None)
    return top3, win


def run_race(
    catalog,
    meta: RaceMeta,
    history,
    n_sims: int,
    pace_unc: float,  # type: ignore[no-untyped-def]
    grid_pace: float,
    prior_laps: float,
    pace_shrink: float = 1.0,
    pass_scale: float = 1.0,
) -> list[Row]:
    feed = catalog.feed(meta.race_id)
    states: list[RaceState] = [feed.state(lap) for lap in range(meta.total_laps + 1)]
    track = track_params(meta.circuit_key, meta.season, history)
    track = replace(track, pass_prob=min(track.pass_prob * pass_scale, 0.8))
    prior = deg_prior(meta.circuit_key, meta.season, history)
    top3, win = actual(meta)
    rows = []
    for fraction, lap in checkpoint_laps(meta):
        inp = build_sim_input(
            meta, states[: lap + 1], track, prior, grid_pace, pace_unc, prior_laps, pace_shrink
        )
        res = simulate(inp, n_sims=n_sims, seed=1000 + meta.round)
        p3, pw = res.p_top3, res.p_win
        for d in inp.drivers:
            rows.append(
                Row(
                    meta.race_id,
                    fraction,
                    d.code,
                    d.position,
                    p3[d.code],
                    pw[d.code],
                    int(d.code in top3),
                    int(d.code == win),
                )
            )
    return rows


def lookup_table(catalog, metas: list[RaceMeta]) -> dict[tuple[float, int], float]:  # type: ignore[no-untyped-def]
    hits: dict[tuple[float, int], list[int]] = defaultdict(lambda: [0, 0])
    for meta in metas:
        feed = catalog.feed(meta.race_id)
        top3, _ = actual(meta)
        for fraction, lap in checkpoint_laps(meta):
            for d in feed.state(lap).drivers:
                if d.status in ("dnf", "dns"):
                    continue
                cell = hits[(fraction, min(d.position, 20))]
                cell[0] += int(d.code in top3)
                cell[1] += 1
    return {k: (h + 1) / (n + 2) for k, (h, n) in hits.items()}  # Laplace smoothing


def score(rows: list[Row], table: dict[tuple[float, int], float]) -> dict[str, dict[str, float]]:
    ys3 = [r.y_top3 for r in rows]
    ysw = [r.y_win for r in rows]
    return {
        "Simulator": {
            "brier3": brier([r.p_top3 for r in rows], ys3),
            "log3": logloss([r.p_top3 for r in rows], ys3),
            "brierw": brier([r.p_win for r in rows], ysw),
        },
        "Order holds": {
            "brier3": brier([float(r.position <= 3) for r in rows], ys3),
            "log3": logloss([float(r.position <= 3) for r in rows], ys3),
            "brierw": brier([float(r.position == 1) for r in rows], ysw),
        },
        "Historical lookup": {
            "brier3": brier(
                [table.get((r.fraction, min(r.position, 20)), 0.15) for r in rows], ys3
            ),
            "log3": logloss(
                [table.get((r.fraction, min(r.position, 20)), 0.15) for r in rows], ys3
            ),
            "brierw": float("nan"),
        },
    }


def calibration(rows: list[Row], bins: int = 10) -> list[tuple[str, int, float, float]]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows if lo <= r.p_top3 < hi or (b == bins - 1 and r.p_top3 == 1.0)]
        if sel:
            out.append(
                (
                    f"{lo:.1f}-{hi:.1f}",
                    len(sel),
                    statistics.fmean(r.p_top3 for r in sel),
                    statistics.fmean(r.y_top3 for r in sel),
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-season", type=int, default=2023)
    parser.add_argument("--test-season", type=int, default=2024)
    parser.add_argument("--sims", type=int, default=1000)
    args = parser.parse_args(argv)

    catalog = ReplayCatalog.default()
    history = load_history(REPO_ROOT / "data" / "sim" / "history.json")
    metas = sorted(catalog.available(), key=lambda m: (m.season, m.round))
    tune = [m for m in metas if m.season == args.tune_season]
    test = [m for m in metas if m.season == args.test_season]
    if len(test) < 10:
        print(f"need >= 10 races in {args.test_season}, have {len(test)}")
        return 1

    # 1) Tune two knobs on the tune season.
    grid = list(product((0.2, 0.35), (0.05, 0.10), (25.0,), (0.5, 0.75, 1.0), (0.5, 1.0)))
    tune_table = lookup_table(catalog, [m for m in metas if m.season < args.tune_season])
    results = {}
    for knobs in grid:
        rows = [r for m in tune for r in run_race(catalog, m, history, args.sims, *knobs)]
        results[knobs] = score(rows, tune_table)["Simulator"]["brier3"]
        print(
            f"tune pace_unc={knobs[0]} grid_pace={knobs[1]} prior_laps={knobs[2]} "
            f"pace_shrink={knobs[3]} pass_scale={knobs[4]}: "
            f"Brier(top3)={results[knobs]:.4f}"
        )
    best = min(results, key=results.__getitem__)

    # 2) Evaluate once on the held-out season.
    start = time.perf_counter()
    rows = [r for m in test for r in run_race(catalog, m, history, args.sims, *best)]
    elapsed = time.perf_counter() - start
    table = lookup_table(catalog, [m for m in metas if m.season < args.test_season])
    overall = score(rows, table)
    by_fraction = {f: score([r for r in rows if r.fraction == f], table) for f in FRACTIONS}
    calib = calibration(rows)
    n_checkpoints = len(test) * len(FRACTIONS)
    write_report(
        args,
        best,
        results,
        overall,
        by_fraction,
        calib,
        test,
        len(rows),
        elapsed / n_checkpoints,
        len(tune),
    )
    print(
        f"best {best}; test Brier(top3) sim={overall['Simulator']['brier3']:.4f} "
        f"lookup={overall['Historical lookup']['brier3']:.4f} "
        f"order={overall['Order holds']['brier3']:.4f}"
    )
    return 0


def write_report(
    args,
    best,
    tuning,
    overall,
    by_fraction,
    calib,
    test,
    n_rows,  # type: ignore[no-untyped-def]
    sec_per_checkpoint,
    n_tune,
) -> None:
    def fmt(x: float) -> str:
        return "n/a" if x != x else f"{x:.3f}"  # NaN check

    lines = [
        "# Simulator backtest",
        "",
        f"Held-out test season **{args.test_season}**: {len(test)} races, {n_rows} "
        f"driver-checkpoint predictions ({args.sims} simulations per checkpoint, "
        f"{sec_per_checkpoint * 1000:.0f} ms each). Knobs tuned on {args.tune_season} "
        f"({n_tune} races): pace uncertainty = {best[0]} s/lap (shrinking as laps are "
        f"observed), grid-pace prior = {best[1]} s/lap per grid slot, observed-pace prior "
        f"weight = {best[2]:g} laps, pace shrink = {best[3]}, pass-probability scale = "
        f"{best[4]}. Track parameters and degradation priors use only "
        "seasons before each race. Lower is better for both scores.",
        "",
        "## Overall",
        "",
        "| Model | Brier P(top 3) | Log loss P(top 3) | Brier P(win) |",
        "|---|---:|---:|---:|",
    ]
    for name, m in overall.items():
        lines.append(f"| {name} | {fmt(m['brier3'])} | {fmt(m['log3'])} | {fmt(m['brierw'])} |")
    lines += [
        "",
        "## By race distance",
        "",
        "| Checkpoint | Simulator | Historical lookup | Order holds |",
        "|---|---:|---:|---:|",
    ]
    for f, m in by_fraction.items():
        label = "Grid (lap 0)" if f == 0 else f"{int(f * 100)}% distance"
        lines.append(
            f"| {label} | {fmt(m['Simulator']['brier3'])} | "
            f"{fmt(m['Historical lookup']['brier3'])} | "
            f"{fmt(m['Order holds']['brier3'])} |"
        )
    lines += [
        "",
        "Brier score of P(top 3).",
        "",
        "## Calibration (simulator P(top 3))",
        "",
        "| Predicted bin | Predictions | Mean predicted | Observed frequency |",
        "|---|---:|---:|---:|",
    ]
    for label, n, mean_p, freq in calib:
        lines.append(f"| {label} | {n} | {mean_p:.2f} | {freq:.2f} |")
    lines += [
        "",
        f"## Tuning grid ({args.tune_season}, Brier P(top 3))",
        "",
        "| Pace uncertainty | Grid-pace prior | Prior laps | Pace shrink | Pass scale | Brier |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for (pu, gp, pl, ps, sc), b in sorted(tuning.items()):
        lines.append(f"| {pu} | {gp} | {pl:g} | {ps} | {sc} | {b:.4f} |")
    lines += [
        "",
        f"Races: {', '.join(m.event_name for m in test)}.",
        "",
        "Reproduce: `make backtest` (needs the races in the local replay cache; "
        "see `scripts/backtest_sim.py`).",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
