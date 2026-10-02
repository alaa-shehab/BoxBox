"""Build a SimInput from RaceFeed ticks only (no direct data-source access).

Driver pace comes from the laps seen so far in this race: the median of recent clean laps,
corrected for fuel, tyre age and compound, relative to the field. It's shrunk towards a
grid-position prior, so early in the race (few laps) the grid dominates, and later the
observed pace does. Degradation is the prior fit (previous race at this circuit),
updated with this race's clean laps so far.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

from core.models import RaceMeta, RaceState
from core.sim.degradation import COMPOUND_OFFSET_S, DEFAULT_DEG, SLICKS, blend
from core.sim.params import SimDriver, SimInput, TrackParams

GRID_PACE_PER_SLOT_S = 0.10  # pre-race prior: each grid slot ~0.1 s/lap slower
PACE_PRIOR_LAPS = 8.0  # observed laps needed to weigh observed pace equally with the prior
PACE_UNCERTAINTY_S = 0.35  # pace uncertainty with no laps observed; shrinks with laps seen
PACE_SHRINK = 1.0  # multiplies the final pace deltas (< 1 = regress towards the field)
RECENT_LAPS = 10
FUEL_LAP_GAIN_S = 0.03 * 100.0  # the whole race's fuel effect, spread per lap below


def _clean(prev: RaceState, cur: RaceState) -> bool:
    flags_ok = set(cur.flags_this_lap) <= {"GREEN", "YELLOW"}
    return cur.lap > 1 and flags_ok and prev.weather.rainfall == cur.weather.rainfall


def build_sim_input(
    meta: RaceMeta,
    history: Sequence[RaceState],
    track: TrackParams,
    deg_prior: dict[str, float] | None = None,
    grid_pace_per_slot_s: float = GRID_PACE_PER_SLOT_S,
    pace_uncertainty_s: float = PACE_UNCERTAINTY_S,
    pace_prior_laps: float = PACE_PRIOR_LAPS,
    pace_shrink: float = PACE_SHRINK,
) -> SimInput:
    """`history` is every tick from lap 0 up to and including the current one."""
    if not history:
        raise ValueError("need at least the grid state")
    cur = history[-1]
    prior = {c: (deg_prior or {}).get(c, DEFAULT_DEG[c]) for c in DEFAULT_DEG}
    fuel_per_lap = FUEL_LAP_GAIN_S / meta.total_laps

    used: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, list[tuple[int, str, int, float]]] = defaultdict(list)
    for prev, state in pairwise(history):
        clean = _clean(prev, state)
        before = {d.code: d for d in prev.drivers}
        for d in state.drivers:
            if d.compound != "UNKNOWN":
                used[d.code].add(d.compound)
            b = before.get(d.code)
            if (
                clean
                and b is not None
                and d.status == "running"
                and d.last_lap_s
                and not d.pitted_this_lap
                and not b.pitted_this_lap
                and d.pit_count == b.pit_count
                and d.laps_down == 0
            ):
                # Remove the fuel effect so early and late laps are comparable.
                t = d.last_lap_s + fuel_per_lap * state.lap
                samples[d.code].append((state.lap, d.compound, d.tyre_age, t))
    for d in history[0].drivers:
        if d.compound != "UNKNOWN":
            used[d.code].add(d.compound)

    # In-race degradation: within-driver-stint slope per compound, blended with the prior.
    deg = dict(prior)
    for compound in DEFAULT_DEG:
        sxy = sxx = 0.0
        n = 0
        for laps in samples.values():
            by_stint: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for lap, comp, age, t in laps:
                if comp == compound:
                    by_stint[lap - age].append((age, t))
            for pts in by_stint.values():
                if len(pts) < 4:
                    continue
                xm = statistics.fmean(a for a, _ in pts)
                ym = statistics.fmean(t for _, t in pts)
                sxy += sum((a - xm) * (t - ym) for a, t in pts)
                sxx += sum((a - xm) ** 2 for a, _ in pts)
                n += len(pts)
        observed = sxy / sxx if sxx > 0 else None
        deg[compound] = min(max(blend(prior[compound], observed, n), 0.0), 0.4)

    # Pace: recent clean laps corrected for compound and tyre age.
    corrected: dict[str, float] = {}
    for code, laps in samples.items():
        recent = laps[-RECENT_LAPS:]
        if recent:
            corrected[code] = statistics.median(
                t - COMPOUND_OFFSET_S.get(c, 0.0) - deg.get(c, 0.07) * age
                for _, c, age, t in recent
            )
    field = statistics.median(corrected.values()) if corrected else None
    recent_raw = [d.last_lap_s for d in cur.drivers if d.last_lap_s and d.status == "running"]
    ref_lap = statistics.median(recent_raw) if recent_raw and cur.lap > 1 else 90.0

    grid = meta.grid
    mid = (len(grid) + 1) / 2
    drivers = []
    last_gap = 0.0
    for d in sorted(cur.drivers, key=lambda d: d.position):
        if d.status in ("dnf", "dns"):
            continue
        prior_pace = grid_pace_per_slot_s * (grid.get(d.code, mid) - mid)
        n_obs = len(samples.get(d.code, []))
        if field is not None and d.code in corrected:
            w = n_obs / (n_obs + pace_prior_laps)
            pace = w * (corrected[d.code] - field) + (1 - w) * prior_pace
        else:
            pace = prior_pace
        gap = d.gap_to_leader_s
        if gap is None:  # grid, or missing timing: keep order with a nominal spacing
            gap = last_gap + (0.25 if cur.lap == 0 else 1.5)
        gap += d.laps_down * ref_lap
        last_gap = gap
        drivers.append(
            SimDriver(
                code=d.code,
                position=d.position,
                gap_s=float(gap),
                compound=d.compound,
                tyre_age=d.tyre_age,
                compounds_used=frozenset(used[d.code] or {d.compound}),
                pit_count=d.pit_count,
                pace_s=float(pace * pace_shrink),
                # Uncertainty about this driver's pace shrinks as more laps are observed.
                pace_sd_s=pace_uncertainty_s * (pace_prior_laps / (pace_prior_laps + n_obs)) ** 0.5,
            )
        )
    neutral = next((f for f in ("SC", "VSC") if cur.flag == f), None)
    is_wet = cur.weather.rainfall or any(d.compound in ("INTERMEDIATE", "WET") for d in drivers)
    return SimInput(
        race_id=meta.race_id,
        lap=cur.lap,
        total_laps=meta.total_laps,
        drivers=tuple(drivers),
        track=track,
        deg=deg,
        ref_lap_s=float(ref_lap),
        is_wet=bool(is_wet),
        neutralised=neutral,
    )


def slick_compounds_used(used: set[str]) -> int:
    return len(used & set(SLICKS))
