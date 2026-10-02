"""Per-race summaries of past races, and track parameters derived from them.

`summarise` turns a replay into one compact row (SC/VSC/red flag, pit loss, on-track
passes, DNFs, degradation). `track_params` aggregates rows strictly from seasons BEFORE
the target race, shrinking each circuit's rates towards the global average (k pseudo-races),
so a circuit with one or two past races doesn't get extreme parameters.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from core.events.detector import EventDetector
from core.feed.data import ReplayData
from core.feed.replay import ReplayFeed
from core.sim.degradation import DEFAULT_DEG, clean_laps, fit_degradation
from core.sim.params import (
    DEFAULT_DNF_PER_CAR_LAP,
    DEFAULT_PASS_PROB,
    DEFAULT_PIT_LOSS_S,
    DEFAULT_SC_PROB,
    DEFAULT_VSC_PROB,
    TrackParams,
)

SHRINK_RACES = 4.0
SHRINK_PASS_RACES = 1.0


@dataclass(frozen=True)
class RaceSummary:
    race_id: str
    circuit_key: str
    season: int
    round: int
    total_laps: int
    is_wet: bool
    had_sc: bool
    had_vsc: bool
    had_red: bool
    pit_loss_s: float | None
    passes: int  # on-track overtakes, excluding lap 1 and neutralised laps
    dnfs: int
    car_laps: int
    median_lap_s: float | None
    deg: dict[str, float] = field(default_factory=dict)


def _pit_loss(data: ReplayData) -> float | None:
    """Median of (in-lap + out-lap) - 2 x the driver's median clean lap, green stops only."""
    clean = clean_laps(data)
    ref = clean.groupby("driver")["lap_time_s"].median()
    laps = data.laps.set_index(["driver", "lap"])
    losses = []
    for (drv, lap), row in laps[laps["pit_in"]].iterrows():
        nxt = (drv, lap + 1)
        if drv not in ref.index or nxt not in laps.index:
            continue
        out = laps.loc[nxt]
        if np.isnan(row["lap_time_s"]) or np.isnan(out["lap_time_s"]):
            continue
        loss = row["lap_time_s"] + out["lap_time_s"] - 2 * ref[drv]
        if 10.0 <= loss <= 45.0:  # drop stops under SC/VSC and long repairs
            losses.append(float(loss))
    return round(statistics.median(losses), 2) if len(losses) >= 3 else None


def summarise(data: ReplayData) -> RaceSummary:
    feed = ReplayFeed(data)
    states = [feed.state(lap) for lap in range(data.meta.total_laps + 1)]
    events = EventDetector(data.meta).run(states)
    flags = set(data.track_status["flag"].astype(str))
    passes = sum(1 for e in events if e.type == "overtake" and e.lap > 1)
    final = states[-1]
    clean = clean_laps(data)
    fit = fit_degradation(data)
    return RaceSummary(
        race_id=data.meta.race_id,
        circuit_key=data.meta.circuit_key,
        season=data.meta.season,
        round=data.meta.round,
        total_laps=data.meta.total_laps,
        is_wet=data.meta.is_wet,
        had_sc="SC" in flags,
        had_vsc="VSC" in flags,
        had_red="RED" in flags,
        pit_loss_s=_pit_loss(data),
        passes=passes,
        dnfs=sum(1 for d in final.drivers if d.status == "dnf"),
        car_laps=int(sum(d.laps_completed for d in final.drivers)),
        median_lap_s=round(float(clean["lap_time_s"].median()), 3) if len(clean) else None,
        deg={c: d.slope_s_per_lap for c, d in fit.by_compound.items()},
    )


def save_history(rows: list[RaceSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r.season, r.round))
    path.write_text(json.dumps([asdict(r) for r in rows], indent=1) + "\n")


def load_history(path: Path) -> list[RaceSummary]:
    if not path.exists():
        return []
    return [RaceSummary(**r) for r in json.loads(path.read_text())]


def _shrunk(local: list[float], global_rate: float, k: float = SHRINK_RACES) -> float:
    return (k * global_rate + sum(local)) / (k + len(local))


def track_params(circuit_key: str, season: int, history: list[RaceSummary]) -> TrackParams:
    """Parameters for a race at `circuit_key` in `season`, from earlier seasons only."""
    past = [r for r in history if r.season < season]
    local = [r for r in past if r.circuit_key == circuit_key]
    if not past:
        return TrackParams(circuit_key)
    g_sc = statistics.fmean(r.had_sc for r in past)
    g_vsc = statistics.fmean(r.had_vsc for r in past)
    car_laps = sum(r.car_laps for r in past) or 1
    g_dnf = sum(r.dnfs for r in past) / car_laps
    dry = [r for r in past if not r.is_wet]
    g_pass_rate = sum(r.passes for r in dry) / max(sum(r.car_laps for r in dry), 1)
    l_dry = [r for r in local if not r.is_wet]
    l_rate = (
        sum(r.passes for r in l_dry) / max(sum(r.car_laps for r in l_dry), 1)
        if l_dry
        else g_pass_rate
    )
    # Passing data is plentiful and circuit-specific (Monaco vs Monza), so it gets lighter
    # shrinkage than the rare SC/VSC events.
    w = len(l_dry) / (len(l_dry) + SHRINK_PASS_RACES)
    rate = w * l_rate + (1 - w) * g_pass_rate
    pass_prob = DEFAULT_PASS_PROB * (rate / g_pass_rate if g_pass_rate > 0 else 1.0)
    losses = [r.pit_loss_s for r in local if r.pit_loss_s]
    g_losses = [r.pit_loss_s for r in past if r.pit_loss_s]
    pit_loss = (
        statistics.median(losses)
        if losses
        else statistics.median(g_losses)
        if g_losses
        else DEFAULT_PIT_LOSS_S
    )
    return TrackParams(
        circuit_key=circuit_key,
        pit_loss_s=round(float(pit_loss), 2),
        sc_prob=round(_shrunk([float(r.had_sc) for r in local], g_sc or DEFAULT_SC_PROB), 3),
        vsc_prob=round(_shrunk([float(r.had_vsc) for r in local], g_vsc or DEFAULT_VSC_PROB), 3),
        dnf_per_car_lap=round(g_dnf or DEFAULT_DNF_PER_CAR_LAP, 5),
        pass_prob=round(float(np.clip(pass_prob, 0.05, 0.8)), 3),
        n_races=len(local),
    )


def deg_prior(circuit_key: str, season: int, history: list[RaceSummary]) -> dict[str, float]:
    """Degradation from the most recent earlier dry race at this circuit, else defaults."""
    local = sorted(
        (r for r in history if r.circuit_key == circuit_key and r.season < season and not r.is_wet),
        key=lambda r: (r.season, r.round),
    )
    prior = dict(DEFAULT_DEG)
    if local:
        for c, v in local[-1].deg.items():
            prior[c] = min(max(v, 0.0), 0.4)
    return prior
