"""Tyre degradation fitted from real stint data (fuel-corrected, within-stint linear fit).

For each compound we fit lap_time ~ tyre_age on clean laps only, with an intercept per
stint (so car pace and track evolution between stints don't leak into the slope):

    slope = sum_s sum_i (age_i - mean_s)(t_i - mean_s) / sum_s sum_i (age_i - mean_s)^2

Lap times are first fuel-corrected: a car gets FUEL_EFFECT_S_PER_KG faster for every kg it
burns, and burns FUEL_START_KG linearly over the race. Clean laps exclude lap 1, in- and
out-laps, deleted laps, anything under SC/VSC/red flag, rain laps for slick compounds,
and per-stint outliers. See core/sim/ASSUMPTIONS.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.feed.data import ReplayData

FUEL_START_KG = 100.0
FUEL_EFFECT_S_PER_KG = 0.03
MIN_STINT_LAPS = 4
MIN_COMPOUND_LAPS = 15

# Fallbacks when a compound has no usable data (s/lap), and fresh-tyre pace offsets vs
# MEDIUM (s/lap; negative = faster). Typical modern-era values; documented in ASSUMPTIONS.md.
DEFAULT_DEG: dict[str, float] = {
    "SOFT": 0.10,
    "MEDIUM": 0.07,
    "HARD": 0.05,
    "INTERMEDIATE": 0.08,
    "WET": 0.06,
}
COMPOUND_OFFSET_S: dict[str, float] = {
    "SOFT": -0.6,
    "MEDIUM": 0.0,
    "HARD": 0.4,
    "INTERMEDIATE": 0.0,
    "WET": 0.0,
}
SLICKS = ("SOFT", "MEDIUM", "HARD")
DEG_BOUNDS = (0.0, 0.4)


@dataclass(frozen=True)
class CompoundDeg:
    slope_s_per_lap: float
    n_laps: int
    n_stints: int
    stderr: float


@dataclass(frozen=True)
class DegFit:
    circuit_key: str
    season: int
    by_compound: dict[str, CompoundDeg]

    def slope(self, compound: str) -> float:
        fit = self.by_compound.get(compound)
        raw = fit.slope_s_per_lap if fit else DEFAULT_DEG.get(compound, 0.07)
        return float(min(max(raw, DEG_BOUNDS[0]), DEG_BOUNDS[1]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "circuit_key": self.circuit_key,
            "season": self.season,
            "by_compound": {c: asdict(d) for c, d in self.by_compound.items()},
        }
        path.write_text(json.dumps(payload, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> DegFit:
        raw = json.loads(path.read_text())
        return cls(
            raw["circuit_key"],
            raw["season"],
            {c: CompoundDeg(**d) for c, d in raw["by_compound"].items()},
        )


def fuel_corrected(lap_time_s: np.ndarray, lap: np.ndarray, total_laps: int) -> np.ndarray:
    remaining_kg = FUEL_START_KG * (1.0 - (lap - 1) / total_laps)
    return lap_time_s - FUEL_EFFECT_S_PER_KG * remaining_kg


def neutralised_laps(data: ReplayData) -> np.ndarray:
    """Boolean mask over data.laps: True if SC, VSC or a red flag was shown during the lap."""
    laps = data.laps
    ts = data.track_status.sort_values("time_s")
    times, flags = ts["time_s"].to_numpy(), ts["flag"].astype(str).to_numpy()
    end = laps["time_s"].to_numpy()
    start = end - laps["lap_time_s"].fillna(120.0).to_numpy()
    bad = np.zeros(len(laps), dtype=bool)
    neutral = np.isin(flags, ["SC", "VSC", "RED"])
    for i, (s, e) in enumerate(zip(start, end, strict=True)):
        if math.isnan(e):
            bad[i] = True
            continue
        before = np.searchsorted(times, s, side="right") - 1
        inside = (times > s) & (times <= e)
        bad[i] = bool((before >= 0 and neutral[before]) or (neutral & inside).any())
    return bad


def rain_laps(data: ReplayData) -> np.ndarray:
    w = data.weather.sort_values("time_s")
    if w.empty:
        return np.zeros(len(data.laps), dtype=bool)
    idx = np.searchsorted(w["time_s"].to_numpy(), data.laps["time_s"].to_numpy(), side="right") - 1
    rain = w["rainfall"].to_numpy()
    return np.array([bool(rain[i]) if i >= 0 else False for i in idx])


def clean_laps(data: ReplayData, up_to_lap: int | None = None) -> pd.DataFrame:
    """Clean racing laps with fuel-corrected times (`t_fc`), optionally only laps <= up_to_lap."""
    laps = data.laps.copy()
    laps["neutral"] = neutralised_laps(data)
    laps["rain"] = rain_laps(data)
    total = data.meta.total_laps
    mask = (
        (laps["lap"] > 1)
        & ~laps["pit_in"]
        & ~laps["pit_out"]
        & ~laps["deleted"]
        & ~laps["neutral"]
        & laps["lap_time_s"].notna()
        & laps["tyre_life"].notna()
        & laps["compound"].isin(list(DEFAULT_DEG))
        & ~(laps["compound"].isin(SLICKS) & laps["rain"])
    )
    if up_to_lap is not None:
        mask &= laps["lap"] <= up_to_lap
    laps = laps[mask].copy()
    laps["t_fc"] = fuel_corrected(laps["lap_time_s"].to_numpy(), laps["lap"].to_numpy(), total)
    # Per-stint outliers (traffic, mistakes): drop laps > 1.5 s off the stint median.
    med = laps.groupby(["driver", "stint"])["t_fc"].transform("median")
    return laps[(laps["t_fc"] - med).abs() <= 1.5]


def fit_degradation(data: ReplayData, up_to_lap: int | None = None) -> DegFit:
    laps = clean_laps(data, up_to_lap)
    out: dict[str, CompoundDeg] = {}
    for compound, g in laps.groupby("compound"):
        stints = [s for _, s in g.groupby(["driver", "stint"]) if len(s) >= MIN_STINT_LAPS]
        n = sum(len(s) for s in stints)
        if n < MIN_COMPOUND_LAPS:
            continue
        sxy = sxx = 0.0
        residuals: list[np.ndarray] = []
        for s in stints:
            x = s["tyre_life"].to_numpy(float)
            y = s["t_fc"].to_numpy(float)
            dx, dy = x - x.mean(), y - y.mean()
            sxy += float((dx * dy).sum())
            sxx += float((dx * dx).sum())
            residuals.append((dx, dy))  # type: ignore[arg-type]
        if sxx <= 0:
            continue
        slope = sxy / sxx
        sse = sum(float(((dy - slope * dx) ** 2).sum()) for dx, dy in residuals)
        dof = max(1, n - len(stints) - 1)
        stderr = math.sqrt(sse / dof / sxx)
        out[str(compound)] = CompoundDeg(round(slope, 4), n, len(stints), round(stderr, 4))
    return DegFit(data.meta.circuit_key, data.meta.season, out)


def blend(
    prior: float, observed: float | None, n_observed: int, prior_weight: float = 40.0
) -> float:
    """Shrink the in-race estimate towards the prior; more observed laps, more trust."""
    if observed is None or n_observed <= 0:
        return prior
    return (prior_weight * prior + n_observed * observed) / (prior_weight + n_observed)
