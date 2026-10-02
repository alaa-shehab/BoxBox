"""Simulator inputs: per-track parameters and per-driver starting state."""

from __future__ import annotations

from dataclasses import dataclass, field

# Global fallbacks when a circuit has no history (documented in ASSUMPTIONS.md).
DEFAULT_PIT_LOSS_S = 22.0
DEFAULT_SC_PROB = 0.45  # probability of at least one safety car in a race
DEFAULT_VSC_PROB = 0.35
DEFAULT_DNF_PER_CAR_LAP = 0.0008  # roughly 4-5% per car per race
DEFAULT_PASS_PROB = 0.35  # chance a faster car completes a pass on a given lap


@dataclass(frozen=True)
class TrackParams:
    circuit_key: str
    pit_loss_s: float = DEFAULT_PIT_LOSS_S
    sc_prob: float = DEFAULT_SC_PROB
    vsc_prob: float = DEFAULT_VSC_PROB
    dnf_per_car_lap: float = DEFAULT_DNF_PER_CAR_LAP
    pass_prob: float = DEFAULT_PASS_PROB  # higher = easier to overtake
    n_races: int = 0  # how many past races these were estimated from


@dataclass(frozen=True)
class SimDriver:
    code: str
    position: int
    gap_s: float  # behind the leader, including laps down (x reference lap)
    compound: str
    tyre_age: int
    compounds_used: frozenset[str]
    pit_count: int = 0
    pace_s: float = 0.0  # per-lap pace vs the field reference (negative = faster)
    pace_sd_s: float = 0.25  # our uncertainty about pace_s (shrinks as laps are observed)
    running: bool = True


@dataclass(frozen=True)
class SimInput:
    race_id: str
    lap: int  # laps completed by the leader
    total_laps: int
    drivers: tuple[SimDriver, ...]
    track: TrackParams
    deg: dict[str, float]  # s/lap per compound
    ref_lap_s: float = 90.0
    is_wet: bool = False
    neutralised: str | None = None  # "SC" or "VSC" if active right now


@dataclass(frozen=True)
class Scenario:
    """Counterfactuals applied on top of the base simulation."""

    name: str = "base"
    sc_in_laps: tuple[int, int] | None = None  # force a safety car between these laps
    pit_now: frozenset[str] = frozenset()  # drivers forced to pit on the next lap
    dnf: frozenset[str] = frozenset()  # drivers forced out on the next lap
    rain_from_lap: int | None = None  # rain arrives: slicks lose time until they pit
    params: dict[str, str] = field(default_factory=dict)
