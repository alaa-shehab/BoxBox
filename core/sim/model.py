"""Vectorised Monte Carlo race simulator (numpy). Deterministic for a given seed.

Simulates the remaining laps of a race many times at once. Arrays are (n_s, n_d):
simulations x drivers.
Every lap:
  1. Neutralisations: a safety car / VSC may start (per-lap hazard from the track's race
     probability) or be forced by a scenario. A safety car bunches the field.
  2. Pit decisions (economic): over the laps left on the current set, compare 0, 1 or 2
     more stops: degradation cost deg*H^2/(2(n+1)) against n pit losses, weighted up where
     passing is hard (track position is worth more). Stop once the stint reaches its
     optimal share, H/(n+1). A stop under SC/VSC costs less, which makes stopping early
     worthwhile. The two-dry-compounds rule forces one stop if needed.
  3. Lap times: reference + driver pace + compound offset + degradation x age + noise.
  4. DNFs at the track's per-car-lap hazard.
  5. Overtaking: a car that would get ahead on time only completes the pass if its
     systematic pace (driver pace + tyre offset + degradation) is better, with probability
     pass_prob x (pace advantage / 0.5 s, capped at 1). Otherwise it's held 0.3 s behind.
Final order: by cumulative time; retired cars last, ordered by how long they lasted.

All assumptions and constants are documented in core/sim/ASSUMPTIONS.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from core.sim.degradation import COMPOUND_OFFSET_S, DEFAULT_DEG, SLICKS
from core.sim.params import Scenario, SimInput

LAP_NOISE_S = 0.4  # lap-to-lap randomness
PIT_NOISE_S = 0.8
SC_LAP_FACTOR, VSC_LAP_FACTOR = 1.40, 1.35
SC_PIT_FACTOR, VSC_PIT_FACTOR = 0.50, 0.60
SC_LAPS = (3, 6)  # a safety car lasts this many laps (inclusive range)
VSC_LAPS = (1, 3)
SC_GAP_S = 0.8  # spacing behind the safety car
HOLD_GAP_S = 0.3  # a blocked car stays this far behind
# Physical safety caps only (Monaco 2024: 77 laps on one set of hards); economics decide.
MAX_TYRE_LIFE = {"SOFT": 45, "MEDIUM": 60, "HARD": 80, "INTERMEDIATE": 60, "WET": 60}
TRACK_POSITION_WEIGHT = 1.0  # pit loss weighted by 1 + w(1 - p)(0.35 / p); p = pass_prob
TYPICAL_PASS_PROB = 0.35
STOP_TIMING_JITTER = (0.85, 1.15)  # strategy diversity between cars and sims
PASS_FULL_ADVANTAGE_S = 0.5  # pace advantage at which a pass succeeds with p = pass_prob
# Fitted degradation is net of track evolution (the track gets faster as rubber goes down,
# hiding part of tyre wear within a stint). Fresh tyres also enjoy the faster track, so
# strategy decisions use the tyres' own wear: fitted deg + evolution.
TRACK_EVOLUTION_S_PER_LAP = 0.03
RAIN_SLICK_PENALTY_S = 8.0  # per lap on slicks once rain arrives
COMPOUND_ORDER = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
_IDX = {c: i for i, c in enumerate(COMPOUND_ORDER)}


@dataclass(frozen=True)
class SimResult:
    codes: tuple[str, ...]
    positions: np.ndarray  # (n_s, n_d) finishing position, 1-based
    lap: int
    n_sims: int
    seed: int
    elapsed_ms: float
    scenario: str = "base"
    stops: np.ndarray | None = None  # (n_s, n_d) pit stops made in the simulated laps

    def p_position_at_most(self, k: int) -> dict[str, float]:
        return {c: float((self.positions[:, i] <= k).mean()) for i, c in enumerate(self.codes)}

    @property
    def p_win(self) -> dict[str, float]:
        return self.p_position_at_most(1)

    @property
    def p_top3(self) -> dict[str, float]:
        return self.p_position_at_most(3)

    @property
    def expected_position(self) -> dict[str, float]:
        return {c: float(self.positions[:, i].mean()) for i, c in enumerate(self.codes)}

    def p_exact_podium(self, p1: str, p2: str, p3: str) -> float:
        idx = {c: i for i, c in enumerate(self.codes)}
        if not {p1, p2, p3} <= set(idx):
            return 0.0
        pos = self.positions
        hit = (pos[:, idx[p1]] == 1) & (pos[:, idx[p2]] == 2) & (pos[:, idx[p3]] == 3)
        return float(hit.mean())

    def p_podium_any_order(self, codes: tuple[str, ...]) -> float:
        idx = {c: i for i, c in enumerate(self.codes)}
        if not set(codes) <= set(idx):
            return 0.0
        return float(np.all([self.positions[:, idx[c]] <= 3 for c in codes], axis=0).mean())


def _hazard(p_race: float, laps: int) -> float:
    """Per-lap probability giving `p_race` chance of at least one event over `laps` laps."""
    p_race = min(max(p_race, 0.0), 0.999)
    return 1.0 - (1.0 - p_race) ** (1.0 / max(laps, 1))


def _remaining_stops(deg_now: np.ndarray, horizon: np.ndarray, stop_cost: np.ndarray) -> np.ndarray:
    """Cost-minimising number of further stops (0-2) for linear degradation.

    With n more stops and equal stints over `horizon` laps (current set's age + laps left),
    the degradation cost is deg*H^2/(2(n+1)) and the pit cost is n*stop_cost."""
    costs = np.stack([deg_now * horizon**2 / (2 * (n + 1)) + n * stop_cost for n in range(3)])
    return np.argmin(costs, axis=0)


def simulate(
    inp: SimInput, n_sims: int = 2000, seed: int = 7, scenario: Scenario | None = None
) -> SimResult:
    start = time.perf_counter()
    scenario = scenario or Scenario()
    rng = np.random.default_rng(seed)
    n_s, n_d = n_sims, len(inp.drivers)
    codes = tuple(d.code for d in inp.drivers)
    laps_left_total = max(inp.total_laps - inp.lap, 0)

    deg = np.array([inp.deg.get(c, DEFAULT_DEG[c]) for c in COMPOUND_ORDER], dtype=float)
    offset = np.array([COMPOUND_OFFSET_S[c] for c in COMPOUND_ORDER], dtype=float)

    t = np.tile(np.array([d.gap_s for d in inp.drivers], dtype=float), (n_s, 1))
    alive = np.tile(np.array([d.running for d in inp.drivers]), (n_s, 1))
    laps_survived = np.zeros((n_s, n_d), dtype=np.int16)
    stops = np.zeros((n_s, n_d), dtype=np.int16)
    comp = np.tile(np.array([_IDX.get(d.compound, 1) for d in inp.drivers]), (n_s, 1))
    age = np.tile(np.array([d.tyre_age for d in inp.drivers], dtype=float), (n_s, 1))
    used_two = np.tile(
        np.array([len(set(d.compounds_used) & set(SLICKS)) >= 2 for d in inp.drivers]), (n_s, 1)
    )
    on_wets = comp >= _IDX["INTERMEDIATE"]
    pace = np.array([d.pace_s for d in inp.drivers], dtype=float)
    pace_sd = np.array([d.pace_sd_s for d in inp.drivers], dtype=float)
    pace = pace[None, :] + rng.normal(0.0, 1.0, (n_s, n_d)) * pace_sd[None, :]
    jitter = rng.uniform(*STOP_TIMING_JITTER, (n_s, n_d))
    max_life = np.array([MAX_TYRE_LIFE[c] for c in COMPOUND_ORDER], dtype=float)
    # Track position is worth more where passing is harder than typical (Monaco: ~7.6x
    # the pit loss; an average track: ~1.65x), so cars there stay out longer.
    p_pass = max(inp.track.pass_prob, 0.02)
    position_factor = 1.0 + TRACK_POSITION_WEIGHT * (1.0 - p_pass) * TYPICAL_PASS_PROB / p_pass
    # Dry rule: a car that hasn't used two dry compounds must stop before the flag.
    mandatory = (not inp.is_wet) & ~used_two & ~on_wets  # `not`: is_wet is a Python bool
    must_by = np.where(
        mandatory, inp.lap + laps_left_total - rng.integers(2, 12, (n_s, n_d)), 10**6
    )

    h_sc = _hazard(inp.track.sc_prob, inp.total_laps)
    h_vsc = _hazard(inp.track.vsc_prob, inp.total_laps)
    sc_left = np.zeros(n_s, dtype=int)
    vsc_left = np.zeros(n_s, dtype=int)
    if inp.neutralised == "SC":
        sc_left[:] = rng.integers(1, SC_LAPS[1], n_s)
    elif inp.neutralised == "VSC":
        vsc_left[:] = rng.integers(1, VSC_LAPS[1], n_s)
    forced_sc_lap = None
    if scenario.sc_in_laps is not None:
        lo, hi = scenario.sc_in_laps
        forced_sc_lap = rng.integers(max(lo, inp.lap + 1), max(hi, lo) + 1, n_s)
    forced_pit = np.array([c in scenario.pit_now for c in codes])
    forced_dnf = np.array([c in scenario.dnf for c in codes])
    rows = np.arange(n_s)

    for lap in range(inp.lap + 1, inp.total_laps + 1):
        laps_left = inp.total_laps - lap
        # 1. neutralisations ----------------------------------------------------------
        new_sc = (sc_left == 0) & (vsc_left == 0) & (rng.random(n_s) < h_sc)
        if forced_sc_lap is not None:
            new_sc |= (forced_sc_lap == lap) & (sc_left == 0)
        new_vsc = (sc_left == 0) & (vsc_left == 0) & ~new_sc & (rng.random(n_s) < h_vsc)
        sc_left = np.where(new_sc, rng.integers(SC_LAPS[0], SC_LAPS[1] + 1, n_s), sc_left)
        vsc_left = np.where(new_vsc, rng.integers(VSC_LAPS[0], VSC_LAPS[1] + 1, n_s), vsc_left)
        under_sc, under_vsc = sc_left > 0, vsc_left > 0
        neutral = under_sc | under_vsc

        # 2. pit decisions --------------------------------------------------------------
        loss_now = np.where(under_sc, SC_PIT_FACTOR, np.where(under_vsc, VSC_PIT_FACTOR, 1.0))
        stop_cost = (inp.track.pit_loss_s * position_factor * loss_now)[:, None]
        horizon = age + laps_left + 1
        n_more = _remaining_stops(deg[comp] + TRACK_EVOLUTION_S_PER_LAP, horizon, stop_cost)
        n_more = np.where(mandatory, np.maximum(n_more, 1), n_more)
        planned = (n_more >= 1) & (age >= jitter * horizon / (n_more + 1))
        worn_out = age >= max_life[comp]
        on_wet_tyres = comp >= _IDX["INTERMEDIATE"]
        due = np.where(on_wet_tyres, worn_out, planned | worn_out) | (lap >= must_by)
        pit = alive & (laps_left >= 2) & due
        if lap == inp.lap + 1:
            pit |= alive & forced_pit[None, :]
        rain = scenario.rain_from_lap is not None and lap >= scenario.rain_from_lap
        if rain:
            on_slicks = comp < _IDX["INTERMEDIATE"]
            pit |= alive & on_slicks & (rng.random((n_s, n_d)) < 0.6)
        if pit.any():
            t += pit * (
                inp.track.pit_loss_s * loss_now[:, None] + rng.normal(0.0, PIT_NOISE_S, (n_s, n_d))
            )
            if rain:
                new_comp = np.full((n_s, n_d), _IDX["INTERMEDIATE"])
            else:
                # Remaining distance decides the compound; respect the two-compound rule.
                choice = np.where(
                    laps_left <= 18,
                    _IDX["SOFT"],
                    np.where(laps_left <= 30, _IDX["MEDIUM"], _IDX["HARD"]),
                )
                choice = np.where(
                    mandatory & (choice == comp),
                    np.where(comp == _IDX["HARD"], _IDX["MEDIUM"], _IDX["HARD"]),
                    choice,
                )
                new_comp = np.where(on_wets, comp, choice)
            stops += pit
            comp = np.where(pit, new_comp, comp)
            age = np.where(pit, 0.0, age)
            used_two |= pit & (new_comp < _IDX["INTERMEDIATE"])
            mandatory &= ~pit
            must_by = np.where(pit, 10**6, must_by)
            jitter = np.where(pit, rng.uniform(*STOP_TIMING_JITTER, (n_s, n_d)), jitter)

        # 3. lap times --------------------------------------------------------------------
        systematic = pace + offset[comp] + deg[comp] * age  # what a pass must be built on
        lap_t = inp.ref_lap_s + systematic + rng.normal(0.0, LAP_NOISE_S, (n_s, n_d))
        if rain:
            lap_t += np.where(comp < _IDX["INTERMEDIATE"], RAIN_SLICK_PENALTY_S, 0.0)
        lap_t = np.where(under_sc[:, None], inp.ref_lap_s * SC_LAP_FACTOR, lap_t)
        lap_t = np.where(under_vsc[:, None], lap_t * VSC_LAP_FACTOR, lap_t)
        prev_order = np.argsort(np.where(alive, t, np.inf), axis=1)
        t_new = t + lap_t
        age += 1

        # 4. DNFs -------------------------------------------------------------------------
        out = alive & (rng.random((n_s, n_d)) < inp.track.dnf_per_car_lap)
        if lap == inp.lap + 1:
            out |= alive & forced_dnf[None, :]
        alive &= ~out
        laps_survived += alive

        # 5. safety car bunching / overtaking ---------------------------------------------
        if new_sc.any():
            order = np.argsort(np.where(alive, t_new, np.inf), axis=1)
            ranks = np.empty_like(order)
            ranks[rows[:, None], order] = np.arange(n_d)[None, :]
            lead_t = np.min(np.where(alive, t_new, np.inf), axis=1)
            bunched = lead_t[:, None] + ranks * SC_GAP_S
            t_new = np.where(new_sc[:, None] & alive, bunched, t_new)
        green = ~neutral
        for i in range(1, n_d):
            a, b = prev_order[:, i - 1], prev_order[:, i]
            ta, tb = t_new[rows, a], t_new[rows, b]
            # A pit stop reordering the cars is not an overtake: skip pairs where either pitted.
            no_pit = ~pit[rows, a] & ~pit[rows, b]
            attempt = green & alive[rows, a] & alive[rows, b] & (tb < ta) & no_pit
            if not attempt.any():
                continue
            advantage = systematic[rows, a] - systematic[rows, b]  # > 0: b is truly faster
            p_pass = inp.track.pass_prob * np.clip(advantage / PASS_FULL_ADVANTAGE_S, 0.0, 1.0)
            blocked = attempt & (rng.random(n_s) >= p_pass)
            t_new[rows[blocked], b[blocked]] = ta[blocked] + HOLD_GAP_S
        t = t_new
        sc_left = np.maximum(sc_left - 1, 0)
        vsc_left = np.maximum(vsc_left - 1, 0)

    # Final classification: running cars by time, then retirements by laps survived.
    big = 1e9
    score = np.where(alive, t, big + (laps_left_total - laps_survived) * 1e3 + t * 1e-6)
    order = np.argsort(score, axis=1, kind="stable")
    positions = np.empty((n_s, n_d), dtype=np.int16)
    positions[rows[:, None], order] = np.arange(1, n_d + 1)[None, :]
    elapsed = (time.perf_counter() - start) * 1000
    return SimResult(codes, positions, inp.lap, n_s, seed, elapsed, scenario.name, stops)
