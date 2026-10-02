"""Turn ReplayData into one RaceState per lap. This is pure and deterministic.

Ordering rules match a timing tower: cars are ranked by laps completed, then by when they
crossed the line on that lap. Gaps are measured at the same timing line against the
first car to complete that lap. Lapped cars report `laps_down`.

A lapped car's lap-L row is shown at tick L, even though it crosses the line slightly
after the leader. Live timing behaves the same way, because each car updates as it
crosses the line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.feed.data import ReplayData
from core.models import (
    DriverState,
    FastestLap,
    RaceControlMsg,
    RaceMeta,
    RaceState,
    TrackFlag,
    Weather,
)

_FINISHED = "finished"


def _f(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


class _DriverLaps:
    """Per-driver lap table indexed by lap number, with cumulative columns precomputed."""

    def __init__(self, df: pd.DataFrame) -> None:
        df = df.sort_values("lap").drop_duplicates("lap", keep="last").set_index("lap")
        valid = df["lap_time_s"].where(~df["deleted"])
        df = df.assign(
            best_lap_s=valid.cummin(),
            # Stops completed before this lap ends: a car that dives in at the end of lap
            # k is counted from lap k+1, when its new tyres are known.
            pit_count=df["pit_in"].astype(int).cumsum() - df["pit_in"].astype(int),
        )
        self.df = df
        self.laps = df.index.to_numpy()
        self.max_lap = int(self.laps.max())

    def row_at_or_before(self, lap: int) -> tuple[int, pd.Series] | None:
        idx = int(np.searchsorted(self.laps, lap, side="right")) - 1
        if idx < 0:
            return None
        k = int(self.laps[idx])
        return k, self.df.loc[k]


class _Timeline:
    """A step function of values over session time."""

    def __init__(self, times: Sequence[float], values: Sequence[object], default: object) -> None:
        self.times = np.asarray(times, dtype=float)
        self.values = list(values)
        self.default = default

    def at(self, t: float) -> object:
        idx = int(np.searchsorted(self.times, t, side="right")) - 1
        return self.default if idx < 0 else self.values[idx]

    def between(self, t0: float, t1: float) -> list[object]:
        """The value at t0 followed by every change in (t0, t1]."""
        out = [self.at(t0)]
        lo = int(np.searchsorted(self.times, t0, side="right"))
        hi = int(np.searchsorted(self.times, t1, side="right"))
        out.extend(self.values[lo:hi])
        return out


def _dedupe_consecutive(flags: list[TrackFlag]) -> tuple[TrackFlag, ...]:
    out: list[TrackFlag] = []
    for f in flags:
        if not out or out[-1] != f:
            out.append(f)
    return tuple(out)


def _leader_times(laps: pd.DataFrame, total: int) -> np.ndarray:
    """lt[L] = first time any car completed lap L (lt[0] = estimated race start)."""
    firsts = laps.groupby("lap")["time_s"].min()
    lt = np.full(total + 1, np.nan)
    for lap, t in firsts.items():
        if 1 <= int(lap) <= total:
            lt[int(lap)] = t
    # Fill rare gaps (missing timing) by interpolation so the series stays monotonic.
    s = pd.Series(lt[1:]).interpolate(limit_direction="both")
    lt[1:] = s.to_numpy()
    lap1 = laps[(laps["lap"] == 1) & laps["lap_time_s"].notna()]
    first_lap = float(lap1["lap_time_s"].min()) if not lap1.empty else 120.0
    lt[0] = lt[1] - first_lap
    return lt


def _first_not_none(*values: object) -> pd.Series | None:
    return next((v for v in values if v is not None), None)  # type: ignore[return-value]


def _weather(row: object) -> Weather:
    if not isinstance(row, pd.Series):
        return Weather()
    return Weather(
        air_temp_c=_f(row["air_temp_c"]),
        track_temp_c=_f(row["track_temp_c"]),
        humidity_pct=_f(row["humidity_pct"]),
        rainfall=bool(row["rainfall"]),
    )


def build_states(data: ReplayData) -> list[RaceState]:
    meta = data.meta
    total = meta.total_laps
    by_driver = {str(code): _DriverLaps(g) for code, g in data.laps.groupby("driver")}
    lt = _leader_times(data.laps, total)
    classified = {r.code for r in meta.results if r.classified.isdigit()}

    flags = _Timeline(data.track_status["time_s"], data.track_status["flag"], "GREEN")
    weather_rows = [row for _, row in data.weather.iterrows()]
    weather = _Timeline(data.weather["time_s"], weather_rows, None)
    messages: dict[int, list[RaceControlMsg]] = {}
    for rec in data.messages.to_dict("records"):
        drivers = tuple(c for c in str(rec["drivers"] or "").split(",") if c)
        flag = rec["flag"]
        messages.setdefault(int(rec["lap"]), []).append(
            RaceControlMsg(
                lap=int(rec["lap"]),
                category=str(rec["category"]),
                message=str(rec["message"]),
                flag=None if flag is None or pd.isna(flag) else str(flag),
                drivers=drivers,
            )
        )

    states = [_grid_state(meta, by_driver, lt, weather)]
    for lap in range(1, total + 1):
        states.append(
            _lap_state(meta, lap, by_driver, lt, classified, flags, weather, messages, data)
        )
    return states


def _grid_state(
    meta: RaceMeta, by_driver: dict[str, _DriverLaps], lt: np.ndarray, weather: _Timeline
) -> RaceState:
    drivers = []
    ordered = sorted(meta.drivers, key=lambda c: (c not in by_driver, meta.grid.get(c, 99)))
    for pos, code in enumerate(ordered, start=1):
        info = meta.drivers[code]
        first = by_driver[code].row_at_or_before(1) if code in by_driver else None
        row = first[1] if first else None
        tyre_life = _f(row["tyre_life"]) if row is not None else None
        drivers.append(
            DriverState(
                code=code,
                team=info.team,
                position=pos,
                status="running" if code in by_driver else "dns",
                compound=str(row["compound"]) if row is not None else "UNKNOWN",  # type: ignore[arg-type]
                tyre_age=max(0, int(tyre_life) - 1) if tyre_life is not None else 0,
            )
        )
    return RaceState(
        race_id=meta.race_id,
        lap=0,
        total_laps=meta.total_laps,
        drivers=tuple(drivers),
        weather=_weather(_first_not_none(weather.at(float(lt[0])), weather.at(float("inf")))),
        session_time_s=float(lt[0]),
    )


def _lap_state(
    meta: RaceMeta,
    lap: int,
    by_driver: dict[str, _DriverLaps],
    lt: np.ndarray,
    classified: set[str],
    flags: _Timeline,
    weather: _Timeline,
    messages: dict[int, list[RaceControlMsg]],
    data: ReplayData,
) -> RaceState:
    total = meta.total_laps
    t_lap = float(lt[lap])
    active: list[tuple[tuple[float, float, float], dict]] = []
    retired: list[tuple[tuple[float, float], dict]] = []

    for code, dl in by_driver.items():
        found = dl.row_at_or_before(lap)
        if found is None:
            continue
        k, row = found
        took_flag = lap == total and (dl.max_lap == total or code in classified)
        if took_flag:
            status = _FINISHED
        elif dl.max_lap < lap and code not in classified:
            status = "dnf"
        else:
            status = "running"
        time_s = _f(row["time_s"])
        record = {
            "code": code,
            "team": meta.drivers[code].team if code in meta.drivers else "",
            "status": status,
            "laps_completed": k,
            "time_s": time_s,
            "last_lap_s": _f(row["lap_time_s"]),
            "best_lap_s": _f(row["best_lap_s"]),
            "compound": str(row["compound"]),
            "tyre_age": int(_f(row["tyre_life"]) or 0),
            "stint": int(_f(row["stint"]) or 1),
            "pit_count": int(row["pit_count"]),
            "pitted_this_lap": bool(row["pit_in"]) and k == lap,
        }
        if status == "dnf":
            retired.append(((-k, time_s if time_s is not None else math.inf), record))
        else:
            ff1_pos = _f(row["position"]) or 99.0
            key = (-float(k), time_s if time_s is not None else math.inf, ff1_pos)
            active.append((key, record))

    active.sort(key=lambda kv: kv[0])
    if lap == total:
        # The chequered flag shows the official classification, so post-race time
        # penalties apply. Unclassified finishers (e.g. later DSQ) keep track order behind.
        rank = {r.code: int(r.classified) for r in meta.results if r.classified.isdigit()}
        order = {id(rec): i for i, (_, rec) in enumerate(active)}
        active.sort(
            key=lambda kv: (
                (0, rank[kv[1]["code"]]) if kv[1]["code"] in rank else (1, order[id(kv[1])])
            )
        )
    retired.sort(key=lambda kv: kv[0])
    dns = sorted(
        (c for c in meta.drivers if c not in by_driver), key=lambda c: meta.grid.get(c, 99)
    )

    # Raw on-track gaps first; then re-base on whoever is P1 (only differs at the flag).
    raw: list[tuple[dict, int, float | None, int]] = []
    for _, rec in active:
        k, time_s = rec.pop("laps_completed"), rec.pop("time_s")
        gap, laps_down = None, 0
        if time_s is not None and not math.isnan(lt[k]):
            gap = max(0.0, time_s - float(lt[k]))
            laps_done_by_leader = int(np.searchsorted(lt[1:], time_s, side="right"))
            laps_down = max(0, laps_done_by_leader - k)
        raw.append((rec, k, gap, laps_down))
    base = raw[0][2] if raw and raw[0][2] is not None else 0.0

    drivers: list[DriverState] = []
    prev_gap: float | None = None
    for pos, (rec, k, raw_gap, laps_down) in enumerate(raw, start=1):
        # After a penalty reshuffle a car can sit behind one it beat on track; its gap to
        # the classified leader is then not meaningful, so report none.
        gap = None if raw_gap is None or raw_gap < base else round(raw_gap - base, 3)
        if pos == 1:
            gap = 0.0
        interval = None
        if pos > 1 and gap is not None and prev_gap is not None and gap >= prev_gap:
            interval = round(gap - prev_gap, 3)
        drivers.append(
            DriverState(
                position=pos,
                laps_completed=k,
                gap_to_leader_s=gap,
                interval_s=interval,
                laps_down=laps_down,
                **rec,
            )
        )
        prev_gap = gap
    for pos, (_, rec) in enumerate(retired, start=len(drivers) + 1):
        k = rec.pop("laps_completed")
        rec.pop("time_s")
        drivers.append(DriverState(position=pos, laps_completed=k, **rec))
    for pos, code in enumerate(dns, start=len(drivers) + 1):
        drivers.append(
            DriverState(code=code, team=meta.drivers[code].team, position=pos, status="dns")
        )

    flags_seen = [f for f in flags.between(float(lt[lap - 1]), t_lap)]
    fastest = _fastest_lap(data.laps, lap)
    return RaceState(
        race_id=meta.race_id,
        lap=lap,
        total_laps=total,
        flag=flags.at(t_lap),  # type: ignore[arg-type]
        flags_this_lap=_dedupe_consecutive(flags_seen),  # type: ignore[arg-type]
        drivers=tuple(drivers),
        weather=_weather(weather.at(t_lap)),
        new_messages=tuple(messages.get(lap, ())),
        fastest_lap=fastest,
        session_time_s=t_lap,
    )


def _fastest_lap(laps: pd.DataFrame, lap: int) -> FastestLap | None:
    pool = laps[(laps["lap"] <= lap) & ~laps["deleted"] & laps["lap_time_s"].notna()]
    if pool.empty:
        return None
    best = pool.loc[pool["lap_time_s"].idxmin()]
    return FastestLap(
        code=str(best["driver"]), lap=int(best["lap"]), time_s=float(best["lap_time_s"])
    )
