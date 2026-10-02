"""Build a ReplayData from a FastF1 race session.

`from_session` is a pure transform over FastF1's dataframes, so it is unit-tested with a
fake session. `build_from_fastf1` adds the (lazy) FastF1 import and download.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.feed.data import ReplayData
from core.log import get_logger
from core.models import COMPOUNDS, ClassifiedResult, DriverInfo, RaceMeta

log = get_logger(__name__)

# FastF1 track status codes -> our flags. "7" (VSC ending) still runs under VSC rules.
TRACK_STATUS_TO_FLAG = {"1": "GREEN", "2": "YELLOW", "4": "SC", "5": "RED", "6": "VSC", "7": "VSC"}
_CODE_IN_MESSAGE = re.compile(r"\(([A-Z]{3})\)")


def race_id(season: int, round_number: int) -> str:
    return f"{season}_{round_number:02d}"


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def _seconds(series: pd.Series) -> pd.Series:
    return series.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(series) else series


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _laps(session: Any) -> pd.DataFrame:
    raw = session.laps
    raw = raw[raw["LapNumber"].notna() & raw["Driver"].notna()]
    compound = raw["Compound"].astype("string").str.upper().fillna("UNKNOWN")
    return pd.DataFrame(
        {
            "driver": raw["Driver"].astype(str),
            "lap": raw["LapNumber"].astype(int),
            "time_s": _seconds(raw["Time"]),
            "lap_time_s": _seconds(raw["LapTime"]),
            "position": raw["Position"],
            "compound": compound.where(compound.isin(COMPOUNDS), "UNKNOWN"),
            "tyre_life": raw["TyreLife"],
            "stint": raw["Stint"],
            "pit_in": raw["PitInTime"].notna(),
            "pit_out": raw["PitOutTime"].notna(),
            "deleted": raw["Deleted"].fillna(False).astype(bool),
        }
    ).sort_values(["driver", "lap"])


def _messages(session: Any, number_to_code: dict[str, str]) -> pd.DataFrame:
    raw = session.race_control_messages
    rows = []
    for rec in raw.to_dict("records"):
        message = str(rec.get("Message") or "")
        codes = list(dict.fromkeys(_CODE_IN_MESSAGE.findall(message)))
        number = rec.get("RacingNumber")
        if number is not None and str(number) in number_to_code:
            codes.insert(0, number_to_code[str(number)])
        flag = rec.get("Flag")
        rows.append(
            {
                "lap": int(_num(rec.get("Lap")) or 1),
                "category": str(rec.get("Category") or "Other"),
                "message": message,
                "flag": None if flag is None or pd.isna(flag) else str(flag),
                "drivers": ",".join(dict.fromkeys(codes)),
            }
        )
    return pd.DataFrame(rows, columns=["lap", "category", "message", "flag", "drivers"])


def _track_status(session: Any) -> pd.DataFrame:
    raw = session.track_status
    flags = raw["Status"].astype(str).map(TRACK_STATUS_TO_FLAG)
    df = pd.DataFrame({"time_s": _seconds(raw["Time"]), "flag": flags})
    return df[df["flag"].notna()].sort_values("time_s")


def _weather(session: Any) -> pd.DataFrame:
    raw = session.weather_data
    return pd.DataFrame(
        {
            "time_s": _seconds(raw["Time"]),
            "air_temp_c": raw["AirTemp"],
            "track_temp_c": raw["TrackTemp"],
            "humidity_pct": raw["Humidity"],
            "rainfall": raw["Rainfall"].fillna(False).astype(bool),
        }
    ).sort_values("time_s")


def _grid_slots(results: pd.DataFrame) -> dict[str, int]:
    """Grid slot per driver. FastF1 uses 0/NaN for pit-lane starts; those go to the back."""
    grid = {str(r["Abbreviation"]): (_num(r["GridPosition"]) or 0.0) for _, r in results.iterrows()}
    ordered = sorted(grid, key=lambda c: (grid[c] <= 0, grid[c]))
    return {code: slot for slot, code in enumerate(ordered, start=1)}


def from_session(session: Any, season: int, round_number: int) -> ReplayData:
    results = session.results
    event = session.event
    drivers = {
        str(r["Abbreviation"]): DriverInfo(
            code=str(r["Abbreviation"]),
            number=str(r["DriverNumber"]),
            name=str(r["FullName"]),
            team=str(r["TeamName"]),
            team_colour=str(r["TeamColor"] or "888888"),
        )
        for _, r in results.iterrows()
    }
    number_to_code = {d.number: d.code for d in drivers.values()}
    laps = _laps(session)
    weather = _weather(session)
    classified = [
        ClassifiedResult(
            code=str(r["Abbreviation"]),
            position=int(p) if (p := _num(r["Position"])) is not None else None,
            classified=str(r["ClassifiedPosition"]),
            status=str(r["Status"]),
            grid=int(g) if (g := _num(r["GridPosition"])) is not None else None,
            points=_num(r["Points"]) or 0.0,
            laps=int(_num(r["Laps"]) or 0),
        )
        for _, r in results.iterrows()
    ]
    is_wet = bool(weather["rainfall"].any()) or bool(
        laps["compound"].isin(["INTERMEDIATE", "WET"]).any()
    )
    scheduled = _num(getattr(session, "total_laps", None))
    meta = RaceMeta(
        race_id=race_id(season, round_number),
        season=season,
        round=round_number,
        event_name=str(event["EventName"]),
        country=str(event["Country"]),
        circuit_key=slugify(str(event["Location"])),
        total_laps=int(laps["lap"].max()),
        scheduled_laps=int(scheduled) if scheduled is not None else None,
        is_wet=is_wet,
        drivers=drivers,
        grid=_grid_slots(results),
        results=classified,
    )
    return ReplayData(
        meta=meta,
        laps=laps,
        messages=_messages(session, number_to_code),
        track_status=_track_status(session),
        weather=weather,
    )


def build_from_fastf1(season: int, round_number: int, cache_dir: Path) -> ReplayData:
    """Download (or read from FastF1's cache) a race and convert it. Needs network."""
    import fastf1  # lazy: heavy import, only needed for building

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    session = fastf1.get_session(season, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    data = from_session(session, season, round_number)
    log.info(
        "replay.built",
        extra={"fields": {"race_id": data.meta.race_id, "event": data.meta.event_name}},
    )
    return data
