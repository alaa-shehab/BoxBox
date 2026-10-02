"""Compact, committed replay format.

A replay is five small files per race in `data/replays/{race_id}/`:
    meta.json            RaceMeta (drivers, grid, classification)
    laps.parquet         one row per driver-lap
    messages.parquet     race control messages, bucketed by lap
    track_status.parquet flag timeline (seconds from session start)
    weather.parquet      weather timeline (seconds from session start)

The format is independent of FastF1, so the deployed app never needs FastF1 or the
network to replay a committed race.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.models import RaceMeta

LAP_COLUMNS = {
    "driver": "string",
    "lap": "int16",
    "time_s": "float64",  # session time when the lap was completed
    "lap_time_s": "float64",
    "position": "float32",  # FastF1's own position (NaN when unknown); tie-breaker only
    "compound": "string",
    "tyre_life": "float32",
    "stint": "float32",
    "pit_in": "bool",  # entered the pit lane at the end of this lap
    "pit_out": "bool",  # left the pit lane at the start of this lap
    "deleted": "bool",  # lap time deleted (track limits), excluded from fastest lap
}
MESSAGE_COLUMNS = {
    "lap": "int16",
    "category": "string",
    "message": "string",
    "flag": "string",
    "drivers": "string",  # comma-separated driver codes mentioned in the message
}
TRACK_STATUS_COLUMNS = {"time_s": "float64", "flag": "string"}
WEATHER_COLUMNS = {
    "time_s": "float64",
    "air_temp_c": "float32",
    "track_temp_c": "float32",
    "humidity_pct": "float32",
    "rainfall": "bool",
}


def conform(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Select and type exactly the schema columns, in order."""
    missing = set(schema) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df[list(schema)].astype(schema).reset_index(drop=True)


@dataclass(frozen=True)
class ReplayData:
    meta: RaceMeta
    laps: pd.DataFrame
    messages: pd.DataFrame
    track_status: pd.DataFrame
    weather: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "laps", conform(self.laps, LAP_COLUMNS))
        object.__setattr__(self, "messages", conform(self.messages, MESSAGE_COLUMNS))
        object.__setattr__(self, "track_status", conform(self.track_status, TRACK_STATUS_COLUMNS))
        object.__setattr__(self, "weather", conform(self.weather, WEATHER_COLUMNS))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meta.json").write_text(self.meta.model_dump_json(indent=1) + "\n")
        for name in ("laps", "messages", "track_status", "weather"):
            getattr(self, name).to_parquet(
                directory / f"{name}.parquet", index=False, compression="zstd"
            )
        return directory

    @classmethod
    def load(cls, directory: Path) -> ReplayData:
        return cls(
            meta=RaceMeta.model_validate_json((directory / "meta.json").read_text()),
            laps=pd.read_parquet(directory / "laps.parquet"),
            messages=pd.read_parquet(directory / "messages.parquet"),
            track_status=pd.read_parquet(directory / "track_status.parquet"),
            weather=pd.read_parquet(directory / "weather.parquet"),
        )
