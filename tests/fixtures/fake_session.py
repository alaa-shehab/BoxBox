"""A tiny, hand-checked fake of a FastF1 race session (6 laps, 5 drivers).

Story (session time in seconds; the race starts at t=1000):
  AAA  P1 from pole, overtaken by BBB on lap 3, pits end of lap 4 (MEDIUM -> HARD).
  BBB  SOFT all race, takes the lead on lap 3, wins. Lap 2 time is deleted.
  CCC  retires after lap 2.
  DDD  pit-lane start, slow, one lap down from lap 4, classified P3 "+1 Lap".
  EEE  did not start (no laps).
Safety car from t=1300 (during lap 4) to t=1400 (during lap 5). Rain from t=1400.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LAP_END = {
    "AAA": [1090.0, 1180.2, 1271.0, 1361.5, 1475.0, 1565.0],
    "BBB": [1091.0, 1180.5, 1270.2, 1360.0, 1452.0, 1542.0],
    "CCC": [1092.0, 1183.0],
    "DDD": [1110.0, 1250.0, 1390.0, 1530.0, 1670.0],
}
COMPOUND = {
    "AAA": ["MEDIUM"] * 4 + ["HARD"] * 2,
    "BBB": ["SOFT"] * 6,
    "CCC": ["MEDIUM", "TEST_UNKNOWN"],
    "DDD": ["HARD"] * 5,
}
TYRE_LIFE = {
    "AAA": [1, 2, 3, 4, 1, 2],
    "BBB": [1, 2, 3, 4, 5, 6],
    "CCC": [3, 4],
    "DDD": [1, 2, 3, 4, 5],
}
STINT = {"AAA": [1, 1, 1, 1, 2, 2], "BBB": [1] * 6, "CCC": [1, 1], "DDD": [1] * 5}
RACE_START = 1000.0


def _td(seconds: float | None) -> pd.Timedelta:
    return pd.NaT if seconds is None else pd.Timedelta(seconds=seconds)


def _laps() -> pd.DataFrame:
    rows = []
    for code, ends in LAP_END.items():
        prev = RACE_START
        for i, end in enumerate(ends):
            lap = i + 1
            rows.append(
                {
                    "Driver": code,
                    "LapNumber": float(lap),
                    "Time": _td(end),
                    "LapTime": _td(end - prev),
                    "Position": np.nan if code == "DDD" else 1.0,  # unreliable on purpose
                    "Compound": COMPOUND[code][i],
                    "TyreLife": float(TYRE_LIFE[code][i]),
                    "Stint": float(STINT[code][i]),
                    "PitInTime": _td(end) if (code == "AAA" and lap == 4) else pd.NaT,
                    "PitOutTime": _td(end - 100) if (code == "AAA" and lap == 5) else pd.NaT,
                    "Deleted": code == "BBB" and lap == 2,
                }
            )
            prev = end
    rows.append({**rows[0], "LapNumber": np.nan})  # junk row without a lap number
    return pd.DataFrame(rows)


def _results() -> pd.DataFrame:
    def r(
        code: str, num: str, grid: float, pos: float, cls: str, status: str, pts: float, laps: float
    ) -> dict[str, Any]:
        return {
            "Abbreviation": code,
            "DriverNumber": num,
            "FullName": f"Driver {code}",
            "TeamName": f"Team {code[0]}",
            "TeamColor": "123456",
            "GridPosition": grid,
            "Position": pos,
            "ClassifiedPosition": cls,
            "Status": status,
            "Points": pts,
            "Laps": laps,
        }

    return pd.DataFrame(
        [
            r("BBB", "2", 2.0, 1.0, "1", "Finished", 25.0, 6.0),
            r("AAA", "1", 1.0, 2.0, "2", "Finished", 18.0, 6.0),
            r("DDD", "4", 0.0, 3.0, "3", "+1 Lap", 15.0, 5.0),
            r("CCC", "3", 3.0, 4.0, "R", "Retired", 0.0, 2.0),
            r("EEE", "5", 5.0, 5.0, "W", "Did not start", 0.0, 0.0),
        ]
    )


def _messages() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Lap": 1,
                "Category": "Flag",
                "Message": "GREEN LIGHT - PIT EXIT OPEN",
                "Flag": "GREEN",
                "RacingNumber": None,
            },
            {
                "Lap": 3,
                "Category": "Other",
                "Message": "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 2 (BBB) - "
                "CAUSING A COLLISION",
                "Flag": None,
                "RacingNumber": None,
            },
            {
                "Lap": 4,
                "Category": "SafetyCar",
                "Message": "SAFETY CAR DEPLOYED",
                "Flag": None,
                "RacingNumber": None,
            },
            {
                "Lap": 4,
                "Category": "Flag",
                "Message": "WAVED BLUE FLAG FOR CAR 4 (DDD)",
                "Flag": "BLUE",
                "RacingNumber": "4",
            },
        ]
    )


def _track_status() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [_td(0), _td(1300), _td(1400), _td(1450)],
            "Status": ["1", "4", "1", "3"],  # "3" is an unused code and must be ignored
            "Message": ["AllClear", "SCDeployed", "AllClear", "?"],
        }
    )


def _weather() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [_td(900), _td(1200), _td(1400)],
            "AirTemp": [20.0, 20.5, 18.0],
            "TrackTemp": [30.0, 31.0, 24.0],
            "Humidity": [60.0, 62.0, 90.0],
            "Rainfall": [False, False, True],
        }
    )


class FakeSession:
    def __init__(self, results: pd.DataFrame | None = None) -> None:
        self.laps = _laps()
        self.results = _results() if results is None else results
        self.race_control_messages = _messages()
        self.track_status = _track_status()
        self.weather_data = _weather()
        self.event = {"EventName": "Test Grand Prix", "Country": "Brazil", "Location": "São Paulo"}
        self.total_laps = 6
