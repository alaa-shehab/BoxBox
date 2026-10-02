"""Small Jolpica (Ergast-compatible) client: paginated, throttled, retried, disk-cached.

Jolpica is a free, volunteer-run API, so the client is deliberately polite: at most one
request per `min_interval_s`, exponential backoff on 429/5xx, and every response cached
on disk. Rebuilding facts therefore costs nothing after the first run.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.log import get_logger

log = get_logger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE = 100


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class JolpicaClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        min_interval_s: float = 0.35,
        timeout_s: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._http = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={
                "User-Agent": "PitWall-fan-project/0.1 (+https://github.com/alaa-shehab/BoxBox)"
            },
        )

    def url(self, path: str) -> str:
        return f"{BASE_URL}/{path.strip('/')}.json"

    def _cache_file(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _fetch(self, url: str) -> dict[str, Any]:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        resp = self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    def get(self, path: str, offset: int = 0, limit: int = PAGE) -> dict[str, Any]:
        url = f"{self.url(path)}?limit={limit}&offset={offset}"
        cached = self._cache_file(url)
        if cached is not None and cached.exists():
            return json.loads(cached.read_text())
        data = self._fetch(url)["MRData"]
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(data))
        return data

    def races(self, path: str) -> list[dict[str, Any]]:
        """All `RaceTable.Races` for a results-style endpoint, across pages.

        Ergast paginates over result rows, so a race can span two pages: merge them."""
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        offset = 0
        while True:
            data = self.get(path, offset=offset)
            for race in data.get("RaceTable", {}).get("Races", []):
                key = (race["season"], race["round"])
                if key in merged:
                    for field in ("Results", "QualifyingResults"):
                        merged[key].setdefault(field, []).extend(race.get(field, []))
                else:
                    merged[key] = race
            offset += int(data["limit"])
            if offset >= int(data["total"]):
                break
        return sorted(merged.values(), key=lambda r: (int(r["season"]), int(r["round"])))

    def standings(self, season: int, round_number: int) -> list[dict[str, Any]]:
        data = self.get(f"{season}/{round_number}/driverStandings")
        lists = data.get("StandingsTable", {}).get("StandingsLists", [])
        return lists[0]["DriverStandings"] if lists else []

    def close(self) -> None:
        self._http.close()
