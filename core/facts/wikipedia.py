"""Fetch a Wikipedia article's plain text and its exact revision (for a stable permalink)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import httpx

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "PitWall-fan-project/0.1 (+https://github.com/alaa-shehab/BoxBox; educational)"


@dataclass(frozen=True)
class Article:
    title: str
    revision_id: int
    extract: str

    @property
    def permalink(self) -> str:
        return (
            f"https://en.wikipedia.org/w/index.php?title={self.title.replace(' ', '_')}"
            f"&oldid={self.revision_id}"
        )


def title_from_url(url: str) -> str:
    return unquote(urlparse(url).path.rsplit("/", 1)[-1]).replace("_", " ")


class WikipediaClient:
    def __init__(
        self, min_interval_s: float = 2.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._http = httpx.Client(
            timeout=30, transport=transport, headers={"User-Agent": USER_AGENT}
        )

    def _get(self, params: dict[str, str], attempts: int = 6) -> httpx.Response:
        """GET with politeness: a minimum interval, and Retry-After on 429/403/5xx."""
        for attempt in range(attempts):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self._http.get(API, params=params)
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                retry_after = resp.headers.get("retry-after", "")
                delay = float(retry_after) if retry_after.isdigit() else 2.0 * 2**attempt
                time.sleep(min(delay, 60.0))
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()
        return resp

    def article(self, title: str) -> Article | None:
        resp = self._get(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "redirects": "1",
                "prop": "extracts|revisions",
                "explaintext": "1",
                "exsectionformat": "wiki",
                "rvprop": "ids",
                "titles": title,
            }
        )
        pages = resp.json().get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        page = pages[0]
        return Article(
            title=page["title"],
            revision_id=int(page["revisions"][0]["revid"]),
            extract=page.get("extract", ""),
        )

    def close(self) -> None:
        self._http.close()
