from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from core.facts.jolpica import JolpicaClient
from core.facts.wikipedia import WikipediaClient, title_from_url


def _page(offset: int, total: int, races: list[dict]) -> dict:
    return {
        "MRData": {
            "limit": "2",
            "offset": str(offset),
            "total": str(total),
            "RaceTable": {"Races": races},
        }
    }


def test_jolpica_paginates_merges_and_caches(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        offset = int(request.url.params["offset"])
        if offset == 0:
            body = _page(0, 3, [{"season": "2024", "round": "1", "Results": [{"p": 1}, {"p": 2}]}])
        else:
            body = _page(2, 3, [{"season": "2024", "round": "1", "Results": [{"p": 3}]}])
        return httpx.Response(200, json=body)

    client = JolpicaClient(
        cache_dir=tmp_path, min_interval_s=0, transport=httpx.MockTransport(handler)
    )
    (merged,) = client.races("circuits/x/results")
    assert [r["p"] for r in merged["Results"]] == [1, 2, 3]  # race split across pages
    assert len(calls) == 2
    client.races("circuits/x/results")
    assert len(calls) == 2  # served from the disk cache
    assert client.url("2024/results") == "https://api.jolpi.ca/ergast/f1/2024/results.json"


def test_jolpica_retries_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    responses = iter([httpx.Response(429), httpx.Response(200, json=_page(0, 0, []))])
    client = JolpicaClient(
        min_interval_s=0, transport=httpx.MockTransport(lambda r: next(responses))
    )
    assert client.races("2024/results") == []


def test_jolpica_standings_empty() -> None:
    body = {
        "MRData": {
            "limit": "100",
            "offset": "0",
            "total": "0",
            "StandingsTable": {"StandingsLists": []},
        }
    }
    client = JolpicaClient(
        min_interval_s=0, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    )
    assert client.standings(2024, 1) == []


def test_wikipedia_honours_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("core.facts.wikipedia.time.sleep", slept.append)
    page = {
        "query": {
            "pages": [
                {
                    "title": "Max Verstappen",
                    "revisions": [{"revid": 123}],
                    "extract": "Max is a driver.",
                }
            ]
        }
    }
    responses = iter(
        [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, json=page)]
    )
    client = WikipediaClient(
        min_interval_s=0, transport=httpx.MockTransport(lambda r: next(responses))
    )
    article = client.article("Max Verstappen")
    assert article is not None and article.revision_id == 123
    assert article.permalink == (
        "https://en.wikipedia.org/w/index.php?title=Max_Verstappen&oldid=123"
    )
    assert 7.0 in slept


def test_wikipedia_missing_page_and_hard_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.facts.wikipedia.time.sleep", lambda s: None)
    missing = {"query": {"pages": [{"title": "Nope", "missing": True}]}}
    client = WikipediaClient(
        min_interval_s=0, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=missing))
    )
    assert client.article("Nope") is None
    blocked = WikipediaClient(
        min_interval_s=0,
        transport=httpx.MockTransport(
            lambda r: httpx.Response(403, text="Please respect our robot policy")
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        blocked.article("Max Verstappen")


def test_title_from_url() -> None:
    assert title_from_url("http://en.wikipedia.org/wiki/Sergio_P%C3%A9rez") == "Sergio Pérez"
    assert json.dumps(title_from_url("https://en.wikipedia.org/wiki/Red_Bull_Racing"))
