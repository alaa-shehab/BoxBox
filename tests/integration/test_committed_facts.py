"""The committed showcase facts: sourced, race-scoped, and free of the race's own result."""

from __future__ import annotations

import pytest

from core.facts.engine import load_corpus, load_race_facts
from core.feed.builder import race_id
from core.feed.catalog import SHOWCASE_RACES

IDS = [race_id(s, r) for s, r in SHOWCASE_RACES]


@pytest.mark.parametrize("rid", IDS)
def test_facts_are_sourced_and_scoped(rid: str) -> None:
    facts = load_race_facts(rid)
    assert len(facts) >= 60
    assert all(f.race_id == rid and f.source.url for f in facts)
    assert all(f.source.url.startswith("https://api.jolpi.ca/") for f in facts)
    assert len({f.id for f in facts}) == len(facts)


def test_no_leak_of_the_replayed_result() -> None:
    """Verstappen won the 2024 São Paulo GP: going into it he had 2 wins there and 61
    career wins, and those are the numbers the facts must show."""
    facts = {(f.kind, f.subject): f for f in load_race_facts("2024_21")}
    assert facts[("circuit_wins", "VER")].values["wins"] == 2
    assert facts[("career_wins", "VER")].values["wins"] == 61
    assert facts[("championship_position", "NOR")].values["position"] == 2


def test_las_vegas_2023_was_an_inaugural_race() -> None:
    facts = load_race_facts("2023_21")
    assert any(f.kind == "circuit_first_race" and f.values["races"] == 0 for f in facts)


def test_wiki_corpus_if_built_is_licensed_and_linked() -> None:
    corpus = load_corpus()
    if corpus is None:
        pytest.skip("Wikipedia corpus not built yet (built on GitHub Actions)")
    assert all(p.url.startswith("https://en.wikipedia.org/w/index.php?") for p in corpus.passages)
    assert all("CC BY-SA" in p.citation_label for p in corpus.passages)
