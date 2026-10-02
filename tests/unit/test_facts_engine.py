from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core.events.types import Event
from core.facts.corpus import WikiCorpus, passages_from_article
from core.facts.engine import FactsEngine, load_race_facts
from core.facts.models import Fact
from core.facts.quiet import QuietPeriodDetector
from core.feed.replay import ReplayFeed
from core.models import Citation, Profile
from core.profiles.repository import SqlRepository
from core.rag.models import HashEmbedder

SRC = Citation(source_type="jolpica", label="Jolpica", url="https://api.jolpi.ca/x")


def fact(subject: str, kind: str, text: str, relevance: float, stype: str = "driver") -> Fact:
    return Fact(
        id=f"2099_01:{kind}:{subject}",
        race_id="2099_01",
        kind=kind,  # type: ignore[arg-type]
        subject_type=stype,
        subject=subject,
        text=text,
        source=SRC,  # type: ignore[arg-type]
        relevance=relevance,
        values={},
    )


FACTS = [
    fact("AAA", "career_wins", "Driver AAA has 3 wins.", 0.6),
    fact("AAA", "circuit_wins", "Driver AAA has won here 2 times.", 1.0),
    fact("BBB", "circuit_debut", "This is Driver BBB's first race here.", 0.9),
    fact("teama", "team_circuit_wins", "Team A has won here 4 times.", 0.85, "team"),
]


def corpus() -> WikiCorpus:
    text = (
        "Driver AAA started karting in 2001 and won a junior title in 2005.\n\n"
        "== Later ==\nIn 2101 Driver AAA did something in the future that is a spoiler."
    )
    ps = passages_from_article(
        "driver",
        "AAA",
        "Driver AAA",
        "Driver AAA",
        "https://w/aaa",
        text.replace(".", ". This sentence pads the passage length enough to be kept.", 1),
    )
    emb = HashEmbedder()
    return WikiCorpus(ps, emb.embed_passages([p.text for p in ps]), emb.name)


@pytest.fixture
def engine(fake_feed: ReplayFeed, repo: SqlRepository) -> FactsEngine:
    return FactsEngine(fake_feed.meta, FACTS, corpus(), repo)


def profile(**kw: object) -> Profile:
    base: dict[str, object] = {
        "nickname": "fan",
        "favourite_team": "Team A",
        "favourite_drivers": ("AAA",),
    }
    return Profile(**{**base, **kw})


def test_quiet_fact_prefers_favourite_and_relevance(engine: FactsEngine) -> None:
    card = engine.next_fact(profile(), "s1")
    assert card is not None and card.fact_id == "2099_01:circuit_wins:AAA"
    assert card.text == "Driver AAA has won here 2 times." and card.phrased_by == "template"
    assert card.source.url == "https://api.jolpi.ca/x" and card.subject == "Driver AAA"


def test_no_repeats_then_wiki_then_none(engine: FactsEngine) -> None:
    seen = [engine.next_fact(profile(favourite_team="Nobody"), "s1") for _ in range(4)]
    ids = [c.fact_id if c else None for c in seen]
    assert ids[:2] == ["2099_01:circuit_wins:AAA", "2099_01:career_wins:AAA"]
    assert ids[2] == "wiki:AAA:0"  # stats exhausted -> Wikipedia
    assert seen[2] is not None and seen[2].origin == "wikipedia"
    assert seen[2].source.label.startswith("Wikipedia: Driver AAA")
    assert ids[3] is None  # the 2101 passage is a "spoiler" for a 2099 race: never shown
    # A different session starts fresh.
    other = engine.next_fact(profile(favourite_team="Nobody"), "s2")
    assert other is not None and other.fact_id == ids[0]


def test_team_fact_when_no_favourite_driver_facts(engine: FactsEngine) -> None:
    card = engine.next_fact(profile(favourite_drivers=()), "s1")
    assert card is not None and card.fact_id == "2099_01:team_circuit_wins:teama"


def test_on_request_resolves_subjects(engine: FactsEngine) -> None:
    card = engine.next_fact(profile(), "s1", subject_text="tell me something about BBB")
    assert card is not None and card.fact_id == "2099_01:circuit_debut:BBB"
    by_name = engine.next_fact(profile(), "s1", subject_text="what about Driver AAA?")
    assert by_name is not None and by_name.subject == "Driver AAA"
    assert engine.next_fact(profile(), "s1", subject_text="the weather") is None
    assert engine.resolve_subject("Team A news") == "teama"


def test_llm_rephrasing_is_guarded(fake_feed: ReplayFeed, repo: SqlRepository) -> None:
    good = FakeListChatModel(responses=["Your driver AAA has won here twice before - 2 wins!"])
    card = FactsEngine(fake_feed.meta, FACTS, None, repo, llm=good).next_fact(profile(), "s1")
    assert card is not None and card.phrased_by == "llm"

    bad = FakeListChatModel(responses=["AAA has 5 wins here.", "AAA has 6 wins here."])
    card = FactsEngine(fake_feed.meta, FACTS, None, repo, llm=bad).next_fact(profile(), "s2")
    assert card is not None and card.phrased_by == "template"
    assert card.text == "Driver AAA has won here 2 times."


def test_facts_from_other_races_are_ignored(fake_feed: ReplayFeed, repo: SqlRepository) -> None:
    other = fact("AAA", "career_wins", "x", 1.0).model_copy(update={"race_id": "1999_01"})
    assert FactsEngine(fake_feed.meta, [other], None, repo).next_fact(profile(), "s") is None


def test_load_drops_unsourced_facts(tmp_path: Path) -> None:
    ok = fact("AAA", "career_wins", "ok", 0.5).model_dump(mode="json")
    bad = {**ok, "id": "bad", "source": {**ok["source"], "url": None}}
    (tmp_path / "2099_01.json").write_text(json.dumps([ok, bad]))
    assert [f.id for f in load_race_facts("2099_01", tmp_path)] == [ok["id"]]
    assert load_race_facts("nope", tmp_path) == []


def test_fact_model_requires_source_url() -> None:
    with pytest.raises(ValueError):
        Fact(
            id="x",
            race_id="r",
            kind="career_wins",
            subject_type="driver",
            subject="A",
            text="t",
            source=Citation(source_type="jolpica", label="j"),
        )


# ------------------------------------------------------------- quiet period
def _ev(importance: float) -> Event:
    return Event(
        id=f"e{importance}",
        race_id="r",
        lap=1,
        type="overtake",
        summary="s",
        base_importance=importance,
    )


def test_quiet_period_triggers_once_per_spell() -> None:
    q = QuietPeriodDetector(laps=3, threshold=0.5)
    assert [q.update(lap, []) for lap in (1, 2, 3, 4, 5, 6)] == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_significant_event_resets_quiet_count() -> None:
    q = QuietPeriodDetector(laps=2, threshold=0.5)
    assert not q.update(1, [])
    assert not q.update(2, [_ev(0.7)])  # significant: reset
    assert not q.update(3, [_ev(0.3)])  # minor events still count as quiet
    assert q.update(4, [])


def test_quiet_period_resets_on_seek_and_ignores_grid() -> None:
    q = QuietPeriodDetector(laps=2)
    assert not q.update(0, [])
    q.update(5, [])
    assert not q.update(3, [])  # went backwards: counting restarts
    assert q.update(4, [])
    with pytest.raises(ValueError):
        QuietPeriodDetector(laps=0)
