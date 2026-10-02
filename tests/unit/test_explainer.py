from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage

from core.agents.explainer import Explainer, reg_query, regs_year
from core.events.types import Event
from core.feed.replay import ReplayFeed
from core.rag.index import IndexedChunk
from core.rag.retriever import RetrievedChunk


class RecordingLLM(FakeListChatModel):
    """FakeListChatModel that also records the prompts it was given."""

    seen: list[list[BaseMessage]] = []  # noqa: RUF012 - pydantic field default, reset per test

    def invoke(self, input: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        self.seen.append(list(input))
        return super().invoke(input, *args, **kwargs)


class FailingLLM(FakeListChatModel):
    def invoke(self, input: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        raise TimeoutError("rate limited")


class StubRetriever:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int | None, str | None]] = []

    def search(
        self, query: str, k: int = 5, year: int | None = None, doc_type: str | None = None, **_: Any
    ) -> list[RetrievedChunk]:
        self.queries.append((query, year, doc_type))
        chunk = IndexedChunk(
            chunk_id="2025_sporting:55.1:0",
            doc_id="2025_sporting",
            doc_title="2025 F1 Sporting Regulations (Issue 5)",
            doc_type="sporting",
            year=2025,
            article="55.1",
            title="SAFETY CAR",
            page=64,
            text="The safety car will be used to neutralise a race.",
            url="https://www.fia.com/x.pdf",
        )
        return [RetrievedChunk(chunk, 1.0, "rerank")]


def _sc_event() -> Event:
    return Event(
        id="2099_01:4:safety_car:deployed",
        race_id="2099_01",
        lap=4,
        type="safety_car",
        summary="Safety car deployed on lap 4.",
        base_importance=0.8,
        payload={"phase": "deployed"},
    )


def _pit_event() -> Event:
    return Event(
        id="2099_01:5:pit_stop:AAA",
        race_id="2099_01",
        lap=5,
        type="pit_stop",
        drivers=("AAA",),
        teams=("Team A",),
        summary="AAA pits under SC from P2: MEDIUM -> HARD, rejoins P2.",
        base_importance=0.65,
        payload={"stop_number": 1, "old_tyre_age": 4, "position_after": 2},
    )


def test_no_llm_uses_template_with_rule_citation() -> None:
    exp = Explainer(None, StubRetriever()).explain(_sc_event(), "rookie")  # type: ignore[arg-type]
    assert exp.source == "template" and exp.text == "Safety car deployed on lap 4."
    assert exp.citations[0].article == "55.1" and exp.citations[0].page == 64


def test_llm_answer_with_citation(fake_feed: ReplayFeed) -> None:
    llm = RecordingLLM(responses=["The safety car came out on lap 4 to neutralise the race [S1]."])
    llm.seen = []
    retriever = StubRetriever()
    exp = Explainer(llm, retriever).explain(  # type: ignore[arg-type]
        _sc_event(), "fan", fake_feed.state(4), fake_feed.meta
    )
    assert exp.source == "llm" and exp.guard_passed and exp.retries == 0
    assert [c.label for c in exp.citations] == [
        "2025 F1 Sporting Regulations (Issue 5), Art. 55.1, p. 64"
    ]
    system, human = llm.seen[0]
    assert "Level: Fan" in system.content and "Never do arithmetic" in system.content
    assert "[S1] 2025 F1 Sporting Regulations" in human.content
    assert retriever.queries[0][1:] == (2026, "sporting")  # 2099 season -> latest regs year


def test_snapshot_includes_involved_driver(fake_feed: ReplayFeed) -> None:
    llm = RecordingLLM(responses=["Your driver AAA switched to hards and is still P2."])
    llm.seen = []
    exp = Explainer(llm, None).explain(_pit_event(), "rookie", fake_feed.state(5), fake_feed.meta)
    assert exp.source == "llm" and exp.citations == ()
    human = llm.seen[0][1].content
    assert '"name": "Driver AAA"' in human and '"compound": "HARD"' in human


def test_guard_retry_then_success(fake_feed: ReplayFeed) -> None:
    llm = RecordingLLM(responses=["AAA lost 7.5 seconds in the stop.", "AAA pitted for hards."])
    llm.seen = []
    exp = Explainer(llm, None).explain(_pit_event(), "expert", fake_feed.state(5), fake_feed.meta)
    assert exp.source == "llm" and exp.retries == 1 and exp.text == "AAA pitted for hards."
    assert "7.5" in llm.seen[1][-1].content  # the retry names the offending number


def test_guard_failure_falls_back_to_template(fake_feed: ReplayFeed) -> None:
    llm = RecordingLLM(responses=["Lost 7.5s.", "Lost 8.5s."])
    llm.seen = []
    exp = Explainer(llm, None).explain(_pit_event(), "fan", fake_feed.state(5), fake_feed.meta)
    assert exp.source == "template" and not exp.guard_passed and exp.retries == 1


def test_llm_error_falls_back_to_template() -> None:
    exp = Explainer(FailingLLM(responses=[]), None).explain(_pit_event(), "fan")
    assert exp.source == "template"


def test_empty_llm_reply_falls_back() -> None:
    exp = Explainer(FakeListChatModel(responses=["  "]), None).explain(_pit_event(), "fan")
    assert exp.source == "template"


def test_unknown_marker_is_not_cited() -> None:
    llm = FakeListChatModel(responses=["Safety car rules apply [S1] [S9]."])
    exp = Explainer(llm, StubRetriever()).explain(_sc_event(), "fan")  # type: ignore[arg-type]
    assert [c.article for c in exp.citations] == ["55.1"]


@pytest.mark.parametrize(
    ("payload", "type_", "expected"),
    [
        (
            {"kind": "time", "reason": "CAUSING A COLLISION"},
            "penalty",
            "time penalty served at pit stop or added to race time causing a collision",
        ),
        ({"kind": "drive_through", "reason": None}, "penalty", "drive through penalty"),
        (
            {"message": "CAR 1 (AAA) UNDER INVESTIGATION - LEAVING THE TRACK"},
            "race_control",
            "incidents investigation by the stewards leaving the track",
        ),
        (
            {"phase": "deployed"},
            "vsc",
            "virtual safety car deployed drivers must stay above the minimum delta time",
        ),
        ({"phase": "ended"}, "red_flag", "resuming a race after suspension restart procedure"),
        ({}, "overtake", None),
    ],
)
def test_reg_query(payload: dict, type_: str, expected: str | None) -> None:
    e = Event(
        id="x",
        race_id="r",
        lap=1,
        type=type_,
        summary="s",
        base_importance=0.5,  # type: ignore[arg-type]
        payload=payload,
    )
    assert reg_query(e) == expected


def test_regs_year() -> None:
    assert regs_year(2023) == 2025 and regs_year(2025) == 2025 and regs_year(2026) == 2026
