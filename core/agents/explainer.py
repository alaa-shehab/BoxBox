"""Explainer: turns a detected event into a level-appropriate explanation.

Pipeline for one event:
  1. Build a structured context: the event's own numbers, a short snapshot of the involved
     drivers from the RaceState, and, for rule events, the top regulation chunks retrieved
     as [S1], [S2], ...
  2. Ask the LLM (system prompt = grounding rules + the fan's level style).
  3. Numeric guard: every number in the answer must come from that context. On failure,
     regenerate once with the offending numbers named, then fall back to the template.
  4. Any LLM error (rate limit, timeout, no provider) also falls back to the template.

The template is the event's deterministic summary, so the feed is never empty or wrong.
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict

from core.agents.guard import check_numbers
from core.events.types import Event
from core.log import get_logger
from core.models import Citation, Level, RaceMeta, RaceState
from core.prompts import system_prompt
from core.rag.retriever import RegRetriever, RetrievedChunk

log = get_logger(__name__)

TASK = """\
Task: explain the race event below to this fan in 2-4 sentences.
- Say what happened and why it matters for the race.
- Use the driver names and numbers from EVENT and RACE SNAPSHOT only.
- If REGULATIONS are provided and relevant, explain the rule and cite it inline with its
  marker, e.g. [S1]. Never cite a marker that isn't provided.
- If the fan's favourites are involved, speak to them directly ("your driver").
- Describe the race as it stands at this lap. Never claim or predict the final result.
- Don't infer positions, gaps or outcomes the data doesn't state (e.g. where a penalty
  will drop someone): say what could happen instead.
- Plain text only: no headings, no bullet lists, no markdown."""

# What to search the rulebook for, per event (refined by payload where useful).
_REG_QUERIES: dict[str, str] = {
    "penalty": "penalties imposed by the stewards time penalty drive through",
    "race_control": "incidents investigation by the stewards",
}
_PHASE_QUERIES: dict[tuple[str, str], str] = {
    ("safety_car", "deployed"): "safety car deployed cars line up behind the safety car no "
    "overtaking",
    ("safety_car", "ended"): "safety car in this lap returns to the pits overtaking allowed "
    "after the line",
    ("vsc", "deployed"): "virtual safety car deployed drivers must stay above the minimum "
    "delta time",
    ("vsc", "ended"): "virtual safety car ending procedure",
    ("red_flag", "deployed"): "suspending a race red flag all cars proceed slowly to the pit lane",
    ("red_flag", "ended"): "resuming a race after suspension restart procedure",
}
_NEUTRAL_FLAGS = ("SC", "VSC", "RED")


class Explanation(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    level: Level
    text: str
    citations: tuple[Citation, ...] = ()
    source: Literal["llm", "template"]
    guard_passed: bool = True
    retries: int = 0
    latency_ms: int = 0


def reg_query(event: Event) -> str | None:
    phase = str(event.payload.get("phase") or "deployed")
    if (event.type, phase) in _PHASE_QUERIES:
        return _PHASE_QUERIES[(event.type, phase)]
    base = _REG_QUERIES.get(event.type)
    if base is None:
        return None
    if event.type == "penalty":
        kind = str(event.payload.get("kind") or "")
        reason = str(event.payload.get("reason") or "").lower()
        base = {
            "time": "time penalty served at pit stop or added to race time",
            "drive_through": "drive through penalty",
            "stop_go": "stop and go penalty",
            "grid": "grid place penalty",
        }.get(kind, base)
        return f"{base} {reason}".strip()
    if event.type == "race_control":
        message = str(event.payload.get("message") or "")
        reason = message.split(" - ", 1)[1].lower() if " - " in message else ""
        return f"{base} {reason}".strip()
    return base


def regs_year(season: int) -> int:
    """We index the 2025 and 2026 regulations; earlier seasons use the closest (2025)."""
    return 2026 if season >= 2026 else 2025


def _snapshot(event: Event, state: RaceState | None, meta: RaceMeta | None) -> dict:
    if state is None:
        return {}
    snap: dict = {"lap": state.lap, "total_laps": state.total_laps}
    if state.flag in _NEUTRAL_FLAGS:  # yellow flags are local; don't invite wrong claims
        snap["race_neutralised_by"] = state.flag
    drivers = []
    for code in event.drivers:
        d = state.driver(code)
        if d is None:
            continue
        info = meta.drivers.get(code) if meta else None
        drivers.append(
            {
                "code": code,
                "name": info.name if info else code,
                "team": d.team,
                "position": d.position,
                "gap_to_leader_s": d.gap_to_leader_s,
                "interval_s": d.interval_s,
                "compound": d.compound,
                "tyre_age_laps": d.tyre_age,
                "pit_stops": d.pit_count,
            }
        )
    if drivers:
        snap["drivers"] = drivers
    return snap


def _event_block(event: Event) -> dict:
    return {
        "type": event.type,
        "lap": event.lap,
        "drivers": list(event.drivers),
        "teams": list(event.teams),
        "summary": event.summary,
        "data": event.payload,
        "involves_fans_favourite": event.involves_favourite,
    }


def _citation(hit: RetrievedChunk) -> Citation:
    c = hit.chunk
    return Citation(
        source_type="fia_regulation",
        label=hit.citation,
        url=c.url,
        article=c.article,
        page=c.page,
    )


class Explainer:
    def __init__(
        self,
        llm: Runnable | None,
        retriever: RegRetriever | None = None,
        reg_hits: int = 3,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reg_hits = reg_hits

    def _regs(self, event: Event, season: int) -> list[RetrievedChunk]:
        query = reg_query(event)
        if query is None or self.retriever is None:
            return []
        try:
            return self.retriever.search(
                query, k=self.reg_hits, year=regs_year(season), doc_type="sporting"
            )
        except Exception:
            log.exception("explainer.retrieval_failed", extra={"fields": {"event": event.id}})
            return []

    def build_context(
        self,
        event: Event,
        state: RaceState | None,
        meta: RaceMeta | None,
        regs: list[RetrievedChunk],
    ) -> str:
        parts = ["EVENT:\n" + json.dumps(_event_block(event), ensure_ascii=False)]
        snap = _snapshot(event, state, meta)
        if snap:
            parts.append("RACE SNAPSHOT:\n" + json.dumps(snap, ensure_ascii=False))
        if regs:
            lines = [f"[S{i}] {h.citation}\n{h.chunk.text[:1200]}" for i, h in enumerate(regs, 1)]
            parts.append("REGULATIONS:\n" + "\n\n".join(lines))
        return "\n\n".join(parts)

    def template(self, event: Event, level: Level, regs: list[RetrievedChunk]) -> Explanation:
        return Explanation(
            event_id=event.id,
            level=level,
            text=event.summary,
            source="template",
            citations=tuple(_citation(h) for h in regs[:1]),
        )

    def explain(
        self,
        event: Event,
        level: Level,
        state: RaceState | None = None,
        meta: RaceMeta | None = None,
    ) -> Explanation:
        season = meta.season if meta else 2025
        regs = self._regs(event, season)
        if self.llm is None:
            return self.template(event, level, regs)

        context = self.build_context(event, state, meta, regs)
        messages = [
            SystemMessage(system_prompt(level, TASK)),
            HumanMessage(context),
        ]
        start = time.perf_counter()
        retries = 0
        try:
            text = self._ask(messages)
            guard = check_numbers(text, context)
            if not guard.ok:
                retries = 1
                messages += [
                    HumanMessage(
                        f"Your answer used numbers that are not in the data: "
                        f"{', '.join(guard.unsupported)}. Rewrite it using only numbers "
                        f"that appear in EVENT, RACE SNAPSHOT or REGULATIONS."
                    )
                ]
                text = self._ask(messages)
                guard = check_numbers(text, context)
        except Exception as exc:
            log.warning(
                "explainer.llm_failed",
                extra={"fields": {"event": event.id, "error": type(exc).__name__}},
            )
            return self.template(event, level, regs)

        latency = int((time.perf_counter() - start) * 1000)
        if not guard.ok:
            log.warning(
                "explainer.guard_failed",
                extra={"fields": {"event": event.id, "numbers": guard.unsupported}},
            )
            fallback = self.template(event, level, regs)
            return fallback.model_copy(
                update={"guard_passed": False, "retries": retries, "latency_ms": latency}
            )
        cited = {int(n) for n in re.findall(r"\[S(\d+)\]", text)}
        citations = tuple(_citation(regs[i - 1]) for i in sorted(cited) if 0 < i <= len(regs))
        return Explanation(
            event_id=event.id,
            level=level,
            text=text.strip(),
            citations=citations,
            source="llm",
            guard_passed=True,
            retries=retries,
            latency_ms=latency,
        )

    def _ask(self, messages: list) -> str:  # type: ignore[type-arg]
        assert self.llm is not None
        reply = self.llm.invoke(messages)
        content = reply.content if hasattr(reply, "content") else str(reply)
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        text = str(content).strip()
        if not text:
            raise ValueError("empty LLM response")
        return text
