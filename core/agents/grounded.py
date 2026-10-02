"""Grounded generation: one LLM call, checked by the numeric guard, with a safe fallback.

    system = grounding rules + level style + task
    reply  -> numeric guard against the context
           -> fails: retry once, naming the unsupported numbers
           -> fails again, an LLM error, or no LLM at all: return the fallback text

Used by the Explainer and the facts engine, so every LLM sentence a fan sees goes
through the same checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from core.agents.guard import check_numbers
from core.log import get_logger
from core.models import Level
from core.prompts import system_prompt

log = get_logger(__name__)


@dataclass(frozen=True)
class Grounded:
    text: str
    source: Literal["llm", "template"]
    guard_passed: bool = True
    retries: int = 0
    latency_ms: int = 0
    error: str | None = None


def _content(reply: object) -> str:
    content = getattr(reply, "content", reply)
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content).strip()


def generate(
    llm: Runnable | None,
    level: Level,
    task: str,
    context: str,
    fallback: str,
    label: str = "",
) -> Grounded:
    if llm is None:
        return Grounded(text=fallback, source="template")
    messages: list[BaseMessage] = [SystemMessage(system_prompt(level, task)), HumanMessage(context)]
    start = time.perf_counter()
    retries = 0
    try:
        text = _content(llm.invoke(messages))
        if not text:
            raise ValueError("empty LLM response")
        guard = check_numbers(text, context)
        if not guard.ok:
            retries = 1
            messages.append(
                HumanMessage(
                    f"Your answer used numbers that are not in the provided data: "
                    f"{', '.join(guard.unsupported)}. Rewrite it using only numbers that appear "
                    f"in the provided data."
                )
            )
            text = _content(llm.invoke(messages))
            guard = check_numbers(text, context)
    except Exception as exc:
        log.warning(
            "grounded.llm_failed", extra={"fields": {"label": label, "error": type(exc).__name__}}
        )
        return Grounded(text=fallback, source="template", error=type(exc).__name__)

    latency = int((time.perf_counter() - start) * 1000)
    if not guard.ok or not text:
        log.warning(
            "grounded.guard_failed",
            extra={"fields": {"label": label, "numbers": guard.unsupported}},
        )
        return Grounded(
            text=fallback,
            source="template",
            guard_passed=False,
            retries=retries,
            latency_ms=latency,
        )
    return Grounded(text=text, source="llm", retries=retries, latency_ms=latency)
