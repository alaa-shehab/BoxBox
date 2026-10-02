"""Per-level style guides plus the grounding rules shared by every LLM call.

`system_prompt(level)` is the single entry point. Every LLM call in core/ builds its system
message from it, which is how the fan's level reaches every call.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

from core.models import Level

LEVELS: tuple[Level, ...] = ("rookie", "fan", "expert")

GROUNDING_RULES = """\
You are Pit Wall, an AI race companion for Formula 1 fans.

Hard rules. Never break them:
1. Use ONLY the facts and numbers in the provided context. Never invent drivers, results,
   statistics, history or regulation text.
2. Never do arithmetic. Every number you state must come from the provided context. You may
   round it (23.215 -> 23.2) but never compute a new one (no sums, differences or averages).
   If a number you need isn't there, describe the situation qualitatively instead.
3. When the context includes sources (regulation articles, URLs), cite them as given.
4. If the context doesn't contain what is needed to answer, say you don't know.
5. Unofficial fan project: don't claim to speak for Formula 1, the FIA or any team.
"""


@cache
def level_style(level: Level) -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    return files("core.prompts").joinpath(f"{level}.md").read_text(encoding="utf-8").strip()


def system_prompt(level: Level, task: str = "") -> str:
    """Grounding rules, then the level style, then the task-specific instructions."""
    parts = [GROUNDING_RULES, level_style(level)]
    if task:
        parts.append(task.strip())
    return "\n\n".join(parts)
