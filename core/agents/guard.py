"""Numeric grounding guard: every number in LLM output must come from the given context.

The LLM may round (23.215 -> 23.2 or 23) but must not compute: no sums, differences or
averages. The check is purely lexical and deterministic, and it doubles as the
"numeric consistency" eval metric.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# A number not glued to a preceding letter (so "P3", "C4.1" are labels, not data), but
# allowed to carry a unit suffix ("23.2s", "5kg", "90%").
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d{3})*(?:\.\d+)?(?![\d])(?!\.\d)")
# Small counting words and ordinals a writer uses without them being data ("two compounds").
ALWAYS_OK = frozenset({0.0, 1.0, 2.0, 3.0})


def numbers_in(text: str) -> list[str]:
    """Numeric tokens in text, ignoring ones glued to letters (P3, C4.1, 2x) and ids."""
    return [m.group(0).replace(",", "") for m in _NUMBER.finditer(text)]


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def _allowed_values(context: str) -> list[float]:
    vals = []
    for token in re.findall(r"\d+(?:\.\d+)?", context):  # also inside ids like "P3", "55.1"
        vals.append(float(token))
    return vals


def supported(token: str, allowed: Iterable[float], tol: float = 1e-9) -> bool:
    value = abs(float(token))
    if value in ALWAYS_OK:
        return True
    d = _decimals(token)
    for a in allowed:
        a = abs(a)
        if abs(a - value) <= tol or abs(round(a, d) - value) <= tol:
            return True
        if d == 0 and abs(round(a) - value) <= tol:
            return True
    return False


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    unsupported: tuple[str, ...]


def check_numbers(output: str, context: str) -> GuardResult:
    allowed = _allowed_values(context)
    bad = tuple(dict.fromkeys(t for t in numbers_in(output) if not supported(t, allowed)))
    return GuardResult(ok=not bad, unsupported=bad)
