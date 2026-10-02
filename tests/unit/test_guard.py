from __future__ import annotations

import pytest

from core.agents.guard import check_numbers, numbers_in

CTX = '{"gap_to_leader_s": 23.215, "position": 4, "tyre_age_laps": 27, "lap": 43} Art. 55.1'


def test_numbers_in_skips_labels_and_keeps_units() -> None:
    text = "P3 in C4.1 with 23.2s gap, 1,200 kg, 90% after lap 43 and 2x stops."
    assert numbers_in(text) == ["23.2", "1200", "90", "43", "2"]


@pytest.mark.parametrize(
    "text",
    [
        "The gap is 23.215s.",
        "About 23.2 seconds.",
        "Roughly 23 seconds.",
        "Fourth place on 27-lap-old tyres at lap 43.",
        "Two compounds, one stop.",
        "See Art. 55.1.",
        "He's in P4.",
    ],
)
def test_supported(text: str) -> None:
    assert check_numbers(text, CTX).ok


@pytest.mark.parametrize(
    ("text", "bad"),
    [
        ("A 24.5s gap.", ("24.5",)),
        ("16 laps to go", ("16",)),
        ("He gained 23.3s", ("23.3",)),
        ("5 places", ("5",)),
    ],
)
def test_unsupported(text: str, bad: tuple[str, ...]) -> None:
    result = check_numbers(text, CTX)
    assert not result.ok and result.unsupported == bad
