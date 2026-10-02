from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import Prediction, Profile


def test_profile_normalises_driver_codes() -> None:
    p = Profile(nickname="tifosi_99", favourite_team="Ferrari", favourite_drivers=("lec", " ham"))
    assert p.favourite_drivers == ("LEC", "HAM")
    assert len(p.profile_id) == 32
    assert p.level == "fan"


def test_profile_dedupes_drivers() -> None:
    p = Profile(nickname="ab", favourite_team="McLaren", favourite_drivers=("NOR", "nor"))
    assert p.favourite_drivers == ("NOR",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"favourite_drivers": ("NOR", "PIA", "VER")},  # too many
        {"favourite_drivers": ("NORRIS",)},  # not a code
        {"nickname": "x"},  # too short
        {"nickname": "<script>"},  # bad chars
        {"favourite_team": "  "},
        {"level": "god"},
    ],
)
def test_profile_rejects_invalid(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {"nickname": "fan1", "favourite_team": "Williams"}
    with pytest.raises(ValidationError):
        Profile(**{**base, **kwargs})


def test_prediction_requires_distinct_podium() -> None:
    with pytest.raises(ValidationError):
        Prediction(profile_id="a", race_id="r", p1="VER", p2="VER", p3="NOR")


def test_prediction_codes_and_lock() -> None:
    pred = Prediction(profile_id="a", race_id="r", p1="ver", p2="nor", p3="lec", first_dnf="sar")
    assert pred.podium == ("VER", "NOR", "LEC")
    assert pred.first_dnf == "SAR"
    assert not pred.is_locked
    assert pred.model_copy(update={"locked_at_lap": 1}).is_locked
