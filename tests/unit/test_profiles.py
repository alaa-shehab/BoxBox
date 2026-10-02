from __future__ import annotations

import pytest

from core.models import DriverInfo
from core.profiles.quiz import BANK, build_quiz, level_for_points, score_quiz
from core.profiles.repository import NicknameTakenError, SqlRepository
from core.profiles.service import ProfileService
from core.profiles.teams import display_name, roster, same_team, team_key
from core.prompts import GROUNDING_RULES, LEVELS, level_style, system_prompt


# ------------------------------------------------------------------- quiz
def test_bank_has_three_questions_per_tier() -> None:
    for tier in ("easy", "medium", "hard"):
        assert len([q for q in BANK if q.tier == tier]) >= 3
    assert len({q.id for q in BANK}) == len(BANK)


def test_build_quiz_is_deterministic_and_ordered() -> None:
    a, b = build_quiz("seed-1"), build_quiz("seed-1")
    assert [q.id for q in a] == [q.id for q in b]
    assert [q.tier for q in a] == ["easy", "medium", "hard"]
    assert len({tuple(q.id for q in build_quiz(s)) for s in range(30)}) > 1  # varies by seed


@pytest.mark.parametrize(
    ("points", "level"),
    [(0, "rookie"), (2, "rookie"), (3, "fan"), (4, "fan"), (5, "expert"), (6, "expert")],
)
def test_level_thresholds(points: int, level: str) -> None:
    assert level_for_points(points) == level


def test_scoring_weights_harder_questions() -> None:
    quiz = build_quiz(0)
    ids = [q.id for q in quiz]
    right = [q.answer for q in quiz]
    wrong = [(q.answer + 1) % len(q.options) for q in quiz]

    assert score_quiz(ids, right).level == "expert"
    assert score_quiz(ids, right).points == 6
    only_easy = score_quiz(ids, [right[0], wrong[1], wrong[2]])
    assert (only_easy.correct, only_easy.points, only_easy.level) == (1, 1, "rookie")
    hard_only = score_quiz(ids, [wrong[0], wrong[1], right[2]])
    assert (hard_only.points, hard_only.level) == (3, "fan")
    skipped = score_quiz(ids, [None, None, None])
    assert skipped.level == "rookie" and skipped.feedback[0].startswith("Not quite")


def test_scoring_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        score_quiz(["yellow_flag"], [0, 1])
    with pytest.raises(ValueError):
        score_quiz(["nope"], [0])


def test_question_validation() -> None:
    from core.profiles.quiz import Question

    with pytest.raises(ValueError):
        Question(id="x", tier="easy", prompt="?", options=("a", "b", "c"), answer=3, explanation="")
    with pytest.raises(ValueError):
        Question(id="x", tier="easy", prompt="?", options=("a", "a", "c"), answer=0, explanation="")


# ------------------------------------------------------------------ teams
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("AlphaTauri", "Racing Bulls"),
        ("RB", "Visa Cash App RB"),
        ("Alfa Romeo", "Kick Sauber"),
        ("Haas F1 Team", "haas"),
        ("Red Bull Racing", "red bull"),
    ],
)
def test_team_lineages(a: str, b: str) -> None:
    assert same_team(a, b)


def test_team_helpers() -> None:
    assert not same_team("Ferrari", "Mercedes")
    assert not same_team("", "Ferrari")
    assert team_key("Some New Team") == "somenewteam"
    assert display_name("AlphaTauri") == "Racing Bulls"
    assert display_name("Some New Team") == "Some New Team"


def test_roster_groups_by_display_team() -> None:
    drivers = {
        "TSU": DriverInfo(code="TSU", number="22", name="Yuki", team="RB"),
        "LAW": DriverInfo(code="LAW", number="30", name="Liam", team="RB"),
        "ALB": DriverInfo(code="ALB", number="23", name="Alex", team="Williams"),
    }
    r = roster(drivers)
    assert list(r) == ["Racing Bulls", "Williams"]
    assert [d.code for d in r["Racing Bulls"]] == ["LAW", "TSU"]


def test_favourite_team_boost_works_across_rebrands() -> None:
    from core.events.importance import personalise
    from core.events.types import Event

    e = Event(
        id="x",
        race_id="r",
        lap=1,
        type="pit_stop",
        drivers=("TSU",),
        teams=("AlphaTauri",),
        summary="s",
        base_importance=0.5,
    )
    (out,) = personalise([e], "Racing Bulls", [])
    assert out.importance == 0.7 and out.involves_favourite


# ---------------------------------------------------------------- service
def test_onboard_resume_and_update(repo: SqlRepository) -> None:
    svc = ProfileService(repo)
    quiz = build_quiz("abc")
    profile, result = svc.onboard(
        "Tifosi",
        "scuderia ferrari",
        ["lec", "ham"],
        [q.id for q in quiz],
        [q.answer for q in quiz],
    )
    assert profile.level == "expert" == result.level
    assert profile.favourite_team == "Ferrari"  # canonical display name
    assert profile.favourite_drivers == ("LEC", "HAM")
    assert profile.quiz_score == 3
    assert svc.resume("tifosi") == profile
    assert svc.get(profile.profile_id) == profile

    rookie = svc.set_level(profile.profile_id, "rookie")
    assert rookie.level == "rookie"
    moved = svc.set_favourites(profile.profile_id, "AlphaTauri", ["TSU"])
    assert (moved.favourite_team, moved.favourite_drivers) == ("Racing Bulls", ("TSU",))
    assert svc.resume("TIFOSI") == moved


def test_onboard_rejects_taken_nickname(repo: SqlRepository) -> None:
    svc = ProfileService(repo)
    ids = [q.id for q in build_quiz(1)]
    svc.onboard("Pitlane", "Williams", ["ALB"], ids, [None, None, None])
    with pytest.raises(NicknameTakenError):
        svc.onboard("pitlane", "McLaren", [], ids, [None, None, None])
    assert svc.resume("nobody") is None


# ---------------------------------------------------------------- prompts
@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_style(level: str) -> None:
    style = level_style(level)  # type: ignore[arg-type]
    assert style.startswith(f"# Level: {level.capitalize()}")
    prompt = system_prompt(level, "Explain the event.")  # type: ignore[arg-type]
    assert prompt.startswith(GROUNDING_RULES) and prompt.endswith("Explain the event.")


def test_levels_differ_in_the_right_direction() -> None:
    assert "jargon" in level_style("rookie")
    assert "article" in level_style("expert").lower()
    assert "Never do arithmetic" in GROUNDING_RULES


def test_unknown_level() -> None:
    with pytest.raises(ValueError):
        level_style("god")  # type: ignore[arg-type]
