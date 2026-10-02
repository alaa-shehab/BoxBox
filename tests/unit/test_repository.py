from __future__ import annotations

import pytest

from core.models import Prediction, Profile, Score
from core.profiles.repository import (
    NicknameTakenError,
    PredictionLockedError,
    ProfileNotFoundError,
    SqlRepository,
    make_engine,
)


def _profile(nickname: str = "BoxBoxBox") -> Profile:
    return Profile(
        nickname=nickname, favourite_team="Williams", favourite_drivers=("ALB", "SAI"), quiz_score=2
    )


def _pred(**kw: object) -> Prediction:
    base: dict[str, object] = {
        "profile_id": "p1",
        "race_id": "2024_21",
        "p1": "VER",
        "p2": "OCO",
        "p3": "GAS",
    }
    return Prediction(**{**base, **kw})


def test_profile_roundtrip(repo: SqlRepository) -> None:
    created = repo.create_profile(_profile())
    loaded = repo.get_profile(created.profile_id)
    assert loaded == created
    assert loaded is not None and loaded.created_at.tzinfo is not None


def test_nickname_lookup_is_case_insensitive(repo: SqlRepository) -> None:
    created = repo.create_profile(_profile("Grid_Walker"))
    assert repo.get_profile_by_nickname("grid_walker ") == created
    assert repo.get_profile_by_nickname("nobody") is None


def test_nickname_unique_case_insensitive(repo: SqlRepository) -> None:
    repo.create_profile(_profile("Pitlane"))
    with pytest.raises(NicknameTakenError):
        repo.create_profile(_profile("PITLANE"))


def test_update_profile_level_and_favourites(repo: SqlRepository) -> None:
    created = repo.create_profile(_profile())
    updated = repo.update_profile(created.profile_id, level="expert", favourite_drivers=("COL",))
    assert updated.level == "expert"
    assert updated.favourite_drivers == ("COL",)
    assert repo.get_profile(created.profile_id) == updated


def test_update_profile_validates(repo: SqlRepository) -> None:
    created = repo.create_profile(_profile())
    with pytest.raises(ValueError):
        repo.update_profile(created.profile_id, favourite_drivers=("A", "B", "C"))
    with pytest.raises(ProfileNotFoundError):
        repo.update_profile("missing", level="rookie")


def test_prediction_upsert_then_lock(repo: SqlRepository) -> None:
    repo.save_prediction(_pred())
    repo.save_prediction(_pred(p3="LEC", pole_to_win=False))
    stored = repo.get_prediction("p1", "2024_21")
    assert stored is not None and stored.p3 == "LEC" and stored.pole_to_win is False

    locked = repo.lock_prediction("p1", "2024_21", lap=1)
    assert locked is not None and locked.locked_at_lap == 1

    with pytest.raises(PredictionLockedError):
        repo.save_prediction(_pred(p1="NOR"))

    # Locking again keeps the original lap.
    again = repo.lock_prediction("p1", "2024_21", lap=5)
    assert again is not None and again.locked_at_lap == 1


def test_lock_without_prediction_returns_none(repo: SqlRepository) -> None:
    assert repo.lock_prediction("nobody", "2024_21", lap=1) is None


def test_predictions_are_per_race(repo: SqlRepository) -> None:
    repo.save_prediction(_pred())
    repo.lock_prediction("p1", "2024_21", lap=1)
    repo.save_prediction(_pred(race_id="2024_22"))  # other race is unaffected
    assert repo.get_prediction("p1", "2024_22") is not None


def test_score_upsert(repo: SqlRepository) -> None:
    repo.save_score(Score(profile_id="p1", race_id="r", points=10, breakdown={"p1": 10}))
    repo.save_score(Score(profile_id="p1", race_id="r", points=25, breakdown={"p1": 10, "p2": 15}))
    score = repo.get_score("p1", "r")
    assert score is not None and score.points == 25 and score.breakdown == {"p1": 10, "p2": 15}
    assert repo.get_score("p1", "other") is None


def test_facts_seen_no_repeats(repo: SqlRepository) -> None:
    assert repo.mark_fact_seen("s1", "fact:alb:first_points") is True
    assert repo.mark_fact_seen("s1", "fact:alb:first_points") is False
    assert repo.mark_fact_seen("s2", "fact:alb:first_points") is True
    assert repo.seen_facts("s1") == {"fact:alb:first_points"}
    assert repo.seen_facts("nobody") == set()


def test_sqlite_file_url_creates_parent_dir(tmp_path: object) -> None:
    from pathlib import Path

    db = Path(str(tmp_path)) / "nested" / "dir" / "pw.db"
    engine = make_engine(f"sqlite:///{db}")
    SqlRepository(engine).create_profile(_profile())
    assert db.exists()
    engine.dispose()
