"""Fan profile use-cases: onboarding, resume by nickname, change level or favourites.

Identity is a nickname only (no passwords). The stable `profile_id` keys every other
record, so a fan can rename later without losing predictions.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.log import get_logger
from core.models import Level, Profile
from core.profiles.quiz import QuizResult, score_quiz
from core.profiles.repository import Repository
from core.profiles.teams import display_name

log = get_logger(__name__)


class ProfileService:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def onboard(
        self,
        nickname: str,
        favourite_team: str,
        favourite_drivers: Sequence[str],
        question_ids: Sequence[str],
        answers: Sequence[int | None],
    ) -> tuple[Profile, QuizResult]:
        """Create a profile, with the level set by the quiz. Raises NicknameTakenError."""
        result = score_quiz(question_ids, answers)
        profile = Profile(
            nickname=nickname,
            favourite_team=display_name(favourite_team),
            favourite_drivers=tuple(favourite_drivers),
            level=result.level,
            quiz_score=result.correct,
        )
        created = self.repo.create_profile(profile)
        log.info(
            "profile.onboarded",
            extra={"fields": {"profile_id": created.profile_id, "level": created.level}},
        )
        return created, result

    def resume(self, nickname: str) -> Profile | None:
        return self.repo.get_profile_by_nickname(nickname)

    def get(self, profile_id: str) -> Profile | None:
        return self.repo.get_profile(profile_id)

    def set_level(self, profile_id: str, level: Level) -> Profile:
        return self.repo.update_profile(profile_id, level=level)

    def set_favourites(
        self, profile_id: str, favourite_team: str, favourite_drivers: Sequence[str]
    ) -> Profile:
        return self.repo.update_profile(
            profile_id,
            favourite_team=display_name(favourite_team),
            favourite_drivers=tuple(favourite_drivers),
        )
