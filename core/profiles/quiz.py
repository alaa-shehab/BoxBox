"""Three-question onboarding quiz that sets the fan's explanation level.

A quiz is one easy, one medium and one hard question, picked deterministically from the
bank by a seed. Scoring weights the harder questions more, so a lucky guess on the easy
question doesn't make someone an expert:

    points = easy*1 + medium*2 + hard*3   (max 6)
    0-2 -> rookie, 3-4 -> fan, 5-6 -> expert

The questions stick to stable, season-independent rules and strategy concepts.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.models import Level

Tier = Literal["easy", "medium", "hard"]
TIER_POINTS: dict[Tier, int] = {"easy": 1, "medium": 2, "hard": 3}


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tier: Tier
    prompt: str
    options: tuple[str, ...] = Field(min_length=3, max_length=4)
    answer: int
    explanation: str

    @model_validator(mode="after")
    def _valid(self) -> Question:
        if not 0 <= self.answer < len(self.options):
            raise ValueError(f"{self.id}: answer index out of range")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"{self.id}: duplicate options")
        return self


BANK: tuple[Question, ...] = (
    # --- easy -----------------------------------------------------------------
    Question(
        id="yellow_flag",
        tier="easy",
        prompt="A marshal waves a yellow flag. What must drivers do?",
        options=(
            "Slow down, be ready to stop, and not overtake",
            "Pit at the end of the lap",
            "Let the car behind pass",
            "Nothing, it is just a warning to spectators",
        ),
        answer=0,
        explanation="Yellow means danger ahead: slow down and no overtaking in that zone.",
    ),
    Question(
        id="win_points",
        tier="easy",
        prompt="How many points does the winner of a Grand Prix score?",
        options=("10", "20", "25", "50"),
        answer=2,
        explanation="The winner scores 25 points, then 18, 15, 12, 10, 8, 6, 4, 2, 1.",
    ),
    Question(
        id="safety_car",
        tier="easy",
        prompt="What is the safety car for?",
        options=(
            "It leads the field at reduced speed while an incident is cleared",
            "It carries spare parts for teams",
            "It starts the race from the front of the grid",
            "It films the race for TV",
        ),
        answer=0,
        explanation="The safety car bunches the field up and slows it down so marshals can work.",
    ),
    # --- medium ---------------------------------------------------------------
    Question(
        id="two_compounds",
        tier="medium",
        prompt="In a fully dry race, what tyre rule must every driver follow?",
        options=(
            "Use at least two different dry compounds",
            "Use the soft tyre at the start",
            "Make exactly one pit stop",
            "Never use the hard tyre",
        ),
        answer=0,
        explanation="Without rain, each driver must run two different dry compounds, so at "
        "least one stop is needed.",
    ),
    Question(
        id="undercut",
        tier="medium",
        prompt="What is an 'undercut'?",
        options=(
            "Pitting before a rival, then using fresh tyres to come out ahead after they stop",
            "Overtaking on the inside of a corner",
            "Cutting a chicane to gain time",
            "Starting the race from the pit lane",
        ),
        answer=0,
        explanation="Fresh tyres are faster, so stopping first can gain enough time to "
        "jump a rival.",
    ),
    Question(
        id="blue_flag",
        tier="medium",
        prompt="A driver is shown a blue flag. What does it mean?",
        options=(
            "A faster car about to lap them is behind; let it through",
            "The track is wet",
            "They have a penalty",
            "Their car has a technical problem",
        ),
        answer=0,
        explanation="Blue flags tell a backmarker to let a lapping car pass.",
    ),
    # --- hard -----------------------------------------------------------------
    Question(
        id="vsc_cheap_stop",
        tier="hard",
        prompt="Why can pitting under a Virtual Safety Car be 'cheap'?",
        options=(
            "Rivals on track must lap slowly, so less time is lost relative to them in the pits",
            "Pit stops under VSC are shorter by regulation",
            "Teams get a free set of tyres under VSC",
            "The pit lane speed limit is removed under VSC",
        ),
        answer=0,
        explanation="Everyone on track is slowed to a delta time, so a stop costs less relative "
        "track position than under green flags.",
    ),
    Question(
        id="deg_rate",
        tier="hard",
        prompt="In strategy models, how is tyre degradation usually expressed?",
        options=(
            "Seconds of lap time lost per lap of tyre age",
            "Millimetres of tread lost per lap",
            "Percentage of grip left after the race",
            "Number of laps until a puncture",
        ),
        answer=0,
        explanation="Degradation is modelled as a lap-time loss per lap of tyre age, e.g. "
        "0.08 s/lap.",
    ),
    Question(
        id="fuel_correction",
        tier="hard",
        prompt="What does a 'fuel-corrected' lap time adjust for?",
        options=(
            "The car getting faster as it burns fuel and gets lighter",
            "Different fuel brands used by teams",
            "Fuel spilled during pit stops",
            "Engine modes used to save fuel",
        ),
        answer=0,
        explanation="Cars get lighter during a race, so raw lap times improve even as tyres wear; "
        "fuel correction removes that effect.",
    ),
)

_BY_ID = {q.id: q for q in BANK}


def build_quiz(seed: int | str = 0) -> list[Question]:
    """One question per tier, easy to hard, chosen deterministically from the seed."""
    rng = random.Random(str(seed))
    return [rng.choice([q for q in BANK if q.tier == tier]) for tier in ("easy", "medium", "hard")]


class QuizResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    correct: int = Field(ge=0, le=3)
    points: int = Field(ge=0, le=6)
    level: Level
    feedback: tuple[str, ...]


def level_for_points(points: int) -> Level:
    if points >= 5:
        return "expert"
    if points >= 3:
        return "fan"
    return "rookie"


def score_quiz(question_ids: Sequence[str], answers: Sequence[int | None]) -> QuizResult:
    """Score answers (option index, or None if skipped) for the given question ids."""
    if len(question_ids) != len(answers):
        raise ValueError("one answer per question is required")
    correct = points = 0
    feedback = []
    for qid, given in zip(question_ids, answers, strict=True):
        q = _BY_ID.get(qid)
        if q is None:
            raise ValueError(f"unknown question {qid!r}")
        right = given == q.answer
        correct += right
        points += TIER_POINTS[q.tier] if right else 0
        feedback.append(("Correct! " if right else "Not quite. ") + q.explanation)
    return QuizResult(
        correct=correct, points=points, level=level_for_points(points), feedback=tuple(feedback)
    )
