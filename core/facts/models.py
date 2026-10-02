"""Facts: short, sourced statements about drivers, teams and circuits."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.models import Citation

SubjectType = Literal["driver", "team", "circuit"]
FactKind = Literal[
    "circuit_wins",
    "circuit_podiums",
    "circuit_no_podium",
    "circuit_debut",
    "career_wins",
    "championship_position",
    "teammate_h2h",
    "team_circuit_wins",
    "circuit_first_race",
    "circuit_most_wins",
    "wikipedia",
]


class Fact(BaseModel):
    """`text` is a deterministic sentence built from `values`. It is what the fan sees
    when no LLM is available, and the ground truth an LLM rephrasing is checked against."""

    model_config = ConfigDict(frozen=True)

    id: str
    race_id: str
    kind: FactKind
    subject_type: SubjectType
    subject: str  # driver code, team lineage key, or circuit id
    text: str
    values: dict[str, int | float | str] = Field(default_factory=dict)
    source: Citation
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)  # circuit-specific facts rank higher

    @field_validator("source")
    @classmethod
    def _source_has_a_link(cls, v: Citation) -> Citation:
        if not v.url:
            raise ValueError("every fact needs a source URL")
        return v
