"""Load the committed simulator data: race history (track params) and deg overrides."""

from __future__ import annotations

import json
from functools import lru_cache

from core.config import REPO_ROOT
from core.models import RaceMeta
from core.sim.history import RaceSummary, deg_prior, load_history, track_params
from core.sim.params import TrackParams

SIM_DIR = REPO_ROOT / "data" / "sim"


@lru_cache(maxsize=1)
def history() -> tuple[RaceSummary, ...]:
    return tuple(load_history(SIM_DIR / "history.json"))


@lru_cache(maxsize=1)
def overrides() -> dict[str, dict[str, float]]:
    path = SIM_DIR / "degradation" / "overrides.json"
    return json.loads(path.read_text()) if path.exists() else {}


def params_for(meta: RaceMeta) -> tuple[TrackParams, dict[str, float]]:
    """Track parameters and degradation prior for a race, from earlier seasons only."""
    rows = list(history())
    deg = deg_prior(meta.circuit_key, meta.season, rows)
    deg.update(overrides().get(meta.circuit_key, {}))
    return track_params(meta.circuit_key, meta.season, rows), deg
