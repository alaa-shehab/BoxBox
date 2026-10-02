from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from core.config import Settings, get_settings
from core.profiles.repository import SqlRepository

POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")

_backends = [pytest.param("sqlite", id="sqlite")]
_backends.append(
    pytest.param(
        "postgres",
        id="postgres",
        marks=[
            pytest.mark.postgres,
            pytest.mark.skipif(not POSTGRES_URL, reason="TEST_POSTGRES_URL not set"),
        ],
    )
)


@pytest.fixture(params=_backends)
def repo(request: pytest.FixtureRequest) -> Iterator[SqlRepository]:
    url = "sqlite://" if request.param == "sqlite" else POSTGRES_URL
    assert url
    repository = SqlRepository.from_url(url)
    repository._truncate_all()
    yield repository
    repository._truncate_all()
    repository.engine.dispose()


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Never let a developer's .env or real keys leak into tests."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_replay():  # type: ignore[no-untyped-def]
    from core.feed.builder import from_session
    from tests.fixtures.fake_session import FakeSession

    return from_session(FakeSession(), season=2099, round_number=1)


@pytest.fixture
def fake_feed(fake_replay):  # type: ignore[no-untyped-def]
    from core.feed.replay import ReplayFeed

    return ReplayFeed(fake_replay)
