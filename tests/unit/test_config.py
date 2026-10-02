from __future__ import annotations

import sys
import types

import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings


def test_defaults_need_no_secrets() -> None:
    s = Settings()
    assert s.backend_mode == "inprocess"
    assert s.llm_provider == "groq"
    assert s.groq_api_key is None
    assert not s.tracing_enabled
    assert s.database_url.startswith("sqlite:///")


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND_MODE", "api")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDERS", '["cerebras", "ollama"]')
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    s = Settings()
    assert s.backend_mode == "api"
    assert s.llm_fallback_providers == ["cerebras", "ollama"]
    assert s.groq_api_key is not None
    assert s.groq_api_key.get_secret_value() == "gsk_test"
    assert "gsk_test" not in repr(s)  # secrets are masked


def test_blank_keys_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "  ")
    assert not Settings().tracing_enabled


def test_tracing_enabled_with_both_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert Settings().tracing_enabled


def test_invalid_values_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND_MODE", "carrier-pigeon")
    with pytest.raises(ValidationError):
        Settings()


def _fake_streamlit(secrets: dict[str, object]) -> types.ModuleType:
    module = types.ModuleType("streamlit")

    class _Secrets:
        def to_dict(self) -> dict[str, object]:
            return secrets

    module.secrets = _Secrets()  # type: ignore[attr-defined]
    return module


def test_streamlit_secrets_used_when_streamlit_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        _fake_streamlit({"DATABASE_URL": "postgresql+psycopg://u:p@h/db", "groq_api_key": "k"}),
    )
    s = Settings()
    assert s.database_url == "postgresql+psycopg://u:p@h/db"
    assert s.groq_api_key is not None


def test_env_beats_streamlit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit({"backend_mode": "api"}))
    monkeypatch.setenv("BACKEND_MODE", "inprocess")
    assert Settings().backend_mode == "inprocess"


def test_streamlit_without_secrets_file(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("streamlit")

    class _Missing:
        def to_dict(self) -> dict[str, object]:
            raise FileNotFoundError("no secrets.toml")

    module.secrets = _Missing()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit", module)
    assert Settings().backend_mode == "inprocess"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
