from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableWithFallbacks

from core.config import Settings
from core.llm import candidates, chat_model


def test_no_keys_means_no_llm() -> None:
    s = Settings(llm_provider="groq", llm_fallback_providers=["cerebras"])
    assert candidates("chat", s) == []
    assert chat_model("explainer", s) is None


def test_fake_provider_has_no_models() -> None:
    assert chat_model("chat", Settings(llm_provider="fake", llm_fallback_providers=[])) is None


def test_groq_chain_with_backup_and_cerebras() -> None:
    s = Settings(
        groq_api_key="gsk_x", cerebras_api_key="csk_x", llm_fallback_providers=["cerebras"]
    )
    names = [getattr(m, "model_name", None) for m in candidates("explainer", s)]
    assert names == ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "gpt-oss-120b"]
    model = chat_model("explainer", s)
    assert isinstance(model, RunnableWithFallbacks) and len(model.fallbacks) == 2


def test_backup_not_duplicated_when_same_as_primary() -> None:
    s = Settings(groq_api_key="gsk_x", llm_fallback_providers=[], judge_model="qwen/qwen3.8-27b")
    assert [m.model_name for m in candidates("judge", s)] == ["qwen/qwen3.8-27b"]  # type: ignore[attr-defined]
    assert candidates("judge", s)[0].temperature < 0.01  # type: ignore[attr-defined]


def test_missing_optional_client_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "langchain_ollama":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    s = Settings(groq_api_key="gsk_x", llm_fallback_providers=["ollama"])
    assert len(candidates("chat", s)) == 2  # groq primary + backup; ollama skipped
