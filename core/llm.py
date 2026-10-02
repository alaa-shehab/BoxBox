"""Chat-model factory: free open-weight model hosts, a fallback chain, client-side limits.

Providers, in the order given by config (`llm_provider` then `llm_fallback_providers`):
    groq      Groq free tier (no card). Primary model per role, then a backup model.
    cerebras  Cerebras free tier, through its OpenAI-compatible endpoint.
    ollama    A local Ollama server (Docker or local dev).

A provider without credentials is skipped. If nothing is configured, `chat_model()` returns
None and every caller falls back to deterministic templates, so the app runs without any
LLM at all. Every call goes through one shared client-side rate limiter per provider, so
free-tier limits show up as a short wait, not as 429 errors.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable

from core.config import Settings, get_settings
from core.log import get_logger

log = get_logger(__name__)

Role = Literal["chat", "explainer", "judge"]

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


@cache
def _limiter(provider: str, rpm: int) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(requests_per_second=rpm / 60.0, max_bucket_size=max(1, rpm // 10))


def _model_for(role: Role, s: Settings) -> str:
    return {"chat": s.chat_model, "explainer": s.explainer_model, "judge": s.judge_model}[role]


def _common(role: Role, s: Settings) -> dict[str, Any]:
    return {
        "temperature": 0.0 if role == "judge" else 0.3,
        "max_retries": 1,
        "max_tokens": 700 if role == "chat" else 400,
    }


def candidates(role: Role, settings: Settings | None = None) -> list[BaseChatModel]:
    """Every configured model for a role, in fallback order."""
    s = settings or get_settings()
    out: list[BaseChatModel] = []
    for provider in dict.fromkeys([s.llm_provider, *s.llm_fallback_providers]):
        try:
            out += _provider_models(provider, role, s)
        except ImportError as exc:  # optional client not installed (e.g. ollama on Cloud)
            log.warning(
                "llm.provider_unavailable",
                extra={"fields": {"provider": provider, "error": str(exc)}},
            )
    return out


def _provider_models(provider: str, role: Role, s: Settings) -> list[BaseChatModel]:
    common = _common(role, s)
    if provider == "groq":
        if s.groq_api_key is None:
            return []
        from langchain_groq import ChatGroq

        names = list(dict.fromkeys([_model_for(role, s), s.backup_model]))
        limiter = _limiter("groq", s.llm_requests_per_minute)
        return [
            ChatGroq(
                model=name,
                api_key=s.groq_api_key,
                request_timeout=s.llm_timeout_s,
                rate_limiter=limiter,
                # gpt-oss models reason before answering; keep that short for latency.
                reasoning_effort="low" if name.startswith("openai/gpt-oss") else None,
                **common,
            )
            for name in names
        ]
    if provider == "cerebras":
        if s.cerebras_api_key is None:
            return []
        from langchain_openai import ChatOpenAI  # OpenAI-compatible client, Cerebras endpoint

        return [
            ChatOpenAI(
                model=s.cerebras_model,
                api_key=s.cerebras_api_key,
                base_url=CEREBRAS_BASE_URL,
                timeout=s.llm_timeout_s,
                rate_limiter=_limiter("cerebras", min(s.llm_requests_per_minute, 5)),
                **common,
            )
        ]
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return [
            ChatOllama(
                model=s.ollama_model, base_url=s.ollama_base_url, temperature=common["temperature"]
            )
        ]
    return []  # "fake" or unknown: no LLM; callers use templates


def chat_model(role: Role, settings: Settings | None = None) -> Runnable | None:
    """A single runnable that tries each configured model in turn, or None if none exist."""
    models = candidates(role, settings)
    if not models:
        log.info("llm.none_configured", extra={"fields": {"role": role}})
        return None
    primary, *rest = models
    return primary.with_fallbacks(rest) if rest else primary
