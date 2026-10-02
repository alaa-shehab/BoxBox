"""Application settings.

There is a single source of truth for configuration. Values are resolved in this order:
init kwargs > environment > `.env` file > Streamlit secrets > defaults.

Streamlit secrets are read only when Streamlit is already imported, i.e. when we run
inside a Streamlit app. Scripts and the API never pay the cost of importing Streamlit.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

LLMProvider = Literal["groq", "cerebras", "ollama", "fake"]


class StreamlitSecretsSource(PydanticBaseSettingsSource):
    """Reads top-level keys from `st.secrets`, matching field names case-insensitively."""

    def _secrets(self) -> dict[str, Any]:
        if "streamlit" not in sys.modules:
            return {}
        try:
            import streamlit as st

            return {str(k).lower(): v for k, v in st.secrets.to_dict().items()}
        except Exception:  # no secrets.toml, or not running under `streamlit run`
            return {}

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._secrets().get(field_name.lower()), field_name, False

    def __call__(self) -> dict[str, Any]:
        secrets = self._secrets()
        return {name: secrets[name] for name in self.settings_cls.model_fields if name in secrets}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app ---
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    backend_mode: Literal["inprocess", "api"] = "inprocess"
    api_base_url: str = "http://localhost:8000"

    # --- persistence ---
    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'pitwall.db'}"

    # --- LLMs (free tiers of open-weight model hosts; see PLAN.md §0.1) ---
    llm_provider: LLMProvider = "groq"
    llm_fallback_providers: list[LLMProvider] = Field(default_factory=lambda: ["cerebras"])
    groq_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "openai/gpt-oss-120b"  # tool-calling chat agent
    explainer_model: str = "openai/gpt-oss-20b"  # high-volume event explanations
    judge_model: str = "qwen/qwen3.8-27b"  # eval judge: a different family than the generator
    backup_model: str = "qwen/qwen3.8-27b"  # same-provider fallback when a model is limited
    cerebras_model: str = "gpt-oss-120b"
    ollama_model: str = "qwen3:4b"
    llm_requests_per_minute: int = Field(default=25, ge=1)
    llm_timeout_s: float = Field(default=30.0, gt=0)

    # --- local retrieval models (fastembed / ONNX) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    onnx_threads: int = Field(default=2, ge=1)  # small pools keep RSS low on 1-2 vCPU hosts
    rerank_enabled: bool = True  # off saves ~130 MB RSS (see evals/results.md for the cost)

    # --- tracing (optional) ---
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- data paths ---
    data_dir: Path = REPO_ROOT / "data"
    replay_dir: Path = REPO_ROOT / "data" / "replays"
    fastf1_cache_dir: Path = REPO_ROOT / "data" / "cache" / "fastf1"

    # --- race features ---
    quiet_period_laps: int = Field(default=5, ge=1)
    sim_n_sims: int = Field(default=2000, ge=100)
    sim_seed: int = 7

    @field_validator(
        "groq_api_key",
        "cerebras_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, v: Any) -> Any:
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def tracing_enabled(self) -> bool:
        return self.langfuse_public_key is not None and self.langfuse_secret_key is not None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            StreamlitSecretsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
