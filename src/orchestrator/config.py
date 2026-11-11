"""Settings from environment variables (prefix ``ORCH_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORCH_", env_file=".env", extra="ignore")

    # Redis-backed shared context store
    redis_url: str = "redis://localhost:6379/0"
    context_ttl_seconds: int = 24 * 3600

    # Configuration directories
    agents_dir: Path = ROOT / "config" / "agents"
    pipelines_dir: Path = ROOT / "config" / "pipelines"
    users_file: Path = ROOT / "config" / "users.yaml"
    knowledge_base_file: Path = ROOT / "data" / "knowledge_base.json"
    # Extra skill sources: a directory of plugin modules and importable packages (comma-separated)
    skill_plugin_dir: Path | None = ROOT / "plugins" / "skills"
    skill_packages: str = ""

    # LLM provider: "mock" runs without an API key; "openai" calls any OpenAI-compatible endpoint
    llm_provider: Literal["mock", "openai"] = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.0
    # How to ask for JSON: "json_schema" (strict structured outputs), "json_object" (JSON mode,
    # widely supported) or "none" (rely on the prompt; output is still parsed and validated)
    llm_response_format: Literal["json_schema", "json_object", "none"] = "json_object"
    # Mock only: fraction of calls that return malformed output, to exercise validation and fallbacks
    mock_invalid_rate: float = 0.0
    mock_seed: int = 7

    # Auth
    jwt_secret: str = "change-me-in-production-please-32b"
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 60
    jwt_issuer: str = "agent-orchestrator"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
