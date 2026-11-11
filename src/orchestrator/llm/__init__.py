"""LLM providers behind one interface. Agents never know which provider is configured."""

from __future__ import annotations

from orchestrator.config import Settings
from orchestrator.llm.base import LLMClient, LLMError, LLMRequest
from orchestrator.llm.mock import MockLLM
from orchestrator.llm.openai_compat import OpenAICompatibleLLM

__all__ = ["LLMClient", "LLMError", "LLMRequest", "build_llm"]


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(settings)
    return MockLLM(invalid_rate=settings.mock_invalid_rate, seed=settings.mock_seed)
