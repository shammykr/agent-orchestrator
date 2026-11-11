"""Client for any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, Azure OpenAI proxies,
vLLM, Ollama, LiteLLM, ...)."""

from __future__ import annotations

from typing import Any

import httpx

from orchestrator.config import Settings
from orchestrator.llm.base import LLMError, LLMRequest


class OpenAICompatibleLLM:
    provider = "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        self._http = httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/"), headers=headers, timeout=settings.llm_timeout_seconds
        )

    async def complete(self, request: LLMRequest) -> str:
        body: dict[str, Any] = {
            "model": self._settings.llm_model,
            "temperature": self._settings.llm_temperature,
            "messages": [{"role": "system", "content": request.system}, *request.messages],
        }
        fmt = self._settings.llm_response_format
        if fmt == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif fmt == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": request.agent, "schema": request.output_schema, "strict": False},
            }
        try:
            resp = await self._http.post("/chat/completions", json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        if not isinstance(content, str) or not content.strip():
            raise LLMError("empty completion")
        return content

    async def aclose(self) -> None:
        await self._http.aclose()
