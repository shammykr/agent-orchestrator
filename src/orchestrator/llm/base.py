from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """The provider failed (network, HTTP error, empty response)."""


@dataclass(frozen=True)
class LLMRequest:
    """A chat completion request for structured output.

    ``system`` and ``messages`` are what a real model sees. ``agent`` and ``payload`` (the
    agent's validated inputs and skill results) are passed alongside so the mock provider can
    produce realistic output without parsing prompts; real providers ignore them.
    """

    agent: str
    system: str
    messages: list[dict[str, str]]
    output_schema: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    provider: str

    async def complete(self, request: LLMRequest) -> str:
        """Return the raw text of the model's reply (expected to contain JSON)."""
        ...

    async def aclose(self) -> None: ...
