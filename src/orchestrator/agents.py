"""Agent framework: agents are declared in YAML and run by one generic LLM agent class.

An agent spec (``config/agents/*.yaml``) is the agent's integration contract:

* ``role``: system prompt describing the agent's speciality;
* ``input_schema``: JSON schema of the inputs a pipeline must map into it;
* ``output_schema``: JSON schema its output is validated against, which downstream stages rely on;
* ``allowed_skills``: which registered skills pipelines may bind to it.

Running an agent: validate inputs against the contract, ask the LLM for JSON, parse and validate
it, and on a validation failure send the errors back for up to ``max_repairs`` correction rounds.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from orchestrator.contracts import AgentContract, ContractError
from orchestrator.llm import LLMClient, LLMRequest
from orchestrator.observability import LLM_CALLS, VALIDATION_FAILURES
from orchestrator.structured import StructuredOutputError, check_schema, parse_structured, validation_errors

log = logging.getLogger(__name__)


class AgentSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = "1.0"
    description: str
    role: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    allowed_skills: list[str] = Field(default_factory=list)
    max_repairs: int = Field(default=1, ge=0, le=3)

    def contract(self) -> AgentContract:
        return AgentContract(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            allowed_skills=self.allowed_skills,
        )


class AgentOutputError(Exception):
    """The agent could not produce schema-valid output within its repair budget."""

    def __init__(self, agent: str, errors: list[str]) -> None:
        super().__init__(f"agent {agent} produced no valid output: {'; '.join(errors[-3:])}")
        self.agent = agent
        self.errors = errors


@dataclass
class AgentRun:
    output: dict[str, Any]
    llm_calls: int
    repaired: bool
    errors: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, spec: AgentSpec, llm: LLMClient) -> None:
        self.spec = spec
        self.llm = llm
        self.system_prompt = (
            f"{spec.role.strip()}\n\n"
            "You receive a JSON object with your `inputs` (validated against your input contract) and "
            "`skill_results` (outputs of tools that already ran for you). Reason over them and respond "
            "with a single JSON object that conforms to this JSON schema. No text outside the JSON.\n\n"
            f"{json.dumps(spec.output_schema)}"
        )

    async def run(self, inputs: dict[str, Any], skill_results: dict[str, Any]) -> AgentRun:
        errors = validation_errors(self.spec.input_schema, inputs)
        if errors:
            raise ContractError(f"agent {self.spec.name}", "input", errors)

        payload = {"inputs": inputs, "skill_results": skill_results}
        messages = [{"role": "user", "content": json.dumps(payload, default=str)}]
        all_errors: list[str] = []
        for attempt in range(1 + self.spec.max_repairs):
            started = time.perf_counter()
            try:
                text = await self.llm.complete(
                    LLMRequest(
                        agent=self.spec.name,
                        system=self.system_prompt,
                        messages=messages,
                        output_schema=self.spec.output_schema,
                        payload={"inputs": inputs, "skills": skill_results},
                    )
                )
            except Exception:
                LLM_CALLS.labels(self.llm.provider, "error").inc()
                raise
            LLM_CALLS.labels(self.llm.provider, "ok").inc()
            try:
                output = parse_structured(text, self.spec.output_schema)
                return AgentRun(output=output, llm_calls=attempt + 1, repaired=attempt > 0, errors=all_errors)
            except StructuredOutputError as e:
                VALIDATION_FAILURES.labels(self.spec.name).inc()
                all_errors.extend(e.errors)
                log.warning(
                    "agent output failed validation",
                    extra={
                        "agent": self.spec.name,
                        "attempt": attempt + 1,
                        "errors": e.errors[:5],
                        "llm_ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                )
                messages += [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "Your reply was not valid against the output schema: "
                        + "; ".join(e.errors[:10])
                        + ". Reply with only the corrected JSON object.",
                    },
                ]
        raise AgentOutputError(self.spec.name, all_errors)


def load_agent_specs(directory: Path) -> dict[str, AgentSpec]:
    specs: dict[str, AgentSpec] = {}
    raw_specs = {path: yaml.safe_load(path.read_text()) for path in sorted(directory.glob("*.y*ml"))}
    by_name = {r.get("name"): r for r in raw_specs.values()}
    for path, raw in raw_specs.items():
        # `extends: <agent>` inherits every field not set here (e.g. a fallback agent reusing the
        # primary agent's input/output contracts with a simpler role prompt).
        if "extends" in raw:
            base = by_name.get(raw["extends"])
            if base is None or "extends" in base:
                raise ValueError(f"{path.name}: cannot extend {raw['extends']!r}")
            raw = {**base, **{k: v for k, v in raw.items() if k != "extends"}}
        spec = AgentSpec.model_validate(raw)
        if spec.name in specs:
            raise ValueError(f"duplicate agent {spec.name!r} in {path.name}")
        for which in ("input_schema", "output_schema"):
            try:
                check_schema(getattr(spec, which))
            except Exception as e:
                raise ValueError(f"agent {spec.name}: invalid {which}: {e}") from e
        specs[spec.name] = spec
    if not specs:
        raise ValueError(f"no agent specs found in {directory}")
    return specs
