"""Integration contracts: the typed interfaces skills and agents expose to the orchestrator.

The orchestrator core only ever talks to these contracts. A new intelligence feature plugs in by
implementing one, as a skill (Python class) or an agent (YAML spec), without core changes:

* :class:`Skill`: a reusable capability with pydantic ``Input`` / ``Output`` models. The
  orchestrator validates arguments against ``Input`` before calling it and the result against
  ``Output`` after, so a skill can never hand an agent malformed data.
* :class:`AgentContract`: an agent's input JSON schema (what a pipeline must map into it) and
  output JSON schema (what downstream stages can rely on).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from orchestrator.config import Settings

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class ContractError(Exception):
    """Data crossing a contract boundary didn't match the declared schema."""

    def __init__(self, component: str, direction: str, errors: list[str]) -> None:
        super().__init__(f"{component} {direction} violates its contract: {'; '.join(errors)}")
        self.component = component
        self.direction = direction
        self.errors = errors


@dataclass(frozen=True)
class SkillContext:
    """What a skill may see besides its arguments: settings and a read-only case snapshot."""

    case_id: str
    settings: Settings
    case_view: Mapping[str, Any] = field(default_factory=dict)


class SkillContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    source: str = ""


class Skill(ABC, Generic[InT, OutT]):
    """Base class for skills. Subclass with concrete Input/Output models and implement :meth:`run`.

    Skills are discovered automatically (see :mod:`orchestrator.skills`): any concrete subclass
    defined in a scanned module is registered under its ``name``.
    """

    name: ClassVar[str]
    version: ClassVar[str] = "1.0"
    description: ClassVar[str] = ""
    input_model: type[InT]
    output_model: type[OutT]

    @abstractmethod
    async def run(self, args: InT, ctx: SkillContext) -> OutT:
        """Do the work. Raise any exception to signal failure; the router records it."""

    @classmethod
    def contract(cls, source: str = "") -> SkillContract:
        return SkillContract(
            name=cls.name,
            version=cls.version,
            description=cls.description,
            input_schema=cls.input_model.model_json_schema(),
            output_schema=cls.output_model.model_json_schema(),
            source=source,
        )

    async def invoke(self, raw_args: Mapping[str, Any], ctx: SkillContext) -> dict[str, Any]:
        """Validate ``raw_args`` against the input contract, run, validate the output contract."""
        try:
            args = self.input_model.model_validate(dict(raw_args))
        except ValidationError as e:
            raise ContractError(f"skill {self.name}", "input", _errors(e)) from e
        result = await self.run(args, ctx)
        try:
            checked = self.output_model.model_validate(result.model_dump())
        except ValidationError as e:
            raise ContractError(f"skill {self.name}", "output", _errors(e)) from e
        return checked.model_dump(mode="json")


class AgentContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    allowed_skills: list[str]


def _errors(e: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()]
