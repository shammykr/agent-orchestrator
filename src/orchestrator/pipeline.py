"""YAML pipeline definitions: stage order, agent and skill bindings, input mappings, routing rules
and fallback strategies. Loading validates every reference against the agent and skill registries,
so a broken pipeline fails at start-up (or on reload), not mid-conversation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from orchestrator.agents import AgentSpec
from orchestrator.skills import SkillRegistry
from orchestrator.structured import resolve, validation_errors


class Condition(BaseModel):
    """All given clauses must hold.

    ``matches``: regex (case-insensitive) searched in ``field`` (default ``$.message``);
    ``exists``: path resolves to a non-null value; ``equals``: {path: value} pairs;
    ``has_history``: whether the case already has earlier turns.
    """

    matches: str | None = None
    field: str = "$.message"
    exists: str | None = None
    equals: dict[str, Any] | None = None
    has_history: bool | None = None

    def evaluate(self, view: dict[str, Any]) -> bool:
        if self.has_history is not None and bool(view.get("history")) != self.has_history:
            return False
        if self.matches is not None:
            value = resolve(self.field, view)
            if not isinstance(value, str) or not re.search(self.matches, value, re.IGNORECASE):
                return False
        if self.exists is not None and resolve(self.exists, view) is None:
            return False
        return not (self.equals and any(resolve(p, view) != v for p, v in self.equals.items()))


class SkillBinding(BaseModel):
    skill: str
    args: dict[str, Any] = Field(default_factory=dict)  # mapping template, see structured.render
    when: Condition | None = None
    required: bool = False  # a failing required skill fails the stage


class FailurePolicy(BaseModel):
    """Applied in order when a stage fails: retries, fallback agent, default output, otherwise."""

    retries: int = Field(default=0, ge=0, le=5)
    fallback_agent: str | None = None
    default_output: dict[str, Any] | None = None
    otherwise: Literal["abort", "skip"] = "abort"


class Stage(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    agent: str
    inputs: dict[str, Any] = Field(default_factory=dict)  # mapping template -> agent input contract
    skills: list[SkillBinding] = Field(default_factory=list)
    when: Condition | None = None
    on_failure: FailurePolicy = Field(default_factory=FailurePolicy)


class ResponseSpec(BaseModel):
    stage: str
    reply_field: str = "customer_reply"
    status_field: str = "status"


class PipelineSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = "1.0"
    description: str
    priority: int = 100  # lower is tried first when routing
    default: bool = False
    route: Condition | None = None
    stages: list[Stage] = Field(min_length=1)
    response: ResponseSpec

    @model_validator(mode="after")
    def _unique_stages(self) -> PipelineSpec:
        ids = [s.id for s in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError(f"pipeline {self.name}: duplicate stage ids")
        if self.response.stage not in ids:
            raise ValueError(f"pipeline {self.name}: response stage {self.response.stage!r} is not a stage")
        return self

    def check_references(self, agents: dict[str, AgentSpec], skills: SkillRegistry) -> None:
        for stage in self.stages:
            where = f"pipeline {self.name}, stage {stage.id}"
            agent = agents.get(stage.agent)
            if agent is None:
                raise ValueError(f"{where}: unknown agent {stage.agent!r}")
            for binding in stage.skills:
                if binding.skill not in skills:
                    raise ValueError(f"{where}: unknown skill {binding.skill!r}")
                if binding.skill not in agent.allowed_skills:
                    raise ValueError(f"{where}: agent {agent.name} may not use skill {binding.skill!r}")
            policy = stage.on_failure
            if policy.fallback_agent is not None:
                fallback = agents.get(policy.fallback_agent)
                if fallback is None:
                    raise ValueError(f"{where}: unknown fallback agent {policy.fallback_agent!r}")
                if fallback.output_schema != agent.output_schema:
                    raise ValueError(f"{where}: fallback agent {fallback.name} has a different output contract")
            if policy.default_output is not None:
                errors = validation_errors(agent.output_schema, policy.default_output)
                if errors:
                    raise ValueError(f"{where}: default_output violates {agent.name}'s output schema: {errors}")


def load_pipelines(directory: Path, agents: dict[str, AgentSpec], skills: SkillRegistry) -> dict[str, PipelineSpec]:
    pipelines: dict[str, PipelineSpec] = {}
    for path in sorted(directory.glob("*.y*ml")):
        spec = PipelineSpec.model_validate(yaml.safe_load(path.read_text()))
        if spec.name in pipelines:
            raise ValueError(f"duplicate pipeline {spec.name!r} in {path.name}")
        spec.check_references(agents, skills)
        pipelines[spec.name] = spec
    defaults = [p.name for p in pipelines.values() if p.default]
    if len(defaults) != 1:
        raise ValueError(f"exactly one pipeline must be marked default, found {defaults}")
    return pipelines


class PipelineRouter:
    """Routes a chat message to a pipeline: explicit request, else first matching route by
    priority, else the default pipeline."""

    def __init__(self, pipelines: dict[str, PipelineSpec]) -> None:
        self.pipelines = pipelines
        self._ordered = sorted(pipelines.values(), key=lambda p: (p.priority, p.name))
        self._default = next(p for p in pipelines.values() if p.default)

    def select(self, view: dict[str, Any], requested: str | None = None) -> tuple[PipelineSpec, str]:
        if requested is not None:
            if requested not in self.pipelines:
                raise KeyError(f"unknown pipeline {requested!r}; available: {', '.join(sorted(self.pipelines))}")
            return self.pipelines[requested], "requested"
        for pipeline in self._ordered:
            if pipeline.route is not None and pipeline.route.evaluate(view):
                return pipeline, "route-match"
        return self._default, "default"
