"""Loads the pluggable parts (skills, agents, pipelines) and checks they fit together."""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.agents import AgentSpec, load_agent_specs
from orchestrator.config import Settings
from orchestrator.pipeline import PipelineSpec, load_pipelines
from orchestrator.skills import SkillRegistry


@dataclass
class Components:
    skills: SkillRegistry
    agents: dict[str, AgentSpec]
    pipelines: dict[str, PipelineSpec]


def load_components(settings: Settings) -> Components:
    packages = [p.strip() for p in settings.skill_packages.split(",") if p.strip()]
    skills = SkillRegistry.discover(packages, settings.skill_plugin_dir)
    agents = load_agent_specs(settings.agents_dir)
    for spec in agents.values():
        missing = [s for s in spec.allowed_skills if s not in skills]
        if missing:
            raise ValueError(f"agent {spec.name} allows unregistered skills: {missing}")
    pipelines = load_pipelines(settings.pipelines_dir, agents, skills)
    return Components(skills, agents, pipelines)
