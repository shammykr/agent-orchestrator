"""The orchestration engine: routes a chat message to a pipeline and coordinates its agents.

For each stage the coordinator:

1. reads the case's shared context from Redis (this turn's findings, earlier turns, history);
2. evaluates the stage's ``when`` condition (skip if false);
3. routes to skills: runs every bound skill whose ``when`` holds, concurrently, through the skill
   contracts, and stores the results;
4. renders the stage's input mapping from context + skill results and checks it against the
   agent's input contract;
5. dispatches to the agent, which returns schema-validated JSON (with repair rounds);
6. on failure applies the stage's policy: retries, fallback agent, default output, then skip or
   abort;
7. writes the validated output to Redis, where downstream agents (and later turns) read it.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from orchestrator.agents import Agent, AgentOutputError, AgentSpec
from orchestrator.config import Settings
from orchestrator.context import ContextStore
from orchestrator.contracts import ContractError, SkillContext
from orchestrator.llm import LLMClient, LLMError
from orchestrator.observability import (
    AGENT_LATENCY,
    AGENT_RUNS,
    FALLBACKS,
    PIPELINE_LATENCY,
    PIPELINE_RUNS,
    SKILL_CALLS,
    case_id_var,
)
from orchestrator.pipeline import PipelineRouter, PipelineSpec, SkillBinding, Stage
from orchestrator.skills import SkillRegistry
from orchestrator.structured import drop_nulls, render

log = logging.getLogger(__name__)

StageStatus = Literal["completed", "fallback_agent", "default_output", "skipped", "failed"]


@dataclass
class StageRecord:
    id: str
    agent: str
    status: StageStatus
    attempts: int = 0
    llm_calls: int = 0
    repaired: bool = False
    duration_ms: float = 0.0
    skills: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    note: str | None = None


@dataclass
class ChatResult:
    case_id: str
    turn: int
    pipeline: str
    routed_by: str
    status: Literal["completed", "degraded", "failed"]
    reply: str | None
    case_status: str | None
    result: dict[str, Any] | None
    stages: list[StageRecord]
    duration_ms: float


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        store: ContextStore,
        llm: LLMClient,
        skills: SkillRegistry,
        agent_specs: dict[str, AgentSpec],
        pipelines: dict[str, PipelineSpec],
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.configure(skills, agent_specs, pipelines)

    def configure(
        self, skills: SkillRegistry, agent_specs: dict[str, AgentSpec], pipelines: dict[str, PipelineSpec]
    ) -> None:
        """(Re)load registries; used at start-up and by the admin reload endpoint."""
        self.skills = skills
        self.agent_specs = agent_specs
        self.agents = {name: Agent(spec, self.llm) for name, spec in agent_specs.items()}
        self.pipelines = pipelines
        self.router = PipelineRouter(pipelines)

    async def handle(
        self, message: str, case_id: str | None, user: dict[str, Any], pipeline_name: str | None = None
    ) -> ChatResult:
        started = time.perf_counter()
        case_id = case_id or f"case-{datetime.now(UTC):%Y%m%d}-{secrets.token_hex(4)}"
        case_id_var.set(case_id)
        await self.store.ensure_case(case_id, user["sub"])

        async with self.store.lock(case_id):
            turn = await self.store.turn_count(case_id) + 1
            previous = {k: v["output"] for k, v in (await self.store.findings(case_id)).items()}
            history = [
                {k: t.get(k) for k in ("turn", "message", "pipeline", "case_status", "reply")}
                for t in await self.store.turns(case_id)
            ]
            base_view = {
                "case_id": case_id,
                "turn": turn,
                "message": message,
                "user": {"sub": user["sub"], "roles": user.get("roles", [])},
                "previous": previous,
                "history": history,
            }
            pipeline, routed_by = self.router.select(base_view, pipeline_name)
            log.info(
                "pipeline selected",
                extra={"pipeline": pipeline.name, "routed_by": routed_by, "turn": turn},
            )

            records: list[StageRecord] = []
            aborted = False
            for stage in pipeline.stages:
                record = await self._run_stage(pipeline, stage, base_view, turn)
                records.append(record)
                if record.status == "failed":
                    aborted = True
                    break

            response_stage = next((r for r in records if r.id == pipeline.response.stage), None)
            result = response_stage.output if response_stage else None
            degraded = any(r.status in ("fallback_agent", "default_output") for r in records) or any(
                r.status == "skipped" and r.note == "failure policy: skip" for r in records
            )
            status: Literal["completed", "degraded", "failed"] = (
                "failed" if aborted else "degraded" if degraded else "completed"
            )
            reply = str(result.get(pipeline.response.reply_field)) if result else None
            case_status = str(result.get(pipeline.response.status_field)) if result else None
            duration = (time.perf_counter() - started) * 1000

            await self.store.append_turn(
                case_id,
                {
                    "turn": turn,
                    "at": datetime.now(UTC).isoformat(timespec="milliseconds"),
                    "user": user["sub"],
                    "message": message,
                    "pipeline": pipeline.name,
                    "routed_by": routed_by,
                    "status": status,
                    "case_status": case_status,
                    "reply": reply,
                    "stages": [
                        {k: getattr(r, k) for k in ("id", "agent", "status", "attempts", "llm_calls", "duration_ms")}
                        for r in records
                    ],
                    "duration_ms": round(duration, 1),
                },
            )
        PIPELINE_RUNS.labels(pipeline.name, status).inc()
        PIPELINE_LATENCY.labels(pipeline.name).observe(duration / 1000)
        log.info(
            "pipeline finished",
            extra={
                "pipeline": pipeline.name,
                "status": status,
                "turn": turn,
                "duration_ms": round(duration, 1),
            },
        )
        return ChatResult(
            case_id=case_id,
            turn=turn,
            pipeline=pipeline.name,
            routed_by=routed_by,
            status=status,
            reply=reply,
            case_status=case_status,
            result=result,
            stages=records,
            duration_ms=round(duration, 1),
        )

    # ---- stages ------------------------------------------------------------------------------

    async def _view(self, base: dict[str, Any], turn: int) -> dict[str, Any]:
        """Fresh read of the shared context: what downstream agents reason over."""
        findings = await self.store.findings(base["case_id"])
        return {
            **base,
            "stages": {k: v["output"] for k, v in findings.items() if v["turn"] == turn},
            "findings": {k: v["output"] for k, v in findings.items()},
        }

    async def _run_stage(self, pipeline: PipelineSpec, stage: Stage, base: dict[str, Any], turn: int) -> StageRecord:
        started = time.perf_counter()
        record = StageRecord(id=stage.id, agent=stage.agent, status="completed")
        view = await self._view(base, turn)

        if stage.when is not None and not stage.when.evaluate(view):
            record.status, record.note = "skipped", "when condition not met"
            AGENT_RUNS.labels(stage.agent, "skipped").inc()
            return record

        skill_results, required_failed = await self._route_skills(stage, view)
        record.skills = [
            {"skill": n, **{k: v for k, v in r.items() if k != "output"}} for n, r in skill_results.items()
        ]
        view["skills"] = {n: r.get("output") for n, r in skill_results.items()}
        visible_skills = {n: r for n, r in skill_results.items() if r["status"] == "ok"}

        output: dict[str, Any] | None = None
        if required_failed:
            record.errors.append(f"required skill(s) failed: {', '.join(required_failed)}")
        else:
            inputs = drop_nulls(render(stage.inputs, view))
            policy = stage.on_failure
            for attempt in range(1 + policy.retries):
                record.attempts = attempt + 1
                output = await self._dispatch(self.agents[stage.agent], inputs, visible_skills, record)
                if output is not None:
                    break
            if output is None and policy.fallback_agent:
                FALLBACKS.labels(pipeline.name, stage.id, "fallback_agent").inc()
                output = await self._dispatch(self.agents[policy.fallback_agent], inputs, visible_skills, record)
                if output is not None:
                    record.status, record.note = "fallback_agent", f"answered by {policy.fallback_agent}"
        if output is None and stage.on_failure.default_output is not None:
            FALLBACKS.labels(pipeline.name, stage.id, "default_output").inc()
            output = dict(stage.on_failure.default_output)
            record.status, record.note = "default_output", "configured default output used"
        if output is None:
            strategy = stage.on_failure.otherwise
            FALLBACKS.labels(pipeline.name, stage.id, strategy).inc()
            record.status = "skipped" if strategy == "skip" else "failed"
            record.note = f"failure policy: {strategy}"

        record.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        AGENT_RUNS.labels(stage.agent, record.status).inc()
        AGENT_LATENCY.labels(stage.agent).observe(record.duration_ms / 1000)
        if output is not None:
            record.output = output
            await self.store.put_finding(base["case_id"], stage.id, output, turn, record.agent)
        log.info(
            "stage finished",
            extra={
                "pipeline": pipeline.name,
                "stage": stage.id,
                "agent": stage.agent,
                "status": record.status,
                "attempts": record.attempts,
                "llm_calls": record.llm_calls,
                "skills": [s["skill"] for s in record.skills],
                "duration_ms": record.duration_ms,
            },
        )
        return record

    async def _dispatch(
        self, agent: Agent, inputs: dict[str, Any], skills: dict[str, Any], record: StageRecord
    ) -> dict[str, Any] | None:
        try:
            run = await agent.run(inputs, skills)
        except AgentOutputError as e:
            record.errors.append(f"{agent.spec.name}: {e}")
            record.llm_calls += agent.spec.max_repairs + 1
            return None
        except LLMError as e:
            record.errors.append(f"{agent.spec.name}: LLM call failed: {e}")
            record.llm_calls += 1
            return None
        except ContractError as e:
            record.errors.append(str(e))
            return None
        record.llm_calls += run.llm_calls
        record.repaired = record.repaired or run.repaired
        return run.output

    async def _route_skills(self, stage: Stage, view: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Skill routing: run each bound skill whose condition holds, concurrently."""
        selected = [b for b in stage.skills if b.when is None or b.when.evaluate(view)]
        ctx = SkillContext(case_id=view["case_id"], settings=self.settings, case_view=view)
        results = await asyncio.gather(*(self._invoke_skill(b, view, ctx) for b in selected))
        out = {b.skill: r for b, r in zip(selected, results, strict=True)}
        failed = [b.skill for b, r in zip(selected, results, strict=True) if b.required and r["status"] != "ok"]
        return out, failed

    async def _invoke_skill(self, binding: SkillBinding, view: dict[str, Any], ctx: SkillContext) -> dict[str, Any]:
        started = time.perf_counter()
        args = drop_nulls(render(binding.args, view))
        try:
            output = await self.skills.get(binding.skill).invoke(args, ctx)
            result: dict[str, Any] = {"status": "ok", "output": output}
        except Exception as e:  # a failing skill is reported, not fatal (unless required)
            result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            log.warning("skill failed", extra={"skill": binding.skill, "error": result["error"]})
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        SKILL_CALLS.labels(binding.skill, result["status"]).inc()
        if result["status"] == "ok":
            await self.store.put_skill_result(view["case_id"], binding.skill, result["output"])
        return result
