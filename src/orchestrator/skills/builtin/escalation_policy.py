"""Decide whether a case must be escalated, to which team, and the SLA that applies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from orchestrator.contracts import Skill, SkillContext

Severity = Literal["low", "medium", "high", "critical"]

_SLA_HOURS = {"critical": 1, "high": 4, "medium": 24, "low": 72}
_TEAMS = {
    "authentication": "identity-platform",
    "billing": "billing-ops",
    "performance": "sre",
    "data_sync": "data-platform",
    "mobile": "mobile-apps",
    "notifications": "messaging",
}


class EscalationInput(BaseModel):
    severity: Severity
    category: str
    customer_plan: str | None = None
    previous_attempts: int = Field(default=0, ge=0)


class EscalationOutput(BaseModel):
    escalate: bool
    team: str
    sla_hours: int
    reasons: list[str]


class EscalationPolicy(Skill[EscalationInput, EscalationOutput]):
    name = "escalation_policy"
    description = "Apply the escalation policy: severity, customer plan and failed attempts -> team and SLA."
    input_model = EscalationInput
    output_model = EscalationOutput

    async def run(self, args: EscalationInput, ctx: SkillContext) -> EscalationOutput:
        reasons = []
        if args.severity == "critical":
            reasons.append("critical severity")
        if args.customer_plan == "enterprise" and args.severity in ("high", "critical"):
            reasons.append("enterprise customer with high-impact issue")
        if args.previous_attempts >= 1:
            reasons.append(f"{args.previous_attempts} previous resolution attempt(s) failed")
        sla = _SLA_HOURS[args.severity]
        if args.customer_plan == "enterprise":
            sla = max(1, sla // 2)
        return EscalationOutput(
            escalate=bool(reasons),
            team=_TEAMS.get(args.category, "tier2-support") if reasons else "tier1-support",
            sla_hours=sla,
            reasons=reasons,
        )
