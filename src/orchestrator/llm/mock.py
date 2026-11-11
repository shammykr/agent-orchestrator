"""Deterministic mock LLM that returns plausible structured JSON, so the platform runs end to end
without an API key.

Known agents (diagnostic, resolution, summary) get purpose-built responders that reason over the
agent's inputs, the skill results and upstream findings, the way a real model would be prompted
to. Any other agent gets output synthesised from its JSON schema, so new agents work with the mock
too. ``invalid_rate`` makes that fraction of calls (repair attempts included) return malformed
output, which exercises validation, the repair loop, stage retries and pipeline fallbacks.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from typing import Any

from orchestrator.llm.base import LLMRequest

Responder = Callable[[dict[str, Any]], dict[str, Any]]

_CATEGORY_WORDS = {
    "authentication": r"log ?in|sso|saml|password|sign ?in|locked|2fa|mfa",
    "billing": r"invoice|charge|payment|card|refund|billing|subscription",
    "performance": r"slow|timeout|time out|latency|504|502|429|rate limit|lag",
    "data_sync": r"sync|integration|salesforce|hubspot|crm|missing records|webhook",
    "mobile": r"mobile|ios|android|app crash|the app",
    "notifications": r"email|notification|alert|spam",
}


class MockLLM:
    provider = "mock"

    def __init__(self, invalid_rate: float = 0.0, seed: int = 7) -> None:
        self._invalid_rate = invalid_rate
        self._rng = random.Random(seed)
        self._responders: dict[str, Responder] = {
            "diagnostic": _diagnostic,
            "resolution": _resolution,
            "summary": _summary,
        }

    async def complete(self, request: LLMRequest) -> str:
        if self._rng.random() < self._invalid_rate:
            return self._malformed(request)
        responder = self._responders.get(request.agent)
        out = responder(request.payload) if responder else _from_schema(request.output_schema, request.agent)
        return "```json\n" + json.dumps(out, indent=2) + "\n```"

    def _malformed(self, request: LLMRequest) -> str:
        """Typical model failures: truncated JSON, or valid JSON missing a required field."""
        if self._rng.random() < 0.5:
            return 'Sure! Here is the analysis: {"category": "auth'
        required = request.output_schema.get("required", [])
        out = _from_schema(request.output_schema, request.agent)
        if required:
            out.pop(required[0], None)
        return json.dumps(out)

    async def aclose(self) -> None:
        return None


# --- responders -----------------------------------------------------------------------------------


def _skill(payload: dict[str, Any], name: str) -> dict[str, Any]:
    result = payload.get("skills", {}).get(name, {})
    return result.get("output") or {} if isinstance(result, dict) else {}


def _guess_category(text: str) -> str:
    for category, pattern in _CATEGORY_WORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return "general"


def _diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs", {})
    message: str = inputs.get("message", "")
    articles = _skill(payload, "knowledge_base_search").get("articles", [])
    logs = _skill(payload, "log_analyzer")
    customer = _skill(payload, "customer_lookup")

    category = articles[0]["category"] if articles else _guess_category(message)
    text = message.lower()
    if re.search(
        r"outage|is down|(all|none) of (our|the) users|no ?one can|nobody can|production|everyone|entire (team|company)",
        text,
    ):
        severity = "critical" if customer.get("plan") == "enterprise" else "high"
    elif re.search(
        r"can'?t|cannot|unable|fail|crash|error|declined|blocked|charged twice|double charged|overcharged", text
    ) or logs.get("has_errors"):
        severity = "high" if "crash" in logs.get("signals", []) else "medium"
    else:
        severity = "low"

    evidence = [f"HTTP {s} in report" for s in logs.get("http_statuses", [])]
    evidence += [f"exception {e}" for e in logs.get("exceptions", [])]
    evidence += [f"signal: {s}" for s in logs.get("signals", [])]
    hypotheses = [
        {
            "hypothesis": a["title"],
            "confidence": round(min(0.95, 0.45 + 0.5 * a["score"]), 2),
            "evidence": [f"matches {a['id']} symptom: {a['symptoms'][0]}", *evidence][:4],
            "kb_article": a["id"],
        }
        for a in articles[:2]
    ] or [
        {
            "hypothesis": "Not enough detail to match a known issue",
            "confidence": 0.3,
            "evidence": evidence[:3] or ["no knowledge-base match"],
            "kb_article": None,
        }
    ]
    needs_info = not articles or len(message) < 40
    plan = customer.get("plan")
    return {
        "category": category,
        "severity": severity,
        "affected_component": articles[0]["component"] if articles else "unknown",
        "hypotheses": hypotheses,
        "needs_more_info": needs_info,
        "clarifying_questions": (
            [
                "When did the problem start, and is it still happening?",
                "How many users or accounts are affected?",
                "Can you share the exact error message or a request id?",
            ]
            if needs_info
            else []
        ),
        "customer_plan": plan,
        "analysis": (
            f"{severity.capitalize()}-severity {category.replace('_', ' ')} issue"
            + (f" for a {plan} customer" if plan else "")
            + f"; most likely cause: {hypotheses[0]['hypothesis']}"
            + (f" (evidence: {', '.join(evidence[:2])})" if evidence else "")
            + "."
        ),
    }


def _owner(step: str) -> str:
    if re.match(r"(ask|advise|recommend)\b", step, re.IGNORECASE):
        return "customer"
    if re.search(r"engineering|crash report", step, re.IGNORECASE):
        return "engineering"
    return "support"


def _resolution(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs", {})
    diagnosis = inputs.get("diagnosis") or {}
    previous = inputs.get("previous_resolution") or {}
    articles = _skill(payload, "knowledge_base_search").get("articles", [])
    escalation = _skill(payload, "escalation_policy")

    tried = {s["action"] for s in previous.get("steps", [])}
    # Only procedures for the diagnosed category, and only steps not already tried.
    category = diagnosis.get("category")
    relevant = [a for a in articles if a["category"] == category] or ([] if tried else articles[:1])
    candidates = [(a["id"], step) for a in relevant for step in a["steps"] if step not in tried]
    steps = [
        {
            "order": i + 1,
            "action": step,
            "owner": _owner(step),
            "expected_outcome": f"Rules out or fixes: {kb}",
        }
        for i, (kb, step) in enumerate(candidates[:4])
    ]
    escalate = bool(escalation.get("escalate")) or not steps
    if escalate:
        steps.append(
            {
                "order": len(steps) + 1,
                "action": f"Escalate to {escalation.get('team', 'tier2-support')} with the diagnosis and steps tried",
                "owner": "engineering",
                "expected_outcome": f"Engineering response within {escalation.get('sla_hours', 24)}h SLA",
            }
        )
    severity = diagnosis.get("severity", "medium")
    reasons = escalation.get("reasons", [])
    return {
        "plan_type": "escalation"
        if escalate
        else ("self_service" if all(s["owner"] == "customer" for s in steps) else "agent_action"),
        "steps": steps,
        "escalate": escalate,
        "escalation_team": escalation.get("team") if escalate else None,
        "sla_hours": int(escalation.get("sla_hours", 24)),
        "risk": {"critical": "high", "high": "medium"}.get(severity, "low"),
        "estimated_minutes": 15 * max(1, len(steps)),
        "kb_articles": sorted({kb for kb, _ in candidates[:4]}),
        "rationale": (
            ("Previous steps did not resolve the issue; " if tried else "")
            + (
                f"escalating because: {', '.join(reasons)}."
                if escalate and reasons
                else "following the matching knowledge-base procedure."
            )
        ),
    }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs", {})
    diagnosis = inputs.get("diagnosis") or {}
    resolution = inputs.get("resolution") or {}
    sentiment = _skill(payload, "sentiment_analyzer")

    opening = (
        "I'm sorry for the trouble this is causing, and thank you for your patience."
        if sentiment.get("sentiment") == "negative"
        else "Thanks for reaching out."
    )
    if diagnosis.get("needs_more_info") and not resolution:
        questions = " ".join(f"({i + 1}) {q}" for i, q in enumerate(diagnosis.get("clarifying_questions", [])))
        reply = f"{opening} To pin this down, could you tell us: {questions}"
        status = "needs_info"
    else:
        cause = (diagnosis.get("hypotheses") or [{}])[0].get("hypothesis", "the issue")
        customer_steps = [
            _customer_voice(s["action"]) for s in resolution.get("steps", []) if s.get("owner") == "customer"
        ]
        reply = f"{opening} It looks like this is related to: {cause.lower()}."
        if customer_steps:
            reply += " Please try the following: " + " ".join(f"{i + 1}. {s}." for i, s in enumerate(customer_steps))
        support_steps = [s for s in resolution.get("steps", []) if s.get("owner") != "customer"]
        if support_steps:
            reply += (
                " On our side, we are "
                + (
                    f"escalating to our {resolution.get('escalation_team')} team"
                    if resolution.get("escalate")
                    else "working through the remaining checks"
                )
                + f" and will update you within {_hours(int(resolution.get('sla_hours', 24)))}."
            )
        status = (
            "escalated"
            if resolution.get("escalate")
            else ("resolved_pending_confirmation" if customer_steps and not support_steps else "in_progress")
        )
    return {
        "status": status,
        "customer_reply": reply,
        "internal_note": (
            f"[{diagnosis.get('severity', '?')}/{diagnosis.get('category', '?')}] "
            f"{diagnosis.get('analysis', '')} Plan: {resolution.get('plan_type', 'n/a')}, "
            f"{len(resolution.get('steps', []))} step(s), KB {', '.join(resolution.get('kb_articles', [])) or 'none'}."
            + (f" Customer sentiment {sentiment['sentiment']} ({sentiment['score']})." if sentiment else "")
        ),
        "next_action": (
            "Wait for the customer's answers"
            if status == "needs_info"
            else f"Hand off to {resolution.get('escalation_team')}"
            if status == "escalated"
            else "Confirm with the customer that the fix worked"
        ),
        "tags": sorted({diagnosis.get("category", "general"), diagnosis.get("severity", "medium"), status}),
    }


def _hours(n: int) -> str:
    return f"{n} hour" if n == 1 else f"{n} hours"


def _customer_voice(step: str) -> str:
    """'Ask the customer to retry X' -> 'Retry X' (the reply is addressed to the customer)."""
    m = re.match(r"(?:ask|advise) the (?:customer|user)s? to (.+)", step, re.IGNORECASE)
    text = m.group(1) if m else step
    return text[0].upper() + text[1:]


def _from_schema(schema: dict[str, Any], agent: str) -> dict[str, Any]:
    value = _value(schema, agent)
    return value if isinstance(value, dict) else {"result": value}


def _value(schema: dict[str, Any], name: str) -> Any:
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "null")
    if t == "object":
        props = schema.get("properties", {})
        return {k: _value(v, k) for k, v in props.items()}
    if t == "array":
        return [_value(schema.get("items", {"type": "string"}), name)] * max(1, schema.get("minItems", 1))
    if t == "integer":
        return int(schema.get("minimum", 1))
    if t == "number":
        return float(schema.get("minimum", 0.5))
    if t == "boolean":
        return False
    if t == "null":
        return None
    return f"mock {name.replace('_', ' ')}".ljust(int(schema.get("minLength", 0)), ".")
