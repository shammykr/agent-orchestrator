"""Extract error signals (status codes, exception names, error codes) from pasted logs or text."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from orchestrator.contracts import Skill, SkillContext

_HTTP = re.compile(r"\b(400|401|403|404|408|409|413|422|429|500|502|503|504)\b")
_EXC = re.compile(r"\b([A-Z][A-Za-z]+(?:Exception|Error))\b")
_CODE = re.compile(r"\b([A-Z]{2,5}-\d{3,5})\b")
_SIGNALS = {
    "timeout": re.compile(r"time[d ]?out|deadline exceeded|504", re.IGNORECASE),
    "rate_limited": re.compile(r"rate.?limit|too many requests|\b429\b", re.IGNORECASE),
    "auth_failure": re.compile(r"unauthori[sz]ed|forbidden|invalid token|\b401\b|\b403\b|saml|sso", re.IGNORECASE),
    "payment_failure": re.compile(r"declined|insufficient funds|card|payment", re.IGNORECASE),
    "crash": re.compile(r"crash|segfault|fatal|panic|stack ?trace|force clos", re.IGNORECASE),
    "data_inconsistency": re.compile(r"mismatch|out of sync|missing records?|duplicate", re.IGNORECASE),
}


class LogInput(BaseModel):
    text: str = Field(min_length=1)


class LogOutput(BaseModel):
    http_statuses: list[int]
    exceptions: list[str]
    error_codes: list[str]
    signals: list[str]
    has_errors: bool


class LogAnalyzer(Skill[LogInput, LogOutput]):
    name = "log_analyzer"
    description = "Pull HTTP error statuses, exception names, error codes and failure signals out of text."
    input_model = LogInput
    output_model = LogOutput

    async def run(self, args: LogInput, ctx: SkillContext) -> LogOutput:
        text = args.text
        statuses = sorted({int(m) for m in _HTTP.findall(text)})
        exceptions = sorted(set(_EXC.findall(text)))
        codes = sorted(set(_CODE.findall(text)) - {c for c in _CODE.findall(text) if c.startswith("CUST-")})
        signals = [name for name, pattern in _SIGNALS.items() if pattern.search(text)]
        return LogOutput(
            http_statuses=statuses,
            exceptions=exceptions,
            error_codes=codes,
            signals=signals,
            has_errors=bool(statuses or exceptions or codes or signals),
        )
