"""Structured JSON logging and Prometheus metrics."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from typing import Any

from prometheus_client import Counter, Histogram

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
case_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("case_id", default=None)

_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line. ``extra=`` fields and the request/case ids become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if (rid := request_id_var.get()) is not None:
            entry["request_id"] = rid
        if (cid := case_id_var.get()) is not None:
            entry["case_id"] = cid
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc"] = "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our access log
    logging.getLogger("httpx").setLevel(logging.WARNING)


# --- metrics (exposed at /metrics) --------------------------------------------------------------

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PIPELINE_RUNS = Counter("pipeline_runs_total", "Pipeline executions", ["pipeline", "status"])
PIPELINE_LATENCY = Histogram("pipeline_duration_seconds", "End-to-end pipeline latency", ["pipeline"])
AGENT_RUNS = Counter("agent_runs_total", "Agent stage executions", ["agent", "status"])
AGENT_LATENCY = Histogram("agent_duration_seconds", "Agent stage latency", ["agent"])
SKILL_CALLS = Counter("skill_invocations_total", "Skill invocations", ["skill", "status"])
LLM_CALLS = Counter("llm_calls_total", "LLM completions", ["provider", "status"])
VALIDATION_FAILURES = Counter(
    "structured_output_validation_failures_total", "Agent outputs rejected by schema validation", ["agent"]
)
FALLBACKS = Counter("pipeline_fallbacks_total", "Fallback strategies applied", ["pipeline", "stage", "strategy"])
AUTH_FAILURES = Counter("auth_failures_total", "Rejected requests", ["reason"])
