"""Structured output: extract JSON from model text, validate it against a JSON schema, and map
values between pipeline stages.

Models wrap JSON in prose or Markdown fences, truncate it, or drop fields. :func:`parse_structured`
turns a raw completion into a validated object or raises :class:`StructuredOutputError` with
messages precise enough to send back to the model in a repair prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(Exception):
    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message + ": " + "; ".join(errors))
        self.errors = errors


def check_schema(schema: Mapping[str, Any]) -> None:
    """Raise ``SchemaError`` if ``schema`` is not a valid JSON schema (checked at config load)."""
    Draft202012Validator.check_schema(schema)


def extract_json(text: str) -> Any:
    """Return the first JSON object in ``text`` (inside a ```json fence, or the first balanced {...})."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        obj = _first_balanced_object(candidate)
        if obj is not None:
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                continue
    raise StructuredOutputError(
        "no JSON object found in model output", [f"no parseable JSON object in: {_preview(text)}"]
    )


def validation_errors(schema: Mapping[str, Any], value: Any) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path)):
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{path}: {err.message}")
    return errors


def parse_structured(text: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    value = extract_json(text)
    if not isinstance(value, dict):
        raise StructuredOutputError("model output is not a JSON object", [f"got {type(value).__name__}"])
    errors = validation_errors(schema, value)
    if errors:
        raise StructuredOutputError("model output does not match the output schema", errors)
    return value


def validate(schema: Mapping[str, Any], value: Any, what: str) -> None:
    errors = validation_errors(schema, value)
    if errors:
        raise StructuredOutputError(f"{what} does not match its schema", errors)


# --- mapping values between stages -----------------------------------------------------------------

_PATH = re.compile(r"^\$(\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*$")


def resolve(path: str, data: Mapping[str, Any]) -> Any:
    """Resolve a ``$.a.b[0].c`` path against ``data``; missing keys resolve to None."""
    if not _PATH.match(path):
        raise ValueError(f"invalid path {path!r} (expected $.field.sub[0])")
    current: Any = data
    for token in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", path):
        key, index = token
        if current is None:
            return None
        if key:
            current = current.get(key) if isinstance(current, Mapping) else None
        else:
            i = int(index)
            current = current[i] if isinstance(current, list) and i < len(current) else None
    return current


def render(template: Any, data: Mapping[str, Any]) -> Any:
    """Build a value from a mapping template.

    * ``"$.path"`` strings are replaced by the value at that path;
    * ``{"$coalesce": ["$.a", "$.b", "default"]}`` takes the first non-null value;
    * ``{"$count": "$.path"}`` gives the length of a list (0 if missing);
    * dicts and lists are rendered recursively; other literals are kept as they are.
    """
    if isinstance(template, str) and template.startswith("$."):
        return resolve(template, data)
    if isinstance(template, Mapping):
        if set(template) == {"$coalesce"}:
            for option in template["$coalesce"]:
                value = render(option, data)
                if value is not None:
                    return value
            return None
        if set(template) == {"$count"}:
            value = render(template["$count"], data)
            return len(value) if isinstance(value, list) else 0
        return {k: render(v, data) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, data) for v in template]
    return template


def drop_nulls(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if v is not None}


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _preview(text: str, n: int = 80) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[:n] + "..."


__all__ = [
    "SchemaError",
    "StructuredOutputError",
    "check_schema",
    "drop_nulls",
    "extract_json",
    "parse_structured",
    "render",
    "resolve",
    "validate",
    "validation_errors",
]
