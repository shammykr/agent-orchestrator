from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    roles: list[str]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000, description="Natural-language message")
    case_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.-]{1,64}$",
        description="Continue an existing case; omit to open one",
    )
    pipeline: str | None = Field(default=None, description="Force a pipeline instead of routing")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Since this morning none of our users can log in with SSO - Okta shows "
                "'invalid SAML response' and we get HTTP 403. Account CUST-1042.",
            }
        }
    }


class StageOut(BaseModel):
    id: str
    agent: str
    status: str
    attempts: int
    llm_calls: int
    repaired: bool
    duration_ms: float
    skills: list[dict[str, Any]]
    output: dict[str, Any] | None
    errors: list[str]
    note: str | None


class ChatResponse(BaseModel):
    case_id: str
    turn: int
    pipeline: str
    routed_by: str
    status: Literal["completed", "degraded", "failed"]
    case_status: str | None
    reply: str | None
    result: dict[str, Any] | None
    stages: list[StageOut]
    duration_ms: float
