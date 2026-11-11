from __future__ import annotations

import dataclasses
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from orchestrator import __version__
from orchestrator.api.schemas import ChatRequest, ChatResponse, StageOut, TokenRequest, TokenResponse
from orchestrator.auth import issue_token, require_role
from orchestrator.context import CaseBusyError
from orchestrator.engine import Orchestrator
from orchestrator.platform import load_components

router = APIRouter()

Viewer = Annotated[dict[str, Any], Depends(require_role("viewer"))]
Operator = Annotated[dict[str, Any], Depends(require_role("operator"))]
Admin = Annotated[dict[str, Any], Depends(require_role("admin"))]


def _orch(request: Request) -> Orchestrator:
    orch: Orchestrator = request.app.state.orchestrator
    return orch


# ---- public -------------------------------------------------------------------------------------


@router.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def token(body: TokenRequest, request: Request) -> TokenResponse:
    roles = request.app.state.users.authenticate(body.username, body.password)
    if roles is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
    access, ttl = issue_token(request.app.state.settings, body.username, roles)
    return TokenResponse(access_token=access, expires_in=ttl, roles=roles)


@router.get("/health", tags=["ops"])
async def health(request: Request) -> JSONResponse:
    orch = _orch(request)
    try:
        redis_ok = await orch.store.ping()
    except Exception:
        redis_ok = False
    body = {
        "status": "ok" if redis_ok else "degraded",
        "version": __version__,
        "uptime_s": round(time.monotonic() - request.app.state.started, 1),
        "checks": {"redis": "up" if redis_ok else "down"},
        "llm_provider": orch.llm.provider,
        "loaded": {
            "pipelines": sorted(orch.pipelines),
            "agents": sorted(orch.agent_specs),
            "skills": orch.skills.names(),
        },
    }
    return JSONResponse(body, status_code=200 if redis_ok else 503)


@router.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---- chat workflow --------------------------------------------------------------------------------


@router.post("/api/chat", response_model=ChatResponse, tags=["chat"])
async def chat(body: ChatRequest, request: Request, user: Operator) -> ChatResponse:
    """Route a natural-language message through an agent pipeline and return structured results."""
    orch = _orch(request)
    if body.case_id is not None and not await orch.store.exists(body.case_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"case {body.case_id} not found")
    try:
        result = await orch.handle(body.message, body.case_id, user, body.pipeline)
    except CaseBusyError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, f"case {e} is already processing a message") from e
    except KeyError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e.args[0])) from e
    return ChatResponse(
        **{k: v for k, v in dataclasses.asdict(result).items() if k != "stages"},
        stages=[StageOut(**dataclasses.asdict(s)) for s in result.stages],
    )


@router.get("/api/cases/{case_id}", tags=["chat"])
async def get_case(case_id: str, request: Request, user: Viewer) -> dict[str, Any]:
    """The case's shared context: turns, latest findings per stage, latest skill results."""
    snapshot = await _orch(request).store.snapshot(case_id)
    if snapshot is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"case {case_id} not found")
    return snapshot


# ---- registry / contracts -------------------------------------------------------------------------


@router.get("/api/pipelines", tags=["registry"])
async def pipelines(request: Request, user: Viewer) -> list[dict[str, Any]]:
    return [p.model_dump(exclude_none=True) for p in _orch(request).pipelines.values()]


@router.get("/api/agents", tags=["registry"])
async def agents(request: Request, user: Viewer) -> list[dict[str, Any]]:
    return [s.contract().model_dump() for s in _orch(request).agent_specs.values()]


@router.get("/api/skills", tags=["registry"])
async def skills(request: Request, user: Viewer) -> list[dict[str, Any]]:
    return [c.model_dump() for c in _orch(request).skills.contracts()]


@router.get("/api/contracts", tags=["registry"])
async def contracts(request: Request, user: Viewer) -> dict[str, Any]:
    """Every integration contract: agent and skill input/output JSON schemas."""
    orch = _orch(request)
    return {
        "agents": {s.name: s.contract().model_dump() for s in orch.agent_specs.values()},
        "skills": {c.name: c.model_dump() for c in orch.skills.contracts()},
    }


@router.post("/api/admin/reload", tags=["admin"])
async def reload(request: Request, user: Admin) -> dict[str, Any]:
    """Re-discover skills and reload agent and pipeline YAML without restarting."""
    orch = _orch(request)
    try:
        components = load_components(request.app.state.settings)
    except Exception as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"reload rejected: {e}") from e
    orch.configure(components.skills, components.agents, components.pipelines)
    return {
        "pipelines": sorted(orch.pipelines),
        "agents": sorted(orch.agent_specs),
        "skills": orch.skills.names(),
    }
