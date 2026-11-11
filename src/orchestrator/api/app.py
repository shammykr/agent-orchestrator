"""FastAPI application wiring: lifespan, middleware, routes."""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from redis.asyncio import Redis

from orchestrator import __version__
from orchestrator.api.routes import router
from orchestrator.auth import JWTAuthMiddleware, UserStore
from orchestrator.config import Settings, get_settings
from orchestrator.context import ContextStore
from orchestrator.engine import Orchestrator
from orchestrator.llm import build_llm
from orchestrator.observability import HTTP_LATENCY, HTTP_REQUESTS, configure_logging, request_id_var
from orchestrator.platform import load_components

log = logging.getLogger("orchestrator.api")
access_log = logging.getLogger("orchestrator.access")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        components = load_components(settings)
        redis = Redis.from_url(settings.redis_url)
        llm = build_llm(settings)
        app.state.settings = settings
        app.state.users = UserStore(settings.users_file)
        app.state.started = time.monotonic()
        app.state.orchestrator = Orchestrator(
            settings,
            ContextStore(redis, settings.context_ttl_seconds),
            llm,
            components.skills,
            components.agents,
            components.pipelines,
        )
        log.info(
            "orchestrator started",
            extra={
                "version": __version__,
                "llm_provider": llm.provider,
                "pipelines": sorted(components.pipelines),
                "agents": sorted(components.agents),
                "skills": components.skills.names(),
            },
        )
        try:
            yield
        finally:
            await llm.aclose()
            await redis.aclose()

    app = FastAPI(
        title="Agent Orchestration Platform",
        version=__version__,
        description="Chat-driven multi-agent workflows with structured output and pluggable skills.",
        lifespan=lifespan,
    )
    app.include_router(router)
    # Added first = runs inside the access-log middleware below.
    app.add_middleware(JWTAuthMiddleware, settings=settings, protected_prefix="/api")

    @app.middleware("http")
    async def access_and_metrics(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            elapsed = time.perf_counter() - started
            route = getattr(request.scope.get("route"), "path", "unmatched")
            HTTP_REQUESTS.labels(request.method, route, str(status_code)).inc()
            HTTP_LATENCY.labels(request.method, route).observe(elapsed)
            principal = getattr(request.state, "principal", None)
            access_log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "route": route,
                    "status": status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                    "user": principal["sub"] if principal else None,
                },
            )
            request_id_var.reset(token)

    return app


app = create_app()
