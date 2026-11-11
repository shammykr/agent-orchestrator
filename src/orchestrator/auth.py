"""JWT authentication middleware and role-based access control.

* ``POST /auth/token`` exchanges username/password (``config/users.yaml``, PBKDF2 hashes) for an
  HS256 JWT with ``sub``, ``roles``, ``iss``, ``iat`` and ``exp`` claims.
* :class:`JWTAuthMiddleware` authenticates every ``/api`` request: it verifies the bearer token's
  signature, expiry and issuer, rejects with 401 otherwise, and puts the principal on
  ``request.state.principal``.
* :func:`require_role` is a FastAPI dependency for authorization. Roles are hierarchical:
  ``viewer`` < ``operator`` < ``admin``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import yaml
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from orchestrator.config import Settings
from orchestrator.observability import AUTH_FAILURES

log = logging.getLogger(__name__)

ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}


class UserStore:
    def __init__(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text()) or {}
        self._users: dict[str, dict[str, Any]] = {u["username"]: u for u in data.get("users", [])}
        for name, user in self._users.items():
            unknown = set(user.get("roles", [])) - set(ROLE_LEVEL)
            if unknown:
                raise ValueError(f"user {name}: unknown roles {sorted(unknown)}")

    def authenticate(self, username: str, password: str) -> list[str] | None:
        user = self._users.get(username)
        if user is None or not verify_password(password, user["password_hash"]):
            return None
        return list(user.get("roles", []))


def hash_password(password: str, salt: bytes, iterations: int = 200_000) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt, _digest = encoded.split("$")
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    candidate = hash_password(password, base64.b64decode(salt), int(iterations))
    return hmac.compare_digest(candidate, encoded)


def issue_token(settings: Settings, username: str, roles: list[str]) -> tuple[str, int]:
    now = datetime.now(UTC)
    ttl = timedelta(minutes=settings.jwt_ttl_minutes)
    claims = {"sub": username, "roles": roles, "iss": settings.jwt_issuer, "iat": now, "exp": now + ttl}
    token = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, int(ttl.total_seconds())


def decode_token(settings: Settings, token: str) -> dict[str, Any]:
    claims: dict[str, Any] = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        issuer=settings.jwt_issuer,
        options={"require": ["sub", "exp", "iat", "iss"]},
    )
    roles = claims.get("roles")
    if not isinstance(roles, list) or not all(r in ROLE_LEVEL for r in roles):
        raise jwt.InvalidTokenError("roles claim missing or invalid")
    return claims


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Authenticates requests under ``protected_prefix``; everything else passes through."""

    def __init__(self, app: ASGIApp, settings: Settings, protected_prefix: str = "/api") -> None:
        super().__init__(app)
        self.settings = settings
        self.prefix = protected_prefix

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if not request.url.path.startswith(self.prefix):
            return await call_next(request)
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return _unauthorized("missing bearer token", "missing_token")
        try:
            request.state.principal = decode_token(self.settings, token)
        except jwt.ExpiredSignatureError:
            return _unauthorized("token expired", "expired")
        except jwt.InvalidTokenError as e:
            return _unauthorized(f"invalid token: {e}", "invalid_token")
        return await call_next(request)


def _unauthorized(detail: str, reason: str) -> JSONResponse:
    AUTH_FAILURES.labels(reason).inc()
    return JSONResponse(
        {"detail": detail}, status_code=status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"}
    )


def require_role(minimum: str) -> Callable[[Request], dict[str, Any]]:
    """Dependency: the caller must hold ``minimum`` or a higher role."""
    needed = ROLE_LEVEL[minimum]

    def check(request: Request) -> dict[str, Any]:
        principal: dict[str, Any] | None = getattr(request.state, "principal", None)
        if principal is None:  # route mounted outside the middleware's prefix
            AUTH_FAILURES.labels("missing_token").inc()
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required")
        level = max((ROLE_LEVEL[r] for r in principal["roles"]), default=0)
        if level < needed:
            AUTH_FAILURES.labels("forbidden").inc()
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires role {minimum!r} or higher")
        return principal

    return check
