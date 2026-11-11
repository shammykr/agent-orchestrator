"""Redis-backed shared context store: the blackboard agents use to reason over each other's work.

Keys for a case (all expire ``context_ttl_seconds`` after the last write):

* ``case:{id}``            hash: created_at, updated_at, created_by, turns
* ``case:{id}:turns``      list: one JSON record per completed chat turn
* ``case:{id}:findings``   hash: stage id -> latest validated output of that stage (any turn)
* ``case:{id}:skills``     hash: skill name -> latest result
* ``case:{id}:lock``       string: held while a pipeline runs on the case

Every stage writes its output to ``findings`` as soon as it is validated, and every stage builds
its inputs from a fresh read of the store. So downstream agents, and agents in later turns,
reason over upstream findings through Redis, not through in-process variables.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis


class CaseBusyError(Exception):
    """Another message for the same case is being processed."""


class ContextStore:
    def __init__(self, redis: Redis, ttl_seconds: int) -> None:
        self.redis = redis
        self.ttl = ttl_seconds

    @staticmethod
    def _k(case_id: str, suffix: str = "") -> str:
        return f"case:{case_id}" + (f":{suffix}" if suffix else "")

    async def ping(self) -> bool:
        return bool(await self.redis.ping())

    async def exists(self, case_id: str) -> bool:
        return bool(await self.redis.exists(self._k(case_id)))

    async def ensure_case(self, case_id: str, user: str) -> None:
        now = _now()
        key = self._k(case_id)
        async with self.redis.pipeline(transaction=True) as p:
            p.hsetnx(key, "created_at", now)
            p.hsetnx(key, "created_by", user)
            p.hsetnx(key, "turns", 0)
            p.hset(key, "updated_at", now)
            await p.execute()

    @asynccontextmanager
    async def lock(self, case_id: str, timeout_seconds: int = 120) -> AsyncIterator[None]:
        """One pipeline run per case at a time (SET NX with a token, released only by its owner)."""
        key, token = self._k(case_id, "lock"), secrets.token_hex(8)
        if not await self.redis.set(key, token, nx=True, ex=timeout_seconds):
            raise CaseBusyError(case_id)
        try:
            yield
        finally:
            # Compare-and-delete so an expired lock taken over by someone else is not released.
            await self.redis.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end return 0",
                1,
                key,
                token,
            )

    async def put_finding(self, case_id: str, stage: str, output: dict[str, Any], turn: int, agent: str) -> None:
        record = {"output": output, "turn": turn, "agent": agent, "at": _now()}
        await self.redis.hset(self._k(case_id, "findings"), stage, json.dumps(record))
        await self._touch(case_id)

    async def put_skill_result(self, case_id: str, skill: str, result: dict[str, Any]) -> None:
        await self.redis.hset(self._k(case_id, "skills"), skill, json.dumps(result))

    async def findings(self, case_id: str) -> dict[str, dict[str, Any]]:
        raw = await self.redis.hgetall(self._k(case_id, "findings"))
        return {_s(k): json.loads(v) for k, v in raw.items()}

    async def turns(self, case_id: str) -> list[dict[str, Any]]:
        return [json.loads(t) for t in await self.redis.lrange(self._k(case_id, "turns"), 0, -1)]

    async def turn_count(self, case_id: str) -> int:
        return int(await self.redis.hget(self._k(case_id), "turns") or 0)

    async def append_turn(self, case_id: str, record: dict[str, Any]) -> None:
        async with self.redis.pipeline(transaction=True) as p:
            p.rpush(self._k(case_id, "turns"), json.dumps(record))
            p.hincrby(self._k(case_id), "turns", 1)
            p.hset(self._k(case_id), "updated_at", _now())
            await p.execute()
        await self._touch(case_id)

    async def snapshot(self, case_id: str) -> dict[str, Any] | None:
        meta = await self.redis.hgetall(self._k(case_id))
        if not meta:
            return None
        skills = await self.redis.hgetall(self._k(case_id, "skills"))
        return {
            "case_id": case_id,
            "meta": {_s(k): _s(v) for k, v in meta.items()},
            "turns": await self.turns(case_id),
            "findings": await self.findings(case_id),
            "skills": {_s(k): json.loads(v) for k, v in skills.items()},
            "ttl_seconds": await self.redis.ttl(self._k(case_id)),
        }

    async def _touch(self, case_id: str) -> None:
        async with self.redis.pipeline(transaction=False) as p:
            for suffix in ("", "turns", "findings", "skills"):
                p.expire(self._k(case_id, suffix), self.ttl)
            await p.execute()


def _now() -> str:
    return datetime.fromtimestamp(time.time(), UTC).isoformat(timespec="milliseconds")


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)
