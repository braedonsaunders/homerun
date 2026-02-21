from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from typing import Any, Optional

from redis.asyncio import Redis

from config import settings
from utils.logger import get_logger

logger = get_logger("redis_streams")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return list(value)
    return str(value)


class RedisStreamClient:
    def __init__(self) -> None:
        self._client: Optional[Redis] = None
        self._lock = asyncio.Lock()

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            await client.aclose()
        except Exception:
            pass

    async def _get_client(self) -> Redis:
        client = self._client
        if client is not None:
            return client
        async with self._lock:
            client = self._client
            if client is not None:
                return client
            kwargs: dict[str, Any] = {
                "host": settings.REDIS_HOST,
                "port": int(settings.REDIS_PORT),
                "db": int(settings.REDIS_DB),
                "decode_responses": True,
                "socket_connect_timeout": float(settings.REDIS_CONNECT_TIMEOUT_SECONDS),
                "socket_timeout": float(settings.REDIS_SOCKET_TIMEOUT_SECONDS),
                "health_check_interval": 30,
            }
            if settings.REDIS_PASSWORD:
                kwargs["password"] = settings.REDIS_PASSWORD
            self._client = Redis(**kwargs)
            return self._client

    async def ping(self) -> bool:
        try:
            client = await self._get_client()
            return bool(await client.ping())
        except Exception as exc:
            logger.debug("Redis ping failed", exc_info=exc)
            await self._close_client()
            return False

    async def append_json(
        self,
        stream: str,
        payload: dict[str, Any],
        *,
        maxlen: int,
    ) -> Optional[str]:
        try:
            client = await self._get_client()
            body = json.dumps(payload, default=_json_default, separators=(",", ":"))
            entry_id = await client.xadd(
                stream,
                {"data": body},
                maxlen=max(1000, int(maxlen)),
                approximate=True,
            )
            return str(entry_id)
        except Exception as exc:
            logger.debug(
                "Redis stream append failed",
                stream=stream,
                exc_info=exc,
            )
            await self._close_client()
            return None

    async def read_raw(
        self,
        stream: str,
        *,
        last_id: str,
        block_ms: int,
        count: int,
    ) -> list[tuple[str, str]]:
        try:
            client = await self._get_client()
            chunks = await client.xread(
                {stream: last_id},
                block=max(1, int(block_ms)),
                count=max(1, int(count)),
            )
        except Exception as exc:
            logger.debug(
                "Redis stream read failed",
                stream=stream,
                exc_info=exc,
            )
            await self._close_client()
            await asyncio.sleep(0.5)
            return []

        out: list[tuple[str, str]] = []
        for _stream_name, entries in chunks:
            for entry_id, fields in entries:
                raw = fields.get("data")
                if isinstance(raw, str):
                    out.append((str(entry_id), raw))
        return out

    async def hgetall_many(self, keys: list[str]) -> list[dict[str, str]]:
        if not keys:
            return []
        try:
            client = await self._get_client()
            pipe = client.pipeline(transaction=False)
            for key in keys:
                pipe.hgetall(key)
            rows = await pipe.execute()
        except Exception as exc:
            logger.debug(
                "Redis hash read failed",
                keys_count=len(keys),
                exc_info=exc,
            )
            await self._close_client()
            return []

        out: list[dict[str, str]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append({str(k): str(v) for k, v in row.items()})
            else:
                out.append({})
        return out

    async def hset_many(self, rows: dict[str, dict[str, str]], *, expire_seconds: int | None = None) -> None:
        if not rows:
            return
        try:
            client = await self._get_client()
            pipe = client.pipeline(transaction=False)
            for key, payload in rows.items():
                if not payload:
                    continue
                pipe.hset(key, mapping=payload)
                if expire_seconds and expire_seconds > 0:
                    pipe.expire(key, int(expire_seconds))
            await pipe.execute()
        except Exception as exc:
            logger.debug(
                "Redis hash write failed",
                keys_count=len(rows),
                exc_info=exc,
            )
            await self._close_client()

    async def close(self) -> None:
        await self._close_client()


redis_streams = RedisStreamClient()
