#!/usr/bin/env python3
"""Wait until the local Redis container is responsive.

Mirrors ensure_postgres_ready.py: a CLI wrapper the GUI calls after
``docker compose up`` to gate worker / backend launch on Redis being
healthy enough to serve PINGs and basic GET/SET.

Exits 0 on success, non-zero on timeout.  Stderr carries the most recent
error so the GUI can surface it in the SYSTEM log.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"redis package not importable: {exc}\n")
    sys.exit(2)


async def _probe(url: str) -> None:
    client = aioredis.Redis.from_url(
        url,
        socket_timeout=2.0,
        socket_connect_timeout=2.0,
        decode_responses=True,
    )
    try:
        pong = await client.ping()
        if not (pong is True or pong == "PONG" or pong == b"PONG"):
            raise RuntimeError(f"unexpected PING reply: {pong!r}")
        await client.set("__ensure_redis_ready__", "1", ex=10)
        value = await client.get("__ensure_redis_ready__")
        if value != "1":
            raise RuntimeError("SET/GET round-trip mismatch")
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


async def _wait(url: str, retries: int, retry_delay_seconds: float) -> None:
    last_error: Optional[BaseException] = None
    for attempt in range(1, max(retries, 1) + 1):
        try:
            await _probe(url)
            return
        except (RedisError, OSError, RuntimeError, asyncio.TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(retry_delay_seconds)
    if last_error is not None:
        sys.stderr.write(f"redis not ready: {last_error}\n")
        raise SystemExit(1)


async def _main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure launcher Redis is reachable + responsive.",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        help="Redis connection URL (defaults to REDIS_URL env var).",
    )
    parser.add_argument("--retries", type=int, default=60)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.25)
    args = parser.parse_args()

    url = str(args.redis_url or "").strip()
    if not url:
        sys.stderr.write("--redis-url (or REDIS_URL) is required\n")
        raise SystemExit(2)

    await _wait(
        url,
        retries=max(1, args.retries),
        retry_delay_seconds=max(0.01, args.retry_delay_seconds),
    )


if __name__ == "__main__":
    asyncio.run(_main_async())
