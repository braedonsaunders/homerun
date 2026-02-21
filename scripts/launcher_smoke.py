#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import os
import socket
from urllib.parse import urlsplit, urlunsplit

import asyncpg


def _redis_ping(host: str, port: int) -> None:
    payload = b"*1\r\n$4\r\nPING\r\n"
    with socket.create_connection((host, port), timeout=2.0) as sock:
        sock.settimeout(2.0)
        sock.sendall(payload)
        data = sock.recv(64)
    if b"+PONG" not in data:
        raise RuntimeError("Redis PING did not return PONG")


async def _postgres_ping(database_url: str) -> None:
    parsed = urlsplit(database_url)
    scheme = parsed.scheme.lower()
    if scheme == "postgresql+asyncpg":
        database_url = urlunsplit(parsed._replace(scheme="postgresql"))
    elif scheme == "postgres+asyncpg":
        database_url = urlunsplit(parsed._replace(scheme="postgres"))

    conn = await asyncpg.connect(database_url, timeout=5)
    try:
        value = await conn.fetchval("SELECT 1")
        if value != 1:
            raise RuntimeError("Postgres probe query returned unexpected result")
    finally:
        await conn.close()


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="Runtime smoke test for launcher-managed Redis + Postgres.")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "127.0.0.1"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    args = parser.parse_args()

    database_url = str(args.database_url or "").strip()
    if not database_url:
        raise ValueError("--database-url (or DATABASE_URL) is required")

    _redis_ping(args.redis_host, int(args.redis_port))
    await _postgres_ping(database_url)


if __name__ == "__main__":
    asyncio.run(_main_async())
