#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import os
from urllib.parse import urlsplit, urlunsplit

import asyncpg


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalize_asyncpg_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    scheme = parsed.scheme.lower()
    if scheme == "postgresql+asyncpg":
        parsed = parsed._replace(scheme="postgresql")
    elif scheme == "postgres+asyncpg":
        parsed = parsed._replace(scheme="postgres")
    return urlunsplit(parsed)


def _admin_database_url(database_url: str) -> tuple[str, str]:
    parsed = urlsplit(_normalize_asyncpg_url(database_url))
    database_name = parsed.path.lstrip("/")
    if not database_name:
        raise ValueError("DATABASE_URL must include a database name")

    admin_path = "/postgres"
    admin_url = urlunsplit((parsed.scheme, parsed.netloc, admin_path, parsed.query, parsed.fragment))
    return admin_url, database_name


async def _database_exists(conn: asyncpg.Connection, database_name: str) -> bool:
    row = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", database_name)
    return bool(row)


async def _ensure_database(database_url: str, retries: int, retry_delay_seconds: float) -> None:
    normalized_database_url = _normalize_asyncpg_url(database_url)
    admin_url, target_db = _admin_database_url(normalized_database_url)
    for attempt in range(1, retries + 1):
        try:
            conn = await asyncpg.connect(admin_url, timeout=5)
            try:
                if not await _database_exists(conn, target_db):
                    await conn.execute(f"CREATE DATABASE {_quote_ident(target_db)}")
            finally:
                await conn.close()

            probe = await asyncpg.connect(normalized_database_url, timeout=5)
            try:
                value = await probe.fetchval("SELECT 1")
                if value != 1:
                    raise RuntimeError("Postgres probe query returned unexpected result")
            finally:
                await probe.close()
            return
        except Exception:
            if attempt >= retries:
                raise
            await asyncio.sleep(retry_delay_seconds)


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="Ensure launcher Postgres database exists and is queryable.")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection URL (defaults to DATABASE_URL env var).",
    )
    parser.add_argument("--retries", type=int, default=60)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.25)
    args = parser.parse_args()

    database_url = str(args.database_url or "").strip()
    if not database_url:
        raise ValueError("--database-url (or DATABASE_URL) is required")

    await _ensure_database(database_url, retries=max(1, args.retries), retry_delay_seconds=max(0.01, args.retry_delay_seconds))


if __name__ == "__main__":
    asyncio.run(_main_async())
