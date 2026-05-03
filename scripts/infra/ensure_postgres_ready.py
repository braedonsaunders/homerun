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
    last_progress_attempt = 0
    recovery_warned = False
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
            if recovery_warned:
                # Acknowledge to the user that the wait paid off, so the
                # progress messages don't end without resolution.
                print(
                    f"Postgres recovery completed after {attempt} attempts "
                    f"({attempt * retry_delay_seconds:.1f}s); database is ready.",
                    flush=True,
                )
            return
        except asyncpg.exceptions.CannotConnectNowError as exc:
            # Postgres is up but still in WAL crash recovery
            # ("Consistent recovery state has not been yet reached").
            # This is a "be patient" signal, not a real failure — recovery
            # on a busy DB can take several minutes after an unclean
            # shutdown (SIGKILL, host sleep, Docker Desktop quit). Print
            # progress so the launcher console doesn't appear hung, and
            # let the retry loop continue.
            if not recovery_warned:
                print(
                    "Postgres is recovering after a prior unclean shutdown; "
                    "waiting for consistent recovery state...",
                    flush=True,
                )
                print(f"  detail: {exc}", flush=True)
                recovery_warned = True
                last_progress_attempt = attempt
            elif attempt - last_progress_attempt >= 20:
                # Heartbeat every ~10s (20 × 0.5s) so the user can see the
                # wait is making progress, not stalled.
                elapsed = attempt * retry_delay_seconds
                print(
                    f"  still recovering... attempt {attempt}/{retries} "
                    f"(~{elapsed:.0f}s elapsed)",
                    flush=True,
                )
                last_progress_attempt = attempt
            if attempt >= retries:
                raise
            await asyncio.sleep(retry_delay_seconds)
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
