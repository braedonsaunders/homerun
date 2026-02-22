from __future__ import annotations

import os
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


DEFAULT_DATABASE_URL = "postgresql+asyncpg://homerun:homerun@127.0.0.1:5432/homerun"


class ManagedAsyncEngine:
    def __init__(self, engine: AsyncEngine, cleanup_fn) -> None:
        self._engine = engine
        self._cleanup_fn = cleanup_fn
        self._disposed = False

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        await self._engine.dispose()
        await self._cleanup_fn()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._engine, item)


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _normalize_base_database_url(database_url: str) -> str:
    parsed = urlsplit(str(database_url or "").strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"postgresql+asyncpg", "postgres+asyncpg", "postgresql", "postgres"}:
        raise ValueError(
            "TEST_DATABASE_URL or DATABASE_URL must use a PostgreSQL scheme "
            "(postgresql+asyncpg, postgres+asyncpg, postgresql, or postgres)"
        )
    if not parsed.path or parsed.path.strip("/") == "":
        raise ValueError("TEST_DATABASE_URL or DATABASE_URL must include a database name")
    return urlunsplit(parsed._replace(scheme="postgresql+asyncpg"))


def _to_asyncpg_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    scheme = parsed.scheme.lower()
    if scheme == "postgresql+asyncpg":
        parsed = parsed._replace(scheme="postgresql")
    elif scheme == "postgres+asyncpg":
        parsed = parsed._replace(scheme="postgres")
    return urlunsplit(parsed)


def _replace_database_in_url(database_url: str, database_name: str) -> str:
    parsed = urlsplit(database_url)
    return urlunsplit(parsed._replace(path=f"/{database_name}"))


def _build_database_name(prefix: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(prefix or "test")).strip("_").lower()
    if not normalized:
        normalized = "test"
    # PostgreSQL identifiers max at 63 bytes.
    base = f"homerun_test_{normalized}"
    return f"{base[:48]}_{uuid.uuid4().hex[:10]}"


async def build_postgres_session_factory(
    base_metadata: Any,
    database_name_prefix: str,
) -> tuple[ManagedAsyncEngine, sessionmaker]:
    base_database_url = _normalize_base_database_url(
        os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    )
    database_name = _build_database_name(database_name_prefix)

    admin_asyncpg_url = _replace_database_in_url(_to_asyncpg_url(base_database_url), "postgres")

    admin_conn = await asyncpg.connect(admin_asyncpg_url, timeout=10)
    try:
        await admin_conn.execute(f"CREATE DATABASE {_quote_ident(database_name)}")
    finally:
        await admin_conn.close()

    test_database_url = _replace_database_in_url(base_database_url, database_name)
    engine = create_async_engine(test_database_url, echo=False, pool_pre_ping=True)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    metadata = getattr(base_metadata, "metadata", base_metadata)
    create_all = getattr(metadata, "create_all", None)
    if create_all is None:
        raise TypeError("base_metadata must be a SQLAlchemy MetaData or declarative base")

    async with engine.begin() as conn:
        await conn.run_sync(create_all)

    async def _drop_database() -> None:
        drop_conn = await asyncpg.connect(admin_asyncpg_url, timeout=10)
        try:
            await drop_conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
                database_name,
            )
            await drop_conn.execute(f"DROP DATABASE IF EXISTS {_quote_ident(database_name)}")
        finally:
            await drop_conn.close()

    return ManagedAsyncEngine(engine, _drop_database), session_factory
