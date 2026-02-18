"""Data Source SDK — stable helpers for source authors and strategy developers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Select, desc, select

from models.database import AsyncSessionLocal, DataSource, DataSourceRecord, DataSourceRun


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _serialize_record(row: DataSourceRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "data_source_id": row.data_source_id,
        "source_slug": row.source_slug,
        "external_id": row.external_id,
        "title": row.title,
        "summary": row.summary,
        "category": row.category,
        "source": row.source,
        "url": row.url,
        "geotagged": bool(row.geotagged),
        "country_iso3": row.country_iso3,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "observed_at": row.observed_at.isoformat() if row.observed_at else None,
        "ingested_at": row.ingested_at.isoformat() if row.ingested_at else None,
        "payload": dict(row.payload_json or {}),
        "transformed": dict(row.transformed_json or {}),
        "tags": list(row.tags_json or []),
    }


def _serialize_run(row: DataSourceRun) -> dict[str, Any]:
    return {
        "id": row.id,
        "data_source_id": row.data_source_id,
        "source_slug": row.source_slug,
        "status": row.status,
        "fetched_count": int(row.fetched_count or 0),
        "transformed_count": int(row.transformed_count or 0),
        "upserted_count": int(row.upserted_count or 0),
        "skipped_count": int(row.skipped_count or 0),
        "error_message": row.error_message,
        "metadata": dict(row.metadata_json or {}),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "duration_ms": row.duration_ms,
    }


class BaseDataSource:
    """Base class for all DB-defined data sources."""

    name = "Custom Data Source"
    description = ""
    default_config: dict[str, Any] = {}

    def __init__(self) -> None:
        self.config: dict[str, Any] = dict(self.default_config)

    def configure(self, config: dict[str, Any] | None) -> None:
        merged = dict(self.default_config)
        if config:
            merged.update(dict(config))
        self.config = merged

    def fetch(self) -> list[dict[str, Any]]:
        return []

    async def fetch_async(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.fetch)

    def transform(self, item: dict[str, Any]) -> dict[str, Any]:
        return dict(item)


class DataSourceSDK:
    """High-level APIs for strategy code and source code."""

    @staticmethod
    async def get_records(
        *,
        source_slug: str | None = None,
        source_slugs: list[str] | None = None,
        limit: int = 200,
        geotagged: bool | None = None,
        category: str | None = None,
        since: str | datetime | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(2000, int(limit)))
        since_dt = _parse_datetime(since)

        query: Select[tuple[DataSourceRecord]] = select(DataSourceRecord)
        if source_slug:
            query = query.where(DataSourceRecord.source_slug == str(source_slug).strip().lower())
        elif source_slugs:
            normalized = [str(value).strip().lower() for value in source_slugs if str(value).strip()]
            if normalized:
                query = query.where(DataSourceRecord.source_slug.in_(normalized))
        if geotagged is not None:
            query = query.where(DataSourceRecord.geotagged == bool(geotagged))
        if category:
            query = query.where(DataSourceRecord.category == str(category).strip().lower())
        if since_dt is not None:
            query = query.where(DataSourceRecord.ingested_at >= since_dt)

        query = query.order_by(
            desc(DataSourceRecord.observed_at),
            desc(DataSourceRecord.ingested_at),
        ).limit(safe_limit)

        async with AsyncSessionLocal() as session:
            rows = (await session.execute(query)).scalars().all()

        return [_serialize_record(row) for row in rows]

    @staticmethod
    async def get_latest_record(
        source_slug: str,
        external_id: str | None = None,
    ) -> dict[str, Any] | None:
        slug = str(source_slug or "").strip().lower()
        if not slug:
            return None

        query: Select[tuple[DataSourceRecord]] = select(DataSourceRecord).where(DataSourceRecord.source_slug == slug)
        if external_id:
            query = query.where(DataSourceRecord.external_id == str(external_id).strip())

        query = query.order_by(
            desc(DataSourceRecord.observed_at),
            desc(DataSourceRecord.ingested_at),
        ).limit(1)

        async with AsyncSessionLocal() as session:
            row = (await session.execute(query)).scalar_one_or_none()

        if row is None:
            return None
        return _serialize_record(row)

    @staticmethod
    async def list_sources(enabled_only: bool = True) -> list[dict[str, Any]]:
        query: Select[tuple[DataSource]] = select(DataSource).order_by(DataSource.sort_order.asc(), DataSource.slug.asc())
        if enabled_only:
            query = query.where(DataSource.enabled == True)  # noqa: E712

        async with AsyncSessionLocal() as session:
            rows = (await session.execute(query)).scalars().all()

        return [
            {
                "id": row.id,
                "slug": row.slug,
                "source_key": row.source_key,
                "source_kind": row.source_kind,
                "name": row.name,
                "description": row.description,
                "enabled": bool(row.enabled),
                "status": row.status,
                "version": int(row.version or 1),
                "is_system": bool(row.is_system),
            }
            for row in rows
        ]

    @staticmethod
    async def get_recent_runs(
        source_slug: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        slug = str(source_slug or "").strip().lower()
        if not slug:
            return []

        safe_limit = max(1, min(200, int(limit)))
        query = (
            select(DataSourceRun)
            .where(DataSourceRun.source_slug == slug)
            .order_by(desc(DataSourceRun.started_at))
            .limit(safe_limit)
        )
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(query)).scalars().all()

        return [_serialize_run(row) for row in rows]

    @staticmethod
    async def run_source(source_slug: str, max_records: int = 500) -> dict[str, Any]:
        slug = str(source_slug or "").strip().lower()
        if not slug:
            raise ValueError("source_slug is required")

        safe_limit = max(1, min(5000, int(max_records)))
        from services.data_source_runner import run_data_source_by_slug

        async with AsyncSessionLocal() as session:
            result = await run_data_source_by_slug(
                session,
                source_slug=slug,
                max_records=safe_limit,
                commit=True,
            )

        return result


__all__ = [
    "BaseDataSource",
    "DataSourceSDK",
    "_utcnow_naive",
]
