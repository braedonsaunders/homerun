"""Data Source SDK — stable helpers for source authors and strategy developers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
from typing import Any
import uuid

from sqlalchemy import Select, desc, select

from models.database import AsyncSessionLocal, DataSource, DataSourceRecord, DataSourceRun, DataSourceTombstone

from services.data_source_loader import (
    DataSourceValidationError,
    data_source_loader,
    validate_data_source_source,
)

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}[a-z0-9]$")


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


def _serialize_source(row: DataSource, runtime: Any | None = None, *, include_code: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": row.id,
        "slug": row.slug,
        "source_key": row.source_key,
        "source_kind": row.source_kind,
        "name": row.name,
        "description": row.description,
        "class_name": row.class_name,
        "is_system": bool(row.is_system),
        "enabled": bool(row.enabled),
        "status": row.status,
        "error_message": row.error_message,
        "version": int(row.version or 1),
        "config": dict(row.config or {}),
        "config_schema": dict(row.config_schema or {}),
        "sort_order": int(row.sort_order or 0),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "runtime": None,
    }
    if include_code:
        out["source_code"] = row.source_code

    if runtime is not None:
        out["runtime"] = {
            "slug": runtime.slug,
            "class_name": runtime.class_name,
            "name": runtime.name,
            "description": runtime.description,
            "loaded_at": runtime.loaded_at.isoformat(),
            "source_hash": runtime.source_hash,
            "run_count": runtime.run_count,
            "error_count": runtime.error_count,
            "last_run": runtime.last_run.isoformat() if runtime.last_run else None,
            "last_error": runtime.last_error,
        }
    return out


def _normalize_slug(value: str) -> str:
    slug = str(value or "").strip().lower()
    if not _SLUG_RE.match(slug):
        raise ValueError(
            "Invalid slug. Must be 3-50 chars, start with a letter, use lowercase letters/numbers/underscores, "
            "and end with letter/number."
        )
    return slug


def _normalize_key(value: str, default: str) -> str:
    out = str(value or "").strip().lower()
    return out or default


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
    async def list_sources(
        enabled_only: bool = True,
        source_key: str | None = None,
        *,
        include_code: bool = False,
    ) -> list[dict[str, Any]]:
        query: Select[tuple[DataSource]] = select(DataSource).order_by(
            DataSource.is_system.desc(),
            DataSource.sort_order.asc(),
            DataSource.slug.asc(),
        )
        if enabled_only:
            query = query.where(DataSource.enabled == True)  # noqa: E712
        if source_key:
            query = query.where(DataSource.source_key == str(source_key).strip().lower())

        async with AsyncSessionLocal() as session:
            rows = (await session.execute(query)).scalars().all()

        out: list[dict[str, Any]] = []
        for row in rows:
            runtime = data_source_loader.get_runtime(str(row.slug or "").strip().lower())
            out.append(_serialize_source(row, runtime, include_code=include_code))
        return out

    @staticmethod
    async def get_source(source_slug: str, *, include_code: bool = True) -> dict[str, Any]:
        slug = str(source_slug or "").strip().lower()
        if not slug:
            raise ValueError("source_slug is required")

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(select(DataSource).where(DataSource.slug == slug))
            ).scalar_one_or_none()
            if row is None:
                raise ValueError(f"Data source '{slug}' not found")

        runtime = data_source_loader.get_runtime(slug)
        return _serialize_source(row, runtime, include_code=include_code)

    @staticmethod
    def validate_source(source_code: str, class_name: str | None = None) -> dict[str, Any]:
        return validate_data_source_source(source_code, class_name=class_name)

    @staticmethod
    async def create_source(
        *,
        slug: str,
        source_code: str,
        source_key: str = "custom",
        source_kind: str = "python",
        name: str | None = None,
        description: str | None = None,
        class_name: str | None = None,
        config: dict[str, Any] | None = None,
        config_schema: dict[str, Any] | None = None,
        enabled: bool = True,
        is_system: bool = False,
        sort_order: int = 0,
    ) -> dict[str, Any]:
        normalized_slug = _normalize_slug(slug)
        normalized_source_key = _normalize_key(source_key, "custom")
        normalized_source_kind = _normalize_key(source_kind, "python")
        validation = validate_data_source_source(source_code, class_name=class_name)
        if not validation["valid"]:
            raise ValueError("; ".join(validation.get("errors") or ["Data source validation failed"]))

        resolved_class_name = validation.get("class_name")
        resolved_name = str(name or validation.get("source_name") or normalized_slug.replace("_", " ").title()).strip()
        if not resolved_name:
            raise ValueError("name is required")
        resolved_description = (
            description
            if description is not None
            else validation.get("source_description")
        )

        status = "unloaded"
        error_message = None

        async with AsyncSessionLocal() as session:
            existing = (
                await session.execute(select(DataSource.id).where(DataSource.slug == normalized_slug))
            ).scalar_one_or_none()
            if existing is not None:
                raise ValueError(f"Data source slug '{normalized_slug}' already exists")

            if enabled:
                try:
                    data_source_loader.load(
                        slug=normalized_slug,
                        source_code=source_code,
                        config=dict(config or {}),
                        class_name=resolved_class_name,
                    )
                    status = "loaded"
                except DataSourceValidationError as exc:
                    status = "error"
                    error_message = str(exc)

            row = DataSource(
                id=uuid.uuid4().hex,
                slug=normalized_slug,
                source_key=normalized_source_key,
                source_kind=normalized_source_kind,
                name=resolved_name,
                description=resolved_description,
                source_code=source_code,
                class_name=resolved_class_name,
                is_system=bool(is_system),
                enabled=bool(enabled),
                status=status,
                error_message=error_message,
                config=dict(config or {}),
                config_schema=dict(config_schema or {}),
                version=1,
                sort_order=int(sort_order),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)

        runtime = data_source_loader.get_runtime(normalized_slug)
        return _serialize_source(row, runtime, include_code=True)

    @staticmethod
    async def update_source(
        source_slug: str,
        *,
        slug: str | None = None,
        source_key: str | None = None,
        source_kind: str | None = None,
        name: str | None = None,
        description: str | None = None,
        source_code: str | None = None,
        class_name: str | None = None,
        config: dict[str, Any] | None = None,
        config_schema: dict[str, Any] | None = None,
        enabled: bool | None = None,
        unlock_system: bool = False,
    ) -> dict[str, Any]:
        normalized_source_slug = str(source_slug or "").strip().lower()
        if not normalized_source_slug:
            raise ValueError("source_slug is required")

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(select(DataSource).where(DataSource.slug == normalized_source_slug))
            ).scalar_one_or_none()
            if row is None:
                raise ValueError(f"Data source '{normalized_source_slug}' not found")
            if bool(row.is_system) and not unlock_system:
                raise PermissionError("System data sources are read-only unless unlock_system=True")

            original_slug = str(row.slug or "").strip().lower()
            slug_changed = False
            runtime_reload_required = False

            if slug is not None:
                next_slug = _normalize_slug(slug)
                if next_slug != original_slug:
                    slug_in_use = (
                        await session.execute(
                            select(DataSource.id).where(DataSource.slug == next_slug, DataSource.id != row.id)
                        )
                    ).scalar_one_or_none()
                    if slug_in_use is not None:
                        raise ValueError(f"Data source slug '{next_slug}' already exists")
                    row.slug = next_slug
                    slug_changed = True

            if source_code is not None and source_code != row.source_code:
                validation = validate_data_source_source(source_code, class_name=class_name)
                if not validation["valid"]:
                    raise ValueError("; ".join(validation.get("errors") or ["Data source validation failed"]))
                row.source_code = source_code
                row.class_name = validation.get("class_name")
                if name is None and validation.get("source_name"):
                    row.name = str(validation.get("source_name"))
                if description is None and validation.get("source_description"):
                    row.description = str(validation.get("source_description"))
                row.version = int(row.version or 1) + 1
                runtime_reload_required = True
            elif class_name is not None:
                row.class_name = class_name
                runtime_reload_required = True

            if source_key is not None:
                row.source_key = _normalize_key(source_key, "custom")
            if source_kind is not None:
                row.source_kind = _normalize_key(source_kind, "python")
            if name is not None:
                row.name = str(name)
            if description is not None:
                row.description = description
            if config is not None:
                row.config = dict(config)
                runtime_reload_required = True
            if config_schema is not None:
                row.config_schema = dict(config_schema)

            enabled_changed = False
            if enabled is not None and bool(enabled) != bool(row.enabled):
                row.enabled = bool(enabled)
                enabled_changed = True

            if slug_changed:
                data_source_loader.unload(original_slug)

            if enabled_changed or runtime_reload_required or slug_changed:
                active_slug = str(row.slug or "").strip().lower()
                if bool(row.enabled):
                    try:
                        runtime = data_source_loader.load(
                            slug=active_slug,
                            source_code=str(row.source_code or ""),
                            config=dict(row.config or {}),
                            class_name=row.class_name,
                        )
                        row.class_name = runtime.class_name
                        row.status = "loaded"
                        row.error_message = None
                    except DataSourceValidationError as exc:
                        row.status = "error"
                        row.error_message = str(exc)
                else:
                    data_source_loader.unload(active_slug)
                    row.status = "unloaded"
                    row.error_message = None

            await session.commit()
            await session.refresh(row)

        runtime = data_source_loader.get_runtime(str(row.slug or "").strip().lower())
        return _serialize_source(row, runtime, include_code=True)

    @staticmethod
    async def delete_source(
        source_slug: str,
        *,
        tombstone_system_source: bool = True,
        unlock_system: bool = False,
        reason: str = "deleted_via_sdk",
    ) -> dict[str, Any]:
        normalized_source_slug = str(source_slug or "").strip().lower()
        if not normalized_source_slug:
            raise ValueError("source_slug is required")

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(select(DataSource).where(DataSource.slug == normalized_source_slug))
            ).scalar_one_or_none()
            if row is None:
                raise ValueError(f"Data source '{normalized_source_slug}' not found")
            if bool(row.is_system) and not unlock_system:
                raise PermissionError("System data sources require unlock_system=True for deletion")

            if bool(row.is_system) and tombstone_system_source:
                existing_tombstone = await session.get(DataSourceTombstone, normalized_source_slug)
                if existing_tombstone is None:
                    session.add(
                        DataSourceTombstone(
                            slug=normalized_source_slug,
                            reason=str(reason or "").strip() or "deleted_via_sdk",
                        )
                    )

            data_source_loader.unload(normalized_source_slug)
            await session.delete(row)
            await session.commit()

        return {"status": "deleted", "slug": normalized_source_slug}

    @staticmethod
    async def reload_source(source_slug: str) -> dict[str, Any]:
        slug = str(source_slug or "").strip().lower()
        if not slug:
            raise ValueError("source_slug is required")

        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(select(DataSource).where(DataSource.slug == slug))
            ).scalar_one_or_none()
            if row is None:
                raise ValueError(f"Data source '{slug}' not found")

            if not bool(row.enabled):
                data_source_loader.unload(slug)
                row.status = "unloaded"
                row.error_message = None
                await session.commit()
                return {"status": "unloaded", "message": "Source is disabled", "runtime": None}

            try:
                runtime = data_source_loader.load(
                    slug=slug,
                    source_code=str(row.source_code or ""),
                    config=dict(row.config or {}),
                    class_name=row.class_name,
                )
                row.status = "loaded"
                row.error_message = None
                row.class_name = runtime.class_name
                await session.commit()
                return {
                    "status": "loaded",
                    "message": f"Reloaded {slug}",
                    "runtime": {
                        "slug": runtime.slug,
                        "class_name": runtime.class_name,
                        "name": runtime.name,
                        "description": runtime.description,
                        "loaded_at": runtime.loaded_at.isoformat(),
                        "source_hash": runtime.source_hash,
                        "run_count": runtime.run_count,
                        "error_count": runtime.error_count,
                        "last_run": runtime.last_run.isoformat() if runtime.last_run else None,
                        "last_error": runtime.last_error,
                    },
                }
            except DataSourceValidationError as exc:
                row.status = "error"
                row.error_message = str(exc)
                await session.commit()
                raise ValueError(str(exc)) from exc

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
