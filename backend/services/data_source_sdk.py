"""Data Source SDK — stable helpers for source authors and strategy developers."""

from __future__ import annotations

import asyncio
import hashlib
import math
from datetime import datetime, timezone
import re
from time import mktime
from typing import Any
import uuid

import feedparser
import httpx
from sqlalchemy import Select, desc, select

from models.database import AsyncSessionLocal, DataSource, DataSourceRecord, DataSourceRun, DataSourceTombstone

from services.data_source_catalog import build_data_source_source_code, default_data_source_retention_policy
from services.data_source_loader import (
    DataSourceValidationError,
    data_source_loader,
    validate_data_source_source,
)

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}[a-z0-9]$")
_SOURCE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_ALLOWED_SOURCE_KINDS = {"python", "rss", "rest_api"}
_TEMPLATED_SOURCE_KINDS = {"rss", "rest_api"}
_RETENTION_BOUNDS: dict[str, tuple[int, int]] = {
    "max_records": (1, 250_000),
    "max_age_days": (1, 3650),
}


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
        "retention": _resolve_retention_policy(
            row.retention,
            slug=row.slug,
            source_key=row.source_key,
            source_kind=row.source_kind,
            strict=False,
        ),
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


def _normalize_source_key(value: str, default: str = "custom") -> str:
    out = _normalize_key(value, default)
    if not _SOURCE_KEY_RE.match(out):
        raise ValueError(
            "Invalid source_key. Must start with a letter, use lowercase letters/numbers/underscores, and be 2-64 chars."
        )
    return out


def _normalize_source_kind(value: str, default: str = "python") -> str:
    out = _normalize_key(value, default)
    if out not in _ALLOWED_SOURCE_KINDS:
        allowed = ", ".join(sorted(_ALLOWED_SOURCE_KINDS))
        raise ValueError(f"Unsupported source_kind '{out}'. Allowed values: {allowed}")
    return out


def _normalize_source_code_text(value: str | None) -> str:
    text = str(value or "")
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return text


def _normalize_retention_policy(retention: Any, *, strict: bool = True) -> dict[str, int]:
    if retention is None:
        return {}
    if not isinstance(retention, dict):
        if strict:
            raise ValueError("retention must be an object")
        return {}

    normalized: dict[str, int] = {}
    for key, bounds in _RETENTION_BOUNDS.items():
        raw_value = retention.get(key)
        if raw_value in (None, ""):
            continue
        low, high = bounds
        try:
            parsed = int(float(raw_value))
        except (TypeError, ValueError):
            if strict:
                raise ValueError(f"retention.{key} must be an integer between {low} and {high}") from None
            continue
        if parsed < low or parsed > high:
            if strict:
                raise ValueError(f"retention.{key} must be between {low} and {high}")
            continue
        normalized[key] = parsed
    return normalized


def _resolve_retention_policy(
    retention: Any,
    *,
    slug: str | None,
    source_key: str | None,
    source_kind: str | None,
    strict: bool = True,
) -> dict[str, int]:
    defaults = default_data_source_retention_policy(
        slug=slug,
        source_key=source_key,
        source_kind=source_kind,
    )
    normalized = _normalize_retention_policy(retention, strict=strict)
    if not normalized:
        return defaults
    out = dict(defaults)
    out.update(normalized)
    return out


def _resolve_source_code_for_kind(
    *,
    source_kind: str,
    slug: str,
    name: str,
    description: str | None,
    source_code: str | None,
    config: dict[str, Any] | None,
) -> str:
    raw_code = _normalize_source_code_text(source_code)
    if raw_code.strip():
        return raw_code

    normalized_kind = _normalize_source_kind(source_kind, "python")
    if normalized_kind in _TEMPLATED_SOURCE_KINDS:
        return build_data_source_source_code(
            source_kind=normalized_kind,
            slug=slug,
            name=name,
            description=description or "",
            config=dict(config or {}),
            include_seed_marker=False,
        )

    raise ValueError("Python data sources require source_code with at least 10 characters")


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

    def _as_int(self, value: Any, default: int, low: int, high: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = int(default)
        return max(int(low), min(int(high), int(parsed)))

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        if out != out:
            return float(default)
        return out

    def _clamp(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(float(low), min(float(high), float(value)))

    def _hours(self) -> int:
        default = int(self.default_config.get("hours", 72) or 72)
        return self._as_int(self.config.get("hours"), default, 1, 87600)

    def _limit(self) -> int:
        default = int(self.default_config.get("limit", 500) or 500)
        return self._as_int(self.config.get("limit"), default, 1, 5000)

    def _min_severity(self) -> float:
        default = float(self.default_config.get("min_severity", 0.0) or 0.0)
        return self._clamp(self._as_float(self.config.get("min_severity"), default), 0.0, 1.0)

    def _source_names(self) -> set[str]:
        return {str(item).strip().lower() for item in (self.config.get("source_names") or []) if str(item).strip()}

    def _signal_types(self) -> set[str]:
        return {str(item).strip().lower() for item in (self.config.get("signal_types") or []) if str(item).strip()}

    def _keep(self, source: Any, category: Any, severity: Any) -> bool:
        source_name = str(source or "").strip().lower()
        category_name = str(category or "").strip().lower()
        severity_value = self._clamp(self._as_float(severity, 0.0), 0.0, 1.0)
        source_names = self._source_names()
        signal_types = self._signal_types()
        if source_names and source_name not in source_names:
            return False
        if signal_types and category_name not in signal_types:
            return False
        if severity_value < self._min_severity():
            return False
        return True

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, tuple) and len(value) >= 6:
            try:
                return datetime.fromtimestamp(mktime(value), tz=timezone.utc)
            except Exception:
                return None
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            try:
                ts = int(text)
                if ts > 10_000_000_000:
                    ts = ts / 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc)
            return parsed.replace(tzinfo=timezone.utc)
        except Exception:
            pass
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
        return None

    def _extract_json_path(self, data: Any, path: str) -> list[dict[str, Any]]:
        parts = str(path or "$[*]").strip().lstrip("$").split(".")
        current: Any = data
        for part in parts:
            if not part:
                continue
            if part.endswith("[*]"):
                key = part[:-3]
                if key:
                    if isinstance(current, dict):
                        current = current.get(key, [])
                    else:
                        return []
                if not isinstance(current, list):
                    return []
                continue
            if isinstance(current, dict):
                current = current.get(part)
                continue
            if isinstance(current, list):
                next_rows: list[Any] = []
                for item in current:
                    if isinstance(item, dict) and part in item:
                        next_rows.append(item.get(part))
                current = next_rows
                continue
            return []
        if isinstance(current, list):
            return [item for item in current if isinstance(item, dict)]
        if isinstance(current, dict):
            return [current]
        return []

    def _haversine_km(self, lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float:
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        d_phi = math.radians(float(lat2) - float(lat1))
        d_lambda = math.radians(float(lon2) - float(lon1))
        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lambda / 2.0) ** 2)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return 6371.0 * c

    def _severity_from_row(self, row: dict[str, Any], default: float = 0.5) -> float:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        transformed = row.get("transformed") if isinstance(row.get("transformed"), dict) else {}
        raw = transformed.get("severity", payload.get("severity", default))
        return self._clamp(self._as_float(raw, default), 0.0, 1.0)

    def _stable_hash(self, *parts: Any, length: int = 24) -> str:
        text = "|".join(str(part or "").strip() for part in parts)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[: max(8, int(length))]

    async def _http_get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float = 30.0,
        default: Any = None,
    ) -> Any:
        target = str(url or "").strip()
        if not target:
            return default
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(target, params=params, headers=headers)
                if response.status_code == 429:
                    return default
                response.raise_for_status()
                try:
                    return response.json()
                except Exception:
                    return default
        except Exception:
            return default

    async def _http_get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float = 30.0,
        default: str = "",
    ) -> str:
        target = str(url or "").strip()
        if not target:
            return default
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(target, params=params, headers=headers)
                if response.status_code == 429:
                    return default
                response.raise_for_status()
                return response.text
        except Exception:
            return default

    async def _fetch_rss(self, url: str, *, timeout: float = 30.0) -> tuple[str, list[dict[str, Any]]]:
        text = await self._http_get_text(url, timeout=timeout, default="")
        if not text:
            return "", []
        parsed = feedparser.parse(text)
        feed = parsed.get("feed") if isinstance(parsed, dict) else {}
        entries = parsed.get("entries") if isinstance(parsed, dict) else []
        feed_title = str((feed or {}).get("title") or "").strip()
        if not isinstance(entries, list):
            entries = []
        return feed_title, [entry for entry in entries if isinstance(entry, dict)]

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
            row = (await session.execute(select(DataSource).where(DataSource.slug == slug))).scalar_one_or_none()
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
        retention: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        config_schema: dict[str, Any] | None = None,
        enabled: bool = True,
        is_system: bool = False,
        sort_order: int = 0,
    ) -> dict[str, Any]:
        normalized_slug = _normalize_slug(slug)
        normalized_source_key = _normalize_source_key(source_key, "custom")
        normalized_source_kind = _normalize_source_kind(source_kind, "python")
        resolved_name = str(name or normalized_slug.replace("_", " ").title()).strip()
        if not resolved_name:
            raise ValueError("name is required")
        resolved_source_code = _resolve_source_code_for_kind(
            source_kind=normalized_source_kind,
            slug=normalized_slug,
            name=resolved_name,
            description=description,
            source_code=source_code,
            config=config,
        )
        validation = validate_data_source_source(resolved_source_code, class_name=class_name)
        if not validation["valid"]:
            raise ValueError("; ".join(validation.get("errors") or ["Data source validation failed"]))
        resolved_class_name = validation.get("class_name")
        if name is None and validation.get("source_name"):
            resolved_name = str(validation.get("source_name"))
        resolved_description = description if description is not None else validation.get("source_description")

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
                        source_code=resolved_source_code,
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
                source_code=resolved_source_code,
                class_name=resolved_class_name,
                is_system=bool(is_system),
                enabled=bool(enabled),
                status=status,
                error_message=error_message,
                retention=_resolve_retention_policy(
                    retention,
                    slug=normalized_slug,
                    source_key=normalized_source_key,
                    source_kind=normalized_source_kind,
                ),
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
        retention: dict[str, Any] | None = None,
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
            source_kind_changed = False

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

            if source_key is not None:
                row.source_key = _normalize_source_key(source_key, "custom")
            if source_kind is not None:
                normalized_kind = _normalize_source_kind(source_kind, "python")
                current_kind = _normalize_source_kind(str(row.source_kind or ""), "python")
                source_kind_changed = normalized_kind != current_kind
                row.source_kind = normalized_kind
            if name is not None:
                row.name = str(name)
            if description is not None:
                row.description = description
            if config is not None:
                row.config = dict(config)
                runtime_reload_required = True
            if config_schema is not None:
                row.config_schema = dict(config_schema)

            effective_kind = _normalize_source_kind(str(row.source_kind or ""), "python")
            if str(row.source_kind or "").strip().lower() != effective_kind:
                row.source_kind = effective_kind
            row.retention = _resolve_retention_policy(
                retention if retention is not None else row.retention,
                slug=str(row.slug or "").strip().lower(),
                source_key=str(row.source_key or "").strip().lower(),
                source_kind=effective_kind,
                strict=retention is not None,
            )

            name_for_template = str(row.name or "").strip() or str(row.slug or "").replace("_", " ").title()
            source_code_input = source_code
            if source_code_input is None:
                if source_kind_changed and effective_kind in _TEMPLATED_SOURCE_KINDS:
                    source_code_input = ""
                else:
                    source_code_input = str(row.source_code or "")

            resolved_source_code = _resolve_source_code_for_kind(
                source_kind=effective_kind,
                slug=str(row.slug or "").strip().lower(),
                name=name_for_template,
                description=row.description,
                source_code=source_code_input,
                config=dict(row.config or {}),
            )

            should_validate = (
                source_code is not None
                or source_kind_changed
                or (effective_kind in _TEMPLATED_SOURCE_KINDS and not str(row.source_code or "").strip())
            )
            if should_validate:
                validation = validate_data_source_source(resolved_source_code, class_name=class_name)
                if not validation["valid"]:
                    raise ValueError("; ".join(validation.get("errors") or ["Data source validation failed"]))
                if str(row.source_code or "") != resolved_source_code:
                    row.source_code = resolved_source_code
                    row.version = int(row.version or 1) + 1
                    runtime_reload_required = True
                row.class_name = validation.get("class_name")
                if name is None and validation.get("source_name"):
                    row.name = str(validation.get("source_name"))
                if description is None and validation.get("source_description"):
                    row.description = str(validation.get("source_description"))
                if source_kind_changed:
                    runtime_reload_required = True
            elif class_name is not None:
                row.class_name = class_name
                runtime_reload_required = True

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
            row = (await session.execute(select(DataSource).where(DataSource.slug == slug))).scalar_one_or_none()
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


__all__ = [
    "BaseDataSource",
    "DataSourceSDK",
    "_utcnow_naive",
]
