"""Execution runner for DB-defined data sources."""

from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import DataSource, DataSourceRecord, DataSourceRun
from services.data_source_loader import DataSourceValidationError, data_source_loader
from utils.logger import get_logger

logger = get_logger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _as_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            item = str(raw or "").strip()
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    return []


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


def _normalize_country_iso3(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if len(text) == 3 and text.isalpha():
        return text
    return None


async def _invoke_callable(func: Any, *args: Any, **kwargs: Any) -> Any:
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _normalize_record(raw: dict[str, Any], transformed: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    final = dict(transformed)

    external_id = _as_text(final.get("external_id")) or _as_text(payload.get("external_id"))
    title = _as_text(final.get("title")) or _as_text(payload.get("title"))
    summary = _as_text(final.get("summary")) or _as_text(payload.get("summary"))
    category_raw = _as_text(final.get("category")) or _as_text(payload.get("category"))
    category = category_raw.lower() if category_raw else None
    source = _as_text(final.get("source")) or _as_text(payload.get("source"))
    url = _as_text(final.get("url")) or _as_text(payload.get("url"))

    latitude = _as_float(final.get("latitude"))
    if latitude is None:
        latitude = _as_float(payload.get("latitude"))
    longitude = _as_float(final.get("longitude"))
    if longitude is None:
        longitude = _as_float(payload.get("longitude"))

    country_iso3 = _normalize_country_iso3(final.get("country_iso3"))
    if country_iso3 is None:
        country_iso3 = _normalize_country_iso3(payload.get("country_iso3"))

    geotagged = _as_bool(final.get("geotagged"), default=False)
    if not geotagged:
        geotagged = latitude is not None and longitude is not None

    observed_at = _parse_datetime(final.get("observed_at"))
    if observed_at is None:
        observed_at = _parse_datetime(payload.get("observed_at"))

    tags = _as_tags(final.get("tags"))
    if not tags:
        tags = _as_tags(payload.get("tags"))

    return {
        "external_id": external_id,
        "title": title,
        "summary": summary,
        "category": category,
        "source": source,
        "url": url,
        "geotagged": geotagged,
        "country_iso3": country_iso3,
        "latitude": latitude,
        "longitude": longitude,
        "observed_at": observed_at,
        "payload_json": payload,
        "transformed_json": final,
        "tags_json": tags,
    }


async def run_data_source(
    session: AsyncSession,
    source: DataSource,
    *,
    max_records: int = 500,
    commit: bool = True,
) -> dict[str, Any]:
    """Run one source, normalize records, and upsert into data_source_records."""

    source_slug = str(source.slug or "").strip().lower()
    if not source_slug:
        raise ValueError("source.slug is required")
    if not bool(source.enabled):
        raise ValueError(f"Source '{source_slug}' is disabled")

    safe_max_records = max(1, min(5000, int(max_records)))
    started_at = _utcnow_naive()
    runtime = data_source_loader.get_runtime(source_slug)

    run_row = DataSourceRun(
        id=uuid.uuid4().hex,
        data_source_id=source.id,
        source_slug=source_slug,
        status="success",
        fetched_count=0,
        transformed_count=0,
        upserted_count=0,
        skipped_count=0,
        error_message=None,
        metadata_json={},
        started_at=started_at,
        completed_at=None,
        duration_ms=None,
    )
    session.add(run_row)

    fetched_count = 0
    transformed_count = 0
    upserted_count = 0
    skipped_count = 0

    try:
        if runtime is None:
            runtime = data_source_loader.load(
                slug=source_slug,
                source_code=str(source.source_code or ""),
                config=dict(source.config or {}),
                class_name=source.class_name,
            )

        instance = runtime.instance
        if hasattr(instance, "fetch_async"):
            fetched = await _invoke_callable(instance.fetch_async)
        elif hasattr(instance, "fetch"):
            fetched = await _invoke_callable(instance.fetch)
        else:
            raise DataSourceValidationError("Source instance has no fetch/fetch_async method")

        if fetched is None:
            fetched_rows: list[Any] = []
        elif isinstance(fetched, list):
            fetched_rows = fetched
        else:
            raise DataSourceValidationError("fetch/fetch_async must return a list of dict records")

        fetched_rows = fetched_rows[:safe_max_records]
        fetched_count = len(fetched_rows)

        normalized_rows: list[dict[str, Any]] = []
        for raw_item in fetched_rows:
            if not isinstance(raw_item, dict):
                skipped_count += 1
                continue

            raw_payload = dict(raw_item)
            transformed_payload = dict(raw_payload)
            if hasattr(instance, "transform"):
                transformed = await _invoke_callable(instance.transform, dict(raw_payload))
                if isinstance(transformed, dict):
                    transformed_payload = transformed
                else:
                    skipped_count += 1
                    continue

            normalized = _normalize_record(raw_payload, transformed_payload)
            normalized_rows.append(normalized)

        transformed_count = len(normalized_rows)

        external_ids = sorted(
            {
                record["external_id"]
                for record in normalized_rows
                if record.get("external_id")
            }
        )

        existing_by_external_id: dict[str, DataSourceRecord] = {}
        if external_ids:
            existing_rows = (
                (
                    await session.execute(
                        select(DataSourceRecord)
                        .where(DataSourceRecord.source_slug == source_slug)
                        .where(DataSourceRecord.external_id.in_(external_ids))
                    )
                )
                .scalars()
                .all()
            )
            existing_by_external_id = {
                str(row.external_id): row
                for row in existing_rows
                if row.external_id
            }

        ingested_at = _utcnow_naive()
        for normalized in normalized_rows:
            external_id = normalized.get("external_id")
            existing = existing_by_external_id.get(str(external_id)) if external_id else None

            if existing is not None:
                existing.title = normalized.get("title")
                existing.summary = normalized.get("summary")
                existing.category = normalized.get("category")
                existing.source = normalized.get("source")
                existing.url = normalized.get("url")
                existing.geotagged = bool(normalized.get("geotagged"))
                existing.country_iso3 = normalized.get("country_iso3")
                existing.latitude = normalized.get("latitude")
                existing.longitude = normalized.get("longitude")
                existing.observed_at = normalized.get("observed_at")
                existing.ingested_at = ingested_at
                existing.payload_json = normalized.get("payload_json")
                existing.transformed_json = normalized.get("transformed_json")
                existing.tags_json = normalized.get("tags_json")
            else:
                session.add(
                    DataSourceRecord(
                        id=uuid.uuid4().hex,
                        data_source_id=source.id,
                        source_slug=source_slug,
                        external_id=normalized.get("external_id"),
                        title=normalized.get("title"),
                        summary=normalized.get("summary"),
                        category=normalized.get("category"),
                        source=normalized.get("source"),
                        url=normalized.get("url"),
                        geotagged=bool(normalized.get("geotagged")),
                        country_iso3=normalized.get("country_iso3"),
                        latitude=normalized.get("latitude"),
                        longitude=normalized.get("longitude"),
                        observed_at=normalized.get("observed_at"),
                        ingested_at=ingested_at,
                        payload_json=normalized.get("payload_json"),
                        transformed_json=normalized.get("transformed_json"),
                        tags_json=normalized.get("tags_json"),
                    )
                )
            upserted_count += 1

        runtime.run_count += 1
        runtime.last_run = datetime.now(timezone.utc)
        runtime.last_error = None

        source.status = "loaded"
        source.error_message = None
        run_row.status = "success"
        run_row.error_message = None
    except Exception as exc:
        if runtime is not None:
            runtime.error_count += 1
            runtime.last_error = str(exc)
        source.status = "error"
        source.error_message = str(exc)
        run_row.status = "error"
        run_row.error_message = str(exc)
        logger.error("Data source run failed", source_slug=source_slug, exc_info=exc)

    completed_at = _utcnow_naive()
    run_row.fetched_count = fetched_count
    run_row.transformed_count = transformed_count
    run_row.upserted_count = upserted_count
    run_row.skipped_count = skipped_count
    run_row.metadata_json = {
        "max_records": safe_max_records,
    }
    run_row.completed_at = completed_at
    run_row.duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    if commit:
        await session.commit()
    else:
        await session.flush()

    return {
        "run_id": run_row.id,
        "source_slug": source_slug,
        "status": run_row.status,
        "fetched_count": fetched_count,
        "transformed_count": transformed_count,
        "upserted_count": upserted_count,
        "skipped_count": skipped_count,
        "error_message": run_row.error_message,
        "duration_ms": run_row.duration_ms,
    }


async def run_data_source_by_slug(
    session: AsyncSession,
    *,
    source_slug: str,
    max_records: int = 500,
    commit: bool = True,
) -> dict[str, Any]:
    slug = str(source_slug or "").strip().lower()
    if not slug:
        raise ValueError("source_slug is required")

    row = (await session.execute(select(DataSource).where(DataSource.slug == slug))).scalar_one_or_none()
    if row is None:
        raise ValueError(f"Data source '{slug}' not found")

    return await run_data_source(
        session,
        row,
        max_records=max_records,
        commit=commit,
    )


__all__ = [
    "run_data_source",
    "run_data_source_by_slug",
]
