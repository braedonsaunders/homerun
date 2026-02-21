"""Execution runner for DB-defined data sources."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, desc, func, select
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import DataSource, DataSourceRecord, DataSourceRun
from services.data_source_catalog import default_data_source_retention_policy
from services.data_source_loader import DataSourceValidationError, data_source_loader
from utils.logger import get_logger

logger = get_logger(__name__)

_RETENTION_BOUNDS: dict[str, tuple[int, int]] = {
    "max_records": (1, 250_000),
    "max_age_days": (1, 3650),
}
_DB_DISCONNECT_RETRY_BASE_DELAY_SECONDS = 0.2


def _is_retryable_db_disconnect_error(exc: Exception) -> bool:
    if not isinstance(exc, (OperationalError, InterfaceError)):
        return False
    message = str(getattr(exc, "orig", exc)).lower()
    return any(
        marker in message
        for marker in (
            "connection is closed",
            "underlying connection is closed",
            "connection has been closed",
            "closed the connection unexpectedly",
            "terminating connection",
            "connection reset by peer",
            "broken pipe",
        )
    )


def _db_disconnect_retry_delay(attempt: int) -> float:
    return min(_DB_DISCONNECT_RETRY_BASE_DELAY_SECONDS * (2**attempt), 1.5)


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).isoformat()
        return value.replace(tzinfo=timezone.utc).isoformat()
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            out[str(key)] = _json_safe(nested)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _normalize_country_iso3(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if len(text) == 3 and text.isalpha():
        return text
    return None


def _normalize_retention_policy(retention: Any) -> dict[str, int]:
    if not isinstance(retention, dict):
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
            continue
        if parsed < low or parsed > high:
            continue
        normalized[key] = parsed
    return normalized


def _resolve_retention_policy(
    retention: Any,
    *,
    slug: str | None,
    source_key: str | None,
    source_kind: str | None,
) -> dict[str, int]:
    defaults = default_data_source_retention_policy(
        slug=slug,
        source_key=source_key,
        source_kind=source_kind,
    )
    normalized = _normalize_retention_policy(retention)
    if not normalized:
        return defaults
    out = dict(defaults)
    out.update(normalized)
    return out


async def _apply_retention_policy(
    session: AsyncSession,
    source: DataSource,
    *,
    retention_policy: dict[str, int],
) -> dict[str, int]:
    source_id = str(source.id or "").strip()
    if not source_id or not retention_policy:
        return {"max_age_deleted": 0, "max_records_deleted": 0}

    max_age_deleted = 0
    max_records_deleted = 0

    max_age_days = retention_policy.get("max_age_days")
    if max_age_days is not None:
        cutoff = _utcnow_naive() - timedelta(days=int(max_age_days))
        stale_ids = (
            (
                await session.execute(
                    select(DataSourceRecord.id)
                    .where(DataSourceRecord.data_source_id == source_id)
                    .where(func.coalesce(DataSourceRecord.observed_at, DataSourceRecord.ingested_at) < cutoff)
                )
            )
            .scalars()
            .all()
        )
        if stale_ids:
            delete_result = await session.execute(
                delete(DataSourceRecord).where(DataSourceRecord.id.in_(list(stale_ids)))
            )
            deleted = int(delete_result.rowcount or 0)
            max_age_deleted = deleted if deleted > 0 else len(stale_ids)

    max_records = retention_policy.get("max_records")
    if max_records is not None:
        overflow_ids = (
            (
                await session.execute(
                    select(DataSourceRecord.id)
                    .where(DataSourceRecord.data_source_id == source_id)
                    .order_by(
                        desc(DataSourceRecord.observed_at),
                        desc(DataSourceRecord.ingested_at),
                        desc(DataSourceRecord.id),
                    )
                    .offset(int(max_records))
                )
            )
            .scalars()
            .all()
        )
        if overflow_ids:
            delete_result = await session.execute(
                delete(DataSourceRecord).where(DataSourceRecord.id.in_(list(overflow_ids)))
            )
            deleted = int(delete_result.rowcount or 0)
            max_records_deleted = deleted if deleted > 0 else len(overflow_ids)

    return {
        "max_age_deleted": int(max_age_deleted),
        "max_records_deleted": int(max_records_deleted),
    }


async def _invoke_callable(func: Any, *args: Any, **kwargs: Any) -> Any:
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _normalize_record(raw: dict[str, Any], transformed: dict[str, Any]) -> dict[str, Any]:
    payload_raw = dict(raw)
    final_raw = dict(transformed)

    external_id = _as_text(final_raw.get("external_id")) or _as_text(payload_raw.get("external_id"))
    title = _as_text(final_raw.get("title")) or _as_text(payload_raw.get("title"))
    summary = _as_text(final_raw.get("summary")) or _as_text(payload_raw.get("summary"))
    category_raw = _as_text(final_raw.get("category")) or _as_text(payload_raw.get("category"))
    category = category_raw.lower() if category_raw else None
    source = _as_text(final_raw.get("source")) or _as_text(payload_raw.get("source"))
    url = _as_text(final_raw.get("url")) or _as_text(payload_raw.get("url"))

    latitude = _as_float(final_raw.get("latitude"))
    if latitude is None:
        latitude = _as_float(payload_raw.get("latitude"))
    longitude = _as_float(final_raw.get("longitude"))
    if longitude is None:
        longitude = _as_float(payload_raw.get("longitude"))

    country_iso3 = _normalize_country_iso3(final_raw.get("country_iso3"))
    if country_iso3 is None:
        country_iso3 = _normalize_country_iso3(payload_raw.get("country_iso3"))

    geotagged = _as_bool(final_raw.get("geotagged"), default=False)
    if not geotagged:
        geotagged = latitude is not None and longitude is not None

    observed_at = _parse_datetime(final_raw.get("observed_at"))
    if observed_at is None:
        observed_at = _parse_datetime(payload_raw.get("observed_at"))

    tags = _as_tags(final_raw.get("tags"))
    if not tags:
        tags = _as_tags(payload_raw.get("tags"))

    payload = _json_safe(payload_raw)
    if not isinstance(payload, dict):
        payload = {}
    final = _json_safe(final_raw)
    if not isinstance(final, dict):
        final = {}

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
    _retry_on_disconnect: bool = True,
) -> dict[str, Any]:
    """Run one source, normalize records, and upsert into data_source_records."""

    source_slug = str(source.slug or "").strip().lower()
    source_id = str(source.id or "")
    if not source_slug:
        raise ValueError("source.slug is required")
    if not bool(source.enabled):
        raise ValueError(f"Source '{source_slug}' is disabled")

    safe_max_records = max(1, min(5000, int(max_records)))
    started_at = _utcnow_naive()
    runtime = data_source_loader.get_runtime(source_slug)

    run_row = DataSourceRun(
        id=uuid.uuid4().hex,
        data_source_id=source_id,
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
    retention_policy = _resolve_retention_policy(
        source.retention,
        slug=source_slug,
        source_key=source.source_key,
        source_kind=source.source_kind,
    )
    source.retention = dict(retention_policy)
    retention_pruned = {"max_age_deleted": 0, "max_records_deleted": 0}

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

        external_ids = sorted({record["external_id"] for record in normalized_rows if record.get("external_id")})

        existing_by_external_id: dict[str, DataSourceRecord] = {}
        if external_ids:
            with session.no_autoflush:
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
            existing_by_external_id = {str(row.external_id): row for row in existing_rows if row.external_id}

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

        retention_pruned = await _apply_retention_policy(
            session,
            source,
            retention_policy=retention_policy,
        )

        if runtime is not None:
            runtime.run_count += 1
            runtime.last_run = datetime.now(timezone.utc)
            runtime.last_error = None

        source.status = "loaded"
        source.error_message = None
        run_row.status = "success"
        run_row.error_message = None
    except Exception as exc:
        if _retry_on_disconnect and _is_retryable_db_disconnect_error(exc):
            try:
                await session.rollback()
            except Exception:
                pass
            logger.warning(
                "Data source run hit transient DB disconnect; retrying once",
                source_slug=source_slug,
                error=str(exc),
            )
            await asyncio.sleep(_db_disconnect_retry_delay(0))
            retry_source = source
            try:
                reloaded_source = await session.get(DataSource, source_id)
                if reloaded_source is not None:
                    retry_source = reloaded_source
            except Exception:
                pass
            return await run_data_source(
                session,
                retry_source,
                max_records=max_records,
                commit=commit,
                _retry_on_disconnect=False,
            )

        if runtime is not None:
            runtime.error_count += 1
            runtime.last_error = str(exc)
        error_message = str(exc)
        logger.error("Data source run failed", source_slug=source_slug, exc_info=exc)

        # The session transaction may be tainted (e.g. autoflush hit
        # a transient transaction error). Roll back so we can persist the error
        # run row cleanly.
        try:
            await session.rollback()
        except Exception:
            pass

        # Re-attach the source and run_row to the fresh transaction so the
        # error state is persisted.
        try:
            source = await session.get(DataSource, source_id)
            if source is not None:
                source.status = "error"
                source.error_message = error_message
        except Exception:
            pass

        run_row = DataSourceRun(
            id=uuid.uuid4().hex,
            data_source_id=source_id,
            source_slug=source_slug,
            status="error",
            fetched_count=fetched_count,
            transformed_count=transformed_count,
            upserted_count=upserted_count,
            skipped_count=skipped_count,
            error_message=error_message,
            metadata_json={},
            started_at=started_at,
            completed_at=None,
            duration_ms=None,
        )
        session.add(run_row)

    completed_at = _utcnow_naive()
    run_row.fetched_count = fetched_count
    run_row.transformed_count = transformed_count
    run_row.upserted_count = upserted_count
    run_row.skipped_count = skipped_count
    run_row.metadata_json = {
        "max_records": safe_max_records,
        "retention": dict(retention_policy),
        "retention_pruned": {
            "max_age_deleted": int(retention_pruned.get("max_age_deleted") or 0),
            "max_records_deleted": int(retention_pruned.get("max_records_deleted") or 0),
            "total_deleted": int(retention_pruned.get("max_age_deleted") or 0)
            + int(retention_pruned.get("max_records_deleted") or 0),
        },
    }
    run_row.completed_at = completed_at
    run_row.duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    if commit:
        await session.commit()
    else:
        await session.flush()

    # Dispatch a DataEvent so subscribed strategies can react when a data
    # source produces new records — bridging the DataSource SDK into the
    # real-time event pipeline.
    if run_row.status == "success" and upserted_count > 0:
        try:
            from services.data_events import DataEvent, EventType
            from services.event_dispatcher import event_dispatcher

            event = DataEvent(
                event_type=EventType.DATA_SOURCE_UPDATE,
                source=f"data_source:{source_slug}",
                timestamp=datetime.now(timezone.utc),
                payload={
                    "source_slug": source_slug,
                    "records_count": upserted_count,
                    "run_id": run_row.id,
                },
            )
            await event_dispatcher.dispatch(event)
        except Exception as exc:
            logger.warning(
                "DataEvent dispatch after data source run failed (non-critical)",
                source_slug=source_slug,
                exc_info=exc,
            )

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
        "retention_pruned_count": int(retention_pruned.get("max_age_deleted") or 0)
        + int(retention_pruned.get("max_records_deleted") or 0),
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


async def preview_data_source(
    source: DataSource,
    *,
    max_records: int = 25,
) -> dict[str, Any]:
    source_slug = str(source.slug or "").strip().lower()
    if not source_slug:
        raise ValueError("source.slug is required")

    safe_max = max(1, min(200, int(max_records)))
    started_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    runtime = data_source_loader.get_runtime(source_slug)
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

    total_fetched = len(fetched_rows)
    fetched_rows = fetched_rows[:safe_max]

    normalized_rows: list[dict[str, Any]] = []
    for raw_item in fetched_rows:
        if not isinstance(raw_item, dict):
            continue
        raw_payload = dict(raw_item)
        transformed_payload = dict(raw_payload)
        if hasattr(instance, "transform"):
            transformed = await _invoke_callable(instance.transform, dict(raw_payload))
            if isinstance(transformed, dict):
                transformed_payload = transformed
            else:
                continue
        normalized = _normalize_record(raw_payload, transformed_payload)
        normalized_rows.append(normalized)

    elapsed_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - started_ms

    return {
        "source_id": source.id,
        "source_slug": source_slug,
        "total_fetched": total_fetched,
        "duration_ms": max(0, elapsed_ms),
        "records": [
            {
                "external_id": row.get("external_id"),
                "title": row.get("title"),
                "summary": row.get("summary"),
                "category": row.get("category"),
                "source": row.get("source"),
                "url": row.get("url"),
                "geotagged": bool(row.get("geotagged")),
                "country_iso3": row.get("country_iso3"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "observed_at": row["observed_at"].isoformat() if row.get("observed_at") else None,
                "payload": row.get("payload_json"),
                "transformed": row.get("transformed_json"),
                "tags": row.get("tags_json"),
            }
            for row in normalized_rows
        ],
    }


__all__ = [
    "run_data_source",
    "run_data_source_by_slug",
    "preview_data_source",
]
