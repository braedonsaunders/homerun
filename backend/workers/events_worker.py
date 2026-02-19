"""Events worker: executes DB-managed events data sources and writes snapshots.

Run from backend dir:
  python -m workers.events_worker
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select

from config import settings
from models.database import (
    AsyncSessionLocal,
    DataSource,
    DataSourceRecord,
    EventsSignal,
    EventsSnapshot,
)
from services.data_source_runner import run_data_source
from services.worker_state import (
    clear_worker_run_request,
    read_worker_control,
    read_worker_snapshot,
    write_worker_snapshot,
)
from utils.logger import setup_logging

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("events_worker")

_IDLE_SLEEP_SECONDS = 5


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_aware(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc)
        return value.replace(tzinfo=timezone.utc)
    return _utcnow()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_iso3(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if len(text) == 3 and text.isalpha():
        return text
    return None


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _source_status_key(source_slug: str) -> str:
    slug = str(source_slug or "").strip().lower()
    if slug.startswith("events_"):
        suffix = slug[len("events_") :].strip()
        if suffix:
            return suffix
    return slug or "events"


def _stable_signal_id(
    source_slug: str, external_id: str | None, url: str | None, title: str, observed_at: datetime
) -> str:
    key = (
        f"{source_slug}:{external_id}"
        if str(external_id or "").strip()
        else f"{source_slug}:{url or ''}:{title}:{observed_at.isoformat()}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"events_{digest}"


def _record_to_signal(record: DataSourceRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {}) if isinstance(record.payload_json, dict) else {}
    transformed = dict(record.transformed_json or {}) if isinstance(record.transformed_json, dict) else {}
    merged: dict[str, Any] = {}
    merged.update(payload)
    merged.update(transformed)

    signal_type = str(record.category or merged.get("signal_type") or "world").strip().lower() or "world"
    source = str(record.source or merged.get("source") or record.source_slug or "events").strip().lower() or "events"
    title = str(record.title or merged.get("title") or "Events signal").strip() or "Events signal"
    summary = str(record.summary or merged.get("summary") or "").strip()
    severity = _clamp(_as_float(merged.get("severity"), _as_float(payload.get("severity"), 0.0)), 0.0, 1.0)

    observed_at = _to_aware(record.observed_at or merged.get("observed_at") or record.ingested_at)
    country_iso3 = _as_iso3(record.country_iso3 or merged.get("country_iso3") or merged.get("country"))

    metadata = merged.get("metadata") if isinstance(merged.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata.setdefault("source_slug", str(record.source_slug or ""))
    metadata.setdefault("external_id", str(record.external_id or "") or None)
    if record.url:
        metadata.setdefault("url", record.url)

    related_market_ids = []
    for item in _as_list(merged.get("related_market_ids")):
        token = str(item or "").strip()
        if token:
            related_market_ids.append(token)

    market_relevance_score = merged.get("market_relevance_score")
    if market_relevance_score is not None:
        market_relevance_score = _as_float(market_relevance_score, 0.0)

    signal_id = _stable_signal_id(
        str(record.source_slug or "events"),
        str(record.external_id or "") or None,
        str(record.url or "") or None,
        title,
        observed_at,
    )

    return {
        "signal_id": signal_id,
        "signal_type": signal_type,
        "severity": severity,
        "country": country_iso3,
        "latitude": record.latitude,
        "longitude": record.longitude,
        "title": title,
        "description": summary,
        "source": source,
        "detected_at": observed_at,
        "metadata": metadata,
        "related_market_ids": related_market_ids,
        "market_relevance_score": market_relevance_score,
    }


def _dedupe_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for signal in signals:
        signal_id = str(signal.get("signal_id") or "").strip()
        if not signal_id:
            continue
        existing = by_id.get(signal_id)
        if existing is None:
            by_id[signal_id] = signal
            continue
        existing_severity = _as_float(existing.get("severity"), 0.0)
        incoming_severity = _as_float(signal.get("severity"), 0.0)
        if incoming_severity >= existing_severity:
            by_id[signal_id] = signal
    out = list(by_id.values())
    out.sort(
        key=lambda item: (
            _to_aware(item.get("detected_at")).timestamp(),
            _as_float(item.get("severity"), 0.0),
        ),
        reverse=True,
    )
    return out


def _build_signal_summary(signals: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    critical = 0
    countries: set[str] = set()

    for signal in signals:
        signal_type = str(signal.get("signal_type") or "unknown").strip().lower() or "unknown"
        by_type[signal_type] = by_type.get(signal_type, 0) + 1
        severity = _as_float(signal.get("severity"), 0.0)
        if severity >= 0.7:
            critical += 1
        country = _as_iso3(signal.get("country"))
        if country:
            countries.add(country)

    return {
        "total": len(signals),
        "critical": critical,
        "countries_tracked": len(countries),
        "by_type": by_type,
    }


async def _persist_signals(signals: list[dict[str, Any]]) -> int:
    if not signals:
        return 0

    try:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        persisted = 0
        async with AsyncSessionLocal() as session:
            for signal in signals:
                signal_id = str(signal.get("signal_id") or "").strip()
                if not signal_id:
                    continue
                signal_type = str(signal.get("signal_type") or "world").strip().lower() or "world"
                country_iso3 = _as_iso3(signal.get("country"))
                detected_at = _to_aware(signal.get("detected_at"))
                related_market_ids = [
                    str(item or "").strip()
                    for item in _as_list(signal.get("related_market_ids"))
                    if str(item or "").strip()
                ]
                metadata = signal.get("metadata") if isinstance(signal.get("metadata"), dict) else {}
                stmt = (
                    sqlite_insert(EventsSignal)
                    .values(
                        id=signal_id,
                        signal_type=signal_type,
                        severity=_clamp(_as_float(signal.get("severity"), 0.0), 0.0, 1.0),
                        country=country_iso3,
                        iso3=country_iso3,
                        latitude=signal.get("latitude"),
                        longitude=signal.get("longitude"),
                        title=str(signal.get("title") or "Events signal").strip() or "Events signal",
                        description=str(signal.get("description") or "").strip(),
                        source=str(signal.get("source") or "events").strip().lower() or "events",
                        detected_at=detected_at,
                        metadata_json=metadata,
                        related_market_ids=related_market_ids,
                        market_relevance_score=(
                            _as_float(signal.get("market_relevance_score"), 0.0)
                            if signal.get("market_relevance_score") is not None
                            else None
                        ),
                    )
                    .on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "signal_type": signal_type,
                            "severity": _clamp(_as_float(signal.get("severity"), 0.0), 0.0, 1.0),
                            "country": country_iso3,
                            "iso3": country_iso3,
                            "latitude": signal.get("latitude"),
                            "longitude": signal.get("longitude"),
                            "title": str(signal.get("title") or "Events signal").strip() or "Events signal",
                            "description": str(signal.get("description") or "").strip(),
                            "source": str(signal.get("source") or "events").strip().lower() or "events",
                            "detected_at": detected_at,
                            "metadata_json": metadata,
                            "related_market_ids": related_market_ids,
                            "market_relevance_score": (
                                _as_float(signal.get("market_relevance_score"), 0.0)
                                if signal.get("market_relevance_score") is not None
                                else None
                            ),
                        },
                    )
                )
                await session.execute(stmt)
                persisted += 1
            await session.commit()
        return persisted
    except Exception as exc:
        logger.warning("Failed to persist events signals: %s", exc)
        return 0


async def _write_snapshot(
    status: dict[str, Any],
    stats: dict[str, Any] | None = None,
    signals: list[dict[str, Any]] | None = None,
    preserve_existing: bool = False,
) -> None:
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            async with AsyncSessionLocal() as session:
                existing = None
                if preserve_existing:
                    existing = await session.get(EventsSnapshot, "latest")

                existing_status = existing.status if existing and isinstance(existing.status, dict) else {}
                next_status = dict(existing_status) if preserve_existing else {}
                next_status.update(status or {})

                next_stats = (
                    (existing.stats if existing and isinstance(existing.stats, dict) else {})
                    if stats is None
                    else stats
                )
                next_signals = (
                    (existing.signals_json if existing and isinstance(existing.signals_json, list) else [])
                    if signals is None
                    else signals
                )

                stmt = (
                    sqlite_insert(EventsSnapshot)
                    .values(
                        id="latest",
                        status=next_status,
                        signals_json=next_signals,
                        stats=next_stats,
                        updated_at=_utcnow(),
                    )
                    .on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "status": next_status,
                            "signals_json": next_signals,
                            "stats": next_stats,
                            "updated_at": _utcnow(),
                        },
                    )
                )
                await session.execute(stmt)
                await session.commit()
            return
        except Exception as exc:
            if "locked" in str(exc).lower() and attempt < max_attempts:
                await asyncio.sleep(0.15 * attempt)
                continue
            logger.warning("Failed to write events snapshot: %s", exc)
            return


async def _read_existing_worker_stats() -> dict[str, Any]:
    try:
        async with AsyncSessionLocal() as session:
            snapshot = await read_worker_snapshot(session, "events")
            stats = snapshot.get("stats")
            if isinstance(stats, dict):
                return stats
    except Exception:
        pass
    return {}


async def _broadcast_update(signals: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    try:
        from api.websocket import broadcast_events_update

        signal_dicts = [
            {
                "signal_id": str(signal.get("signal_id") or ""),
                "signal_type": str(signal.get("signal_type") or "world"),
                "severity": round(_as_float(signal.get("severity"), 0.0), 3),
                "country": _as_iso3(signal.get("country")),
                "latitude": signal.get("latitude"),
                "longitude": signal.get("longitude"),
                "title": str(signal.get("title") or "Events signal"),
                "description": str(signal.get("description") or ""),
                "source": str(signal.get("source") or "events"),
                "detected_at": _to_aware(signal.get("detected_at")).isoformat(),
                "related_market_ids": [str(item) for item in _as_list(signal.get("related_market_ids"))],
                "market_relevance_score": (
                    round(_as_float(signal.get("market_relevance_score"), 0.0), 3)
                    if signal.get("market_relevance_score") is not None
                    else None
                ),
            }
            for signal in signals[:50]
        ]
        await broadcast_events_update(signal_dicts, summary)
    except Exception:
        pass


async def _load_enabled_event_sources() -> list[DataSource]:
    async with AsyncSessionLocal() as session:
        query = (
            select(DataSource)
            .where(DataSource.source_key == "events")
            .where(DataSource.enabled == True)  # noqa: E712
            .order_by(DataSource.sort_order.asc(), DataSource.slug.asc())
        )
        rows = (await session.execute(query)).scalars().all()
    return list(rows)


async def _run_event_data_sources_once() -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], list[str], dict[str, int]
]:
    sources = await _load_enabled_event_sources()
    source_status: dict[str, dict[str, Any]] = {}
    source_errors: list[str] = []
    all_signals: list[dict[str, Any]] = []

    run_metrics = {
        "sources_total": len(sources),
        "sources_succeeded": 0,
        "sources_failed": 0,
    }

    for source in sources:
        slug = str(source.slug or "").strip().lower()
        health_key = _source_status_key(slug)
        max_records = max(1, min(5000, int((source.config or {}).get("limit") or 500)))
        run_start = datetime.now(timezone.utc).replace(tzinfo=None)

        run_result: dict[str, Any]
        records: list[DataSourceRecord]

        try:
            async with AsyncSessionLocal() as session:
                row = await session.get(DataSource, source.id)
                if row is None or not bool(row.enabled):
                    continue
                run_result = await run_data_source(session, row, max_records=max_records, commit=True)
                records = (
                    (
                        await session.execute(
                            select(DataSourceRecord)
                            .where(DataSourceRecord.source_slug == slug)
                            .where(DataSourceRecord.ingested_at >= run_start)
                            .order_by(
                                desc(DataSourceRecord.observed_at),
                                desc(DataSourceRecord.ingested_at),
                            )
                            .limit(max_records)
                        )
                    )
                    .scalars()
                    .all()
                )
        except Exception as exc:
            run_result = {
                "status": "error",
                "error_message": str(exc),
                "duration_ms": 0,
            }
            records = []

        source_signals = [_record_to_signal(record) for record in records]
        all_signals.extend(source_signals)

        run_ok = str(run_result.get("status") or "").strip().lower() == "success"
        error_text = str(run_result.get("error_message") or "").strip() or None
        duration_seconds = round((_as_float(run_result.get("duration_ms"), 0.0) / 1000.0), 3)

        source_status[health_key] = {
            "ok": bool(run_ok),
            "count": len(source_signals),
            "duration_seconds": duration_seconds,
            "error": error_text,
            "last_error": error_text,
            "last_updated": _utcnow().isoformat(),
            "source_slug": slug,
        }

        if run_ok:
            run_metrics["sources_succeeded"] += 1
        else:
            run_metrics["sources_failed"] += 1
            if error_text:
                source_errors.append(f"{health_key}: {error_text}")

    return _dedupe_signals(all_signals), source_status, source_errors, run_metrics


async def _run_loop() -> None:
    logger.info("Events worker started")

    from services.data_source_catalog import ensure_all_data_sources_seeded
    from services.data_source_loader import data_source_loader

    try:
        async with AsyncSessionLocal() as session:
            await ensure_all_data_sources_seeded(session)
            await data_source_loader.refresh_all_from_db(session=session)
    except Exception as exc:
        logger.warning("Events data source seed/refresh failed: %s", exc)

    next_scheduled_run_at: datetime | None = None

    while True:
        try:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(session, "events")

            interval = int(
                max(
                    30,
                    min(
                        3600,
                        control.get("interval_seconds", settings.EVENTS_INTERVAL_SECONDS)
                        or settings.EVENTS_INTERVAL_SECONDS,
                    ),
                )
            )
            control_enabled = bool(control.get("is_enabled", True))
            paused = bool(control.get("is_paused", False))
            settings_enabled = bool(getattr(settings, "EVENTS_ENABLED", True))
            requested = control.get("requested_run_at") is not None
            enabled = settings_enabled and control_enabled and not paused

            now = _utcnow()
            should_run_scheduled = enabled and (next_scheduled_run_at is None or now >= next_scheduled_run_at)
            should_run = requested or should_run_scheduled

            if not should_run:
                next_scan = next_scheduled_run_at.isoformat() if next_scheduled_run_at and enabled else None
                activity = (
                    "Disabled by EVENTS_ENABLED setting"
                    if not settings_enabled
                    else "Paused"
                    if paused
                    else "Idle - waiting for next collection cycle."
                )
                existing_stats = await _read_existing_worker_stats()
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        "events",
                        running=True,
                        enabled=enabled,
                        current_activity=activity,
                        interval_seconds=interval,
                        last_run_at=None,
                        last_error=None,
                        stats=existing_stats,
                    )
                await _write_snapshot(
                    status={
                        "running": True,
                        "enabled": enabled,
                        "interval_seconds": interval,
                        "next_scan": next_scan,
                        "current_activity": activity,
                    },
                    stats=None,
                    signals=None,
                    preserve_existing=True,
                )
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)
                continue

            await _write_snapshot(
                status={
                    "running": True,
                    "enabled": enabled,
                    "interval_seconds": interval,
                    "current_activity": "Running events data-source cycle...",
                    "next_scan": None,
                },
                stats=None,
                signals=None,
                preserve_existing=True,
            )

            cycle_start = _utcnow()
            signals, source_status, source_errors, run_metrics = await _run_event_data_sources_once()
            cycle_duration = (_utcnow() - cycle_start).total_seconds()

            persisted_signals = await _persist_signals(signals)
            summary = _build_signal_summary(signals)

            emitted_signals = 0

            await _broadcast_update(signals, summary)

            completed_at = _utcnow()
            next_scheduled_run_at = completed_at.replace(microsecond=0) + timedelta(seconds=interval)

            top_signals = [
                {
                    "signal_id": str(signal.get("signal_id") or ""),
                    "signal_type": str(signal.get("signal_type") or "world"),
                    "severity": round(_as_float(signal.get("severity"), 0.0), 3),
                    "country": _as_iso3(signal.get("country")),
                    "latitude": signal.get("latitude"),
                    "longitude": signal.get("longitude"),
                    "title": str(signal.get("title") or "Events signal"),
                    "description": str(signal.get("description") or ""),
                    "source": str(signal.get("source") or "events"),
                    "detected_at": _to_aware(signal.get("detected_at")).isoformat(),
                    "related_market_ids": [
                        str(item) for item in _as_list(signal.get("related_market_ids")) if str(item)
                    ],
                    "market_relevance_score": (
                        round(_as_float(signal.get("market_relevance_score"), 0.0), 3)
                        if signal.get("market_relevance_score") is not None
                        else None
                    ),
                    "metadata": signal.get("metadata") if isinstance(signal.get("metadata"), dict) else {},
                }
                for signal in signals[:200]
            ]

            stats = {
                "total_signals": len(signals),
                "persisted_signals": persisted_signals,
                "persisted_scores": 0,
                "persisted_tensions": 0,
                "persisted_conflicts": 0,
                "emitted_trade_signals": int(emitted_signals),
                "cycle_duration_seconds": round(cycle_duration, 2),
                "critical_signals": int(summary.get("critical", 0) or 0),
                "countries_tracked": int(summary.get("countries_tracked", 0) or 0),
                "tension_pairs_tracked": int(summary.get("by_type", {}).get("tension", 0) or 0),
                "signal_breakdown": dict(summary.get("by_type") or {}),
                "source_status": source_status,
                "source_errors": source_errors,
                "source_runs": run_metrics,
                "country_reference": {},
                "ucdp_conflicts": {},
                "mid_reference": {},
                "trade_dependencies": {},
                "chokepoint_reference": {},
                "gdelt_news": {},
                "runtime_state": {},
            }

            async with AsyncSessionLocal() as session:
                await clear_worker_run_request(session, "events")

            await _write_snapshot(
                status={
                    "running": True,
                    "enabled": enabled,
                    "current_activity": "Idle - waiting for next collection cycle.",
                    "last_scan": completed_at.isoformat(),
                    "interval_seconds": interval,
                    "next_scan": next_scheduled_run_at.isoformat(),
                    "last_error": None,
                },
                stats=stats,
                signals=top_signals,
            )

            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    "events",
                    running=True,
                    enabled=enabled,
                    current_activity="Idle - waiting for next collection cycle.",
                    interval_seconds=interval,
                    last_run_at=completed_at.replace(tzinfo=None),
                    last_error=None,
                    stats=stats,
                )

            logger.info(
                "Events cycle complete: signals=%d persisted=%d duration=%.2fs sources=%d",
                len(signals),
                persisted_signals,
                cycle_duration,
                run_metrics.get("sources_total", 0),
            )
            await asyncio.sleep(_IDLE_SLEEP_SECONDS)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Events cycle failed: %s", exc)
            try:
                existing_stats = await _read_existing_worker_stats()
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        "events",
                        running=True,
                        enabled=True,
                        current_activity=f"Last cycle error: {exc}",
                        interval_seconds=settings.EVENTS_INTERVAL_SECONDS,
                        last_run_at=_utcnow().replace(tzinfo=None),
                        last_error=str(exc),
                        stats=existing_stats,
                    )
                    await clear_worker_run_request(session, "events")
                await _write_snapshot(
                    status={
                        "running": True,
                        "enabled": True,
                        "current_activity": f"Error: {exc}",
                        "last_error": str(exc),
                    },
                    stats=None,
                    signals=None,
                    preserve_existing=True,
                )
            except Exception:
                pass
            await asyncio.sleep(min(_IDLE_SLEEP_SECONDS, settings.EVENTS_INTERVAL_SECONDS))


async def start_loop() -> None:
    """Run the events worker loop (called from API process lifespan)."""
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Events worker shutting down")
