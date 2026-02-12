"""World Intelligence worker: collects global signals and writes DB snapshots.

Run from backend dir:
  python -m workers.world_intelligence_worker
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

from config import settings
from models.database import (
    AsyncSessionLocal,
    WorldIntelligenceSignal,
    WorldIntelligenceSnapshot,
    CountryInstabilityRecord,
    TensionPairRecord,
    ConflictEventRecord,
    init_database,
)
from services.worker_state import write_worker_snapshot, read_worker_control
from services.worker_state import read_worker_snapshot
from services.worker_state import clear_worker_run_request

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("world_intelligence_worker")

_IDLE_SLEEP_SECONDS = 5


async def _persist_signals(signals) -> int:
    """Persist world signals to database."""
    if not signals:
        return 0
    try:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        persisted = 0
        async with AsyncSessionLocal() as session:
            for s in signals:
                iso3 = (
                    s.country
                    if isinstance(s.country, str) and len(s.country) == 3
                    else None
                )
                stmt = sqlite_insert(WorldIntelligenceSignal).values(
                    id=s.signal_id,
                    signal_type=s.signal_type,
                    severity=s.severity,
                    country=s.country,
                    iso3=iso3,
                    latitude=s.latitude,
                    longitude=s.longitude,
                    title=s.title,
                    description=s.description,
                    source=s.source,
                    detected_at=s.detected_at,
                    metadata_json=s.metadata,
                    related_market_ids=s.related_market_ids,
                    market_relevance_score=s.market_relevance_score,
                ).on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "signal_type": s.signal_type,
                        "severity": s.severity,
                        "country": s.country,
                        "iso3": iso3,
                        "latitude": s.latitude,
                        "longitude": s.longitude,
                        "title": s.title,
                        "description": s.description,
                        "source": s.source,
                        "detected_at": s.detected_at,
                        "metadata_json": s.metadata,
                        "related_market_ids": s.related_market_ids,
                        "market_relevance_score": s.market_relevance_score,
                    },
                )
                await session.execute(stmt)
                persisted += 1
            await session.commit()
        return persisted
    except Exception as e:
        logger.warning("Failed to persist world signals: %s", e)
        return 0


async def _persist_instability_scores(scores) -> int:
    """Persist instability scores to database."""
    if not scores:
        return 0
    try:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        persisted = 0
        async with AsyncSessionLocal() as session:
            for iso3, score in scores.items():
                record_id = f"cii_{iso3}_{datetime.now(timezone.utc).strftime('%Y%m%d%H')}"
                stmt = sqlite_insert(CountryInstabilityRecord).values(
                    id=record_id,
                    country=score.country,
                    iso3=score.iso3,
                    score=score.score,
                    components=score.components,
                    trend=score.trend,
                    computed_at=score.last_updated or datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["id"],
                    set_={"score": score.score, "components": score.components, "trend": score.trend},
                )
                await session.execute(stmt)
                persisted += 1
            await session.commit()
        return persisted
    except Exception as e:
        logger.warning("Failed to persist instability scores: %s", e)
        return 0


async def _persist_tension_pairs(pairs) -> int:
    """Persist tension pair records to database."""
    if not pairs:
        return 0
    try:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        persisted = 0
        async with AsyncSessionLocal() as session:
            for p in pairs:
                record_id = f"tension_{p.country_a}_{p.country_b}_{datetime.now(timezone.utc).strftime('%Y%m%d%H')}"
                stmt = sqlite_insert(TensionPairRecord).values(
                    id=record_id,
                    country_a=p.country_a,
                    country_b=p.country_b,
                    tension_score=p.tension_score,
                    event_count=p.event_count,
                    avg_goldstein_scale=p.avg_goldstein_scale,
                    trend=p.trend,
                    computed_at=p.last_updated or datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["id"],
                    set_={"tension_score": p.tension_score, "trend": p.trend},
                )
                await session.execute(stmt)
                persisted += 1
            await session.commit()
        return persisted
    except Exception as e:
        logger.warning("Failed to persist tension pairs: %s", e)
        return 0


async def _persist_conflict_events(events) -> int:
    """Persist ACLED conflict events to database."""
    if not events:
        return 0
    try:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        persisted = 0
        async with AsyncSessionLocal() as session:
            for evt in events:
                event_date = None
                severity_score = None
                try:
                    if isinstance(evt.event_date, str) and evt.event_date:
                        event_date = datetime.fromisoformat(evt.event_date)
                    elif isinstance(evt.event_date, datetime):
                        event_date = evt.event_date
                except Exception:
                    event_date = None
                try:
                    from services.world_intelligence.acled_client import ACLEDClient

                    severity_score = float(ACLEDClient.get_severity_score(evt))
                except Exception:
                    severity_score = None
                stmt = sqlite_insert(ConflictEventRecord).values(
                    id=str(evt.event_id),
                    event_type=evt.event_type,
                    sub_event_type=evt.sub_event_type,
                    country=evt.country,
                    iso3=evt.iso3,
                    latitude=evt.latitude,
                    longitude=evt.longitude,
                    fatalities=evt.fatalities,
                    event_date=event_date,
                    source=evt.source,
                    notes=evt.notes[:500] if evt.notes else None,
                    severity_score=severity_score,
                    fetched_at=datetime.now(timezone.utc),
                ).on_conflict_do_nothing()
                await session.execute(stmt)
                persisted += 1
            await session.commit()
        return persisted
    except Exception as e:
        logger.warning("Failed to persist conflict events: %s", e)
        return 0


async def _write_snapshot(
    status: dict,
    stats: dict | None = None,
    signals: list[dict] | None = None,
    preserve_existing: bool = False,
):
    """Write world intelligence snapshot to DB."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            async with AsyncSessionLocal() as session:
                existing = None
                if preserve_existing:
                    existing = await session.get(WorldIntelligenceSnapshot, "latest")

                existing_status = (
                    existing.status if existing and isinstance(existing.status, dict) else {}
                )
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

                stmt = sqlite_insert(WorldIntelligenceSnapshot).values(
                    id="latest",
                    status=next_status,
                    signals_json=next_signals,
                    stats=next_stats,
                    updated_at=datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "status": next_status,
                        "signals_json": next_signals,
                        "stats": next_stats,
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                await session.execute(stmt)
                await session.commit()
            return
        except Exception as e:
            message = str(e).lower()
            if "locked" in message and attempt < max_attempts:
                await asyncio.sleep(0.15 * attempt)
                continue
            logger.warning("Failed to write world intelligence snapshot: %s", e)
            return


async def _read_existing_worker_stats() -> dict:
    """Best-effort read of last worker stats so idle/error updates don't blank telemetry."""
    try:
        async with AsyncSessionLocal() as session:
            snapshot = await read_worker_snapshot(session, "world_intelligence")
            stats = snapshot.get("stats")
            if isinstance(stats, dict):
                return stats
    except Exception:
        pass
    return {}


async def _broadcast_update(signals, summary):
    """Broadcast world intelligence update via WebSocket."""
    try:
        from api.websocket import broadcast_world_intelligence_update

        signal_dicts = [
            {
                "signal_id": s.signal_id,
                "signal_type": s.signal_type,
                "severity": round(s.severity, 3),
                "country": s.country,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "title": s.title,
                "description": s.description,
                "source": s.source,
                "detected_at": s.detected_at.isoformat() if s.detected_at else None,
                "related_market_ids": s.related_market_ids,
                "market_relevance_score": round(s.market_relevance_score, 3) if s.market_relevance_score else None,
            }
            for s in (signals or [])[:50]
        ]
        await broadcast_world_intelligence_update(signal_dicts, summary)
    except Exception:
        pass  # WS not available in worker process


async def _run_loop() -> None:
    logger.info("World Intelligence worker started")

    # Import intelligence modules
    from services.world_intelligence import signal_aggregator

    try:
        async with AsyncSessionLocal() as session:
            snapshot = await session.get(WorldIntelligenceSnapshot, "latest")
        runtime_state = {}
        if snapshot is not None and isinstance(snapshot.stats, dict):
            runtime_state = (snapshot.stats or {}).get("runtime_state") or {}
        if isinstance(runtime_state, dict) and runtime_state:
            signal_aggregator.import_runtime_state(runtime_state)
            logger.info("World Intelligence runtime state restored from snapshot")
    except Exception as exc:
        logger.debug("Failed restoring world-intelligence runtime state: %s", exc)

    next_scheduled_run_at: datetime | None = None

    while True:
        try:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(session, "world_intelligence")

            interval = int(
                max(
                    30,
                    min(
                        3600,
                        control.get(
                            "interval_seconds",
                            settings.WORLD_INTELLIGENCE_INTERVAL_SECONDS,
                        )
                        or settings.WORLD_INTELLIGENCE_INTERVAL_SECONDS,
                    ),
                )
            )
            control_enabled = bool(control.get("is_enabled", True))
            paused = bool(control.get("is_paused", False))
            settings_enabled = bool(getattr(settings, "WORLD_INTELLIGENCE_ENABLED", True))
            requested = control.get("requested_run_at") is not None
            enabled = settings_enabled and control_enabled and not paused

            now = datetime.now(timezone.utc)
            should_run_scheduled = enabled and (
                next_scheduled_run_at is None or now >= next_scheduled_run_at
            )
            should_run = requested or should_run_scheduled

            if not should_run:
                next_scan = (
                    next_scheduled_run_at.isoformat()
                    if next_scheduled_run_at and enabled
                    else None
                )
                activity = (
                    "Disabled by WORLD_INTELLIGENCE_ENABLED setting"
                    if not settings_enabled
                    else "Paused"
                    if paused
                    else "Idle - waiting for next collection cycle."
                )
                existing_stats = await _read_existing_worker_stats()
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        "world_intelligence",
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

            # Update activity status
            await _write_snapshot(
                status={
                    "running": True,
                    "enabled": enabled,
                    "interval_seconds": interval,
                    "current_activity": "Running collection cycle...",
                    "next_scan": None,
                },
                stats=None,
                signals=None,
                preserve_existing=True,
            )

            # Run collection cycle
            cycle_start = datetime.now(timezone.utc)
            signals = await signal_aggregator.run_collection_cycle()
            cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()

            # Persist results
            persisted_signals = await _persist_signals(signals)

            # Persist instability scores
            from services.world_intelligence import instability_scorer, tension_tracker
            scores = instability_scorer.get_all_scores()
            persisted_scores = await _persist_instability_scores(scores)

            tensions = tension_tracker.get_all_tensions()
            persisted_tensions = await _persist_tension_pairs(tensions)

            # Persist conflict events from ACLED
            from services.world_intelligence import acled_client
            # Events were already fetched during collection cycle, just grab cached
            conflict_events = getattr(acled_client, "_last_events", [])
            persisted_conflicts = await _persist_conflict_events(conflict_events)

            # World intelligence should run independently of trader orchestrator by default.
            emitted_signals = 0
            if bool(getattr(settings, "WORLD_INTEL_EMIT_TRADE_SIGNALS", False)):
                try:
                    from services.world_intelligence.signal_emitter import emit_world_intelligence_signals

                    async with AsyncSessionLocal() as sig_session:
                        emitted_signals = await emit_world_intelligence_signals(
                            sig_session,
                            signals,
                            max_age_minutes=120,
                        )
                except Exception as sig_exc:
                    logger.debug("World intelligence signal emission failed: %s", sig_exc)
                    emitted_signals = 0

            # Get summary for broadcast
            summary = signal_aggregator.get_signal_summary()
            source_status = signal_aggregator.get_source_status()
            source_errors = signal_aggregator.get_last_errors()

            # Broadcast
            await _broadcast_update(signals, summary)

            # Compute stats
            stats = {
                "total_signals": len(signals),
                "persisted_signals": persisted_signals,
                "persisted_scores": persisted_scores,
                "persisted_tensions": persisted_tensions,
                "persisted_conflicts": persisted_conflicts,
                "emitted_trade_signals": emitted_signals,
                "cycle_duration_seconds": round(cycle_duration, 2),
                "critical_signals": len([s for s in signals if s.severity >= 0.7]),
                "countries_tracked": len(scores),
                "tension_pairs_tracked": len(tensions),
                "signal_breakdown": summary.get("by_type", {}),
                "source_status": source_status,
                "source_errors": source_errors,
                "runtime_state": signal_aggregator.export_runtime_state(),
            }

            completed_at = datetime.now(timezone.utc)
            next_scheduled_run_at = completed_at.replace(microsecond=0) + timedelta(
                seconds=interval
            )
            top_signals = [
                {
                    "signal_id": s.signal_id,
                    "signal_type": s.signal_type,
                    "severity": round(s.severity, 3),
                    "country": s.country,
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                    "title": s.title,
                    "description": s.description,
                    "source": s.source,
                    "detected_at": s.detected_at.isoformat()
                    if s.detected_at
                    else None,
                    "related_market_ids": list(s.related_market_ids or []),
                    "market_relevance_score": (
                        round(float(s.market_relevance_score), 3)
                        if s.market_relevance_score is not None
                        else None
                    ),
                    "metadata": s.metadata,
                }
                for s in signals[:200]
            ]

            async with AsyncSessionLocal() as session:
                await clear_worker_run_request(session, "world_intelligence")

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
                    "world_intelligence",
                    running=True,
                    enabled=enabled,
                    current_activity="Idle - waiting for next collection cycle.",
                    interval_seconds=interval,
                    last_run_at=completed_at.replace(tzinfo=None),
                    last_error=None,
                    stats=stats,
                )

            logger.info(
                "World Intelligence cycle complete: %d signals, %.1fs duration",
                len(signals),
                cycle_duration,
            )
            await asyncio.sleep(_IDLE_SLEEP_SECONDS)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("World Intelligence cycle failed: %s", exc)
            try:
                existing_stats = await _read_existing_worker_stats()
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        "world_intelligence",
                        running=True,
                        enabled=True,
                        current_activity=f"Last cycle error: {exc}",
                        interval_seconds=settings.WORLD_INTELLIGENCE_INTERVAL_SECONDS,
                        last_run_at=datetime.now(timezone.utc).replace(tzinfo=None),
                        last_error=str(exc),
                        stats=existing_stats,
                    )
                    await clear_worker_run_request(session, "world_intelligence")
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
            await asyncio.sleep(min(_IDLE_SLEEP_SECONDS, settings.WORLD_INTELLIGENCE_INTERVAL_SECONDS))


async def main() -> None:
    await init_database()
    logger.info("Database initialized")
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("World Intelligence worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
