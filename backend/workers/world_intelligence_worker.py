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
from datetime import datetime, timezone

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
                stmt = sqlite_insert(WorldIntelligenceSignal).values(
                    id=s.signal_id,
                    signal_type=s.signal_type,
                    severity=s.severity,
                    country=s.country,
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
                        "severity": s.severity,
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
                stmt = sqlite_insert(ConflictEventRecord).values(
                    id=str(evt.event_id),
                    event_type=evt.event_type,
                    sub_event_type=evt.sub_event_type,
                    country=evt.country,
                    iso3=evt.iso3,
                    latitude=evt.latitude,
                    longitude=evt.longitude,
                    fatalities=evt.fatalities,
                    event_date=evt.event_date,
                    source=evt.source,
                    notes=evt.notes[:500] if evt.notes else None,
                    severity_score=evt.severity_score if hasattr(evt, "severity_score") else None,
                    fetched_at=datetime.now(timezone.utc),
                ).on_conflict_do_nothing()
                await session.execute(stmt)
                persisted += 1
            await session.commit()
        return persisted
    except Exception as e:
        logger.warning("Failed to persist conflict events: %s", e)
        return 0


async def _write_snapshot(status: dict, stats: dict):
    """Write world intelligence snapshot to DB."""
    try:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        async with AsyncSessionLocal() as session:
            stmt = sqlite_insert(WorldIntelligenceSnapshot).values(
                id="latest",
                status=status,
                stats=stats,
                updated_at=datetime.now(timezone.utc),
            ).on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "status": status,
                    "stats": stats,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as e:
        logger.warning("Failed to write world intelligence snapshot: %s", e)


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

    if not settings.WORLD_INTELLIGENCE_ENABLED:
        logger.info("World Intelligence is disabled; worker idling")
        while True:
            await asyncio.sleep(60)

    # Import intelligence modules
    from services.world_intelligence import signal_aggregator

    while True:
        try:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(session, "world_intelligence")

            interval = control.get("interval_seconds", settings.WORLD_INTELLIGENCE_INTERVAL_SECONDS)
            enabled = control.get("is_enabled", True)
            paused = control.get("is_paused", False)

            if not enabled or paused:
                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        "world_intelligence",
                        running=True,
                        enabled=enabled and not paused,
                        current_activity="Paused" if paused else "Disabled",
                        interval_seconds=interval,
                        last_run_at=None,
                        last_error=None,
                        stats={},
                    )
                await _write_snapshot(
                    status={"running": True, "enabled": False, "current_activity": "Paused" if paused else "Disabled"},
                    stats={},
                )
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)
                continue

            # Update activity status
            await _write_snapshot(
                status={"running": True, "enabled": True, "current_activity": "Running collection cycle..."},
                stats={},
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

            # Emit trade signals into the signal bus for autotrader consumption
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
            }

            completed_at = datetime.now(timezone.utc)

            await _write_snapshot(
                status={
                    "running": True,
                    "enabled": True,
                    "current_activity": "Idle - waiting for next collection cycle.",
                    "last_scan": completed_at.isoformat(),
                    "interval_seconds": interval,
                },
                stats=stats,
            )

            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    "world_intelligence",
                    running=True,
                    enabled=True,
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

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("World Intelligence cycle failed: %s", exc)
            try:
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
                        stats={},
                    )
                await _write_snapshot(
                    status={
                        "running": True,
                        "enabled": True,
                        "current_activity": f"Error: {exc}",
                        "last_error": str(exc),
                    },
                    stats={},
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
