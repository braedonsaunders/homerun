"""Weather worker: runs independent weather workflow and writes DB snapshot.

Run from backend dir:
  python -m workers.weather_worker
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import InterfaceError, OperationalError

from config import settings
from models.database import AsyncSessionLocal
from services.data_events import DataEvent
from services.event_dispatcher import event_dispatcher
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_signal_bridge import bridge_opportunities_to_signals
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.weather.workflow_orchestrator import weather_workflow_orchestrator
from services.weather import shared_state
from services.worker_state import write_worker_snapshot
from utils.logger import setup_logging

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("weather_worker")


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


async def _run_loop() -> None:
    logger.info("Weather worker started")

    try:
        async with AsyncSessionLocal() as session:
            await ensure_all_strategies_seeded(session)
            await refresh_strategy_runtime_if_needed(
                session,
                source_keys=["weather"],
                force=True,
            )
    except Exception as exc:
        logger.warning("Weather worker strategy startup sync failed: %s", exc)

    # Ensure initial snapshot exists for UI status.
    try:
        async with AsyncSessionLocal() as session:
            await shared_state.write_weather_snapshot(
                session,
                opportunities=[],
                status={
                    "running": True,
                    "enabled": True,
                    "interval_seconds": settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS,
                    "last_scan": None,
                    "current_activity": "Weather worker started; first scan pending.",
                },
                stats={},
            )
            await write_worker_snapshot(
                session,
                "weather",
                running=True,
                enabled=True,
                current_activity="Weather worker started; first scan pending.",
                interval_seconds=settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS,
                last_run_at=None,
                last_error=None,
                stats={"pending_intents": 0, "signals_emitted_last_run": 0},
            )
    except Exception:
        pass

    next_scheduled_run_at: datetime | None = None

    while True:
        async with AsyncSessionLocal() as session:
            control = await shared_state.read_weather_control(session)
            wf_settings = await shared_state.get_weather_settings(session)
            try:
                await refresh_strategy_runtime_if_needed(
                    session,
                    source_keys=["weather"],
                )
            except Exception as exc:
                logger.warning("Weather worker strategy refresh check failed: %s", exc)

        interval = int(
            max(
                300,
                min(
                    86400,
                    control.get("scan_interval_seconds")
                    or wf_settings.get("scan_interval_seconds")
                    or settings.WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS,
                ),
            )
        )
        paused = bool(control.get("is_paused", False))
        requested = control.get("requested_scan_at") is not None
        enabled = bool(wf_settings.get("enabled", True))
        auto_run = bool(wf_settings.get("auto_run", True))
        now = datetime.now(timezone.utc)

        should_run_scheduled = (
            enabled and auto_run and not paused and (next_scheduled_run_at is None or now >= next_scheduled_run_at)
        )
        should_run = requested or should_run_scheduled

        if not should_run:
            try:
                async with AsyncSessionLocal() as session:
                    try:
                        pending = len(
                            await shared_state.list_weather_intents(session, status_filter="pending", limit=2000)
                        )
                    except Exception:
                        pending = 0
                    await write_worker_snapshot(
                        session,
                        "weather",
                        running=True,
                        enabled=enabled and not paused,
                        current_activity=("Paused" if paused else "Idle - waiting for next weather workflow cycle."),
                        interval_seconds=interval,
                        last_run_at=None,
                        last_error=None,
                        stats={
                            "pending_intents": int(pending),
                            "signals_emitted_last_run": 0,
                        },
                    )
            except Exception:
                pass
            await asyncio.sleep(min(10, interval))
            continue

        try:
            result: dict = {}
            pending_rows: list = []
            enriched_intents: list[dict] = []
            max_db_attempts = 2
            for attempt in range(max_db_attempts):
                try:
                    async with AsyncSessionLocal() as session:
                        result = await weather_workflow_orchestrator.run_cycle(session)
                        await shared_state.clear_weather_scan_request(session)
                        try:
                            pending_rows = await shared_state.list_weather_intents(
                                session,
                                status_filter="pending",
                                limit=2000,
                            )
                        except Exception:
                            pending_rows = []
                        enriched_intents = shared_state.get_enriched_weather_intents()
                    break
                except Exception as exc:
                    is_retryable_disconnect = _is_retryable_db_disconnect_error(exc)
                    if not is_retryable_disconnect or attempt >= max_db_attempts - 1:
                        raise
                    logger.warning("Weather workflow DB connection dropped; retrying cycle: %s", exc)
                    await asyncio.sleep(0.25 * (attempt + 1))

            intent_dicts: list[dict] = []
            if enriched_intents:
                intent_dicts = [dict(intent) for intent in enriched_intents if isinstance(intent, dict)]
            else:
                for row in pending_rows:
                    intent_dict = {
                        "id": row.id,
                        "market_id": row.market_id,
                        "market_question": row.market_question,
                        "direction": row.direction,
                        "entry_price": row.entry_price,
                        "take_profit_price": row.take_profit_price,
                        "stop_loss_pct": row.stop_loss_pct,
                        "model_probability": row.model_probability,
                        "edge_percent": row.edge_percent,
                        "confidence": row.confidence,
                        "model_agreement": row.model_agreement,
                        "suggested_size_usd": row.suggested_size_usd,
                        "status": row.status,
                        "created_at": row.created_at,
                    }
                    metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
                    intent_dict.update(metadata)
                    intent_dicts.append(intent_dict)

            weather_event = DataEvent(
                event_type="weather_update",
                source="weather_worker",
                timestamp=datetime.now(timezone.utc),
                payload={"intents": intent_dicts},
            )
            dispatched_opportunities = await event_dispatcher.dispatch(weather_event)
            deduped_dispatched_opportunities: list = []
            seen_bridge_keys: set[str] = set()
            for opp in dispatched_opportunities:
                stable_id = str(getattr(opp, "stable_id", "") or "").strip()
                opportunity_id = str(getattr(opp, "id", "") or "").strip()
                strategy = str(getattr(opp, "strategy", "") or "").strip()
                key = f"{strategy}:{stable_id or opportunity_id}"
                if not key or key in seen_bridge_keys:
                    continue
                seen_bridge_keys.add(key)
                deduped_dispatched_opportunities.append(opp)
            async with AsyncSessionLocal() as session:
                emitted = await bridge_opportunities_to_signals(
                    session,
                    deduped_dispatched_opportunities,
                    source="weather",
                    sweep_missing=True,
                    refresh_prices=False,
                )
            try:
                async with AsyncSessionLocal() as session:
                    _, existing_status = await shared_state.read_weather_snapshot(session)
                    stats_payload = dict(existing_status.get("stats") or {})
                    cycle_stats = result.get("stats")
                    if isinstance(cycle_stats, dict):
                        stats_payload.update(cycle_stats)
                    stats_payload["strategy_opportunities"] = len(deduped_dispatched_opportunities)
                    stats_payload["signals_emitted_last_run"] = int(emitted)
                    await shared_state.write_weather_snapshot(
                        session,
                        opportunities=sorted(
                            deduped_dispatched_opportunities,
                            key=lambda opp: float(getattr(opp, "roi_percent", 0.0) or 0.0),
                            reverse=True,
                        ),
                        status={
                            "running": True,
                            "enabled": enabled and not paused,
                            "interval_seconds": interval,
                            "last_scan": datetime.now(timezone.utc).isoformat(),
                            "current_activity": (
                                "Weather strategy cycle complete: "
                                f"{len(deduped_dispatched_opportunities)} opportunities from {len(intent_dicts)} intents."
                            ),
                        },
                        stats=stats_payload,
                    )
            except Exception as snapshot_exc:
                logger.warning("Weather strategy snapshot update failed: %s", snapshot_exc)
            next_scheduled_run_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=interval)
            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    "weather",
                    running=True,
                    enabled=enabled and not paused,
                    current_activity="Idle - waiting for next weather workflow cycle.",
                    interval_seconds=interval,
                    last_run_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    last_error=None,
                    stats={
                        "pending_intents": len(pending_rows),
                        "signals_emitted_last_run": int(emitted),
                        "strategy_opportunities": len(deduped_dispatched_opportunities),
                        "cycle_result": result,
                    },
                )
            logger.info(
                "Weather cycle complete",
                extra={
                    "markets": result.get("markets"),
                    "opportunities": result.get("opportunities"),
                    "intents": result.get("intents"),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Weather workflow cycle failed: %s", exc)
            try:
                async with AsyncSessionLocal() as session:
                    try:
                        existing, status = await shared_state.read_weather_snapshot(session)
                    except Exception:
                        existing, status = [], {}
                    await shared_state.write_weather_snapshot(
                        session,
                        opportunities=existing,
                        status={
                            "running": True,
                            "enabled": not paused,
                            "interval_seconds": interval,
                            "last_scan": datetime.now(timezone.utc).isoformat(),
                            "current_activity": f"Last weather scan error: {exc}",
                        },
                        stats=status.get("stats") or {},
                    )
                    try:
                        pending = len(
                            await shared_state.list_weather_intents(session, status_filter="pending", limit=2000)
                        )
                    except Exception:
                        pending = 0
                    await write_worker_snapshot(
                        session,
                        "weather",
                        running=True,
                        enabled=enabled and not paused,
                        current_activity=f"Last weather scan error: {exc}",
                        interval_seconds=interval,
                        last_run_at=datetime.now(timezone.utc).replace(tzinfo=None),
                        last_error=str(exc),
                        stats={
                            "pending_intents": int(pending),
                            "signals_emitted_last_run": 0,
                        },
                    )
            except Exception:
                pass

        await asyncio.sleep(min(10, interval))


async def start_loop() -> None:
    """Run the weather worker loop (called from API process lifespan)."""
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Weather worker shutting down")
