"""News worker: runs independent news workflow and writes DB snapshot.

Run from backend dir:
  python -m workers.news_worker
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from config import settings
from models.database import AsyncSessionLocal, recover_pool
from services.news import shared_state
from services.data_events import DataEvent
from services.event_dispatcher import event_dispatcher
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_signal_bridge import bridge_opportunities_to_signals
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.news.workflow_orchestrator import workflow_orchestrator
from services.worker_state import write_worker_snapshot
from utils.logger import setup_logging

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("news_worker")

_IDLE_SLEEP_SECONDS = 5
_RUN_CYCLE_DB_RETRY_ATTEMPTS = 2
_RUN_CYCLE_DB_RETRY_BASE_DELAY_SECONDS = 0.25
_DB_POOL_RECOVERY_COOLDOWN_SECONDS = 2.0
_last_pool_recovery_at_monotonic = 0.0

_DB_DISCONNECT_MARKERS = (
    "connection is closed",
    "underlying connection is closed",
    "connection has been closed",
    "closed the connection unexpectedly",
    "terminating connection",
    "connection reset by peer",
    "broken pipe",
    "connection was closed",
    "connectiondoesnotexist",
    "closed in the middle of operation",
    "transaction.rollback(): the underlying connection is closed",
    "another operation",
    "cannot switch to state",
)


def _is_db_disconnect_error(exc: Exception) -> bool:
    exc_module = type(exc).__module__ or ""
    is_db_exc = isinstance(exc, (DBAPIError, InterfaceError, OperationalError)) or exc_module.startswith("asyncpg")
    if not is_db_exc:
        return False
    full_msg = str(exc).lower()
    return any(marker in full_msg for marker in _DB_DISCONNECT_MARKERS)


async def emit_news_intent_signals(session, opportunities: list) -> int:
    return await bridge_opportunities_to_signals(
        session,
        opportunities,
        source="news",
        sweep_missing=True,
        refresh_prices=False,
    )


def _interval_from_control_and_settings(control: dict, wf_settings: dict) -> int:
    return int(
        max(
            30,
            min(
                3600,
                control.get("scan_interval_seconds")
                or wf_settings.get("scan_interval_seconds")
                or getattr(settings, "NEWS_SCAN_INTERVAL_SECONDS", 120),
            ),
        )
    )


def _next_scan_for_state(
    now: datetime,
    interval: int,
    enabled: bool,
    auto_run: bool,
    paused: bool,
    next_scheduled_run_at: datetime | None,
) -> datetime | None:
    if not enabled or not auto_run or paused:
        return None
    if next_scheduled_run_at is not None:
        return next_scheduled_run_at
    return now.replace(microsecond=0) + timedelta(seconds=interval)


async def _run_loop() -> None:
    global _last_pool_recovery_at_monotonic
    logger.info("News worker started")
    owner = f"{socket.gethostname()}:{os.getpid()}"

    try:
        async with AsyncSessionLocal() as session:
            await ensure_all_strategies_seeded(session)
            await refresh_strategy_runtime_if_needed(
                session,
                source_keys=["news"],
                force=True,
            )
    except Exception as exc:
        logger.warning("News worker strategy startup sync failed: %s", exc)

    try:
        from services.ai import initialize_ai

        llm_manager = await initialize_ai()
        logger.info("AI initialized in news worker (available=%s)", llm_manager.is_available())
    except Exception as exc:
        logger.warning("AI init in news worker failed (fallback mode): %s", exc)

    # Load persisted articles into worker-local memory cache.
    try:
        from services.news.feed_service import news_feed_service

        await news_feed_service.load_from_db()
    except Exception as exc:
        logger.warning("News feed preload failed (continuing): %s", exc)

    next_scheduled_run_at: datetime | None = None

    # Ensure initial snapshot exists.
    try:
        async with AsyncSessionLocal() as session:
            control = await shared_state.read_news_control(session)
            wf_settings = await shared_state.get_news_settings(session)
            interval = _interval_from_control_and_settings(control, wf_settings)
            paused = bool(control.get("is_paused", False))
            enabled = bool(control.get("is_enabled", True)) and bool(wf_settings.get("enabled", True))
            auto_run = bool(wf_settings.get("auto_run", True))
            next_scan_at = _next_scan_for_state(
                now=datetime.now(timezone.utc),
                interval=interval,
                enabled=enabled,
                auto_run=auto_run,
                paused=paused,
                next_scheduled_run_at=next_scheduled_run_at,
            )
            pending = await shared_state.count_pending_news_intents(session)
            await shared_state.write_news_snapshot(
                session,
                status={
                    "running": True,
                    "enabled": enabled,
                    "interval_seconds": interval,
                    "next_scan": next_scan_at.isoformat() if next_scan_at else None,
                    "current_activity": "News worker started; first cycle pending.",
                    "last_error": None,
                    "degraded_mode": False,
                },
                stats={"pending_intents": pending},
            )
            await write_worker_snapshot(
                session,
                "news",
                running=True,
                enabled=enabled and not paused,
                current_activity="News worker started; first cycle pending.",
                interval_seconds=interval,
                last_run_at=None,
                last_error=None,
                stats={"pending_intents": int(pending), "signals_emitted_last_run": 0},
            )
    except Exception:
        pass

    while True:
        try:
            async with AsyncSessionLocal() as session:
                control = await shared_state.read_news_control(session)
                wf_settings = await shared_state.get_news_settings(session)
                try:
                    await refresh_strategy_runtime_if_needed(
                        session,
                        source_keys=["news"],
                    )
                except Exception as exc:
                    logger.warning("News worker strategy refresh check failed: %s", exc)
        except Exception as exc:
            if _is_db_disconnect_error(exc):
                logger.warning("News workflow DB connection dropped; retrying cycle: %s", exc)
                now_monotonic = time.monotonic()
                if now_monotonic - _last_pool_recovery_at_monotonic >= _DB_POOL_RECOVERY_COOLDOWN_SECONDS:
                    try:
                        await recover_pool()
                        _last_pool_recovery_at_monotonic = time.monotonic()
                        logger.warning("Recovered DB pool after news workflow disconnect")
                    except Exception as pool_exc:
                        logger.warning("News workflow DB pool recovery failed: %s", pool_exc)
                await asyncio.sleep(_RUN_CYCLE_DB_RETRY_BASE_DELAY_SECONDS)
                continue
            raise

        interval = _interval_from_control_and_settings(control, wf_settings)
        paused = bool(control.get("is_paused", False))
        requested = control.get("requested_scan_at") is not None
        enabled = bool(control.get("is_enabled", True)) and bool(wf_settings.get("enabled", True))
        auto_run = bool(wf_settings.get("auto_run", True))
        now = datetime.now(timezone.utc)

        try:
            async with AsyncSessionLocal() as session:
                await shared_state.expire_stale_news_intents(
                    session,
                    max_age_minutes=int(
                        max(
                            1,
                            wf_settings.get("orchestrator_max_age_minutes", 120) or 120,
                        )
                    ),
                )
        except Exception as exc:
            logger.debug("News intent expiry pass failed: %s", exc)

        should_run_scheduled = (
            enabled and auto_run and not paused and (next_scheduled_run_at is None or now >= next_scheduled_run_at)
        )
        should_run = requested or should_run_scheduled

        if not should_run:
            next_scan_at = _next_scan_for_state(
                now=now,
                interval=interval,
                enabled=enabled,
                auto_run=auto_run,
                paused=paused,
                next_scheduled_run_at=next_scheduled_run_at,
            )
            if next_scheduled_run_at is None and next_scan_at is not None:
                next_scheduled_run_at = next_scan_at

            try:
                async with AsyncSessionLocal() as session:
                    pending = await shared_state.count_pending_news_intents(session)
                    await shared_state.write_news_snapshot(
                        session,
                        status={
                            "running": True,
                            "enabled": enabled,
                            "interval_seconds": interval,
                            "next_scan": next_scan_at.isoformat() if next_scan_at else None,
                            "current_activity": (
                                "Paused" if paused else "Idle - waiting for next news workflow cycle."
                            ),
                            "degraded_mode": False,
                        },
                        stats={"pending_intents": pending},
                    )
                    await write_worker_snapshot(
                        session,
                        "news",
                        running=True,
                        enabled=enabled and not paused,
                        current_activity=("Paused" if paused else "Idle - waiting for next news workflow cycle."),
                        interval_seconds=interval,
                        last_run_at=None,
                        last_error=None,
                        stats={"pending_intents": int(pending), "signals_emitted_last_run": 0},
                    )
            except Exception:
                pass

            await asyncio.sleep(min(_IDLE_SLEEP_SECONDS, interval))
            continue

        acquired = False
        lease_ttl_seconds = int(max(300, interval * 4))
        try:
            async with AsyncSessionLocal() as session:
                acquired = await shared_state.try_acquire_news_lease(
                    session,
                    owner=owner,
                    ttl_seconds=lease_ttl_seconds,
                )
        except Exception as exc:
            logger.debug("Failed acquiring news worker lease: %s", exc)

        if not acquired:
            await asyncio.sleep(min(_IDLE_SLEEP_SECONDS, interval))
            continue

        try:
            async with AsyncSessionLocal() as session:
                pending = await shared_state.count_pending_news_intents(session)
                await shared_state.write_news_snapshot(
                    session,
                    status={
                        "running": True,
                        "enabled": enabled,
                        "interval_seconds": interval,
                        "next_scan": None,
                        "current_activity": "Running news workflow cycle...",
                        "last_error": None,
                    },
                    stats={"pending_intents": pending},
                )

            result: dict = {}
            for attempt in range(_RUN_CYCLE_DB_RETRY_ATTEMPTS):
                try:
                    async with AsyncSessionLocal() as session:
                        result = await workflow_orchestrator.run_cycle(session)
                    break
                except Exception as exc:
                    is_disconnect = _is_db_disconnect_error(exc)
                    is_last_attempt = attempt >= _RUN_CYCLE_DB_RETRY_ATTEMPTS - 1
                    if not is_disconnect or is_last_attempt:
                        raise
                    logger.warning("News workflow DB connection dropped; retrying cycle: %s", exc)
                    now_monotonic = time.monotonic()
                    if now_monotonic - _last_pool_recovery_at_monotonic >= _DB_POOL_RECOVERY_COOLDOWN_SECONDS:
                        try:
                            await recover_pool()
                            _last_pool_recovery_at_monotonic = time.monotonic()
                            logger.warning("Recovered DB pool after news workflow disconnect")
                        except Exception as pool_exc:
                            logger.warning("News workflow DB pool recovery failed: %s", pool_exc)
                    await asyncio.sleep(_RUN_CYCLE_DB_RETRY_BASE_DELAY_SECONDS * (attempt + 1))

            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            next_scheduled_run_at = completed_at + timedelta(seconds=interval)

            async with AsyncSessionLocal() as session:
                await shared_state.clear_news_scan_request(session)
                expired = await shared_state.expire_stale_news_intents(
                    session,
                    max_age_minutes=int(
                        max(
                            1,
                            wf_settings.get("orchestrator_max_age_minutes", 120) or 120,
                        )
                    ),
                )
                pending = await shared_state.count_pending_news_intents(session)
                cycle_stats = dict(result.get("stats") or {})
                cycle_stats["pending_intents"] = pending
                cycle_stats["expired_intents"] = expired
                cycle_stats["budget_skip_count"] = int(cycle_stats.get("llm_calls_skipped", 0) or 0)
                await shared_state.write_news_snapshot(
                    session,
                    status={
                        "running": True,
                        "enabled": enabled,
                        "interval_seconds": interval,
                        "last_scan": completed_at.isoformat(),
                        "next_scan": (
                            next_scheduled_run_at.isoformat() if enabled and auto_run and not paused else None
                        ),
                        "current_activity": "Idle - waiting for next news workflow cycle.",
                        "last_error": result.get("error") if result.get("status") == "error" else None,
                        "degraded_mode": bool(result.get("degraded_mode", False)),
                        "budget_remaining": result.get("budget_remaining"),
                    },
                    stats=cycle_stats,
                )
                pending_rows = await shared_state.list_news_intents(session, status_filter="pending", limit=2000)
                finding_rows = await shared_state.list_news_findings(
                    session,
                    limit=2000,
                    max_age_minutes=int(
                        max(
                            1,
                            wf_settings.get("orchestrator_max_age_minutes", 120) or 120,
                        )
                    ),
                )

            # Serialize news intents to dicts for strategy consumption
            intent_dicts = []
            for row in pending_rows:
                intent_dict = {
                    "id": row.id,
                    "market_id": row.market_id,
                    "market_question": row.market_question,
                    "direction": getattr(row, "direction", None),
                    "entry_price": getattr(row, "entry_price", None),
                    "edge_percent": getattr(row, "edge_percent", None),
                    "confidence": getattr(row, "confidence", None),
                    "status": row.status,
                    "created_at": row.created_at,
                }
                metadata = getattr(row, "metadata_json", None) or {}
                if isinstance(metadata, dict):
                    intent_dict.update(metadata)
                intent_dicts.append(intent_dict)

            finding_dicts = []
            for row in finding_rows:
                finding_dicts.append(
                    {
                        "id": row.id,
                        "article_id": row.article_id,
                        "market_id": row.market_id,
                        "article_title": row.article_title,
                        "article_source": row.article_source,
                        "article_url": row.article_url,
                        "market_question": row.market_question,
                        "market_price": row.market_price,
                        "model_probability": row.model_probability,
                        "edge_percent": row.edge_percent,
                        "direction": row.direction,
                        "confidence": row.confidence,
                        "reasoning": row.reasoning,
                        "actionable": row.actionable,
                        "signal_key": row.signal_key,
                        "cache_key": row.cache_key,
                        "event_graph": row.event_graph if isinstance(row.event_graph, dict) else {},
                        "evidence": row.evidence if isinstance(row.evidence, dict) else {},
                        "created_at": row.created_at,
                    }
                )

            news_event = DataEvent(
                event_type="news_update",
                source="news_worker",
                timestamp=datetime.now(timezone.utc),
                payload={"intents": intent_dicts, "findings": finding_dicts},
            )
            opportunities = await event_dispatcher.dispatch(news_event)
            async with AsyncSessionLocal() as session:
                emitted = await emit_news_intent_signals(session, opportunities)
                await write_worker_snapshot(
                    session,
                    "news",
                    running=True,
                    enabled=enabled and not paused,
                    current_activity="Idle - waiting for next news workflow cycle.",
                    interval_seconds=interval,
                    last_run_at=completed_at.replace(tzinfo=None),
                    last_error=result.get("error") if result.get("status") == "error" else None,
                    stats={
                        "pending_intents": int(pending),
                        "expired_intents": int(expired),
                        "signals_emitted_last_run": int(emitted),
                        "cycle_stats": cycle_stats,
                    },
                )

            logger.info(
                "News cycle complete",
                extra={
                    "status": result.get("status"),
                    "findings": result.get("findings"),
                    "intents": result.get("intents"),
                    "degraded_mode": result.get("degraded_mode", False),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if _is_db_disconnect_error(exc):
                logger.warning("News workflow cycle hit transient DB disconnect (will retry): %s", exc)
                now_monotonic = time.monotonic()
                if now_monotonic - _last_pool_recovery_at_monotonic >= _DB_POOL_RECOVERY_COOLDOWN_SECONDS:
                    try:
                        await recover_pool()
                        _last_pool_recovery_at_monotonic = time.monotonic()
                        logger.warning("Recovered DB pool after news workflow disconnect")
                    except Exception as pool_exc:
                        logger.warning("News workflow DB pool recovery failed: %s", pool_exc)
            else:
                logger.exception("News workflow cycle failed: %s", exc)
            next_scheduled_run_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=interval)
            try:
                async with AsyncSessionLocal() as session:
                    await shared_state.clear_news_scan_request(session)
                    pending = await shared_state.count_pending_news_intents(session)
                    await shared_state.write_news_snapshot(
                        session,
                        status={
                            "running": True,
                            "enabled": enabled,
                            "interval_seconds": interval,
                            "next_scan": next_scheduled_run_at.isoformat(),
                            "current_activity": f"Last news cycle error: {exc}",
                            "last_error": str(exc),
                            "degraded_mode": True,
                        },
                        stats={"pending_intents": pending},
                    )
                    await write_worker_snapshot(
                        session,
                        "news",
                        running=True,
                        enabled=enabled and not paused,
                        current_activity=f"Last news cycle error: {exc}",
                        interval_seconds=interval,
                        last_run_at=datetime.now(timezone.utc).replace(tzinfo=None),
                        last_error=str(exc),
                        stats={"pending_intents": int(pending), "signals_emitted_last_run": 0},
                    )
            except Exception:
                pass
        finally:
            try:
                async with AsyncSessionLocal() as session:
                    await shared_state.release_news_lease(session, owner)
            except Exception:
                pass

        await asyncio.sleep(min(_IDLE_SLEEP_SECONDS, interval))


async def start_loop() -> None:
    """Run the news worker loop (called from API process lifespan)."""
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("News worker shutting down")
