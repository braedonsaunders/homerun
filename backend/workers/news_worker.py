"""News worker: runs independent news workflow and writes DB snapshot.

Run from backend dir:
  python -m workers.news_worker
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
from datetime import datetime, timedelta, timezone

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

from config import settings
from models.database import AsyncSessionLocal, init_database
from services.news import shared_state
from services.signal_bus import emit_news_intent_signals
from services.news.workflow_orchestrator import workflow_orchestrator
from services.worker_state import write_worker_snapshot

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("news_worker")

_IDLE_SLEEP_SECONDS = 5


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
    logger.info("News worker started")
    owner = f"{socket.gethostname()}:{os.getpid()}"

    try:
        from services.ai import initialize_ai

        llm_manager = await initialize_ai()
        logger.info(
            "AI initialized in news worker (available=%s)", llm_manager.is_available()
        )
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
            enabled = bool(control.get("is_enabled", True)) and bool(
                wf_settings.get("enabled", True)
            )
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
        async with AsyncSessionLocal() as session:
            control = await shared_state.read_news_control(session)
            wf_settings = await shared_state.get_news_settings(session)

        interval = _interval_from_control_and_settings(control, wf_settings)
        paused = bool(control.get("is_paused", False))
        requested = control.get("requested_scan_at") is not None
        enabled = bool(control.get("is_enabled", True)) and bool(
            wf_settings.get("enabled", True)
        )
        auto_run = bool(wf_settings.get("auto_run", True))
        now = datetime.now(timezone.utc)

        try:
            async with AsyncSessionLocal() as session:
                await shared_state.expire_stale_news_intents(
                    session,
                    max_age_minutes=int(
                        max(
                            1,
                            wf_settings.get("auto_trader_max_age_minutes", 120) or 120,
                        )
                    ),
                )
        except Exception as exc:
            logger.debug("News intent expiry pass failed: %s", exc)

        should_run_scheduled = (
            enabled
            and auto_run
            and not paused
            and (next_scheduled_run_at is None or now >= next_scheduled_run_at)
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
                            "next_scan": next_scan_at.isoformat()
                            if next_scan_at
                            else None,
                            "current_activity": (
                                "Paused"
                                if paused
                                else "Idle - waiting for next news workflow cycle."
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
                        current_activity=(
                            "Paused"
                            if paused
                            else "Idle - waiting for next news workflow cycle."
                        ),
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

            async with AsyncSessionLocal() as session:
                result = await workflow_orchestrator.run_cycle(session)

            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            next_scheduled_run_at = completed_at + timedelta(seconds=interval)

            async with AsyncSessionLocal() as session:
                await shared_state.clear_news_scan_request(session)
                expired = await shared_state.expire_stale_news_intents(
                    session,
                    max_age_minutes=int(
                        max(
                            1,
                            wf_settings.get("auto_trader_max_age_minutes", 120) or 120,
                        )
                    ),
                )
                pending = await shared_state.count_pending_news_intents(session)
                cycle_stats = dict(result.get("stats") or {})
                cycle_stats["pending_intents"] = pending
                cycle_stats["expired_intents"] = expired
                cycle_stats["budget_skip_count"] = int(
                    cycle_stats.get("llm_calls_skipped", 0) or 0
                )
                await shared_state.write_news_snapshot(
                    session,
                    status={
                        "running": True,
                        "enabled": enabled,
                        "interval_seconds": interval,
                        "last_scan": completed_at.isoformat(),
                        "next_scan": (
                            next_scheduled_run_at.isoformat()
                            if enabled and auto_run and not paused
                            else None
                        ),
                        "current_activity": "Idle - waiting for next news workflow cycle.",
                        "last_error": result.get("error")
                        if result.get("status") == "error"
                        else None,
                        "degraded_mode": bool(result.get("degraded_mode", False)),
                        "budget_remaining": result.get("budget_remaining"),
                    },
                    stats=cycle_stats,
                )
                pending_rows = await shared_state.list_news_intents(
                    session, status_filter="pending", limit=2000
                )
                emitted = await emit_news_intent_signals(
                    session,
                    pending_rows,
                    max_age_minutes=int(
                        max(
                            1,
                            wf_settings.get("auto_trader_max_age_minutes", 120) or 120,
                        )
                    ),
                )
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
            logger.exception("News workflow cycle failed: %s", exc)
            next_scheduled_run_at = datetime.now(timezone.utc).replace(
                microsecond=0
            ) + timedelta(seconds=interval)
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


async def main() -> None:
    await init_database()
    logger.info("Database initialized")
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("News worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
