"""Dedicated live reconciliation worker for trader order/position state."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm.exc import StaleDataError

from models.database import AsyncSessionLocal, init_database, recover_pool
from services.event_bus import event_bus
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.live_execution_service import live_execution_service
from services.trader_orchestrator.position_lifecycle import reconcile_live_positions
from services.trader_orchestrator_state import (
    create_trader_event,
    list_traders,
    reconcile_live_provider_orders,
    sync_trader_position_inventory,
)
from services.wallet_ws_monitor import wallet_ws_monitor
from services.worker_state import (
    _is_retryable_db_error,
    clear_worker_run_request,
    read_worker_control,
    write_worker_snapshot,
)
from utils.logger import setup_logging
from utils.utcnow import utcnow

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("trader_reconciliation_worker")

WORKER_NAME = "trader_reconciliation"
DEFAULT_INTERVAL_SECONDS = 1
_IDLE_SLEEP_SECONDS = 1
_MAX_CONSECUTIVE_DB_FAILURES = 3
_CONTROL_REFRESH_SECONDS = 1.0
_ACTIVE_POSITION_TICK_SECONDS = 0.25
_EVENT_QUEUE_MAXSIZE = 4096
_WALLET_MONITOR_REFRESH_SECONDS = 15.0
_TRADER_RECONCILE_ATTEMPTS = 3
_TRADER_RECONCILE_RETRY_BASE_DELAY_SECONDS = 0.05
_TRADER_RECONCILE_RETRY_MAX_DELAY_SECONDS = 0.3
_RECONCILE_TRIGGER_EVENTS = frozenset(
    {
        "trader_order",
        "execution_order",
        "wallet_trade",
        "trade_signal_batch",
        "trade_signal_emission",
        "crypto_markets_update",
        "opportunities_update",
        "events_update",
        "weather_update",
        "news_workflow_update",
        "signals_update",
    }
)


def _reconcile_retry_delay_seconds(attempt: int) -> float:
    return min(
        _TRADER_RECONCILE_RETRY_BASE_DELAY_SECONDS * (2**attempt),
        _TRADER_RECONCILE_RETRY_MAX_DELAY_SECONDS,
    )


def _default_strategy_params(trader: dict[str, Any]) -> dict[str, Any]:
    source_configs = trader.get("source_configs")
    if not isinstance(source_configs, list):
        return {}
    for row in source_configs:
        if not isinstance(row, dict):
            continue
        params = row.get("strategy_params")
        if isinstance(params, dict):
            return dict(params)
    return {}


def _empty_cycle_summary() -> dict[str, Any]:
    return {
        "traders_seen": 0,
        "traders_processed": 0,
        "provider_unavailable_traders": 0,
        "provider_active_seen": 0,
        "provider_status_changes": 0,
        "provider_updates": 0,
        "provider_notional_updates": 0,
        "provider_price_updates": 0,
        "positions_would_close": 0,
        "positions_closed": 0,
        "inventory_open_positions": 0,
        "inventory_updates": 0,
        "inventory_inserts": 0,
        "inventory_closures": 0,
        "failures": 0,
    }


def _wallet_monitor_snapshot_stats() -> dict[str, Any]:
    status = wallet_ws_monitor.get_status()
    raw_stats = status.get("stats")
    stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}
    return {
        "wallet_monitor_running": bool(status.get("running")),
        "wallet_monitor_ws_connected": bool(status.get("ws_connected")),
        "wallet_monitor_fallback_polling": bool(status.get("fallback_polling")),
        "wallet_monitor_tracked_wallets": int(status.get("tracked_wallets") or 0),
        "wallet_monitor_last_event_detected_at": str(stats.get("last_event_detected_at") or ""),
        "wallet_monitor_last_block_seen_at": str(stats.get("last_block_seen_at") or ""),
        "wallet_monitor_last_block_processed_at": str(stats.get("last_block_processed_at") or ""),
        "wallet_monitor_last_fallback_poll_at": str(stats.get("last_fallback_poll_at") or ""),
        "wallet_monitor_events_detected_total": int(stats.get("events_detected") or 0),
        "wallet_monitor_blocks_processed_total": int(stats.get("blocks_processed") or 0),
        "wallet_monitor_errors_total": int(stats.get("errors") or 0),
    }


async def _sync_live_wallet_monitor_source(current_wallet: str) -> str:
    execution_wallet = ""
    try:
        if await live_execution_service.ensure_initialized():
            execution_wallet = str(live_execution_service.get_execution_wallet_address() or "").strip().lower()
    except Exception as exc:
        logger.warning("Failed to initialize trading service for wallet WS monitor sync", exc_info=exc)

    if execution_wallet != current_wallet:
        wallet_ws_monitor.set_wallets_for_source(WORKER_NAME, [execution_wallet] if execution_wallet else [])
        current_wallet = execution_wallet

    if execution_wallet:
        try:
            await wallet_ws_monitor.start()
        except Exception as exc:
            logger.warning("Failed to start wallet WS monitor for reconciliation worker", exc_info=exc)

    return current_wallet


async def _reconcile_live_state_for_trader(
    trader: dict[str, Any],
    *,
    provider_pass: bool,
) -> dict[str, Any]:
    trader_id = str(trader.get("id") or "").strip()
    if not trader_id:
        return {
            "provider": {"active_seen": 0, "status_changes": 0, "updated_orders": 0},
            "lifecycle": {"would_close": 0, "closed": 0},
            "inventory": {"open_positions": 0, "updates": 0, "inserts": 0, "closures": 0},
        }

    provider_result: dict[str, Any] = {
        "provider_ready": True,
        "active_seen": 0,
        "status_changes": 0,
        "updated_orders": 0,
        "notional_updates": 0,
        "price_updates": 0,
    }
    if provider_pass:
        async with AsyncSessionLocal() as session:
            provider_result = await reconcile_live_provider_orders(
                session,
                trader_id=trader_id,
                commit=True,
                broadcast=True,
            )

    active_seen = int(provider_result.get("active_seen", 0) or 0)
    lifecycle_result: dict[str, Any] = {"would_close": 0, "closed": 0}
    trader_params = _default_strategy_params(trader)
    if (not provider_pass) or active_seen > 0:
        async with AsyncSessionLocal() as session:
            lifecycle_result = await reconcile_live_positions(
                session,
                trader_id=trader_id,
                trader_params=trader_params,
                dry_run=False,
                reason="reconciliation_worker",
            )
    async with AsyncSessionLocal() as session:
        inventory_result = await sync_trader_position_inventory(
            session,
            trader_id=trader_id,
            mode="live",
            commit=True,
        )

    return {
        "provider": provider_result,
        "lifecycle": lifecycle_result,
        "inventory": inventory_result,
    }


async def _run_reconciliation_cycle(
    *,
    reason: str,
    emit_event: bool,
    provider_pass: bool,
) -> dict[str, Any]:
    summary = _empty_cycle_summary()
    async with AsyncSessionLocal() as session:
        traders = await list_traders(session)

    summary["traders_seen"] = len(traders)
    for trader in traders:
        trader_id = str(trader.get("id") or "").strip()
        if not trader_id:
            continue
        result: dict[str, Any] | None = None
        for attempt in range(_TRADER_RECONCILE_ATTEMPTS):
            try:
                result = await _reconcile_live_state_for_trader(trader, provider_pass=provider_pass)
                break
            except StaleDataError as exc:
                if attempt < _TRADER_RECONCILE_ATTEMPTS - 1:
                    logger.warning(
                        "Live reconciliation stale-row conflict for trader=%s reason=%s attempt=%d/%d; retrying",
                        trader_id,
                        reason,
                        attempt + 1,
                        _TRADER_RECONCILE_ATTEMPTS,
                        exc_info=exc,
                    )
                    await asyncio.sleep(_reconcile_retry_delay_seconds(attempt))
                    continue
                summary["failures"] = int(summary["failures"]) + 1
                logger.warning(
                    "Live reconciliation failed for trader=%s reason=%s error_type=%s retryable_db=%s",
                    trader_id,
                    reason,
                    type(exc).__name__,
                    False,
                    exc_info=exc,
                )
            except Exception as exc:
                retryable_db = _is_retryable_db_error(exc)
                if retryable_db and attempt < _TRADER_RECONCILE_ATTEMPTS - 1:
                    logger.warning(
                        "Live reconciliation retrying trader=%s reason=%s attempt=%d/%d due retryable DB error (%s)",
                        trader_id,
                        reason,
                        attempt + 1,
                        _TRADER_RECONCILE_ATTEMPTS,
                        type(exc).__name__,
                        exc_info=exc,
                    )
                    await asyncio.sleep(_reconcile_retry_delay_seconds(attempt))
                    continue
                summary["failures"] = int(summary["failures"]) + 1
                logger.warning(
                    "Live reconciliation failed for trader=%s reason=%s error_type=%s retryable_db=%s",
                    trader_id,
                    reason,
                    type(exc).__name__,
                    retryable_db,
                    exc_info=exc,
                )
            break
        if result is None:
            continue

        provider = result.get("provider") or {}
        lifecycle = result.get("lifecycle") or {}
        inventory = result.get("inventory") or {}
        summary["traders_processed"] = int(summary["traders_processed"]) + 1
        if not bool(provider.get("provider_ready", True)):
            summary["provider_unavailable_traders"] = int(summary["provider_unavailable_traders"]) + 1
        summary["provider_active_seen"] = int(summary["provider_active_seen"]) + int(
            provider.get("active_seen", 0) or 0
        )
        summary["provider_status_changes"] = int(summary["provider_status_changes"]) + int(
            provider.get("status_changes", 0) or 0
        )
        summary["provider_updates"] = int(summary["provider_updates"]) + int(provider.get("updated_orders", 0) or 0)
        summary["provider_notional_updates"] = int(summary["provider_notional_updates"]) + int(
            provider.get("notional_updates", 0) or 0
        )
        summary["provider_price_updates"] = int(summary["provider_price_updates"]) + int(
            provider.get("price_updates", 0) or 0
        )
        summary["positions_would_close"] = int(summary["positions_would_close"]) + int(
            lifecycle.get("would_close", 0) or 0
        )
        summary["positions_closed"] = int(summary["positions_closed"]) + int(lifecycle.get("closed", 0) or 0)
        summary["inventory_open_positions"] = int(summary["inventory_open_positions"]) + int(
            inventory.get("open_positions", 0) or 0
        )
        summary["inventory_updates"] = int(summary["inventory_updates"]) + int(inventory.get("updates", 0) or 0)
        summary["inventory_inserts"] = int(summary["inventory_inserts"]) + int(inventory.get("inserts", 0) or 0)
        summary["inventory_closures"] = int(summary["inventory_closures"]) + int(inventory.get("closures", 0) or 0)

    if emit_event:
        async with AsyncSessionLocal() as session:
            await create_trader_event(
                session,
                trader_id=None,
                event_type="live_reconciliation_cycle",
                source="worker",
                message=(
                    f"Live reconciliation {reason}: "
                    f"processed={int(summary['traders_processed'])}/{int(summary['traders_seen'])}, "
                    f"provider_updates={int(summary['provider_updates'])}, "
                    f"positions_closed={int(summary['positions_closed'])}"
                ),
                payload={
                    "reason": reason,
                    "provider_pass": bool(provider_pass),
                    **summary,
                },
                commit=True,
            )
    return summary


async def run_worker_loop() -> None:
    logger.info("Starting trader reconciliation worker loop")

    try:
        async with AsyncSessionLocal() as session:
            await ensure_all_strategies_seeded(session)
            await refresh_strategy_runtime_if_needed(session, source_keys=None, force=True)
    except Exception as exc:
        logger.warning("Reconciliation strategy startup sync failed", exc_info=exc)

    consecutive_db_failures = 0
    wallet_monitor_wallet = ""
    wallet_monitor_refresh_at = 0.0
    wallet_monitor_wallet = await _sync_live_wallet_monitor_source(wallet_monitor_wallet)
    wallet_monitor_refresh_at = time.monotonic() + _WALLET_MONITOR_REFRESH_SECONDS

    startup_summary = _empty_cycle_summary()
    try:
        startup_summary = await _run_reconciliation_cycle(
            reason="startup",
            emit_event=True,
            provider_pass=True,
        )
    except Exception as exc:
        logger.warning("Startup full reconciliation failed", exc_info=exc)

    try:
        async with AsyncSessionLocal() as session:
            await write_worker_snapshot(
                session,
                WORKER_NAME,
                running=True,
                enabled=True,
                current_activity=(
                    f"Startup full sync processed={int(startup_summary['traders_processed'])}/"
                    f"{int(startup_summary['traders_seen'])}"
                ),
                interval_seconds=DEFAULT_INTERVAL_SECONDS,
                last_run_at=utcnow(),
                stats={
                    **startup_summary,
                    **_wallet_monitor_snapshot_stats(),
                    "cycle_reason": "startup",
                    "provider_pass": True,
                },
            )
    except Exception as exc:
        logger.warning("Failed to persist startup reconciliation snapshot", exc_info=exc)

    interval_seconds = DEFAULT_INTERVAL_SECONDS
    is_enabled = True
    is_paused = False
    requested_run = False
    last_open_positions = int(startup_summary.get("inventory_open_positions", 0) or 0)
    last_provider_pass_at = time.monotonic()
    control_refresh_at = 0.0
    trigger_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_EVENT_QUEUE_MAXSIZE)

    async def _on_runtime_event(event_type: str, data: dict[str, Any]) -> None:
        _ = data
        if event_type not in _RECONCILE_TRIGGER_EVENTS:
            return
        try:
            trigger_queue.put_nowait(event_type)
        except asyncio.QueueFull:
            logger.warning("Reconciliation event queue full; dropping trigger", event_type=event_type)

    await event_bus.start()
    event_bus.subscribe("*", _on_runtime_event)

    try:
        while True:
            try:
                now_mono = time.monotonic()
                if now_mono >= control_refresh_at:
                    async with AsyncSessionLocal() as session:
                        control = await read_worker_control(
                            session,
                            WORKER_NAME,
                            default_interval=DEFAULT_INTERVAL_SECONDS,
                        )
                        try:
                            await refresh_strategy_runtime_if_needed(session, source_keys=None)
                        except Exception as exc:
                            logger.warning("Reconciliation strategy runtime refresh failed", exc_info=exc)

                        interval_seconds = max(1, int(control.get("interval_seconds") or DEFAULT_INTERVAL_SECONDS))
                        is_enabled = bool(control.get("is_enabled", True))
                        is_paused = bool(control.get("is_paused", False))
                        requested_run = bool(control.get("requested_run_at") is not None)
                        if requested_run:
                            await clear_worker_run_request(session, WORKER_NAME)

                        if not is_enabled or is_paused:
                            await write_worker_snapshot(
                                session,
                                WORKER_NAME,
                                running=False,
                                enabled=bool(is_enabled and not is_paused),
                                current_activity="Paused" if is_paused else "Disabled",
                                interval_seconds=interval_seconds,
                                stats={
                                    "requested_run": bool(requested_run),
                                    "open_positions_last_cycle": int(last_open_positions),
                                    **_wallet_monitor_snapshot_stats(),
                                },
                            )

                    control_refresh_at = now_mono + _CONTROL_REFRESH_SECONDS

                if now_mono >= wallet_monitor_refresh_at:
                    wallet_monitor_wallet = await _sync_live_wallet_monitor_source(wallet_monitor_wallet)
                    wallet_monitor_refresh_at = now_mono + _WALLET_MONITOR_REFRESH_SECONDS

                if not is_enabled or is_paused:
                    await asyncio.sleep(max(_IDLE_SLEEP_SECONDS, interval_seconds))
                    continue

                cycle_reason = "scheduled"
                provider_pass = True
                emit_event = False

                if requested_run:
                    cycle_reason = "requested"
                    provider_pass = True
                    emit_event = True
                    requested_run = False
                else:
                    wait_seconds = _ACTIVE_POSITION_TICK_SECONDS if last_open_positions > 0 else float(interval_seconds)
                    try:
                        event_type = await asyncio.wait_for(
                            trigger_queue.get(),
                            timeout=max(0.05, wait_seconds),
                        )
                        cycle_reason = f"event:{event_type}"
                        provider_pass = True
                    except asyncio.TimeoutError:
                        if last_open_positions > 0:
                            elapsed_provider_seconds = time.monotonic() - last_provider_pass_at
                            if elapsed_provider_seconds >= float(interval_seconds):
                                cycle_reason = "scheduled"
                                provider_pass = True
                            else:
                                cycle_reason = "position_tick"
                                provider_pass = False
                        else:
                            cycle_reason = "scheduled"
                            provider_pass = True

                cycle_summary = await _run_reconciliation_cycle(
                    reason=cycle_reason,
                    emit_event=emit_event,
                    provider_pass=provider_pass,
                )
                consecutive_db_failures = 0
                last_open_positions = int(cycle_summary.get("inventory_open_positions", 0) or 0)
                if provider_pass:
                    last_provider_pass_at = time.monotonic()

                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        WORKER_NAME,
                        running=True,
                        enabled=True,
                        current_activity=(
                            f"Reconciled {int(cycle_summary['traders_processed'])}/"
                            f"{int(cycle_summary['traders_seen'])} traders, "
                            f"provider_updates={int(cycle_summary['provider_updates'])}, "
                            f"positions_closed={int(cycle_summary['positions_closed'])}, "
                            f"provider_unavailable={int(cycle_summary['provider_unavailable_traders'])}"
                        ),
                        interval_seconds=interval_seconds,
                        last_run_at=utcnow(),
                        stats={
                            **cycle_summary,
                            "cycle_reason": cycle_reason,
                            "provider_pass": bool(provider_pass),
                            **_wallet_monitor_snapshot_stats(),
                        },
                    )
            except DBAPIError as exc:
                if not _is_retryable_db_error(exc):
                    raise
                consecutive_db_failures += 1
                logger.warning(
                    "Trader reconciliation cycle hit transient DB error (failure=%d)",
                    consecutive_db_failures,
                    exc_info=exc,
                )
                if consecutive_db_failures >= _MAX_CONSECUTIVE_DB_FAILURES:
                    await recover_pool()
                    consecutive_db_failures = 0
                    logger.warning("Recovered DB connection pool after trader reconciliation disconnects")
                control_refresh_at = 0.0
                wallet_monitor_refresh_at = 0.0
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)
            except Exception as exc:
                logger.exception("Trader reconciliation worker cycle failed: %s", exc)
                try:
                    async with AsyncSessionLocal() as session:
                        await write_worker_snapshot(
                            session,
                            WORKER_NAME,
                            running=False,
                            enabled=True,
                            current_activity="Worker error",
                            interval_seconds=interval_seconds,
                            last_error=str(exc),
                        )
                except Exception:
                    pass
                control_refresh_at = 0.0
                wallet_monitor_refresh_at = 0.0
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)
    finally:
        event_bus.unsubscribe("*", _on_runtime_event)


async def start_loop() -> None:
    try:
        await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Trader reconciliation worker shutting down")


async def main() -> None:
    await init_database()
    try:
        await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Trader reconciliation worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
