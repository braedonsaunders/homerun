"""Dedicated trader orchestrator worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any


from config import settings
from models.database import (
    AppSettings,
    AsyncSessionLocal,
    SimulationAccount,
    Trader,
    init_database,
)
from services.trader_orchestrator.order_manager import submit_order
from services.trader_orchestrator.live_market_context import (
    RuntimeTradeSignalView,
    build_live_signal_contexts,
)
from services.trader_orchestrator.position_lifecycle import reconcile_paper_positions
from services.trader_orchestrator.risk_manager import evaluate_risk
from services.trader_orchestrator.strategies.registry import get_strategy
from services.trader_orchestrator_state import (
    compute_orchestrator_metrics,
    create_trader_decision,
    create_trader_decision_checks,
    create_trader_event,
    create_trader_order,
    get_consecutive_loss_count,
    get_daily_realized_pnl,
    get_gross_exposure,
    get_last_resolved_loss_at,
    get_market_exposure,
    get_open_position_count_for_trader,
    list_traders,
    list_unconsumed_trade_signals,
    read_orchestrator_control,
    record_signal_consumption,
    sync_trader_position_inventory,
    write_orchestrator_snapshot,
)
from services.signal_bus import expire_stale_signals
from utils.secrets import decrypt_secret

logger = logging.getLogger("trader_orchestrator_worker")
_RESUME_POLICIES = {"resume_full", "manage_only", "flatten_then_start"}


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _parse_hhmm_utc(value: Any) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except Exception:
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute


def _is_within_trading_window_utc(metadata: dict[str, Any], now_utc: datetime) -> bool:
    window = metadata.get("trading_window_utc")
    if not isinstance(window, dict):
        return True

    start = _parse_hhmm_utc(window.get("start"))
    end = _parse_hhmm_utc(window.get("end"))
    if start is None or end is None:
        return True

    start_minutes = (start[0] * 60) + start[1]
    end_minutes = (end[0] * 60) + end[1]
    now_minutes = (now_utc.hour * 60) + now_utc.minute

    if start_minutes == end_minutes:
        return True
    if start_minutes < end_minutes:
        return start_minutes <= now_minutes < end_minutes
    return now_minutes >= start_minutes or now_minutes < end_minutes


def _supports_live_market_context(signal: Any) -> bool:
    """Only apply HTTP live-market enrichment to non-crypto signals."""
    source = str(getattr(signal, "source", "") or "").strip().lower()
    return source != "crypto"


def _normalize_resume_policy(value: Any) -> str:
    policy = str(value or "").strip().lower()
    if policy in _RESUME_POLICIES:
        return policy
    return "resume_full"


def _is_due(trader: dict[str, Any], now: datetime) -> bool:
    requested = _parse_iso(trader.get("requested_run_at"))
    if requested is not None:
        return True

    last_run = _parse_iso(trader.get("last_run_at"))
    interval = max(1, int(trader.get("interval_seconds") or 60))
    if last_run is None:
        return True
    return (now - last_run.astimezone(timezone.utc)).total_seconds() >= interval


def _checks_to_payload(checks: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for check in checks:
        out.append(
            {
                "check_key": str(getattr(check, "key", "check")),
                "check_label": str(getattr(check, "label", "Check")),
                "passed": bool(getattr(check, "passed", False)),
                "score": getattr(check, "score", None),
                "detail": getattr(check, "detail", None),
                "payload": getattr(check, "payload", {}) or {},
            }
        )
    return out


def _is_live_credentials_configured(app_settings: AppSettings | None) -> bool:
    if app_settings is None:
        return False

    polymarket_ready = bool(
        decrypt_secret(app_settings.polymarket_api_key)
        and decrypt_secret(app_settings.polymarket_api_secret)
        and decrypt_secret(app_settings.polymarket_api_passphrase)
    )
    kalshi_ready = bool(
        (app_settings.kalshi_email or "").strip()
        and decrypt_secret(app_settings.kalshi_password)
        and decrypt_secret(app_settings.kalshi_api_key)
    )
    return polymarket_ready or kalshi_ready


async def _run_trader_once(
    trader: dict[str, Any],
    control: dict[str, Any],
    *,
    process_signals: bool = True,
) -> tuple[int, int]:
    decisions_written = 0
    orders_written = 0

    async with AsyncSessionLocal() as session:
        trader_id = str(trader["id"])
        strategy = get_strategy(str(trader.get("strategy_key") or ""))
        params = dict(trader.get("params") or {})
        risk_limits = dict(trader.get("risk_limits") or {})
        metadata = dict(trader.get("metadata") or {})
        run_mode = str(control.get("mode") or "paper").strip().lower()
        resume_policy = _normalize_resume_policy(metadata.get("resume_policy"))
        open_positions = 0

        if run_mode == "paper":
            force_flatten = resume_policy == "flatten_then_start"
            lifecycle_result = await reconcile_paper_positions(
                session,
                trader_id=trader_id,
                trader_params=params,
                dry_run=False,
                force_mark_to_market=force_flatten,
                reason="worker_flatten_then_start" if force_flatten else "worker_lifecycle",
            )
            closed_positions = int(lifecycle_result.get("closed", 0))
            if closed_positions > 0:
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="paper_positions_closed",
                    source="worker",
                    message=f"Closed {closed_positions} paper position(s)",
                    payload={
                        "matched": lifecycle_result.get("matched"),
                        "closed": closed_positions,
                        "held": lifecycle_result.get("held"),
                        "skipped": lifecycle_result.get("skipped"),
                        "total_realized_pnl": lifecycle_result.get("total_realized_pnl"),
                        "by_status": lifecycle_result.get("by_status"),
                    },
                )
        await sync_trader_position_inventory(
            session,
            trader_id=trader_id,
            mode=run_mode,
        )
        open_positions = await get_open_position_count_for_trader(
            session,
            trader_id,
            mode=run_mode,
        )

        block_entries_reason = None
        if resume_policy == "manage_only":
            block_entries_reason = "Resume policy manage_only blocks new entries"
        elif resume_policy == "flatten_then_start" and open_positions > 0:
            if run_mode == "paper":
                block_entries_reason = (
                    f"Resume policy flatten_then_start waiting to flatten {open_positions} open paper position(s)"
                )
            else:
                block_entries_reason = (
                    f"Resume policy flatten_then_start blocked: {open_positions} open live position(s) require manual flattening"
                )

        effective_process_signals = bool(process_signals)
        if block_entries_reason is not None and process_signals:
            effective_process_signals = False
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="trader_resume_policy",
                severity="warn" if run_mode == "live" else "info",
                source="worker",
                message=block_entries_reason,
                payload={
                    "resume_policy": resume_policy,
                    "mode": run_mode,
                    "open_positions": open_positions,
                },
            )

        if not effective_process_signals:
            return 0, 0

        max_signals_per_cycle = int(max(1, min(500, _safe_int(params.get("max_signals_per_cycle"), 200))))
        scan_batch_size = int(
            max(
                1,
                min(
                    500,
                    _safe_int(params.get("scan_batch_size"), max_signals_per_cycle),
                ),
            )
        )
        query_limit = max(max_signals_per_cycle, scan_batch_size)
        raw_sources = trader.get("sources")
        sources = None if raw_sources is None else [str(item) for item in raw_sources]
        signals = await list_unconsumed_trade_signals(
            session,
            trader_id=trader_id,
            sources=sources,
            limit=query_limit,
        )
        if len(signals) > max_signals_per_cycle:
            signals = signals[:max_signals_per_cycle]

        control_settings = control.get("settings") or {}
        enable_live_market_context = bool(control_settings.get("enable_live_market_context", True))
        history_window_seconds = int(
            max(
                300,
                min(
                    21600,
                    _safe_int(
                        control_settings.get("live_market_history_window_seconds", 7200),
                        7200,
                    ),
                ),
            )
        )
        history_fidelity_seconds = int(
            max(
                30,
                min(
                    1800,
                    _safe_int(
                        control_settings.get("live_market_history_fidelity_seconds", 300),
                        300,
                    ),
                ),
            )
        )
        max_history_points = int(
            max(
                20,
                min(
                    240,
                    _safe_int(
                        control_settings.get("live_market_history_max_points", 120),
                        120,
                    ),
                ),
            )
        )

        now_utc = datetime.now(timezone.utc)
        trading_window_ok = _is_within_trading_window_utc(metadata, now_utc)
        global_limits = dict((control.get("settings") or {}).get("global_risk") or {})
        effective_risk_limits = dict(risk_limits)
        if "max_orders_per_cycle" not in effective_risk_limits:
            fallback_cycle_limit = _safe_int(global_limits.get("max_orders_per_cycle"), 50)
            if fallback_cycle_limit > 0:
                effective_risk_limits["max_orders_per_cycle"] = fallback_cycle_limit
        if "max_daily_loss_usd" not in effective_risk_limits:
            fallback_daily_loss = _safe_float(global_limits.get("max_daily_loss_usd"), 0.0) or 0.0
            if fallback_daily_loss > 0:
                effective_risk_limits["max_daily_loss_usd"] = fallback_daily_loss

        global_daily_pnl = await get_daily_realized_pnl(session, mode=run_mode)
        trader_daily_pnl = await get_daily_realized_pnl(session, trader_id=trader_id, mode=run_mode)
        trader_loss_streak = await get_consecutive_loss_count(
            session,
            trader_id=trader_id,
            mode=run_mode,
        )
        last_loss_at = await get_last_resolved_loss_at(
            session,
            trader_id=trader_id,
            mode=run_mode,
        )
        cooldown_seconds = max(0, _safe_int(effective_risk_limits.get("cooldown_seconds"), 0))
        cooldown_active = False
        cooldown_remaining_seconds = 0
        if cooldown_seconds > 0 and last_loss_at is not None and trader_loss_streak > 0:
            cooldown_until = last_loss_at + timedelta(seconds=cooldown_seconds)
            now_naive = now_utc.replace(tzinfo=None)
            if cooldown_until > now_naive:
                cooldown_active = True
                cooldown_remaining_seconds = int((cooldown_until - now_naive).total_seconds())

        live_contexts: dict[str, dict[str, Any]] = {}
        if enable_live_market_context and signals:
            context_candidates = [sig for sig in signals if _supports_live_market_context(sig)]
            try:
                live_contexts = await build_live_signal_contexts(
                    context_candidates,
                    history_window_seconds=history_window_seconds,
                    history_fidelity_seconds=history_fidelity_seconds,
                    max_history_points=max_history_points,
                )
            except Exception as exc:
                logger.warning("Live market context refresh failed: %s", exc)
                live_contexts = {}

        for signal in signals:
            live_context = live_contexts.get(str(signal.id), {})
            runtime_signal = RuntimeTradeSignalView(signal, live_context=live_context)
            decision_obj = strategy.evaluate(
                runtime_signal,
                {
                    "params": params,
                    "trader": trader,
                    "mode": control.get("mode", "paper"),
                    "live_market": live_context,
                },
            )
            checks_payload = _checks_to_payload(decision_obj.checks)

            if live_context:
                live_price = live_context.get("live_selected_price")
                checks_payload.append(
                    {
                        "check_key": "live_market_price",
                        "check_label": "Live market price",
                        "passed": live_price is not None,
                        "score": live_price,
                        "detail": (
                            "Using live selected-outcome midpoint"
                            if live_price is not None
                            else "Live selected-outcome midpoint unavailable"
                        ),
                        "payload": {
                            "selected_outcome": live_context.get("selected_outcome"),
                            "fetched_at": live_context.get("fetched_at"),
                        },
                    }
                )
                drift_pct = live_context.get("entry_price_delta_pct")
                drift_score = _safe_float(drift_pct)
                checks_payload.append(
                    {
                        "check_key": "live_entry_drift",
                        "check_label": "Entry drift from signal",
                        "passed": drift_score is None or abs(drift_score) <= 1000.0,
                        "score": drift_score,
                        "detail": (
                            f"drift={drift_score:.2f}%"
                            if drift_score is not None
                            else "Signal entry unavailable; drift skipped"
                        ),
                        "payload": {
                            "signal_entry_price": live_context.get("signal_entry_price"),
                            "live_selected_price": live_context.get("live_selected_price"),
                            "adverse_price_move": live_context.get("adverse_price_move"),
                        },
                    }
                )

            final_decision = decision_obj.decision
            final_reason = decision_obj.reason
            score = decision_obj.score
            size_usd = float(max(1.0, decision_obj.size_usd or 10.0))
            risk_snapshot = {}

            if final_decision == "selected" and not trading_window_ok:
                final_decision = "blocked"
                final_reason = "Outside configured trading window (UTC)"

            if final_decision == "selected":
                gross_exposure = await get_gross_exposure(session, mode=run_mode)
                market_exposure = await get_market_exposure(session, str(signal.market_id), mode=run_mode)
                risk_result = evaluate_risk(
                    size_usd=size_usd,
                    gross_exposure_usd=gross_exposure,
                    trader_open_positions=open_positions,
                    market_exposure_usd=market_exposure,
                    global_limits=global_limits,
                    trader_limits=effective_risk_limits,
                    global_daily_realized_pnl_usd=global_daily_pnl,
                    trader_daily_realized_pnl_usd=trader_daily_pnl,
                    trader_consecutive_losses=trader_loss_streak,
                    cycle_orders_placed=orders_written,
                    cooldown_active=cooldown_active,
                )
                risk_snapshot = {
                    "allowed": risk_result.allowed,
                    "reason": risk_result.reason,
                    "global_daily_realized_pnl_usd": global_daily_pnl,
                    "trader_daily_realized_pnl_usd": trader_daily_pnl,
                    "trader_consecutive_losses": trader_loss_streak,
                    "cooldown_seconds": cooldown_seconds,
                    "cooldown_active": cooldown_active,
                    "cooldown_remaining_seconds": cooldown_remaining_seconds,
                    "trader_open_positions": open_positions,
                    "checks": [
                        {
                            "check_key": check.key,
                            "check_label": check.key,
                            "passed": check.passed,
                            "score": check.score,
                            "detail": check.detail,
                        }
                        for check in risk_result.checks
                    ],
                }
                checks_payload.extend(
                    {
                        "check_key": check.key,
                        "check_label": check.key,
                        "passed": check.passed,
                        "score": check.score,
                        "detail": check.detail,
                    }
                    for check in risk_result.checks
                )
                if not risk_result.allowed:
                    final_decision = "blocked"
                    final_reason = risk_result.reason

            if bool(control.get("kill_switch")) and final_decision == "selected":
                final_decision = "blocked"
                final_reason = "Kill switch is enabled"

            decision_row = await create_trader_decision(
                session,
                trader_id=trader_id,
                signal=signal,
                strategy_key=str(trader.get("strategy_key") or ""),
                decision=final_decision,
                reason=final_reason,
                score=score,
                checks_summary={"count": len(checks_payload)},
                risk_snapshot=risk_snapshot,
                payload={
                    "strategy_payload": decision_obj.payload,
                    "size_usd": size_usd,
                    "live_market": {
                        "available": bool(live_context.get("available")),
                        "fetched_at": live_context.get("fetched_at"),
                        "selected_outcome": live_context.get("selected_outcome"),
                        "live_selected_price": live_context.get("live_selected_price"),
                        "signal_entry_price": live_context.get("signal_entry_price"),
                        "entry_price_delta": live_context.get("entry_price_delta"),
                        "entry_price_delta_pct": live_context.get("entry_price_delta_pct"),
                        "live_edge_percent": live_context.get("live_edge_percent"),
                        "history_summary": live_context.get("history_summary") or {},
                        "history_tail": live_context.get("history_tail") or [],
                    },
                    "risk_runtime": {
                        "global_daily_realized_pnl_usd": global_daily_pnl,
                        "trader_daily_realized_pnl_usd": trader_daily_pnl,
                        "trader_consecutive_losses": trader_loss_streak,
                        "cooldown_seconds": cooldown_seconds,
                        "cooldown_active": cooldown_active,
                        "cooldown_remaining_seconds": cooldown_remaining_seconds,
                        "trader_open_positions": open_positions,
                        "trading_window_ok": trading_window_ok,
                    },
                },
            )
            decisions_written += 1

            await create_trader_decision_checks(
                session,
                decision_id=decision_row.id,
                checks=checks_payload,
            )

            order_status = None
            if final_decision == "selected":
                status, effective_price, error_message, execution_payload = await submit_order(
                    mode=str(control.get("mode", "paper")),
                    signal=runtime_signal,
                    size_usd=size_usd,
                )
                await create_trader_order(
                    session,
                    trader_id=trader_id,
                    signal=runtime_signal,
                    decision_id=decision_row.id,
                    mode=str(control.get("mode", "paper")),
                    status=status,
                    notional_usd=size_usd,
                    effective_price=effective_price,
                    reason=final_reason,
                    payload=execution_payload,
                    error_message=error_message,
                )
                if str(status).strip().lower() in {"submitted", "executed", "open"}:
                    open_positions = await get_open_position_count_for_trader(
                        session,
                        trader_id,
                        mode=run_mode,
                    )
                orders_written += 1
                order_status = status

            await record_signal_consumption(
                session,
                trader_id=trader_id,
                signal_id=str(signal.id),
                decision_id=decision_row.id,
                outcome=order_status or final_decision,
                reason=final_reason,
            )

            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="decision",
                source=str(signal.source),
                message=final_reason,
                payload={
                    "decision_id": decision_row.id,
                    "signal_id": signal.id,
                    "decision": final_decision,
                    "order_status": order_status,
                },
            )

        row = await session.get(Trader, trader_id)
        if row is not None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            row.last_run_at = now
            row.requested_run_at = None
            row.updated_at = now
            await session.commit()

    return decisions_written, orders_written


async def run_worker_loop() -> None:
    logger.info("Starting trader orchestrator worker loop")

    while True:
        try:
            async with AsyncSessionLocal() as session:
                await expire_stale_signals(session)

                control = await read_orchestrator_control(session)
                interval = max(1, int(control.get("run_interval_seconds") or 2))

                if not control.get("is_enabled") or control.get("is_paused"):
                    await write_orchestrator_snapshot(
                        session,
                        running=False,
                        enabled=bool(control.get("is_enabled", False)),
                        current_activity="Paused",
                        interval_seconds=interval,
                        stats=await compute_orchestrator_metrics(session),
                    )
                    await asyncio.sleep(interval)
                    continue

                mode = str(control.get("mode") or "paper").strip().lower()
                control_settings = control.get("settings") or {}

                if mode == "paper":
                    paper_account_id = str(control_settings.get("paper_account_id") or "").strip()
                    if not paper_account_id:
                        await write_orchestrator_snapshot(
                            session,
                            running=False,
                            enabled=True,
                            current_activity="Blocked: select a sandbox account for paper mode",
                            interval_seconds=interval,
                            last_error=None,
                            stats=await compute_orchestrator_metrics(session),
                        )
                        await asyncio.sleep(interval)
                        continue
                    paper_account = await session.get(SimulationAccount, paper_account_id)
                    if paper_account is None:
                        await write_orchestrator_snapshot(
                            session,
                            running=False,
                            enabled=True,
                            current_activity="Blocked: selected sandbox account no longer exists",
                            interval_seconds=interval,
                            last_error=None,
                            stats=await compute_orchestrator_metrics(session),
                        )
                        await asyncio.sleep(interval)
                        continue

                if mode == "live":
                    app_settings = await session.get(AppSettings, "default")
                    trading_enabled = bool(settings.TRADING_ENABLED) and bool(
                        app_settings.trading_enabled if app_settings is not None else False
                    )
                    if not trading_enabled:
                        await write_orchestrator_snapshot(
                            session,
                            running=False,
                            enabled=True,
                            current_activity="Blocked: live trading disabled in config/settings",
                            interval_seconds=interval,
                            last_error=None,
                            stats=await compute_orchestrator_metrics(session),
                        )
                        await asyncio.sleep(interval)
                        continue
                    if not _is_live_credentials_configured(app_settings):
                        await write_orchestrator_snapshot(
                            session,
                            running=False,
                            enabled=True,
                            current_activity="Blocked: live credentials missing",
                            interval_seconds=interval,
                            last_error=None,
                            stats=await compute_orchestrator_metrics(session),
                        )
                        await asyncio.sleep(interval)
                        continue

                traders = await list_traders(session)

            total_decisions = 0
            total_orders = 0
            now = datetime.now(timezone.utc)
            for trader in traders:
                if not trader.get("is_enabled", True):
                    continue

                is_paused = bool(trader.get("is_paused", False))
                due = _is_due(trader, now)

                if is_paused:
                    if mode == "paper":
                        await _run_trader_once(trader, control, process_signals=False)
                    continue

                if not due:
                    if mode == "paper":
                        await _run_trader_once(trader, control, process_signals=False)
                    continue

                decisions, orders = await _run_trader_once(trader, control, process_signals=True)
                total_decisions += decisions
                total_orders += orders

            async with AsyncSessionLocal() as session:
                metrics = await compute_orchestrator_metrics(session)
                metrics["decisions_last_cycle"] = total_decisions
                metrics["orders_last_cycle"] = total_orders
                await write_orchestrator_snapshot(
                    session,
                    running=True,
                    enabled=True,
                    current_activity=f"Cycle decisions={total_decisions} orders={total_orders}",
                    interval_seconds=interval,
                    last_run_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    stats=metrics,
                )

            await asyncio.sleep(interval)
        except Exception as exc:
            logger.exception("Trader orchestrator worker cycle failed: %s", exc)
            async with AsyncSessionLocal() as session:
                control = await read_orchestrator_control(session)
                await write_orchestrator_snapshot(
                    session,
                    running=False,
                    enabled=bool(control.get("is_enabled", False)),
                    current_activity="Worker error",
                    interval_seconds=max(1, int(control.get("run_interval_seconds") or 2)),
                    last_error=str(exc),
                    stats=await compute_orchestrator_metrics(session),
                )
            await asyncio.sleep(2)


async def main() -> None:
    """Initialize DB schema before entering orchestrator loop."""
    await init_database()
    logger.info("Database initialized")

    notifier = None
    try:
        from services.notifier import notifier as notifier_service

        notifier = notifier_service
        await notifier.start()
        logger.info("Autotrader notifier started")
    except Exception as exc:
        logger.warning("Autotrader notifier start failed (non-critical): %s", exc)

    try:
        await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Trader orchestrator worker shutting down")
    finally:
        if notifier is not None:
            try:
                await notifier.shutdown()
            except Exception as exc:
                logger.debug("Notifier shutdown skipped: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
