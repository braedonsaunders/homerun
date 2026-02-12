"""Dedicated trader orchestrator worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from models.database import AsyncSessionLocal, Trader
from services.trader_orchestrator.order_manager import submit_order
from services.trader_orchestrator.risk_manager import evaluate_risk
from services.trader_orchestrator.strategies.registry import get_strategy
from services.trader_orchestrator_state import (
    compute_orchestrator_metrics,
    create_trader_decision,
    create_trader_decision_checks,
    create_trader_event,
    create_trader_order,
    get_gross_exposure,
    get_market_exposure,
    get_open_order_count_for_trader,
    list_traders,
    list_unconsumed_trade_signals,
    read_orchestrator_control,
    seed_default_traders,
    write_orchestrator_snapshot,
    record_signal_consumption,
)
from services.signal_bus import expire_stale_signals

logger = logging.getLogger("trader_orchestrator_worker")


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


async def _run_trader_once(
    trader: dict[str, Any],
    control: dict[str, Any],
) -> tuple[int, int]:
    decisions_written = 0
    orders_written = 0

    async with AsyncSessionLocal() as session:
        sources = [str(item) for item in (trader.get("sources") or [])]
        signals = await list_unconsumed_trade_signals(
            session,
            trader_id=str(trader["id"]),
            sources=sources or None,
            limit=200,
        )

        strategy = get_strategy(str(trader.get("strategy_key") or ""))
        params = trader.get("params") or {}
        risk_limits = trader.get("risk_limits") or {}

        for signal in signals:
            decision_obj = strategy.evaluate(
                signal,
                {
                    "params": params,
                    "trader": trader,
                    "mode": control.get("mode", "paper"),
                },
            )
            checks_payload = _checks_to_payload(decision_obj.checks)

            final_decision = decision_obj.decision
            final_reason = decision_obj.reason
            score = decision_obj.score
            size_usd = float(max(1.0, decision_obj.size_usd or 10.0))
            risk_snapshot = {}

            if final_decision == "selected":
                gross_exposure = await get_gross_exposure(session)
                open_orders = await get_open_order_count_for_trader(session, str(trader["id"]))
                market_exposure = await get_market_exposure(session, str(signal.market_id))
                global_limits = (control.get("settings") or {}).get("global_risk") or {}
                risk_result = evaluate_risk(
                    size_usd=size_usd,
                    gross_exposure_usd=gross_exposure,
                    trader_open_orders=open_orders,
                    market_exposure_usd=market_exposure,
                    global_limits=global_limits,
                    trader_limits=risk_limits,
                )
                risk_snapshot = {
                    "allowed": risk_result.allowed,
                    "reason": risk_result.reason,
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
                trader_id=str(trader["id"]),
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
                    signal=signal,
                    size_usd=size_usd,
                )
                await create_trader_order(
                    session,
                    trader_id=str(trader["id"]),
                    signal=signal,
                    decision_id=decision_row.id,
                    mode=str(control.get("mode", "paper")),
                    status=status,
                    notional_usd=size_usd,
                    effective_price=effective_price,
                    reason=final_reason,
                    payload=execution_payload,
                    error_message=error_message,
                )
                orders_written += 1
                order_status = status

            await record_signal_consumption(
                session,
                trader_id=str(trader["id"]),
                signal_id=str(signal.id),
                decision_id=decision_row.id,
                outcome=order_status or final_decision,
                reason=final_reason,
            )

            await create_trader_event(
                session,
                trader_id=str(trader["id"]),
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

        row = await session.get(Trader, str(trader["id"]))
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
                await seed_default_traders(session)
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

                traders = await list_traders(session)

            total_decisions = 0
            total_orders = 0
            now = datetime.now(timezone.utc)
            for trader in traders:
                if not trader.get("is_enabled", True) or trader.get("is_paused", False):
                    continue
                if not _is_due(trader, now):
                    continue
                decisions, orders = await _run_trader_once(trader, control)
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


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
