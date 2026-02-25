#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import desc, func, select


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from models.database import (  # noqa: E402
    AsyncSessionLocal,
    ExecutionSessionOrder,
    Trader,
    TraderDecision,
    TraderEvent,
    TraderOrder,
    TraderOrchestratorSnapshot,
    WorkerSnapshot,
)
from services.polymarket import polymarket_client  # noqa: E402
from services.trader_orchestrator_state import _extract_live_fill_metrics  # noqa: E402
from services.live_execution_service import live_execution_service as trading_service  # noqa: E402
from utils.converters import safe_float, safe_int  # noqa: E402


LIVE_ACTIVE_STATUSES = {"open", "executed", "submitted"}
LIVE_TERMINAL_STATUSES = {"closed_win", "closed_loss", "resolved_win", "resolved_loss", "cancelled", "failed"}
WATCHED_TRANSITION_STATUSES = LIVE_ACTIVE_STATUSES | LIVE_TERMINAL_STATUSES
TERMINAL_NON_RESOLUTION_TRIGGERS = {
    "take_profit",
    "stop_loss",
    "trailing_stop",
    "max_hold",
    "market_inactive",
    "manual_mark_to_market",
    "failed_exit_exhausted",
}
PENDING_EXIT_STUCK_SECONDS = 90
ALERT_DEDUP_SECONDS = 15
DECISION_FRESHNESS_SECONDS = 120
WALLET_EVENT_STALE_SECONDS = 120
WALLET_BLOCK_STALE_SECONDS = 60
DB_ALERT_WORKERS = {"trader_orchestrator", "trader_reconciliation", "event_dispatcher", "snapshot_broadcaster"}
WALLET_CONTEXT_REFRESH_SECONDS = 60.0
PROVIDER_SNAPSHOT_REFRESH_SECONDS = 30.0


@dataclass
class MonitorConfig:
    duration_seconds: int
    poll_seconds: float
    trader_id: Optional[str]
    trader_name: Optional[str]
    enable_provider_checks: bool


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        t_index = text.find("T")
        has_offset = False
        if t_index >= 0:
            time_part = text[t_index + 1 : -1]
            has_offset = ("+" in time_part) or ("-" in time_part[1:])
        if has_offset:
            text = text[:-1]
        else:
            text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _extract_token_id(payload: dict[str, Any]) -> str:
    for key in ("token_id", "selected_token_id"):
        raw = payload.get(key)
        token_id = str(raw or "").strip()
        if token_id:
            return token_id
    provider_reconciliation = payload.get("provider_reconciliation")
    if isinstance(provider_reconciliation, dict):
        snapshot = provider_reconciliation.get("snapshot")
        if isinstance(snapshot, dict):
            raw = snapshot.get("asset_id") or snapshot.get("asset") or snapshot.get("token_id")
            token_id = str(raw or "").strip()
            if token_id:
                return token_id
    return ""


def _extract_provider_clob_id(
    payload: dict[str, Any],
    execution_orders_for_order: list[ExecutionSessionOrder],
) -> str:
    provider_reconciliation = payload.get("provider_reconciliation")
    if isinstance(provider_reconciliation, dict):
        snapshot = provider_reconciliation.get("snapshot")
        if isinstance(snapshot, dict):
            clob = str(
                snapshot.get("clob_order_id")
                or snapshot.get("id")
                or ""
            ).strip()
            if clob:
                return clob
        clob = str(provider_reconciliation.get("provider_clob_order_id") or "").strip()
        if clob:
            return clob
    clob = str(payload.get("provider_clob_order_id") or "").strip()
    if clob:
        return clob
    for row in execution_orders_for_order:
        clob = str(row.provider_clob_order_id or "").strip()
        if clob:
            return clob
    return ""


def _extract_close_provider_clob_id(payload: dict[str, Any]) -> str:
    pending_exit = payload.get("pending_live_exit")
    if not isinstance(pending_exit, dict):
        return ""
    direct = str(
        pending_exit.get("provider_clob_order_id")
        or pending_exit.get("exit_order_clob_id")
        or ""
    ).strip()
    if direct:
        return direct
    close_order = pending_exit.get("close_order")
    if isinstance(close_order, dict):
        clob = str(close_order.get("provider_clob_order_id") or "").strip()
        if clob:
            return clob
    fallback = str(pending_exit.get("exit_order_id") or "").strip()
    if fallback.startswith("0x") or fallback.isdigit():
        return fallback
    if fallback:
        try:
            cached = trading_service.get_order(fallback)
        except Exception:
            cached = None
        if cached is not None:
            clob = str(getattr(cached, "clob_order_id", "") or "").strip()
            if clob:
                return clob
    return ""


def _extract_market_question(order: TraderOrder) -> str:
    return str(order.market_question or "").strip()


def _extract_order_reason(order: TraderOrder, payload: dict[str, Any]) -> str:
    pending_exit = payload.get("pending_live_exit")
    if isinstance(pending_exit, dict):
        trigger = str(pending_exit.get("close_trigger") or "").strip()
        if trigger:
            return trigger
    position_close = payload.get("position_close")
    if isinstance(position_close, dict):
        trigger = str(position_close.get("close_trigger") or "").strip()
        if trigger:
            return trigger
    return str(order.reason or "").strip()


def _extract_wallet_token_map(positions: list[dict[str, Any]]) -> dict[str, float]:
    token_sizes: dict[str, float] = {}
    for pos in positions:
        token_id = str(
            pos.get("asset")
            or pos.get("asset_id")
            or pos.get("assetId")
            or pos.get("token_id")
            or pos.get("tokenId")
            or ""
        ).strip()
        if not token_id:
            continue
        size = max(0.0, safe_float(pos.get("size"), 0.0) or 0.0)
        token_sizes[token_id] = token_sizes.get(token_id, 0.0) + size
    return token_sizes


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


async def _resolve_target_trader(config: MonitorConfig) -> Trader:
    async with AsyncSessionLocal() as session:
        if config.trader_id:
            trader = await session.get(Trader, config.trader_id)
            if trader is None:
                raise RuntimeError(f"Trader not found: {config.trader_id}")
            return trader

        base_query = (
            select(Trader)
            .where(
                Trader.is_enabled == True,  # noqa: E712
                func.lower(func.coalesce(Trader.strategy_key, "")) == "btc_eth_highfreq",
            )
            .order_by(desc(Trader.updated_at))
        )
        if config.trader_name:
            name_key = config.trader_name.strip().lower()
            if name_key:
                base_query = (
                    select(Trader)
                    .where(
                        Trader.is_enabled == True,  # noqa: E712
                        func.lower(func.coalesce(Trader.name, "")) == name_key,
                    )
                    .order_by(desc(Trader.updated_at))
                )

        result = await session.execute(base_query.limit(1))
        trader = result.scalar_one_or_none()
        if trader is None:
            raise RuntimeError("No enabled trader found for monitor target.")
        return trader


async def _load_cycle_state(trader_id: str) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        trader = await session.get(Trader, trader_id)
        if trader is None:
            raise RuntimeError(f"Trader not found during monitoring: {trader_id}")

        order_rows = list(
            (
                await session.execute(
                    select(TraderOrder)
                    .where(
                        TraderOrder.trader_id == trader_id,
                        func.lower(func.coalesce(TraderOrder.mode, "")) == "live",
                    )
                    .order_by(desc(TraderOrder.updated_at))
                    .limit(300)
                )
            )
            .scalars()
            .all()
        )

        order_ids = [str(row.id) for row in order_rows]
        execution_rows: list[ExecutionSessionOrder] = []
        if order_ids:
            execution_rows = list(
                (
                    await session.execute(
                        select(ExecutionSessionOrder)
                        .where(ExecutionSessionOrder.trader_order_id.in_(order_ids))
                        .order_by(desc(ExecutionSessionOrder.created_at))
                    )
                )
                .scalars()
                .all()
            )

        decisions = list(
            (
                await session.execute(
                    select(TraderDecision)
                    .where(TraderDecision.trader_id == trader_id)
                    .order_by(desc(TraderDecision.created_at))
                    .limit(1)
                )
            )
            .scalars()
            .all()
        )
        events = list(
            (
                await session.execute(
                    select(TraderEvent)
                    .where(
                        (TraderEvent.trader_id == trader_id) | (TraderEvent.trader_id.is_(None))
                    )
                    .order_by(desc(TraderEvent.created_at))
                    .limit(1)
                )
            )
            .scalars()
            .all()
        )

        orchestrator_snapshot = await session.get(TraderOrchestratorSnapshot, "latest")
        worker_rows = list(
            (
                await session.execute(
                    select(WorkerSnapshot).where(
                        WorkerSnapshot.worker_name.in_(
                            ("trader_reconciliation", "crypto", "events", "news", "scanner")
                        )
                    )
                )
            )
            .scalars()
            .all()
        )

        status_counts_rows = (
            await session.execute(
                select(
                    TraderOrder.status,
                    func.count(TraderOrder.id).label("count"),
                )
                .where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "live",
                )
                .group_by(TraderOrder.status)
            )
        ).all()
        status_counts: dict[str, int] = {}
        for row in status_counts_rows:
            key = normalize_status(row.status)
            status_counts[key] = status_counts.get(key, 0) + int(row.count or 0)

        return {
            "trader": trader,
            "orders": order_rows,
            "execution_orders": execution_rows,
            "decisions": decisions,
            "events": events,
            "orchestrator_snapshot": orchestrator_snapshot,
            "worker_snapshots": worker_rows,
            "status_counts": status_counts,
        }


async def _fetch_wallet_context() -> dict[str, Any]:
    ready = False
    error: Optional[str] = None
    try:
        ready = bool(await trading_service.ensure_initialized())
    except Exception as exc:  # pragma: no cover - defensive for live monitor runtime
        error = str(exc)

    runtime_sig_type = getattr(trading_service, "_balance_signature_type", None)
    signature_type = safe_int(
        runtime_sig_type if isinstance(runtime_sig_type, int) else getattr(settings, "POLYMARKET_SIGNATURE_TYPE", 1),
        1,
    )
    eoa_address = str(getattr(trading_service, "_eoa_address", "") or "").strip()
    wallet_address = str(trading_service._get_wallet_address() or "").strip()
    proxy_funder = str(getattr(trading_service, "_proxy_funder_address", "") or "").strip()
    execution_wallet = str(
        getattr(trading_service, "get_execution_wallet_address", lambda: None)() or ""
    ).strip()
    if not execution_wallet and hasattr(trading_service, "_funder_for_signature_type"):
        execution_wallet = str(trading_service._funder_for_signature_type(signature_type) or "").strip()
    if not execution_wallet:
        execution_wallet = wallet_address

    eoa_positions: list[dict[str, Any]] = []
    proxy_positions: list[dict[str, Any]] = []
    execution_positions: list[dict[str, Any]] = []
    if ready:
        try:
            if wallet_address:
                eoa_positions = await polymarket_client.get_wallet_positions(wallet_address)
            if proxy_funder and proxy_funder.lower() != wallet_address.lower():
                proxy_positions = await polymarket_client.get_wallet_positions(proxy_funder)
            if execution_wallet:
                execution_positions = await polymarket_client.get_wallet_positions(execution_wallet)
        except Exception as exc:  # pragma: no cover
            if error:
                error = f"{error}; {exc}"
            else:
                error = str(exc)

    return {
        "ready": ready,
        "error": error,
        "signature_type": signature_type,
        "eoa_address": eoa_address,
        "wallet_address": wallet_address,
        "proxy_funder": proxy_funder,
        "execution_wallet": execution_wallet,
        "eoa_token_sizes": _extract_wallet_token_map(eoa_positions),
        "proxy_token_sizes": _extract_wallet_token_map(proxy_positions),
        "execution_token_sizes": _extract_wallet_token_map(execution_positions),
    }


def _empty_wallet_context(error: str = "wallet_context_unavailable") -> dict[str, Any]:
    return {
        "ready": False,
        "error": error,
        "signature_type": safe_int(getattr(settings, "POLYMARKET_SIGNATURE_TYPE", 1), 1),
        "eoa_address": "",
        "wallet_address": "",
        "proxy_funder": "",
        "execution_wallet": "",
        "eoa_token_sizes": {},
        "proxy_token_sizes": {},
        "execution_token_sizes": {},
    }


def _build_provider_id_set(orders: list[TraderOrder], exec_rows_by_order: dict[str, list[ExecutionSessionOrder]]) -> set[str]:
    clob_ids: set[str] = set()
    for order in orders:
        payload = dict(order.payload_json or {})
        for candidate in (
            _extract_provider_clob_id(payload, exec_rows_by_order.get(str(order.id), [])),
            _extract_close_provider_clob_id(payload),
        ):
            if candidate:
                clob_ids.add(candidate)
    return clob_ids


def _make_alert_record(
    *,
    ts: datetime,
    trader_id: str,
    order: Optional[TraderOrder],
    signal_id: Optional[str],
    market_question: str,
    local_status_reason: str,
    provider_clob_order_id: str,
    provider_status: str,
    provider_filled_size: Optional[float],
    provider_price: Optional[float],
    wallet_position_size: Optional[float],
    verdict: str,
    root_cause: str,
    required_fix: str,
    rule: str,
) -> dict[str, Any]:
    return {
        "timestamp_utc": to_iso(ts),
        "trader_id": trader_id,
        "order_id": str(order.id) if order is not None else "",
        "signal_id": signal_id or (str(order.signal_id) if order is not None and order.signal_id else ""),
        "market_question": market_question,
        "local_status_reason": local_status_reason,
        "provider": {
            "clob_order_id": provider_clob_order_id,
            "status": provider_status,
            "filled_size": provider_filled_size,
            "price": provider_price,
        },
        "wallet_position_size": wallet_position_size,
        "lifecycle_verdict": verdict,
        "root_cause": root_cause,
        "required_fix": required_fix,
        "rule": rule,
    }


async def run_monitor(config: MonitorConfig) -> int:
    trader = await _resolve_target_trader(config)
    trader_id = str(trader.id)
    started_at = utcnow()
    ends_at = started_at + timedelta(seconds=max(1, config.duration_seconds))
    started_mono = time.monotonic()
    ends_mono = started_mono + float(max(1, config.duration_seconds))
    next_cycle_mono = started_mono

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"live_truth_monitor_{stamp}_{trader_id[:8]}.jsonl"
    summary_path = out_dir / f"live_truth_monitor_{stamp}_{trader_id[:8]}_summary.json"

    print(
        _json_dumps(
            {
                "event": "monitor_start",
                "timestamp_utc": to_iso(started_at),
                "duration_seconds": config.duration_seconds,
                "poll_seconds": config.poll_seconds,
                "trader_id": trader_id,
                "trader_name": str(trader.name),
                "strategy_key": str(trader.strategy_key),
                "provider_checks_enabled": bool(config.enable_provider_checks),
                "report_path": str(report_path),
            }
        ),
        flush=True,
    )

    seen_alerts: dict[str, datetime] = {}
    prev_order_state: dict[str, dict[str, Any]] = {}
    prev_decision_id = ""
    alerts_count_by_rule: dict[str, int] = {}
    total_cycles = 0
    monitor_errors: list[str] = []
    cached_wallet_ctx: Optional[dict[str, Any]] = None
    cached_wallet_ctx_mono = -1.0
    wallet_ctx_refresh_task: Optional[asyncio.Task[dict[str, Any]]] = None
    cached_provider_ids: tuple[str, ...] = ()
    cached_provider_snapshots: dict[str, dict[str, Any]] = {}
    cached_provider_snapshots_mono = -1.0

    with report_path.open("w", encoding="utf-8") as report:
        while True:
            now_mono = time.monotonic()
            if now_mono >= ends_mono:
                break
            if now_mono < next_cycle_mono:
                await asyncio.sleep(min(next_cycle_mono - now_mono, ends_mono - now_mono))
                continue
            total_cycles += 1
            try:
                cycle_state = await _load_cycle_state(trader_id)
                if config.enable_provider_checks:
                    if wallet_ctx_refresh_task is not None and wallet_ctx_refresh_task.done():
                        try:
                            cached_wallet_ctx = wallet_ctx_refresh_task.result()
                            cached_wallet_ctx_mono = now_mono
                        except Exception as exc:
                            monitor_errors.append(f"wallet_context_error:{exc}")
                            if cached_wallet_ctx is None:
                                cached_wallet_ctx = _empty_wallet_context(str(exc))
                        finally:
                            wallet_ctx_refresh_task = None
                    if (
                        wallet_ctx_refresh_task is None
                        and (
                            cached_wallet_ctx is None
                            or cached_wallet_ctx_mono < 0.0
                            or (now_mono - cached_wallet_ctx_mono) >= WALLET_CONTEXT_REFRESH_SECONDS
                        )
                    ):
                        wallet_ctx_refresh_task = asyncio.create_task(_fetch_wallet_context())
                    wallet_ctx = cached_wallet_ctx or _empty_wallet_context(
                        "wallet_context_refresh_pending"
                        if wallet_ctx_refresh_task is not None
                        else "wallet_context_unavailable"
                    )
                else:
                    wallet_ctx = _empty_wallet_context("provider_checks_disabled")
                orders: list[TraderOrder] = cycle_state["orders"]
                exec_rows: list[ExecutionSessionOrder] = cycle_state["execution_orders"]
                decisions: list[TraderDecision] = cycle_state["decisions"]
                events: list[TraderEvent] = cycle_state["events"]

                exec_rows_by_order: dict[str, list[ExecutionSessionOrder]] = {}
                for exec_row in exec_rows:
                    key = str(exec_row.trader_order_id or "").strip()
                    if not key:
                        continue
                    exec_rows_by_order.setdefault(key, []).append(exec_row)

                provider_ids = tuple(sorted(_build_provider_id_set(orders, exec_rows_by_order)))
                provider_snapshots: dict[str, dict[str, Any]] = {}
                if config.enable_provider_checks and provider_ids:
                    should_refresh_provider_snapshots = (
                        cached_provider_snapshots_mono < 0.0
                        or provider_ids != cached_provider_ids
                        or (now_mono - cached_provider_snapshots_mono) >= PROVIDER_SNAPSHOT_REFRESH_SECONDS
                    )
                    if should_refresh_provider_snapshots:
                        try:
                            cached_provider_snapshots = await trading_service.get_order_snapshots_by_clob_ids(
                                list(provider_ids)
                            )
                            cached_provider_snapshots_mono = now_mono
                            cached_provider_ids = provider_ids
                        except Exception as exc:
                            monitor_errors.append(f"provider_snapshot_error:{exc}")
                            if provider_ids != cached_provider_ids:
                                cached_provider_ids = provider_ids
                    provider_snapshots = {
                        clob_id: dict(cached_provider_snapshots.get(clob_id) or {})
                        for clob_id in provider_ids
                    }
                else:
                    cached_provider_ids = ()
                    cached_provider_snapshots = {}
                    cached_provider_snapshots_mono = now_mono

                new_alerts: list[dict[str, Any]] = []
                now = utcnow()

                # Hard check: trader unexpectedly paused or disabled during monitoring.
                if bool(cycle_state["trader"].is_paused):
                    new_alerts.append(
                        _make_alert_record(
                            ts=now,
                            trader_id=trader_id,
                            order=None,
                            signal_id="",
                            market_question="",
                            local_status_reason="trader is_paused=true",
                            provider_clob_order_id="",
                            provider_status="",
                            provider_filled_size=None,
                            provider_price=None,
                            wallet_position_size=None,
                            verdict="drift",
                            root_cause="Trader was auto-paused by runtime controls during live monitoring.",
                            required_fix="Review circuit breaker thresholds and recent loss streak before re-enabling.",
                            rule="trader_paused",
                        )
                    )
                if not bool(cycle_state["trader"].is_enabled):
                    new_alerts.append(
                        _make_alert_record(
                            ts=now,
                            trader_id=trader_id,
                            order=None,
                            signal_id="",
                            market_question="",
                            local_status_reason="trader is_enabled=false",
                            provider_clob_order_id="",
                            provider_status="",
                            provider_filled_size=None,
                            provider_price=None,
                            wallet_position_size=None,
                            verdict="drift",
                            root_cause="Trader was disabled while monitor expected active live execution.",
                            required_fix="Re-enable trader only after verifying live risk and wallet funding state.",
                            rule="trader_disabled",
                        )
                    )

                if cycle_state["orchestrator_snapshot"] is not None:
                    orchestrator = cycle_state["orchestrator_snapshot"]
                    if not bool(orchestrator.running):
                        new_alerts.append(
                            _make_alert_record(
                                ts=now,
                                trader_id=trader_id,
                                order=None,
                                signal_id="",
                                market_question="",
                                local_status_reason="orchestrator running=false",
                                provider_clob_order_id="",
                                provider_status="",
                                provider_filled_size=None,
                                provider_price=None,
                                wallet_position_size=None,
                                verdict="drift",
                                root_cause="Trader orchestrator snapshot indicates the lifecycle loop is not running.",
                                required_fix="Restart or unpause trader orchestrator worker and confirm heartbeat updates.",
                                rule="orchestrator_down",
                            )
                        )

                worker_snapshots: list[WorkerSnapshot] = cycle_state["worker_snapshots"]
                for row in worker_snapshots:
                    worker_name = str(row.worker_name or "")
                    worker_key = worker_name.strip().lower()
                    last_error = str(row.last_error or "").strip()
                    if not last_error:
                        continue
                    lowered = last_error.lower()
                    worker_in_scope = worker_key in DB_ALERT_WORKERS or worker_key.startswith("trader_")
                    if worker_in_scope and ("interfaceerror" in lowered or "connection is closed" in lowered):
                        new_alerts.append(
                            _make_alert_record(
                                ts=now,
                                trader_id=trader_id,
                                order=None,
                                signal_id="",
                                market_question="",
                                local_status_reason=f"{worker_name} last_error",
                                provider_clob_order_id="",
                                provider_status="",
                                provider_filled_size=None,
                                provider_price=None,
                                wallet_position_size=None,
                                verdict="drift",
                                root_cause="Worker snapshot shows repeated database disconnect symptoms.",
                                required_fix="Stabilize DB pool/connection lifecycle so reconcile cycles are not skipped.",
                                rule="db_disconnect_loop",
                            )
                        )
                    if "event_dispatcher" in lowered and "timeout" in lowered:
                        new_alerts.append(
                            _make_alert_record(
                                ts=now,
                                trader_id=trader_id,
                                order=None,
                                signal_id="",
                                market_question="",
                                local_status_reason=f"{worker_name} last_error",
                                provider_clob_order_id="",
                                provider_status="",
                                provider_filled_size=None,
                                provider_price=None,
                                wallet_position_size=None,
                                verdict="drift",
                                root_cause="Event dispatcher timeouts are delaying lifecycle event propagation.",
                                required_fix="Reduce event backlog and restore dispatcher responsiveness before new live orders.",
                                rule="event_dispatcher_timeout",
                            )
                        )

                for order in orders:
                    order_id = str(order.id)
                    status = normalize_status(order.status)
                    if status not in WATCHED_TRANSITION_STATUSES:
                        continue
                    payload = dict(order.payload_json or {})
                    pending_exit = payload.get("pending_live_exit")
                    pending_exit_status = ""
                    pending_exit_trigger_dt: Optional[datetime] = None
                    if isinstance(pending_exit, dict):
                        pending_exit_status = normalize_status(pending_exit.get("status"))
                        pending_exit_trigger_dt = (
                            parse_iso(pending_exit.get("triggered_at"))
                            or parse_iso(pending_exit.get("last_attempt_at"))
                        )

                    token_id = _extract_token_id(payload)
                    order_provider_clob_id = _extract_provider_clob_id(payload, exec_rows_by_order.get(order_id, []))
                    snapshot = provider_snapshots.get(order_provider_clob_id, {})
                    provider_status = normalize_status(
                        snapshot.get("normalized_status") or snapshot.get("raw_status")
                    )
                    provider_filled_size = safe_float(snapshot.get("filled_size"))
                    provider_price = safe_float(snapshot.get("average_fill_price")) or safe_float(snapshot.get("limit_price"))
                    wallet_position_size = (
                        wallet_ctx["execution_token_sizes"].get(token_id)
                        if token_id
                        else None
                    )

                    local_reason = _extract_order_reason(order, payload)
                    market_question = _extract_market_question(order)

                    # Local active with no provider evidence.
                    if (
                        config.enable_provider_checks
                        and wallet_ctx["ready"]
                        and status in LIVE_ACTIVE_STATUSES
                        and not order_provider_clob_id
                    ):
                        new_alerts.append(
                            _make_alert_record(
                                ts=now,
                                trader_id=trader_id,
                                order=order,
                                signal_id=str(order.signal_id or ""),
                                market_question=market_question,
                                local_status_reason=f"{status} with missing provider_clob_order_id",
                                provider_clob_order_id="",
                                provider_status="missing",
                                provider_filled_size=None,
                                provider_price=None,
                                wallet_position_size=wallet_position_size,
                                verdict="drift",
                                root_cause="Local live order is active but missing provider linkage IDs.",
                                required_fix="Populate provider order IDs from execution_session_orders and reconcile immediately.",
                                rule="active_no_provider_id",
                            )
                        )

                    # Entry truth: provider filled but local failed/pending stale.
                    if (
                        config.enable_provider_checks
                        and provider_status in {"filled", "matched", "executed"}
                        and status in {"failed", "submitted"}
                    ):
                        order_age_seconds = max(
                            0.0,
                            (now - (order.updated_at.replace(tzinfo=timezone.utc) if order.updated_at and order.updated_at.tzinfo is None else (order.updated_at or now))).total_seconds()
                            if order.updated_at is not None
                            else 0.0,
                        )
                        if status == "failed" or order_age_seconds > 15.0:
                            new_alerts.append(
                                _make_alert_record(
                                    ts=now,
                                    trader_id=trader_id,
                                    order=order,
                                    signal_id=str(order.signal_id or ""),
                                    market_question=market_question,
                                    local_status_reason=f"{status} while provider={provider_status}",
                                    provider_clob_order_id=order_provider_clob_id,
                                    provider_status=provider_status,
                                    provider_filled_size=provider_filled_size,
                                    provider_price=provider_price,
                                    wallet_position_size=wallet_position_size,
                                    verdict="drift",
                                    root_cause="Provider indicates entry filled but local lifecycle state is not converged to open.",
                                    required_fix="Force provider reconciliation to normalize status to open for filled entries.",
                                    rule="entry_state_stale",
                                )
                            )

                    # Stuck pending live exit.
                    if isinstance(pending_exit, dict) and pending_exit_status in {"pending", "submitted", "failed"}:
                        if pending_exit_trigger_dt is not None:
                            pending_age = (now - pending_exit_trigger_dt).total_seconds()
                            if pending_age >= PENDING_EXIT_STUCK_SECONDS:
                                new_alerts.append(
                                    _make_alert_record(
                                        ts=now,
                                        trader_id=trader_id,
                                        order=order,
                                        signal_id=str(order.signal_id or ""),
                                        market_question=market_question,
                                        local_status_reason=f"pending_live_exit={pending_exit_status} age={pending_age:.0f}s",
                                        provider_clob_order_id=order_provider_clob_id,
                                        provider_status=provider_status,
                                        provider_filled_size=provider_filled_size,
                                        provider_price=provider_price,
                                        wallet_position_size=wallet_position_size,
                                        verdict="drift",
                                        root_cause="pending_live_exit is stale and not progressing toward fill reconciliation.",
                                        required_fix="Re-submit or cancel/rebuild close order using current provider state and allowance.",
                                        rule="pending_exit_stuck",
                                    )
                                )

                    # Non-strategy fallback exits.
                    position_close = payload.get("position_close")
                    close_trigger = ""
                    if isinstance(position_close, dict):
                        close_trigger = normalize_status(position_close.get("close_trigger"))
                    elif isinstance(pending_exit, dict):
                        close_trigger = normalize_status(pending_exit.get("close_trigger"))
                    if close_trigger and close_trigger in TERMINAL_NON_RESOLUTION_TRIGGERS:
                        new_alerts.append(
                            _make_alert_record(
                                ts=now,
                                trader_id=trader_id,
                                order=order,
                                signal_id=str(order.signal_id or ""),
                                market_question=market_question,
                                local_status_reason=f"close_trigger={close_trigger}",
                                provider_clob_order_id=order_provider_clob_id,
                                provider_status=provider_status,
                                provider_filled_size=provider_filled_size,
                                provider_price=provider_price,
                                wallet_position_size=wallet_position_size,
                                verdict="drift",
                                root_cause="Exit trigger came from lifecycle fallback rather than strategy should_exit or resolution.",
                                required_fix="Constrain live exits to strategy/resolution paths and disable fallback-based closures.",
                                rule="non_strategy_exit_trigger",
                            )
                        )

                    # Terminal close with partial or missing provider close fill.
                    if config.enable_provider_checks and status in LIVE_TERMINAL_STATUSES and status.startswith("closed"):
                        pending_exit_status = ""
                        pending_exit_filled_size = 0.0
                        pending_exit_fill_price = None
                        pending_exit_required_size = 0.0
                        pending_exit_provider_status = ""
                        if isinstance(pending_exit, dict):
                            pending_exit_status = normalize_status(pending_exit.get("status"))
                            pending_exit_filled_size = safe_float(pending_exit.get("filled_size"), 0.0) or 0.0
                            pending_exit_fill_price = safe_float(pending_exit.get("average_fill_price"))
                            pending_exit_required_size = safe_float(pending_exit.get("exit_size"), 0.0) or 0.0
                            pending_exit_provider_status = normalize_status(pending_exit.get("provider_status"))
                        if (
                            close_trigger in {"resolution", "resolution_inferred", "external_wallet_flatten"}
                            or pending_exit_status in {"superseded_resolution", "superseded_external"}
                        ):
                            continue
                        close_provider_clob_id = _extract_close_provider_clob_id(payload)
                        close_snapshot = provider_snapshots.get(close_provider_clob_id, {}) if close_provider_clob_id else {}
                        close_provider_status = normalize_status(
                            close_snapshot.get("normalized_status") or close_snapshot.get("raw_status")
                        )
                        close_filled_size = safe_float(close_snapshot.get("filled_size"), 0.0) or 0.0
                        if close_provider_status in {"", "missing", "invalid"} and pending_exit_provider_status:
                            close_provider_status = pending_exit_provider_status
                        if close_filled_size <= 0.0 and pending_exit_filled_size > 0.0:
                            close_filled_size = pending_exit_filled_size
                        _, entry_filled_size, _ = _extract_live_fill_metrics(payload)
                        required_size = max(0.0, pending_exit_required_size if pending_exit_required_size > 0 else entry_filled_size)
                        if required_size > 0 and (
                            close_provider_status not in {"filled", "matched", "executed"}
                            or close_filled_size + 1e-9 < (required_size * 0.98)
                        ):
                            new_alerts.append(
                                _make_alert_record(
                                    ts=now,
                                    trader_id=trader_id,
                                    order=order,
                                    signal_id=str(order.signal_id or ""),
                                    market_question=market_question,
                                    local_status_reason=f"terminal={status} required_close_size={required_size:.6f}",
                                    provider_clob_order_id=close_provider_clob_id,
                                    provider_status=close_provider_status or "missing",
                                    provider_filled_size=close_filled_size,
                                    provider_price=safe_float(close_snapshot.get("average_fill_price")) or pending_exit_fill_price,
                                    wallet_position_size=wallet_position_size,
                                    verdict="drift",
                                    root_cause="Order is locally terminal but provider close fill is partial or absent.",
                                    required_fix="Gate terminal close state on verified close fill ratio reaching threshold.",
                                    rule="false_terminal_close",
                                )
                            )

                    # Account visibility mismatch (proxy funder vs EOA).
                    eoa_size = wallet_ctx["eoa_token_sizes"].get(token_id, 0.0) if token_id else 0.0
                    proxy_size = wallet_ctx["proxy_token_sizes"].get(token_id, 0.0) if token_id else 0.0
                    execution_wallet = str(wallet_ctx.get("execution_wallet") or "").strip().lower()
                    proxy_wallet = str(wallet_ctx.get("proxy_funder") or "").strip().lower()
                    execution_tracks_proxy = bool(
                        execution_wallet
                        and proxy_wallet
                        and execution_wallet == proxy_wallet
                    )
                    if (
                        config.enable_provider_checks
                        and
                        token_id
                        and provider_status in {"filled", "matched", "executed"}
                        and eoa_size <= 1e-9
                        and proxy_size > 1e-9
                        and not execution_tracks_proxy
                    ):
                        new_alerts.append(
                            _make_alert_record(
                                ts=now,
                                trader_id=trader_id,
                                order=order,
                                signal_id=str(order.signal_id or ""),
                                market_question=market_question,
                                local_status_reason=f"{status} provider={provider_status}",
                                provider_clob_order_id=order_provider_clob_id,
                                provider_status=provider_status,
                                provider_filled_size=provider_filled_size,
                                provider_price=provider_price,
                                wallet_position_size=proxy_size,
                                verdict="drift",
                                root_cause="Filled trade appears on proxy funder wallet while EOA view is empty.",
                                required_fix="Surface proxy funder holdings in UI and align execution wallet context.",
                                rule="proxy_visibility_mismatch",
                            )
                        )

                    # Open local without provider evidence and without wallet evidence.
                    if config.enable_provider_checks and wallet_ctx["ready"] and status in LIVE_ACTIVE_STATUSES:
                        execution_size = wallet_ctx["execution_token_sizes"].get(token_id, 0.0) if token_id else 0.0
                        if (
                            provider_status in {"", "missing", "cancelled"}
                            and execution_size <= 1e-9
                        ):
                            new_alerts.append(
                                _make_alert_record(
                                    ts=now,
                                    trader_id=trader_id,
                                    order=order,
                                    signal_id=str(order.signal_id or ""),
                                    market_question=market_question,
                                    local_status_reason=f"{status} with no provider/wallet evidence",
                                    provider_clob_order_id=order_provider_clob_id,
                                    provider_status=provider_status or "missing",
                                    provider_filled_size=provider_filled_size,
                                    provider_price=provider_price,
                                    wallet_position_size=execution_size,
                                    verdict="drift",
                                    root_cause="Local active position has no provider evidence and no wallet token inventory.",
                                    required_fix="Mark order non-active or resync from provider snapshot source of truth.",
                                    rule="local_open_without_provider_truth",
                                )
                            )

                    # Transition-driven reconciliation requirement.
                    previous = prev_order_state.get(order_id)
                    current_state = {
                        "status": status,
                        "pending_exit_status": pending_exit_status,
                        "provider_status": provider_status,
                        "updated_at": to_iso(order.updated_at),
                    }
                    if previous is not None and (
                        previous["status"] != current_state["status"]
                        or previous["pending_exit_status"] != current_state["pending_exit_status"]
                    ):
                        if status in WATCHED_TRANSITION_STATUSES:
                            transition_note = {
                                "event": "transition",
                                "timestamp_utc": to_iso(now),
                                "trader_id": trader_id,
                                "order_id": order_id,
                                "from": previous,
                                "to": current_state,
                                "provider_clob_order_id": order_provider_clob_id,
                            }
                            report.write(_json_dumps(transition_note) + "\n")
                    prev_order_state[order_id] = current_state

                # Risk gating mismatch: blocked on open positions while not actually near cap.
                max_open_positions = max(
                    1,
                    safe_int(
                        (cycle_state["trader"].risk_limits_json or {}).get("max_open_positions"),
                        10,
                    ),
                )
                max_open_orders = max(
                    1,
                    safe_int(
                        (cycle_state["trader"].risk_limits_json or {}).get("max_open_orders"),
                        max_open_positions,
                    ),
                )
                active_count = sum(cycle_state["status_counts"].get(k, 0) for k in LIVE_ACTIVE_STATUSES)
                latest_decision = decisions[0] if decisions else None
                if latest_decision is not None:
                    latest_decision_id = str(latest_decision.id)
                    if latest_decision_id != prev_decision_id:
                        prev_decision_id = latest_decision_id
                        decision_created_at = latest_decision.created_at
                        decision_age_seconds = None
                        if decision_created_at is not None:
                            decision_created_at_aware = (
                                decision_created_at.replace(tzinfo=timezone.utc)
                                if decision_created_at.tzinfo is None
                                else decision_created_at.astimezone(timezone.utc)
                            )
                            decision_age_seconds = (now - decision_created_at_aware).total_seconds()
                        decision_reason = str(latest_decision.reason or "").strip().lower()
                        risk_snapshot_json = dict(latest_decision.risk_snapshot_json or {})
                        snapshot_open_positions = max(
                            0,
                            safe_int(risk_snapshot_json.get("trader_open_positions"), active_count),
                        )
                        snapshot_open_orders = max(
                            0,
                            safe_int(risk_snapshot_json.get("trader_open_orders"), active_count),
                        )
                        if (
                            "risk blocked: trader_open_positions" in decision_reason
                            and snapshot_open_positions < max_open_positions
                            and decision_age_seconds is not None
                            and decision_age_seconds <= DECISION_FRESHNESS_SECONDS
                        ):
                            new_alerts.append(
                                _make_alert_record(
                                    ts=now,
                                    trader_id=trader_id,
                                    order=None,
                                    signal_id=str(latest_decision.signal_id or ""),
                                    market_question="",
                                    local_status_reason=(
                                        f"active_count={active_count} snapshot_open_positions={snapshot_open_positions} "
                                        f"max_open_positions={max_open_positions} "
                                        f"decision_age={decision_age_seconds:.1f}s"
                                    ),
                                    provider_clob_order_id="",
                                    provider_status="",
                                    provider_filled_size=None,
                                    provider_price=None,
                                    wallet_position_size=None,
                                    verdict="drift",
                                    root_cause="Risk gate blocked on trader_open_positions below configured threshold.",
                                    required_fix="Use reconciled open-position count derived from provider truth before blocking.",
                                    rule="false_open_position_block",
                                )
                            )
                        if (
                            "risk blocked: trader_open_orders" in decision_reason
                            and snapshot_open_orders < max_open_orders
                            and decision_age_seconds is not None
                            and decision_age_seconds <= DECISION_FRESHNESS_SECONDS
                        ):
                            new_alerts.append(
                                _make_alert_record(
                                    ts=now,
                                    trader_id=trader_id,
                                    order=None,
                                    signal_id=str(latest_decision.signal_id or ""),
                                    market_question="",
                                    local_status_reason=(
                                        f"active_count={active_count} snapshot_open_orders={snapshot_open_orders} "
                                        f"max_open_orders={max_open_orders} decision_age={decision_age_seconds:.1f}s"
                                    ),
                                    provider_clob_order_id="",
                                    provider_status="",
                                    provider_filled_size=None,
                                    provider_price=None,
                                    wallet_position_size=None,
                                    verdict="drift",
                                    root_cause="Risk gate blocked on trader_open_orders below configured threshold.",
                                    required_fix="Use reconciled active-order count derived from provider truth before blocking.",
                                    rule="false_open_order_block",
                                )
                            )

                # Wallet monitor down + no provider fallback.
                provider_updates_recent = False
                provider_fallback_recent = False
                provider_fallback_age = None
                wallet_monitor_running = False
                wallet_monitor_ws_connected = False
                wallet_monitor_fallback_polling = False
                wallet_monitor_tracked_wallets = 0
                wallet_monitor_events_detected_total = 0
                wallet_monitor_last_detected_at_iso = ""
                wallet_last_event_age = None
                wallet_last_block_age = None
                wallet_last_fallback_poll_age = None
                for worker_row in worker_snapshots:
                    if str(worker_row.worker_name) == "trader_reconciliation":
                        stats_json = dict(worker_row.stats_json or {})
                        provider_updates_count = safe_int(stats_json.get("provider_updates"), 0)
                        provider_updates_recent = provider_updates_count > 0
                        provider_pass_raw = stats_json.get("provider_pass")
                        provider_pass = bool(provider_pass_raw) if provider_pass_raw is not None else False
                        cycle_reason = normalize_status(stats_json.get("cycle_reason"))
                        provider_cycle_indicates_pass = (
                            provider_pass
                            or cycle_reason.startswith("event:")
                            or cycle_reason in {"scheduled", "requested", "startup"}
                        )
                        worker_last_run = worker_row.last_run_at
                        worker_age_seconds = None
                        if worker_last_run is not None:
                            worker_last_run_aware = (
                                worker_last_run.replace(tzinfo=timezone.utc)
                                if worker_last_run.tzinfo is None
                                else worker_last_run.astimezone(timezone.utc)
                            )
                            worker_age_seconds = (now - worker_last_run_aware).total_seconds()
                        worker_interval_seconds = max(
                            1.0,
                            safe_float(getattr(worker_row, "interval_seconds", None), 1.0) or 1.0,
                        )
                        provider_staleness_limit = max(3.0, worker_interval_seconds * 3.0)
                        if (
                            provider_cycle_indicates_pass
                            and worker_age_seconds is not None
                            and worker_age_seconds <= provider_staleness_limit
                        ):
                            provider_fallback_recent = True
                            provider_fallback_age = worker_age_seconds
                        wallet_monitor_running = bool(stats_json.get("wallet_monitor_running"))
                        wallet_monitor_ws_connected = bool(stats_json.get("wallet_monitor_ws_connected"))
                        wallet_monitor_fallback_polling = bool(stats_json.get("wallet_monitor_fallback_polling"))
                        wallet_monitor_tracked_wallets = max(
                            0,
                            safe_int(stats_json.get("wallet_monitor_tracked_wallets"), 0),
                        )
                        wallet_monitor_events_detected_total = max(
                            wallet_monitor_events_detected_total,
                            max(
                                0,
                                safe_int(stats_json.get("wallet_monitor_events_detected_total"), 0),
                            ),
                        )
                        worker_wallet_event_at = parse_iso(stats_json.get("wallet_monitor_last_event_detected_at"))
                        worker_wallet_block_at = parse_iso(
                            stats_json.get("wallet_monitor_last_block_processed_at")
                            or stats_json.get("wallet_monitor_last_block_seen_at")
                        )
                        worker_wallet_poll_at = parse_iso(stats_json.get("wallet_monitor_last_fallback_poll_at"))
                        if worker_wallet_event_at is not None:
                            wallet_last_event_age = (now - worker_wallet_event_at).total_seconds()
                            wallet_monitor_last_detected_at_iso = to_iso(worker_wallet_event_at)
                        if worker_wallet_block_at is not None:
                            wallet_last_block_age = (now - worker_wallet_block_at).total_seconds()
                        if worker_wallet_poll_at is not None:
                            wallet_last_fallback_poll_age = (now - worker_wallet_poll_at).total_seconds()
                wallet_event_recent = (
                    wallet_last_event_age is not None
                    and wallet_last_event_age <= WALLET_EVENT_STALE_SECONDS
                )
                wallet_transport_recent = (
                    wallet_last_block_age is not None
                    and wallet_last_block_age <= WALLET_BLOCK_STALE_SECONDS
                ) or (
                    wallet_monitor_fallback_polling
                    and wallet_last_fallback_poll_age is not None
                    and wallet_last_fallback_poll_age <= WALLET_BLOCK_STALE_SECONDS
                )
                if (
                    active_count > 0
                    and not wallet_event_recent
                    and not wallet_transport_recent
                    and not provider_fallback_recent
                    and not provider_updates_recent
                ):
                    new_alerts.append(
                        _make_alert_record(
                            ts=now,
                            trader_id=trader_id,
                            order=None,
                            signal_id="",
                            market_question="",
                            local_status_reason=(
                                f"wallet_event_age={wallet_last_event_age} "
                                f"wallet_block_age={wallet_last_block_age} "
                                f"wallet_fallback_poll_age={wallet_last_fallback_poll_age} "
                                f"wallet_running={wallet_monitor_running} "
                                f"wallet_ws_connected={wallet_monitor_ws_connected} "
                                f"wallet_fallback_polling={wallet_monitor_fallback_polling} "
                                f"wallet_tracked_wallets={wallet_monitor_tracked_wallets} "
                                f"provider_pass_age={provider_fallback_age} "
                                f"provider_updates={provider_updates_recent}"
                            ),
                            provider_clob_order_id="",
                            provider_status="",
                            provider_filled_size=None,
                            provider_price=None,
                            wallet_position_size=None,
                            verdict="drift",
                            root_cause="Wallet monitor has no recent event/block activity and provider reconciliation fallback is stale.",
                            required_fix="Restore wallet monitor transport heartbeat or force fresh provider reconciliation while active positions exist.",
                            rule="wallet_monitor_down_no_fallback",
                        )
                    )

                # Emit alerts with short dedupe window.
                for alert in new_alerts:
                    dedupe_key = "|".join(
                        [
                            alert["rule"],
                            alert["trader_id"],
                            alert["order_id"],
                            alert["provider"]["clob_order_id"],
                            alert["root_cause"],
                        ]
                    )
                    last_seen = seen_alerts.get(dedupe_key)
                    if last_seen is not None and (now - last_seen).total_seconds() < ALERT_DEDUP_SECONDS:
                        continue
                    seen_alerts[dedupe_key] = now
                    alerts_count_by_rule[alert["rule"]] = alerts_count_by_rule.get(alert["rule"], 0) + 1
                    report.write(_json_dumps(alert) + "\n")
                    print(_json_dumps({"event": "alert", "alert": alert}), flush=True)

                # Heartbeat each ~30 seconds.
                if total_cycles == 1 or total_cycles % max(1, int(round(30 / config.poll_seconds))) == 0:
                    latest_event = events[0] if events else None
                    heartbeat = {
                        "event": "heartbeat",
                        "timestamp_utc": to_iso(now),
                        "trader_id": trader_id,
                        "trader_name": str(cycle_state["trader"].name or ""),
                        "is_enabled": bool(cycle_state["trader"].is_enabled),
                        "is_paused": bool(cycle_state["trader"].is_paused),
                        "status_counts": cycle_state["status_counts"],
                        "wallet_monitor_total": wallet_monitor_events_detected_total,
                        "wallet_monitor_last_detected_at": wallet_monitor_last_detected_at_iso,
                        "provider_checks_enabled": bool(config.enable_provider_checks),
                        "provider_ready": bool(wallet_ctx["ready"]),
                        "execution_wallet": wallet_ctx["execution_wallet"],
                        "eoa_wallet": wallet_ctx["wallet_address"],
                        "proxy_funder_wallet": wallet_ctx["proxy_funder"],
                        "latest_event_type": str(latest_event.event_type) if latest_event is not None else "",
                        "latest_event_at": to_iso(latest_event.created_at) if latest_event is not None else "",
                        "alerts_so_far": int(sum(alerts_count_by_rule.values())),
                    }
                    report.write(_json_dumps(heartbeat) + "\n")
                    print(_json_dumps(heartbeat), flush=True)

                report.flush()
            except Exception as exc:  # pragma: no cover
                error_entry = f"cycle_error:{exc}"
                monitor_errors.append(error_entry)
                print(_json_dumps({"event": "cycle_error", "timestamp_utc": to_iso(utcnow()), "error": error_entry}), flush=True)

            next_cycle_mono += config.poll_seconds
            cycle_finished_mono = time.monotonic()
            if cycle_finished_mono > next_cycle_mono:
                missed_ticks = int((cycle_finished_mono - next_cycle_mono) // config.poll_seconds)
                if missed_ticks > 0:
                    next_cycle_mono += missed_ticks * config.poll_seconds
    if wallet_ctx_refresh_task is not None and not wallet_ctx_refresh_task.done():
        wallet_ctx_refresh_task.cancel()

    finished_at = utcnow()
    expected_cycles = max(1, int(math.ceil(config.duration_seconds / config.poll_seconds)))
    missed_cycles = max(0, expected_cycles - total_cycles)
    summary = {
        "event": "monitor_complete",
        "started_at_utc": to_iso(started_at),
        "finished_at_utc": to_iso(finished_at),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "target_trader_id": trader_id,
        "target_trader_name": str(trader.name),
        "total_cycles": total_cycles,
        "expected_cycles": expected_cycles,
        "missed_cycles": missed_cycles,
        "cycle_coverage_pct": (float(total_cycles) / float(expected_cycles) * 100.0),
        "alerts_total": int(sum(alerts_count_by_rule.values())),
        "alerts_by_rule": alerts_count_by_rule,
        "provider_checks_enabled": bool(config.enable_provider_checks),
        "monitor_errors": monitor_errors,
        "report_path": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(_json_dumps(summary), flush=True)
    print(_json_dumps({"event": "summary_path", "path": str(summary_path)}), flush=True)
    return 0


def parse_args() -> MonitorConfig:
    parser = argparse.ArgumentParser(description="Run 1s live-trading truth monitor.")
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--trader-id", type=str, default=os.environ.get("TRADER_ID"))
    parser.add_argument("--trader-name", type=str, default=os.environ.get("TRADER_NAME"))
    parser.add_argument(
        "--provider-checks",
        dest="enable_provider_checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable provider and wallet API reconciliation checks (use --no-provider-checks to disable).",
    )
    args = parser.parse_args()
    return MonitorConfig(
        duration_seconds=max(1, int(args.duration_seconds)),
        poll_seconds=max(0.2, float(args.poll_seconds)),
        trader_id=str(args.trader_id).strip() if args.trader_id else None,
        trader_name=str(args.trader_name).strip() if args.trader_name else None,
        enable_provider_checks=bool(args.enable_provider_checks),
    )


def main() -> int:
    config = parse_args()
    return asyncio.run(run_monitor(config))


if __name__ == "__main__":
    raise SystemExit(main())
