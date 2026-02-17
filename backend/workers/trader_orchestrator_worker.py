"""Dedicated trader orchestrator worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from config import settings
from models.database import (
    AppSettings,
    AsyncSessionLocal,
    DiscoveredWallet,
    SimulationAccount,
    TraderOrder,
    TrackedWallet,
    Trader,
    TraderGroupMember,
    init_database,
)
from services.trader_orchestrator.order_manager import submit_order
from services.trader_orchestrator.live_market_context import (
    RuntimeTradeSignalView,
    build_live_signal_contexts,
)
from services.trader_orchestrator.position_lifecycle import reconcile_paper_positions
from services.simulation import simulation_service
from services.trader_orchestrator.risk_manager import evaluate_risk
from services.trader_orchestrator.sources.registry import (
    list_source_aliases,
    normalize_source_key,
)
from services.trader_orchestrator.strategy_catalog import (
    ensure_system_trader_strategies_seeded,
)
from services.trader_orchestrator.strategy_db_loader import strategy_db_loader
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
    get_open_market_ids_for_trader,
    get_open_position_count_for_trader,
    get_trader_signal_cursor,
    list_traders,
    list_unconsumed_trade_signals,
    read_orchestrator_control,
    record_signal_consumption,
    sync_trader_position_inventory,
    upsert_trader_signal_cursor,
    write_orchestrator_snapshot,
)
from services.signal_bus import expire_stale_signals, set_trade_signal_status
from utils.utcnow import utcnow
from utils.secrets import decrypt_secret

logger = logging.getLogger("trader_orchestrator_worker")
_RESUME_POLICIES = {"resume_full", "manage_only", "flatten_then_start"}
_PAPER_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "open"}


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


def _normalize_wallet(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_source_configs(trader: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_configs_raw = trader.get("source_configs")
    if not isinstance(source_configs_raw, list):
        source_configs_raw = []

    normalized: dict[str, dict[str, Any]] = {}
    for raw in source_configs_raw:
        if not isinstance(raw, dict):
            continue
        source_key = normalize_source_key(raw.get("source_key"))
        strategy_key = str(raw.get("strategy_key") or "").strip().lower()
        if not source_key or not strategy_key:
            continue
        normalized[source_key] = {
            "source_key": source_key,
            "strategy_key": strategy_key,
            "strategy_params": dict(raw.get("strategy_params") or {}),
            "traders_scope": dict(raw.get("traders_scope") or {}),
        }
    return normalized


def _query_sources_for_configs(source_configs: dict[str, dict[str, Any]]) -> list[str]:
    if not source_configs:
        return []
    source_aliases = list_source_aliases()
    sources: set[str] = set(source_configs.keys())
    for alias, canonical in source_aliases.items():
        if canonical in source_configs:
            sources.add(alias)
    return sorted(sources)


def _signal_wallets(signal: Any) -> set[str]:
    payload = getattr(signal, "payload_json", None)
    if not isinstance(payload, dict):
        payload = {}

    wallets: set[str] = set()
    for raw in payload.get("wallets") or []:
        normalized = _normalize_wallet(raw)
        if normalized:
            wallets.add(normalized)
    for raw in payload.get("wallet_addresses") or []:
        normalized = _normalize_wallet(raw)
        if normalized:
            wallets.add(normalized)
    for item in payload.get("top_wallets") or []:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_wallet(item.get("address"))
        if normalized:
            wallets.add(normalized)
    return wallets


async def _backfill_simulation_ledger_for_active_paper_orders(
    session: Any,
    *,
    trader_id: str,
    paper_account_id: str | None,
) -> dict[str, Any]:
    if not paper_account_id:
        return {"attempted": 0, "backfilled": 0, "skipped": 0, "errors": []}

    rows = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "paper",
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(_PAPER_ACTIVE_ORDER_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )

    attempted = 0
    backfilled = 0
    skipped = 0
    errors: list[str] = []
    now = utcnow()

    for row in rows:
        payload = dict(row.payload_json or {})
        if isinstance(payload.get("simulation_ledger"), dict):
            continue

        attempted += 1
        entry_price = _safe_float(row.effective_price, None)
        if entry_price is None or entry_price <= 0:
            entry_price = _safe_float(row.entry_price, None)
        notional = _safe_float(row.notional_usd, None)
        if entry_price is None or entry_price <= 0 or notional is None or notional <= 0:
            skipped += 1
            continue

        token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip() or None
        try:
            ledger_entry = await simulation_service.record_orchestrator_paper_fill(
                account_id=paper_account_id,
                trader_id=trader_id,
                signal_id=str(row.signal_id or row.id),
                market_id=str(row.market_id or ""),
                market_question=str(row.market_question or row.market_id or ""),
                direction=str(row.direction or ""),
                notional_usd=float(notional),
                entry_price=float(entry_price),
                strategy_type=str(row.source or "trader_orchestrator_backfill"),
                token_id=token_id,
                payload={
                    "source": str(row.source or ""),
                    "backfilled_from_order_id": str(row.id),
                    "backfilled_at": now.isoformat() + "Z",
                    "edge_percent": _safe_float(row.edge_percent, 0.0) or 0.0,
                    "confidence": _safe_float(row.confidence, 0.0) or 0.0,
                },
                session=session,
                commit=False,
            )
            payload["simulation_ledger"] = ledger_entry
            payload["simulation_backfill"] = {
                "order_id": str(row.id),
                "backfilled_at": now.isoformat() + "Z",
            }
            row.payload_json = payload
            row.updated_at = now
            backfilled += 1
        except Exception as exc:
            errors.append(f"{row.id}:{exc}")

    return {
        "attempted": attempted,
        "backfilled": backfilled,
        "skipped": skipped,
        "errors": errors,
    }


async def _build_traders_scope_context(session: Any, traders_scope: dict[str, Any]) -> dict[str, Any]:
    modes = {
        str(mode or "").strip().lower()
        for mode in (traders_scope.get("modes") or [])
        if str(mode or "").strip()
    }
    context: dict[str, Any] = {
        "modes": modes,
        "individual_wallets": {
            _normalize_wallet(wallet)
            for wallet in (traders_scope.get("individual_wallets") or [])
            if _normalize_wallet(wallet)
        },
        "group_ids": [
            str(group_id or "").strip()
            for group_id in (traders_scope.get("group_ids") or [])
            if str(group_id or "").strip()
        ],
        "tracked_wallets": set(),
        "pool_wallets": set(),
        "group_wallets": set(),
    }

    if "tracked" in modes:
        tracked_rows = (
            await session.execute(select(TrackedWallet.address))
        ).scalars().all()
        context["tracked_wallets"] = {
            _normalize_wallet(address) for address in tracked_rows if _normalize_wallet(address)
        }

    if "pool" in modes:
        pool_rows = (
            await session.execute(
                select(DiscoveredWallet.address).where(DiscoveredWallet.in_top_pool == True)  # noqa: E712
            )
        ).scalars().all()
        context["pool_wallets"] = {
            _normalize_wallet(address) for address in pool_rows if _normalize_wallet(address)
        }

    if "group" in modes and context["group_ids"]:
        group_rows = (
            await session.execute(
                select(TraderGroupMember.wallet_address).where(
                    TraderGroupMember.group_id.in_(context["group_ids"])
                )
            )
        ).scalars().all()
        context["group_wallets"] = {
            _normalize_wallet(address) for address in group_rows if _normalize_wallet(address)
        }

    return context


def _signal_matches_traders_scope(signal: Any, scope_context: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    wallets = _signal_wallets(signal)
    modes: set[str] = set(scope_context.get("modes") or set())
    matched_modes: list[str] = []

    if "tracked" in modes and wallets.intersection(scope_context.get("tracked_wallets") or set()):
        matched_modes.append("tracked")
    if "pool" in modes and wallets.intersection(scope_context.get("pool_wallets") or set()):
        matched_modes.append("pool")
    if "individual" in modes and wallets.intersection(scope_context.get("individual_wallets") or set()):
        matched_modes.append("individual")
    if "group" in modes and wallets.intersection(scope_context.get("group_wallets") or set()):
        matched_modes.append("group")

    payload = {
        "signal_wallets": sorted(wallets),
        "selected_modes": sorted(modes),
        "matched_modes": sorted(matched_modes),
    }
    return bool(matched_modes), payload


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
        source_configs = _normalize_source_configs(trader)
        if not source_configs:
            return 0, 0
        default_source_config = next(iter(source_configs.values()))
        default_strategy_params = dict(default_source_config.get("strategy_params") or {})
        risk_limits = dict(trader.get("risk_limits") or {})
        metadata = dict(trader.get("metadata") or {})
        run_mode = str(control.get("mode") or "paper").strip().lower()
        resume_policy = _normalize_resume_policy(metadata.get("resume_policy"))
        open_positions = 0
        open_market_ids: set[str] = set()

        if run_mode == "paper":
            backfill_result = await _backfill_simulation_ledger_for_active_paper_orders(
                session,
                trader_id=trader_id,
                paper_account_id=str((control.get("settings") or {}).get("paper_account_id") or "").strip() or None,
            )
            if backfill_result.get("backfilled"):
                await session.commit()
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="paper_ledger_backfill",
                    source="worker",
                    message=(
                        f"Backfilled {int(backfill_result['backfilled'])} paper order(s) into simulation ledger"
                    ),
                    payload=backfill_result,
                )

            force_flatten = resume_policy == "flatten_then_start"
            lifecycle_result = await reconcile_paper_positions(
                session,
                trader_id=trader_id,
                trader_params=default_strategy_params,
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
        open_market_ids = await get_open_market_ids_for_trader(
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

        max_signals_per_cycle = int(
            max(
                1,
                min(
                    500,
                    _safe_int(default_strategy_params.get("max_signals_per_cycle"), 200),
                ),
            )
        )
        scan_batch_size = int(
            max(
                1,
                min(
                    500,
                    _safe_int(default_strategy_params.get("scan_batch_size"), max_signals_per_cycle),
                ),
            )
        )
        sources = _query_sources_for_configs(source_configs)

        control_settings = control.get("settings") or {}
        paper_account_id = str(control_settings.get("paper_account_id") or "").strip() or None
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
        allow_averaging = bool(effective_risk_limits.get("allow_averaging", False))

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

        traders_scope_context: dict[str, Any] | None = None
        traders_source_config = source_configs.get("traders")
        if traders_source_config is not None:
            traders_scope_context = await _build_traders_scope_context(
                session,
                dict(traders_source_config.get("traders_scope") or {}),
            )

        cursor_created_at, cursor_signal_id = await get_trader_signal_cursor(
            session,
            trader_id=trader_id,
        )
        processed_signals = 0

        while processed_signals < max_signals_per_cycle:
            batch_limit = min(scan_batch_size, max_signals_per_cycle - processed_signals)
            signals = await list_unconsumed_trade_signals(
                session,
                trader_id=trader_id,
                sources=sources,
                statuses=["pending"],
                cursor_created_at=cursor_created_at,
                cursor_signal_id=cursor_signal_id,
                limit=batch_limit,
            )
            if not signals:
                break

            live_contexts: dict[str, dict[str, Any]] = {}
            if enable_live_market_context:
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
                signal_id = str(signal.id)
                signal_source = normalize_source_key(getattr(signal, "source", ""))
                source_config = source_configs.get(signal_source)

                try:
                    if source_config is None:
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            outcome="skipped",
                            reason="No source configuration for signal source",
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=signal.created_at,
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await session.commit()
                        cursor_created_at = signal.created_at
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        continue

                    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
                    strategy_params = dict(source_config.get("strategy_params") or {})
                    strategy_status = strategy_db_loader.get_availability(strategy_key)
                    resolved_strategy_key = (
                        strategy_status.resolved_key or strategy_key
                    )
                    live_context = live_contexts.get(signal_id, {})
                    runtime_signal = RuntimeTradeSignalView(signal, live_context=live_context)
                    runtime_signal.source = signal_source
                    traders_scope_payload: dict[str, Any] | None = None

                    if not strategy_status.available:
                        blocked_reason = f"strategy_unavailable:{resolved_strategy_key}"
                        checks_payload = [
                            {
                                "check_key": "strategy_available",
                                "check_label": "Strategy available",
                                "passed": False,
                                "score": None,
                                "detail": str(strategy_status.reason or blocked_reason),
                                "payload": {
                                    "requested_strategy_key": strategy_key,
                                    "resolved_strategy_key": resolved_strategy_key,
                                },
                            }
                        ]
                        decision_row = await create_trader_decision(
                            session,
                            trader_id=trader_id,
                            signal=runtime_signal,
                            strategy_key=resolved_strategy_key,
                            decision="blocked",
                            reason=blocked_reason,
                            score=0.0,
                            checks_summary={"count": len(checks_payload)},
                            risk_snapshot={},
                            payload={
                                "source_key": signal_source,
                                "source_config": source_config,
                                "strategy_runtime_error": strategy_status.reason,
                            },
                            commit=False,
                        )
                        decisions_written += 1
                        await create_trader_decision_checks(
                            session,
                            decision_id=decision_row.id,
                            checks=checks_payload,
                            commit=False,
                        )
                        await set_trade_signal_status(
                            session,
                            signal_id=signal_id,
                            status="skipped",
                            commit=False,
                        )
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            decision_id=decision_row.id,
                            outcome="blocked",
                            reason=blocked_reason,
                            commit=False,
                        )
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="strategy_unavailable",
                            severity="warn",
                            source=signal_source,
                            message=blocked_reason,
                            payload={
                                "decision_id": decision_row.id,
                                "signal_id": signal.id,
                                "requested_strategy_key": strategy_key,
                                "resolved_strategy_key": resolved_strategy_key,
                                "error": strategy_status.reason,
                            },
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=signal.created_at,
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await session.commit()
                        cursor_created_at = signal.created_at
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        continue

                    strategy = strategy_db_loader.get_strategy(resolved_strategy_key)
                    if strategy is None:
                        blocked_reason = f"strategy_unavailable:{resolved_strategy_key}"
                        checks_payload = [
                            {
                                "check_key": "strategy_available",
                                "check_label": "Strategy available",
                                "passed": False,
                                "score": None,
                                "detail": "Strategy cache miss",
                                "payload": {
                                    "requested_strategy_key": strategy_key,
                                    "resolved_strategy_key": resolved_strategy_key,
                                },
                            }
                        ]
                        decision_row = await create_trader_decision(
                            session,
                            trader_id=trader_id,
                            signal=runtime_signal,
                            strategy_key=resolved_strategy_key,
                            decision="blocked",
                            reason=blocked_reason,
                            score=0.0,
                            checks_summary={"count": len(checks_payload)},
                            risk_snapshot={},
                            payload={
                                "source_key": signal_source,
                                "source_config": source_config,
                                "strategy_runtime_error": "strategy cache miss",
                            },
                            commit=False,
                        )
                        decisions_written += 1
                        await create_trader_decision_checks(
                            session,
                            decision_id=decision_row.id,
                            checks=checks_payload,
                            commit=False,
                        )
                        await set_trade_signal_status(
                            session,
                            signal_id=signal_id,
                            status="skipped",
                            commit=False,
                        )
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="strategy_unavailable",
                            severity="warn",
                            source=signal_source,
                            message=blocked_reason,
                            payload={
                                "decision_id": decision_row.id,
                                "signal_id": signal.id,
                                "requested_strategy_key": strategy_key,
                                "resolved_strategy_key": resolved_strategy_key,
                                "error": "strategy cache miss",
                            },
                            commit=False,
                        )
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            decision_id=decision_row.id,
                            outcome="blocked",
                            reason=blocked_reason,
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=signal.created_at,
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await session.commit()
                        cursor_created_at = signal.created_at
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        continue

                    if signal_source == "traders":
                        if traders_scope_context is None:
                            traders_scope_context = await _build_traders_scope_context(
                                session,
                                dict(source_config.get("traders_scope") or {}),
                            )
                        scope_ok, scope_payload = _signal_matches_traders_scope(runtime_signal, traders_scope_context)
                        traders_scope_payload = scope_payload
                        if not scope_ok:
                            checks_payload = [
                                {
                                    "check_key": "traders_scope",
                                    "check_label": "Traders scope",
                                    "passed": False,
                                    "score": None,
                                    "detail": "Signal wallets did not match selected traders_scope modes.",
                                    "payload": scope_payload,
                                }
                            ]
                            decision_row = await create_trader_decision(
                                session,
                                trader_id=trader_id,
                                signal=runtime_signal,
                                strategy_key=resolved_strategy_key,
                                decision="skipped",
                                reason="Signal excluded by traders_scope",
                                score=0.0,
                                checks_summary={"count": len(checks_payload)},
                                risk_snapshot={},
                                payload={
                                    "source_key": signal_source,
                                    "source_config": source_config,
                                    "traders_scope": scope_payload,
                                },
                                commit=False,
                            )
                            decisions_written += 1
                            await create_trader_decision_checks(
                                session,
                                decision_id=decision_row.id,
                                checks=checks_payload,
                                commit=False,
                            )
                            await record_signal_consumption(
                                session,
                                trader_id=trader_id,
                                signal_id=signal_id,
                                decision_id=decision_row.id,
                                outcome="skipped",
                                reason="Signal excluded by traders_scope",
                                commit=False,
                            )
                            await create_trader_event(
                                session,
                                trader_id=trader_id,
                                event_type="decision",
                                source=signal_source,
                                message="Signal excluded by traders_scope",
                                payload={
                                    "decision_id": decision_row.id,
                                    "signal_id": signal.id,
                                    "decision": "skipped",
                                    "order_status": None,
                                    "traders_scope": scope_payload,
                                },
                                commit=False,
                            )
                            await upsert_trader_signal_cursor(
                                session,
                                trader_id=trader_id,
                                last_signal_created_at=signal.created_at,
                                last_signal_id=signal_id,
                                commit=False,
                            )
                            await session.commit()
                            cursor_created_at = signal.created_at
                            cursor_signal_id = signal_id
                            processed_signals += 1
                            continue

                    decision_obj = strategy.evaluate(
                        runtime_signal,
                        {
                            "params": strategy_params,
                            "trader": trader,
                            "mode": control.get("mode", "paper"),
                            "live_market": live_context,
                            "source_config": source_config,
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

                    if final_decision == "selected" and not allow_averaging:
                        signal_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
                        stacking_blocked = bool(signal_market_id) and signal_market_id in open_market_ids
                        checks_payload.append(
                            {
                                "check_key": "stacking_guard",
                                "check_label": "One active entry per market",
                                "passed": not stacking_blocked,
                                "score": None,
                                "detail": (
                                    "allow_averaging=false and market already has an open position"
                                    if stacking_blocked
                                    else "allow_averaging=false and no open position exists for this market"
                                ),
                                "payload": {
                                    "allow_averaging": False,
                                    "market_id": signal_market_id or None,
                                },
                            }
                        )
                        if stacking_blocked:
                            final_decision = "blocked"
                            final_reason = "Stacking guard: market already open while allow_averaging=false"

                    if bool(control.get("kill_switch")) and final_decision == "selected":
                        final_decision = "blocked"
                        final_reason = "Kill switch is enabled"

                    decision_row = await create_trader_decision(
                        session,
                        trader_id=trader_id,
                        signal=runtime_signal,
                        strategy_key=resolved_strategy_key,
                        decision=final_decision,
                        reason=final_reason,
                        score=score,
                        checks_summary={"count": len(checks_payload)},
                        risk_snapshot=risk_snapshot,
                        payload={
                            "source_key": signal_source,
                            "source_config": source_config,
                            "strategy_payload": decision_obj.payload,
                            "size_usd": size_usd,
                            "traders_scope": traders_scope_payload,
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
                        commit=False,
                    )
                    decisions_written += 1

                    await create_trader_decision_checks(
                        session,
                        decision_id=decision_row.id,
                        checks=checks_payload,
                        commit=False,
                    )

                    order_status = None
                    if final_decision == "selected":
                        await set_trade_signal_status(
                            session,
                            signal_id=signal_id,
                            status="selected",
                            commit=False,
                        )
                        status, effective_price, error_message, execution_payload = await submit_order(
                            mode=str(control.get("mode", "paper")),
                            signal=runtime_signal,
                            size_usd=size_usd,
                        )
                        normalized_order_status = str(status or "").strip().lower()

                        if run_mode == "paper" and normalized_order_status in {"submitted", "executed", "open"}:
                            if not paper_account_id:
                                status = "failed"
                                normalized_order_status = "failed"
                                error_message = "Paper account is not configured; set paper_account_id to execute paper trades."
                            else:
                                entry_price = _safe_float(effective_price, None)
                                if entry_price is None or entry_price <= 0:
                                    entry_price = _safe_float(getattr(runtime_signal, "entry_price", None), None)
                                if entry_price is None or entry_price <= 0:
                                    live_price = None
                                    if isinstance(live_context, dict):
                                        live_price = _safe_float(live_context.get("live_selected_price"), None)
                                    entry_price = live_price

                                signal_payload = getattr(runtime_signal, "payload_json", None)
                                signal_payload = signal_payload if isinstance(signal_payload, dict) else {}
                                signal_live_context = (
                                    runtime_signal.live_context
                                    if isinstance(runtime_signal.live_context, dict)
                                    else {}
                                )
                                token_id = (
                                    str(
                                        signal_live_context.get("selected_token_id")
                                        or signal_payload.get("selected_token_id")
                                        or signal_payload.get("token_id")
                                        or ""
                                    ).strip()
                                    or None
                                )

                                if entry_price is None or entry_price <= 0:
                                    status = "failed"
                                    normalized_order_status = "failed"
                                    error_message = "Paper execution missing a valid entry price."
                                else:
                                    try:
                                        ledger_entry = await simulation_service.record_orchestrator_paper_fill(
                                            account_id=paper_account_id,
                                            trader_id=trader_id,
                                            signal_id=signal_id,
                                            market_id=str(getattr(runtime_signal, "market_id", "") or ""),
                                            market_question=str(getattr(runtime_signal, "market_question", "") or ""),
                                            direction=str(getattr(runtime_signal, "direction", "") or ""),
                                            notional_usd=size_usd,
                                            entry_price=float(entry_price),
                                            strategy_type=resolved_strategy_key,
                                            token_id=token_id,
                                            payload={
                                                "source": signal_source,
                                                "edge_percent": _safe_float(getattr(runtime_signal, "edge_percent", None), 0.0) or 0.0,
                                                "confidence": _safe_float(getattr(runtime_signal, "confidence", None), 0.0) or 0.0,
                                            },
                                            session=session,
                                            commit=False,
                                        )
                                        payload_copy = dict(execution_payload or {})
                                        payload_copy["simulation_ledger"] = ledger_entry
                                        execution_payload = payload_copy
                                    except Exception as exc:
                                        logger.warning(
                                            "Paper simulation ledger write failed for trader %s signal %s: %s",
                                            trader_id,
                                            signal_id,
                                            exc,
                                        )
                                        status = "failed"
                                        normalized_order_status = "failed"
                                        error_message = str(exc)
                                        payload_copy = dict(execution_payload or {})
                                        payload_copy["simulation_ledger_error"] = str(exc)
                                        execution_payload = payload_copy

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
                            commit=False,
                        )
                        if normalized_order_status in {"submitted", "open"}:
                            await set_trade_signal_status(
                                session,
                                signal_id=signal_id,
                                status="submitted",
                                effective_price=effective_price,
                                commit=False,
                            )
                        elif normalized_order_status == "executed":
                            await set_trade_signal_status(
                                session,
                                signal_id=signal_id,
                                status="executed",
                                effective_price=effective_price,
                                commit=False,
                            )
                        elif normalized_order_status == "failed":
                            await set_trade_signal_status(
                                session,
                                signal_id=signal_id,
                                status="failed",
                                effective_price=effective_price,
                                commit=False,
                            )

                        if normalized_order_status in {"submitted", "executed", "open"}:
                            open_positions = await get_open_position_count_for_trader(
                                session,
                                trader_id,
                                mode=run_mode,
                            )
                            opened_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
                            if opened_market_id:
                                open_market_ids.add(opened_market_id)
                        orders_written += 1
                        order_status = status

                    await record_signal_consumption(
                        session,
                        trader_id=trader_id,
                        signal_id=signal_id,
                        decision_id=decision_row.id,
                        outcome=order_status or final_decision,
                        reason=final_reason,
                        commit=False,
                    )

                    await create_trader_event(
                        session,
                        trader_id=trader_id,
                        event_type="decision",
                        source=signal_source,
                        message=final_reason,
                        payload={
                            "decision_id": decision_row.id,
                            "signal_id": signal.id,
                            "decision": final_decision,
                            "order_status": order_status,
                        },
                        commit=False,
                    )
                    await upsert_trader_signal_cursor(
                        session,
                        trader_id=trader_id,
                        last_signal_created_at=signal.created_at,
                        last_signal_id=signal_id,
                        commit=False,
                    )
                    await session.commit()
                    cursor_created_at = signal.created_at
                    cursor_signal_id = signal_id
                    processed_signals += 1
                except Exception:
                    await session.rollback()
                    logger.exception(
                        "Trader %s failed to process signal %s",
                        trader_id,
                        signal_id,
                    )
                    await upsert_trader_signal_cursor(
                        session,
                        trader_id=trader_id,
                        last_signal_created_at=signal.created_at,
                        last_signal_id=signal_id,
                        commit=False,
                    )
                    await create_trader_event(
                        session,
                        trader_id=trader_id,
                        event_type="decision_error",
                        severity="warn",
                        source=signal_source,
                        message="Signal processing failed; advanced cursor to avoid hard loop.",
                        payload={"signal_id": signal_id},
                        commit=False,
                    )
                    await session.commit()
                    cursor_created_at = signal.created_at
                    cursor_signal_id = signal_id
                    processed_signals += 1

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

    try:
        async with AsyncSessionLocal() as session:
            seeded = await ensure_system_trader_strategies_seeded(session)
        if seeded > 0:
            logger.info("Seeded/updated %s trader strategy definitions", seeded)
    except Exception as exc:
        logger.warning("Failed to ensure trader strategy definitions: %s", exc)

    while True:
        try:
            async with AsyncSessionLocal() as session:
                await expire_stale_signals(session)
                try:
                    await strategy_db_loader.refresh_from_db(session=session)
                except Exception as exc:
                    logger.warning("Failed to refresh DB strategy registry: %s", exc)

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
