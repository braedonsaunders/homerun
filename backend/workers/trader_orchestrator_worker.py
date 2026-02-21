"""Dedicated trader orchestrator worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, text

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
from services.trader_orchestrator.live_market_context import (
    RuntimeTradeSignalView,
    build_live_signal_contexts,
)
from services.trader_orchestrator.session_engine import ExecutionSessionEngine
from services.trader_orchestrator.position_lifecycle import reconcile_live_positions, reconcile_paper_positions
from services.simulation import simulation_service
from services.trader_orchestrator.risk_manager import evaluate_risk
from services.trader_orchestrator.decision_gates import (
    apply_platform_decision_gates,
    is_within_trading_window_utc,
)
from services.worker_state import _commit_with_retry, _is_sqlite_lock_error
from services.trader_orchestrator.sources.registry import (
    normalize_source_key,
)
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_loader import strategy_loader
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.trader_orchestrator_state import (
    ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
    cleanup_trader_open_orders,
    compute_orchestrator_metrics,
    create_trader_decision,
    create_trader_decision_checks,
    create_trader_event,
    create_trader_order as _create_trader_order,
    get_consecutive_loss_count,
    get_daily_realized_pnl,
    get_gross_exposure,
    get_unrealized_pnl,
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
    update_orchestrator_control,
    upsert_trader_signal_cursor,
    write_orchestrator_snapshot,
)
from services.signal_bus import expire_stale_signals, set_trade_signal_status
from utils.utcnow import utcnow
from utils.converters import safe_float, safe_int
from utils.secrets import decrypt_secret
from sqlalchemy.exc import OperationalError

logger = logging.getLogger("trader_orchestrator_worker")
strategy_db_loader = strategy_loader
create_trader_order = _create_trader_order
_RESUME_POLICIES = {"resume_full", "manage_only", "flatten_then_start"}
_PAPER_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "open"}
_ORPHAN_CLEANUP_MAX_TRADERS_PER_CYCLE = 32
_ORCHESTRATOR_CYCLE_LOCK_KEY = 0x54524F5243485354  # "TRORCHST"
_ORCHESTRATOR_SQLITE_BUSY_TIMEOUT_MS = 250
_orchestrator_lock_ctx: tuple[Any, Any] | None = None
_TRADER_IDLE_MAINTENANCE_INTERVAL_SECONDS = 60
_trader_idle_maintenance_last_run: dict[str, datetime] = {}


def _session_dialect_name(session: Any) -> str:
    try:
        bind = session.get_bind()
        return str(getattr(getattr(bind, "dialect", None), "name", "") or "").lower()
    except Exception:
        return ""


async def _apply_orchestrator_sqlite_busy_timeout(session: Any) -> None:
    if "sqlite" not in settings.DATABASE_URL.lower():
        return
    if not hasattr(session, "execute"):
        return
    try:
        await session.execute(text(f"PRAGMA busy_timeout={_ORCHESTRATOR_SQLITE_BUSY_TIMEOUT_MS}"))
    except Exception:
        return


async def _write_orchestrator_snapshot_best_effort(session: Any, **snapshot_kwargs: Any) -> None:
    try:
        await write_orchestrator_snapshot(session, **snapshot_kwargs)
    except OperationalError as exc:
        if not _is_sqlite_lock_error(exc):
            raise
        if hasattr(session, "rollback"):
            await session.rollback()
        logger.warning("Skipped orchestrator snapshot write due to SQLite lock")


async def submit_order(
    *,
    session_engine: ExecutionSessionEngine,
    trader_id: str,
    signal: RuntimeTradeSignalView,
    decision_id: str,
    strategy_key: str,
    strategy_params: dict[str, Any],
    risk_limits: dict[str, Any],
    mode: str,
    size_usd: float,
    reason: str,
):
    return await session_engine.execute_signal(
        trader_id=trader_id,
        signal=signal,
        decision_id=decision_id,
        strategy_key=strategy_key,
        strategy_params=strategy_params,
        risk_limits=risk_limits,
        mode=mode,
        size_usd=size_usd,
        reason=reason,
    )


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


def _strategy_instance_from_loaded(candidate: Any) -> Any:
    if candidate is None:
        return None
    instance = getattr(candidate, "instance", None)
    if instance is not None:
        return instance
    return candidate


async def _try_acquire_orchestrator_cycle_lock(session: Any) -> bool:
    """Acquire a cross-process lock when running on PostgreSQL.

    For non-PostgreSQL dialects (e.g. sqlite in local tests), fall back to
    allowing the cycle to proceed.
    """
    dialect_name = _session_dialect_name(session)

    if dialect_name != "postgresql":
        return True

    try:
        result = await session.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": int(_ORCHESTRATOR_CYCLE_LOCK_KEY)},
        )
        return bool(result.scalar())
    except Exception as exc:
        logger.warning("Unable to acquire orchestrator cycle advisory lock: %s", exc)
        return False


async def _release_orchestrator_cycle_lock(session: Any) -> None:
    """Release the PostgreSQL advisory lock, if applicable."""
    dialect_name = _session_dialect_name(session)

    if dialect_name != "postgresql":
        return

    try:
        await session.execute(
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": int(_ORCHESTRATOR_CYCLE_LOCK_KEY)},
        )
    except Exception as exc:
        logger.warning("Unable to release orchestrator cycle advisory lock: %s", exc)


async def _ensure_orchestrator_cycle_lock_owner() -> bool:
    """Own the cross-process orchestrator lock for this worker process."""
    global _orchestrator_lock_ctx

    if _orchestrator_lock_ctx is not None:
        return True

    session_ctx = AsyncSessionLocal()
    session = await session_ctx.__aenter__()
    keep_lock_session = False
    try:
        acquired = await _try_acquire_orchestrator_cycle_lock(session)
        if not acquired:
            return False

        if _session_dialect_name(session) == "postgresql":
            _orchestrator_lock_ctx = (session_ctx, session)
            keep_lock_session = True
            logger.info("Acquired orchestrator cross-process cycle lock")

        return True
    finally:
        if not keep_lock_session:
            await session_ctx.__aexit__(None, None, None)


async def _release_orchestrator_cycle_lock_owner() -> None:
    """Release owned lock session on shutdown/error."""
    global _orchestrator_lock_ctx
    if _orchestrator_lock_ctx is None:
        return

    session_ctx, session = _orchestrator_lock_ctx
    _orchestrator_lock_ctx = None
    try:
        await _release_orchestrator_cycle_lock(session)
        logger.info("Released orchestrator cross-process cycle lock")
    finally:
        await session_ctx.__aexit__(None, None, None)


def _query_sources_for_configs(source_configs: dict[str, dict[str, Any]]) -> list[str]:
    if not source_configs:
        return []
    return sorted(source_configs.keys())


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
        entry_price = safe_float(row.effective_price, None)
        if entry_price is None or entry_price <= 0:
            entry_price = safe_float(row.entry_price, None)
        notional = safe_float(row.notional_usd, None)
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
                    "edge_percent": safe_float(row.edge_percent, 0.0) or 0.0,
                    "confidence": safe_float(row.confidence, 0.0) or 0.0,
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
    modes = {str(mode or "").strip().lower() for mode in (traders_scope.get("modes") or []) if str(mode or "").strip()}
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
        tracked_rows = (await session.execute(select(TrackedWallet.address))).scalars().all()
        context["tracked_wallets"] = {
            _normalize_wallet(address) for address in tracked_rows if _normalize_wallet(address)
        }

    if "pool" in modes:
        pool_rows = (
            (
                await session.execute(
                    select(DiscoveredWallet.address).where(DiscoveredWallet.in_top_pool == True)  # noqa: E712
                )
            )
            .scalars()
            .all()
        )
        context["pool_wallets"] = {_normalize_wallet(address) for address in pool_rows if _normalize_wallet(address)}

    if "group" in modes and context["group_ids"]:
        group_rows = (
            (
                await session.execute(
                    select(TraderGroupMember.wallet_address).where(TraderGroupMember.group_id.in_(context["group_ids"]))
                )
            )
            .scalars()
            .all()
        )
        context["group_wallets"] = {_normalize_wallet(address) for address in group_rows if _normalize_wallet(address)}

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
        await _apply_orchestrator_sqlite_busy_timeout(session)
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
        sources = _query_sources_for_configs(source_configs)
        cursor_created_at, cursor_signal_id = await get_trader_signal_cursor(
            session,
            trader_id=trader_id,
        )
        prefetched_signals: list[Any] | None = None
        if process_signals and sources:
            pending_preview = await list_unconsumed_trade_signals(
                session,
                trader_id=trader_id,
                sources=sources,
                statuses=["pending"],
                cursor_created_at=cursor_created_at,
                cursor_signal_id=cursor_signal_id,
                limit=1,
            )
            if pending_preview:
                prefetched_signals = pending_preview
            else:
                now = utcnow()
                idle_maintenance_interval_seconds = int(
                    max(
                        15,
                        min(
                            900,
                            safe_int(
                                default_strategy_params.get("idle_maintenance_interval_seconds"),
                                _TRADER_IDLE_MAINTENANCE_INTERVAL_SECONDS,
                            ),
                        ),
                    )
                )
                last_idle_maintenance = _trader_idle_maintenance_last_run.get(trader_id)
                if last_idle_maintenance is not None:
                    elapsed_seconds = (now - last_idle_maintenance).total_seconds()
                    if elapsed_seconds < idle_maintenance_interval_seconds:
                        return 0, 0
                _trader_idle_maintenance_last_run[trader_id] = now
        open_positions = 0
        open_market_ids: set[str] = set()

        if run_mode == "paper":
            backfill_result = await _backfill_simulation_ledger_for_active_paper_orders(
                session,
                trader_id=trader_id,
                paper_account_id=str((control.get("settings") or {}).get("paper_account_id") or "").strip() or None,
            )
            if backfill_result.get("backfilled"):
                await _commit_with_retry(session)
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="paper_ledger_backfill",
                    source="worker",
                    message=(f"Backfilled {int(backfill_result['backfilled'])} paper order(s) into simulation ledger"),
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
        elif run_mode == "live":
            force_flatten = resume_policy == "flatten_then_start"
            lifecycle_result = await reconcile_live_positions(
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
                    event_type="live_positions_closed",
                    source="worker",
                    message=f"Closed {closed_positions} live position(s)",
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
        session_engine = ExecutionSessionEngine(session)
        if hasattr(session, "execute"):
            reconcile_result = await session_engine.reconcile_active_sessions(
                mode=run_mode,
                trader_id=trader_id,
            )
            if reconcile_result.get("expired", 0) > 0:
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="execution_sessions_expired",
                    severity="warn",
                    source="worker",
                    message=f"Expired {int(reconcile_result['expired'])} execution session(s)",
                    payload=reconcile_result,
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
                block_entries_reason = f"Resume policy flatten_then_start blocked: {open_positions} open live position(s) require manual flattening"

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
                    safe_int(default_strategy_params.get("max_signals_per_cycle"), 200),
                ),
            )
        )
        scan_batch_size = int(
            max(
                1,
                min(
                    500,
                    safe_int(default_strategy_params.get("scan_batch_size"), max_signals_per_cycle),
                ),
            )
        )
        control_settings = control.get("settings") or {}
        enable_live_market_context = bool(control_settings.get("enable_live_market_context", True))
        history_window_seconds = int(
            max(
                300,
                min(
                    21600,
                    safe_int(
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
                    safe_int(
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
                    safe_int(
                        control_settings.get("live_market_history_max_points", 120),
                        120,
                    ),
                ),
            )
        )
        live_market_context_timeout_seconds = float(
            max(
                1.0,
                min(
                    12.0,
                    safe_float(
                        control_settings.get("live_market_context_timeout_seconds"),
                        4.0,
                    )
                    or 4.0,
                ),
            )
        )
        live_market_context_request_timeout_seconds = float(
            max(
                0.5,
                min(
                    4.0,
                    live_market_context_timeout_seconds / 2.0,
                ),
            )
        )

        now_utc = datetime.now(timezone.utc)
        trading_window_ok = is_within_trading_window_utc(metadata, now_utc)
        global_limits = dict((control.get("settings") or {}).get("global_risk") or {})
        effective_risk_limits = dict(risk_limits)
        if "max_orders_per_cycle" not in effective_risk_limits:
            fallback_cycle_limit = safe_int(global_limits.get("max_orders_per_cycle"), 50)
            if fallback_cycle_limit > 0:
                effective_risk_limits["max_orders_per_cycle"] = fallback_cycle_limit
        if "max_daily_loss_usd" not in effective_risk_limits:
            fallback_daily_loss = safe_float(global_limits.get("max_daily_loss_usd"), 0.0) or 0.0
            if fallback_daily_loss > 0:
                effective_risk_limits["max_daily_loss_usd"] = fallback_daily_loss
        allow_averaging = bool(effective_risk_limits.get("allow_averaging", False))

        global_daily_pnl = await get_daily_realized_pnl(session, mode=run_mode)
        trader_daily_pnl = await get_daily_realized_pnl(session, trader_id=trader_id, mode=run_mode)
        # Compute unrealized PnL (mark-to-market) for open positions so the
        # daily loss limit accounts for positions that are deeply underwater.
        try:
            global_unrealized_pnl = await get_unrealized_pnl(session, mode=run_mode)
            trader_unrealized_pnl = await get_unrealized_pnl(session, trader_id=trader_id, mode=run_mode)
        except Exception:
            global_unrealized_pnl = 0.0
            trader_unrealized_pnl = 0.0
        # Track cumulative notional committed within this cycle so that
        # subsequent risk checks account for intra-cycle exposure even
        # before PnL is realized in the database.
        intra_cycle_committed_usd: float = 0.0
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
        cooldown_seconds = max(0, safe_int(effective_risk_limits.get("cooldown_seconds"), 0))
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

        processed_signals = 0

        # Kill switch is the very first gate: short-circuit before any signal
        # fetching, live-market context building, or risk evaluation.
        if bool(control.get("kill_switch")):
            logger.info(
                "Kill switch active for trader %s — skipping all signal processing",
                trader_id,
            )
            return 0, 0

        while processed_signals < max_signals_per_cycle:
            batch_limit = min(scan_batch_size, max_signals_per_cycle - processed_signals)
            if prefetched_signals is not None:
                signals = prefetched_signals[:batch_limit]
                prefetched_signals = None
            else:
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
                    live_contexts = await asyncio.wait_for(
                        build_live_signal_contexts(
                            context_candidates,
                            history_window_seconds=history_window_seconds,
                            history_fidelity_seconds=history_fidelity_seconds,
                            max_history_points=max_history_points,
                            market_fetch_timeout_seconds=live_market_context_request_timeout_seconds,
                            prices_batch_timeout_seconds=live_market_context_request_timeout_seconds,
                            history_fetch_timeout_seconds=live_market_context_request_timeout_seconds,
                        ),
                        timeout=live_market_context_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Live market context refresh failed: %s", exc)
                    live_contexts = {}

            for signal in signals:
                signal_id = str(signal.id)
                signal_source = normalize_source_key(getattr(signal, "source", ""))
                source_config = source_configs.get(signal_source)
                strategy_key = ""
                resolved_strategy_key = ""

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
                        await _commit_with_retry(session)
                        cursor_created_at = signal.created_at
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        continue

                    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
                    strategy_params = dict(source_config.get("strategy_params") or {})
                    strategy_status = strategy_loader.get_availability(strategy_key)
                    resolved_strategy_key = strategy_status.resolved_key or strategy_key
                    live_context = live_contexts.get(signal_id, {})
                    runtime_signal = RuntimeTradeSignalView(signal, live_context=live_context)
                    runtime_signal.source = signal_source
                    traders_scope_payload: dict[str, Any] | None = None

                    # ── Strategy resolution (unified loader) ─────────────
                    strategy = None

                    # 1. Try the configured strategy_key
                    if strategy_status.available:
                        loaded = strategy_loader.get_strategy(resolved_strategy_key)
                        strategy = _strategy_instance_from_loaded(loaded)

                    # 2. Fallback: try the signal's strategy_type slug
                    if strategy is None and strategy_status.available:
                        signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
                        if signal_strategy_type:
                            loaded = strategy_loader.get_strategy(signal_strategy_type)
                            candidate = _strategy_instance_from_loaded(loaded)
                            if candidate is not None and hasattr(candidate, "evaluate"):
                                strategy = candidate

                    # 3. Final fallback: try source key as slug
                    if strategy is None and strategy_status.available:
                        loaded = strategy_loader.get_strategy(signal_source)
                        candidate = _strategy_instance_from_loaded(loaded)
                        if candidate is not None and hasattr(candidate, "evaluate"):
                            strategy = candidate

                    if strategy is None:
                        blocked_reason = f"strategy_unavailable:{resolved_strategy_key}"
                        strategy_detail = str(strategy_status.reason or blocked_reason)
                        if strategy_status.available:
                            strategy_detail = "Strategy cache miss"
                        checks_payload = [
                            {
                                "check_key": "strategy_available",
                                "check_label": "Strategy available",
                                "passed": False,
                                "score": None,
                                "detail": strategy_detail,
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
                                "strategy_runtime_error": strategy_detail,
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
                                "error": strategy_detail,
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
                        await _commit_with_retry(session)
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
                            await _commit_with_retry(session)
                            cursor_created_at = signal.created_at
                            cursor_signal_id = signal_id
                            processed_signals += 1
                            continue

                    loop = asyncio.get_running_loop()
                    decision_obj = await loop.run_in_executor(
                        None,
                        strategy.evaluate,
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
                        drift_score = safe_float(drift_pct)
                        max_drift = safe_float(
                            effective_risk_limits.get("max_entry_drift_pct"),
                            10.0,
                        )
                        checks_payload.append(
                            {
                                "check_key": "live_entry_drift",
                                "check_label": "Entry drift from signal",
                                "passed": drift_score is None or abs(drift_score) <= max_drift,
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

                    risk_runtime_payload = {
                        "global_daily_realized_pnl_usd": global_daily_pnl,
                        "trader_daily_realized_pnl_usd": trader_daily_pnl,
                        "intra_cycle_committed_usd": intra_cycle_committed_usd,
                        "trader_consecutive_losses": trader_loss_streak,
                        "cooldown_seconds": cooldown_seconds,
                        "cooldown_active": cooldown_active,
                        "cooldown_remaining_seconds": cooldown_remaining_seconds,
                        "trader_open_positions": open_positions,
                        "trading_window_ok": trading_window_ok,
                    }

                    risk_evaluator = None
                    if decision_obj.decision == "selected" and trading_window_ok:
                        gross_exposure = await get_gross_exposure(session, mode=run_mode)
                        market_exposure = await get_market_exposure(session, str(signal.market_id), mode=run_mode)
                        adjusted_global_daily_pnl = global_daily_pnl - intra_cycle_committed_usd
                        adjusted_trader_daily_pnl = trader_daily_pnl - intra_cycle_committed_usd
                        risk_snapshot_base = {
                            "global_daily_realized_pnl_usd": global_daily_pnl,
                            "trader_daily_realized_pnl_usd": trader_daily_pnl,
                            "global_unrealized_pnl_usd": global_unrealized_pnl,
                            "trader_unrealized_pnl_usd": trader_unrealized_pnl,
                            "intra_cycle_committed_usd": intra_cycle_committed_usd,
                            "adjusted_global_daily_pnl_usd": adjusted_global_daily_pnl,
                            "adjusted_trader_daily_pnl_usd": adjusted_trader_daily_pnl,
                            "trader_consecutive_losses": trader_loss_streak,
                            "cooldown_seconds": cooldown_seconds,
                            "cooldown_active": cooldown_active,
                            "cooldown_remaining_seconds": cooldown_remaining_seconds,
                            "trader_open_positions": open_positions,
                        }

                        def _evaluate_runtime_risk(size_for_eval: float):
                            risk_result = evaluate_risk(
                                size_usd=size_for_eval,
                                gross_exposure_usd=gross_exposure,
                                trader_open_positions=open_positions,
                                market_exposure_usd=market_exposure,
                                global_limits=global_limits,
                                trader_limits=effective_risk_limits,
                                global_daily_realized_pnl_usd=adjusted_global_daily_pnl,
                                trader_daily_realized_pnl_usd=adjusted_trader_daily_pnl,
                                global_unrealized_pnl_usd=global_unrealized_pnl,
                                trader_unrealized_pnl_usd=trader_unrealized_pnl,
                                trader_consecutive_losses=trader_loss_streak,
                                cycle_orders_placed=orders_written,
                                cooldown_active=cooldown_active,
                                mode=run_mode,
                            )
                            return risk_result, dict(risk_snapshot_base)

                        risk_evaluator = _evaluate_runtime_risk

                    gate_result = apply_platform_decision_gates(
                        decision_obj=decision_obj,
                        runtime_signal=runtime_signal,
                        strategy=strategy,
                        checks_payload=checks_payload,
                        trading_window_ok=trading_window_ok,
                        trading_window_config=metadata.get("trading_window_utc"),
                        global_limits=global_limits,
                        effective_risk_limits=effective_risk_limits,
                        allow_averaging=allow_averaging,
                        open_market_ids=open_market_ids,
                        risk_evaluator=risk_evaluator,
                        invoke_hooks=True,
                    )
                    final_decision = gate_result["final_decision"]
                    final_reason = gate_result["final_reason"]
                    score = gate_result["score"]
                    size_usd = gate_result["size_usd"]
                    checks_payload = gate_result["checks_payload"]
                    risk_snapshot = gate_result["risk_snapshot"]

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
                            "strategy_decision": {
                                "decision": gate_result["strategy_decision"],
                                "reason": gate_result["strategy_reason"],
                            },
                            "platform_gates": gate_result["platform_gates"],
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
                                "global_daily_realized_pnl_usd": risk_runtime_payload["global_daily_realized_pnl_usd"],
                                "trader_daily_realized_pnl_usd": risk_runtime_payload["trader_daily_realized_pnl_usd"],
                                "intra_cycle_committed_usd": risk_runtime_payload["intra_cycle_committed_usd"],
                                "trader_consecutive_losses": risk_runtime_payload["trader_consecutive_losses"],
                                "cooldown_seconds": risk_runtime_payload["cooldown_seconds"],
                                "cooldown_active": risk_runtime_payload["cooldown_active"],
                                "cooldown_remaining_seconds": risk_runtime_payload["cooldown_remaining_seconds"],
                                "trader_open_positions": risk_runtime_payload["trader_open_positions"],
                                "trading_window_ok": risk_runtime_payload["trading_window_ok"],
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
                        submit_result = await submit_order(
                            session_engine=session_engine,
                            trader_id=trader_id,
                            signal=runtime_signal,
                            decision_id=decision_row.id,
                            strategy_key=resolved_strategy_key,
                            strategy_params=strategy_params,
                            risk_limits=effective_risk_limits,
                            mode=str(control.get("mode", "paper")),
                            size_usd=size_usd,
                            reason=final_reason,
                        )
                        if isinstance(submit_result, tuple):
                            normalized_order_status = str(submit_result[0] or "").strip().lower()
                            order_status = normalized_order_status
                            if normalized_order_status in {
                                "executed",
                                "completed",
                                "working",
                                "hedging",
                                "partial",
                                "open",
                            }:
                                orders_written += 1
                        else:
                            normalized_order_status = str(submit_result.status or "").strip().lower()
                            order_status = normalized_order_status
                            orders_written += int(submit_result.orders_written or 0)

                        if normalized_order_status in {"completed", "working", "hedging", "partial"}:
                            open_positions = await get_open_position_count_for_trader(
                                session,
                                trader_id,
                                mode=run_mode,
                            )
                            opened_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
                            if opened_market_id:
                                open_market_ids.add(opened_market_id)
                            intra_cycle_committed_usd += size_usd

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
                    await _commit_with_retry(session)
                    cursor_created_at = signal.created_at
                    cursor_signal_id = signal_id
                    processed_signals += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await session.rollback()
                    logger.exception(
                        "Trader %s failed to process signal %s",
                        trader_id,
                        signal_id,
                    )
                    error_type = exc.__class__.__name__
                    error_message = str(exc or "").strip() or error_type
                    strategy_for_error = resolved_strategy_key or strategy_key or "unknown_strategy"
                    failure_reason = f"Signal processing failed ({error_type})"
                    decision_row = await create_trader_decision(
                        session,
                        trader_id=trader_id,
                        signal=signal,
                        strategy_key=strategy_for_error,
                        decision="failed",
                        reason=failure_reason,
                        score=0.0,
                        checks_summary={"count": 0},
                        risk_snapshot={},
                        payload={
                            "source_key": signal_source,
                            "source_config": source_config or {},
                            "error_type": error_type,
                            "error_message": error_message,
                        },
                        commit=False,
                    )
                    decisions_written += 1
                    await set_trade_signal_status(
                        session,
                        signal_id=signal_id,
                        status="failed",
                        commit=False,
                    )
                    await record_signal_consumption(
                        session,
                        trader_id=trader_id,
                        signal_id=signal_id,
                        decision_id=decision_row.id,
                        outcome="failed",
                        reason=failure_reason,
                        payload={
                            "error_type": error_type,
                            "error_message": error_message,
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
                    await create_trader_event(
                        session,
                        trader_id=trader_id,
                        event_type="decision_error",
                        severity="warn",
                        source=signal_source,
                        message="Signal processing failed; advanced cursor to avoid hard loop.",
                        payload={
                            "signal_id": signal_id,
                            "decision_id": decision_row.id,
                            "error_type": error_type,
                            "error_message": error_message,
                        },
                        commit=False,
                    )
                    await _commit_with_retry(session)
                    cursor_created_at = signal.created_at
                    cursor_signal_id = signal_id
                    processed_signals += 1

        if process_signals:
            row = await session.get(Trader, trader_id)
            if row is not None:
                should_persist_heartbeat = (
                    processed_signals > 0
                    or decisions_written > 0
                    or orders_written > 0
                    or row.requested_run_at is not None
                )
                if not should_persist_heartbeat:
                    return decisions_written, orders_written
                now = utcnow()
                row.last_run_at = now
                row.requested_run_at = None
                row.updated_at = now
                await _commit_with_retry(
                    session,
                    retry_attempts=2,
                    busy_timeout_ms=250,
                    base_delay_seconds=0.05,
                    max_delay_seconds=0.1,
                )

    return decisions_written, orders_written


async def _reconcile_orphan_open_orders(session: Any) -> dict[str, int]:
    """Close/cancel open orders for trader_ids that no longer exist."""
    rows = (
        await session.execute(
            select(
                TraderOrder.trader_id,
                func.lower(func.coalesce(TraderOrder.mode, "")).label("mode_key"),
                func.count(TraderOrder.id).label("count"),
            )
            .select_from(TraderOrder)
            .outerjoin(Trader, Trader.id == TraderOrder.trader_id)
            .where(Trader.id.is_(None))
            .where(func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(_PAPER_ACTIVE_ORDER_STATUSES)))
            .group_by(TraderOrder.trader_id, func.lower(func.coalesce(TraderOrder.mode, "")))
            .limit(_ORPHAN_CLEANUP_MAX_TRADERS_PER_CYCLE)
        )
    ).all()

    if not rows:
        return {
            "traders_seen": 0,
            "rows_seen": 0,
            "paper_closed": 0,
            "non_paper_cancelled": 0,
        }

    per_trader_mode: dict[tuple[str, str], int] = {}
    for row in rows:
        trader_id = str(row.trader_id or "").strip()
        if not trader_id:
            continue
        mode_key = str(row.mode_key or "").strip().lower()
        if mode_key == "":
            mode_key = "other"
        per_trader_mode[(trader_id, mode_key)] = int(row.count or 0)

    paper_closed = 0
    non_paper_cancelled = 0
    for trader_id, mode_key in per_trader_mode:
        if mode_key == "paper":
            result = await reconcile_paper_positions(
                session,
                trader_id=trader_id,
                trader_params={},
                dry_run=False,
                force_mark_to_market=False,
                reason="orphan_trader_lifecycle",
            )
            paper_closed += int(result.get("closed", 0))
            await sync_trader_position_inventory(session, trader_id=trader_id, mode="paper")
            continue

        cleanup = await cleanup_trader_open_orders(
            session,
            trader_id=trader_id,
            scope="all",
            dry_run=False,
            target_status="cancelled",
            reason="orphan_trader_cleanup",
        )
        non_paper_cancelled += int(cleanup.get("updated", 0))

    await create_trader_event(
        session,
        trader_id=None,
        event_type="orphan_orders_reconciled",
        severity="warn",
        source="worker",
        message=(
            f"Reconciled orphan trader open orders "
            f"(paper_closed={paper_closed}, non_paper_cancelled={non_paper_cancelled})"
        ),
        payload={
            "traders_seen": len({tid for tid, _ in per_trader_mode.keys()}),
            "rows_seen": len(per_trader_mode),
            "paper_closed": paper_closed,
            "non_paper_cancelled": non_paper_cancelled,
        },
        commit=True,
    )

    return {
        "traders_seen": len({tid for tid, _ in per_trader_mode.keys()}),
        "rows_seen": len(per_trader_mode),
        "paper_closed": paper_closed,
        "non_paper_cancelled": non_paper_cancelled,
    }


async def _run_trader_once_with_timeout(
    trader: dict[str, Any],
    control: dict[str, Any],
    *,
    process_signals: bool,
    timeout_seconds: float,
) -> tuple[int, int]:
    timeout = max(1.0, float(timeout_seconds))
    trader_id = str(trader.get("id") or "")
    try:
        return await asyncio.wait_for(
            _run_trader_once(
                trader,
                control,
                process_signals=process_signals,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Trader cycle timed out for trader=%s process_signals=%s timeout=%.1fs",
            trader_id,
            process_signals,
            timeout,
        )
        if trader_id:
            try:
                async with AsyncSessionLocal() as session:
                    await _apply_orchestrator_sqlite_busy_timeout(session)
                    await create_trader_event(
                        session,
                        trader_id=trader_id,
                        event_type="cycle_timeout",
                        severity="warn",
                        source="worker",
                        message=f"Trader cycle timed out after {timeout:.1f}s",
                        payload={
                            "process_signals": bool(process_signals),
                            "timeout_seconds": timeout,
                        },
                        commit=True,
                    )
            except Exception as exc:
                logger.warning("Failed to persist trader timeout event: %s", exc)
        return 0, 0
    except OperationalError as exc:
        if not _is_sqlite_lock_error(exc):
            raise
        logger.warning(
            "Trader cycle skipped due to SQLite lock for trader=%s process_signals=%s",
            trader_id,
            process_signals,
        )
        return 0, 0


async def run_worker_loop() -> None:
    logger.info("Starting trader orchestrator worker loop")

    try:
        async with AsyncSessionLocal() as session:
            await _apply_orchestrator_sqlite_busy_timeout(session)
            await ensure_all_strategies_seeded(session)
            await refresh_strategy_runtime_if_needed(session, source_keys=None, force=True)
    except Exception as exc:
        logger.warning("Orchestrator strategy startup sync failed: %s", exc)

    try:
        while True:
            cycle_interval = ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS
            sleep_seconds = 2
            skip_cycle = False
            force_cycle = False
            mode = "paper"
            control: dict[str, Any] = {}
            traders: list[dict[str, Any]] = []
            try:
                if not await _ensure_orchestrator_cycle_lock_owner():
                    logger.debug(
                        "Orchestrator cycle lock held by another worker instance; waiting to retry",
                    )
                    await asyncio.sleep(2)
                    continue

                async with AsyncSessionLocal() as session:
                    await _apply_orchestrator_sqlite_busy_timeout(session)
                    try:
                        await expire_stale_signals(session)
                    except OperationalError as exc:
                        if _is_sqlite_lock_error(exc):
                            if hasattr(session, "rollback"):
                                await session.rollback()
                            logger.warning("Skipped stale-signal expiry due to SQLite lock")
                        else:
                            raise
                    except Exception as exc:
                        if hasattr(session, "rollback"):
                            await session.rollback()
                        logger.warning("Failed stale-signal expiry pass: %s", exc)
                    try:
                        await refresh_strategy_runtime_if_needed(session, source_keys=None)
                    except Exception as exc:
                        logger.warning("Failed strategy runtime refresh pass: %s", exc)

                    try:
                        orphan_cleanup = await _reconcile_orphan_open_orders(session)
                        if orphan_cleanup.get("rows_seen", 0) > 0:
                            logger.warning(
                                "Reconciled orphan trader orders",
                                extra={"orphan_cleanup": orphan_cleanup},
                            )
                    except Exception as exc:
                        logger.warning("Failed orphan-order reconciliation pass: %s", exc)

                    control = await read_orchestrator_control(session)
                    cycle_interval = max(
                        1,
                        int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
                    )
                    sleep_seconds = cycle_interval
                    default_trader_cycle_timeout = float(max(8, cycle_interval * 2))
                    trader_cycle_timeout_seconds = float(
                        max(
                            3.0,
                            min(
                                120.0,
                                safe_float(
                                    (control.get("settings") or {}).get("trader_cycle_timeout_seconds"),
                                    default_trader_cycle_timeout,
                                )
                                or default_trader_cycle_timeout,
                            ),
                        )
                    )
                    force_cycle = _parse_iso(control.get("requested_run_at")) is not None

                    if not control.get("is_enabled") or control.get("is_paused"):
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            running=False,
                            enabled=bool(control.get("is_enabled", False)),
                            current_activity="Paused",
                            interval_seconds=cycle_interval,
                            stats=await compute_orchestrator_metrics(session),
                        )
                        skip_cycle = True

                    # Kill switch is the highest-priority gate after enabled/paused.
                    # Short-circuit before ANY mode checks, trader iteration, or
                    # signal processing so no work is wasted while the switch is on.
                    if not skip_cycle and bool(control.get("kill_switch")):
                        logger.info("Kill switch active — skipping entire orchestrator cycle")
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            running=True,
                            enabled=True,
                            current_activity="Kill switch active — all signal processing blocked",
                            interval_seconds=cycle_interval,
                            stats=await compute_orchestrator_metrics(session),
                        )
                        skip_cycle = True

                    if not skip_cycle:
                        mode = str(control.get("mode") or "paper").strip().lower()
                        control_settings = control.get("settings") or {}

                        if mode == "paper":
                            paper_account_id = str(control_settings.get("paper_account_id") or "").strip()
                            if not paper_account_id:
                                await _write_orchestrator_snapshot_best_effort(
                                    session,
                                    running=False,
                                    enabled=True,
                                    current_activity="Blocked: select a sandbox account for paper mode",
                                    interval_seconds=cycle_interval,
                                    last_error=None,
                                    stats=await compute_orchestrator_metrics(session),
                                )
                                skip_cycle = True
                            else:
                                paper_account = await session.get(SimulationAccount, paper_account_id)
                                if paper_account is None:
                                    await _write_orchestrator_snapshot_best_effort(
                                        session,
                                        running=False,
                                        enabled=True,
                                        current_activity="Blocked: selected sandbox account no longer exists",
                                        interval_seconds=cycle_interval,
                                        last_error=None,
                                        stats=await compute_orchestrator_metrics(session),
                                    )
                                    skip_cycle = True

                        if not skip_cycle and mode == "live":
                            app_settings = await session.get(AppSettings, "default")
                            trading_enabled = bool(settings.TRADING_ENABLED) and bool(
                                app_settings.trading_enabled if app_settings is not None else False
                            )
                            if not trading_enabled:
                                await _write_orchestrator_snapshot_best_effort(
                                    session,
                                    running=False,
                                    enabled=True,
                                    current_activity="Blocked: live trading disabled in config/settings",
                                    interval_seconds=cycle_interval,
                                    last_error=None,
                                    stats=await compute_orchestrator_metrics(session),
                                )
                                skip_cycle = True
                            elif not _is_live_credentials_configured(app_settings):
                                await _write_orchestrator_snapshot_best_effort(
                                    session,
                                    running=False,
                                    enabled=True,
                                    current_activity="Blocked: live credentials missing",
                                    interval_seconds=cycle_interval,
                                    last_error=None,
                                    stats=await compute_orchestrator_metrics(session),
                                )
                                skip_cycle = True

                        if not skip_cycle:
                            traders = await list_traders(session)

                if skip_cycle:
                    await asyncio.sleep(sleep_seconds)
                    continue

                total_decisions = 0
                total_orders = 0
                now = datetime.now(timezone.utc)
                for trader in traders:
                    if not trader.get("is_enabled", True):
                        continue

                    is_paused = bool(trader.get("is_paused", False))
                    due = force_cycle or _is_due(trader, now)

                    if is_paused:
                        if mode in ("paper", "live"):
                            await _run_trader_once_with_timeout(
                                trader,
                                control,
                                process_signals=False,
                                timeout_seconds=trader_cycle_timeout_seconds,
                            )
                        continue

                    if not due:
                        if mode in ("paper", "live"):
                            await _run_trader_once_with_timeout(
                                trader,
                                control,
                                process_signals=False,
                                timeout_seconds=trader_cycle_timeout_seconds,
                            )
                        continue

                    decisions, orders = await _run_trader_once_with_timeout(
                        trader,
                        control,
                        process_signals=True,
                        timeout_seconds=trader_cycle_timeout_seconds,
                    )
                    total_decisions += decisions
                    total_orders += orders

                async with AsyncSessionLocal() as session:
                    await _apply_orchestrator_sqlite_busy_timeout(session)
                    if force_cycle:
                        await update_orchestrator_control(session, requested_run_at=None)
                    metrics = await compute_orchestrator_metrics(session)
                    metrics["decisions_last_cycle"] = total_decisions
                    metrics["orders_last_cycle"] = total_orders
                    await _write_orchestrator_snapshot_best_effort(
                        session,
                        running=True,
                        enabled=True,
                        current_activity=f"Cycle decisions={total_decisions} orders={total_orders}",
                        interval_seconds=cycle_interval,
                        last_run_at=utcnow(),
                        stats=metrics,
                    )

                await asyncio.sleep(sleep_seconds)
            except Exception as exc:
                logger.exception("Trader orchestrator worker cycle failed: %s", exc)
                try:
                    async with AsyncSessionLocal() as session:
                        await _apply_orchestrator_sqlite_busy_timeout(session)
                        control = await read_orchestrator_control(session)
                        await _write_orchestrator_snapshot_best_effort(
                            session,
                            running=False,
                            enabled=bool(control.get("is_enabled", False)),
                            current_activity="Worker error",
                            interval_seconds=max(
                                1, int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS)
                            ),
                            last_error=str(exc),
                            stats=await compute_orchestrator_metrics(session),
                        )
                except Exception as snapshot_exc:
                    logger.warning(
                        "Failed to persist orchestrator worker error state",
                        error=str(snapshot_exc),
                    )
                await asyncio.sleep(2)
    finally:
        await _release_orchestrator_cycle_lock_owner()


async def start_loop() -> None:
    """Run the trader orchestrator worker loop (called from API process lifespan)."""
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


async def main() -> None:
    await init_database()
    try:
        await run_worker_loop()
    except asyncio.CancelledError:
        logger.info("Trader orchestrator worker shutting down")


if __name__ == "__main__":
    asyncio.run(main())
