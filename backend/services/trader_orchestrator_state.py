"""DB-backed state for trader orchestrator runtime and APIs."""

from __future__ import annotations

import uuid
import asyncio
import time
import copy
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from config import settings
from services.execution_latency_metrics import execution_latency_metrics, snapshot_from_events as _latency_snapshot_from_events
from sqlalchemy import and_, case, desc, func, or_, select, text as sa_text, update as sa_update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from models.database import (
    AppSettings,
    AsyncSessionLocal,
    ExecutionSession,
    ExecutionSessionEvent,
    ExecutionSessionLeg,
    ExecutionSessionOrder,
    LiveTradingOrder,
    LiveTradingPosition,
    TradeSignal,
    Strategy,
    Trader,
    TraderConfigRevision,
    TraderDecision,
    TraderDecisionCheck,
    TraderEvent,
    TraderOrder,
    TraderPosition,
    TraderOrchestratorControl,
    TraderOrchestratorSnapshot,
    TraderSignalCursor,
    TraderSignalConsumption,
    StrategyExperimentAssignment,
    release_conn,
)
from utils.logger import get_logger
from services.event_bus import event_bus
from services.market_roster import ensure_market_roster_payload
from services.market_cache import CachedMarket
from services.runtime_status import runtime_status
from services.worker_state import (
    DB_RETRY_ATTEMPTS,
    _commit_with_retry,
    _is_retryable_db_error,
    _db_retry_delay,
    read_worker_snapshot,
    summarize_worker_stats,
)
from services.live_execution_adapter import execute_live_order
from services.live_execution_service import live_execution_service
from services.live_pressure import maybe_mark_db_pressure
from services.trader_orchestrator.sources.registry import normalize_source_key
from services.opportunity_strategy_catalog import (
    build_system_opportunity_strategy_rows,
    list_system_strategy_keys,
)
from services.strategy_versioning import normalize_strategy_version, resolve_strategy_version
from services.strategy_sdk import StrategySDK
from services.strategies.news_edge import validate_news_edge_config
from services.strategies.traders_copy_trade import validate_traders_copy_trade_config
from services.trader_orchestrator.templates import (
    DEFAULT_GLOBAL_RISK,
    TRADER_TEMPLATES,
    get_template,
)
from services.trader_order_verification import (
    TRADER_ORDER_HIDDEN_VERIFICATION_STATUSES,
    TRADER_ORDER_VERIFICATION_LOCAL,
    TRADER_ORDER_VERIFICATION_VENUE_FILL,
    TRADER_ORDER_VERIFICATION_VENUE_ORDER,
    TRADER_ORDER_VERIFICATION_WALLET_POSITION,
    append_trader_order_verification_event,
    apply_trader_order_verification,
    derive_trader_order_verification,
    extract_trader_order_execution_wallet_address,
    extract_trader_order_provider_clob_order_id,
    extract_trader_order_provider_order_id,
    extract_trader_order_verification_tx_hash,
    trader_order_verification_hidden,
    trader_order_verification_rank,
)
from utils.utcnow import utcnow
from utils.secrets import decrypt_secret
from utils.converters import coerce_bool, safe_float, safe_int, to_iso
from utils.market_urls import build_polymarket_market_url

logger = get_logger(__name__)

ORCHESTRATOR_CONTROL_ID = "default"
ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS = 30
_UNSET = object()  # Sentinel: distinguish "not provided" from explicit None
ORCHESTRATOR_SNAPSHOT_ID = "latest"
OPEN_ORDER_STATUSES = {"submitted", "executed", "completed", "open"}
RESOLVED_ORDER_STATUSES = {"resolved", "resolved_win", "resolved_loss"}
UNFILLED_ORDER_STATUSES = {"submitted", "open"}
SHADOW_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "completed", "open"}
LIVE_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "completed", "open"}
CLEANUP_ELIGIBLE_ORDER_STATUSES = {"submitted", "executed", "completed", "open"}

# Cap on per-call cleanup work.  cleanup_trader_open_orders sequentially
# fires one (occasionally two) Polymarket cancel HTTP calls per live
# candidate, each with an 8s timeout (LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS).
# A trader with 50+ stuck unfilled live orders blows the 20s maintenance
# step budget every cycle.  Cap candidate processing per invocation so a
# burst of stale orders drains over multiple cycles instead of starving
# the cycle.  Oldest-first ordering means the most-overdue cancellations
# happen first; the rest land on the next maintenance pass.
_OPEN_ORDER_CLEANUP_MAX_CANDIDATES_PER_CALL = 10
_LIVE_ORDER_AUTHORITY_RECOVERY_WINDOW_HOURS = 24
_LIVE_ORDER_AUTHORITY_RECOVERY_GENERAL_MAX_ROWS = 150
_LIVE_ORDER_AUTHORITY_RECOVERY_CANDIDATE_CACHE_BUCKET_SECONDS = 60
_LIVE_ORDER_AUTHORITY_RECOVERY_RECENT_TERMINAL_LOOKBACK_MINUTES = 30
_LIVE_ORDER_AUTHORITY_RECOVERY_DECISION_LOOKBACK_MINUTES = 30
PENDING_LIVE_EXIT_TERMINAL_STATUSES = {
    "filled",
    "superseded_resolution",
    "superseded_external",
    "cancelled",
}
DEFAULT_PENDING_LIVE_EXIT_GUARD = {
    "max_pending_exits": 0,
    "identity_guard_enabled": True,
    "terminal_statuses": sorted(PENDING_LIVE_EXIT_TERMINAL_STATUSES),
}
DEFAULT_LIVE_MARKET_CONTEXT = {
    "enabled": True,
    "history_window_seconds": 7200,
    "history_fidelity_seconds": 300,
    "max_history_points": 120,
    "timeout_seconds": 4.0,
    "strict_ws_pricing_only": True,
    "max_market_data_age_ms": 10000,
}
DEFAULT_LIVE_PROVIDER_HEALTH = {
    "window_seconds": 180,
    "min_errors": 2,
    "block_seconds": 120,
}
LEGACY_IMPLICIT_LIVE_RISK_CLAMPS = {
    "enforce_allow_averaging_off": True,
    "min_cooldown_seconds": 60,
    "max_consecutive_losses_cap": 3,
    "max_open_orders_cap": 6,
    "max_open_positions_cap": 8,
    "max_trade_notional_usd_cap": 10.0,
    "max_per_market_exposure_usd_cap": 10.0,
    "max_orders_per_cycle_cap": 3,
    "enforce_halt_on_consecutive_losses": True,
}
LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS = 8.0
DEFAULT_TIMEOUT_TAKER_RESCUE_PRICE_BPS = 35.0
DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE = "IOC"
REALIZED_WIN_ORDER_STATUSES = {"resolved_win", "closed_win", "win"}
REALIZED_LOSS_ORDER_STATUSES = {"resolved_loss", "closed_loss", "loss"}
REALIZED_ORDER_STATUSES = REALIZED_WIN_ORDER_STATUSES | REALIZED_LOSS_ORDER_STATUSES
ACTIVE_POSITION_STATUS = "open"
INACTIVE_POSITION_STATUS = "closed"
ACTIVE_EXECUTION_SESSION_STATUSES = {"pending", "placing", "working", "partial", "hedging", "paused"}
TERMINAL_EXECUTION_SESSION_STATUSES = {"completed", "failed", "cancelled", "expired", "skipped"}
_TRADER_SCOPE_MODES = set(StrategySDK.TRADER_SCOPE_MODE_CANONICAL)
_ORCHESTRATOR_SNAPSHOT_STALE_MULTIPLIER = 30.0
_ORCHESTRATOR_SNAPSHOT_STALE_MIN_SECONDS = 120.0
_TRADER_MODES = {"shadow", "live"}
_LATENCY_CLASSES = {"fast", "normal", "slow"}
_MANUAL_LIVE_POSITION_SOURCE = "manual_wallet_position"
_LIVE_ORDER_AUTHORITY_RECOVERY_LOCK_KEY = 674021913
_LEGACY_STRATEGY_ALIASES: dict[str, str] = {
    "news_reaction": "news_edge",
}


def _now() -> datetime:
    return utcnow()


def _new_id() -> str:
    return uuid.uuid4().hex


def _normalize_mode_key(mode: Any) -> str:
    key = str(mode or "").strip().lower()
    if key in _TRADER_MODES:
        return key
    return "other"


def _normalize_status_key(status: Any) -> str:
    return str(status or "").strip().lower()


def _visible_trader_order_query_clause():
    # Phase 3 step 3: tolerate the verification status living EITHER on
    # trader_orders.verification_status (legacy) OR on
    # trader_order_verification.verification_status (new side table).
    # During dual-write the listener keeps them identical; once we drop
    # the old column (step 4), this clause continues to work because
    # ``func.coalesce`` skips the legacy NULL.  We don't add an explicit
    # JOIN here because most callers SELECT TraderOrder rows and bring
    # the side table along via the relationship loader (or accept the
    # fall-through to the legacy column for now).  When step 4 lands,
    # this becomes a JOIN clause; until then the legacy column is the
    # source of truth and the listener guarantees it stays in sync.
    verification_status_key = func.lower(
        func.coalesce(TraderOrder.verification_status, TRADER_ORDER_VERIFICATION_LOCAL)
    )
    return ~verification_status_key.in_(tuple(TRADER_ORDER_HIDDEN_VERIFICATION_STATUSES))


def _expand_trader_order_status_filter(status: Optional[str]) -> tuple[str, ...] | None:
    status_key = str(status or "").strip().lower()
    compound_status_key = status_key.replace("+", "_").replace("-", "_").replace(" ", "_").strip("_")
    if not status_key or status_key == "all":
        return None
    if status_key == "open":
        return tuple(sorted(OPEN_ORDER_STATUSES))
    if status_key == "resolved":
        return tuple(sorted(RESOLVED_ORDER_STATUSES))
    if status_key == "failed":
        return ("cancelled", "error", "failed", "rejected")
    if compound_status_key == "open_resolved":
        return tuple(sorted(OPEN_ORDER_STATUSES | RESOLVED_ORDER_STATUSES))
    return (status_key,)


def _visible_trader_order_row(row: TraderOrder) -> bool:
    return not trader_order_verification_hidden(getattr(row, "verification_status", None))


def _coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out or list(fallback)


def _normalize_pending_live_exit_terminal_statuses(value: Any) -> list[str]:
    if not isinstance(value, list):
        return list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        status = _normalize_status_key(item)
        if not status or status in seen:
            continue
        seen.add(status)
        out.append(status)
    return out or list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])


def _normalize_pending_live_exit_guard(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "max_pending_exits": max(0, min(1000, safe_int(source.get("max_pending_exits"), 0))),
        "identity_guard_enabled": bool(
            source.get("identity_guard_enabled", DEFAULT_PENDING_LIVE_EXIT_GUARD["identity_guard_enabled"])
        ),
        "terminal_statuses": _normalize_pending_live_exit_terminal_statuses(source.get("terminal_statuses")),
    }


def _live_risk_clamp_was_legacy_implicit(key: str, value: Any) -> bool:
    if key not in LEGACY_IMPLICIT_LIVE_RISK_CLAMPS:
        return False
    legacy_value = LEGACY_IMPLICIT_LIVE_RISK_CLAMPS[key]
    if isinstance(legacy_value, bool):
        return bool(value) is legacy_value
    if isinstance(legacy_value, int):
        return safe_int(value, -1) == legacy_value
    return abs(safe_float(value, -1.0) - float(legacy_value)) < 0.000001


def _normalize_live_risk_clamps(value: Any, *, explicit: bool = False) -> dict[str, Any]:
    """Normalize explicitly configured live risk clamps."""
    source = value if isinstance(value, dict) else {}
    out: dict[str, Any] = {}
    source_is_explicit = bool(explicit or source.get("_explicit"))

    def should_apply(key: str) -> bool:
        return key in source and (source_is_explicit or not _live_risk_clamp_was_legacy_implicit(key, source[key]))

    if should_apply("enforce_allow_averaging_off"):
        out["enforce_allow_averaging_off"] = bool(source["enforce_allow_averaging_off"])
    if should_apply("min_cooldown_seconds"):
        out["min_cooldown_seconds"] = max(0, min(86400, safe_int(source["min_cooldown_seconds"], 0)))
    if should_apply("max_consecutive_losses_cap"):
        out["max_consecutive_losses_cap"] = max(1, min(1000, safe_int(source["max_consecutive_losses_cap"], 1000)))
    if should_apply("max_open_orders_cap"):
        out["max_open_orders_cap"] = max(1, min(1000, safe_int(source["max_open_orders_cap"], 1000)))
    if should_apply("max_open_positions_cap"):
        out["max_open_positions_cap"] = max(1, min(1000, safe_int(source["max_open_positions_cap"], 1000)))
    if should_apply("max_trade_notional_usd_cap"):
        out["max_trade_notional_usd_cap"] = max(1.0, min(1_000_000.0, safe_float(source["max_trade_notional_usd_cap"], 1_000_000.0)))
    if should_apply("max_per_market_exposure_usd_cap"):
        out["max_per_market_exposure_usd_cap"] = max(1.0, min(1_000_000.0, safe_float(source["max_per_market_exposure_usd_cap"], 1_000_000.0)))
    if should_apply("max_orders_per_cycle_cap"):
        out["max_orders_per_cycle_cap"] = max(1, min(1000, safe_int(source["max_orders_per_cycle_cap"], 1000)))
    if should_apply("enforce_halt_on_consecutive_losses"):
        out["enforce_halt_on_consecutive_losses"] = bool(source["enforce_halt_on_consecutive_losses"])
    return out


def _normalize_live_market_context(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "enabled": bool(source.get("enabled", DEFAULT_LIVE_MARKET_CONTEXT["enabled"])),
        "history_window_seconds": max(
            300,
            min(
                21600,
                safe_int(source.get("history_window_seconds"), DEFAULT_LIVE_MARKET_CONTEXT["history_window_seconds"]),
            ),
        ),
        "history_fidelity_seconds": max(
            30,
            min(
                1800,
                safe_int(
                    source.get("history_fidelity_seconds"),
                    DEFAULT_LIVE_MARKET_CONTEXT["history_fidelity_seconds"],
                ),
            ),
        ),
        "max_history_points": max(
            20,
            min(240, safe_int(source.get("max_history_points"), DEFAULT_LIVE_MARKET_CONTEXT["max_history_points"])),
        ),
        "timeout_seconds": max(
            1.0,
            min(12.0, safe_float(source.get("timeout_seconds"), DEFAULT_LIVE_MARKET_CONTEXT["timeout_seconds"]) or 4.0),
        ),
        "strict_ws_pricing_only": bool(
            source.get("strict_ws_pricing_only", DEFAULT_LIVE_MARKET_CONTEXT["strict_ws_pricing_only"])
        ),
        "max_market_data_age_ms": max(
            25,
            min(
                30_000,
                safe_int(source.get("max_market_data_age_ms"), DEFAULT_LIVE_MARKET_CONTEXT["max_market_data_age_ms"]),
            ),
        ),
    }


def _normalize_live_provider_health(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "window_seconds": max(
            30,
            min(
                900,
                safe_int(source.get("window_seconds"), DEFAULT_LIVE_PROVIDER_HEALTH["window_seconds"]),
            ),
        ),
        "min_errors": max(1, min(20, safe_int(source.get("min_errors"), DEFAULT_LIVE_PROVIDER_HEALTH["min_errors"]))),
        "block_seconds": max(
            15,
            min(3600, safe_int(source.get("block_seconds"), DEFAULT_LIVE_PROVIDER_HEALTH["block_seconds"])),
        ),
    }


def _normalize_global_runtime_settings(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    timeout_override = source.get("trader_cycle_timeout_seconds")
    timeout_value = safe_float(timeout_override, 0.0)
    trader_cycle_timeout_seconds = None if timeout_value <= 0 else max(3.0, min(120.0, float(timeout_value)))
    return {
        "pending_live_exit_guard": _normalize_pending_live_exit_guard(source.get("pending_live_exit_guard")),
        "live_risk_clamps": _normalize_live_risk_clamps(
            source.get("live_risk_clamps"),
            explicit=bool(source.get("live_risk_clamps_explicit")),
        ),
        "live_risk_clamps_explicit": bool(source.get("live_risk_clamps_explicit", False)),
        "live_market_context": _normalize_live_market_context(source.get("live_market_context")),
        "live_provider_health": _normalize_live_provider_health(source.get("live_provider_health")),
        "trader_cycle_timeout_seconds": trader_cycle_timeout_seconds,
    }


def _normalize_global_risk(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "max_gross_exposure_usd": max(
            1.0,
            min(
                1_000_000.0,
                safe_float(source.get("max_gross_exposure_usd"), DEFAULT_GLOBAL_RISK["max_gross_exposure_usd"]),
            ),
        ),
        "max_daily_loss_usd": max(
            0.0,
            min(1_000_000.0, safe_float(source.get("max_daily_loss_usd"), DEFAULT_GLOBAL_RISK["max_daily_loss_usd"])),
        ),
        "max_orders_per_cycle": max(
            1,
            min(1000, safe_int(source.get("max_orders_per_cycle"), DEFAULT_GLOBAL_RISK["max_orders_per_cycle"])),
        ),
    }


def _normalize_control_settings(value: Any) -> dict[str, Any]:
    settings = dict(value) if isinstance(value, dict) else {}
    removed_keys = {
        "paper_account_id",
        "enable_live_market_context",
        "live_market_history_window_seconds",
        "live_market_history_fidelity_seconds",
        "live_market_history_max_points",
        "live_market_context_timeout_seconds",
        "trader_cycle_timeout_seconds",
    }
    normalized_keys = {
        "global_risk",
        "global_runtime",
        "trading_domains",
        "enabled_strategies",
        "llm_verify_trades",
    }

    normalized = {
        "global_risk": _normalize_global_risk(settings.get("global_risk")),
        "global_runtime": _normalize_global_runtime_settings(settings.get("global_runtime")),
        "trading_domains": _coerce_string_list(settings.get("trading_domains"), ["event_markets", "crypto"]),
        "enabled_strategies": _coerce_string_list(settings.get("enabled_strategies"), list_system_strategy_keys()),
        "llm_verify_trades": bool(settings.get("llm_verify_trades", False)),
    }
    for key, raw_value in settings.items():
        if key in normalized_keys or key in removed_keys:
            continue
        normalized[key] = raw_value
    return normalized


async def _list_enabled_strategy_keys(session: AsyncSession) -> list[str]:
    rows = (
        (
            await session.execute(
                select(Strategy.slug)
                .where(Strategy.enabled == True)  # noqa: E712
                .order_by(Strategy.slug.asc())
            )
        )
        .scalars()
        .all()
    )
    enabled: list[str] = []
    seen: set[str] = set()
    for raw_slug in rows:
        slug = str(raw_slug or "").strip().lower()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        enabled.append(slug)
    if enabled:
        return enabled
    return list_system_strategy_keys()


async def _normalize_control_settings_for_session(
    session: AsyncSession,
    value: Any,
) -> dict[str, Any]:
    normalized = _normalize_control_settings(value)
    normalized["enabled_strategies"] = await _list_enabled_strategy_keys(session)
    return normalized


def _normalize_condition_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 66 or not text.startswith("0x"):
        return ""
    body = text[2:]
    if not body:
        return ""
    for char in body:
        if char not in "0123456789abcdef":
            return ""
    return text


def _extract_order_condition_id(row: TraderOrder) -> str:
    payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    return (
        _normalize_condition_id(payload.get("condition_id"))
        or _normalize_condition_id(payload.get("conditionId"))
        or _normalize_condition_id(payload.get("market_id"))
        or _normalize_condition_id(payload.get("marketId"))
        or _normalize_condition_id(getattr(row, "market_id", None))
    )


def _extract_order_token_id(row: TraderOrder) -> str:
    payload = dict(row.payload_json or {})
    token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip()
    if token_id:
        return token_id
    provider_reconciliation = payload.get("provider_reconciliation")
    if isinstance(provider_reconciliation, dict):
        snapshot = provider_reconciliation.get("snapshot")
        if isinstance(snapshot, dict):
            token_id = str(snapshot.get("asset_id") or snapshot.get("asset") or snapshot.get("token_id") or "").strip()
            if token_id:
                return token_id
    return ""


def _normalize_bundle_side(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"buy", "sell"}:
        return text
    return None


def _normalize_bundle_outcome(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"yes", "no"}:
        return text
    return None


def _trade_bundle_payload(signal: TradeSignal | None, row: TraderOrder) -> dict[str, Any]:
    signal_payload = signal.payload_json if signal is not None and isinstance(signal.payload_json, dict) else {}
    row_payload = row.payload_json if isinstance(row.payload_json, dict) else {}
    row_strategy_context = row_payload.get("strategy_context")
    row_strategy_context = row_strategy_context if isinstance(row_strategy_context, dict) else {}

    row_execution_plan = row_payload.get("execution_plan")
    if not isinstance(row_execution_plan, dict):
        row_execution_plan = row_strategy_context.get("execution_plan")
    row_execution_plan = row_execution_plan if isinstance(row_execution_plan, dict) else {}

    row_execution_session = row_payload.get("execution_session")
    row_execution_session = row_execution_session if isinstance(row_execution_session, dict) else {}
    row_positions = row_payload.get("positions_to_take")
    if not isinstance(row_positions, list):
        row_positions = row_strategy_context.get("positions_to_take")
    row_positions = row_positions if isinstance(row_positions, list) else []

    row_leg = row_payload.get("leg")
    row_leg = row_leg if isinstance(row_leg, dict) else {}
    row_market_roster = row_payload.get("market_roster")
    if not isinstance(row_market_roster, dict):
        row_market_roster = row_strategy_context.get("market_roster")
    row_market_roster = row_market_roster if isinstance(row_market_roster, dict) else {}
    row_markets = row_payload.get("markets")
    if not isinstance(row_markets, list):
        row_markets = row_strategy_context.get("markets")
    row_markets = row_markets if isinstance(row_markets, list) else []

    row_policy = str(row_execution_plan.get("policy") or row_execution_session.get("policy") or "").strip().upper()
    row_has_execution_shape = bool(row_execution_plan or row_positions or row_leg or row_execution_session)
    if not row_has_execution_shape:
        return signal_payload if signal_payload else row_payload

    payload = dict(signal_payload) if signal_payload else {}
    if row_execution_plan:
        payload["execution_plan"] = row_execution_plan
    elif row_policy == "SINGLE_LEG":
        payload["execution_plan"] = {
            "policy": row_policy,
            "legs": [row_leg] if row_leg else [],
        }
    else:
        payload["execution_plan"] = {}
    payload["positions_to_take"] = row_positions
    if row_market_roster:
        payload["market_roster"] = row_market_roster
    if row_markets:
        payload["markets"] = row_markets
    return payload


def _row_counts_as_materialized_bundle_leg(row: TraderOrder) -> bool:
    status_key = _normalize_status_key(getattr(row, "status", None))
    payload = dict(getattr(row, "payload_json", None) or {})
    filled_notional_usd, filled_shares, _ = _extract_live_fill_metrics(payload)
    verification_rank = trader_order_verification_rank(getattr(row, "verification_status", None))

    if status_key in {"failed", "cancelled", "rejected", "error"}:
        return (
            filled_notional_usd > 0.0
            or filled_shares > 0.0
            or verification_rank >= trader_order_verification_rank(TRADER_ORDER_VERIFICATION_VENUE_FILL)
        )

    mode_key = _normalize_mode_key(getattr(row, "mode", None))
    if mode_key != "live":
        return bool(status_key)

    if filled_notional_usd > 0.0 or filled_shares > 0.0:
        return True
    if verification_rank >= trader_order_verification_rank(TRADER_ORDER_VERIFICATION_VENUE_FILL):
        return True
    if verification_rank >= trader_order_verification_rank(TRADER_ORDER_VERIFICATION_WALLET_POSITION):
        return True

    provider_order_id = str(
        getattr(row, "provider_order_id", None)
        or extract_trader_order_provider_order_id(payload)
        or ""
    ).strip()
    provider_clob_order_id = str(
        getattr(row, "provider_clob_order_id", None)
        or extract_trader_order_provider_clob_order_id(payload)
        or ""
    ).strip()
    if status_key in {"resolved", "resolved_win", "resolved_loss", "closed_win", "closed_loss"}:
        return True
    if status_key in LIVE_ACTIVE_ORDER_STATUSES:
        return bool(provider_order_id or provider_clob_order_id) or (
            verification_rank >= trader_order_verification_rank(TRADER_ORDER_VERIFICATION_VENUE_ORDER)
        )
    return False


def _observed_bundle_token_ids(
    row: TraderOrder,
    sibling_rows: list[TraderOrder] | None,
) -> set[str]:
    observed: set[str] = set()
    candidates = sibling_rows if isinstance(sibling_rows, list) and sibling_rows else [row]
    for candidate in candidates:
        if not _visible_trader_order_row(candidate):
            continue
        if not _row_counts_as_materialized_bundle_leg(candidate):
            continue
        token_id = _extract_order_token_id(candidate)
        if token_id:
            observed.add(token_id)
    return observed


def _build_trade_bundle(
    signal: TradeSignal | None,
    row: TraderOrder,
    sibling_rows: list[TraderOrder] | None = None,
) -> dict[str, Any] | None:
    signal_payload = _trade_bundle_payload(signal, row)
    execution_plan = signal_payload.get("execution_plan")
    execution_plan = execution_plan if isinstance(execution_plan, dict) else {}
    raw_plan_legs = execution_plan.get("legs")
    raw_plan_legs = raw_plan_legs if isinstance(raw_plan_legs, list) else []
    raw_positions = signal_payload.get("positions_to_take")
    raw_positions = raw_positions if isinstance(raw_positions, list) else []
    if len(raw_plan_legs) <= 1 and len(raw_positions) <= 1:
        return None

    raw_markets = signal_payload.get("markets")
    raw_markets = raw_markets if isinstance(raw_markets, list) else []
    raw_market_roster = signal_payload.get("market_roster")
    raw_market_roster = raw_market_roster if isinstance(raw_market_roster, dict) else {}
    raw_roster_markets = raw_market_roster.get("markets")
    raw_roster_markets = raw_roster_markets if isinstance(raw_roster_markets, list) else []

    market_lookup_by_token: dict[str, dict[str, Any]] = {}
    for market in raw_markets:
        if not isinstance(market, dict):
            continue
        token_ids = market.get("clob_token_ids")
        if not isinstance(token_ids, list):
            continue
        market_question = str(market.get("question") or market.get("market_question") or "").strip()
        condition_id = str(market.get("condition_id") or market.get("conditionId") or "").strip()
        for token in token_ids:
            token_id = str(token or "").strip()
            if not token_id:
                continue
            market_lookup_by_token[token_id] = {
                "market_question": market_question or None,
                "condition_id": condition_id or None,
            }

    position_lookup_by_token: dict[str, dict[str, Any]] = {}
    for position in raw_positions:
        if not isinstance(position, dict):
            continue
        token_id = str(position.get("token_id") or "").strip()
        if token_id:
            position_lookup_by_token[token_id] = position

    bundle_legs: list[dict[str, Any]] = []
    order_token_id = _extract_order_token_id(row)
    current_leg_id: str | None = None
    current_leg_index: int | None = None

    def _bundle_market_key(value: dict[str, Any]) -> str | None:
        if not isinstance(value, dict):
            return None
        raw_key = str(
            value.get("condition_id")
            or value.get("id")
            or value.get("market_id")
            or value.get("question")
            or value.get("market_question")
            or ""
        ).strip().lower()
        return raw_key or None

    def _append_leg(
        *,
        leg_index: int,
        leg_id: str | None,
        market_id: str | None,
        market_question: str | None,
        token_id: str | None,
        side: str | None,
        outcome: str | None,
        limit_price: float | None,
        notional_weight: float | None,
        condition_id: str | None,
    ) -> None:
        nonlocal current_leg_id, current_leg_index
        normalized_token = str(token_id or "").strip() or None
        normalized_market_id = str(market_id or "").strip() or None
        normalized_question = str(market_question or "").strip() or normalized_market_id
        normalized_leg_id = str(leg_id or f"leg_{leg_index + 1}").strip() or f"leg_{leg_index + 1}"
        bundle_legs.append(
            {
                "leg_index": int(leg_index),
                "leg_id": normalized_leg_id,
                "market_id": normalized_market_id,
                "market_question": normalized_question,
                "token_id": normalized_token,
                "side": side,
                "outcome": outcome,
                "limit_price": float(limit_price) if limit_price is not None and limit_price > 0.0 else None,
                "notional_weight": float(notional_weight) if notional_weight is not None and notional_weight > 0.0 else None,
                "condition_id": str(condition_id or "").strip() or None,
            }
        )
        if not current_leg_id and normalized_token and order_token_id and normalized_token == order_token_id:
            current_leg_id = normalized_leg_id
            current_leg_index = int(leg_index)

    if raw_plan_legs:
        for index, raw_leg in enumerate(raw_plan_legs):
            if not isinstance(raw_leg, dict):
                continue
            token_id = str(raw_leg.get("token_id") or "").strip() or None
            position = position_lookup_by_token.get(token_id or "", {})
            market_meta = market_lookup_by_token.get(token_id or "", {})
            _append_leg(
                leg_index=index,
                leg_id=str(raw_leg.get("leg_id") or "").strip() or None,
                market_id=str(raw_leg.get("market_id") or position.get("market") or "").strip() or None,
                market_question=(
                    str(raw_leg.get("market_question") or position.get("market") or market_meta.get("market_question") or "").strip()
                    or None
                ),
                token_id=token_id,
                side=_normalize_bundle_side(raw_leg.get("side") or position.get("action")),
                outcome=_normalize_bundle_outcome(raw_leg.get("outcome") or position.get("outcome")),
                limit_price=safe_float(raw_leg.get("limit_price"), safe_float(position.get("price"), None)),
                notional_weight=safe_float(raw_leg.get("notional_weight"), None),
                condition_id=market_meta.get("condition_id"),
            )
    else:
        for index, raw_position in enumerate(raw_positions):
            if not isinstance(raw_position, dict):
                continue
            token_id = str(raw_position.get("token_id") or "").strip() or None
            market_meta = market_lookup_by_token.get(token_id or "", {})
            _append_leg(
                leg_index=index,
                leg_id=None,
                market_id=str(raw_position.get("market_id") or raw_position.get("market") or "").strip() or None,
                market_question=(
                    str(raw_position.get("market_question") or raw_position.get("market") or market_meta.get("market_question") or "").strip()
                    or None
                ),
                token_id=token_id,
                side=_normalize_bundle_side(raw_position.get("action")),
                outcome=_normalize_bundle_outcome(raw_position.get("outcome")),
                limit_price=safe_float(raw_position.get("price"), None),
                notional_weight=safe_float(raw_position.get("notional_weight"), None),
                condition_id=market_meta.get("condition_id"),
            )

    if len(bundle_legs) <= 1:
        return None

    mode_key = _normalize_mode_key(getattr(row, "mode", None))
    observed_token_ids = _observed_bundle_token_ids(row, sibling_rows)
    status_key = _normalize_status_key(getattr(row, "status", None))
    if mode_key == "live":
        if sibling_rows is not None and not observed_token_ids:
            return None
        if (
            sibling_rows is not None
            and status_key in {"failed", "cancelled", "rejected", "error"}
            and not _row_counts_as_materialized_bundle_leg(row)
        ):
            return None
        if observed_token_ids:
            filtered_bundle_legs: list[dict[str, Any]] = []
            seen_tokens: set[str] = set()
            for leg in bundle_legs:
                token_id = str(leg.get("token_id") or "").strip()
                if not token_id or token_id not in observed_token_ids or token_id in seen_tokens:
                    continue
                seen_tokens.add(token_id)
                filtered_bundle_legs.append(leg)
            bundle_legs = filtered_bundle_legs

    if len(bundle_legs) <= 1:
        return None

    distinct_market_keys = {
        str(leg.get("condition_id") or leg.get("market_id") or leg.get("market_question") or "").strip().lower()
        for leg in bundle_legs
        if str(leg.get("condition_id") or leg.get("market_id") or leg.get("market_question") or "").strip()
    }
    roster_market_keys = {key for key in (_bundle_market_key(market) for market in raw_roster_markets) if key}
    outcome_set = {str(leg.get("outcome") or "").strip().lower() for leg in bundle_legs if str(leg.get("outcome") or "").strip()}

    if len(bundle_legs) == 2 and len(distinct_market_keys) == 1 and outcome_set == {"yes", "no"}:
        bundle_kind = "paired_binary"
        bundle_label = "Binary paired trade"
    elif len(bundle_legs) >= 2 and outcome_set == {"yes"}:
        bundle_kind = "multi_outcome_yes"
        bundle_label = "Mutually exclusive YES bundle"
    else:
        bundle_kind = "multi_leg"
        bundle_label = "Multi-leg trade"

    signal_is_guaranteed = bool(signal_payload.get("is_guaranteed", False))
    guarantee_proven = False
    guarantee_reason: str | None = None
    if signal_is_guaranteed:
        if bundle_kind == "paired_binary":
            guarantee_proven = len(bundle_legs) == 2 and len(distinct_market_keys) == 1 and outcome_set == {"yes", "no"}
            guarantee_reason = "paired_binary_complete" if guarantee_proven else "paired_binary_incomplete"
        elif roster_market_keys and distinct_market_keys == roster_market_keys:
            guarantee_proven = True
            guarantee_reason = "full_market_roster"
        elif not roster_market_keys:
            guarantee_reason = "missing_market_roster"
        else:
            guarantee_reason = "partial_market_roster"

    return {
        "bundle_id": str(signal.id if signal is not None else row.signal_id or "").strip() or str(row.id),
        "plan_id": str(execution_plan.get("plan_id") or "").strip() or None,
        "kind": bundle_kind,
        "label": bundle_label,
        "leg_count": len(bundle_legs),
        "is_guaranteed": guarantee_proven,
        "signal_is_guaranteed": signal_is_guaranteed,
        "guarantee_proven": guarantee_proven,
        "guarantee_reason": guarantee_reason,
        "roi_type": str(signal_payload.get("roi_type") or "").strip() or None,
        "mispricing_type": str(signal_payload.get("mispricing_type") or "").strip() or None,
        "total_cost": safe_float(signal_payload.get("total_cost"), None),
        "expected_payout": safe_float(signal_payload.get("expected_payout"), None),
        "gross_profit": safe_float(signal_payload.get("gross_profit"), None),
        "net_profit": safe_float(signal_payload.get("net_profit"), safe_float(signal_payload.get("guaranteed_profit"), None)),
        "roi_percent": safe_float(signal_payload.get("roi_percent"), None),
        "planned_market_count": len(distinct_market_keys),
        "market_roster_count": len(roster_market_keys) if roster_market_keys else None,
        "current_leg_id": current_leg_id,
        "current_leg_index": current_leg_index,
        "current_leg_token_id": order_token_id or None,
        "legs": bundle_legs,
    }


def _extract_copy_source_wallet_from_payload(payload: Any) -> str:
    data = payload if isinstance(payload, dict) else {}
    copy_attribution = data.get("copy_attribution")
    copy_attribution = copy_attribution if isinstance(copy_attribution, dict) else {}
    source_trade = data.get("source_trade")
    source_trade = source_trade if isinstance(source_trade, dict) else {}
    strategy_context = data.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    copy_event = strategy_context.get("copy_event")
    copy_event = copy_event if isinstance(copy_event, dict) else {}
    execution_plan = data.get("execution_plan")
    execution_plan = execution_plan if isinstance(execution_plan, dict) else {}
    legs = execution_plan.get("legs")
    legs = legs if isinstance(legs, list) else []

    for raw_wallet in (
        copy_attribution.get("source_wallet"),
        copy_event.get("wallet_address"),
        copy_event.get("source_wallet"),
        source_trade.get("wallet_address"),
        source_trade.get("source_wallet"),
        data.get("wallet_address"),
        data.get("source_wallet"),
    ):
        normalized = StrategySDK.normalize_trader_wallet(raw_wallet)
        if normalized:
            return normalized

    for leg in legs:
        if not isinstance(leg, dict):
            continue
        metadata = leg.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        normalized = StrategySDK.normalize_trader_wallet(metadata.get("source_wallet"))
        if normalized:
            return normalized
    return ""


def _extract_copy_side_from_payload(payload: Any) -> str:
    data = payload if isinstance(payload, dict) else {}
    copy_attribution = data.get("copy_attribution")
    copy_attribution = copy_attribution if isinstance(copy_attribution, dict) else {}
    source_trade = data.get("source_trade")
    source_trade = source_trade if isinstance(source_trade, dict) else {}
    strategy_context = data.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    copy_event = strategy_context.get("copy_event")
    copy_event = copy_event if isinstance(copy_event, dict) else {}
    raw_side = (
        copy_attribution.get("side")
        or copy_event.get("side")
        or source_trade.get("side")
        or data.get("side")
    )
    side = str(raw_side or "").strip().upper()
    if side in {"BUY", "SELL"}:
        return side
    return ""


async def _resolve_execution_wallet_address() -> str:
    wallet = ""
    try:
        wallet = str(live_execution_service.get_execution_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    if wallet:
        return wallet

    try:
        await live_execution_service.ensure_initialized()
    except Exception:
        pass

    try:
        wallet = str(live_execution_service.get_execution_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    if wallet:
        return wallet

    try:
        runtime_signature_type = getattr(live_execution_service, "_balance_signature_type", None)
        signature_type = (
            int(runtime_signature_type)
            if isinstance(runtime_signature_type, int)
            else int(getattr(settings, "POLYMARKET_SIGNATURE_TYPE", 1))
        )
    except Exception:
        signature_type = 1

    try:
        wallet = str(live_execution_service._funder_for_signature_type(signature_type) or "").strip()
    except Exception:
        wallet = ""
    if wallet:
        return wallet

    try:
        wallet = str(live_execution_service._get_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    return wallet


def _wallet_position_token_key(token_id: Any) -> str:
    return str(token_id or "").strip().lower()


def _read_wallet_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _read_wallet_float(payload: dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        parsed = safe_float(payload.get(key))
        if parsed is not None:
            return float(parsed)
    return None


def _wallet_outcome_index(outcome: Any) -> Optional[int]:
    outcome_text = str(outcome or "").strip().lower()
    if outcome_text in {"yes", "buy_yes"}:
        return 0
    if outcome_text in {"no", "buy_no"}:
        return 1
    return None


def _live_trading_position_to_wallet_row(row: LiveTradingPosition) -> Optional[dict[str, Any]]:
    token_id = str(row.token_id or "").strip()
    if not token_id:
        return None
    size = max(0.0, safe_float(row.size, 0.0) or 0.0)
    if size <= 0.0:
        return None
    market_id = str(row.market_id or "").strip() or token_id
    average_cost = safe_float(row.average_cost, None)
    current_price = safe_float(row.current_price, None)
    initial_value = (
        float(size * average_cost)
        if average_cost is not None and average_cost > 0.0
        else None
    )
    current_value = (
        float(size * current_price)
        if current_price is not None and current_price > 0.0
        else None
    )
    unrealized_pnl = safe_float(row.unrealized_pnl, None)
    outcome_text = str(row.outcome or "").strip()
    outcome_index = _wallet_outcome_index(outcome_text)
    payload: dict[str, Any] = {
        "asset": token_id,
        "asset_id": token_id,
        "token_id": token_id,
        "tokenId": token_id,
        "conditionId": market_id,
        "condition_id": market_id,
        "market": market_id,
        "market_id": market_id,
        "marketId": market_id,
        "title": str(row.market_question or "").strip(),
        "market_question": str(row.market_question or "").strip(),
        "marketQuestion": str(row.market_question or "").strip(),
        "question": str(row.market_question or "").strip(),
        "outcome": outcome_text,
        "position_side": outcome_text.lower(),
        "side": outcome_text.lower(),
        "size": float(size),
        "positionSize": float(size),
        "shares": float(size),
        "avgPrice": float(average_cost) if average_cost is not None and average_cost > 0.0 else None,
        "avg_price": float(average_cost) if average_cost is not None and average_cost > 0.0 else None,
        "avgCost": float(average_cost) if average_cost is not None and average_cost > 0.0 else None,
        "average_cost": float(average_cost) if average_cost is not None and average_cost > 0.0 else None,
        "currentPrice": float(current_price) if current_price is not None and current_price > 0.0 else None,
        "current_price": float(current_price) if current_price is not None and current_price > 0.0 else None,
        "curPrice": float(current_price) if current_price is not None and current_price > 0.0 else None,
        "cur_price": float(current_price) if current_price is not None and current_price > 0.0 else None,
        "markPrice": float(current_price) if current_price is not None and current_price > 0.0 else None,
        "mark_price": float(current_price) if current_price is not None and current_price > 0.0 else None,
        "initialValue": initial_value,
        "initial_value": initial_value,
        "currentValue": current_value,
        "current_value": current_value,
        "cashPnl": unrealized_pnl,
        "cash_pnl": unrealized_pnl,
        "unrealizedPnl": unrealized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "redeemable": bool(getattr(row, "redeemable", False)),
        "counts_as_open": coerce_bool(getattr(row, "counts_as_open", True), default=True),
        "endDate": str(getattr(row, "end_date", "") or "").strip() or None,
        "end_date": str(getattr(row, "end_date", "") or "").strip() or None,
    }
    if outcome_index is not None:
        payload["outcomeIndex"] = outcome_index
        payload["outcome_index"] = outcome_index
    return payload


def _extract_market_token_ids(market_info: dict[str, Any]) -> list[str]:
    for key in ("token_ids", "tokenIds", "clob_token_ids", "clobTokenIds"):
        raw = market_info.get(key)
        if not isinstance(raw, list):
            continue
        out = [str(token_id or "").strip() for token_id in raw if str(token_id or "").strip()]
        if out:
            return out
    return []


def _extract_market_outcomes(market_info: dict[str, Any]) -> list[str]:
    raw = market_info.get("outcomes")
    if not isinstance(raw, list):
        return []
    return [str(value or "").strip().lower() for value in raw if str(value or "").strip()]


def _extract_binary_market_prices(market_info: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    yes_price = safe_float(market_info.get("yes_price"))
    no_price = safe_float(market_info.get("no_price"))
    if yes_price is not None and no_price is not None:
        return yes_price, no_price

    outcome_prices = market_info.get("outcome_prices")
    if not isinstance(outcome_prices, list):
        outcome_prices = market_info.get("outcomePrices")
    if not isinstance(outcome_prices, list) or len(outcome_prices) < 2:
        return yes_price, no_price

    outcomes = _extract_market_outcomes(market_info)
    if outcomes and len(outcomes) == len(outcome_prices):
        for index, label in enumerate(outcomes):
            parsed_price = safe_float(outcome_prices[index])
            if parsed_price is None:
                continue
            if label == "yes":
                yes_price = parsed_price
            elif label == "no":
                no_price = parsed_price

    if yes_price is None:
        yes_price = safe_float(outcome_prices[0])
    if no_price is None:
        no_price = safe_float(outcome_prices[1])
    return yes_price, no_price


def _normalize_wallet_outcome_and_direction(
    wallet_position: dict[str, Any],
    *,
    market_info: dict[str, Any],
    token_id: str,
) -> tuple[str, str]:
    outcome_text = _read_wallet_text(wallet_position, "outcome", "position_side", "side").lower()
    if outcome_text in {"yes", "buy_yes"}:
        return "yes", "buy_yes"
    if outcome_text in {"no", "buy_no"}:
        return "no", "buy_no"

    outcome_index = safe_int(
        wallet_position.get("outcomeIndex") if wallet_position.get("outcomeIndex") is not None else wallet_position.get("outcome_index"),
        None,
    )
    token_ids = _extract_market_token_ids(market_info)
    if outcome_index is None and token_ids:
        normalized_token = _wallet_position_token_key(token_id)
        for index, known_token in enumerate(token_ids):
            if _wallet_position_token_key(known_token) == normalized_token:
                outcome_index = index
                break

    if outcome_index is None:
        return "", ""

    outcomes = _extract_market_outcomes(market_info)
    if outcomes and 0 <= outcome_index < len(outcomes):
        label = outcomes[outcome_index]
        if label == "yes":
            return "yes", "buy_yes"
        if label == "no":
            return "no", "buy_no"

    if outcome_index == 0:
        return "yes", "buy_yes"
    if outcome_index == 1:
        return "no", "buy_no"
    return "", ""


def _normalize_wallet_position_row(
    wallet_position: dict[str, Any],
    *,
    market_info: dict[str, Any],
) -> Optional[dict[str, Any]]:
    token_id = _read_wallet_text(wallet_position, "asset", "asset_id", "assetId", "token_id", "tokenId")
    if not token_id:
        return None

    size = _read_wallet_float(wallet_position, "size", "positionSize", "shares", "amount")
    if size is None or size <= 0.0:
        return None

    condition_id = _normalize_condition_id(
        _read_wallet_text(wallet_position, "conditionId", "condition_id", "market", "market_id", "marketId")
    )
    market_id = condition_id or _read_wallet_text(wallet_position, "market", "market_id", "marketId") or token_id

    avg_price = _read_wallet_float(wallet_position, "avgPrice", "avg_price", "avgCost", "average_cost")
    initial_value = _read_wallet_float(wallet_position, "initialValue", "initial_value")
    current_price = _read_wallet_float(
        wallet_position,
        "currentPrice",
        "current_price",
        "curPrice",
        "cur_price",
        "markPrice",
        "mark_price",
        "price",
    )
    current_value = _read_wallet_float(wallet_position, "currentValue", "current_value")
    unrealized_pnl = _read_wallet_float(wallet_position, "cashPnl", "cash_pnl", "unrealizedPnl", "unrealized_pnl")

    if (avg_price is None or avg_price <= 0.0) and initial_value is not None and initial_value > 0.0 and size > 0.0:
        avg_price = initial_value / size
    if (initial_value is None or initial_value <= 0.0) and avg_price is not None and avg_price > 0.0:
        initial_value = size * avg_price
    if (current_price is None or current_price <= 0.0) and current_value is not None and current_value > 0.0 and size > 0.0:
        current_price = current_value / size
    if current_value is None and current_price is not None and current_price > 0.0:
        current_value = current_price * size
    if unrealized_pnl is None and current_value is not None and initial_value is not None:
        unrealized_pnl = current_value - initial_value

    outcome, direction = _normalize_wallet_outcome_and_direction(
        wallet_position,
        market_info=market_info,
        token_id=token_id,
    )

    token_ids = _extract_market_token_ids(market_info)
    outcomes = _extract_market_outcomes(market_info)
    yes_token_id = ""
    no_token_id = ""
    if token_ids:
        yes_index = 0
        no_index = 1 if len(token_ids) > 1 else 0
        for index, label in enumerate(outcomes):
            if label == "yes":
                yes_index = index
            elif label == "no":
                no_index = index
        if 0 <= yes_index < len(token_ids):
            yes_token_id = str(token_ids[yes_index] or "").strip()
        if 0 <= no_index < len(token_ids):
            no_token_id = str(token_ids[no_index] or "").strip()

    yes_price, no_price = _extract_binary_market_prices(market_info)
    market_slug = _read_wallet_text(market_info, "slug")
    event_slug = _read_wallet_text(market_info, "event_slug", "eventSlug")
    market_question = (
        _read_wallet_text(wallet_position, "title", "market_question", "marketQuestion", "question")
        or _read_wallet_text(market_info, "question", "market_question", "title")
    )
    market_url = (
        build_polymarket_market_url(
            event_slug=event_slug or None,
            market_slug=market_slug or None,
            market_id=market_id,
            condition_id=condition_id or market_id,
        )
        or ""
    ).strip()

    return {
        "token_id": token_id,
        "market_id": market_id,
        "condition_id": condition_id or None,
        "market_question": market_question or None,
        "outcome": outcome or None,
        "direction": direction or None,
        "size": float(size),
        "avg_price": float(avg_price) if avg_price is not None and avg_price > 0.0 else None,
        "current_price": float(current_price) if current_price is not None and current_price > 0.0 else None,
        "initial_value": float(initial_value) if initial_value is not None and initial_value > 0.0 else None,
        "current_value": float(current_value) if current_value is not None and current_value > 0.0 else None,
        "unrealized_pnl": float(unrealized_pnl) if unrealized_pnl is not None else None,
        "yes_token_id": yes_token_id or None,
        "no_token_id": no_token_id or None,
        "yes_price": float(yes_price) if yes_price is not None else None,
        "no_price": float(no_price) if no_price is not None else None,
        "market_slug": market_slug or None,
        "event_slug": event_slug or None,
        "market_url": market_url or None,
        "redeemable": coerce_bool(wallet_position.get("redeemable"), default=False),
        "counts_as_open": coerce_bool(wallet_position.get("counts_as_open"), default=True),
        "end_date": _read_wallet_text(wallet_position, "endDate", "end_date") or None,
    }


def _active_statuses_for_mode(mode: Any) -> set[str]:
    mode_key = _normalize_mode_key(mode)
    if mode_key == "shadow":
        return SHADOW_ACTIVE_ORDER_STATUSES
    if mode_key == "live":
        return LIVE_ACTIVE_ORDER_STATUSES
    return OPEN_ORDER_STATUSES


def _is_active_order_status(mode: Any, status: Any) -> bool:
    return _normalize_status_key(status) in _active_statuses_for_mode(mode)


def _normalize_position_status(status: Any) -> str:
    value = str(status or "").strip().lower()
    if value == ACTIVE_POSITION_STATUS:
        return ACTIVE_POSITION_STATUS
    return INACTIVE_POSITION_STATUS


def _position_identity_key(mode: Any, market_id: Any, direction: Any) -> tuple[str, str, str]:
    return (
        _normalize_mode_key(mode),
        str(market_id or "").strip(),
        str(direction or "").strip().lower(),
    )


def _normalize_crypto_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


def _normalize_crypto_timeframe(value: Any) -> str:
    raw = str(value or "").strip().lower()
    compact = raw.replace("_", "").replace("-", "").replace(" ", "")
    if compact in {"5m", "5min", "5minute", "5minutes"}:
        return "5m"
    if compact in {"15m", "15min", "15minute", "15minutes"}:
        return "15m"
    if compact in {"1h", "1hr", "1hour", "60m", "60min"}:
        return "1h"
    if compact in {"4h", "4hr", "4hour", "240m", "240min"}:
        return "4h"
    return raw


def _extract_asset_timeframe_from_payload(payload: Any) -> tuple[str, str]:
    payload_dict = payload if isinstance(payload, dict) else {}
    strategy_context = payload_dict.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    live_market = payload_dict.get("live_market")
    live_market = live_market if isinstance(live_market, dict) else {}
    position_state = payload_dict.get("position_state")
    position_state = position_state if isinstance(position_state, dict) else {}

    candidates = [payload_dict, strategy_context, live_market, position_state]

    asset = ""
    timeframe = ""
    for candidate in candidates:
        if not asset:
            asset = _normalize_crypto_asset(candidate.get("asset") or candidate.get("coin") or candidate.get("symbol"))
        if not timeframe:
            timeframe = _normalize_crypto_timeframe(
                candidate.get("timeframe") or candidate.get("cadence") or candidate.get("interval")
            )
        if asset and timeframe:
            break
    return asset, timeframe


def _position_cap_scope_key(
    *,
    position_cap_scope: str,
    mode: Any,
    market_id: Any,
    direction: Any,
    payload: Any,
) -> tuple[str, str, str] | None:
    mode_key = _normalize_mode_key(mode)
    market_key = str(market_id or "").strip()
    if not market_key:
        return None
    direction_key = str(direction or "").strip().lower() or "__unknown__"
    scope = str(position_cap_scope or "market_direction").strip().lower()
    if scope == "market":
        return mode_key, market_key, "__market__"
    if scope == "asset_timeframe":
        asset, timeframe = _extract_asset_timeframe_from_payload(payload)
        asset_key = asset or f"market:{market_key}"
        timeframe_key = timeframe or "__unknown_timeframe__"
        return mode_key, asset_key, timeframe_key
    return mode_key, market_key, direction_key


def _normalize_direction_side(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    token = raw.replace("-", "_").replace(" ", "_")
    if token in {"yes", "buy_yes", "sell_yes", "buy", "long", "up"}:
        return "YES"
    if token in {"no", "buy_no", "sell_no", "sell", "short", "down"}:
        return "NO"
    if token.startswith("buy_yes") or token.startswith("sell_yes"):
        return "YES"
    if token.startswith("buy_no") or token.startswith("sell_no"):
        return "NO"
    return None


def _extract_outcome_labels(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    labels: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            label = str(
                item.get("label")
                or item.get("outcome")
                or item.get("name")
                or item.get("title")
                or item.get("value")
                or ""
            ).strip()
        else:
            label = str(item or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _extract_market_outcome_labels(market: dict[str, Any]) -> list[str]:
    for key in ("outcome_labels", "outcomeLabels", "outcomes", "labels"):
        labels = _extract_outcome_labels(market.get(key))
        if labels:
            return labels
    return _extract_outcome_labels(market.get("tokens"))


def _find_signal_market(signal_payload: dict[str, Any], market_id: str) -> Optional[dict[str, Any]]:
    raw_markets = signal_payload.get("markets")
    if not isinstance(raw_markets, list):
        return None
    normalized_market_id = str(market_id or "").strip().lower()
    first_market: Optional[dict[str, Any]] = None
    market_dict_count = 0
    for raw_market in raw_markets:
        if not isinstance(raw_market, dict):
            continue
        market_dict_count += 1
        if first_market is None:
            first_market = raw_market
        if not normalized_market_id:
            continue
        for candidate in (
            raw_market.get("id"),
            raw_market.get("market_id"),
            raw_market.get("condition_id"),
            raw_market.get("conditionId"),
            raw_market.get("slug"),
            raw_market.get("market_slug"),
            raw_market.get("ticker"),
        ):
            candidate_text = str(candidate or "").strip().lower()
            if candidate_text and candidate_text == normalized_market_id:
                return raw_market
    if market_dict_count == 1:
        return first_market
    return None


def _resolve_direction_label_from_signal(
    signal_payload: dict[str, Any],
    *,
    market_id: str,
    direction_side: str,
) -> Optional[str]:
    if direction_side not in {"YES", "NO"}:
        return None
    labels: list[str] = []
    matched_market = _find_signal_market(signal_payload, market_id)
    if isinstance(matched_market, dict):
        labels = _extract_market_outcome_labels(matched_market)
    if not labels:
        for key in ("outcome_labels", "outcomeLabels", "outcomes", "labels"):
            labels = _extract_outcome_labels(signal_payload.get(key))
            if labels:
                break
    index = 0 if direction_side == "YES" else 1
    if len(labels) <= index:
        return None
    label = str(labels[index] or "").strip()
    return label or None


def _resolve_binary_labels_from_signal(
    signal_payload: dict[str, Any],
    *,
    market_id: str,
) -> tuple[Optional[str], Optional[str]]:
    labels: list[str] = []
    matched_market = _find_signal_market(signal_payload, market_id)
    if isinstance(matched_market, dict):
        labels = _extract_market_outcome_labels(matched_market)
    if not labels:
        for key in ("outcome_labels", "outcomeLabels", "outcomes", "labels"):
            labels = _extract_outcome_labels(signal_payload.get(key))
            if labels:
                break
    yes_label = str(labels[0] or "").strip() if len(labels) > 0 else ""
    no_label = str(labels[1] or "").strip() if len(labels) > 1 else ""
    return (yes_label or None, no_label or None)


def _resolve_binary_labels_from_market_payload(
    market_payload: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    labels = _extract_market_outcome_labels(market_payload)
    yes_label = str(labels[0] or "").strip() if len(labels) > 0 else ""
    no_label = str(labels[1] or "").strip() if len(labels) > 1 else ""
    return (yes_label or None, no_label or None)


def _resolve_direction_label_from_payload(payload: dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    leg = payload.get("leg")
    if isinstance(leg, dict):
        for key in ("outcome_label", "outcomeLabel", "outcome_name", "outcomeName"):
            label = str(leg.get(key) or "").strip()
            if label:
                return label
        outcome = str(leg.get("outcome") or "").strip()
        if outcome and _normalize_direction_side(outcome) is None:
            return outcome
    for key in ("outcome_label", "outcomeLabel"):
        label = str(payload.get(key) or "").strip()
        if label:
            return label
    outcome = str(payload.get("outcome") or "").strip()
    if outcome and _normalize_direction_side(outcome) is None:
        return outcome
    return None


def _resolve_direction_presentation(
    *,
    direction: Any,
    payload: Optional[dict[str, Any]] = None,
    signal_payload: Optional[dict[str, Any]] = None,
    market_payload: Optional[dict[str, Any]] = None,
    market_id: Any = None,
) -> tuple[Optional[str], str, Optional[str], Optional[str]]:
    payload_obj = payload if isinstance(payload, dict) else {}
    signal_payload_obj = signal_payload if isinstance(signal_payload, dict) else {}
    market_payload_obj = market_payload if isinstance(market_payload, dict) else {}
    direction_side = _normalize_direction_side(direction)
    leg = payload_obj.get("leg")
    leg = leg if isinstance(leg, dict) else {}
    leg_outcome_side = _normalize_direction_side(leg.get("outcome"))
    if leg_outcome_side:
        direction_side = leg_outcome_side
    resolved_market_id = str(market_id or leg.get("market_id") or payload_obj.get("market_id") or "").strip()
    yes_label, no_label = _resolve_binary_labels_from_signal(
        signal_payload_obj,
        market_id=resolved_market_id,
    )
    if not yes_label or not no_label:
        market_yes_label, market_no_label = _resolve_binary_labels_from_market_payload(market_payload_obj)
        yes_label = yes_label or market_yes_label
        no_label = no_label or market_no_label
    direction_label = _resolve_direction_label_from_payload(payload_obj)
    if not direction_label and direction_side:
        direction_label = yes_label if direction_side == "YES" else no_label
    if not direction_label and direction_side:
        direction_label = _resolve_direction_label_from_signal(
            signal_payload_obj, market_id=resolved_market_id, direction_side=direction_side
        )
    if not direction_label and direction_side:
        direction_label = direction_side
    if not direction_label:
        raw_direction = str(direction or "").strip()
        direction_label = raw_direction.upper() if raw_direction else "N/A"
    return direction_side, direction_label, yes_label, no_label


def _first_float_from_dicts(candidates: list[Any], keys: tuple[str, ...]) -> Optional[float]:
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in keys:
            parsed = safe_float(candidate.get(key))
            if parsed is not None:
                return float(parsed)
    return None


def _extract_live_provider_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    reconciliation = payload.get("provider_reconciliation")
    if not isinstance(reconciliation, dict):
        reconciliation = {}
    snapshot = reconciliation.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    provider_snapshot = payload.get("provider_snapshot")
    if not isinstance(provider_snapshot, dict):
        provider_snapshot = {}
    return {
        "reconciliation": reconciliation,
        "snapshot": snapshot,
        "provider_snapshot": provider_snapshot,
        "payload": payload,
    }


def _extract_live_fill_metrics(payload: dict[str, Any]) -> tuple[float, float, Optional[float]]:
    snapshots = _extract_live_provider_snapshot(payload)
    candidates: list[Any] = [
        snapshots["reconciliation"],
        snapshots["snapshot"],
        snapshots["provider_snapshot"],
        payload,
    ]
    filled_shares = max(
        0.0,
        _first_float_from_dicts(
            candidates,
            (
                "filled_size",
                "size_matched",
                "sizeMatched",
                "matched_size",
                "filled_shares",
                "executed_size",
            ),
        )
        or 0.0,
    )
    average_fill_price = _first_float_from_dicts(
        candidates,
        (
            "average_fill_price",
            "avg_fill_price",
            "avg_price",
            "avgFillPrice",
            "matched_price",
            "price",
            "limit_price",
        ),
    )
    filled_notional_usd = max(
        0.0,
        _first_float_from_dicts(
            candidates,
            (
                "filled_notional_usd",
                "filled_notional",
                "matched_notional",
                "matched_amount",
                "executed_notional",
            ),
        )
        or 0.0,
    )
    if filled_notional_usd <= 0.0 and filled_shares > 0.0 and average_fill_price is not None and average_fill_price > 0:
        filled_notional_usd = filled_shares * average_fill_price
    if filled_shares <= 0.0 and filled_notional_usd > 0.0 and average_fill_price is not None and average_fill_price > 0:
        filled_shares = filled_notional_usd / average_fill_price
    return filled_notional_usd, filled_shares, average_fill_price


def _live_active_notional(mode: Any, status: Any, row_notional: float, payload: dict[str, Any], executed_at: Any = None) -> float:
    mode_key = _normalize_mode_key(mode)
    if mode_key not in {"live", "shadow"}:
        return max(0.0, abs(row_notional))

    status_key = _normalize_status_key(status)
    filled_notional_usd, _, _ = _extract_live_fill_metrics(payload)
    if status_key in {"executed", "completed"}:
        if filled_notional_usd > 0.0:
            return filled_notional_usd
        return max(0.0, abs(row_notional))
    if status_key in {"open", "submitted"}:
        if filled_notional_usd > 0.0:
            return filled_notional_usd
        if executed_at is not None:
            return max(0.0, abs(row_notional))
        return 0.0
    return 0.0


def _resolve_provider_order_ids(
    *,
    order_payload: dict[str, Any],
    session_order: Optional[ExecutionSessionOrder],
) -> tuple[Optional[str], Optional[str]]:
    provider_order_candidates: list[str] = []
    provider_clob_candidates: list[str] = []
    execution_payload = order_payload.get("execution_session")
    if not isinstance(execution_payload, dict):
        execution_payload = {}

    for value in (
        order_payload.get("order_id"),
        order_payload.get("provider_order_id"),
        execution_payload.get("provider_order_id"),
        getattr(session_order, "provider_order_id", None),
    ):
        text = str(value or "").strip()
        if text:
            provider_order_candidates.append(text)

    for value in (
        order_payload.get("clob_order_id"),
        order_payload.get("provider_clob_order_id"),
        execution_payload.get("provider_clob_order_id"),
        getattr(session_order, "provider_clob_order_id", None),
    ):
        text = str(value or "").strip()
        if text:
            provider_clob_candidates.append(text)

    provider_order_id = provider_order_candidates[0] if provider_order_candidates else None
    provider_clob_order_id = provider_clob_candidates[0] if provider_clob_candidates else None
    return provider_order_id, provider_clob_order_id


async def _acquire_live_order_authority_recovery_lock(session: AsyncSession) -> None:
    await session.execute(
        sa_text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": _LIVE_ORDER_AUTHORITY_RECOVERY_LOCK_KEY},
    )


async def _reassign_deleted_live_order_references(
    session: AsyncSession,
    *,
    duplicate_order_id: str,
    canonical_order_id: str,
) -> None:
    if not duplicate_order_id or not canonical_order_id or duplicate_order_id == canonical_order_id:
        return
    await session.execute(
        sa_update(ExecutionSessionOrder)
        .where(ExecutionSessionOrder.trader_order_id == duplicate_order_id)
        .values(trader_order_id=canonical_order_id)
    )
    await session.execute(
        sa_update(StrategyExperimentAssignment)
        .where(StrategyExperimentAssignment.order_id == duplicate_order_id)
        .values(order_id=canonical_order_id)
    )


def _extract_live_order_authority_ids_from_payload(payload: dict[str, Any]) -> tuple[str, str]:
    provider_clob_id = str(payload.get("provider_clob_order_id") or "").strip()
    authority_payload = payload.get("live_wallet_authority")
    if not isinstance(authority_payload, dict):
        authority_payload = {}
    live_order_id = str(authority_payload.get("live_trading_order_id") or "").strip()
    return provider_clob_id, live_order_id


def _extract_pending_live_exit_authority_ids(payload: dict[str, Any]) -> tuple[str, str]:
    pending_exit = payload.get("pending_live_exit")
    if not isinstance(pending_exit, dict):
        return "", ""
    provider_clob_id = str(
        pending_exit.get("provider_clob_order_id")
        or pending_exit.get("exit_order_clob_id")
        or ""
    ).strip()
    exit_order_id = str(pending_exit.get("exit_order_id") or "").strip()
    return provider_clob_id, exit_order_id


def _provider_side_from_payload(payload: dict[str, Any]) -> str:
    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    raw_snapshot = snapshot.get("raw")
    if not isinstance(raw_snapshot, dict):
        raw_snapshot = {}
    return str(raw_snapshot.get("side") or snapshot.get("side") or "").strip().upper()


def _is_recovered_sell_authority_row(row: TraderOrder) -> bool:
    payload = dict(getattr(row, "payload_json", None) or {})
    return (
        str(getattr(row, "reason", None) or "").strip().lower() == "recovered from live venue authority"
        and _provider_side_from_payload(payload) == "SELL"
    )


def _value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _merge_missing_mapping_values(target: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    merged = dict(target)
    changed = False
    for key, source_value in source.items():
        target_value = merged.get(key)
        if isinstance(source_value, dict):
            if isinstance(target_value, dict):
                nested_value, nested_changed = _merge_missing_mapping_values(target_value, source_value)
                if nested_changed:
                    merged[key] = nested_value
                    changed = True
                continue
            if _value_is_missing(target_value):
                merged[key] = copy.deepcopy(source_value)
                changed = True
            continue
        if _value_is_missing(source_value) or not _value_is_missing(target_value):
            continue
        merged[key] = copy.deepcopy(source_value)
        changed = True
    return merged, changed


def _live_order_authority_strategy_payload(
    *,
    trader_source_configs: Any,
    source_key: str,
    strategy_key: str,
) -> dict[str, Any]:
    normalized_source_key = normalize_source_key(source_key)
    normalized_strategy_key = _normalize_strategy_for_source(normalized_source_key, strategy_key)
    strategy_payload: dict[str, Any] = {}
    if normalized_strategy_key:
        strategy_payload["strategy_type"] = normalized_strategy_key
    if not normalized_source_key:
        return strategy_payload
    try:
        source_configs = _normalize_source_configs(trader_source_configs)
    except Exception:
        return strategy_payload
    matched_source_config = next(
        (
            source_config
            for source_config in source_configs
            if str(source_config.get("source_key") or "").strip().lower() == normalized_source_key
        ),
        None,
    )
    if matched_source_config is None:
        return strategy_payload
    matched_strategy_key = str(matched_source_config.get("strategy_key") or "").strip().lower()
    resolved_strategy_key = normalized_strategy_key or matched_strategy_key
    if resolved_strategy_key:
        strategy_payload["strategy_type"] = resolved_strategy_key
    if normalized_strategy_key and matched_strategy_key and matched_strategy_key != normalized_strategy_key:
        return strategy_payload
    explicit_strategy_params = dict(matched_source_config.get("strategy_params") or {})
    if explicit_strategy_params:
        strategy_payload["strategy_exit_config"] = explicit_strategy_params
    return strategy_payload


def _live_order_authority_key_from_payload(payload: dict[str, Any]) -> str:
    provider_clob_id, live_order_id = _extract_live_order_authority_ids_from_payload(payload)
    if provider_clob_id:
        return f"clob:{provider_clob_id.lower()}"
    if live_order_id:
        return f"live:{live_order_id.lower()}"
    return ""


def _live_order_authority_key_from_row(row: TraderOrder) -> str:
    payload = dict(getattr(row, "payload_json", None) or {})
    provider_clob_id = str(
        getattr(row, "provider_clob_order_id", None)
        or extract_trader_order_provider_clob_order_id(payload)
        or ""
    ).strip()
    if provider_clob_id:
        return f"clob:{provider_clob_id.lower()}"
    provider_order_id = str(
        getattr(row, "provider_order_id", None)
        or extract_trader_order_provider_order_id(payload)
        or ""
    ).strip()
    if provider_order_id:
        return f"provider:{provider_order_id.lower()}"
    _provider_clob_id, live_order_id = _extract_live_order_authority_ids_from_payload(payload)
    if live_order_id:
        return f"live:{live_order_id.lower()}"
    return ""


def _live_order_authority_keys_from_row(row: TraderOrder) -> list[str]:
    payload = dict(getattr(row, "payload_json", None) or {})
    keys: list[str] = []
    primary_key = _live_order_authority_key_from_row(row)
    if primary_key:
        keys.append(primary_key)
    row_clob_id = str(getattr(row, "provider_clob_order_id", None) or "").strip()
    payload_clob_id, payload_live_order_id = _extract_live_order_authority_ids_from_payload(payload)
    for clob_id in (row_clob_id, payload_clob_id):
        if clob_id:
            keys.append(f"clob:{clob_id.lower()}")
    provider_order_id = str(
        getattr(row, "provider_order_id", None)
        or extract_trader_order_provider_order_id(payload)
        or ""
    ).strip()
    for live_order_id in (provider_order_id, payload_live_order_id):
        if live_order_id:
            keys.append(f"live:{live_order_id.lower()}")
            keys.append(f"provider:{live_order_id.lower()}")
    return list(dict.fromkeys(keys))


def _is_live_authority_terminal_order_status(status: Any) -> bool:
    key = _normalize_status_key(status)
    return key in RESOLVED_ORDER_STATUSES or key in REALIZED_ORDER_STATUSES or key.startswith("closed")


def _position_close_has_live_terminal_authority(position_close: dict[str, Any]) -> bool:
    price_source = str(position_close.get("price_source") or "").strip().lower()
    close_trigger = str(position_close.get("close_trigger") or "").strip().lower()
    if price_source in {"wallet_activity", "wallet_trade", "wallet_flat_override"}:
        return True
    if close_trigger in {"wallet_activity", "wallet_summary_recovery", "external_wallet_flatten"}:
        return True
    for key in (
        "wallet_trade_id",
        "wallet_activity_transaction_hash",
        "wallet_activity_id",
        "wallet_closed_position_timestamp",
    ):
        if str(position_close.get(key) or "").strip():
            return True
    return False


def _terminal_status_for_position_close(position_close: dict[str, Any], existing_profit: Any) -> tuple[str, float]:
    pnl = safe_float(position_close.get("realized_pnl"), safe_float(existing_profit, 0.0) or 0.0) or 0.0
    normalized_pnl = 0.0 if abs(float(pnl)) <= 1e-9 else float(pnl)
    return ("closed_win" if normalized_pnl >= 0.0 else "closed_loss", normalized_pnl)


def _live_order_authority_row_sort_key(row: TraderOrder) -> tuple[int, int, int, int, float, float, str]:
    updated_at = row.updated_at or row.executed_at or row.created_at or _now()
    created_at = row.created_at or row.updated_at or updated_at
    return (
        1 if str(row.reason or "").strip().lower() == "recovered from live venue authority" else 0,
        1 if not str(row.decision_id or "").strip() else 0,
        1 if not str(row.signal_id or "").strip() else 0,
        0 if _is_live_authority_terminal_order_status(row.status) else 1,
        -updated_at.timestamp(),
        -created_at.timestamp(),
        str(row.id or ""),
    )


def _dedupe_live_authority_rows(rows: list[TraderOrder]) -> list[TraderOrder]:
    canonical_by_key: dict[str, TraderOrder] = {}
    ordered_rows: list[TraderOrder] = []
    for row in rows:
        if _normalize_mode_key(getattr(row, "mode", None)) != "live":
            ordered_rows.append(row)
            continue
        authority_key = _live_order_authority_key_from_row(row)
        if not authority_key:
            ordered_rows.append(row)
            continue
        current = canonical_by_key.get(authority_key)
        if current is None:
            canonical_by_key[authority_key] = row
            ordered_rows.append(row)
            continue
        if _live_order_authority_row_sort_key(row) < _live_order_authority_row_sort_key(current):
            canonical_by_key[authority_key] = row
            replace_index = ordered_rows.index(current)
            ordered_rows[replace_index] = row
    return ordered_rows


def _sync_pending_live_exit_from_live_order(
    row: TraderOrder,
    *,
    live_row: LiveTradingOrder,
    now: datetime,
) -> bool:
    payload = copy.deepcopy(row.payload_json or {})
    pending_exit = payload.get("pending_live_exit")
    if not isinstance(pending_exit, dict):
        return False

    normalized_provider_status = _normalize_status_key(live_row.status)
    filled_size = max(0.0, safe_float(live_row.filled_size, 0.0) or 0.0)
    average_fill_price = safe_float(live_row.average_fill_price, None)
    filled_notional_usd = 0.0
    if average_fill_price is not None and average_fill_price > 0.0 and filled_size > 0.0:
        filled_notional_usd = filled_size * average_fill_price
    elif filled_size > 0.0:
        limit_price = safe_float(live_row.price, 0.0) or 0.0
        if limit_price > 0.0:
            filled_notional_usd = filled_size * limit_price

    if normalized_provider_status == "filled":
        pending_status = "filled"
    elif normalized_provider_status in {"cancelled", "expired"}:
        pending_status = "cancelled"
    elif normalized_provider_status in {"failed", "rejected", "error"}:
        pending_status = "cancelled"
    else:
        pending_status = "submitted"

    snapshot = {
        "id": str(live_row.id or ""),
        "clob_order_id": str(live_row.clob_order_id or ""),
        "normalized_status": normalized_provider_status,
        "status": str(live_row.status or ""),
        "side": str(live_row.side or "").strip().upper() or None,
        "size": safe_float(live_row.size, None),
        "filled_size": filled_size,
        "average_fill_price": average_fill_price,
        "filled_notional_usd": filled_notional_usd,
        "limit_price": safe_float(live_row.price, None),
        "updated_at": to_iso(live_row.updated_at),
    }

    changed = False
    for key, value in (
        ("status", pending_status),
        ("provider_status", normalized_provider_status or str(live_row.status or "").strip().lower() or None),
        ("exit_order_id", str(live_row.id or "").strip() or None),
        ("provider_clob_order_id", str(live_row.clob_order_id or "").strip() or None),
        ("filled_size", filled_size if filled_size > 0.0 else None),
        ("filled_notional_usd", filled_notional_usd if filled_notional_usd > 0.0 else None),
        ("average_fill_price", average_fill_price if average_fill_price is not None and average_fill_price > 0.0 else None),
        ("last_snapshot_at", to_iso(live_row.updated_at or now)),
        ("snapshot", snapshot),
    ):
        if value is None:
            continue
        if pending_exit.get(key) != value:
            pending_exit[key] = value
            changed = True

    if pending_status == "filled" and pending_exit.get("resolved_at") is None:
        pending_exit["resolved_at"] = to_iso(now)
        changed = True
    elif pending_status == "cancelled":
        last_error = str(live_row.error_message or "").strip() or f"provider_exit_status:{normalized_provider_status or 'cancelled'}"
        if pending_exit.get("last_error") != last_error:
            pending_exit["last_error"] = last_error
            changed = True

    if changed:
        payload["pending_live_exit"] = pending_exit
        row.payload_json = payload
        row.updated_at = now
    return changed


def _build_live_order_authority_payload(
    *,
    live_row: LiveTradingOrder,
    position_row: LiveTradingPosition | None,
    now: datetime,
) -> dict[str, Any]:
    live_status_key = _normalize_status_key(live_row.status)
    fill_size = max(0.0, safe_float(live_row.filled_size, 0.0) or 0.0)
    order_size = max(0.0, safe_float(live_row.size, 0.0) or 0.0)
    average_fill_price = safe_float(live_row.average_fill_price, None)
    if fill_size <= 0.0 and live_status_key == "filled" and order_size > 0.0:
        fill_size = order_size
    if position_row is not None:
        wallet_position_size = max(0.0, safe_float(position_row.size, 0.0) or 0.0)
        wallet_average_cost = safe_float(position_row.average_cost, None)
        if fill_size <= 0.0 and live_status_key == "filled" and wallet_position_size > 0.0:
            fill_size = wallet_position_size
        if (
            (average_fill_price is None or average_fill_price <= 0.0)
            and wallet_average_cost is not None
            and wallet_average_cost > 0.0
        ):
            average_fill_price = wallet_average_cost
    filled_notional_usd = 0.0
    if fill_size > 0.0 and average_fill_price is not None and average_fill_price > 0.0:
        filled_notional_usd = fill_size * average_fill_price
    if filled_notional_usd <= 0.0:
        fallback_price = safe_float(live_row.price, 0.0) or 0.0
        if fill_size > 0.0 and fallback_price > 0.0:
            filled_notional_usd = fill_size * fallback_price

    trader_status = _map_live_trading_order_status_to_trader_order_status(
        live_status=live_row.status,
        filled_notional_usd=filled_notional_usd,
        live_side=live_row.side,
    )
    entry_price = average_fill_price if average_fill_price is not None and average_fill_price > 0.0 else safe_float(live_row.price, None)
    order_notional_usd = filled_notional_usd

    provider_clob_id = str(live_row.clob_order_id or "").strip()
    provider_snapshot = {
        "normalized_status": _normalize_status_key(live_row.status),
        "status": str(live_row.status or ""),
        "side": str(live_row.side or "").strip().upper() or None,
        "clob_order_id": provider_clob_id or None,
        "asset_id": str(live_row.token_id or "").strip(),
        "filled_size": fill_size,
        "average_fill_price": average_fill_price,
        "filled_notional_usd": filled_notional_usd,
        "limit_price": safe_float(live_row.price, None),
        "size": safe_float(live_row.size, None),
        "updated_at": to_iso(live_row.updated_at),
    }
    payload_json: dict[str, Any] = {
        "provider_clob_order_id": provider_clob_id or None,
        "provider_reconciliation": {
            "source": "live_trading_orders_recovery",
            "snapshot_status": _normalize_status_key(live_row.status),
            "mapped_status": trader_status,
            "reconciled_at": to_iso(now),
            "filled_size": fill_size,
            "average_fill_price": average_fill_price,
            "filled_notional_usd": filled_notional_usd,
            "snapshot": provider_snapshot,
        },
        "live_wallet_authority": {
            "source": "live_trading_orders",
            "live_trading_order_id": str(live_row.id),
            "wallet_address": str(live_row.wallet_address or ""),
            "token_id": str(live_row.token_id or ""),
            "recovered_at": to_iso(now),
        },
        "token_id": str(live_row.token_id or "").strip(),
        "order_type": str(live_row.order_type or "").strip(),
    }
    if position_row is not None:
        payload_json["position_state"] = {
            "last_mark_price": safe_float(position_row.average_cost, None),
            "last_mark_source": "live_wallet_recovery",
            "last_marked_at": to_iso(position_row.updated_at),
        }
    return {
        "payload_json": payload_json,
        "trader_status": trader_status,
        "entry_price": entry_price,
        "order_notional_usd": order_notional_usd,
    }


def _terminal_live_order_recovery_status(
    *,
    recovered_status: str,
    live_row: LiveTradingOrder,
) -> str:
    status_key = _normalize_status_key(recovered_status)
    if status_key != "failed":
        return status_key
    error_text = str(getattr(live_row, "error_message", None) or "").strip().lower()
    if "no orders found to match with fak order" in error_text:
        return "rejected"
    return status_key


def _should_adopt_terminal_live_order_status(
    *,
    existing_status: str,
    recovered_status: str,
    recovery_state: dict[str, Any],
    merged_payload: dict[str, Any],
) -> bool:
    if existing_status not in LIVE_ACTIVE_ORDER_STATUSES:
        return False
    if recovered_status not in {"cancelled", "failed", "rejected", "error"}:
        return False

    filled_notional = max(0.0, safe_float(recovery_state.get("order_notional_usd"), 0.0) or 0.0)
    if filled_notional > 0.0:
        return False

    provider_reconciliation = merged_payload.get("provider_reconciliation")
    provider_reconciliation = provider_reconciliation if isinstance(provider_reconciliation, dict) else {}
    snapshot = provider_reconciliation.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    filled_size = max(
        0.0,
        safe_float(
            provider_reconciliation.get("filled_size"),
            safe_float(snapshot.get("filled_size"), 0.0),
        )
        or 0.0,
    )
    if filled_size > 0.0:
        return False

    snapshot_status = _normalize_status_key(
        provider_reconciliation.get("snapshot_status")
        or provider_reconciliation.get("mapped_status")
        or snapshot.get("normalized_status")
        or snapshot.get("status")
    )
    if snapshot_status in {"filled", "partially_filled", "open", "submitted", "pending"}:
        return False

    return True


def _sync_recovered_order_hot_state(row: TraderOrder) -> None:
    import services.trader_hot_state as _hs

    payload = dict(row.payload_json or {})
    token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip()
    status_key = _normalize_status_key(row.status)
    copy_source_wallet = _extract_copy_source_wallet_from_payload(payload)

    if status_key in LIVE_ACTIVE_ORDER_STATUSES:
        _hs.upsert_active_order(
            trader_id=str(row.trader_id or ""),
            mode=str(row.mode or "live"),
            order_id=str(row.id or ""),
            status=str(row.status or "open"),
            market_id=str(row.market_id or ""),
            direction=str(row.direction or ""),
            source=str(row.source or ""),
            notional_usd=float(row.notional_usd or 0.0),
            entry_price=float(row.entry_price or row.effective_price or 0.0),
            token_id=token_id,
            filled_shares=0.0,
            payload=payload,
            copy_source_wallet=copy_source_wallet,
        )
        return

    if status_key in REALIZED_ORDER_STATUSES or status_key.startswith("closed"):
        _hs.record_order_resolved(
            trader_id=str(row.trader_id or ""),
            mode=str(row.mode or "live"),
            order_id=str(row.id or ""),
            market_id=str(row.market_id or ""),
            direction=str(row.direction or ""),
            source=str(row.source or ""),
            status=str(row.status or ""),
            actual_profit=float(row.actual_profit or 0.0),
            payload=payload,
            copy_source_wallet=copy_source_wallet,
            updated_at=getattr(row, "updated_at", None),
        )
        return

    _hs.record_order_cancelled(
        trader_id=str(row.trader_id or ""),
        mode=str(row.mode or "live"),
        order_id=str(row.id or ""),
        market_id=str(row.market_id or ""),
        source=str(row.source or ""),
        copy_source_wallet=copy_source_wallet,
    )


def _cleanup_timeout_reason(note_reason: str) -> bool:
    normalized = str(note_reason or "").strip().lower()
    return normalized.startswith("max_open_order_timeout")


def _cleanup_rescue_side_from_direction(direction: Any) -> str:
    direction_key = str(direction or "").strip().lower()
    if direction_key.startswith("sell"):
        return "SELL"
    return "BUY"


def _cleanup_rescue_token_id(
    *,
    order_payload: dict[str, Any],
    session_order: Optional[ExecutionSessionOrder],
) -> str:
    candidates: list[str] = []
    execution_payload = order_payload.get("execution_session")
    if not isinstance(execution_payload, dict):
        execution_payload = {}
    leg_payload = order_payload.get("leg")
    if not isinstance(leg_payload, dict):
        leg_payload = {}
    session_payload = dict(session_order.payload_json or {}) if session_order is not None else {}
    session_leg_payload = session_payload.get("leg")
    if not isinstance(session_leg_payload, dict):
        session_leg_payload = {}

    for value in (
        order_payload.get("token_id"),
        order_payload.get("selected_token_id"),
        leg_payload.get("token_id"),
        execution_payload.get("token_id"),
        session_payload.get("token_id"),
        session_leg_payload.get("token_id"),
        order_payload.get("yes_token_id"),
        order_payload.get("no_token_id"),
    ):
        text = str(value or "").strip()
        if text:
            candidates.append(text)
    return candidates[0] if candidates else ""


def _cleanup_rescue_shares(
    *,
    row: TraderOrder,
    order_payload: dict[str, Any],
    session_order: Optional[ExecutionSessionOrder],
) -> float:
    session_payload = dict(session_order.payload_json or {}) if session_order is not None else {}
    requested_shares = safe_float(order_payload.get("requested_shares"), None)
    if requested_shares is None:
        requested_shares = safe_float((order_payload.get("leg") or {}).get("requested_shares"), None)
    if requested_shares is None:
        requested_shares = safe_float(session_payload.get("requested_shares"), None)
    if requested_shares is None:
        requested_shares = safe_float(order_payload.get("shares"), None)
    if requested_shares is None:
        requested_shares = safe_float(order_payload.get("size"), None)
    if requested_shares is None:
        requested_notional = safe_float(row.notional_usd, 0.0) or 0.0
        entry_price = safe_float(row.entry_price, None)
        if entry_price is None or entry_price <= 0.0:
            entry_price = safe_float(order_payload.get("resolved_price"), None)
        if entry_price is not None and entry_price > 0.0 and requested_notional > 0.0:
            requested_shares = requested_notional / entry_price
    return max(0.0, float(requested_shares or 0.0))


def _cleanup_rescue_price(
    *,
    side: str,
    row: TraderOrder,
    order_payload: dict[str, Any],
    rescue_price_bps: float,
) -> float | None:
    base_price = safe_float(row.entry_price, None)
    if base_price is None or base_price <= 0.0:
        base_price = safe_float(row.effective_price, None)
    if base_price is None or base_price <= 0.0:
        base_price = safe_float(order_payload.get("resolved_price"), None)
    if base_price is None or base_price <= 0.0:
        base_price = safe_float(order_payload.get("fallback_price"), None)
    if base_price is None or base_price <= 0.0:
        return None
    bps = max(0.0, float(rescue_price_bps))
    if side == "SELL":
        adjusted = float(base_price) * (1.0 - (bps / 10_000.0))
    else:
        adjusted = float(base_price) * (1.0 + (bps / 10_000.0))
    return min(0.99, max(0.01, adjusted))


async def _attempt_timeout_taker_rescue(
    *,
    row: TraderOrder,
    order_payload: dict[str, Any],
    session_order: Optional[ExecutionSessionOrder],
    rescue_price_bps: float,
    rescue_time_in_force: str,
) -> dict[str, Any]:
    token_id = _cleanup_rescue_token_id(order_payload=order_payload, session_order=session_order)
    side = _cleanup_rescue_side_from_direction(row.direction)
    shares = _cleanup_rescue_shares(row=row, order_payload=order_payload, session_order=session_order)
    fallback_price = _cleanup_rescue_price(
        side=side,
        row=row,
        order_payload=order_payload,
        rescue_price_bps=rescue_price_bps,
    )
    result: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "status": "failed",
        "token_id": token_id or None,
        "side": side,
        "requested_shares": shares,
        "fallback_price": fallback_price,
        "time_in_force": rescue_time_in_force,
        "price_bps": float(rescue_price_bps),
        "order_id": None,
        "clob_order_id": None,
        "effective_price": None,
        "error": None,
        "payload": {},
    }
    if not token_id:
        result["error"] = "missing_token_id"
        return result
    if shares <= 0.0:
        result["error"] = "missing_shares"
        return result
    if fallback_price is None or fallback_price <= 0.0:
        result["error"] = "missing_price"
        return result

    result["attempted"] = True
    try:
        execution = await execute_live_order(
            token_id=token_id,
            side=side,
            size=float(shares),
            fallback_price=float(fallback_price),
            market_question=str(row.market_question or ""),
            opportunity_id=str(row.signal_id or row.id or ""),
            time_in_force=str(rescue_time_in_force or DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE).strip().upper() or "IOC",
            post_only=False,
            resolve_live_price=True,
            prefer_cached_price=True,
            enforce_fallback_bound=True,
        )
    except Exception as exc:
        result["error"] = str(exc)
        return result

    status_key = _normalize_status_key(execution.status)
    result["status"] = status_key or "failed"
    result["success"] = status_key in {"executed", "open", "submitted"}
    result["order_id"] = str(execution.order_id or "").strip() or None
    result["clob_order_id"] = str((execution.payload or {}).get("clob_order_id") or "").strip() or None
    result["effective_price"] = safe_float(execution.effective_price, None)
    result["error"] = str(execution.error_message or "").strip() or None
    result["payload"] = dict(execution.payload or {})
    return result


def _map_provider_snapshot_status(snapshot_status: str, filled_notional_usd: float) -> str:
    status_key = str(snapshot_status or "").strip().lower()
    if status_key in {"filled"}:
        return "executed"
    if status_key in {"partially_filled", "open"}:
        return "open"
    if status_key in {"pending"}:
        return "submitted"
    if status_key in {"cancelled", "expired", "failed"}:
        if filled_notional_usd > 0.0:
            return "executed"
        return "cancelled" if status_key in {"cancelled", "expired"} else "failed"
    return ""


def _map_live_trading_order_status_to_trader_order_status(
    *,
    live_status: Any,
    filled_notional_usd: float,
    live_side: Any,
) -> str:
    mapped_status = _map_provider_snapshot_status(str(live_status or ""), filled_notional_usd)
    side = str(live_side or "").strip().upper()
    if mapped_status == "executed":
        if side == "SELL":
            return "resolved"
        return "executed"
    if mapped_status == "open":
        return "open" if filled_notional_usd > 0.0 else "submitted"
    if mapped_status:
        return mapped_status
    return "submitted"


async def _infer_live_order_authority_candidate(
    session: AsyncSession,
    *,
    market_question: str,
    created_at: datetime,
    trader_ids: set[str] | None,
) -> dict[str, Any] | None:
    question = str(market_question or "").strip()
    if not question:
        return None

    window_start = created_at - timedelta(minutes=_LIVE_ORDER_AUTHORITY_RECOVERY_DECISION_LOOKBACK_MINUTES)
    window_end = created_at + timedelta(minutes=5)
    rows = (
        await session.execute(
            select(
                TraderDecision.id.label("decision_id"),
                TraderDecision.trader_id.label("trader_id"),
                TraderDecision.strategy_key.label("strategy_key"),
                TraderDecision.strategy_version.label("strategy_version"),
                TraderDecision.decision.label("decision"),
                TraderDecision.reason.label("decision_reason"),
                TraderDecision.created_at.label("decision_created_at"),
                TradeSignal.id.label("signal_id"),
                TradeSignal.source.label("signal_source"),
                TradeSignal.market_id.label("market_id"),
                TradeSignal.market_question.label("market_question"),
                TradeSignal.direction.label("direction"),
                TradeSignal.payload_json.label("signal_payload_json"),
                Trader.is_enabled.label("is_enabled"),
                Trader.is_paused.label("is_paused"),
                Trader.source_configs_json.label("trader_source_configs_json"),
            )
            .join(TradeSignal, TradeSignal.id == TraderDecision.signal_id)
            .join(Trader, Trader.id == TraderDecision.trader_id)
            .where(
                Trader.mode == "live",
                TradeSignal.market_question == question,
                TraderDecision.created_at >= window_start,
                TraderDecision.created_at <= window_end,
            )
            .order_by(desc(TraderDecision.created_at))
        )
    ).all()
    if not rows:
        return None

    best_by_trader: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in rows:
        trader_id = str(row.trader_id or "").strip()
        if not trader_id:
            continue
        if trader_ids is not None and trader_id not in trader_ids:
            continue
        direction = str(row.direction or "").strip().lower()
        market_id = str(row.market_id or "").strip()
        if not direction or not market_id:
            continue

        decision_key = _normalize_status_key(row.decision)
        decision_created_at = row.decision_created_at
        if decision_created_at is None:
            continue
        if decision_created_at.tzinfo is None:
            decision_created_at = decision_created_at.replace(tzinfo=timezone.utc)
        else:
            decision_created_at = decision_created_at.astimezone(timezone.utc)
        delta_seconds = abs((created_at - decision_created_at).total_seconds())
        if decision_key == "selected":
            score = 400.0
        elif decision_key == "failed":
            score = 320.0
        elif decision_key == "blocked":
            score = 180.0
        elif decision_key == "skipped":
            score = 120.0
        else:
            score = 60.0
        score -= min(delta_seconds / 60.0, 180.0)
        if bool(row.is_enabled):
            score += 8.0
        if bool(row.is_paused):
            score -= 4.0
        candidate = {
            "decision_id": str(row.decision_id or "").strip() or None,
            "trader_id": trader_id,
            "strategy_key": str(row.strategy_key or "").strip() or None,
            "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
            "signal_id": str(row.signal_id or "").strip() or None,
            "source": str(row.signal_source or "").strip() or "scanner",
            "decision_reason": str(row.decision_reason or "").strip() or None,
            "market_id": market_id,
            "market_question": question,
            "direction": direction,
            "decision_created_at": decision_created_at,
            "signal_payload": dict(row.signal_payload_json or {}),
            "trader_source_configs": row.trader_source_configs_json,
        }
        existing = best_by_trader.get(trader_id)
        if existing is None or score > existing[0]:
            best_by_trader[trader_id] = (score, candidate)

    if not best_by_trader:
        return None

    ranked = sorted(
        best_by_trader.values(),
        key=lambda item: (
            item[0],
            item[1]["decision_created_at"],
        ),
        reverse=True,
    )
    top_score, top_candidate = ranked[0]
    if len(ranked) > 1 and abs(top_score - ranked[1][0]) < 0.001:
        return None
    return top_candidate


async def recover_missing_live_trader_orders(
    session: AsyncSession,
    *,
    trader_ids: list[str] | None = None,
    commit: bool = True,
    broadcast: bool = True,
) -> dict[str, Any]:
    trader_id_filter = {str(value or "").strip() for value in (trader_ids or []) if str(value or "").strip()}
    await _acquire_live_order_authority_recovery_lock(session)
    # Only recover live authority from the last 24 hours, and only keep
    # terminal venue rows in a short lookback window. Old failed/cancelled
    # rows are already final and repeatedly rescanning them was starving the
    # pool without producing any new local authority.
    now = _now()
    _recovery_cutoff = now - timedelta(hours=_LIVE_ORDER_AUTHORITY_RECOVERY_WINDOW_HOURS)
    _recent_terminal_cutoff = now - timedelta(
        minutes=_LIVE_ORDER_AUTHORITY_RECOVERY_RECENT_TERMINAL_LOOKBACK_MINUTES
    )
    _live_row_timestamp = func.coalesce(LiveTradingOrder.updated_at, LiveTradingOrder.created_at)
    live_rows_query = (
        select(LiveTradingOrder)
        .where(
            or_(
                func.lower(func.coalesce(LiveTradingOrder.status, "")).in_(
                    ("open", "filled", "pending", "partially_filled")
                ),
                and_(
                    func.lower(func.coalesce(LiveTradingOrder.status, "")).in_(
                        ("failed", "cancelled", "expired")
                    ),
                    _live_row_timestamp >= _recent_terminal_cutoff,
                ),
            )
        )
        .where(_live_row_timestamp >= _recovery_cutoff)
        .order_by(desc(_live_row_timestamp))
    )
    if not trader_id_filter:
        live_rows_query = live_rows_query.limit(_LIVE_ORDER_AUTHORITY_RECOVERY_GENERAL_MAX_ROWS)
    live_rows = list(
        (
            await session.execute(live_rows_query)
        )
        .scalars()
        .all()
    )
    if not live_rows:
        if commit and session.in_transaction():
            await session.rollback()
        return {
            "recovered_orders": 0,
            "affected_traders": 0,
            "ambiguous_orders": 0,
            "collapsed_duplicates": 0,
            "adopted_existing_orders": 0,
        }

    live_provider_clob_ids = {
        str(row.clob_order_id or "").strip().lower()
        for row in live_rows
        if str(row.clob_order_id or "").strip()
    }
    live_order_ids = {
        str(row.id or "").strip().lower()
        for row in live_rows
        if str(row.id or "").strip()
    }

    live_orders_by_token: dict[str, LiveTradingPosition] = {}
    relevant_token_ids = {
        str(row.token_id or "").strip()
        for row in live_rows
        if str(row.token_id or "").strip()
    }
    position_rows: list[LiveTradingPosition] = []
    if relevant_token_ids:
        position_rows = list(
            (
                await session.execute(
                    select(LiveTradingPosition).where(LiveTradingPosition.token_id.in_(list(relevant_token_ids)))
                )
            )
            .scalars()
            .all()
        )
    for row in position_rows:
        token_key = _wallet_position_token_key(row.token_id)
        if not token_key:
            continue
        current = live_orders_by_token.get(token_key)
        if current is None or (current.updated_at or current.created_at or _now()) < (row.updated_at or row.created_at or _now()):
            live_orders_by_token[token_key] = row

    existing_rows_query = (
        select(TraderOrder)
        .where(TraderOrder.mode == "live")
        .where(
            or_(
                func.lower(func.coalesce(TraderOrder.status, "")).in_(
                    tuple(
                        LIVE_ACTIVE_ORDER_STATUSES
                        | RESOLVED_ORDER_STATUSES
                        | {"placing"}
                        | {"cancelled", "failed", "rejected", "error"}
                    )
                ),
                TraderOrder.created_at >= _recovery_cutoff,
                TraderOrder.updated_at >= _recovery_cutoff,
                TraderOrder.executed_at >= _recovery_cutoff,
            )
        )
    )
    if trader_id_filter:
        existing_rows_query = existing_rows_query.where(TraderOrder.trader_id.in_(list(trader_id_filter)))
    else:
        relevant_market_ids = {
            str(getattr(position_row, "market_id", "") or "").strip()
            for position_row in position_rows
            if str(getattr(position_row, "market_id", "") or "").strip()
        }
        relevant_provider_clob_ids = set(live_provider_clob_ids)
        if relevant_market_ids or relevant_provider_clob_ids:
            market_filters: list[Any] = [
                TraderOrder.market_id.is_(None),
                TraderOrder.market_id == "",
            ]
            if relevant_market_ids:
                market_filters.append(TraderOrder.market_id.in_(list(relevant_market_ids)))
            if relevant_provider_clob_ids:
                market_filters.append(
                    func.lower(func.coalesce(TraderOrder.provider_clob_order_id, "")).in_(list(relevant_provider_clob_ids))
                )
            existing_rows_query = existing_rows_query.where(or_(*market_filters))
    existing_rows = list(
        (
            await session.execute(existing_rows_query)
        )
        .scalars()
        .all()
    )
    global_authority_rows_by_key: dict[str, TraderOrder] = {}
    if trader_id_filter and (live_provider_clob_ids or live_order_ids):
        global_authority_filters: list[Any] = []
        if live_provider_clob_ids:
            global_authority_filters.append(
                func.lower(func.coalesce(TraderOrder.provider_clob_order_id, "")).in_(list(live_provider_clob_ids))
            )
        if live_order_ids:
            global_authority_filters.append(
                func.lower(func.coalesce(TraderOrder.provider_order_id, "")).in_(list(live_order_ids))
            )
        if global_authority_filters:
            global_authority_rows = list(
                (
                    await session.execute(
                        select(TraderOrder)
                        .where(TraderOrder.mode == "live")
                        .where(or_(*global_authority_filters))
                    )
                )
                .scalars()
                .all()
            )
            for global_authority_row in global_authority_rows:
                for authority_key in _live_order_authority_keys_from_row(global_authority_row):
                    current = global_authority_rows_by_key.get(authority_key)
                    if current is None or _live_order_authority_row_sort_key(
                        global_authority_row
                    ) < _live_order_authority_row_sort_key(current):
                        global_authority_rows_by_key[authority_key] = global_authority_row
    existing_order_ids = [str(row.id or "").strip() for row in existing_rows if str(row.id or "").strip()]
    execution_order_rows: list[ExecutionSessionOrder] = []
    if existing_order_ids:
        execution_order_rows = list(
            (
                await session.execute(
                    select(ExecutionSessionOrder)
                    .where(ExecutionSessionOrder.trader_order_id.in_(existing_order_ids))
                    .order_by(desc(ExecutionSessionOrder.created_at))
                )
            )
            .scalars()
            .all()
        )
    execution_order_by_trader_order: dict[str, ExecutionSessionOrder] = {}
    for execution_order_row in execution_order_rows:
        trader_order_id = str(execution_order_row.trader_order_id or "").strip()
        if not trader_order_id or trader_order_id in execution_order_by_trader_order:
            continue
        execution_order_by_trader_order[trader_order_id] = execution_order_row
    existing_trader_ids = {str(row.trader_id or "").strip() for row in existing_rows if str(row.trader_id or "").strip()}
    trader_source_configs_by_id: dict[str, Any] = {}
    if existing_trader_ids:
        trader_rows = (
            await session.execute(
                select(
                    Trader.id,
                    Trader.source_configs_json,
                ).where(Trader.id.in_(list(existing_trader_ids)))
            )
        ).all()
        trader_source_configs_by_id = {
            str(row.id or "").strip(): row.source_configs_json
            for row in trader_rows
            if str(row.id or "").strip()
        }
    existing_provider_rows_by_clob_id: dict[str, TraderOrder] = {}
    existing_rows_by_live_order_id: dict[str, TraderOrder] = {}
    pending_exit_rows_by_clob_id: dict[str, TraderOrder] = {}
    pending_exit_rows_by_live_order_id: dict[str, TraderOrder] = {}
    recovered_sell_rows_by_authority_key: dict[str, list[TraderOrder]] = {}
    rows_by_authority_key: dict[str, list[TraderOrder]] = {}
    rows_without_authority: list[TraderOrder] = []
    pre_submit_rows_by_token_key: dict[str, list[TraderOrder]] = {}
    existing_occupied_markets: set[tuple[str, str]] = set()
    recently_closed_recovery_guard_markets: set[tuple[str, str]] = set()
    affected_traders: set[str] = set()
    updated_execution_order_rows: dict[str, ExecutionSessionOrder] = {}
    updated_execution_leg_rows: dict[str, ExecutionSessionLeg] = {}
    updated_execution_session_rows: dict[str, ExecutionSession] = {}
    leg_state_updates: dict[str, dict[str, Any]] = {}
    touched_session_ids: set[str] = set()
    for row in existing_rows:
        payload = dict(row.payload_json or {})
        authority_key = _live_order_authority_key_from_row(row)
        is_recovered_sell_authority = _is_recovered_sell_authority_row(row)
        if authority_key:
            rows_by_authority_key.setdefault(authority_key, []).append(row)
            if is_recovered_sell_authority:
                recovered_sell_rows_by_authority_key.setdefault(authority_key, []).append(row)
        else:
            rows_without_authority.append(row)
        pending_exit_clob_id, pending_exit_live_order_id = _extract_pending_live_exit_authority_ids(payload)
        if pending_exit_clob_id:
            pending_exit_rows_by_clob_id[pending_exit_clob_id.lower()] = row
        if pending_exit_live_order_id:
            pending_exit_rows_by_live_order_id[pending_exit_live_order_id.lower()] = row
        row_status = str(row.status or "").strip().lower()
        provider_order_id = str(
            getattr(row, "provider_order_id", None)
            or extract_trader_order_provider_order_id(payload)
            or ""
        ).strip()
        submission_intent = payload.get("submission_intent")
        submission_intent_state = (
            _normalize_status_key(submission_intent.get("state"))
            if isinstance(submission_intent, dict)
            else ""
        )
        if (
            not authority_key
            and not provider_order_id
            and (
                (row_status == "placing" and submission_intent_state == "pre_submit_persisted")
                or (row_status == "failed" and submission_intent_state == "submit_timeout")
            )
            and isinstance(submission_intent, dict)
        ):
            token_key = _wallet_position_token_key(_extract_order_token_id(row))
            if token_key:
                pre_submit_rows_by_token_key.setdefault(token_key, []).append(row)
        row_trader_id = str(row.trader_id or "").strip()
        row_market_id = str(row.market_id or "").strip()
        has_authority = bool(authority_key or provider_order_id)
        if row_trader_id and row_market_id and not is_recovered_sell_authority and not has_authority:
            # "placing" must also occupy the market: if a pre-submit placeholder
            # exists for (trader, market), the provider-side order it represents
            # may not yet have its clob_order_id linked back to the row.  Without
            # this guard, authority recovery sees the unmatched live_trading_order
            # and creates a *second* trader_order for the same leg, producing
            # ghost duplicates (one with NULL provider_clob_order_id, one with
            # the real id).
            if row_status in OPEN_ORDER_STATUSES or row_status == "placing":
                existing_occupied_markets.add((row_trader_id, row_market_id))
            # Also block recovery for recently closed/resolved orders to prevent
            # phantom duplicates when a sell exit creates a new LiveTradingOrder
            # that the recovery sees as unmatched.
            elif row_status in RESOLVED_ORDER_STATUSES or row_status.startswith("closed"):
                row_updated = row.updated_at
                if row_updated is not None:
                    if row_updated.tzinfo is None:
                        row_updated = row_updated.replace(tzinfo=timezone.utc)
                    if (now - row_updated).total_seconds() < 600:  # 10 minutes
                        recently_closed_recovery_guard_markets.add((row_trader_id, row_market_id))

    collapsed_duplicates = 0
    for group in rows_by_authority_key.values():
        ordered_group = sorted(group, key=_live_order_authority_row_sort_key)
        canonical_row = ordered_group[0]
        canonical_trader_id = str(canonical_row.trader_id or "").strip()
        if canonical_trader_id:
            affected_traders.add(canonical_trader_id)
        canonical_payload = dict(canonical_row.payload_json or {})
        canonical_payload_changed = False

        for duplicate_row in ordered_group[1:]:
            duplicate_payload = dict(duplicate_row.payload_json or {})
            merged_payload, merged_changed = _merge_missing_mapping_values(canonical_payload, duplicate_payload)
            if merged_changed:
                canonical_payload = merged_payload
                canonical_payload_changed = True
            await _reassign_deleted_live_order_references(
                session,
                duplicate_order_id=str(duplicate_row.id or ""),
                canonical_order_id=str(canonical_row.id or ""),
            )
            await session.delete(duplicate_row)
            collapsed_duplicates += 1

        if canonical_payload_changed:
            canonical_row.payload_json = canonical_payload
            canonical_row.updated_at = now

        provider_clob_id, live_order_id = _extract_live_order_authority_ids_from_payload(canonical_payload)
        provider_order_id = str(
            getattr(canonical_row, "provider_order_id", None)
            or extract_trader_order_provider_order_id(canonical_payload)
            or ""
        ).strip()
        if provider_clob_id:
            provider_clob_id = provider_clob_id.lower()
            existing_provider_rows_by_clob_id[provider_clob_id] = canonical_row
        if live_order_id:
            existing_rows_by_live_order_id[live_order_id.lower()] = canonical_row
        if provider_order_id:
            existing_rows_by_live_order_id[provider_order_id.lower()] = canonical_row

    for row in rows_without_authority:
        payload = dict(row.payload_json or {})
        provider_clob_id, live_order_id = _extract_live_order_authority_ids_from_payload(payload)
        provider_order_id = str(
            getattr(row, "provider_order_id", None)
            or extract_trader_order_provider_order_id(payload)
            or ""
        ).strip()
        if provider_clob_id:
            provider_clob_id = provider_clob_id.lower()
            existing_provider_rows_by_clob_id[provider_clob_id] = row
        if live_order_id:
            existing_rows_by_live_order_id[live_order_id.lower()] = row
        if provider_order_id:
            existing_rows_by_live_order_id[provider_order_id.lower()] = row

    recovered_rows: list[TraderOrder] = []
    hot_state_sync_rows: dict[str, TraderOrder] = {}
    ambiguous_orders = 0
    adopted_existing_orders = 0
    adopted_existing_order_ids: set[str] = set()
    candidate_cache: dict[tuple[str, int, tuple[str, ...] | None], dict[str, Any] | None] = {}

    for live_row in live_rows:
        clob_order_id = str(live_row.clob_order_id or "").strip().lower()
        live_order_id = str(live_row.id or "").strip().lower()
        live_side = str(live_row.side or "").strip().upper()
        live_authority_keys = [
            key
            for key in (
                f"clob:{clob_order_id}" if clob_order_id else "",
                f"live:{live_order_id}" if live_order_id else "",
                f"provider:{live_order_id}" if live_order_id else "",
            )
            if key
        ]
        global_authority_row = None
        if trader_id_filter:
            for live_authority_key in live_authority_keys:
                global_authority_row = global_authority_rows_by_key.get(live_authority_key)
                if global_authority_row is not None:
                    break
            if global_authority_row is not None:
                global_authority_trader_id = str(global_authority_row.trader_id or "").strip()
                if global_authority_trader_id and global_authority_trader_id not in trader_id_filter:
                    adopted_existing_orders += 1
                    affected_traders.add(global_authority_trader_id)
                    continue

        token_key = _wallet_position_token_key(live_row.token_id)
        position_row = live_orders_by_token.get(token_key)
        market_question = str(live_row.market_question or (position_row.market_question if position_row is not None else "") or "").strip()
        created_at = live_row.created_at or live_row.updated_at or now
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = created_at.astimezone(timezone.utc)

        recovery_state = _build_live_order_authority_payload(
            live_row=live_row,
            position_row=position_row,
            now=now,
        )
        recovered_payload_json = dict(recovery_state["payload_json"] or {})
        recovered_filled_notional_usd, recovered_filled_shares, recovered_fill_price = _extract_live_fill_metrics(
            recovered_payload_json
        )
        recovered_entry_price = safe_float(recovery_state.get("entry_price"), recovered_fill_price)
        recovered_notional = max(0.0, safe_float(recovery_state.get("order_notional_usd"), 0.0) or 0.0)
        if recovered_notional <= 0.0:
            recovered_notional = max(0.0, recovered_filled_notional_usd)
        if live_side == "SELL":
            pending_exit_parent = None
            if clob_order_id:
                pending_exit_parent = pending_exit_rows_by_clob_id.get(clob_order_id)
            if pending_exit_parent is None and live_order_id:
                pending_exit_parent = pending_exit_rows_by_live_order_id.get(live_order_id)
            if pending_exit_parent is not None:
                if _sync_pending_live_exit_from_live_order(
                    pending_exit_parent,
                    live_row=live_row,
                    now=now,
                ):
                    adopted_existing_orders += 1
                parent_trader_id = str(pending_exit_parent.trader_id or "").strip()
                if parent_trader_id:
                    affected_traders.add(parent_trader_id)
                for authority_key in tuple(
                    key
                    for key in (
                        f"clob:{clob_order_id}" if clob_order_id else "",
                        f"live:{live_order_id}" if live_order_id else "",
                    )
                    if key
                ):
                    duplicate_rows = recovered_sell_rows_by_authority_key.get(authority_key, [])
                    for duplicate_row in list(duplicate_rows):
                        if duplicate_row.id == pending_exit_parent.id:
                            continue
                        await _reassign_deleted_live_order_references(
                            session,
                            duplicate_order_id=str(duplicate_row.id or ""),
                            canonical_order_id=str(pending_exit_parent.id or ""),
                        )
                        await session.delete(duplicate_row)
                        collapsed_duplicates += 1
                    recovered_sell_rows_by_authority_key.pop(authority_key, None)
                continue
            ambiguous_orders += 1
            continue
        existing_row = None
        matched_pre_submit_placeholder = False
        if clob_order_id:
            existing_row = existing_provider_rows_by_clob_id.get(clob_order_id)
        if existing_row is None and live_order_id:
            existing_row = existing_rows_by_live_order_id.get(live_order_id)
        if existing_row is None and global_authority_row is not None:
            global_authority_trader_id = str(global_authority_row.trader_id or "").strip()
            if not trader_id_filter or global_authority_trader_id in trader_id_filter:
                existing_row = global_authority_row
        if existing_row is None and token_key:
            live_market_id = str(getattr(position_row, "market_id", None) or "").strip()
            live_condition_id = _normalize_condition_id(live_market_id)
            placeholder_candidates: list[tuple[float, float, TraderOrder]] = []
            for placeholder_row in pre_submit_rows_by_token_key.get(token_key, []):
                placeholder_payload = dict(placeholder_row.payload_json or {})
                submission_intent = placeholder_payload.get("submission_intent")
                if not isinstance(submission_intent, dict):
                    continue
                if _normalize_status_key(submission_intent.get("state")) not in {
                    "pre_submit_persisted",
                    "submit_timeout",
                }:
                    continue
                if _live_order_authority_key_from_row(placeholder_row):
                    continue
                placeholder_market_id = str(placeholder_row.market_id or placeholder_payload.get("market_id") or "").strip()
                placeholder_condition_id = _normalize_condition_id(placeholder_market_id)
                if live_market_id and placeholder_market_id and placeholder_market_id != live_market_id:
                    # Token identity is stricter than market_id formatting; allow
                    # numeric trader market ids to match condition-id wallet rows.
                    if placeholder_condition_id and live_condition_id and placeholder_condition_id != live_condition_id:
                        continue
                placeholder_created_at = placeholder_row.created_at or placeholder_row.updated_at or now
                if placeholder_created_at.tzinfo is None:
                    placeholder_created_at = placeholder_created_at.replace(tzinfo=timezone.utc)
                else:
                    placeholder_created_at = placeholder_created_at.astimezone(timezone.utc)
                placeholder_candidates.append(
                    (
                        abs((placeholder_created_at - created_at).total_seconds()),
                        -placeholder_created_at.timestamp(),
                        placeholder_row,
                    )
                )
            if placeholder_candidates:
                placeholder_candidates.sort(key=lambda item: (item[0], item[1], str(item[2].id or "")))
                if len(placeholder_candidates) == 1 or placeholder_candidates[0][0] < placeholder_candidates[1][0]:
                    existing_row = placeholder_candidates[0][2]
                    matched_pre_submit_placeholder = True
        if existing_row is not None:
            existing_row_id = str(existing_row.id or "").strip()
            existing_payload = dict(existing_row.payload_json or {})
            merged_payload, payload_changed = _merge_missing_mapping_values(
                existing_payload,
                recovered_payload_json,
            )
            recovered_provider_reconciliation = recovered_payload_json.get("provider_reconciliation")
            if isinstance(recovered_provider_reconciliation, dict):
                existing_provider_reconciliation = dict(merged_payload.get("provider_reconciliation") or {})
                next_provider_reconciliation = dict(existing_provider_reconciliation)
                next_provider_reconciliation.update(copy.deepcopy(recovered_provider_reconciliation))
                existing_provider_material = dict(existing_provider_reconciliation)
                next_provider_material = dict(next_provider_reconciliation)
                existing_provider_material.pop("reconciled_at", None)
                next_provider_material.pop("reconciled_at", None)
                if existing_provider_material != next_provider_material:
                    merged_payload["provider_reconciliation"] = next_provider_reconciliation
                    payload_changed = True
            recovered_live_wallet_authority = recovered_payload_json.get("live_wallet_authority")
            if isinstance(recovered_live_wallet_authority, dict):
                existing_live_wallet_authority = dict(merged_payload.get("live_wallet_authority") or {})
                next_live_wallet_authority = dict(existing_live_wallet_authority)
                next_live_wallet_authority.update(copy.deepcopy(recovered_live_wallet_authority))
                existing_authority_material = dict(existing_live_wallet_authority)
                next_authority_material = dict(next_live_wallet_authority)
                existing_authority_material.pop("recovered_at", None)
                next_authority_material.pop("recovered_at", None)
                if existing_authority_material != next_authority_material:
                    merged_payload["live_wallet_authority"] = next_live_wallet_authority
                    payload_changed = True
            if matched_pre_submit_placeholder:
                submission_intent = dict(merged_payload.get("submission_intent") or {})
                submission_intent["state"] = "recovered_from_live_authority"
                submission_intent["authority_recovered_at"] = to_iso(now)
                if live_order_id:
                    submission_intent["live_trading_order_id"] = live_order_id
                if clob_order_id:
                    submission_intent["provider_clob_order_id"] = clob_order_id
                merged_payload["submission_intent"] = submission_intent
                payload_changed = True
            recovery_strategy_payload = _live_order_authority_strategy_payload(
                trader_source_configs=trader_source_configs_by_id.get(str(existing_row.trader_id or "").strip()),
                source_key=str(existing_row.source or "").strip(),
                strategy_key=str(existing_row.strategy_key or "").strip(),
            )
            if recovery_strategy_payload:
                merged_payload, strategy_payload_changed = _merge_missing_mapping_values(
                    merged_payload,
                    recovery_strategy_payload,
                )
                payload_changed = payload_changed or strategy_payload_changed
            recovered_verification = derive_trader_order_verification(
                mode="live",
                status=str(recovery_state["trader_status"]),
                reason="Recovered from live venue authority",
                payload=merged_payload,
            )
            existing_status = _normalize_status_key(existing_row.status)
            recovered_status = _normalize_status_key(recovery_state["trader_status"])
            existing_status_is_terminal = (
                existing_status in RESOLVED_ORDER_STATUSES
                or existing_status in REALIZED_ORDER_STATUSES
                or existing_status.startswith("closed")
            )
            status_changed = False
            existing_position_close = merged_payload.get("position_close")
            # If polymarket_trade_verifier wrote a verified_close
            # block (matched to on-chain settlement or trade), it is
            # the authoritative actual_profit. Don't recompute from
            # the stale position_close payload, which still carries
            # whatever wallet_summary_recovery / inferred close wrote
            # before the verifier ran. (This was the path that kept
            # snapping Flash Crash from -$2.31 back to +$32.82.)
            _verified_close = merged_payload.get("verified_close") if isinstance(merged_payload, dict) else None
            _has_verified_close = (
                isinstance(_verified_close, dict)
                and str(getattr(existing_row, "verification_status", "") or "").strip().lower() == "wallet_activity"
            )
            if isinstance(existing_position_close, dict) and _position_close_has_live_terminal_authority(
                existing_position_close
            ):
                prior_existing_status = existing_status
                terminal_status, terminal_pnl = _terminal_status_for_position_close(
                    existing_position_close,
                    existing_row.actual_profit,
                )
                if existing_status != terminal_status:
                    existing_row.status = terminal_status
                    existing_status = terminal_status
                    status_changed = True
                # Only overwrite actual_profit from the position_close
                # payload if no Polymarket-verified close exists. The
                # verifier owns the value once it has stamped one.
                if not _has_verified_close:
                    if safe_float(existing_row.actual_profit, None) != terminal_pnl:
                        existing_row.actual_profit = float(terminal_pnl)
                        status_changed = True
                existing_status_is_terminal = True
                if not str(existing_position_close.get("closed_at") or "").strip():
                    existing_position_close["closed_at"] = to_iso(now)
                    payload_changed = True
                merged_payload["position_close"] = existing_position_close
                repair_reason = "live_authority_recovery_preserved_terminal_position_close"
                repair_state = merged_payload.get("position_close_status_repair")
                if (
                    status_changed
                    or not isinstance(repair_state, dict)
                    or str(repair_state.get("reason") or "").strip() != repair_reason
                ):
                    merged_payload["position_close_status_repair"] = {
                        "repaired_at": to_iso(now),
                        "prior_status": prior_existing_status,
                        "reason": repair_reason,
                    }
                    payload_changed = True
            if (
                str(live_row.side or "").strip().upper() != "SELL"
                and recovered_status in LIVE_ACTIVE_ORDER_STATUSES
                and not existing_status_is_terminal
                and existing_status in {"placing", "submitted", "executed", "completed", "open", "failed", "cancelled", "rejected", "error"}
                and existing_status != recovered_status
            ):
                existing_row.status = recovered_status
                existing_row.actual_profit = None
                existing_row.error_message = None
                if recovered_status in {"executed", "completed", "open"} and existing_row.executed_at is None:
                    existing_row.executed_at = live_row.updated_at or live_row.created_at or now
                status_changed = True
            elif (
                str(live_row.side or "").strip().upper() != "SELL"
                and not existing_status_is_terminal
                and _should_adopt_terminal_live_order_status(
                    existing_status=existing_status,
                    recovered_status=recovered_status,
                    recovery_state=recovery_state,
                    merged_payload=merged_payload,
                )
            ):
                terminal_status = _terminal_live_order_recovery_status(
                    recovered_status=recovered_status,
                    live_row=live_row,
                )
                if existing_status != terminal_status:
                    existing_row.status = terminal_status
                    status_changed = True
                existing_row.notional_usd = 0.0
                existing_row.actual_profit = None
                existing_row.error_message = str(live_row.error_message or "").strip() or existing_row.error_message
                existing_row.executed_at = None
            # Don't downgrade rows polymarket_trade_verifier already
            # verified against on-chain truth. The live-authority
            # adoption path uses force=True which would otherwise
            # bypass the rank gate and clobber wallet_activity status
            # back to wallet_position (visible as a downgrade event
            # right after every closed_positions sweep).
            _existing_verification = str(getattr(existing_row, "verification_status", None) or "").strip().lower()
            _existing_status_lower = str(getattr(existing_row, "status", None) or "").strip().lower()
            _is_resolved_status_for_guard = _existing_status_lower in {
                "resolved", "resolved_win", "resolved_loss",
                "closed_win", "closed_loss", "win", "loss",
            }
            if _existing_verification == "wallet_activity" and _is_resolved_status_for_guard:
                verification_changed = False
            else:
                verification_changed = apply_trader_order_verification(
                    existing_row,
                    verification_status=str(recovered_verification.get("verification_status") or TRADER_ORDER_VERIFICATION_LOCAL),
                    verification_source=str(recovered_verification.get("verification_source") or "").strip() or None,
                    verification_reason=str(recovered_verification.get("verification_reason") or "").strip() or None,
                    provider_order_id=str(recovered_verification.get("provider_order_id") or "").strip() or None,
                    provider_clob_order_id=str(recovered_verification.get("provider_clob_order_id") or "").strip() or None,
                    execution_wallet_address=str(recovered_verification.get("execution_wallet_address") or "").strip() or None,
                    verification_tx_hash=str(recovered_verification.get("verification_tx_hash") or "").strip() or None,
                    verified_at=now,
                    force=True,
                )
            field_changed = False
            if recovered_entry_price is not None and recovered_entry_price > 0.0:
                if safe_float(existing_row.entry_price, 0.0) <= 0.0:
                    existing_row.entry_price = float(recovered_entry_price)
                    field_changed = True
                if safe_float(existing_row.effective_price, 0.0) <= 0.0:
                    existing_row.effective_price = float(recovered_entry_price)
                    field_changed = True
            if recovered_notional > 0.0:
                existing_notional = safe_float(existing_row.notional_usd, 0.0) or 0.0
                if abs(existing_notional - recovered_notional) > 1e-9:
                    existing_row.notional_usd = float(recovered_notional)
                    field_changed = True
            elif recovered_notional <= 0.0 and _normalize_status_key(existing_row.status) in {"cancelled", "failed", "rejected", "error"}:
                if safe_float(existing_row.notional_usd, 0.0) != 0.0:
                    existing_row.notional_usd = 0.0
                    field_changed = True
            if payload_changed:
                existing_row.payload_json = merged_payload
            execution_order = execution_order_by_trader_order.get(str(existing_row.id or "").strip())
            if execution_order is not None:
                execution_payload = dict(execution_order.payload_json or {})
                execution_payload, execution_payload_changed = _merge_missing_mapping_values(
                    execution_payload,
                    recovered_payload_json,
                )
                if matched_pre_submit_placeholder:
                    execution_submission_intent = dict(execution_payload.get("submission_intent") or {})
                    execution_submission_intent["state"] = "recovered_from_live_authority"
                    execution_submission_intent["authority_recovered_at"] = to_iso(now)
                    if live_order_id:
                        execution_submission_intent["live_trading_order_id"] = live_order_id
                    if clob_order_id:
                        execution_submission_intent["provider_clob_order_id"] = clob_order_id
                    execution_payload["submission_intent"] = execution_submission_intent
                    execution_payload_changed = True
                execution_order_changed = False
                if str(execution_order.status or "").strip().lower() != recovered_status:
                    execution_order.status = recovered_status or str(execution_order.status or "")
                    execution_order_changed = True
                if recovered_entry_price is not None and recovered_entry_price > 0.0:
                    existing_execution_price = safe_float(execution_order.price, None)
                    if existing_execution_price is None or abs(existing_execution_price - recovered_entry_price) > 1e-9:
                        execution_order.price = float(recovered_entry_price)
                        execution_order_changed = True
                if recovered_filled_shares > 0.0 and safe_float(execution_order.size, 0.0) != recovered_filled_shares:
                    execution_order.size = float(recovered_filled_shares)
                    execution_order_changed = True
                if recovered_notional > 0.0 and safe_float(execution_order.notional_usd, 0.0) != recovered_notional:
                    execution_order.notional_usd = float(recovered_notional)
                    execution_order_changed = True
                elif recovered_notional <= 0.0 and recovered_status in {"cancelled", "failed", "rejected", "error"}:
                    if safe_float(execution_order.notional_usd, 0.0) != 0.0:
                        execution_order.notional_usd = 0.0
                        execution_order_changed = True
                if existing_row.provider_order_id and str(execution_order.provider_order_id or "") != str(existing_row.provider_order_id):
                    execution_order.provider_order_id = str(existing_row.provider_order_id)
                    execution_order_changed = True
                if existing_row.provider_clob_order_id and str(execution_order.provider_clob_order_id or "") != str(existing_row.provider_clob_order_id):
                    execution_order.provider_clob_order_id = str(existing_row.provider_clob_order_id)
                    execution_order_changed = True
                next_execution_error = None if recovered_status in LIVE_ACTIVE_ORDER_STATUSES else str(live_row.error_message or "").strip() or execution_order.error_message
                if execution_order.error_message != next_execution_error:
                    execution_order.error_message = next_execution_error
                    execution_order_changed = True
                if execution_payload_changed:
                    execution_order.payload_json = execution_payload
                    execution_order_changed = True
                if execution_order_changed:
                    execution_order.updated_at = now
                    updated_execution_order_rows[str(execution_order.id)] = execution_order
                leg_id = str(execution_order.leg_id or "").strip()
                if leg_id:
                    leg_state_updates[leg_id] = {
                        "status": recovered_status,
                        "filled_notional_usd": float(recovered_notional),
                        "filled_shares": float(max(0.0, recovered_filled_shares)),
                        "avg_fill_price": float(recovered_entry_price) if recovered_entry_price is not None and recovered_entry_price > 0.0 else None,
                        "provider_order_id": existing_row.provider_order_id,
                        "provider_clob_order_id": existing_row.provider_clob_order_id,
                        "snapshot_status": _normalize_status_key(live_row.status),
                    }
                session_id = str(execution_order.session_id or "").strip()
                if session_id:
                    touched_session_ids.add(session_id)
            if payload_changed or status_changed or verification_changed or field_changed:
                existing_row.updated_at = now
                if existing_row_id not in adopted_existing_order_ids:
                    event_payload = dict(existing_row.payload_json or {})
                    event_payload.update(
                        {
                            "live_trading_order_id": live_order_id or None,
                            "live_trading_status": str(live_row.status or ""),
                            "adopted_status": str(existing_row.status or ""),
                        }
                    )
                    append_trader_order_verification_event(
                        session,
                        trader_order_id=str(existing_row.id),
                        verification_status=str(existing_row.verification_status or TRADER_ORDER_VERIFICATION_LOCAL),
                        source=existing_row.verification_source,
                        event_type="authority_adopted_existing",
                        reason=existing_row.verification_reason or "Recovered from live venue authority",
                        provider_order_id=existing_row.provider_order_id,
                        provider_clob_order_id=existing_row.provider_clob_order_id,
                        execution_wallet_address=existing_row.execution_wallet_address,
                        tx_hash=existing_row.verification_tx_hash,
                        payload_json=event_payload,
                        created_at=now,
                    )
                    adopted_existing_order_ids.add(existing_row_id)
                    adopted_existing_orders += 1
                hot_state_sync_rows[str(existing_row.id or "")] = existing_row
            if matched_pre_submit_placeholder and token_key:
                pre_submit_rows_by_token_key[token_key] = [
                    row for row in pre_submit_rows_by_token_key.get(token_key, []) if str(row.id or "") != str(existing_row.id or "")
                ]
                if not pre_submit_rows_by_token_key[token_key]:
                    pre_submit_rows_by_token_key.pop(token_key, None)
            if clob_order_id:
                existing_provider_rows_by_clob_id[clob_order_id] = existing_row
            if live_order_id:
                existing_rows_by_live_order_id[live_order_id] = existing_row
            continue

        live_wallet_size = max(0.0, safe_float(getattr(position_row, "size", None), 0.0) or 0.0)
        recovered_status = _normalize_status_key(recovery_state.get("trader_status"))
        if (
            live_side != "SELL"
            and recovered_status in LIVE_ACTIVE_ORDER_STATUSES
            and live_wallet_size <= 0.0
            and recovered_notional <= 0.0
        ):
            ambiguous_orders += 1
            continue

        age_seconds = (now - created_at).total_seconds() if created_at else 0
        if age_seconds < 90 and len(trader_id_filter) != 1:
            ambiguous_orders += 1
            continue

        candidate_cache_key = (
            market_question,
            int(created_at.timestamp()) // _LIVE_ORDER_AUTHORITY_RECOVERY_CANDIDATE_CACHE_BUCKET_SECONDS,
            tuple(sorted(trader_id_filter)) if trader_id_filter else None,
        )
        if candidate_cache_key in candidate_cache:
            candidate = candidate_cache[candidate_cache_key]
        else:
            candidate = await _infer_live_order_authority_candidate(
                session,
                market_question=market_question,
                created_at=created_at,
                trader_ids=trader_id_filter or None,
            )
            candidate_cache[candidate_cache_key] = candidate
        if candidate is None:
            ambiguous_orders += 1
            continue

        candidate_trader_id = str(candidate["trader_id"]).strip()
        candidate_market_id = str(candidate["market_id"]).strip()
        if (
            (candidate_trader_id, candidate_market_id) in existing_occupied_markets
            or (candidate_trader_id, candidate_market_id) in recently_closed_recovery_guard_markets
        ):
            ambiguous_orders += 1
            continue

        recovered_payload = dict(recovery_state["payload_json"] or {})
        candidate_signal_payload = dict(candidate.get("signal_payload") or {})
        recovered_payload, _ = _merge_missing_mapping_values(recovered_payload, candidate_signal_payload)
        recovery_strategy_payload = _live_order_authority_strategy_payload(
            trader_source_configs=candidate.get("trader_source_configs"),
            source_key=str(candidate.get("source") or "").strip(),
            strategy_key=str(candidate.get("strategy_key") or "").strip(),
        )
        if recovery_strategy_payload:
            recovered_payload, _ = _merge_missing_mapping_values(recovered_payload, recovery_strategy_payload)
        recovered_verification = derive_trader_order_verification(
            mode="live",
            status=str(recovery_state["trader_status"]),
            reason="Recovered from live venue authority",
            payload=recovered_payload,
        )
        recovered_row = TraderOrder(
            id=_new_id(),
            trader_id=str(candidate["trader_id"]),
            signal_id=candidate.get("signal_id"),
            decision_id=candidate.get("decision_id"),
            source=str(candidate.get("source") or "scanner"),
            strategy_key=candidate.get("strategy_key"),
            strategy_version=candidate.get("strategy_version"),
            market_id=str(candidate["market_id"]),
            market_question=market_question or None,
            direction=str(candidate["direction"]),
            event_id=None,
            trace_id=None,
            mode="live",
            status=str(recovery_state["trader_status"]),
            notional_usd=float(recovered_notional) if recovered_notional > 0.0 else None,
            entry_price=float(recovered_entry_price) if recovered_entry_price is not None and recovered_entry_price > 0.0 else None,
            effective_price=float(recovered_entry_price) if recovered_entry_price is not None and recovered_entry_price > 0.0 else None,
            execution_wallet_address=recovered_verification.get("execution_wallet_address"),
            provider_order_id=recovered_verification.get("provider_order_id"),
            provider_clob_order_id=recovered_verification.get("provider_clob_order_id"),
            verification_status=str(recovered_verification.get("verification_status") or TRADER_ORDER_VERIFICATION_LOCAL),
            verification_source=recovered_verification.get("verification_source"),
            verification_reason=recovered_verification.get("verification_reason"),
            verification_tx_hash=recovered_verification.get("verification_tx_hash"),
            verified_at=now,
            edge_percent=None,
            confidence=None,
            actual_profit=None,
            reason=str(candidate.get("decision_reason") or "").strip() or "Recovered from live venue authority",
            payload_json=recovered_payload,
            error_message=live_row.error_message,
            created_at=created_at,
            executed_at=(
                live_row.updated_at or live_row.created_at or now
                if recovered_notional > 0.0
                or str(recovery_state["trader_status"]) == "executed"
                else None
            ),
            updated_at=live_row.updated_at or now,
        )
        session.add(recovered_row)
        event_payload = dict(recovered_row.payload_json or {})
        event_payload.update(
            {
                "market_question": market_question,
                "candidate_signal_id": candidate.get("signal_id"),
                "candidate_decision_id": candidate.get("decision_id"),
            }
        )
        append_trader_order_verification_event(
            session,
            trader_order_id=str(recovered_row.id),
            verification_status=str(recovered_row.verification_status or TRADER_ORDER_VERIFICATION_LOCAL),
            source=recovered_row.verification_source,
            event_type="authority_recovered",
            reason=recovered_row.verification_reason or "Recovered from live venue authority",
            provider_order_id=recovered_row.provider_order_id,
            provider_clob_order_id=recovered_row.provider_clob_order_id,
            execution_wallet_address=recovered_row.execution_wallet_address,
            tx_hash=recovered_row.verification_tx_hash,
            payload_json=event_payload,
            created_at=now,
        )
        recovered_rows.append(recovered_row)
        hot_state_sync_rows[str(recovered_row.id or "")] = recovered_row
        affected_traders.add(str(candidate["trader_id"]))
        if clob_order_id:
            existing_provider_rows_by_clob_id[clob_order_id] = recovered_row
        if live_order_id:
            existing_rows_by_live_order_id[live_order_id] = recovered_row

    if leg_state_updates:
        leg_rows = list(
            (
                await session.execute(
                    select(ExecutionSessionLeg).where(ExecutionSessionLeg.id.in_(list(leg_state_updates.keys())))
                )
            )
            .scalars()
            .all()
        )
        for leg_row in leg_rows:
            leg_update = leg_state_updates.get(str(leg_row.id))
            if leg_update is None:
                continue
            mapped_status = str(leg_update["status"] or "").strip().lower()
            leg_row.status = _map_live_order_status_to_leg_status(mapped_status)
            leg_row.filled_notional_usd = float(max(0.0, safe_float(leg_update.get("filled_notional_usd"), 0.0) or 0.0))
            leg_row.filled_shares = float(max(0.0, safe_float(leg_update.get("filled_shares"), 0.0) or 0.0))
            avg_fill_price = safe_float(leg_update.get("avg_fill_price"))
            if avg_fill_price is not None and avg_fill_price > 0:
                leg_row.avg_fill_price = float(avg_fill_price)
            if mapped_status in {"failed", "cancelled"}:
                leg_row.last_error = f"authority_recovery:{mapped_status}"
            else:
                leg_row.last_error = None
            metadata_json = dict(leg_row.metadata_json or {})
            metadata_json["provider_recovery"] = {
                "recovered_at": to_iso(now),
                "provider_order_id": leg_update.get("provider_order_id"),
                "provider_clob_order_id": leg_update.get("provider_clob_order_id"),
                "snapshot_status": leg_update.get("snapshot_status"),
                "mapped_status": mapped_status,
            }
            leg_row.metadata_json = metadata_json
            leg_row.updated_at = now
            updated_execution_leg_rows[str(leg_row.id)] = leg_row
            touched_session_ids.add(str(leg_row.session_id))

    if touched_session_ids:
        await session.flush()
    for session_id in touched_session_ids:
        session_row = await _refresh_execution_session_rollups(session, session_id=session_id)
        if session_row is None:
            continue
        previous_session_status = _normalize_status_key(session_row.status)
        next_session_status = previous_session_status
        legs_open = int(session_row.legs_open or 0)
        legs_completed = int(session_row.legs_completed or 0)
        legs_failed = int(session_row.legs_failed or 0)
        if legs_open <= 0:
            if legs_completed > 0 and legs_failed == 0:
                next_session_status = "completed"
                session_row.error_message = None
            elif legs_completed == 0 and legs_failed > 0:
                next_session_status = "failed"
                if not session_row.error_message:
                    session_row.error_message = "All execution legs failed during live authority recovery."
            elif legs_completed > 0 and legs_failed > 0:
                next_session_status = "failed"
                if not session_row.error_message:
                    session_row.error_message = "Execution session closed with partial fills and failed legs during live authority recovery."
        elif previous_session_status in {"pending", "placing", "partial", "hedging"}:
            next_session_status = "working"
        if next_session_status in TERMINAL_EXECUTION_SESSION_STATUSES and session_row.completed_at is None:
            session_row.completed_at = now
        session_row.status = next_session_status
        session_row.updated_at = now
        updated_execution_session_rows[str(session_row.id)] = session_row

    if (
        not recovered_rows
        and collapsed_duplicates <= 0
        and adopted_existing_orders <= 0
        and not updated_execution_order_rows
        and not updated_execution_leg_rows
        and not updated_execution_session_rows
    ):
        if commit and session.in_transaction():
            await session.rollback()
        return {
            "recovered_orders": 0,
            "affected_traders": 0,
            "ambiguous_orders": ambiguous_orders,
            "collapsed_duplicates": 0,
            "adopted_existing_orders": 0,
        }

    await session.flush()
    for trader_id in sorted(affected_traders):
        trader_recovered_order_ids = [row.id for row in recovered_rows if row.trader_id == trader_id]
        if trader_recovered_order_ids:
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="live_order_authority_recovered",
                severity="info",
                source="live_wallet_authority",
                message="Recovered missing live venue orders into trader order authority.",
                payload={
                    "recovered_order_ids": trader_recovered_order_ids,
                },
                commit=False,
            )
        await sync_trader_position_inventory(
            session,
            trader_id=trader_id,
            mode="live",
            commit=False,
        )

    if commit:
        await _commit_with_retry(session)
    else:
        await session.flush()

    # Notify hot state so occupancy and cooldown reflect authority reconciliation immediately.
    try:
        for row in hot_state_sync_rows.values():
            _sync_recovered_order_hot_state(row)
    except Exception:
        pass

    if broadcast:
        published_order_ids: set[str] = set()
        for row in recovered_rows:
            try:
                await event_bus.publish("trader_order", _serialize_order(row))
                published_order_ids.add(str(row.id or ""))
            except Exception:
                continue
        for row in hot_state_sync_rows.values():
            row_id = str(row.id or "")
            if not row_id or row_id in published_order_ids:
                continue
            try:
                await event_bus.publish("trader_order", _serialize_order(row))
            except Exception:
                continue
        for row in updated_execution_order_rows.values():
            try:
                await event_bus.publish("execution_order", _serialize_execution_order(row))
            except Exception:
                continue
        for row in updated_execution_leg_rows.values():
            try:
                await event_bus.publish("execution_leg", _serialize_execution_leg(row))
            except Exception:
                continue
        for row in updated_execution_session_rows.values():
            try:
                await event_bus.publish("execution_session", _serialize_execution_session(row))
            except Exception:
                continue

    return {
        "recovered_orders": len(recovered_rows),
        "affected_traders": len(affected_traders),
        "ambiguous_orders": ambiguous_orders,
        "collapsed_duplicates": collapsed_duplicates,
        "adopted_existing_orders": adopted_existing_orders,
    }


def _map_live_order_status_to_leg_status(order_status: str) -> str:
    status_key = _normalize_status_key(order_status)
    if status_key == "executed":
        return "completed"
    if status_key in {"open", "submitted"}:
        return "open"
    if status_key in {"cancelled", "failed", "expired"}:
        return "failed"
    return "open"


def _is_terminal_order_status(status: Any) -> bool:
    return _normalize_status_key(status) in {
        "cancelled",
        "failed",
        "expired",
        "resolved_win",
        "resolved_loss",
        "closed_win",
        "closed_loss",
        "win",
        "loss",
    }


def _sync_order_runtime_payload(
    *,
    payload: dict[str, Any],
    status: Any,
    now: datetime,
    provider_snapshot_status: Optional[str] = None,
    mapped_status: Optional[str] = None,
) -> dict[str, Any]:
    status_key = _normalize_status_key(status)
    if status_key:
        payload["trading_status"] = status_key

    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    provider_snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(provider_snapshot, dict):
        provider_snapshot = {}

    snapshot_status_key = _normalize_status_key(
        provider_snapshot_status
        or provider_reconciliation.get("snapshot_status")
        or provider_snapshot.get("normalized_status")
        or provider_snapshot.get("status")
    )
    if snapshot_status_key:
        provider_reconciliation["snapshot_status"] = snapshot_status_key

    mapped_status_key = _normalize_status_key(mapped_status or provider_reconciliation.get("mapped_status"))
    if mapped_status_key:
        provider_reconciliation["mapped_status"] = mapped_status_key

    if _is_terminal_order_status(status_key):
        if not snapshot_status_key:
            snapshot_status_key = mapped_status_key or status_key
            provider_reconciliation["snapshot_status"] = snapshot_status_key
        if not mapped_status_key:
            mapped_status_key = "cancelled" if status_key in {"cancelled", "failed", "expired"} else "executed"
            provider_reconciliation["mapped_status"] = mapped_status_key
        provider_reconciliation["reconciled_at"] = to_iso(now)

    if snapshot_status_key:
        provider_snapshot["normalized_status"] = snapshot_status_key
        provider_reconciliation["snapshot"] = provider_snapshot
    elif mapped_status_key:
        provider_snapshot["normalized_status"] = mapped_status_key
        provider_reconciliation["snapshot"] = provider_snapshot

    if provider_reconciliation:
        payload["provider_reconciliation"] = provider_reconciliation

    return payload


def _active_order_notional_for_metrics_fields(
    mode: Any,
    status: Any,
    notional_usd: Any,
    payload_json: Any,
) -> float:
    mode_key = _normalize_mode_key(mode)
    if not _is_active_order_status(mode_key, status):
        return 0.0
    payload = dict(payload_json or {})
    row_notional = safe_float(notional_usd, 0.0) or 0.0
    return _live_active_notional(mode_key, status, row_notional, payload)


def _active_order_notional_for_metrics(order: TraderOrder) -> float:
    return _active_order_notional_for_metrics_fields(
        mode=order.mode,
        status=order.status,
        notional_usd=order.notional_usd,
        payload_json=order.payload_json,
    )


def _normalize_confidence_fraction(value: Any, default: float = 0.0) -> float:
    parsed = safe_float(value, default)
    if parsed < 0.0:
        parsed = 0.0
    elif parsed <= 1.0:
        # Already a 0-1 fraction; use as-is.
        pass
    elif parsed <= 100.0:
        # Percentage scale (e.g. 75 -> 0.75).
        parsed = parsed / 100.0
    else:
        # Values above 100 are nonsensical; clamp to 1.0.
        parsed = 1.0
    return parsed


def _normalize_resume_policy(value: Any) -> str:
    return StrategySDK.validate_trader_runtime_metadata({"resume_policy": value}).get("resume_policy", "resume_full")


def _normalize_strategy_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _canonical_strategy_key(value: Any) -> str:
    key = _normalize_strategy_key(value)
    return _LEGACY_STRATEGY_ALIASES.get(key, key)


def _normalize_trader_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if not mode:
        return "shadow"
    if mode not in _TRADER_MODES:
        raise ValueError("mode must be 'shadow' or 'live'")
    return mode


def _normalize_trader_latency_class(value: Any) -> str:
    """Coerce a trader latency_class input to the canonical set.

    Defaults to ``normal`` when absent so pre-existing traders keep their
    current execution path unchanged.  Raises ValueError on unknown values.
    """
    cls = str(value or "").strip().lower()
    if not cls:
        return "normal"
    if cls not in _LATENCY_CLASSES:
        raise ValueError("latency_class must be 'fast', 'normal', or 'slow'")
    return cls


def _normalize_strategy_for_source(source_key: str, strategy_key: str) -> str:
    del source_key
    return _canonical_strategy_key(strategy_key)


def _normalize_timeframe_scope(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if token in {"5m", "5min", "5"}:
            token = "5m"
        elif token in {"15m", "15min", "15"}:
            token = "15m"
        elif token in {"1h", "1hr", "60m", "60min"}:
            token = "1h"
        elif token in {"4h", "4hr", "240m", "240min"}:
            token = "4h"
        else:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


async def _fetch_enabled_strategy_catalog(
    session: AsyncSession,
) -> tuple[set[str], dict[str, set[str]]]:
    rows = list(
        (
            await session.execute(
                select(
                    Strategy.slug,
                    Strategy.source_key,
                ).where(Strategy.enabled == True)  # noqa: E712
            )
        ).all()
    )
    if not rows:
        fallback_rows = build_system_opportunity_strategy_rows()
        valid_keys: set[str] = set()
        by_source: dict[str, set[str]] = {}
        for row in fallback_rows:
            slug = str(row.get("slug") or "").strip().lower()
            src = str(row.get("source_key") or "").strip().lower()
            if not slug or not src:
                continue
            valid_keys.add(slug)
            by_source.setdefault(src, set()).add(slug)
        return valid_keys, by_source

    valid_keys: set[str] = set()
    by_source: dict[str, set[str]] = {}
    for strategy_key, source_key in rows:
        skey = str(strategy_key or "").strip().lower()
        src = str(source_key or "").strip().lower()
        if not skey or not src:
            continue
        valid_keys.add(skey)
        by_source.setdefault(src, set()).add(skey)
    return valid_keys, by_source


async def _validate_strategy_key(session: AsyncSession, strategy_key: str) -> None:
    valid_keys, _ = await _fetch_enabled_strategy_catalog(session)
    canonical_strategy_key = _canonical_strategy_key(strategy_key)
    if not canonical_strategy_key:
        raise ValueError("strategy_key is required")
    if canonical_strategy_key not in valid_keys:
        allowed = ", ".join(sorted(valid_keys))
        raise ValueError(f"Unknown strategy_key '{strategy_key}'. Valid strategy keys: {allowed}")


def _normalize_strategy_params(value: Any, source_key: str, strategy_key: str = "") -> dict[str, Any]:
    params = value if isinstance(value, dict) else {}
    out = dict(params)
    normalized_source = str(source_key or "").strip().lower()
    normalized_strategy_key = str(strategy_key or "").strip().lower()
    if normalized_source == "traders":
        if normalized_strategy_key == "traders_copy_trade":
            return validate_traders_copy_trade_config(out)
        return StrategySDK.validate_trader_filter_config(out)
    if normalized_source == "news":
        return validate_news_edge_config(out)
    # Crypto strategies own their own defaults inline (see each strategy's
    # default_config) — no shared merge needed at this layer; configure()
    # in the strategy class handles overlay against its own defaults.
    if "min_confidence" in out:
        out["min_confidence"] = _normalize_confidence_fraction(out.get("min_confidence"), 0.0)
    return StrategySDK.normalize_strategy_retention_config(out)


def _validate_traders_scope(scope: dict[str, Any]) -> None:
    modes = list(scope.get("modes") or [])
    if not modes:
        raise ValueError("traders_scope.modes must include at least one mode")
    invalid = [mode for mode in modes if mode not in _TRADER_SCOPE_MODES]
    if invalid:
        raise ValueError(
            f"Invalid traders_scope.modes: {', '.join(invalid)}. Allowed: {', '.join(sorted(_TRADER_SCOPE_MODES))}"
        )
    if "individual" in modes and not list(scope.get("individual_wallets") or []):
        raise ValueError("traders_scope.individual_wallets is required when mode 'individual' is selected")
    if "group" in modes and not list(scope.get("group_ids") or []):
        raise ValueError("traders_scope.group_ids is required when mode 'group' is selected")


def _normalize_source_config(raw: Any) -> dict[str, Any]:
    item = raw if isinstance(raw, dict) else {}
    source_key = normalize_source_key(item.get("source_key"))
    requested_strategy_key = _normalize_strategy_key(item.get("strategy_key"))
    raw_strategy_params = dict(item.get("strategy_params") or {})
    strategy_key = _normalize_strategy_for_source(
        source_key,
        requested_strategy_key,
    )
    if requested_strategy_key and requested_strategy_key != strategy_key:
        accepted = raw_strategy_params.get("accepted_signal_strategy_types")
        if isinstance(accepted, list):
            accepted_list = [str(item).strip().lower() for item in accepted if str(item or "").strip()]
        elif isinstance(accepted, str):
            accepted_list = [part.strip().lower() for part in accepted.split(",") if part.strip()]
        else:
            accepted_list = []
        if requested_strategy_key not in accepted_list:
            accepted_list.append(requested_strategy_key)
        raw_strategy_params["accepted_signal_strategy_types"] = accepted_list
    strategy_version = normalize_strategy_version(item.get("strategy_version"))
    strategy_params = _normalize_strategy_params(raw_strategy_params, source_key, strategy_key)
    normalized: dict[str, Any] = {
        "source_key": source_key,
        "strategy_key": strategy_key,
        "strategy_version": strategy_version,
        "strategy_params": strategy_params,
    }
    return normalized


def _normalize_source_configs(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for raw_item in items:
        source_config = _normalize_source_config(raw_item)
        source_key = source_config.get("source_key", "")
        if not source_key:
            continue
        if source_key in seen_sources:
            raise ValueError(f"Duplicate source_key '{source_key}' in source_configs")
        seen_sources.add(source_key)
        out.append(source_config)
    return out


async def _validate_source_strategy_pair(
    session: AsyncSession,
    source_key: str,
    strategy_key: str,
    strategy_version: int | None = None,
) -> None:
    _, by_source = await _fetch_enabled_strategy_catalog(session)
    valid_strategies = by_source.get(source_key)
    if not valid_strategies:
        allowed_sources = ", ".join(sorted(by_source.keys()))
        raise ValueError(f"Unknown source_key '{source_key}'. Allowed source keys: {allowed_sources}")
    await _validate_strategy_key(session, strategy_key)
    if strategy_key not in valid_strategies:
        allowed = ", ".join(sorted(valid_strategies))
        raise ValueError(
            f"Invalid strategy_key '{strategy_key}' for source_key '{source_key}'. Allowed strategies: {allowed}"
        )
    if strategy_version is not None:
        try:
            await resolve_strategy_version(
                session,
                strategy_key=strategy_key,
                requested_version=int(strategy_version),
            )
        except ValueError as exc:
            raise ValueError(str(exc))


async def _validate_source_configs(
    session: AsyncSession,
    source_configs: list[dict[str, Any]],
) -> None:
    if not source_configs:
        raise ValueError("source_configs must include at least one source")
    if len(source_configs) > 1:
        # A trader runs exactly one strategy on one source. Reject anything else.
        raise ValueError(
            "source_configs must contain exactly one entry "
            "(a trader runs one strategy on one source)"
        )
    for source_config in source_configs:
        source_key = str(source_config.get("source_key") or "").strip().lower()
        strategy_key = _normalize_strategy_key(source_config.get("strategy_key"))
        strategy_version = normalize_strategy_version(source_config.get("strategy_version"))
        await _validate_source_strategy_pair(
            session,
            source_key,
            strategy_key,
            strategy_version=strategy_version,
        )
        if source_key == "traders":
            strategy_params = dict(source_config.get("strategy_params") or {})
            _validate_traders_scope(StrategySDK.validate_trader_scope_config(strategy_params.get("traders_scope")))


def _default_control_settings() -> dict[str, Any]:
    return _normalize_control_settings({})


def _serialize_control(row: TraderOrchestratorControl) -> dict[str, Any]:
    return {
        "id": row.id,
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "mode": _normalize_trader_mode(row.mode),
        "run_interval_seconds": int(row.run_interval_seconds or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
        "requested_run_at": to_iso(row.requested_run_at),
        "kill_switch": bool(row.kill_switch),
        "settings": _normalize_control_settings(row.settings_json),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_snapshot(row: TraderOrchestratorSnapshot) -> dict[str, Any]:
    interval_seconds = int(row.interval_seconds or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS)
    lag_seconds: Optional[float] = None
    if row.last_run_at is not None:
        try:
            lag_seconds = max(0.0, (_now() - row.last_run_at).total_seconds())
        except Exception:
            lag_seconds = None

    running = bool(row.running)
    if running and lag_seconds is not None:
        stale_after_seconds = max(
            _ORCHESTRATOR_SNAPSHOT_STALE_MIN_SECONDS,
            float(max(1, interval_seconds)) * _ORCHESTRATOR_SNAPSHOT_STALE_MULTIPLIER,
        )
        if lag_seconds > stale_after_seconds:
            running = False

    return {
        "id": row.id,
        "updated_at": to_iso(row.updated_at),
        "last_run_at": to_iso(row.last_run_at),
        "running": running,
        "enabled": bool(row.enabled),
        "current_activity": row.current_activity,
        "interval_seconds": interval_seconds,
        "traders_total": int(row.traders_total or 0),
        "traders_running": int(row.traders_running or 0),
        "decisions_count": int(row.decisions_count or 0),
        "orders_count": int(row.orders_count or 0),
        "open_orders": int(row.open_orders or 0),
        "gross_exposure_usd": float(row.gross_exposure_usd or 0.0),
        "daily_pnl": float(row.daily_pnl or 0.0),
        "last_error": row.last_error,
        "stats": row.stats_json or {},
    }


async def _apply_wallet_live_exposure_floor(session: AsyncSession, snapshot: dict[str, Any]) -> dict[str, Any]:
    try:
        wallet_live_exposure = await get_gross_exposure(session, mode="live")
    except Exception as exc:
        maybe_mark_db_pressure(exc, component="orchestrator_snapshot_exposure_floor")
        return snapshot
    current_exposure = safe_float(snapshot.get("gross_exposure_usd"), 0.0) or 0.0
    if wallet_live_exposure <= current_exposure:
        return snapshot

    patched = dict(snapshot)
    stats = dict(patched.get("stats") or {})
    patched["gross_exposure_usd"] = float(wallet_live_exposure)
    stats["gross_exposure_usd"] = float(wallet_live_exposure)
    stats["wallet_gross_exposure_floor_usd"] = float(wallet_live_exposure)
    patched["stats"] = stats
    return patched


def _serialize_trader(row: Trader) -> dict[str, Any]:
    metadata = dict(row.metadata_json or {})
    metadata["resume_policy"] = _normalize_resume_policy(metadata.get("resume_policy"))
    source_configs = _normalize_source_configs(row.source_configs_json)
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "mode": _normalize_trader_mode(row.mode),
        "latency_class": _normalize_trader_latency_class(getattr(row, "latency_class", None)),
        "source_configs": source_configs,
        "risk_limits": row.risk_limits_json or {},
        "metadata": metadata,
        "is_enabled": bool(row.is_enabled),
        "is_paused": bool(row.is_paused),
        "block_new_orders": bool(getattr(row, "block_new_orders", False)),
        "interval_seconds": int(row.interval_seconds or 60),
        "requested_run_at": to_iso(row.requested_run_at),
        "last_run_at": to_iso(row.last_run_at),
        "next_run_at": to_iso(row.next_run_at),
        "created_at": to_iso(row.created_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_decision(row: TraderDecision) -> dict[str, Any]:
    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "source": row.source,
        "strategy_key": row.strategy_key,
        "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
        "decision": row.decision,
        "reason": row.reason,
        "score": row.score,
        "event_id": row.event_id,
        "trace_id": row.trace_id,
        "checks_summary": row.checks_summary_json or {},
        "risk_snapshot": row.risk_snapshot_json or {},
        "payload": row.payload_json or {},
        "created_at": to_iso(row.created_at),
    }


def _serialize_decision_with_signal(
    row: TraderDecision,
    signal: TradeSignal | None,
) -> dict[str, Any]:
    serialized = _serialize_decision(row)

    signal_payload_full: dict[str, Any] = {}
    signal_payload: dict[str, Any] = {}
    signal_strategy_context: dict[str, Any] = {}
    market_id: str | None = None
    market_question: str | None = None
    direction: str | None = None
    market_price: float | None = None
    edge_percent: float | None = None
    confidence: float | None = None
    model_probability: float | None = None

    if signal is not None:
        market_id = str(signal.market_id or "").strip() or None
        market_question = str(signal.market_question or "").strip() or None
        direction = str(signal.direction or "").strip() or None
        market_price = signal.effective_price if signal.effective_price is not None else signal.entry_price
        edge_percent = signal.edge_percent
        confidence = signal.confidence
        if isinstance(signal.payload_json, dict):
            signal_payload_full = signal.payload_json
            for key in ("execution_plan", "markets", "positions_to_take"):
                value = signal_payload_full.get(key)
                if value is not None:
                    signal_payload[key] = value
        if isinstance(signal.strategy_context_json, dict):
            signal_strategy_context = signal.strategy_context_json

    ai_analysis = signal_payload_full.get("ai_analysis") if isinstance(signal_payload_full, dict) else None
    if isinstance(ai_analysis, dict):
        model_probability = safe_float(
            ai_analysis.get("model_probability")
            or ai_analysis.get("predicted_probability")
            or ai_analysis.get("estimated_probability")
            or ai_analysis.get("probability"),
            None,
        )
    if model_probability is None and isinstance(signal_payload_full, dict):
        model_probability = safe_float(
            signal_payload_full.get("model_probability")
            or signal_payload_full.get("predicted_probability")
            or signal_payload_full.get("probability"),
            None,
        )

    direction_side, direction_label, yes_label, no_label = _resolve_direction_presentation(
        direction=direction,
        payload=row.payload_json if isinstance(row.payload_json, dict) else {},
        signal_payload=signal_payload_full if isinstance(signal_payload_full, dict) else {},
        market_id=market_id,
    )

    serialized.update(
        {
            "market_id": market_id,
            "market_question": market_question,
            "direction": direction,
            "direction_side": direction_side,
            "direction_label": direction_label,
            "yes_label": yes_label,
            "no_label": no_label,
            "market_price": market_price,
            "model_probability": model_probability,
            "edge_percent": edge_percent,
            "confidence": confidence,
            "signal_score": row.score,
            "signal_payload": signal_payload,
            "signal_strategy_context": signal_strategy_context,
        }
    )
    return serialized


def _serialize_order(
    row: TraderOrder,
    signal: TradeSignal | None = None,
    cached_market: CachedMarket | None = None,
    sibling_rows: list[TraderOrder] | None = None,
) -> dict[str, Any]:
    payload = dict(row.payload_json or {})
    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    provider_snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(provider_snapshot, dict):
        provider_snapshot = {}
    position_state = payload.get("position_state")
    if not isinstance(position_state, dict):
        position_state = {}
    position_close = payload.get("position_close")
    if not isinstance(position_close, dict):
        position_close = {}
    pending_live_exit = payload.get("pending_live_exit")
    if not isinstance(pending_live_exit, dict):
        pending_live_exit = {}
    close_trigger = str(position_close.get("close_trigger") or pending_live_exit.get("close_trigger") or "").strip()
    close_reason = str(position_close.get("reason") or pending_live_exit.get("reason") or row.reason or "").strip()
    filled_notional_usd, filled_shares, average_fill_price = _extract_live_fill_metrics(payload)
    price_anchor = average_fill_price
    if price_anchor is None or price_anchor <= 0:
        price_anchor = safe_float(row.effective_price) or safe_float(row.entry_price)
    current_price = safe_float(position_state.get("last_mark_price"))
    if current_price is None or current_price <= 0:
        current_price = safe_float(
            payload.get("market_price")
            or payload.get("resolved_price")
            or provider_snapshot.get("price")
            or provider_snapshot.get("limit_price")
        )
    filled_notional_display = filled_notional_usd
    if filled_notional_display <= 0:
        filled_notional_display = max(0.0, safe_float(row.notional_usd, 0.0) or 0.0)
    quantity = filled_shares
    if quantity <= 0 and price_anchor is not None and price_anchor > 0 and filled_notional_display > 0:
        quantity = filled_notional_display / price_anchor

    status_key = _normalize_status_key(row.status)
    provider_snapshot_status = _normalize_status_key(
        provider_reconciliation.get("snapshot_status")
        or provider_snapshot.get("normalized_status")
        or provider_snapshot.get("status")
        or payload.get("trading_status")
    )
    if (
        status_key
        and status_key not in LIVE_ACTIVE_ORDER_STATUSES
        and provider_snapshot_status
        in {
            "open",
            "submitted",
            "pending",
            "placing",
            "working",
            "partially_filled",
            "partial",
        }
    ):
        provider_snapshot_status = status_key
    if status_key and status_key not in LIVE_ACTIVE_ORDER_STATUSES and not provider_snapshot_status:
        provider_snapshot_status = status_key

    serialized_payload = dict(payload)
    serialized_payload["trading_status"] = status_key
    if provider_reconciliation:
        serialized_provider_reconciliation = dict(provider_reconciliation)
        if provider_snapshot_status:
            serialized_provider_reconciliation["snapshot_status"] = provider_snapshot_status
        serialized_payload["provider_reconciliation"] = serialized_provider_reconciliation

    cached_market_payload = (
        cached_market.extra_data if cached_market is not None and isinstance(cached_market.extra_data, dict) else {}
    )
    cached_market_slug = str(
        cached_market_payload.get("slug") or (cached_market.slug if cached_market is not None else "") or ""
    ).strip()
    cached_event_slug = str(
        cached_market_payload.get("event_slug") or cached_market_payload.get("eventSlug") or ""
    ).strip()
    cached_condition_id = str(
        cached_market_payload.get("condition_id")
        or cached_market_payload.get("conditionId")
        or (cached_market.condition_id if cached_market is not None else "")
        or ""
    ).strip()
    cached_market_id = str(cached_market_payload.get("id") or "").strip()

    existing_condition_id = str(
        serialized_payload.get("condition_id") or serialized_payload.get("conditionId") or ""
    ).strip()
    condition_id = existing_condition_id or cached_condition_id or _extract_order_condition_id(row)
    if condition_id:
        serialized_payload["condition_id"] = condition_id

    market_slug = str(
        serialized_payload.get("market_slug")
        or serialized_payload.get("marketSlug")
        or serialized_payload.get("slug")
        or ""
    ).strip()
    if not market_slug:
        market_slug = cached_market_slug
    if market_slug:
        serialized_payload["market_slug"] = market_slug
        if not str(serialized_payload.get("slug") or "").strip():
            serialized_payload["slug"] = market_slug

    event_slug = str(serialized_payload.get("event_slug") or serialized_payload.get("eventSlug") or "").strip()
    if not event_slug:
        event_slug = cached_event_slug
    if event_slug:
        serialized_payload["event_slug"] = event_slug

    existing_market_url = str(
        serialized_payload.get("market_url")
        or serialized_payload.get("marketUrl")
        or serialized_payload.get("url")
        or ""
    ).strip()
    existing_polymarket_url = str(
        serialized_payload.get("polymarket_url") or serialized_payload.get("polymarketUrl") or ""
    ).strip()
    market_url = existing_market_url or existing_polymarket_url
    if not market_url:
        market_url = (
            build_polymarket_market_url(
                event_slug=event_slug or None,
                market_slug=market_slug or None,
                market_id=cached_market_id or row.market_id,
                condition_id=condition_id or row.market_id,
            )
            or ""
        ).strip()
    if market_url:
        if not existing_market_url:
            serialized_payload["market_url"] = market_url
            serialized_payload["url"] = market_url
        if not existing_polymarket_url and "polymarket.com" in market_url.lower():
            serialized_payload["polymarket_url"] = market_url

    mark_updated_at = str(position_state.get("last_marked_at") or provider_reconciliation.get("reconciled_at") or "")

    unrealized_pnl = None
    if (
        status_key in {"submitted", "open", "executed"}
        and current_price is not None
        and current_price > 0
        and quantity > 0
    ):
        unrealized_pnl = (quantity * current_price) - filled_notional_display

    signal_payload = signal.payload_json if signal is not None and isinstance(signal.payload_json, dict) else {}
    direction_side, direction_label, yes_label, no_label = _resolve_direction_presentation(
        direction=row.direction,
        payload=payload,
        signal_payload=signal_payload,
        market_payload=cached_market_payload,
        market_id=row.market_id,
    )

    source_trade_payload = serialized_payload.get("source_trade")
    source_trade_payload = source_trade_payload if isinstance(source_trade_payload, dict) else {}
    strategy_context_payload = serialized_payload.get("strategy_context")
    strategy_context_payload = strategy_context_payload if isinstance(strategy_context_payload, dict) else {}
    copy_event_payload = strategy_context_payload.get("copy_event")
    copy_event_payload = copy_event_payload if isinstance(copy_event_payload, dict) else {}
    copy_attribution = serialized_payload.get("copy_attribution")
    copy_attribution = dict(copy_attribution) if isinstance(copy_attribution, dict) else {}
    source_wallet = _extract_copy_source_wallet_from_payload(serialized_payload)
    if source_wallet and not str(copy_attribution.get("source_wallet") or "").strip():
        copy_attribution["source_wallet"] = source_wallet

    source_price = safe_float(
        copy_attribution.get("source_price"),
        safe_float(source_trade_payload.get("price"), safe_float(copy_event_payload.get("price"), None)),
    )
    follower_effective_price = safe_float(
        copy_attribution.get("follower_effective_price"),
        average_fill_price if average_fill_price is not None and average_fill_price > 0 else safe_float(row.effective_price),
    )
    if source_price is not None and source_price > 0.0:
        copy_attribution["source_price"] = float(source_price)
    if follower_effective_price is not None and follower_effective_price > 0.0:
        copy_attribution["follower_effective_price"] = float(follower_effective_price)

    side = _extract_copy_side_from_payload(serialized_payload)
    if side and not str(copy_attribution.get("side") or "").strip():
        copy_attribution["side"] = side
    if (
        source_price is not None
        and source_price > 0.0
        and follower_effective_price is not None
        and follower_effective_price > 0.0
    ):
        slippage_bps = ((follower_effective_price - source_price) / source_price) * 10_000.0
        adverse_slippage_bps = slippage_bps
        if side == "BUY":
            adverse_slippage_bps = max(0.0, slippage_bps)
        elif side == "SELL":
            adverse_slippage_bps = max(0.0, -slippage_bps)
        copy_attribution["slippage_bps"] = float(slippage_bps)
        copy_attribution["adverse_slippage_bps"] = float(adverse_slippage_bps)

    if not str(copy_attribution.get("source_tx_hash") or "").strip():
        source_tx_hash = str(copy_event_payload.get("tx_hash") or source_trade_payload.get("tx_hash") or "").strip()
        if source_tx_hash:
            copy_attribution["source_tx_hash"] = source_tx_hash
    if "source_size" not in copy_attribution:
        source_size = safe_float(source_trade_payload.get("size"), safe_float(copy_event_payload.get("size"), None))
        if source_size is not None and source_size > 0.0:
            copy_attribution["source_size"] = float(source_size)
    if "source_notional_usd" not in copy_attribution:
        source_notional = safe_float(source_trade_payload.get("source_notional_usd"), None)
        if source_notional is not None and source_notional > 0.0:
            copy_attribution["source_notional_usd"] = float(source_notional)
    if "source_detection_latency_ms" not in copy_attribution:
        source_detection_latency_ms = safe_float(copy_event_payload.get("latency_ms"), None)
        if source_detection_latency_ms is not None and source_detection_latency_ms >= 0.0:
            copy_attribution["source_detection_latency_ms"] = float(source_detection_latency_ms)
    detected_at_text = str(
        copy_attribution.get("detected_at")
        or copy_event_payload.get("detected_at")
        or copy_event_payload.get("timestamp")
        or source_trade_payload.get("detected_at")
        or ""
    ).strip()
    if detected_at_text and "detected_at" not in copy_attribution:
        copy_attribution["detected_at"] = detected_at_text
    if detected_at_text and "copy_latency_ms" not in copy_attribution and row.created_at is not None:
        try:
            parsed_detected_at = datetime.fromisoformat(detected_at_text.replace("Z", "+00:00"))
        except Exception:
            parsed_detected_at = None
        if parsed_detected_at is not None:
            if parsed_detected_at.tzinfo is None:
                parsed_detected_at = parsed_detected_at.replace(tzinfo=timezone.utc)
            created_at_utc = row.created_at if row.created_at.tzinfo is not None else row.created_at.replace(tzinfo=timezone.utc)
            copy_latency_ms = max(0.0, (created_at_utc.astimezone(timezone.utc) - parsed_detected_at.astimezone(timezone.utc)).total_seconds() * 1000.0)
            copy_attribution["copy_latency_ms"] = float(copy_latency_ms)

    if copy_attribution:
        serialized_payload["copy_attribution"] = copy_attribution

    trade_bundle = _build_trade_bundle(signal, row, sibling_rows)

    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "decision_id": row.decision_id,
        "source": row.source,
        "strategy_key": row.strategy_key,
        "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
        "market_id": row.market_id,
        "market_question": row.market_question,
        "direction": row.direction,
        "direction_side": direction_side,
        "direction_label": direction_label,
        "yes_label": yes_label,
        "no_label": no_label,
        "mode": row.mode,
        "status": row.status,
        "notional_usd": row.notional_usd,
        "entry_price": row.entry_price,
        "effective_price": row.effective_price,
        "execution_wallet_address": str(
            getattr(row, "execution_wallet_address", None)
            or extract_trader_order_execution_wallet_address(payload)
            or ""
        ),
        "provider_order_id": str(getattr(row, "provider_order_id", None) or payload.get("provider_order_id") or ""),
        "provider_clob_order_id": str(
            getattr(row, "provider_clob_order_id", None) or payload.get("provider_clob_order_id") or ""
        ),
        "verification_status": str(getattr(row, "verification_status", None) or TRADER_ORDER_VERIFICATION_LOCAL),
        "verification_source": str(getattr(row, "verification_source", None) or ""),
        "verification_reason": getattr(row, "verification_reason", None),
        "verification_tx_hash": str(
            getattr(row, "verification_tx_hash", None) or extract_trader_order_verification_tx_hash(payload) or ""
        ),
        "verification_disputed": trader_order_verification_hidden(getattr(row, "verification_status", None)),
        "verified_at": to_iso(getattr(row, "verified_at", None)),
        "provider_snapshot_status": provider_snapshot_status,
        "filled_shares": float(max(0.0, filled_shares)),
        "filled_notional_usd": float(max(0.0, filled_notional_display)),
        "average_fill_price": float(average_fill_price)
        if average_fill_price is not None and average_fill_price > 0
        else None,
        "current_price": float(current_price) if current_price is not None and current_price > 0 else None,
        "mark_source": str(position_state.get("last_mark_source") or ""),
        "mark_updated_at": mark_updated_at,
        "unrealized_pnl": float(unrealized_pnl) if unrealized_pnl is not None else None,
        "edge_percent": row.edge_percent,
        "confidence": row.confidence,
        # Returns row.actual_profit directly. The DB-layer guard in
        # models/database.py (_enforce_pnl_verification_guard) coerces
        # this column to NULL on insert/update unless verification_status
        # is "wallet_activity" — so anything that surfaces here is
        # already truth-matched against an on-chain Polymarket trade.
        "actual_profit": row.actual_profit,
        "reason": row.reason,
        "close_trigger": close_trigger or None,
        "close_reason": close_reason or None,
        "trade_bundle": trade_bundle,
        "payload": serialized_payload,
        "copy_attribution": copy_attribution,
        "error_message": row.error_message,
        "event_id": row.event_id,
        "trace_id": row.trace_id,
        "created_at": to_iso(row.created_at),
        "executed_at": to_iso(row.executed_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_execution_session(row: ExecutionSession) -> dict[str, Any]:
    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "signal_id": row.signal_id,
        "decision_id": row.decision_id,
        "source": row.source,
        "strategy_key": row.strategy_key,
        "strategy_version": int(row.strategy_version) if row.strategy_version is not None else None,
        "mode": row.mode,
        "status": row.status,
        "policy": row.policy,
        "plan_id": row.plan_id,
        "market_ids": row.market_ids_json or [],
        "legs_total": int(row.legs_total or 0),
        "legs_completed": int(row.legs_completed or 0),
        "legs_failed": int(row.legs_failed or 0),
        "legs_open": int(row.legs_open or 0),
        "requested_notional_usd": float(row.requested_notional_usd or 0.0),
        "executed_notional_usd": float(row.executed_notional_usd or 0.0),
        "max_unhedged_notional_usd": float(row.max_unhedged_notional_usd or 0.0),
        "unhedged_notional_usd": float(row.unhedged_notional_usd or 0.0),
        "trace_id": row.trace_id,
        "started_at": to_iso(row.started_at),
        "completed_at": to_iso(row.completed_at),
        "expires_at": to_iso(row.expires_at),
        "error_message": row.error_message,
        "payload": row.payload_json or {},
        "created_at": to_iso(row.created_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_execution_leg(row: ExecutionSessionLeg) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "leg_index": int(row.leg_index or 0),
        "leg_id": row.leg_id,
        "market_id": row.market_id,
        "market_question": row.market_question,
        "token_id": row.token_id,
        "side": row.side,
        "outcome": row.outcome,
        "price_policy": row.price_policy,
        "time_in_force": row.time_in_force,
        "target_price": row.target_price,
        "requested_notional_usd": row.requested_notional_usd,
        "requested_shares": row.requested_shares,
        "filled_notional_usd": float(row.filled_notional_usd or 0.0),
        "filled_shares": float(row.filled_shares or 0.0),
        "avg_fill_price": row.avg_fill_price,
        "status": row.status,
        "last_error": row.last_error,
        "metadata": row.metadata_json or {},
        "created_at": to_iso(row.created_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_execution_order(row: ExecutionSessionOrder) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "leg_id": row.leg_id,
        "trader_order_id": row.trader_order_id,
        "provider_order_id": row.provider_order_id,
        "provider_clob_order_id": row.provider_clob_order_id,
        "action": row.action,
        "side": row.side,
        "price": row.price,
        "size": row.size,
        "notional_usd": row.notional_usd,
        "status": row.status,
        "reason": row.reason,
        "payload": row.payload_json or {},
        "error_message": row.error_message,
        "created_at": to_iso(row.created_at),
        "updated_at": to_iso(row.updated_at),
    }


def _serialize_execution_event(row: ExecutionSessionEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "leg_id": row.leg_id,
        "event_type": row.event_type,
        "severity": row.severity,
        "message": row.message,
        "payload": row.payload_json or {},
        "created_at": to_iso(row.created_at),
    }


def _serialize_event(row: TraderEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "trader_id": row.trader_id,
        "event_type": row.event_type,
        "severity": row.severity,
        "verbosity": row.verbosity,
        "source": row.source,
        "operator": row.operator,
        "message": row.message,
        "trace_id": row.trace_id,
        "payload": row.payload_json or {},
        "created_at": to_iso(row.created_at),
    }


async def ensure_orchestrator_control(session: AsyncSession) -> TraderOrchestratorControl:
    row = await session.get(TraderOrchestratorControl, ORCHESTRATOR_CONTROL_ID)
    if row is None:
        row = TraderOrchestratorControl(
            id=ORCHESTRATOR_CONTROL_ID,
            is_enabled=False,
            is_paused=True,
            mode="shadow",
            run_interval_seconds=ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
            kill_switch=False,
            settings_json=_default_control_settings(),
            updated_at=_now(),
        )
        session.add(row)
        await _commit_with_retry(session)
        await session.refresh(row)
    else:
        normalized_settings = await _normalize_control_settings_for_session(session, row.settings_json)
        if row.settings_json != normalized_settings:
            row.settings_json = normalized_settings
            row.updated_at = _now()
            await _commit_with_retry(session)
            await session.refresh(row)
    if not isinstance(row.settings_json, dict):
        row.settings_json = await _normalize_control_settings_for_session(session, {})
        row.updated_at = _now()
        await _commit_with_retry(session)
        await session.refresh(row)
    return row


async def ensure_orchestrator_snapshot(session: AsyncSession) -> TraderOrchestratorSnapshot:
    row = await session.get(TraderOrchestratorSnapshot, ORCHESTRATOR_SNAPSHOT_ID)
    if row is None:
        row = TraderOrchestratorSnapshot(
            id=ORCHESTRATOR_SNAPSHOT_ID,
            running=False,
            enabled=False,
            current_activity="Waiting for trader orchestrator worker.",
            interval_seconds=ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
            stats_json={},
            updated_at=_now(),
        )
        session.add(row)
        await _commit_with_retry(session)
        await session.refresh(row)
    return row


async def read_orchestrator_control(session: AsyncSession) -> dict[str, Any]:
    return _serialize_control(await ensure_orchestrator_control(session))


async def read_orchestrator_snapshot(session: AsyncSession) -> dict[str, Any]:
    runtime_snapshot = runtime_status.get_orchestrator()
    if os.environ.get("HOMERUN_PROCESS_ROLE") == "worker" and runtime_snapshot.get("updated_at") is not None:
        stats = dict(runtime_snapshot.get("stats") or {})
        return await _apply_wallet_live_exposure_floor(session, {
            "id": ORCHESTRATOR_SNAPSHOT_ID,
            "updated_at": runtime_snapshot.get("updated_at"),
            "last_run_at": runtime_snapshot.get("last_run_at"),
            "running": bool(runtime_snapshot.get("running")),
            "enabled": bool(runtime_snapshot.get("enabled")),
            "current_activity": runtime_snapshot.get("current_activity"),
            "interval_seconds": int(runtime_snapshot.get("interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
            "traders_total": int(stats.get("traders_total", 0) or 0),
            "traders_running": int(stats.get("traders_running", 0) or 0),
            "decisions_count": int(stats.get("decisions_count", 0) or 0),
            "orders_count": int(stats.get("orders_count", 0) or 0),
            "open_orders": int(stats.get("open_orders", 0) or 0),
            "gross_exposure_usd": float(stats.get("gross_exposure_usd", 0.0) or 0.0),
            "daily_pnl": float(stats.get("daily_pnl", 0.0) or 0.0),
            "last_error": runtime_snapshot.get("last_error"),
            "stats": stats,
        })
    return await _apply_wallet_live_exposure_floor(session, _serialize_snapshot(await ensure_orchestrator_snapshot(session)))


async def update_orchestrator_control(session: AsyncSession, **updates: Any) -> dict[str, Any]:
    row = await ensure_orchestrator_control(session)
    payload: dict[str, Any] = {}

    for field in ("is_enabled", "is_paused", "mode", "run_interval_seconds", "kill_switch"):
        if field in updates and updates[field] is not None:
            payload[field] = updates[field]

    if "requested_run_at" in updates:
        payload["requested_run_at"] = updates["requested_run_at"]

    if isinstance(updates.get("settings_json"), dict):
        merged = dict(await _normalize_control_settings_for_session(session, row.settings_json))
        merged.update(updates["settings_json"])
        payload["settings_json"] = await _normalize_control_settings_for_session(session, merged)

    payload["updated_at"] = _now()

    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            await session.execute(
                sa_update(TraderOrchestratorControl)
                .where(TraderOrchestratorControl.id == ORCHESTRATOR_CONTROL_ID)
                .values(**payload)
            )
            await session.commit()
            break
        except OperationalError as exc:
            await session.rollback()
            is_locked = _is_retryable_db_error(exc)
            is_last = attempt >= DB_RETRY_ATTEMPTS - 1
            if not is_locked or is_last:
                raise
            await asyncio.sleep(_db_retry_delay(attempt))

    refreshed = await session.get(TraderOrchestratorControl, ORCHESTRATOR_CONTROL_ID)
    if refreshed is None:
        refreshed = await ensure_orchestrator_control(session)
    return _serialize_control(refreshed)


async def write_orchestrator_snapshot(
    session: AsyncSession,
    *,
    running: bool,
    enabled: bool,
    current_activity: Optional[str],
    interval_seconds: Optional[int],
    last_run_at: Optional[datetime] = None,
    last_error: Optional[str] = None,
    stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if isinstance(stats, dict):
        wallet_live_exposure = await get_gross_exposure(session, mode="live")
        if wallet_live_exposure > safe_float(stats.get("gross_exposure_usd"), 0.0):
            stats = dict(stats)
            stats["gross_exposure_usd"] = float(wallet_live_exposure)
            stats["wallet_gross_exposure_floor_usd"] = float(wallet_live_exposure)
    persisted_stats = summarize_worker_stats(stats) if isinstance(stats, dict) else None
    runtime_status.update_orchestrator(
        running=running,
        enabled=enabled,
        current_activity=current_activity,
        interval_seconds=interval_seconds,
        last_run_at=to_iso(last_run_at) if last_run_at is not None else None,
        last_error=last_error,
        stats=stats,
    )
    row = await ensure_orchestrator_snapshot(session)
    row.updated_at = _now()
    row.running = bool(running)
    row.enabled = bool(enabled)
    row.current_activity = current_activity
    if interval_seconds is not None:
        row.interval_seconds = max(1, int(interval_seconds))
    if last_run_at is not None:
        row.last_run_at = last_run_at
    row.last_error = last_error
    if isinstance(stats, dict):
        row.stats_json = persisted_stats or {}
        row.traders_total = int(stats.get("traders_total", row.traders_total or 0) or 0)
        row.traders_running = int(stats.get("traders_running", row.traders_running or 0) or 0)
        row.decisions_count = int(stats.get("decisions_count", row.decisions_count or 0) or 0)
        row.orders_count = int(stats.get("orders_count", row.orders_count or 0) or 0)
        row.open_orders = int(stats.get("open_orders", row.open_orders or 0) or 0)
        row.gross_exposure_usd = float(stats.get("gross_exposure_usd", row.gross_exposure_usd or 0.0) or 0.0)
        row.daily_pnl = float(stats.get("daily_pnl", row.daily_pnl or 0.0) or 0.0)
    await _commit_with_retry(
        session,
        retry_attempts=2,
        base_delay_seconds=0.05,
        max_delay_seconds=0.1,
    )
    await session.refresh(row)
    snapshot_data = _serialize_snapshot(row)

    # Publish orchestrator status event.
    try:
        await event_bus.publish("trader_orchestrator_status", snapshot_data)
    except Exception:
        pass  # fire-and-forget

    return snapshot_data


def list_trader_templates() -> list[dict[str, Any]]:
    return [
        {
            "id": template["id"],
            "name": template["name"],
            "description": template.get("description"),
            "source_configs": _normalize_source_configs(template.get("source_configs") or []),
            "interval_seconds": int(template.get("interval_seconds", 60) or 60),
            "risk_limits": template.get("risk_limits", {}),
        }
        for template in TRADER_TEMPLATES
    ]


async def _normalize_trader_payload(
    session: AsyncSession,
    payload: dict[str, Any],
) -> dict[str, Any]:
    source_configs = _normalize_source_configs(payload.get("source_configs"))
    await _validate_source_configs(session, source_configs)

    risk_limits = StrategySDK.validate_trader_risk_config(payload.get("risk_limits"))
    metadata = StrategySDK.validate_trader_runtime_metadata(payload.get("metadata"))

    return {
        "name": str(payload.get("name") or "").strip(),
        "description": payload.get("description"),
        "mode": _normalize_trader_mode(payload.get("mode")),
        "latency_class": _normalize_trader_latency_class(payload.get("latency_class")),
        "source_configs": source_configs,
        "risk_limits": risk_limits,
        "metadata": metadata,
        "is_enabled": bool(payload.get("is_enabled", True)),
        "is_paused": bool(payload.get("is_paused", False)),
        "block_new_orders": bool(payload.get("block_new_orders", False)),
        "interval_seconds": max(1, min(86400, safe_int(payload.get("interval_seconds"), 60))),
    }


async def list_traders(session: AsyncSession, mode: Optional[str] = None) -> list[dict[str, Any]]:
    query = select(Trader)
    if mode is not None:
        mode_key = _normalize_trader_mode(mode)
        if mode_key == "shadow":
            query = query.where(func.lower(func.coalesce(Trader.mode, "")).in_(("shadow", "")))
        else:
            query = query.where(func.lower(func.coalesce(Trader.mode, "")) == mode_key)
    rows = (await session.execute(query.order_by(Trader.name.asc()))).scalars().all()
    return [_serialize_trader(row) for row in rows]


async def get_trader(session: AsyncSession, trader_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    return _serialize_trader(row) if row else None


async def list_fast_traders(session: AsyncSession) -> list[dict[str, Any]]:
    """Return enabled, unpaused traders on the fast latency tier.

    These traders are owned by ``fast_trader_runtime`` — the shared
    orchestrator loop skips them.  A trader must be explicitly set to
    ``latency_class='fast'`` to opt into the fast-tier runtime.
    """
    query = (
        select(Trader)
        .where(func.lower(func.coalesce(Trader.latency_class, "normal")) == "fast")
        .where(Trader.is_enabled.is_(True), Trader.is_paused.is_(False))
        .order_by(Trader.name.asc())
    )
    rows = (await session.execute(query)).scalars().all()
    return [_serialize_trader(row) for row in rows]


async def seed_default_traders(session: AsyncSession) -> None:
    count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)
    if count > 0:
        return

    for template in TRADER_TEMPLATES:
        source_configs = _normalize_source_configs(template.get("source_configs") or [])
        session.add(
            Trader(
                id=_new_id(),
                name=template["name"],
                description=template.get("description"),
                source_configs_json=source_configs,
                risk_limits_json=template.get("risk_limits") or {},
                metadata_json={"template_id": template["id"]},
                mode="shadow",
                is_enabled=True,
                is_paused=False,
                interval_seconds=int(template.get("interval_seconds", 60) or 60),
                created_at=_now(),
                updated_at=_now(),
            )
        )
    await _commit_with_retry(session)


async def create_trader(session: AsyncSession, payload: dict[str, Any]) -> dict[str, Any]:
    create_payload = dict(payload or {})
    copy_from_trader_id = str(create_payload.pop("copy_from_trader_id", "") or "").strip()
    if copy_from_trader_id:
        source_row = await session.get(Trader, copy_from_trader_id)
        if source_row is None:
            raise ValueError("Source trader not found")
        source_trader = _serialize_trader(source_row)
        copied_payload: dict[str, Any] = {
            "description": source_trader.get("description"),
            "mode": source_trader.get("mode", "shadow"),
            "source_configs": copy.deepcopy(source_trader.get("source_configs", [])),
            "interval_seconds": int(source_trader.get("interval_seconds", 60) or 60),
            "risk_limits": copy.deepcopy(source_trader.get("risk_limits", {})),
            "metadata": copy.deepcopy(source_trader.get("metadata", {})),
            "is_enabled": bool(source_trader.get("is_enabled", True)),
            "is_paused": bool(source_trader.get("is_paused", False)),
            "block_new_orders": bool(source_trader.get("block_new_orders", False)),
        }
        copied_payload.update(create_payload)
        create_payload = copied_payload

    live_mode_requested = _normalize_trader_mode(create_payload.get("mode")) == "live"
    live_start_requested = bool(create_payload.get("is_enabled", True)) and not bool(create_payload.get("is_paused", False))
    raw_risk_limits = create_payload.get("risk_limits")
    explicit_live_trade_cap = safe_float(
        raw_risk_limits.get("max_trade_notional_usd") if isinstance(raw_risk_limits, dict) else None,
        None,
    )
    explicit_live_position_cap = safe_float(
        raw_risk_limits.get("max_position_notional_usd") if isinstance(raw_risk_limits, dict) else None,
        None,
    )
    if (
        live_mode_requested
        and live_start_requested
        and (
            explicit_live_trade_cap is None
            or explicit_live_trade_cap <= 0.0
            or explicit_live_position_cap is None
            or explicit_live_position_cap <= 0.0
        )
    ):
        create_payload["is_paused"] = True

    normalized = await _normalize_trader_payload(session, create_payload)
    if not normalized["name"]:
        raise ValueError("Trader name is required")

    existing = (
        (await session.execute(select(Trader).where(func.lower(Trader.name) == normalized["name"].lower())))
        .scalars()
        .first()
    )
    if existing is not None:
        raise ValueError("Trader name already exists")

    row = Trader(
        id=_new_id(),
        name=normalized["name"],
        description=normalized["description"],
        source_configs_json=normalized["source_configs"],
        risk_limits_json=normalized["risk_limits"],
        metadata_json=normalized["metadata"],
        mode=normalized["mode"],
        latency_class=normalized["latency_class"],
        is_enabled=normalized["is_enabled"],
        is_paused=normalized["is_paused"],
        block_new_orders=normalized["block_new_orders"],
        interval_seconds=normalized["interval_seconds"],
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(row)
    await _commit_with_retry(session)
    await session.refresh(row)
    return _serialize_trader(row)


async def create_trader_from_template(
    session: AsyncSession,
    template_id: str,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    template = get_template(template_id)
    if template is None:
        raise ValueError("Unknown template")

    payload: dict[str, Any] = {
        "name": template["name"],
        "description": template.get("description"),
        "source_configs": template.get("source_configs", []),
        "interval_seconds": template.get("interval_seconds", 60),
        "risk_limits": template.get("risk_limits", {}),
        "metadata": {"template_id": template_id},
        "is_enabled": True,
        "is_paused": False,
    }
    if isinstance(overrides, dict):
        payload.update(overrides)
    return await create_trader(session, payload)


async def update_trader(
    session: AsyncSession,
    trader_id: str,
    payload: dict[str, Any],
) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    if row is None:
        return None

    normalized = await _normalize_trader_payload(session, {**_serialize_trader(row), **payload})
    if "name" in payload:
        row.name = normalized["name"]
    if "description" in payload:
        row.description = normalized["description"]
    if "source_configs" in payload:
        row.source_configs_json = normalized["source_configs"]
    if "risk_limits" in payload:
        row.risk_limits_json = normalized["risk_limits"]
    if "metadata" in payload:
        row.metadata_json = normalized["metadata"]
    if "mode" in payload:
        row.mode = normalized["mode"]
    if "latency_class" in payload:
        row.latency_class = normalized["latency_class"]
    if "is_enabled" in payload:
        row.is_enabled = bool(payload.get("is_enabled"))
    if "is_paused" in payload:
        row.is_paused = bool(payload.get("is_paused"))
    if "block_new_orders" in payload:
        row.block_new_orders = bool(payload.get("block_new_orders"))
    if "interval_seconds" in payload:
        row.interval_seconds = normalized["interval_seconds"]

    row.updated_at = _now()
    await _commit_with_retry(session)
    await session.refresh(row)
    return _serialize_trader(row)


async def delete_trader(session: AsyncSession, trader_id: str, *, force: bool = False) -> bool:
    row = await session.get(Trader, trader_id)
    if row is None:
        return False

    await sync_trader_position_inventory(session, trader_id=trader_id)
    open_position_summary = await get_open_position_summary_for_trader(session, trader_id)
    open_order_summary = await get_open_order_summary_for_trader(session, trader_id)

    open_live_positions = int(open_position_summary.get("live", 0))
    open_shadow_positions = int(open_position_summary.get("shadow", 0))
    open_other_positions = int(open_position_summary.get("other", 0))
    open_total_positions = int(open_position_summary.get("total", 0))
    open_live_orders = int(open_order_summary.get("live", 0))
    open_shadow_orders = int(open_order_summary.get("shadow", 0))
    open_other_orders = int(open_order_summary.get("other", 0))
    open_total_orders = int(open_order_summary.get("total", 0))

    if (
        open_live_positions > 0
        or open_shadow_positions > 0
        or open_other_positions > 0
        or open_live_orders > 0
        or open_shadow_orders > 0
        or open_other_orders > 0
    ) and not force:
        raise ValueError(
            f"Trader {trader_id} has active exposure: "
            f"{open_live_positions} live open position(s), "
            f"{open_shadow_positions} shadow open position(s), "
            f"{open_other_positions} unknown-mode open position(s), "
            f"{open_live_orders} live active order(s), and "
            f"{open_shadow_orders} shadow active order(s), and "
            f"{open_other_orders} unknown-mode active order(s). "
            "Flatten active exposure first, or pass force=True to delete anyway."
        )

    if force and (open_total_orders > 0 or open_total_positions > 0):
        # Force-delete is an explicit operator override. Skip provider calls and
        # close local active orders immediately so manual Polymarket cleanup does
        # not block account hygiene.
        now = _now()
        active_rows = list(
            (
                await session.execute(
                    select(TraderOrder).where(
                        TraderOrder.trader_id == trader_id,
                        func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(OPEN_ORDER_STATUSES)),
                        _visible_trader_order_query_clause(),
                    )
                )
            )
            .scalars()
            .all()
        )
        for active_row in active_rows:
            payload = dict(active_row.payload_json or {})
            payload["cleanup"] = {
                "previous_status": str(active_row.status or ""),
                "target_status": "cancelled",
                "reason": "force_delete_override",
                "performed_at": to_iso(now),
            }
            if _normalize_mode_key(active_row.mode) == "live":
                payload["provider_cancel"] = {
                    "skipped": True,
                    "reason": "force_delete_override",
                    "performed_at": to_iso(now),
                }
            _prev_status_for_cb = str(active_row.status or "").strip().lower()
            _unfilled_for_cb = float(active_row.notional_usd or 0.0)
            active_row.status = "cancelled"
            active_row.updated_at = now
            active_row.payload_json = payload
            existing_reason = str(active_row.reason or "").strip()
            active_row.reason = (
                f"{existing_reason} | cleanup:force_delete_override"
                if existing_reason
                else "cleanup:force_delete_override"
            )
            import services.trader_hot_state as _hs
            _hs.record_order_cancelled(
                trader_id=str(active_row.trader_id or ""),
                mode=str(active_row.mode or ""),
                order_id=str(active_row.id or ""),
                market_id=str(active_row.market_id or ""),
                source=str(active_row.source or ""),
                copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
            )

            # Strategy hook: notify on the cancel.  Skip transitions
            # that were already cancelled / failed / executed (no new
            # cancellation event).
            if _prev_status_for_cb not in {"cancelled", "failed", "rejected", "executed", "closed_win", "closed_loss"}:
                try:
                    _slug = str(active_row.strategy_key or "").strip()
                    if _slug:
                        import asyncio as _asyncio
                        from services.strategy_callbacks import dispatch_on_cancel

                        _asyncio.create_task(
                            dispatch_on_cancel(
                                strategy_slug=_slug,
                                order=active_row,
                                mode=str(active_row.mode or "shadow"),
                                reason="force_delete_override",
                                unfilled_shares=_unfilled_for_cb,
                            )
                        )
                except Exception:
                    pass  # never raise from a cancel notification
        await _commit_with_retry(session)
        await sync_trader_position_inventory(session, trader_id=trader_id)

    # Re-fetch the row: sync_trader_position_inventory may rollback the
    # session when there are no changes (line 5791), which expires the
    # original ORM instance and triggers MissingGreenlet on delete.
    row = await session.get(Trader, trader_id)
    if row is None:
        return False
    await session.delete(row)
    await _commit_with_retry(session)
    return True


async def transfer_open_trades(
    session: AsyncSession,
    from_trader_id: str,
    to_trader_id: str,
) -> dict[str, int]:
    """Reassign all open orders and open positions from one trader to another."""
    now = _now()

    # Transfer open orders
    open_orders = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == from_trader_id,
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(OPEN_ORDER_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )
    for order in open_orders:
        payload = dict(order.payload_json or {})
        payload["transferred"] = {
            "from_trader_id": from_trader_id,
            "to_trader_id": to_trader_id,
            "transferred_at": to_iso(now),
        }
        order.trader_id = to_trader_id
        order.payload_json = payload
        order.updated_at = now

    # Transfer open positions
    open_positions = list(
        (
            await session.execute(
                select(TraderPosition).where(
                    TraderPosition.trader_id == from_trader_id,
                    TraderPosition.status == "open",
                )
            )
        )
        .scalars()
        .all()
    )
    for position in open_positions:
        payload = dict(position.payload_json or {})
        payload["transferred"] = {
            "from_trader_id": from_trader_id,
            "to_trader_id": to_trader_id,
            "transferred_at": to_iso(now),
        }
        position.trader_id = to_trader_id
        position.payload_json = payload
        position.updated_at = now

    await _commit_with_retry(session)

    # Resync inventory for both traders
    await sync_trader_position_inventory(session, trader_id=from_trader_id)
    await sync_trader_position_inventory(session, trader_id=to_trader_id)

    return {"orders": len(open_orders), "positions": len(open_positions)}


async def set_trader_paused(session: AsyncSession, trader_id: str, paused: bool) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    if row is None:
        return None
    row.is_paused = bool(paused)
    row.updated_at = _now()
    await _commit_with_retry(session)
    await session.refresh(row)
    return _serialize_trader(row)


async def request_trader_run(session: AsyncSession, trader_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(Trader, trader_id)
    if row is None:
        return None
    row.requested_run_at = _now()
    row.updated_at = _now()
    await _commit_with_retry(session)
    await session.refresh(row)
    return _serialize_trader(row)


async def clear_trader_run_request(session: AsyncSession, trader_id: str) -> None:
    row = await session.get(Trader, trader_id)
    if row is None:
        return
    row.requested_run_at = None
    row.updated_at = _now()
    await _commit_with_retry(session)


async def create_config_revision(
    session: AsyncSession,
    *,
    trader_id: Optional[str],
    operator: Optional[str],
    reason: Optional[str],
    orchestrator_before: Optional[dict[str, Any]],
    orchestrator_after: Optional[dict[str, Any]],
    trader_before: Optional[dict[str, Any]],
    trader_after: Optional[dict[str, Any]],
) -> None:
    session.add(
        TraderConfigRevision(
            id=_new_id(),
            trader_id=trader_id,
            operator=operator,
            reason=reason,
            orchestrator_before_json=orchestrator_before or {},
            orchestrator_after_json=orchestrator_after or {},
            trader_before_json=trader_before or {},
            trader_after_json=trader_after or {},
            created_at=_now(),
        )
    )
    await _commit_with_retry(session)


async def create_trader_event(
    session: AsyncSession,
    *,
    event_type: str,
    severity: str = "info",
    verbosity: Optional[str] = None,
    trader_id: Optional[str] = None,
    source: Optional[str] = None,
    operator: Optional[str] = None,
    message: Optional[str] = None,
    trace_id: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    commit: bool = True,
    flush_on_add: bool = True,
) -> TraderEvent:
    row = TraderEvent(
        id=_new_id(),
        trader_id=trader_id,
        event_type=str(event_type),
        severity=str(severity or "info"),
        verbosity=str(verbosity).lower() if verbosity else None,
        source=source,
        operator=operator,
        message=message,
        trace_id=trace_id,
        payload_json=payload or {},
        created_at=_now(),
    )
    session.add(row)
    serialized_row = _serialize_event(row)

    async def _publish_event_buses(row_serialized: dict[str, Any]) -> None:
        # In-process event bus (workers within the same process).
        try:
            await event_bus.publish("trader_event", row_serialized)
        except Exception:
            pass
        # Cross-process Redis pub/sub — UI subscribers (via API plane
        # WS bridge) get bot terminal output in <5ms instead of the
        # previous N-second DB-poll latency.  Soft fail: if Redis is
        # down, in-process subscribers still get the event.
        try:
            from services import redis_client
            import json as _json

            client = redis_client.get_client_or_none()
            if client is not None:
                await client.publish(
                    redis_client.namespaced("trader_events"),
                    _json.dumps(row_serialized, default=str),
                )
        except Exception:
            pass

    if not commit:
        await _publish_event_buses(serialized_row)
    if commit:
        try:
            await _commit_with_retry(session)
        except IntegrityError:
            await session.rollback()
            row.trader_id = None
            session.add(row)
            await _commit_with_retry(session)
            serialized_row = _serialize_event(row)
        await _publish_event_buses(serialized_row)
    elif flush_on_add:
        # Same flush rationale as ``create_trader_decision``: detects
        # invalid trader_id FK early.  Hot-path callers that already
        # validated trader_id pass ``flush_on_add=False`` and accept
        # that any IntegrityError will surface at commit time and
        # rollback the per-signal transaction.
        try:
            await session.flush()
        except IntegrityError:
            await session.rollback()
            row.trader_id = None
            session.add(row)
            await session.flush()

    return row


async def list_trader_events(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    limit: int = 200,
    cursor: Optional[str] = None,
    event_types: Optional[list[str]] = None,
) -> tuple[list[TraderEvent], Optional[str]]:
    query = select(TraderEvent).order_by(desc(TraderEvent.created_at), desc(TraderEvent.id))
    if trader_id:
        query = query.where(TraderEvent.trader_id == trader_id)
    if event_types:
        query = query.where(TraderEvent.event_type.in_(event_types))
    if cursor:
        cursor_row = await session.get(TraderEvent, cursor)
        if cursor_row is not None:
            query = query.where(
                or_(
                    TraderEvent.created_at < cursor_row.created_at,
                    and_(TraderEvent.created_at == cursor_row.created_at, TraderEvent.id < cursor_row.id),
                )
            )
    query = query.limit(max(1, min(limit, 500)) + 1)
    rows = list((await session.execute(query)).scalars().all())
    next_cursor = None
    if len(rows) > limit:
        next_cursor = rows[-1].id
        rows = rows[:limit]
    return rows, next_cursor


async def list_trader_decisions(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    trader_ids: Optional[list[str]] = None,
    decision: Optional[str] = None,
    limit: int = 200,
    per_trader_limit: Optional[int] = None,
) -> list[TraderDecision]:
    normalized_trader_ids: list[str] = []
    if trader_ids:
        seen: set[str] = set()
        for raw in trader_ids:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized_trader_ids.append(value)

    if trader_id:
        query = select(TraderDecision).where(TraderDecision.trader_id == trader_id)
        if decision:
            query = query.where(TraderDecision.decision == decision)
        query = query.order_by(desc(TraderDecision.created_at), desc(TraderDecision.id))
        query = query.limit(max(1, min(limit, 1000)))
        return list((await session.execute(query)).scalars().all())

    max_limit = 5000 if normalized_trader_ids else 1000
    capped_limit = max(1, min(limit, max_limit))
    capped_per_trader_limit = (
        max(1, min(int(per_trader_limit or 1), 1000))
        if per_trader_limit is not None and normalized_trader_ids
        else None
    )

    if capped_per_trader_limit is not None:
        ranked_decisions = (
            select(
                TraderDecision.id.label("id"),
                func.row_number()
                .over(
                    partition_by=TraderDecision.trader_id,
                    order_by=(TraderDecision.created_at.desc(), TraderDecision.id.desc()),
                )
                .label("rn"),
            )
            .where(TraderDecision.trader_id.in_(normalized_trader_ids))
        )
        if decision:
            ranked_decisions = ranked_decisions.where(TraderDecision.decision == decision)
        ranked_subquery = ranked_decisions.subquery()
        query = (
            select(TraderDecision)
            .join(ranked_subquery, TraderDecision.id == ranked_subquery.c.id)
            .where(ranked_subquery.c.rn <= capped_per_trader_limit)
            .order_by(desc(TraderDecision.created_at), desc(TraderDecision.id))
            .limit(capped_limit)
        )
        return list((await session.execute(query)).scalars().all())

    query = select(TraderDecision).order_by(desc(TraderDecision.created_at), desc(TraderDecision.id))
    if normalized_trader_ids:
        query = query.where(TraderDecision.trader_id.in_(normalized_trader_ids))
    if decision:
        query = query.where(TraderDecision.decision == decision)
    query = query.limit(capped_limit)
    return list((await session.execute(query)).scalars().all())


async def list_trader_orders(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    status: Optional[str] = None,
    mode: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    since_seconds: Optional[int] = None,
) -> list[TraderOrder]:
    status_values = _expand_trader_order_status_filter(status)
    status_key_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    mode_key = str(mode or "").strip().lower()
    include_all_open_orders = status_values is None or any(
        status_value in OPEN_ORDER_STATUSES for status_value in status_values
    )

    # When fetching without a specific status filter, ALWAYS include all
    # open orders regardless of pagination.  Open positions must never be
    # silently dropped by a row limit — this is a financial application.
    open_orders: list[TraderOrder] = []
    if include_all_open_orders:
        open_query = (
            select(TraderOrder)
            .where(status_key_expr.in_(tuple(OPEN_ORDER_STATUSES)))
            .where(_visible_trader_order_query_clause())
        )
        if trader_id:
            open_query = open_query.where(TraderOrder.trader_id == trader_id)
        if mode_key:
            open_query = open_query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
        open_orders = list((await session.execute(open_query)).scalars().all())

    # Fetch the paginated historical window.
    query = select(TraderOrder).order_by(desc(TraderOrder.created_at))
    query = query.where(_visible_trader_order_query_clause())
    if trader_id:
        query = query.where(TraderOrder.trader_id == trader_id)
    if status_values is not None:
        query = query.where(status_key_expr.in_(status_values))
    if mode_key:
        query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    # Cap the historical window by date so an active trader with
    # hundreds of resolved orders over months doesn't force every
    # poll to ship its full lifetime history.  Open orders are still
    # returned unbounded above (financial correctness — never silently
    # hide an open position); only the paginated *historical* tail is
    # date-filtered.
    if since_seconds is not None and since_seconds > 0:
        cutoff = utcnow() - timedelta(seconds=int(since_seconds))
        cutoff_naive = cutoff.replace(tzinfo=None) if cutoff.tzinfo is not None else cutoff
        query = query.where(TraderOrder.created_at >= cutoff_naive)
    if offset > 0:
        query = query.offset(offset)
    query = query.limit(max(1, min(limit, 20000)))
    paginated = list((await session.execute(query)).scalars().all())

    # Merge: open orders first, then paginated (deduplicated).
    if not open_orders:
        return paginated
    seen_ids = {str(o.id) for o in open_orders}
    merged = list(open_orders)
    for order in paginated:
        if str(order.id) not in seen_ids:
            merged.append(order)
    return merged


async def get_trader_decision_detail(session: AsyncSession, decision_id: str) -> Optional[dict[str, Any]]:
    row = await session.get(TraderDecision, decision_id)
    if row is None:
        return None
    signal_row = None
    if row.signal_id:
        signal_row = await session.get(TradeSignal, row.signal_id)

    checks = (
        (
            await session.execute(
                select(TraderDecisionCheck)
                .where(TraderDecisionCheck.decision_id == decision_id)
                .order_by(TraderDecisionCheck.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    orders = (
        (
            await session.execute(
                select(TraderOrder)
                .where(TraderOrder.decision_id == decision_id)
                .where(_visible_trader_order_query_clause())
                .order_by(desc(TraderOrder.created_at))
            )
        )
        .scalars()
        .all()
    )

    return {
        "decision": _serialize_decision_with_signal(row, signal_row),
        "checks": [
            {
                "id": check.id,
                "check_key": check.check_key,
                "check_label": check.check_label,
                "passed": bool(check.passed),
                "score": check.score,
                "detail": check.detail,
                "payload": check.payload_json or {},
                "created_at": to_iso(check.created_at),
            }
            for check in checks
        ],
        "orders": [_serialize_order(order) for order in orders],
    }


async def create_trader_decision(
    session: AsyncSession,
    *,
    decision_id: str | None = None,
    trader_id: str,
    signal: TradeSignal,
    strategy_key: str,
    strategy_version: int | None = None,
    decision: str,
    reason: Optional[str] = None,
    score: Optional[float] = None,
    checks_summary: Optional[dict[str, Any]] = None,
    risk_snapshot: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    commit: bool = True,
    flush_on_add: bool = True,
) -> TraderDecision:
    row = TraderDecision(
        id=decision_id or _new_id(),
        trader_id=trader_id,
        signal_id=signal.id,
        source=str(signal.source),
        strategy_key=str(strategy_key),
        strategy_version=int(strategy_version) if strategy_version is not None else None,
        decision=str(decision),
        reason=reason,
        score=score,
        trace_id=trace_id,
        checks_summary_json=checks_summary or {},
        risk_snapshot_json=risk_snapshot or {},
        payload_json=payload or {},
        created_at=_now(),
    )
    session.add(row)
    serialized_row = _serialize_decision(row)
    if not commit:
        try:
            await event_bus.publish("trader_decision", serialized_row)
        except Exception:
            pass
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
        try:
            await event_bus.publish("trader_decision", serialized_row)
        except Exception:
            pass
    elif flush_on_add:
        # Flush exists for early IntegrityError detection (invalid
        # trader_id FK).  Hot-path callers that already validated the
        # trader_id pass ``flush_on_add=False`` to skip the round-trip;
        # any IntegrityError will surface at the eventual commit and
        # rollback the per-signal transaction, which is the same
        # outcome with one fewer round-trip on every call.
        await session.flush()

    return row


async def update_trader_decision(
    session: AsyncSession,
    *,
    decision_id: str,
    decision: str | None = None,
    reason: str | None = None,
    payload_patch: dict[str, Any] | None = None,
    checks_summary_patch: dict[str, Any] | None = None,
    commit: bool = True,
) -> TraderDecision | None:
    row = await session.get(TraderDecision, decision_id)
    if row is None:
        return None

    if decision is not None:
        row.decision = str(decision)
    if reason is not None:
        row.reason = reason
    if payload_patch:
        merged_payload = dict(row.payload_json or {})
        merged_payload.update(payload_patch)
        row.payload_json = merged_payload
    if checks_summary_patch:
        merged_checks_summary = dict(row.checks_summary_json or {})
        merged_checks_summary.update(checks_summary_patch)
        row.checks_summary_json = merged_checks_summary

    serialized_row = _serialize_decision(row)
    if not commit:
        try:
            await event_bus.publish("trader_decision", serialized_row)
        except Exception:
            pass
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
        try:
            await event_bus.publish("trader_decision", serialized_row)
        except Exception:
            pass
    else:
        await session.flush()

    return row


async def create_trader_decision_checks(
    session: AsyncSession,
    *,
    decision_id: str,
    checks: list[dict[str, Any]],
    commit: bool = True,
) -> None:
    if not checks:
        return
    for check in checks:
        session.add(
            TraderDecisionCheck(
                id=_new_id(),
                decision_id=decision_id,
                check_key=str(check.get("check_key") or check.get("key") or "check"),
                check_label=str(check.get("check_label") or check.get("label") or "Check"),
                passed=bool(check.get("passed", False)),
                score=check.get("score"),
                detail=check.get("detail"),
                payload_json=check.get("payload") or {},
                created_at=_now(),
            )
        )
    if commit:
        await _commit_with_retry(session)
    else:
        await session.flush()


async def create_trader_order(
    session: AsyncSession,
    *,
    trader_id: str,
    signal: TradeSignal,
    decision_id: Optional[str],
    strategy_key: str | None,
    strategy_version: int | None,
    mode: str,
    status: str,
    notional_usd: Optional[float],
    effective_price: Optional[float],
    reason: Optional[str],
    payload: Optional[dict[str, Any]],
    error_message: Optional[str] = None,
    trace_id: Optional[str] = None,
    commit: bool = True,
) -> TraderOrder:
    now = _now()
    row = build_trader_order_row(
        trader_id=trader_id,
        signal=signal,
        decision_id=decision_id,
        strategy_key=strategy_key,
        strategy_version=strategy_version,
        mode=mode,
        status=status,
        notional_usd=notional_usd,
        effective_price=effective_price,
        reason=reason,
        payload=payload,
        error_message=error_message,
        trace_id=trace_id,
        created_at=now,
    )
    session.add(row)
    event_payload = dict(row.payload_json or {})
    event_payload.update({"status": str(row.status or ""), "mode": str(row.mode or "")})
    append_trader_order_verification_event(
        session,
        trader_order_id=str(row.id),
        verification_status=str(row.verification_status or TRADER_ORDER_VERIFICATION_LOCAL),
        source=row.verification_source,
        event_type="order_created",
        reason=row.verification_reason,
        provider_order_id=row.provider_order_id,
        provider_clob_order_id=row.provider_clob_order_id,
        execution_wallet_address=row.execution_wallet_address,
        tx_hash=row.verification_tx_hash,
        payload_json=event_payload,
        created_at=now,
    )
    serialized_row = _serialize_order(row)
    if not commit:
        try:
            await event_bus.publish("trader_order", serialized_row)
        except Exception:
            pass
    await session.flush()
    if _is_active_order_status(mode, status):
        await sync_trader_position_inventory(
            session,
            trader_id=trader_id,
            mode=str(mode),
            commit=False,
        )
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
        try:
            await event_bus.publish("trader_order", serialized_row)
        except Exception:
            pass

    return row


def build_trader_order_row(
    *,
    trader_id: str,
    signal: Any,
    decision_id: Optional[str],
    strategy_key: str | None,
    strategy_version: int | None,
    mode: str,
    status: str,
    notional_usd: Optional[float],
    effective_price: Optional[float],
    reason: Optional[str],
    payload: Optional[dict[str, Any]],
    error_message: Optional[str] = None,
    trace_id: Optional[str] = None,
    created_at: datetime | None = None,
    market_id: Optional[str] = None,
    market_question: Optional[str] = None,
    direction: Optional[str] = None,
    entry_price: Optional[float] = None,
    edge_percent: Optional[float] = None,
    confidence: Optional[float] = None,
) -> TraderOrder:
    now = created_at or _now()
    order_payload = _sync_order_runtime_payload(
        payload=dict(payload or {}),
        status=status,
        now=now,
    )
    signal_payload = signal.payload_json if isinstance(getattr(signal, "payload_json", None), dict) else {}
    source_trade_payload = signal_payload.get("source_trade")
    source_trade_payload = source_trade_payload if isinstance(source_trade_payload, dict) else {}
    strategy_context_payload = signal_payload.get("strategy_context")
    strategy_context_payload = strategy_context_payload if isinstance(strategy_context_payload, dict) else {}
    copy_event_payload = strategy_context_payload.get("copy_event")
    copy_event_payload = copy_event_payload if isinstance(copy_event_payload, dict) else {}

    if source_trade_payload and not isinstance(order_payload.get("source_trade"), dict):
        order_payload["source_trade"] = dict(source_trade_payload)

    strategy_context_order_payload = (
        dict(order_payload.get("strategy_context"))
        if isinstance(order_payload.get("strategy_context"), dict)
        else {}
    )
    if strategy_context_payload and not strategy_context_order_payload:
        strategy_context_order_payload = dict(strategy_context_payload)
    elif copy_event_payload and not isinstance(strategy_context_order_payload.get("copy_event"), dict):
        strategy_context_order_payload["copy_event"] = dict(copy_event_payload)
    if strategy_context_order_payload:
        order_payload["strategy_context"] = strategy_context_order_payload

    signal_source_key = str(getattr(signal, "source", "") or "").strip().lower()
    strategy_key_normalized = str(strategy_key or "").strip().lower()
    has_copy_payload = bool(source_trade_payload) or bool(copy_event_payload)
    if signal_source_key == "traders" and (strategy_key_normalized == "traders_copy_trade" or has_copy_payload):
        copy_attribution = dict(order_payload.get("copy_attribution") or {})
        source_wallet = _extract_copy_source_wallet_from_payload(order_payload)
        if source_wallet and not str(copy_attribution.get("source_wallet") or "").strip():
            copy_attribution["source_wallet"] = source_wallet
        if "side" not in copy_attribution:
            copy_side = _extract_copy_side_from_payload(order_payload)
            if copy_side:
                copy_attribution["side"] = copy_side

        source_price = safe_float(
            copy_attribution.get("source_price"),
            safe_float(source_trade_payload.get("price"), safe_float(copy_event_payload.get("price"), None)),
        )
        if source_price is not None and source_price > 0.0:
            copy_attribution["source_price"] = float(source_price)

        source_size = safe_float(
            source_trade_payload.get("size"),
            safe_float(copy_event_payload.get("size"), None),
        )
        if source_size is not None and source_size > 0.0 and "source_size" not in copy_attribution:
            copy_attribution["source_size"] = float(source_size)
        source_notional_usd = safe_float(source_trade_payload.get("source_notional_usd"), None)
        if source_notional_usd is not None and source_notional_usd > 0.0 and "source_notional_usd" not in copy_attribution:
            copy_attribution["source_notional_usd"] = float(source_notional_usd)

        source_tx_hash = str(copy_event_payload.get("tx_hash") or source_trade_payload.get("tx_hash") or "").strip()
        if source_tx_hash and not str(copy_attribution.get("source_tx_hash") or "").strip():
            copy_attribution["source_tx_hash"] = source_tx_hash
        source_order_hash = str(copy_event_payload.get("order_hash") or source_trade_payload.get("order_hash") or "").strip()
        if source_order_hash and not str(copy_attribution.get("source_order_hash") or "").strip():
            copy_attribution["source_order_hash"] = source_order_hash

        source_detection_latency_ms = safe_float(copy_event_payload.get("latency_ms"), None)
        if source_detection_latency_ms is not None and source_detection_latency_ms >= 0.0:
            copy_attribution["source_detection_latency_ms"] = float(source_detection_latency_ms)

        detected_at_text = str(
            copy_event_payload.get("detected_at")
            or source_trade_payload.get("detected_at")
            or copy_event_payload.get("timestamp")
            or ""
        ).strip()
        if detected_at_text and "detected_at" not in copy_attribution:
            copy_attribution["detected_at"] = detected_at_text
        if detected_at_text and "copy_latency_ms" not in copy_attribution:
            try:
                parsed_detected_at = datetime.fromisoformat(detected_at_text.replace("Z", "+00:00"))
            except Exception:
                parsed_detected_at = None
            if parsed_detected_at is not None:
                if parsed_detected_at.tzinfo is None:
                    parsed_detected_at = parsed_detected_at.replace(tzinfo=timezone.utc)
                copy_latency_ms = max(
                    0.0,
                    (now.astimezone(timezone.utc) - parsed_detected_at.astimezone(timezone.utc)).total_seconds()
                    * 1000.0,
                )
                copy_attribution["copy_latency_ms"] = float(copy_latency_ms)

        follower_effective_price = safe_float(
            effective_price,
            safe_float(getattr(signal, "effective_price", None), safe_float(getattr(signal, "entry_price", None), None)),
        )
        if follower_effective_price is not None and follower_effective_price > 0.0:
            copy_attribution["follower_effective_price"] = float(follower_effective_price)
        if (
            source_price is not None
            and source_price > 0.0
            and follower_effective_price is not None
            and follower_effective_price > 0.0
        ):
            slippage_bps = ((follower_effective_price - source_price) / source_price) * 10_000.0
            side = str(copy_attribution.get("side") or "").strip().upper()
            adverse_slippage_bps = slippage_bps
            if side == "BUY":
                adverse_slippage_bps = max(0.0, slippage_bps)
            elif side == "SELL":
                adverse_slippage_bps = max(0.0, -slippage_bps)
            copy_attribution["slippage_bps"] = float(slippage_bps)
            copy_attribution["adverse_slippage_bps"] = float(adverse_slippage_bps)

        order_payload["copy_attribution"] = copy_attribution

    resolved_market_id = str(market_id or getattr(signal, "market_id", "") or "")
    resolved_market_question = (
        str(market_question).strip()
        if market_question is not None
        else str(getattr(signal, "market_question", "") or "").strip()
    )
    resolved_direction = str(direction or getattr(signal, "direction", "") or "").strip()
    resolved_entry_price = entry_price if entry_price is not None else getattr(signal, "entry_price", None)
    resolved_edge_percent = edge_percent if edge_percent is not None else getattr(signal, "edge_percent", None)
    resolved_confidence = confidence if confidence is not None else getattr(signal, "confidence", None)
    verification_fields = derive_trader_order_verification(
        mode=mode,
        status=status,
        reason=reason,
        payload=order_payload,
    )

    row = TraderOrder(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=signal.id,
        decision_id=decision_id,
        source=str(signal.source),
        strategy_key=str(strategy_key or "").strip().lower() or None,
        strategy_version=int(strategy_version) if strategy_version is not None else None,
        market_id=resolved_market_id,
        market_question=resolved_market_question or None,
        direction=resolved_direction or None,
        mode=str(mode),
        status=str(status),
        notional_usd=notional_usd,
        entry_price=resolved_entry_price,
        effective_price=effective_price,
        execution_wallet_address=verification_fields.get("execution_wallet_address"),
        provider_order_id=verification_fields.get("provider_order_id"),
        provider_clob_order_id=verification_fields.get("provider_clob_order_id"),
        verification_status=str(verification_fields.get("verification_status") or TRADER_ORDER_VERIFICATION_LOCAL),
        verification_source=verification_fields.get("verification_source"),
        verification_reason=verification_fields.get("verification_reason"),
        verification_tx_hash=verification_fields.get("verification_tx_hash"),
        verified_at=now,
        edge_percent=resolved_edge_percent,
        confidence=resolved_confidence,
        reason=reason,
        payload_json=order_payload,
        error_message=error_message,
        trace_id=trace_id,
        created_at=now,
        executed_at=now if status in {"executed", "open"} else None,
        updated_at=now,
    )
    return row


async def _refresh_execution_session_rollups(
    session: AsyncSession,
    *,
    session_id: str,
) -> ExecutionSession | None:
    row = await session.get(ExecutionSession, session_id)
    if row is None:
        return None

    status_key = func.lower(func.coalesce(ExecutionSessionLeg.status, ""))
    side_key = func.lower(func.coalesce(ExecutionSessionLeg.side, ""))
    leg_notional = func.abs(func.coalesce(ExecutionSessionLeg.filled_notional_usd, 0.0))
    failed_statuses = ("failed", "cancelled", "expired")
    open_statuses = ("open", "submitted", "placing", "working", "partial", "hedging", "pending")
    rollup_row = (
        await session.execute(
            select(
                func.count(ExecutionSessionLeg.id).label("legs_total"),
                func.sum(case((status_key == "completed", 1), else_=0)).label("legs_completed"),
                func.sum(case((status_key.in_(failed_statuses), 1), else_=0)).label("legs_failed"),
                func.sum(case((status_key.in_(open_statuses), 1), else_=0)).label("legs_open"),
                func.sum(leg_notional).label("executed_notional_usd"),
                func.sum(case((side_key == "sell", leg_notional), else_=0.0)).label("sell_notional_usd"),
                func.sum(case((side_key == "sell", 0.0), else_=leg_notional)).label("buy_notional_usd"),
            ).where(ExecutionSessionLeg.session_id == session_id)
        )
    ).one()

    row.legs_total = safe_int(rollup_row.legs_total, 0)
    row.legs_completed = safe_int(rollup_row.legs_completed, 0)
    row.legs_failed = safe_int(rollup_row.legs_failed, 0)
    row.legs_open = safe_int(rollup_row.legs_open, 0)
    row.executed_notional_usd = safe_float(rollup_row.executed_notional_usd, 0.0)
    buy_notional = safe_float(rollup_row.buy_notional_usd, 0.0)
    sell_notional = safe_float(rollup_row.sell_notional_usd, 0.0)
    row.unhedged_notional_usd = abs(buy_notional - sell_notional)
    row.updated_at = _now()
    await session.flush()
    return row


def _apply_execution_session_rollups_from_rows(
    row: ExecutionSession,
    leg_rows: list[ExecutionSessionLeg],
) -> None:
    failed_statuses = {"failed", "cancelled", "expired"}
    open_statuses = {"open", "submitted", "placing", "working", "partial", "hedging", "pending"}
    legs_total = 0
    legs_completed = 0
    legs_failed = 0
    legs_open = 0
    executed_notional_usd = 0.0
    buy_notional_usd = 0.0
    sell_notional_usd = 0.0

    for leg_row in leg_rows:
        legs_total += 1
        status_key = _normalize_status_key(leg_row.status)
        if status_key == "completed":
            legs_completed += 1
        elif status_key in failed_statuses:
            legs_failed += 1
        elif status_key in open_statuses:
            legs_open += 1

        leg_notional = abs(safe_float(leg_row.filled_notional_usd, 0.0) or 0.0)
        executed_notional_usd += leg_notional
        if _normalize_status_key(leg_row.side) == "sell":
            sell_notional_usd += leg_notional
        else:
            buy_notional_usd += leg_notional

    row.legs_total = legs_total
    row.legs_completed = legs_completed
    row.legs_failed = legs_failed
    row.legs_open = legs_open
    row.executed_notional_usd = executed_notional_usd
    row.unhedged_notional_usd = abs(buy_notional_usd - sell_notional_usd)
    row.updated_at = _now()


def build_execution_session_rows(
    *,
    trader_id: str,
    signal: Any,
    decision_id: str | None,
    strategy_key: str | None,
    strategy_version: int | None,
    mode: str,
    policy: str,
    plan_id: str | None,
    legs: list[dict[str, Any]],
    requested_notional_usd: float,
    max_unhedged_notional_usd: float,
    expires_at: datetime | None,
    payload: dict[str, Any] | None = None,
    trace_id: str | None = None,
    created_at: datetime | None = None,
) -> tuple[ExecutionSession, list[ExecutionSessionLeg]]:
    now = created_at or _now()
    signal_payload = getattr(signal, "payload_json", None)
    signal_payload = signal_payload if isinstance(signal_payload, dict) else {}
    normalized_payload = ensure_market_roster_payload(
        payload,
        markets=list(signal_payload.get("markets") or []),
        market_id=str(getattr(signal, "market_id", "") or "") or None,
        market_question=str(getattr(signal, "market_question", "") or "") or None,
        event_id=str(signal_payload.get("event_id") or "") or None,
        event_slug=str(signal_payload.get("event_slug") or "") or None,
        event_title=str(signal_payload.get("event_title") or "") or None,
        category=str(signal_payload.get("category") or "") or None,
    )
    row = ExecutionSession(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=signal.id,
        decision_id=decision_id,
        source=str(signal.source),
        strategy_key=str(strategy_key or "") or None,
        strategy_version=int(strategy_version) if strategy_version is not None else None,
        mode=str(mode),
        status="pending",
        policy=str(policy or "SINGLE_LEG"),
        plan_id=str(plan_id or "") or None,
        market_ids_json=sorted(
            {str(leg.get("market_id") or "").strip() for leg in legs if str(leg.get("market_id") or "").strip()}
        ),
        legs_total=0,
        legs_completed=0,
        legs_failed=0,
        legs_open=0,
        requested_notional_usd=float(max(0.0, requested_notional_usd)),
        executed_notional_usd=0.0,
        max_unhedged_notional_usd=float(max(0.0, max_unhedged_notional_usd)),
        unhedged_notional_usd=0.0,
        trace_id=trace_id,
        started_at=now,
        completed_at=None,
        expires_at=expires_at,
        payload_json=normalized_payload,
        created_at=now,
        updated_at=now,
    )

    leg_rows: list[ExecutionSessionLeg] = []
    for index, leg in enumerate(legs):
        target_price = safe_float(leg.get("limit_price"), None)
        requested_notional = safe_float(leg.get("requested_notional_usd"), None)
        requested_shares = safe_float(leg.get("requested_shares"), None)
        leg_rows.append(
            ExecutionSessionLeg(
                id=_new_id(),
                session_id=row.id,
                leg_index=index,
                leg_id=str(leg.get("leg_id") or f"leg_{index + 1}"),
                market_id=str(leg.get("market_id") or signal.market_id),
                market_question=str(leg.get("market_question") or signal.market_question or ""),
                token_id=str(leg.get("token_id") or "").strip() or None,
                side=str(leg.get("side") or "buy"),
                outcome=str(leg.get("outcome") or "").strip() or None,
                price_policy=str(leg.get("price_policy") or "maker_limit"),
                time_in_force=str(leg.get("time_in_force") or "GTC"),
                post_only=bool(leg.get("post_only", False)),
                target_price=target_price,
                requested_notional_usd=requested_notional,
                requested_shares=requested_shares,
                filled_notional_usd=0.0,
                filled_shares=0.0,
                avg_fill_price=None,
                status="pending",
                last_error=None,
                metadata_json=dict(leg.get("metadata") or {}),
                created_at=now,
                updated_at=now,
            )
        )

    _apply_execution_session_rollups_from_rows(row, leg_rows)
    return row, leg_rows


async def create_execution_session(
    session: AsyncSession,
    *,
    trader_id: str,
    signal: TradeSignal,
    decision_id: str | None,
    strategy_key: str | None,
    strategy_version: int | None,
    mode: str,
    policy: str,
    plan_id: str | None,
    legs: list[dict[str, Any]],
    requested_notional_usd: float,
    max_unhedged_notional_usd: float,
    expires_at: datetime | None,
    payload: dict[str, Any] | None = None,
    trace_id: str | None = None,
    commit: bool = True,
) -> ExecutionSession:
    row, leg_rows = build_execution_session_rows(
        trader_id=trader_id,
        signal=signal,
        decision_id=decision_id,
        strategy_key=strategy_key,
        strategy_version=strategy_version,
        mode=mode,
        policy=policy,
        plan_id=plan_id,
        legs=legs,
        requested_notional_usd=requested_notional_usd,
        max_unhedged_notional_usd=max_unhedged_notional_usd,
        expires_at=expires_at,
        payload=payload,
        trace_id=trace_id,
    )
    session.add(row)
    await session.flush()
    for leg_row in leg_rows:
        session.add(leg_row)
    await session.flush()
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)

    try:
        await event_bus.publish("execution_session", _serialize_execution_session(row))
    except Exception:
        pass

    return row


async def list_execution_sessions(
    session: AsyncSession,
    *,
    trader_id: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[ExecutionSession]:
    query = select(ExecutionSession).order_by(desc(ExecutionSession.created_at))
    if trader_id:
        query = query.where(ExecutionSession.trader_id == trader_id)
    if status:
        query = query.where(func.lower(func.coalesce(ExecutionSession.status, "")) == str(status).strip().lower())
    query = query.limit(max(1, min(limit, 1000)))
    return list((await session.execute(query)).scalars().all())


async def list_active_execution_sessions(
    session: AsyncSession,
    *,
    trader_id: str | None = None,
    mode: str | None = None,
    limit: int = 500,
) -> list[ExecutionSession]:
    query = select(ExecutionSession).where(
        func.lower(func.coalesce(ExecutionSession.status, "")).in_(tuple(ACTIVE_EXECUTION_SESSION_STATUSES))
    )
    if trader_id:
        query = query.where(ExecutionSession.trader_id == trader_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(ExecutionSession.mode, "")).not_in(list(_TRADER_MODES)))
        else:
            query = query.where(func.lower(func.coalesce(ExecutionSession.mode, "")) == mode_key)
    query = query.order_by(ExecutionSession.created_at.asc()).limit(max(1, min(limit, 2000)))
    return list((await session.execute(query)).scalars().all())


async def get_execution_session_leg_rollups(
    session: AsyncSession,
    session_ids: list[str],
) -> dict[str, dict[str, int]]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for raw_session_id in session_ids:
        session_id = str(raw_session_id or "").strip()
        if not session_id or session_id in seen:
            continue
        seen.add(session_id)
        ordered_ids.append(session_id)
    if not ordered_ids:
        return {}

    status_key = func.lower(func.coalesce(ExecutionSessionLeg.status, ""))
    failed_statuses = ("failed", "cancelled", "expired")
    open_statuses = ("open", "submitted", "placing", "working", "partial", "hedging", "pending")
    query = (
        select(
            ExecutionSessionLeg.session_id.label("session_id"),
            func.count(ExecutionSessionLeg.id).label("legs_total"),
            func.sum(case((status_key == "completed", 1), else_=0)).label("legs_completed"),
            func.sum(case((status_key.in_(failed_statuses), 1), else_=0)).label("legs_failed"),
            func.sum(case((status_key.in_(open_statuses), 1), else_=0)).label("legs_open"),
        )
        .where(ExecutionSessionLeg.session_id.in_(ordered_ids))
        .group_by(ExecutionSessionLeg.session_id)
    )
    rows = (await session.execute(query)).all()
    out: dict[str, dict[str, int]] = {
        session_id: {"legs_total": 0, "legs_completed": 0, "legs_failed": 0, "legs_open": 0}
        for session_id in ordered_ids
    }
    for row in rows:
        session_id = str(row.session_id or "").strip()
        if not session_id:
            continue
        out[session_id] = {
            "legs_total": int(row.legs_total or 0),
            "legs_completed": int(row.legs_completed or 0),
            "legs_failed": int(row.legs_failed or 0),
            "legs_open": int(row.legs_open or 0),
        }
    return out

async def get_execution_session_detail(
    session: AsyncSession,
    session_id: str,
) -> dict[str, Any] | None:
    row = await session.get(ExecutionSession, session_id)
    if row is None:
        return None
    legs = (
        (
            await session.execute(
                select(ExecutionSessionLeg)
                .where(ExecutionSessionLeg.session_id == session_id)
                .order_by(ExecutionSessionLeg.leg_index.asc())
            )
        )
        .scalars()
        .all()
    )
    orders = (
        (
            await session.execute(
                select(ExecutionSessionOrder)
                .where(ExecutionSessionOrder.session_id == session_id)
                .order_by(desc(ExecutionSessionOrder.created_at))
            )
        )
        .scalars()
        .all()
    )
    events = (
        (
            await session.execute(
                select(ExecutionSessionEvent)
                .where(ExecutionSessionEvent.session_id == session_id)
                .order_by(desc(ExecutionSessionEvent.created_at))
            )
        )
        .scalars()
        .all()
    )
    return {
        "session": _serialize_execution_session(row),
        "legs": [_serialize_execution_leg(leg) for leg in legs],
        "orders": [_serialize_execution_order(order) for order in orders],
        "events": [_serialize_execution_event(event) for event in events],
    }


async def update_execution_session_status(
    session: AsyncSession,
    *,
    session_id: str,
    status: str,
    error_message: str | None = None,
    payload_patch: dict[str, Any] | None = None,
    commit: bool = True,
) -> ExecutionSession | None:
    row = await session.get(ExecutionSession, session_id)
    if row is None:
        return None
    now = _now()
    row.status = str(status)
    row.error_message = error_message
    if payload_patch:
        merged = dict(row.payload_json or {})
        merged.update(payload_patch)
        row.payload_json = merged
    if str(status).strip().lower() in TERMINAL_EXECUTION_SESSION_STATUSES:
        row.completed_at = now
    row.updated_at = now
    await _refresh_execution_session_rollups(session, session_id=session_id)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)

    try:
        await event_bus.publish("execution_session", _serialize_execution_session(row))
    except Exception:
        pass

    return row


async def update_execution_leg(
    session: AsyncSession,
    *,
    leg_row_id: str,
    status: str | None = None,
    filled_notional_usd: float | None = None,
    filled_shares: float | None = None,
    avg_fill_price: float | None = None,
    last_error: str | None = None,
    metadata_patch: dict[str, Any] | None = None,
    commit: bool = True,
) -> ExecutionSessionLeg | None:
    row = await session.get(ExecutionSessionLeg, leg_row_id)
    if row is None:
        return None
    if status is not None:
        row.status = str(status)
    if filled_notional_usd is not None:
        row.filled_notional_usd = float(max(0.0, filled_notional_usd))
    if filled_shares is not None:
        row.filled_shares = float(max(0.0, filled_shares))
    if avg_fill_price is not None:
        row.avg_fill_price = float(avg_fill_price)
    if last_error is not None:
        row.last_error = last_error
    if metadata_patch:
        merged = dict(row.metadata_json or {})
        merged.update(metadata_patch)
        row.metadata_json = merged
    row.updated_at = _now()

    session_row = await _refresh_execution_session_rollups(session, session_id=row.session_id)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
        if session_row is not None:
            await session.refresh(session_row)

    try:
        await event_bus.publish("execution_leg", _serialize_execution_leg(row))
    except Exception:
        pass
    if session_row is not None:
        try:
            await event_bus.publish("execution_session", _serialize_execution_session(session_row))
        except Exception:
            pass

    return row


async def create_execution_session_order(
    session: AsyncSession,
    *,
    session_id: str,
    leg_id: str,
    trader_order_id: str | None,
    provider_order_id: str | None,
    provider_clob_order_id: str | None,
    action: str,
    side: str,
    price: float | None,
    size: float | None,
    notional_usd: float | None,
    status: str,
    reason: str | None = None,
    payload: dict[str, Any] | None = None,
    error_message: str | None = None,
    commit: bool = True,
) -> ExecutionSessionOrder:
    row = ExecutionSessionOrder(
        id=_new_id(),
        session_id=session_id,
        leg_id=leg_id,
        trader_order_id=trader_order_id,
        provider_order_id=provider_order_id,
        provider_clob_order_id=provider_clob_order_id,
        action=str(action),
        side=str(side),
        price=price,
        size=size,
        notional_usd=notional_usd,
        status=str(status),
        reason=reason,
        payload_json=payload or {},
        error_message=error_message,
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(row)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
    else:
        await session.flush()

    try:
        await event_bus.publish("execution_order", _serialize_execution_order(row))
    except Exception:
        pass

    return row


async def create_execution_session_event(
    session: AsyncSession,
    *,
    session_id: str,
    event_type: str,
    severity: str = "info",
    leg_id: str | None = None,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
    commit: bool = True,
) -> ExecutionSessionEvent:
    row = ExecutionSessionEvent(
        id=_new_id(),
        session_id=session_id,
        leg_id=leg_id,
        event_type=str(event_type),
        severity=str(severity or "info"),
        message=message,
        payload_json=payload or {},
        created_at=_now(),
    )
    session.add(row)
    if commit:
        await _commit_with_retry(session)
        await session.refresh(row)
    else:
        await session.flush()

    try:
        await event_bus.publish("execution_session_event", _serialize_execution_event(row))
    except Exception:
        pass

    return row


async def record_signal_consumption(
    session: AsyncSession,
    *,
    trader_id: str,
    signal_id: str,
    outcome: str,
    reason: Optional[str] = None,
    decision_id: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    signal_updated_at: Optional[datetime] = None,
    commit: bool = True,
) -> None:
    """Record a per-trader consumption row.

    ``signal_updated_at`` (optional): pass the signal's ``updated_at``
    if the caller already has it in memory.  Without this hint, we use
    ``now`` as ``consumed_at`` — which is correct unless the signal was
    updated *after* we computed ``now`` (clock skew or sub-millisecond
    update race), in which case the signal harmlessly re-emerges on the
    next ``list_unconsumed_trade_signals`` call.  Pre-2026-04-30 we did
    ``await session.get(TradeSignal, signal_id)`` here defensively, but
    that round-trip dominated the per-signal write cost
    (``ps_decision_writes`` 1-4s under contention) — and the orchestrator
    hot path always has the signal object in scope, so passing it via
    ``signal_updated_at`` is strictly better.
    """
    now = _now()
    consumed_at = now
    if signal_updated_at is not None and signal_updated_at > consumed_at:
        consumed_at = signal_updated_at
    stmt = pg_insert(TraderSignalConsumption).values(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=signal_id,
        decision_id=decision_id,
        outcome=outcome,
        reason=reason,
        payload_json=payload or {},
        consumed_at=consumed_at,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_trader_signal_consumption",
        set_={
            "decision_id": stmt.excluded.decision_id,
            "outcome": stmt.excluded.outcome,
            "reason": stmt.excluded.reason,
            "payload_json": stmt.excluded.payload_json,
            "consumed_at": stmt.excluded.consumed_at,
        },
    )
    await session.execute(stmt)
    if commit:
        await _commit_with_retry(session)

async def list_unconsumed_trade_signals(
    session: AsyncSession,
    *,
    trader_id: str,
    sources: Optional[list[str]] = None,
    statuses: Optional[list[str]] = None,
    strategy_types_by_source: Optional[dict[str, Any]] = None,
    cursor_runtime_sequence: Optional[int] = None,
    cursor_created_at: Optional[datetime] = None,
    cursor_signal_id: Optional[str] = None,
    signal_ids: Optional[list[str]] = None,
    exclude_market_ids: Optional[list[str]] = None,
    limit: int = 200,
    defer_heavy_columns: bool = False,
) -> list[TradeSignal]:
    now = _now().replace(tzinfo=None)
    scanner_runtime_ready_clause = or_(
        func.lower(func.coalesce(TradeSignal.source, "")) != "scanner",
        TradeSignal.runtime_sequence.is_not(None),
    )
    normalized_cursor_runtime_sequence = safe_int(cursor_runtime_sequence, None)
    normalized_cursor_created_at = cursor_created_at
    if isinstance(normalized_cursor_created_at, datetime) and normalized_cursor_created_at.tzinfo is not None:
        normalized_cursor_created_at = normalized_cursor_created_at.astimezone(timezone.utc).replace(tzinfo=None)
    normalized_cursor_signal_id = str(cursor_signal_id or "").strip()
    cursor_signal_consumed = False
    if normalized_cursor_signal_id:
        cursor_signal_consumed = bool(
            (
                await session.execute(
                    select(func.count())
                    .select_from(TraderSignalConsumption)
                    .where(TraderSignalConsumption.trader_id == trader_id)
                    .where(TraderSignalConsumption.signal_id == normalized_cursor_signal_id)
                )
            ).scalar_one()
            or 0
        ) > 0
    signal_sort_ts = func.coalesce(TradeSignal.updated_at, TradeSignal.created_at)
    # Replace the LEFT JOIN + GROUP BY MAX with two correlated NOT EXISTS
    # subqueries. The uq_trader_signal_consumption (trader_id, signal_id)
    # constraint guarantees at most one row per pair, so MAX(consumed_at)
    # was always redundant work. Anti-join via NOT EXISTS lets the planner
    # short-circuit per row using the unique index instead of materializing
    # the entire consumption ledger for the trader.
    unconsumed_or_reactivated_clause = ~(
        select(TraderSignalConsumption.id)
        .where(
            and_(
                TraderSignalConsumption.trader_id == trader_id,
                TraderSignalConsumption.signal_id == TradeSignal.id,
                TraderSignalConsumption.consumed_at >= signal_sort_ts,
            )
        )
        .exists()
    )
    never_consumed_clause = ~(
        select(TraderSignalConsumption.id)
        .where(
            and_(
                TraderSignalConsumption.trader_id == trader_id,
                TraderSignalConsumption.signal_id == TradeSignal.id,
            )
        )
        .exists()
    )
    pending_unconsumed_clause = and_(
        never_consumed_clause,
        func.lower(func.coalesce(TradeSignal.status, "")) == "pending",
    )
    # The cursor-bypass branch is only meaningful when the cursor signal
    # has already been consumed; otherwise it can never widen the result
    # set (cursor advances strictly forward), so collapse the clause to
    # FALSE rather than emit an EXISTS subquery the planner has to evaluate.
    if cursor_signal_consumed:
        pending_unconsumed_cursor_bypass_clause = pending_unconsumed_clause
    else:
        pending_unconsumed_cursor_bypass_clause = sa_text("FALSE")
    # Within the result set (outer WHERE already enforces
    # unconsumed_or_reactivated), "reactivated" is exactly the complement
    # of "never consumed" — i.e. the signal has a consumption row that
    # predates the latest signal_sort_ts. We use this for cursor-bypass
    # branches so reactivated signals re-emerge even when their
    # runtime_sequence falls below the cursor.
    reactivated_after_consumption_clause = ~never_consumed_clause
    # ``defer_heavy_columns=True`` skips ``payload_json`` and
    # ``strategy_context_json`` at SELECT time.  Each is ~3KB JSON
    # decoded by asyncpg ON THE EVENT LOOP — for a 200-row result the
    # fast trader's cycle blocked 2-4 seconds on JSON decode alone,
    # exceeding the 2.5s fast-tier ``statement_timeout`` and corrupting
    # asyncpg protocol state.  Fast trader never reads these columns
    # so it opts in; the orchestrator worker keeps the default eager
    # load because it does need the payload for execution.
    query = select(TradeSignal)
    if defer_heavy_columns:
        query = query.options(
            defer(TradeSignal.payload_json),
            defer(TradeSignal.strategy_context_json),
            defer(TradeSignal.quality_rejection_reasons),
        )
    query = (
        query
        .where(unconsumed_or_reactivated_clause)
        .where(or_(TradeSignal.expires_at.is_(None), TradeSignal.expires_at >= now))
        .where(scanner_runtime_ready_clause)
        .limit(max(1, min(limit, 5000)))
    )
    normalized_signal_ids = [str(signal_id or "").strip() for signal_id in (signal_ids or []) if str(signal_id or "").strip()]
    if signal_ids is not None:
        if not normalized_signal_ids:
            return []
        query = query.where(TradeSignal.id.in_(normalized_signal_ids))
    normalized_excluded_market_ids = [
        str(market_id or "").strip()
        for market_id in (exclude_market_ids or [])
        if str(market_id or "").strip()
    ]
    if normalized_excluded_market_ids:
        query = query.where(
            or_(
                TradeSignal.market_id.is_(None),
                TradeSignal.market_id == "",
                TradeSignal.market_id.not_in(normalized_excluded_market_ids),
            )
        )
    if normalized_cursor_runtime_sequence is not None:
        query = query.where(
            or_(
                pending_unconsumed_cursor_bypass_clause,
                reactivated_after_consumption_clause,
                TradeSignal.runtime_sequence.is_(None),
                TradeSignal.runtime_sequence > normalized_cursor_runtime_sequence,
            )
        )
    if normalized_cursor_created_at is not None:
        if normalized_cursor_signal_id:
            query = query.where(
                or_(
                    pending_unconsumed_cursor_bypass_clause,
                    reactivated_after_consumption_clause,
                    signal_sort_ts > normalized_cursor_created_at,
                    and_(signal_sort_ts == normalized_cursor_created_at, TradeSignal.id > normalized_cursor_signal_id),
                )
            )
        else:
            query = query.where(
                or_(
                    pending_unconsumed_cursor_bypass_clause,
                    reactivated_after_consumption_clause,
                    signal_sort_ts > normalized_cursor_created_at,
                )
            )
    normalized_statuses = (
        [str(status or "").strip().lower() for status in statuses if str(status or "").strip()]
        if statuses is not None
        else ["pending"]
    )
    if normalized_statuses:
        query = query.where(func.lower(func.coalesce(TradeSignal.status, "")).in_(normalized_statuses))
    else:
        return []

    if sources is not None:
        normalized_sources = [str(source).strip().lower() for source in sources if str(source).strip()]
        if not normalized_sources:
            return []
        query = query.where(func.lower(func.coalesce(TradeSignal.source, "")).in_(normalized_sources))

    normalized_strategy_types_by_source: dict[str, list[str]] = {}
    if isinstance(strategy_types_by_source, dict):
        for raw_source, raw_strategy_types in strategy_types_by_source.items():
            source_key = str(raw_source or "").strip().lower()
            if not source_key:
                continue
            if isinstance(raw_strategy_types, str):
                candidates = [raw_strategy_types]
            elif isinstance(raw_strategy_types, (list, tuple, set)):
                candidates = list(raw_strategy_types)
            else:
                candidates = []
            normalized: list[str] = []
            seen: set[str] = set()
            for raw_strategy in candidates:
                strategy_type = str(raw_strategy or "").strip().lower()
                if not strategy_type or strategy_type in seen:
                    continue
                seen.add(strategy_type)
                normalized.append(strategy_type)
            if normalized:
                normalized_strategy_types_by_source[source_key] = normalized
    if normalized_strategy_types_by_source:
        source_strategy_clauses = []
        for source_key, allowed_strategy_types in normalized_strategy_types_by_source.items():
            source_strategy_clauses.append(
                and_(
                    func.lower(func.coalesce(TradeSignal.source, "")) == source_key,
                    func.lower(func.coalesce(TradeSignal.strategy_type, "")).in_(allowed_strategy_types),
                )
            )
        if not source_strategy_clauses:
            return []
        query = query.where(or_(*source_strategy_clauses))
    if normalized_cursor_runtime_sequence is not None:
        query = query.order_by(
            TradeSignal.runtime_sequence.asc().nulls_last(),
            signal_sort_ts.asc(),
            TradeSignal.id.asc(),
        )
    else:
        query = query.order_by(TradeSignal.runtime_sequence.desc().nulls_last(), signal_sort_ts.asc(), TradeSignal.id.asc())
    return list((await session.execute(query)).scalars().all())


async def get_trader_signal_cursor(
    session: AsyncSession,
    *,
    trader_id: str,
) -> tuple[Optional[datetime], Optional[str]]:
    row = await session.get(TraderSignalCursor, trader_id)
    if row is None:
        return None, None
    return row.last_signal_created_at, row.last_signal_id


async def upsert_trader_signal_cursor(
    session: AsyncSession,
    *,
    trader_id: str,
    last_signal_created_at: Optional[datetime],
    last_signal_id: Optional[str],
    last_runtime_sequence: Optional[int] = None,
    commit: bool = True,
) -> None:
    """Upsert the per-trader signal cursor.

    Uses ``INSERT ... ON CONFLICT DO UPDATE`` so we issue a single SQL
    round-trip per call — pre-2026-04-30 we did
    ``await session.get(TraderSignalCursor, trader_id)`` followed by
    ``session.add`` or attribute mutation, which doubled the cost on
    every cursor advance and contributed to the ~1-4s
    ``ps_decision_writes`` cost in the orchestrator's per-signal
    pipeline.
    """
    normalized_runtime_sequence = safe_int(last_runtime_sequence, None)
    now = _now()
    stmt = pg_insert(TraderSignalCursor).values(
        trader_id=trader_id,
        last_signal_created_at=last_signal_created_at,
        last_signal_id=str(last_signal_id or "") or None,
        last_runtime_sequence=normalized_runtime_sequence,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[TraderSignalCursor.trader_id],
        set_={
            "last_signal_created_at": stmt.excluded.last_signal_created_at,
            "last_signal_id": stmt.excluded.last_signal_id,
            "last_runtime_sequence": stmt.excluded.last_runtime_sequence,
            "updated_at": stmt.excluded.updated_at,
        },
    )
    await session.execute(stmt)
    if commit:
        await _commit_with_retry(session)


async def reconcile_orphaned_fast_submissions(
    session: AsyncSession,
    *,
    trader_id: str,
    min_age_seconds: float = 30.0,
    commit: bool = True,
) -> dict[str, Any]:
    """Resolve fast-tier rows whose CLOB submission outcome was lost.

    The fast path writes a skeleton ``TraderOrder`` row before the CLOB
    submission and updates it in place when the venue replies. Two
    failure modes leave the row in a half-resolved state:

    1. ``fast_submission_state == "in_flight"`` — process killed
       between CLOB success and the post-update flush. Venue may or may
       not have an order; we don't know which.
    2. ``fast_submission_state == "clob_exception"`` — submit raised; we
       don't know if the request reached the venue.
    3. ``fast_submission_state == "post_update_failed"`` — CLOB
       succeeded, venue has the order, but our post-update DB write
       failed so ``provider_clob_order_id`` is missing.

    This function looks up live open orders at the venue by the
    deterministic metadata key the fast path stamps on every
    submission. For each match it patches ``provider_clob_order_id``
    onto the local row, allowing the existing
    ``reconcile_live_provider_orders`` to take over status syncing on
    the next cycle. Rows older than ``min_age_seconds`` whose key is
    not present at the venue are marked ``failed`` with reason
    ``orphan_no_venue_match`` — the venue never received the order.

    Returns a structured result for telemetry / log consumption.
    """
    from services.live_execution_service import live_execution_service
    from services.trader_orchestrator.fast_idempotency import (
        is_fast_idempotency_key,
        normalize_metadata_for_match,
    )

    cutoff = _now() - timedelta(seconds=max(0.0, float(min_age_seconds)))
    candidate_rows = list(
        (
            await session.execute(
                select(TraderOrder)
                .where(TraderOrder.trader_id == trader_id)
                .where(TraderOrder.mode == "live")
                .where(TraderOrder.provider_clob_order_id.is_(None))
                .where(TraderOrder.created_at < cutoff)
                .where(_visible_trader_order_query_clause())
            )
        )
        .scalars()
        .all()
    )

    eligible: list[tuple[TraderOrder, str]] = []
    for row in candidate_rows:
        payload = row.payload_json if isinstance(row.payload_json, dict) else None
        if not payload:
            continue
        if not bool(payload.get("fast_tier")):
            continue
        submission_state = str(payload.get("fast_submission_state") or "").strip()
        if submission_state not in {"in_flight", "clob_exception", "post_update_failed"}:
            continue
        idempotency_key = str(payload.get("fast_idempotency_key") or "").strip()
        if not is_fast_idempotency_key(idempotency_key):
            continue
        eligible.append((row, idempotency_key))

    result: dict[str, Any] = {
        "trader_id": trader_id,
        "candidates_seen": len(candidate_rows),
        "eligible": len(eligible),
        "matched": 0,
        "marked_orphan": 0,
        "venue_unreachable": False,
        "errors": [],
    }
    if not eligible:
        if commit:
            await _commit_with_retry(session)
        return result

    snapshots_by_metadata: dict[str, list[dict[str, Any]]] = {}
    venue_reachable = False
    # Don't ``release_conn`` here — that would detach the eligible ORM
    # objects, and the patches below would silently fail to flush. The
    # orphan sweep is a low-frequency operation (startup + periodic), so
    # holding the connection across one venue HTTP call is acceptable.
    try:
        snapshots_by_metadata = await live_execution_service.get_open_order_snapshots_by_metadata()
        venue_reachable = True
    except Exception as exc:
        logger.warning(
            "Orphan reconcile: failed to fetch venue snapshots",
            trader_id=trader_id,
            exc_info=exc,
        )
        result["errors"].append(f"venue_fetch_failed: {type(exc).__name__}")
    result["venue_unreachable"] = not venue_reachable

    if not venue_reachable:
        # Don't mark anything orphan when we can't verify against the venue.
        if commit:
            await _commit_with_retry(session)
        return result

    now_ts = _now()
    for row, idempotency_key in eligible:
        normalized = normalize_metadata_for_match(idempotency_key)
        if not normalized:
            continue
        venue_orders = snapshots_by_metadata.get(normalized) or []
        if venue_orders:
            # Pick the most-recent snapshot if duplicates exist (shouldn't,
            # but be defensive — same metadata + same maker + same market
            # should be 1:1).
            snapshot = venue_orders[0]
            clob_order_id = str(snapshot.get("clob_order_id") or "").strip()
            if not clob_order_id:
                continue
            row.provider_clob_order_id = clob_order_id
            updated_payload = dict(row.payload_json or {})
            updated_payload["fast_submission_state"] = "reconciled"
            updated_payload["reconciled_at_iso"] = now_ts.isoformat()
            updated_payload["reconciled_via"] = "orphan_metadata_match"
            updated_payload["provider_clob_order_id"] = clob_order_id
            row.payload_json = updated_payload
            row.updated_at = now_ts
            result["matched"] += 1
        else:
            # Venue has no order with this metadata. The submission never
            # reached the venue (or was cancelled before our sweep ran).
            # Either way, the local row should not stay live.
            row.status = "failed"
            row.error_message = (row.error_message or "orphan: no venue match for fast_idempotency_key")
            updated_payload = dict(row.payload_json or {})
            updated_payload["fast_submission_state"] = "orphan_no_venue_match"
            updated_payload["resolved_at_iso"] = now_ts.isoformat()
            row.payload_json = updated_payload
            row.updated_at = now_ts
            result["marked_orphan"] += 1

    if commit:
        await _commit_with_retry(session)
    return result


async def reconcile_live_provider_orders(
    session: AsyncSession,
    *,
    trader_id: str,
    commit: bool = True,
    broadcast: bool = True,
) -> dict[str, Any]:
    active_order_ids = [
        str(order_id)
        for order_id in (
            await session.execute(
                select(TraderOrder.id).where(
                    TraderOrder.trader_id == trader_id,
                    TraderOrder.mode == "live",
                    TraderOrder.status.in_(tuple(UNFILLED_ORDER_STATUSES)),
                    _visible_trader_order_query_clause(),
                )
            )
        )
        .scalars()
        .all()
        if str(order_id).strip()
    ]
    if not active_order_ids:
        if commit and session.in_transaction():
            await session.rollback()
        return {
            "trader_id": trader_id,
            "provider_ready": True,
            "active_seen": 0,
            "provider_tagged": 0,
            "snapshots_found": 0,
            "updated_orders": 0,
            "updated_session_orders": 0,
            "updated_legs": 0,
            "status_changes": 0,
            "notional_updates": 0,
            "price_updates": 0,
        }

    # Release the DB connection while initializing the external trading provider.
    if commit:
        await session.commit()
    provider_ready = False
    async with release_conn(session):
        try:
            provider_ready = bool(await live_execution_service.ensure_initialized())
        except Exception as exc:
            logger.warning(
                "Live provider reconciliation failed to initialize trading service", trader_id=trader_id, exc_info=exc
            )
            provider_ready = False
    if not provider_ready:
        return {
            "trader_id": trader_id,
            "provider_ready": False,
            "active_seen": len(active_order_ids),
            "provider_tagged": 0,
            "snapshots_found": 0,
            "updated_orders": 0,
            "updated_session_orders": 0,
            "updated_legs": 0,
            "status_changes": 0,
            "notional_updates": 0,
            "price_updates": 0,
        }

    order_reconciliation_targets: dict[str, dict[str, Any]] = {}
    provider_clob_ids: set[str] = set()
    provider_tagged = 0
    tagged_payload_updates = 0
    active_order_rows = list(
        (
            await session.execute(select(TraderOrder).where(TraderOrder.id.in_(active_order_ids)))
        )
        .scalars()
        .all()
    )
    session_order_rows = list(
        (
            await session.execute(
                select(ExecutionSessionOrder)
                .where(ExecutionSessionOrder.trader_order_id.in_(active_order_ids))
                .order_by(desc(ExecutionSessionOrder.created_at))
            )
        )
        .scalars()
        .all()
    )
    session_orders_by_order: dict[str, list[ExecutionSessionOrder]] = {}
    for row in session_order_rows:
        key = str(row.trader_order_id or "").strip()
        if key:
            session_orders_by_order.setdefault(key, []).append(row)

    for order in active_order_rows:
        order_id = str(order.id)
        payload = dict(order.payload_json or {})
        previous_provider_order_id = str(payload.get("provider_order_id") or "").strip()
        previous_provider_clob_order_id = str(payload.get("provider_clob_order_id") or "").strip()
        session_orders = session_orders_by_order.get(order_id, [])
        session_order = session_orders[0] if session_orders else None
        provider_order_id, provider_clob_order_id = _resolve_provider_order_ids(
            order_payload=payload,
            session_order=session_order,
        )
        if provider_order_id:
            payload.setdefault("provider_order_id", provider_order_id)
        if provider_clob_order_id:
            payload.setdefault("provider_clob_order_id", provider_clob_order_id)
            provider_clob_ids.add(provider_clob_order_id)
            provider_tagged += 1
        if (
            str(payload.get("provider_order_id") or "").strip() != previous_provider_order_id
            or str(payload.get("provider_clob_order_id") or "").strip() != previous_provider_clob_order_id
        ):
            tagged_payload_updates += 1
        order_reconciliation_targets[order_id] = {
            "provider_order_id": provider_order_id,
            "provider_clob_order_id": provider_clob_order_id,
        }

    snapshots: dict[str, dict[str, Any]] = {}
    if provider_clob_ids:
        async with release_conn(session):
            try:
                snapshots = await live_execution_service.get_order_snapshots_by_clob_ids(sorted(provider_clob_ids))
            except Exception as exc:
                logger.warning("Live provider reconciliation failed to fetch snapshots", trader_id=trader_id, exc_info=exc)
                snapshots = {}

    now = _now()
    updated_orders = 0
    updated_session_orders = 0
    updated_legs = 0
    updated_sessions = 0
    terminal_session_cancels = 0
    status_changes = 0
    session_status_changes = 0
    notional_updates = 0
    price_updates = 0
    updated_order_rows: dict[str, TraderOrder] = {}
    updated_execution_order_rows: dict[str, ExecutionSessionOrder] = {}
    updated_execution_session_rows: dict[str, ExecutionSession] = {}
    touched_session_ids: set[str] = set()
    leg_state_updates: dict[str, dict[str, Any]] = {}

    active_rows = list(
        (
            await session.execute(select(TraderOrder).where(TraderOrder.id.in_(active_order_ids)))
        )
        .scalars()
        .all()
    )
    session_order_rows = list(
        (
            await session.execute(
                select(ExecutionSessionOrder)
                .where(ExecutionSessionOrder.trader_order_id.in_(active_order_ids))
                .order_by(desc(ExecutionSessionOrder.created_at))
            )
        )
        .scalars()
        .all()
    )
    session_orders_by_order = {}
    linked_session_ids: set[str] = set()
    for row in session_order_rows:
        key = str(row.trader_order_id or "").strip()
        if not key:
            continue
        session_orders_by_order.setdefault(key, []).append(row)
        session_id_key = str(row.session_id or "").strip()
        if session_id_key:
            linked_session_ids.add(session_id_key)

    session_status_by_id: dict[str, str] = {}
    if linked_session_ids:
        linked_sessions = list(
            (
                await session.execute(
                    select(ExecutionSession.id, ExecutionSession.status).where(
                        ExecutionSession.id.in_(list(linked_session_ids))
                    )
                )
            ).all()
        )
        for linked_session in linked_sessions:
            session_status_by_id[str(linked_session.id)] = str(linked_session.status or "")

    for order in active_rows:
        with session.no_autoflush:
            await session.refresh(
                order,
                attribute_names=[
                    "status",
                    "payload_json",
                    "notional_usd",
                    "effective_price",
                    "entry_price",
                    "updated_at",
                    "executed_at",
                ],
            )
        if _normalize_status_key(order.status) not in LIVE_ACTIVE_ORDER_STATUSES:
            continue

        payload = dict(order.payload_json or {})
        order_target = order_reconciliation_targets.get(str(order.id), {})
        session_orders = session_orders_by_order.get(str(order.id), [])
        session_order = session_orders[0] if session_orders else None
        provider_order_id = str(order_target.get("provider_order_id") or "").strip()
        provider_clob_order_id = str(order_target.get("provider_clob_order_id") or "").strip()
        if not provider_order_id and session_order is not None:
            provider_order_id = str(session_order.provider_order_id or "").strip()
        if not provider_clob_order_id and session_order is not None:
            provider_clob_order_id = str(session_order.provider_clob_order_id or "").strip()
        if not provider_order_id or not provider_clob_order_id:
            resolved_provider_order_id, resolved_provider_clob_order_id = _resolve_provider_order_ids(
                order_payload=payload,
                session_order=session_order,
            )
            if not provider_order_id:
                provider_order_id = resolved_provider_order_id
            if not provider_clob_order_id:
                provider_clob_order_id = resolved_provider_clob_order_id
        if not provider_clob_order_id:
            continue
        snapshot = snapshots.get(provider_clob_order_id)
        if not isinstance(snapshot, dict):
            continue

        snapshot_status = str(snapshot.get("normalized_status") or "").strip().lower()
        filled_size = max(0.0, safe_float(snapshot.get("filled_size"), 0.0) or 0.0)
        avg_fill_price = safe_float(snapshot.get("average_fill_price"))
        filled_notional_usd = max(0.0, safe_float(snapshot.get("filled_notional_usd"), 0.0) or 0.0)
        if filled_notional_usd <= 0.0 and filled_size > 0.0:
            reference_price = (
                avg_fill_price
                if avg_fill_price is not None and avg_fill_price > 0
                else safe_float(snapshot.get("limit_price"))
            )
            if reference_price is None or reference_price <= 0:
                reference_price = safe_float(order.effective_price, 0.0) or safe_float(order.entry_price, 0.0)
            if reference_price and reference_price > 0:
                filled_notional_usd = filled_size * reference_price

        mapped_status = _map_provider_snapshot_status(snapshot_status, filled_notional_usd)
        if not mapped_status:
            continue

        linked_session_terminal = False
        for session_order_row in session_orders:
            session_status = _normalize_status_key(session_status_by_id.get(str(session_order_row.session_id or "")))
            if session_status in TERMINAL_EXECUTION_SESSION_STATUSES:
                linked_session_terminal = True
                break

        if (
            linked_session_terminal
            and mapped_status in {"submitted", "open"}
            and filled_notional_usd <= 0.0
            and filled_size <= 0.0
        ):
            cancel_targets: list[str] = []
            if provider_clob_order_id:
                cancel_targets.append(provider_clob_order_id)
            if provider_order_id and provider_order_id not in cancel_targets:
                cancel_targets.append(provider_order_id)
            cancel_success = False
            if commit:
                await session.commit()
            for cancel_target in cancel_targets:
                try:
                    async with release_conn(session):
                        if await live_execution_service.cancel_order(cancel_target):
                            cancel_success = True
                            break
                except Exception as exc:
                    logger.warning(
                        "Failed to cancel terminal-session live provider order",
                        trader_id=trader_id,
                        order_id=order.id,
                        cancel_target=cancel_target,
                        exc_info=exc,
                    )
            if cancel_success:
                terminal_session_cancels += 1
                mapped_status = "cancelled"
                snapshot_status = "cancelled"
                snapshot = dict(snapshot)
                snapshot["normalized_status"] = "cancelled"
                snapshot["raw_status"] = "cancelled_session_terminal"

        previous_status = _normalize_status_key(order.status)
        previous_notional = max(0.0, abs(safe_float(order.notional_usd, 0.0) or 0.0))
        previous_effective_price = safe_float(order.effective_price)
        previous_verification_status = str(order.verification_status or "")
        previous_verification_source = str(order.verification_source or "")
        previous_verification_reason = str(order.verification_reason or "")
        previous_verification_tx_hash = str(order.verification_tx_hash or "")
        previous_execution_wallet = str(order.execution_wallet_address or "")
        order_status_to_persist = mapped_status

        if mapped_status in {"submitted", "open"}:
            next_notional = max(0.0, filled_notional_usd)
        elif mapped_status == "executed":
            next_notional = filled_notional_usd if filled_notional_usd > 0.0 else previous_notional
        else:
            next_notional = 0.0

        next_effective_price = previous_effective_price
        if avg_fill_price is not None and avg_fill_price > 0:
            next_effective_price = avg_fill_price

        order.status = order_status_to_persist
        order.notional_usd = float(next_notional)
        if next_effective_price is not None:
            order.effective_price = float(next_effective_price)
        if next_notional > 0.0 and order.executed_at is None:
            order.executed_at = now
        order.updated_at = now
        payload["provider_order_id"] = provider_order_id
        payload["provider_clob_order_id"] = provider_clob_order_id
        payload["provider_reconciliation"] = {
            "reconciled_at": to_iso(now),
            "provider_order_id": provider_order_id,
            "provider_clob_order_id": provider_clob_order_id,
            "snapshot": snapshot,
            "snapshot_status": snapshot_status,
            "mapped_status": mapped_status,
            "filled_size": filled_size,
            "average_fill_price": avg_fill_price,
            "filled_notional_usd": filled_notional_usd,
        }
        if linked_session_terminal:
            payload["provider_reconciliation"]["terminal_session"] = True
        order.payload_json = _sync_order_runtime_payload(
            payload=payload,
            status=order_status_to_persist,
            now=now,
            provider_snapshot_status=snapshot_status,
            mapped_status=mapped_status,
        )
        snapshot_verification_status = (
            TRADER_ORDER_VERIFICATION_VENUE_FILL
            if snapshot_status in {"filled", "partially_filled"} or filled_notional_usd > 0.0 or filled_size > 0.0
            else TRADER_ORDER_VERIFICATION_VENUE_ORDER
        )
        execution_wallet_address = (
            extract_trader_order_execution_wallet_address(payload)
            or str(live_execution_service.get_execution_wallet_address() or "").strip().lower()
            or None
        )
        verification_changed = apply_trader_order_verification(
            order,
            verification_status=snapshot_verification_status,
            verification_source="live_order_snapshot",
            verification_reason=snapshot_status or None,
            provider_order_id=provider_order_id or None,
            provider_clob_order_id=provider_clob_order_id or None,
            execution_wallet_address=execution_wallet_address,
            verification_tx_hash=extract_trader_order_verification_tx_hash(payload),
            verified_at=now,
        )
        if verification_changed or (
            previous_verification_status != str(order.verification_status or "")
            or previous_verification_source != str(order.verification_source or "")
            or previous_verification_reason != str(order.verification_reason or "")
            or previous_verification_tx_hash != str(order.verification_tx_hash or "")
            or previous_execution_wallet != str(order.execution_wallet_address or "")
        ):
            event_payload = dict(order.payload_json or {})
            event_payload.update(
                {
                    "snapshot_status": snapshot_status,
                    "mapped_status": mapped_status,
                    "filled_size": filled_size,
                    "filled_notional_usd": filled_notional_usd,
                }
            )
            append_trader_order_verification_event(
                session,
                trader_order_id=str(order.id),
                verification_status=str(order.verification_status or TRADER_ORDER_VERIFICATION_LOCAL),
                source=order.verification_source,
                event_type="provider_snapshot",
                reason=order.verification_reason,
                provider_order_id=order.provider_order_id,
                provider_clob_order_id=order.provider_clob_order_id,
                execution_wallet_address=order.execution_wallet_address,
                tx_hash=order.verification_tx_hash,
                payload_json=event_payload,
                created_at=now,
            )
        if _is_active_order_status(str(order.mode or ""), order_status_to_persist):
            import services.trader_hot_state as _hs
            _hs.upsert_active_order(
                trader_id=trader_id,
                mode=str(order.mode or ""),
                order_id=str(order.id or ""),
                status=order_status_to_persist,
                market_id=str(order.market_id or ""),
                direction=str(order.direction or ""),
                source=str(order.source or ""),
                notional_usd=float(order.notional_usd or 0.0),
                entry_price=float(order.effective_price or order.entry_price or 0.0),
                token_id=str(payload.get("token_id") or payload.get("selected_token_id") or ""),
                filled_shares=float(filled_size or 0.0),
                payload=dict(order.payload_json or {}),
                copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
            )
        elif order_status_to_persist in {"cancelled", "failed"}:
            import services.trader_hot_state as _hs
            _hs.record_order_cancelled(
                trader_id=trader_id,
                mode=str(order.mode or ""),
                order_id=str(order.id or ""),
                market_id=str(order.market_id or ""),
                source=str(order.source or ""),
                copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
            )
        updated_orders += 1
        updated_order_rows[str(order.id)] = order
        if previous_status != _normalize_status_key(order_status_to_persist):
            status_changes += 1
        if abs(previous_notional - next_notional) > 1e-9:
            notional_updates += 1
        if (
            previous_effective_price is None
            and next_effective_price is not None
            or previous_effective_price is not None
            and next_effective_price is not None
            and abs(previous_effective_price - next_effective_price) > 1e-9
        ):
            price_updates += 1

        for session_order_row in session_orders:
            session_order_row.status = mapped_status
            if avg_fill_price is not None and avg_fill_price > 0:
                session_order_row.price = float(avg_fill_price)
            if filled_size > 0.0:
                session_order_row.size = float(filled_size)
            if mapped_status in {"submitted", "open", "executed"}:
                session_order_row.notional_usd = float(next_notional)
            else:
                session_order_row.notional_usd = float(filled_notional_usd)
            if provider_order_id:
                session_order_row.provider_order_id = provider_order_id
            if provider_clob_order_id:
                session_order_row.provider_clob_order_id = provider_clob_order_id
            session_payload = dict(session_order_row.payload_json or {})
            session_payload["provider_reconciliation"] = {
                "reconciled_at": to_iso(now),
                "snapshot_status": snapshot_status,
                "mapped_status": mapped_status,
                "filled_size": filled_size,
                "average_fill_price": avg_fill_price,
                "filled_notional_usd": filled_notional_usd,
                "provider_order_id": provider_order_id,
                "provider_clob_order_id": provider_clob_order_id,
            }
            session_order_row.payload_json = session_payload
            session_order_row.updated_at = now
            updated_session_orders += 1
            updated_execution_order_rows[str(session_order_row.id)] = session_order_row
            leg_state_updates[str(session_order_row.leg_id)] = {
                "status": mapped_status,
                "filled_notional_usd": float(
                    next_notional if mapped_status in {"submitted", "open", "executed"} else 0.0
                ),
                "filled_shares": float(filled_size),
                "avg_fill_price": float(avg_fill_price) if avg_fill_price is not None and avg_fill_price > 0 else None,
                "provider_order_id": provider_order_id,
                "provider_clob_order_id": provider_clob_order_id,
                "snapshot_status": snapshot_status,
            }
            touched_session_ids.add(str(session_order_row.session_id))

    if leg_state_updates:
        await session.flush()
        leg_rows = list(
            (
                await session.execute(
                    select(ExecutionSessionLeg).where(ExecutionSessionLeg.id.in_(list(leg_state_updates.keys())))
                )
            )
            .scalars()
            .all()
        )
        for leg_row in leg_rows:
            leg_update = leg_state_updates.get(str(leg_row.id))
            if leg_update is None:
                continue
            mapped_status = str(leg_update["status"])
            leg_row.status = _map_live_order_status_to_leg_status(mapped_status)
            leg_row.filled_notional_usd = float(max(0.0, safe_float(leg_update.get("filled_notional_usd"), 0.0) or 0.0))
            leg_row.filled_shares = float(max(0.0, safe_float(leg_update.get("filled_shares"), 0.0) or 0.0))
            avg_fill_price = safe_float(leg_update.get("avg_fill_price"))
            if avg_fill_price is not None and avg_fill_price > 0:
                leg_row.avg_fill_price = float(avg_fill_price)
            if mapped_status in {"failed", "cancelled"}:
                leg_row.last_error = f"provider_reconcile:{mapped_status}"
            else:
                leg_row.last_error = None
            metadata_json = dict(leg_row.metadata_json or {})
            metadata_json["provider_reconciliation"] = {
                "reconciled_at": to_iso(now),
                "provider_order_id": leg_update.get("provider_order_id"),
                "provider_clob_order_id": leg_update.get("provider_clob_order_id"),
                "snapshot_status": leg_update.get("snapshot_status"),
                "mapped_status": mapped_status,
            }
            leg_row.metadata_json = metadata_json
            leg_row.updated_at = now
            updated_legs += 1
            touched_session_ids.add(str(leg_row.session_id))

    if touched_session_ids:
        await session.flush()
    for session_id in touched_session_ids:
        session_row = await _refresh_execution_session_rollups(session, session_id=session_id)
        if session_row is None:
            continue
        previous_session_status = _normalize_status_key(session_row.status)
        next_session_status = previous_session_status
        legs_open = int(session_row.legs_open or 0)
        legs_completed = int(session_row.legs_completed or 0)
        legs_failed = int(session_row.legs_failed or 0)
        if legs_open <= 0:
            if legs_completed > 0 and legs_failed == 0:
                next_session_status = "completed"
                session_row.error_message = None
            elif legs_completed == 0 and legs_failed > 0:
                next_session_status = "failed"
                if not session_row.error_message:
                    session_row.error_message = "All execution legs failed during provider reconciliation."
            elif legs_completed > 0 and legs_failed > 0:
                next_session_status = "failed"
                if not session_row.error_message:
                    session_row.error_message = "Execution session closed with partial fills and failed legs."
        elif previous_session_status in {"pending", "placing", "partial", "hedging"}:
            next_session_status = "working"

        if next_session_status in TERMINAL_EXECUTION_SESSION_STATUSES and session_row.completed_at is None:
            session_row.completed_at = now
        session_row.status = next_session_status
        session_row.updated_at = now
        updated_sessions += 1
        if next_session_status != previous_session_status:
            session_status_changes += 1
        updated_execution_session_rows[str(session_row.id)] = session_row

    if commit and (
        updated_orders > 0
        or updated_session_orders > 0
        or updated_legs > 0
        or updated_sessions > 0
        or tagged_payload_updates > 0
    ):
        await _commit_with_retry(session)
    elif commit and session.in_transaction():
        await session.rollback()
    elif not commit and (
        updated_orders > 0
        or updated_session_orders > 0
        or updated_legs > 0
        or updated_sessions > 0
        or tagged_payload_updates > 0
    ):
        await session.flush()

    if broadcast and (updated_order_rows or updated_execution_order_rows or updated_execution_session_rows):
        order_payloads = [_serialize_order(row) for row in updated_order_rows.values()]
        execution_order_payloads = [_serialize_execution_order(row) for row in updated_execution_order_rows.values()]
        execution_session_payloads = [
            _serialize_execution_session(row) for row in updated_execution_session_rows.values()
        ]

        for payload in order_payloads:
            try:
                await event_bus.publish("trader_order", payload)
            except Exception:
                continue
        for payload in execution_order_payloads:
            try:
                await event_bus.publish("execution_order", payload)
            except Exception:
                continue
        for payload in execution_session_payloads:
            try:
                await event_bus.publish("execution_session", payload)
            except Exception:
                continue

    return {
        "trader_id": trader_id,
        "provider_ready": True,
        "active_seen": len(active_order_ids),
        "provider_tagged": provider_tagged,
        "payload_tag_updates": tagged_payload_updates,
        "snapshots_found": len(snapshots),
        "updated_orders": updated_orders,
        "updated_session_orders": updated_session_orders,
        "updated_legs": updated_legs,
        "updated_sessions": updated_sessions,
        "terminal_session_cancels": terminal_session_cancels,
        "status_changes": status_changes,
        "session_status_changes": session_status_changes,
        "notional_updates": notional_updates,
        "price_updates": price_updates,
    }


async def list_live_wallet_positions_for_trader(
    session: AsyncSession,
    *,
    trader_id: str,
    include_managed: bool = True,
) -> dict[str, Any]:
    trader_row = await session.get(Trader, trader_id)
    if trader_row is None:
        raise ValueError("Trader not found")

    async with release_conn(session):
        wallet_address = await _resolve_execution_wallet_address()
    if not wallet_address:
        return {
            "trader_id": trader_id,
            "wallet_address": None,
            "positions": [],
            "managed_token_ids": [],
            "managed_order_ids": [],
            "summary": {
                "total_positions": 0,
                "managed_positions": 0,
                "unmanaged_positions": 0,
                "returned_positions": 0,
            },
        }

    wallet_address_key = str(wallet_address).strip().lower()
    wallet_position_rows = list(
        (
            await session.execute(
                select(LiveTradingPosition)
                .where(func.lower(func.coalesce(LiveTradingPosition.wallet_address, "")) == wallet_address_key)
                .order_by(LiveTradingPosition.updated_at.desc(), LiveTradingPosition.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    wallet_positions_raw = [
        wallet_position
        for wallet_position in (_live_trading_position_to_wallet_row(row) for row in wallet_position_rows)
        if isinstance(wallet_position, dict)
    ]

    active_live_rows = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "live",
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(LIVE_ACTIVE_ORDER_STATUSES)),
                    _visible_trader_order_query_clause(),
                )
            )
        )
        .scalars()
        .all()
    )
    managed_by_token: dict[str, str] = {}
    for row in active_live_rows:
        token_id = _wallet_position_token_key(_extract_order_token_id(row))
        if token_id and token_id not in managed_by_token:
            managed_by_token[token_id] = str(row.id)

    total_positions = 0
    open_positions = 0
    managed_positions = 0
    managed_open_positions = 0
    normalized_positions: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for wallet_position in wallet_positions_raw:
        if not isinstance(wallet_position, dict):
            continue
        token_id = _read_wallet_text(wallet_position, "asset", "asset_id", "assetId", "token_id", "tokenId")
        token_key = _wallet_position_token_key(token_id)
        if not token_key or token_key in seen_tokens:
            continue
        seen_tokens.add(token_key)

        normalized = _normalize_wallet_position_row(
            wallet_position,
            market_info={},
        )
        if normalized is None:
            continue

        total_positions += 1
        counts_as_open = bool(normalized.get("counts_as_open", True))
        if counts_as_open:
            open_positions += 1
        managed_order_id = managed_by_token.get(token_key)
        is_managed = managed_order_id is not None
        if is_managed:
            managed_positions += 1
            if counts_as_open:
                managed_open_positions += 1
        normalized["is_managed"] = is_managed
        normalized["managed_order_id"] = managed_order_id
        if include_managed or not is_managed:
            normalized_positions.append(normalized)

    normalized_positions.sort(
        key=lambda item: (
            not bool(item.get("counts_as_open", True)),
            bool(item.get("is_managed", False)),
            -float(item.get("current_value") or item.get("initial_value") or 0.0),
        )
    )

    return {
        "trader_id": trader_id,
        "wallet_address": wallet_address,
        "positions": normalized_positions,
        "managed_token_ids": sorted(managed_by_token.keys()),
        "managed_order_ids": sorted(set(managed_by_token.values())),
        "summary": {
            "total_positions": total_positions,
            "open_positions": open_positions,
            "managed_positions": managed_positions,
            "managed_open_positions": managed_open_positions,
            "unmanaged_positions": max(0, total_positions - managed_positions),
            "unmanaged_open_positions": max(0, open_positions - managed_open_positions),
            "returned_positions": len(normalized_positions),
        },
    }


async def adopt_live_wallet_position(
    session: AsyncSession,
    *,
    trader_id: str,
    token_id: str,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    trader_row = await session.get(Trader, trader_id)
    if trader_row is None:
        raise ValueError("Trader not found")

    trader_mode = _normalize_trader_mode(trader_row.mode)
    if trader_mode != "live":
        raise ValueError("Trader mode must be live to adopt wallet positions")

    requested_token_key = _wallet_position_token_key(token_id)
    if not requested_token_key:
        raise ValueError("token_id is required")
    metadata_json = trader_row.metadata_json if isinstance(trader_row.metadata_json, dict) else {}
    assigned_strategy_key = _normalize_strategy_key(metadata_json.get("manual_position_strategy_key"))
    if not assigned_strategy_key:
        source_configs = _normalize_source_configs(trader_row.source_configs_json)
        for source_config in source_configs:
            candidate = _normalize_strategy_key(source_config.get("strategy_key"))
            if candidate:
                assigned_strategy_key = candidate
                break
    if not assigned_strategy_key:
        raise ValueError("Trader must have a configured strategy before adopting live wallet positions")
    await _validate_strategy_key(session, assigned_strategy_key)

    wallet_snapshot = await list_live_wallet_positions_for_trader(
        session,
        trader_id=trader_id,
        include_managed=True,
    )
    wallet_address = str(wallet_snapshot.get("wallet_address") or "").strip()
    if not wallet_address:
        raise ValueError("Execution wallet address is not configured")

    positions = list(wallet_snapshot.get("positions") or [])
    target_position = next(
        (
            position
            for position in positions
            if _wallet_position_token_key(position.get("token_id")) == requested_token_key
        ),
        None,
    )
    if target_position is None:
        raise ValueError(f"Token '{token_id}' not found in execution wallet positions")
    if bool(target_position.get("is_managed")):
        managed_order_id = str(target_position.get("managed_order_id") or "").strip()
        if managed_order_id:
            raise ValueError(f"Token '{token_id}' is already managed by order {managed_order_id}")
        raise ValueError(f"Token '{token_id}' is already managed by this trader")

    direction = str(target_position.get("direction") or "").strip().lower()
    if direction not in {"buy_yes", "buy_no"}:
        raise ValueError("Unable to infer binary direction for wallet position")

    size = max(0.0, safe_float(target_position.get("size"), 0.0) or 0.0)
    if size <= 0.0:
        raise ValueError("Wallet position size must be greater than zero")

    entry_price = safe_float(target_position.get("avg_price"))
    initial_value = safe_float(target_position.get("initial_value"))
    if (entry_price is None or entry_price <= 0.0) and initial_value is not None and initial_value > 0.0:
        entry_price = initial_value / size
    if entry_price is None or entry_price <= 0.0:
        raise ValueError("Unable to infer average entry price for wallet position")

    notional_usd = initial_value if initial_value is not None and initial_value > 0.0 else (size * entry_price)
    if notional_usd <= 0.0:
        raise ValueError("Unable to infer position notional value for wallet position")

    current_price = safe_float(target_position.get("current_price"))
    if current_price is None or current_price <= 0.0:
        current_price = entry_price

    resolved_token_id = str(target_position.get("token_id") or "").strip()
    condition_id = _normalize_condition_id(target_position.get("condition_id") or target_position.get("market_id"))
    market_id = str(target_position.get("market_id") or condition_id or resolved_token_id).strip()
    if not market_id:
        raise ValueError("Unable to resolve market id for wallet position")

    selected_outcome = "yes" if direction == "buy_yes" else "no"
    yes_token_id = str(target_position.get("yes_token_id") or "").strip()
    no_token_id = str(target_position.get("no_token_id") or "").strip()
    yes_price = safe_float(target_position.get("yes_price"))
    no_price = safe_float(target_position.get("no_price"))
    market_slug = str(target_position.get("market_slug") or "").strip()
    event_slug = str(target_position.get("event_slug") or "").strip()
    market_url = str(target_position.get("market_url") or "").strip()
    market_question = str(target_position.get("market_question") or "").strip() or None

    now = _now()
    reason_text = str(reason or "manual_wallet_position_adopt").strip() or "manual_wallet_position_adopt"
    mark_price = float(current_price)
    provider_snapshot = {
        "normalized_status": "filled",
        "status": "filled",
        "asset_id": resolved_token_id,
        "token_id": resolved_token_id,
        "filled_size": float(size),
        "average_fill_price": float(entry_price),
        "filled_notional_usd": float(notional_usd),
    }
    payload_json: dict[str, Any] = {
        "source": _MANUAL_LIVE_POSITION_SOURCE,
        "strategy_type": assigned_strategy_key,
        "execution_wallet_address": wallet_address,
        "market_id": market_id,
        "condition_id": condition_id or None,
        "token_id": resolved_token_id,
        "selected_token_id": resolved_token_id,
        "selected_outcome": selected_outcome,
        "yes_token_id": yes_token_id or None,
        "no_token_id": no_token_id or None,
        "market_slug": market_slug or None,
        "event_slug": event_slug or None,
        "market_url": market_url or None,
        "live_market": {
            "market_data_source": "wallet_position_import",
            "condition_id": condition_id or None,
            "selected_token_id": resolved_token_id,
            "selected_outcome": selected_outcome,
            "yes_token_id": yes_token_id or None,
            "no_token_id": no_token_id or None,
            "live_selected_price": float(mark_price),
            "live_yes_price": float(yes_price) if yes_price is not None else None,
            "live_no_price": float(no_price) if no_price is not None else None,
            "live_market_fetched_at": to_iso(now),
            "fetched_at": to_iso(now),
        },
        "provider_reconciliation": {
            "source": "wallet_position_import",
            "snapshot_status": "filled",
            "mapped_status": "executed",
            "reconciled_at": to_iso(now),
            "filled_size": float(size),
            "average_fill_price": float(entry_price),
            "filled_notional_usd": float(notional_usd),
            "snapshot": provider_snapshot,
        },
        "position_state": {
            "highest_price": float(mark_price),
            "lowest_price": float(mark_price),
            "last_mark_price": float(mark_price),
            "last_mark_source": "wallet_mark",
            "last_marked_at": to_iso(now),
        },
        "wallet_position_import": {
            "wallet_address": wallet_address,
            "token_id": resolved_token_id,
            "market_id": market_id,
            "size": float(size),
            "avg_price": float(entry_price),
            "current_price": float(mark_price),
            "initial_value": float(notional_usd),
            "current_value": safe_float(target_position.get("current_value")),
            "imported_at": to_iso(now),
            "reason": reason_text,
        },
    }
    verification_fields = derive_trader_order_verification(
        mode="live",
        status="executed",
        reason=reason_text,
        payload=payload_json,
    )

    order_row = TraderOrder(
        id=_new_id(),
        trader_id=trader_id,
        signal_id=None,
        decision_id=None,
        source=_MANUAL_LIVE_POSITION_SOURCE,
        strategy_key=assigned_strategy_key,
        strategy_version=None,
        market_id=market_id,
        market_question=market_question,
        direction=direction,
        mode="live",
        status="executed",
        notional_usd=float(notional_usd),
        entry_price=float(entry_price),
        effective_price=float(entry_price),
        execution_wallet_address=verification_fields.get("execution_wallet_address"),
        provider_order_id=verification_fields.get("provider_order_id"),
        provider_clob_order_id=verification_fields.get("provider_clob_order_id"),
        verification_status=str(verification_fields.get("verification_status") or TRADER_ORDER_VERIFICATION_LOCAL),
        verification_source=verification_fields.get("verification_source"),
        verification_reason=verification_fields.get("verification_reason"),
        verification_tx_hash=verification_fields.get("verification_tx_hash"),
        verified_at=now,
        edge_percent=None,
        confidence=None,
        actual_profit=None,
        reason=reason_text,
        payload_json=payload_json,
        error_message=None,
        created_at=now,
        executed_at=now,
        updated_at=now,
    )
    session.add(order_row)
    event_payload = dict(order_row.payload_json or {})
    event_payload.update(
        {
            "token_id": resolved_token_id,
            "market_id": market_id,
            "size": float(size),
        }
    )
    append_trader_order_verification_event(
        session,
        trader_order_id=str(order_row.id),
        verification_status=str(order_row.verification_status or TRADER_ORDER_VERIFICATION_LOCAL),
        source=order_row.verification_source,
        event_type="wallet_position_adopted",
        reason=order_row.verification_reason or reason_text,
        provider_order_id=order_row.provider_order_id,
        provider_clob_order_id=order_row.provider_clob_order_id,
        execution_wallet_address=order_row.execution_wallet_address,
        tx_hash=order_row.verification_tx_hash,
        payload_json=event_payload,
        created_at=now,
    )
    await _commit_with_retry(session)
    await session.refresh(order_row)

    position_inventory = await sync_trader_position_inventory(
        session,
        trader_id=trader_id,
        mode="live",
    )
    return {
        "status": "adopted",
        "trader_id": trader_id,
        "wallet_address": wallet_address,
        "strategy_key": assigned_strategy_key,
        "token_id": resolved_token_id,
        "market_id": market_id,
        "direction": direction,
        "order": _serialize_order(order_row),
        "position_inventory": position_inventory,
    }


async def sync_trader_position_inventory(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    commit: bool = True,
) -> dict[str, Any]:
    query = select(TraderOrder).where(
        TraderOrder.trader_id == trader_id,
        TraderOrder.status.in_(tuple(OPEN_ORDER_STATUSES)),
        _visible_trader_order_query_clause(),
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(TraderOrder.mode.not_in(list(_TRADER_MODES)))
        else:
            query = query.where(TraderOrder.mode == mode_key)

    order_rows = _dedupe_live_authority_rows(list((await session.execute(query)).scalars().all()))

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in order_rows:
        if not _visible_trader_order_row(row):
            continue
        mode_key = _normalize_mode_key(row.mode)
        if not _is_active_order_status(mode_key, row.status):
            continue
        identity = _position_identity_key(mode_key, row.market_id, row.direction)
        if not identity[1]:
            continue

        payload = dict(row.payload_json or {})
        position_state = payload.get("position_state")
        if not isinstance(position_state, dict):
            position_state = {}
        row_notional = safe_float(row.notional_usd, 0.0) or 0.0
        notional = _live_active_notional(row.mode, row.status, row_notional, payload, executed_at=row.executed_at)
        _, _, fill_price = _extract_live_fill_metrics(payload)
        entry_price = (
            fill_price
            if fill_price is not None and fill_price > 0
            else (safe_float(row.effective_price, 0.0) or safe_float(row.entry_price, 0.0))
        )
        mark_price = safe_float(position_state.get("last_mark_price"))
        mark_source = str(position_state.get("last_mark_source") or "")
        mark_updated_at = str(position_state.get("last_marked_at") or "")
        if notional <= 0.0:
            continue

        bucket = grouped.get(identity)
        if bucket is None:
            bucket = {
                "mode": mode_key,
                "market_id": str(row.market_id or ""),
                "market_question": row.market_question,
                "direction": str(row.direction or ""),
                "open_order_count": 0,
                "total_notional_usd": 0.0,
                "weighted_entry_numerator": 0.0,
                "weighted_entry_denominator": 0.0,
                "first_order_at": row.created_at,
                "last_order_at": row.updated_at or row.executed_at or row.created_at,
                "open_order_ids": [],
                "mark_anchor": None,
                "last_mark_price": None,
                "last_mark_source": "",
                "last_marked_at": "",
                "strategy_exit_config": None,
            }
            grouped[identity] = bucket

        if bucket.get("strategy_exit_config") is None:
            order_exit_config = payload.get("strategy_exit_config")
            if isinstance(order_exit_config, dict) and order_exit_config:
                bucket["strategy_exit_config"] = order_exit_config

        bucket["open_order_count"] = int(bucket["open_order_count"]) + 1
        bucket["total_notional_usd"] = float(bucket["total_notional_usd"]) + max(0.0, notional)
        open_order_ids = bucket.get("open_order_ids")
        if isinstance(open_order_ids, list):
            open_order_ids.append(str(row.id))
        if entry_price and entry_price > 0 and notional > 0:
            bucket["weighted_entry_numerator"] = float(bucket["weighted_entry_numerator"]) + (entry_price * notional)
            bucket["weighted_entry_denominator"] = float(bucket["weighted_entry_denominator"]) + notional
        if row.created_at and (bucket["first_order_at"] is None or row.created_at < bucket["first_order_at"]):
            bucket["first_order_at"] = row.created_at
        last_order_at = row.updated_at or row.executed_at or row.created_at
        if last_order_at and (bucket["last_order_at"] is None or last_order_at > bucket["last_order_at"]):
            bucket["last_order_at"] = last_order_at
        if mark_price is not None and mark_price > 0:
            mark_anchor = last_order_at
            previous_anchor = bucket.get("mark_anchor")
            if previous_anchor is None or (mark_anchor is not None and mark_anchor >= previous_anchor):
                bucket["mark_anchor"] = mark_anchor
                bucket["last_mark_price"] = float(mark_price)
                bucket["last_mark_source"] = mark_source
                bucket["last_marked_at"] = mark_updated_at

    existing_query = select(TraderPosition).where(TraderPosition.trader_id == trader_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            existing_query = existing_query.where(TraderPosition.mode.not_in(list(_TRADER_MODES)))
        else:
            existing_query = existing_query.where(TraderPosition.mode == mode_key)
    existing_rows = list((await session.execute(existing_query)).scalars().all())
    existing_by_identity = {
        _position_identity_key(row.mode, row.market_id, row.direction): row for row in existing_rows
    }

    now = _now()
    updates = 0
    inserts = 0
    closures = 0

    for identity, bucket in grouped.items():
        row = existing_by_identity.get(identity)
        avg_entry_price = None
        weighted_den = float(bucket.get("weighted_entry_denominator") or 0.0)
        if weighted_den > 0:
            avg_entry_price = float(bucket.get("weighted_entry_numerator") or 0.0) / weighted_den
        position_payload: dict[str, Any] = {
            "sync_source": "order_inventory",
            "open_order_ids": list(
                dict.fromkeys([str(order_id) for order_id in list(bucket.get("open_order_ids") or [])])
            ),
            "last_mark_price": bucket.get("last_mark_price"),
            "last_mark_source": str(bucket.get("last_mark_source") or ""),
            "last_marked_at": str(bucket.get("last_marked_at") or ""),
        }
        exit_config = bucket.get("strategy_exit_config")
        if isinstance(exit_config, dict) and exit_config:
            position_payload["strategy_exit_config"] = exit_config

        if row is None:
            row = TraderPosition(
                id=_new_id(),
                trader_id=trader_id,
                mode=str(bucket["mode"]),
                market_id=str(bucket["market_id"]),
                market_question=bucket.get("market_question"),
                direction=str(bucket.get("direction") or ""),
                status=ACTIVE_POSITION_STATUS,
                open_order_count=int(bucket.get("open_order_count") or 0),
                total_notional_usd=float(bucket.get("total_notional_usd") or 0.0),
                avg_entry_price=avg_entry_price,
                first_order_at=bucket.get("first_order_at"),
                last_order_at=bucket.get("last_order_at"),
                closed_at=None,
                payload_json=position_payload,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            inserts += 1
            continue

        row.market_question = bucket.get("market_question")
        row.status = ACTIVE_POSITION_STATUS
        row.open_order_count = int(bucket.get("open_order_count") or 0)
        row.total_notional_usd = float(bucket.get("total_notional_usd") or 0.0)
        row.avg_entry_price = avg_entry_price
        row.first_order_at = bucket.get("first_order_at")
        row.last_order_at = bucket.get("last_order_at")
        row.closed_at = None
        existing_payload = dict(row.payload_json or {})
        existing_payload.update(position_payload)
        row.payload_json = existing_payload
        row.updated_at = now
        updates += 1

    grouped_keys = set(grouped.keys())
    for identity, row in existing_by_identity.items():
        if identity in grouped_keys:
            continue
        if _normalize_position_status(row.status) != ACTIVE_POSITION_STATUS:
            continue
        row.status = INACTIVE_POSITION_STATUS
        row.open_order_count = 0
        row.total_notional_usd = 0.0
        row.closed_at = now
        existing_payload = dict(row.payload_json or {})
        existing_payload["sync_source"] = "order_inventory"
        existing_payload["open_order_ids"] = []
        row.payload_json = existing_payload
        row.updated_at = now
        closures += 1

    if commit:
        if inserts > 0 or updates > 0 or closures > 0:
            await _commit_with_retry(session)
        elif session.in_transaction():
            await session.rollback()

    return {
        "trader_id": trader_id,
        "mode": _normalize_mode_key(mode) if mode is not None else "all",
        "open_positions": len(grouped),
        "inserts": inserts,
        "updates": updates,
        "closures": closures,
    }


# Per-trader cap-count cache.  ``get_open_position_count_for_trader`` and
# ``get_open_order_count_for_trader`` are called every orchestrator cycle
# per trader to enforce position caps.  Both fetch full ``TraderOrder``
# rows including payload_json (~8KB/row × hundreds of rows = several MB
# transferred + Python dedup) — observed at 2-5s/call in production
# (pg_stat_statements ``mean_exec_time=2872ms calls=27``).  A 2-second
# TTL means at most one full computation per cap check per trader per
# 2 seconds; cap accuracy is preserved within this window because:
#   * Position counts only increase via the orchestrator's own writes
#     (which ALSO invalidate this cache via ``invalidate_cap_cache``)
#   * Position counts decrease via reconcile/exit which run on >5s
#     cadence — a 2s stale cap is one cycle's worth of slack at worst,
#     and the staleness biases towards "more open" so caps tighten
#     conservatively, never loosen.
_CAP_COUNT_CACHE_TTL_SECONDS = 2.0
_cap_count_cache: dict[tuple[str, str, str, str], tuple[float, int]] = {}
_cap_count_cache_lock = asyncio.Lock()


def invalidate_cap_count_cache(trader_id: str | None = None) -> None:
    """Drop cached cap counts.  Called after writes that change position/
    order state (entry submitted, exit confirmed, etc.) so the next
    cap check re-reads fresh DB state.  Pass ``None`` to drop all.
    """
    if trader_id is None:
        _cap_count_cache.clear()
        return
    keys_to_drop = [k for k in _cap_count_cache if k[0] == str(trader_id)]
    for key in keys_to_drop:
        _cap_count_cache.pop(key, None)


async def get_open_position_count_for_trader(
    session: AsyncSession,
    trader_id: str,
    mode: Optional[str] = None,
    position_cap_scope: str | None = None,
) -> int:
    scope = str(position_cap_scope or "market_direction").strip().lower()
    if scope not in {"market_direction", "market", "asset_timeframe"}:
        scope = "market_direction"

    cache_key = ("pos", str(trader_id or ""), str(mode or ""), scope)
    cached = _cap_count_cache.get(cache_key)
    if cached is not None and (time.monotonic() - cached[0]) < _CAP_COUNT_CACHE_TTL_SECONDS:
        return cached[1]

    position_query = (
        select(
            TraderPosition.mode,
            TraderPosition.market_id,
            TraderPosition.direction,
            TraderPosition.payload_json,
        )
        .where(TraderPosition.trader_id == trader_id)
        .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            position_query = position_query.where(
                func.lower(func.coalesce(TraderPosition.mode, "")).not_in(list(_TRADER_MODES))
            )
        else:
            position_query = position_query.where(func.lower(func.coalesce(TraderPosition.mode, "")) == mode_key)
    position_rows = (await session.execute(position_query)).all()
    active_position_keys: set[tuple[str, str, str]] = set()
    for row in position_rows:
        key = _position_cap_scope_key(
            position_cap_scope=scope,
            mode=(mode if mode is not None else row.mode),
            market_id=row.market_id,
            direction=row.direction,
            payload=row.payload_json,
        )
        if key is not None:
            active_position_keys.add(key)

    order_status_expr = func.lower(func.trim(func.coalesce(TraderOrder.status, "")))
    order_query = (
        select(TraderOrder)
        .where(TraderOrder.trader_id == trader_id)
        .where(order_status_expr.in_(tuple(OPEN_ORDER_STATUSES)))
        .where(_visible_trader_order_query_clause())
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            order_query = order_query.where(func.lower(func.coalesce(TraderOrder.mode, "")).not_in(list(_TRADER_MODES)))
        else:
            order_query = order_query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    order_rows = _dedupe_live_authority_rows(list((await session.execute(order_query)).scalars().all()))
    active_order_position_keys: set[tuple[str, str, str]] = set()
    for row in order_rows:
        row_mode = _normalize_mode_key(mode if mode is not None else row.mode)
        if not _is_active_order_status(row_mode, row.status):
            continue
        row_payload = dict(row.payload_json or {})
        row_notional = safe_float(row.notional_usd, 0.0) or 0.0
        active_notional = _live_active_notional(row_mode, row.status, row_notional, row_payload, executed_at=row.executed_at)
        if active_notional <= 0.0:
            continue
        key = _position_cap_scope_key(
            position_cap_scope=scope,
            mode=row_mode,
            market_id=row.market_id,
            direction=row.direction,
            payload=row_payload,
        )
        if key is not None:
            active_order_position_keys.add(key)

    result = max(len(active_position_keys), len(active_order_position_keys))
    _cap_count_cache[cache_key] = (time.monotonic(), result)
    return result


async def get_open_position_summary_for_trader(session: AsyncSession, trader_id: str) -> dict[str, int]:
    rows = (
        await session.execute(
            select(
                TraderPosition.mode,
                func.count(TraderPosition.id).label("count"),
            )
            .where(TraderPosition.trader_id == trader_id)
            .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
            .group_by(TraderPosition.mode)
        )
    ).all()

    summary = {"live": 0, "shadow": 0, "other": 0, "total": 0}
    for row in rows:
        mode_key = _normalize_mode_key(row.mode)
        count = int(row.count or 0)
        if mode_key == "live":
            summary["live"] += count
        elif mode_key == "shadow":
            summary["shadow"] += count
        else:
            summary["other"] += count
        summary["total"] += count
    return summary


async def get_open_order_count_for_trader(
    session: AsyncSession,
    trader_id: str,
    mode: Optional[str] = None,
    statuses: Optional[set[str]] = None,
) -> int:
    # Same per-cycle-per-trader cap cache as
    # ``get_open_position_count_for_trader`` — see comment there.
    # The ``statuses`` override path (rare) bypasses the cache because
    # cache key would explode for arbitrary status sets.
    use_cache = statuses is None
    cache_key = ("ord", str(trader_id or ""), str(mode or ""), "default")
    if use_cache:
        cached = _cap_count_cache.get(cache_key)
        if cached is not None and (time.monotonic() - cached[0]) < _CAP_COUNT_CACHE_TTL_SECONDS:
            return cached[1]

    effective_statuses = statuses if statuses is not None else UNFILLED_ORDER_STATUSES
    status_key_expr = func.lower(func.trim(func.coalesce(TraderOrder.status, "")))
    # NOTE: do NOT ``defer(TraderOrder.payload_json)`` here.  The earlier
    # comment ("the dedup logic doesn't read payload_json") was wrong —
    # ``_live_order_authority_key_from_row`` (called by
    # ``_dedupe_live_authority_rows`` below) DOES read ``payload_json``
    # for rows where the direct ``provider_clob_order_id`` /
    # ``provider_order_id`` columns are NULL.  With defer, that access
    # triggers a lazy column-load which bypasses SQLAlchemy's async
    # greenlet bridge and raises ``MissingGreenlet``, breaking
    # reconciliation for every cycle on every trader that has even one
    # such row.  Eager-loading the column adds a few KB per row to the
    # transfer (filtered to one trader's open orders, typically <50
    # rows) — a tiny price for not silently breaking the lifecycle.
    query = (
        select(TraderOrder)
        .where(TraderOrder.trader_id == trader_id)
        .where(status_key_expr.in_(tuple(effective_statuses)))
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = _dedupe_live_authority_rows(list((await session.execute(query)).scalars().all()))
    result = len(rows)
    if use_cache:
        _cap_count_cache[cache_key] = (time.monotonic(), result)
    return result


async def get_open_order_summary_for_trader(session: AsyncSession, trader_id: str) -> dict[str, int]:
    summary = {"live": 0, "shadow": 0, "other": 0, "total": 0}
    rows = _dedupe_live_authority_rows(
        list(
            (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .where(func.lower(func.trim(func.coalesce(TraderOrder.status, ""))).in_(tuple(UNFILLED_ORDER_STATUSES)))
                )
            )
            .scalars()
            .all()
        )
    )
    for row in rows:
        mode = _normalize_mode_key(row.mode)
        count = 1
        if mode == "live":
            summary["live"] += count
        elif mode == "shadow":
            summary["shadow"] += count
        else:
            summary["other"] += count
        summary["total"] += count
    return summary

async def get_pending_live_exit_summary_for_trader(
    session: AsyncSession,
    trader_id: str,
    mode: str = "live",
    terminal_statuses: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    mode_key = _normalize_mode_key(mode)
    if mode_key != "live":
        return {
            "count": 0,
            "order_ids": [],
            "market_ids": [],
            "signal_ids": [],
            "statuses": {},
            "terminal_statuses": list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"]),
            "identities": [],
            "identity_keys": [],
        }

    configured_terminal_statuses = _normalize_pending_live_exit_terminal_statuses(
        list(terminal_statuses) if isinstance(terminal_statuses, (list, set, tuple)) else None
    )
    terminal_status_set = set(configured_terminal_statuses)

    status_key_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    rows = _dedupe_live_authority_rows(
        list(
            (
                await session.execute(
                    select(TraderOrder)
                    .where(TraderOrder.trader_id == trader_id)
                    .where(func.lower(func.coalesce(TraderOrder.mode, "")) == "live")
                    .where(status_key_expr.in_(tuple(_active_statuses_for_mode("live"))))
                    .where(_visible_trader_order_query_clause())
                )
            )
            .scalars()
            .all()
        )
    )

    order_ids: list[str] = []
    market_ids: set[str] = set()
    signal_ids: set[str] = set()
    statuses: dict[str, int] = {}
    identities: list[dict[str, Any]] = []
    identity_keys: set[str] = set()
    for row in rows:
        payload = dict(row.payload_json or {})
        pending_live_exit = payload.get("pending_live_exit")
        if not isinstance(pending_live_exit, dict):
            continue
        order_id = str(row.id or "").strip()
        market_id = str(row.market_id or "").strip()
        direction = str(row.direction or "").strip().lower()
        signal_id = str(row.signal_id or "").strip()
        status_key = _normalize_status_key(pending_live_exit.get("status")) or "unknown"
        if status_key in terminal_status_set:
            continue
        statuses[status_key] = int(statuses.get(status_key, 0)) + 1
        if order_id:
            order_ids.append(order_id)
        if market_id:
            market_ids.add(market_id)
        if signal_id:
            signal_ids.add(signal_id)
        identities.append(
            {
                "order_id": order_id or None,
                "market_id": market_id or None,
                "direction": direction or None,
                "signal_id": signal_id or None,
                "status": status_key,
            }
        )
        if market_id and direction:
            identity_keys.add(f"{market_id}|{direction}|{signal_id or '*'}")

    return {
        "count": len(order_ids),
        "order_ids": order_ids,
        "market_ids": sorted(market_ids),
        "signal_ids": sorted(signal_ids),
        "statuses": statuses,
        "terminal_statuses": configured_terminal_statuses,
        "identities": identities,
        "identity_keys": sorted(identity_keys),
    }


def _order_has_fill(row: TraderOrder) -> bool:
    mode_key = _normalize_mode_key(row.mode)
    status_key = _normalize_status_key(row.status)
    payload = dict(row.payload_json or {})
    row_notional = max(0.0, abs(safe_float(row.notional_usd, 0.0) or 0.0))

    if mode_key in {"live", "shadow"}:
        filled_notional_usd, filled_shares, _ = _extract_live_fill_metrics(payload)
        if filled_notional_usd > 0.0 or filled_shares > 0.0:
            return True
        return status_key == "executed" and row_notional > 0.0

    return row_notional > 0.0


async def cleanup_trader_open_orders(
    session: AsyncSession,
    *,
    trader_id: str,
    scope: str = "shadow",
    max_age_hours: Optional[int] = None,
    max_age_seconds: Optional[float] = None,
    source: Optional[str] = None,
    require_unfilled: bool = False,
    dry_run: bool = False,
    target_status: str = "cancelled",
    reason: Optional[str] = None,
    attempt_live_taker_rescue: bool = False,
    live_taker_rescue_price_bps: Optional[float] = None,
    live_taker_rescue_time_in_force: str = DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE,
) -> dict[str, Any]:
    scope_key = str(scope or "shadow").strip().lower()
    if scope_key == "all":
        allowed_modes = {"shadow", "live", "other"}
    elif scope_key in {"shadow", "live"}:
        allowed_modes = {scope_key}
    else:
        raise ValueError("scope must be one of: shadow, live, all")

    query = select(TraderOrder).where(TraderOrder.trader_id == trader_id).where(_visible_trader_order_query_clause())
    rows = list((await session.execute(query)).scalars().all())
    source_filter = normalize_source_key(source) if source is not None else ""
    cutoff: Optional[datetime] = None
    if max_age_seconds is not None:
        cutoff = _now() - timedelta(seconds=max(1.0, float(max_age_seconds)))
    elif max_age_hours is not None:
        cutoff = _now() - timedelta(hours=max(1, int(max_age_hours)))

    candidates: list[TraderOrder] = []
    for row in rows:
        mode_key = _normalize_mode_key(row.mode)
        if mode_key not in allowed_modes:
            continue
        if source_filter and normalize_source_key(row.source) != source_filter:
            continue

        status_key = _normalize_status_key(row.status)
        if status_key not in CLEANUP_ELIGIBLE_ORDER_STATUSES:
            continue
        if require_unfilled and _order_has_fill(row):
            continue

        if cutoff is not None:
            if require_unfilled:
                age_anchor = row.created_at or row.updated_at or row.executed_at
            else:
                age_anchor = row.executed_at or row.updated_at or row.created_at
            if age_anchor is None or age_anchor > cutoff:
                continue

        candidates.append(row)

    # Cap candidates per call to bound the per-cycle wall time on the
    # per-order live-cancel HTTP fan-out (8s timeout × N orders).
    # Sort oldest-first by the same anchor we filtered on, so the most
    # overdue cancellations get the budget; the rest drain on the next
    # cycle.  See _OPEN_ORDER_CLEANUP_MAX_CANDIDATES_PER_CALL above.
    candidates_total = len(candidates)
    if candidates_total > _OPEN_ORDER_CLEANUP_MAX_CANDIDATES_PER_CALL:
        def _age_anchor(row: TraderOrder) -> datetime:
            if require_unfilled:
                anchor = row.created_at or row.updated_at or row.executed_at
            else:
                anchor = row.executed_at or row.updated_at or row.created_at
            return anchor or datetime.max
        candidates.sort(key=_age_anchor)  # oldest first
        candidates = candidates[:_OPEN_ORDER_CLEANUP_MAX_CANDIDATES_PER_CALL]
    candidates_deferred = candidates_total - len(candidates)

    mode_breakdown = {"shadow": 0, "live": 0, "other": 0}
    status_breakdown: dict[str, int] = {}
    for row in candidates:
        mode_key = _normalize_mode_key(row.mode)
        status_key = _normalize_status_key(row.status)
        mode_breakdown[mode_key] = int(mode_breakdown.get(mode_key, 0)) + 1
        status_breakdown[status_key] = int(status_breakdown.get(status_key, 0)) + 1

    updated = 0
    provider_cancel_attempted = 0
    provider_cancelled = 0
    provider_cancel_failed = 0
    taker_rescue_attempted = 0
    taker_rescue_succeeded = 0
    taker_rescue_failed = 0
    execution_order_updates = 0
    execution_leg_updates = 0
    execution_session_updates = 0
    if not dry_run and candidates:
        now = _now()
        note_reason = str(reason or "manual_position_cleanup").strip()
        rescue_price_bps = safe_float(live_taker_rescue_price_bps, DEFAULT_TIMEOUT_TAKER_RESCUE_PRICE_BPS)
        rescue_price_bps = max(0.0, float(rescue_price_bps or 0.0))
        rescue_time_in_force = str(live_taker_rescue_time_in_force or DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE).strip().upper()
        if rescue_time_in_force not in {"IOC", "FOK", "GTC"}:
            rescue_time_in_force = DEFAULT_TIMEOUT_TAKER_RESCUE_TIME_IN_FORCE
        allow_timeout_taker_rescue = bool(attempt_live_taker_rescue and require_unfilled and _cleanup_timeout_reason(note_reason))
        candidate_ids = [str(row.id) for row in candidates]
        execution_order_rows = list(
            (
                await session.execute(
                    select(ExecutionSessionOrder)
                    .where(ExecutionSessionOrder.trader_order_id.in_(candidate_ids))
                    .order_by(desc(ExecutionSessionOrder.created_at))
                )
            )
            .scalars()
            .all()
        )
        execution_order_by_trader_order: dict[str, ExecutionSessionOrder] = {}
        for execution_order in execution_order_rows:
            key = str(execution_order.trader_order_id or "").strip()
            if not key or key in execution_order_by_trader_order:
                continue
            execution_order_by_trader_order[key] = execution_order
        touched_leg_ids: set[str] = set()
        touched_session_ids: set[str] = set()

        for row in candidates:
            mode_key = _normalize_mode_key(row.mode)
            previous_status = str(row.status or "")
            existing_payload = dict(row.payload_json or {})
            execution_order = execution_order_by_trader_order.get(str(row.id))

            if mode_key == "live" and not _is_active_order_status(mode_key, target_status):
                provider_cancel_attempted += 1
                provider_order_id, provider_clob_order_id = _resolve_provider_order_ids(
                    order_payload=existing_payload,
                    session_order=execution_order,
                )
                cancel_targets: list[str] = []
                if provider_clob_order_id:
                    cancel_targets.append(provider_clob_order_id)
                if provider_order_id and provider_order_id not in cancel_targets:
                    cancel_targets.append(provider_order_id)
                if not cancel_targets:
                    provider_cancel_failed += 1
                    raise ValueError(
                        f"Cannot cancel live order {row.id}: missing provider order identifiers in payload/session state."
                    )

                provider_cancel_success = False
                last_target = ""
                cancel_failure_reason = ""
                async with release_conn(session):
                    for target in cancel_targets:
                        last_target = target
                        try:
                            cancelled = await asyncio.wait_for(
                                live_execution_service.cancel_order(target),
                                timeout=LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            cancelled = False
                            cancel_failure_reason = f"timeout after {LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS:.1f}s"
                        except Exception as exc:
                            cancelled = False
                            cancel_failure_reason = f"error: {exc}"
                        if cancelled:
                            provider_cancel_success = True
                            cancel_failure_reason = ""
                            break
                        if not cancel_failure_reason:
                            cancel_failure_reason = "provider returned unsuccessful cancellation"

                existing_payload["provider_cancel"] = {
                    "attempted_at": to_iso(now),
                    "provider_order_id": provider_order_id,
                    "provider_clob_order_id": provider_clob_order_id,
                    "targets": cancel_targets,
                    "success": provider_cancel_success,
                    "failure_reason": cancel_failure_reason or None,
                    "timeout_seconds": LIVE_PROVIDER_CANCEL_TIMEOUT_SECONDS,
                }
                if not provider_cancel_success:
                    provider_cancel_failed += 1
                    raise ValueError(
                        f"Provider cancellation failed for live order {row.id} "
                        f"(target={last_target or 'unknown'}, reason={cancel_failure_reason or 'unknown'})."
                    )
                provider_cancelled += 1

                if allow_timeout_taker_rescue and source_filter == "crypto":
                    taker_rescue_attempted += 1
                    rescue = await _attempt_timeout_taker_rescue(
                        row=row,
                        order_payload=existing_payload,
                        session_order=execution_order,
                        rescue_price_bps=rescue_price_bps,
                        rescue_time_in_force=rescue_time_in_force,
                    )
                    existing_payload["timeout_taker_rescue"] = {
                        **rescue,
                        "attempted_at": to_iso(now),
                    }
                    if bool(rescue.get("success")):
                        taker_rescue_succeeded += 1
                        rescue_payload = dict(rescue.get("payload") or {})
                        rescue_status = _normalize_status_key(rescue.get("status"))
                        row.status = rescue_status if rescue_status in {"executed", "open", "submitted"} else "failed"
                        row.updated_at = now
                        if row.status == "failed":
                            import services.trader_hot_state as _hs
                            _hs.record_order_cancelled(
                                trader_id=str(row.trader_id or ""),
                                mode=str(row.mode or ""),
                                order_id=str(row.id or ""),
                                market_id=str(row.market_id or ""),
                                source=str(row.source or ""),
                                copy_source_wallet=_extract_copy_source_wallet_from_payload(existing_payload),
                            )
                        existing_payload["cleanup"] = {
                            "previous_status": previous_status,
                            "target_status": row.status,
                            "reason": "timeout_taker_rescue",
                            "performed_at": to_iso(now),
                        }
                        if rescue_payload:
                            existing_payload.update(
                                {
                                    "order_id": str(rescue.get("order_id") or rescue_payload.get("order_id") or ""),
                                    "clob_order_id": str(rescue.get("clob_order_id") or rescue_payload.get("clob_order_id") or ""),
                                    "provider_order_id": str(rescue.get("order_id") or rescue_payload.get("order_id") or ""),
                                    "provider_clob_order_id": str(
                                        rescue.get("clob_order_id") or rescue_payload.get("clob_order_id") or ""
                                    ),
                                    "effective_price": safe_float(rescue.get("effective_price"), None),
                                    "filled_size": safe_float(rescue_payload.get("filled_size"), 0.0) or 0.0,
                                    "average_fill_price": safe_float(rescue_payload.get("average_fill_price"), None),
                                    "submission": "timeout_taker_rescue",
                                    "post_only": False,
                                    "time_in_force": rescue_time_in_force,
                                }
                            )
                        row.payload_json = existing_payload
                        if row.reason:
                            row.reason = f"{row.reason} | rescue:timeout_taker_conversion"
                        else:
                            row.reason = "rescue:timeout_taker_conversion"

                        if execution_order is not None:
                            execution_order.status = str(row.status)
                            execution_order.updated_at = now
                            execution_order.provider_order_id = str(rescue.get("order_id") or "") or None
                            execution_order.provider_clob_order_id = str(rescue.get("clob_order_id") or "") or None
                            execution_payload = dict(execution_order.payload_json or {})
                            execution_payload["timeout_taker_rescue"] = {
                                **rescue,
                                "attempted_at": to_iso(now),
                            }
                            execution_order.payload_json = execution_payload
                            execution_order.error_message = None
                            execution_order.reason = "timeout_taker_rescue"
                            execution_order_updates += 1
                        updated += 1
                        continue

                    taker_rescue_failed += 1

            row.status = target_status
            row.updated_at = now
            existing_payload["cleanup"] = {
                "previous_status": previous_status,
                "target_status": target_status,
                "reason": note_reason,
                "performed_at": to_iso(now),
            }
            row.payload_json = existing_payload
            if note_reason:
                if row.reason:
                    row.reason = f"{row.reason} | cleanup:{note_reason}"
                else:
                    row.reason = f"cleanup:{note_reason}"
            import services.trader_hot_state as _hs
            _hs.record_order_cancelled(
                trader_id=str(row.trader_id or ""),
                mode=str(row.mode or ""),
                order_id=str(row.id or ""),
                market_id=str(row.market_id or ""),
                source=str(row.source or ""),
                copy_source_wallet=_extract_copy_source_wallet_from_payload(existing_payload),
            )

            if execution_order is not None and not _is_active_order_status(mode_key, target_status):
                execution_order.status = str(target_status)
                execution_order.updated_at = now
                if note_reason:
                    if execution_order.reason:
                        execution_order.reason = f"{execution_order.reason} | cleanup:{note_reason}"
                    else:
                        execution_order.reason = f"cleanup:{note_reason}"
                    execution_order.error_message = note_reason
                execution_payload = dict(execution_order.payload_json or {})
                execution_payload["cleanup"] = {
                    "target_status": str(target_status),
                    "reason": note_reason,
                    "performed_at": to_iso(now),
                }
                execution_order.payload_json = execution_payload
                execution_order_updates += 1
                leg_id = str(execution_order.leg_id or "").strip()
                if leg_id:
                    touched_leg_ids.add(leg_id)
                session_id = str(execution_order.session_id or "").strip()
                if session_id:
                    touched_session_ids.add(session_id)
            updated += 1

        if touched_leg_ids and not _is_active_order_status(scope_key, target_status):
            leg_target_status = str(target_status).strip().lower()
            if leg_target_status not in {"cancelled", "failed", "expired"}:
                leg_target_status = "cancelled"
            for leg_id in sorted(touched_leg_ids):
                leg_row = await session.get(ExecutionSessionLeg, leg_id)
                if leg_row is None:
                    continue
                leg_status_key = _normalize_status_key(leg_row.status)
                if leg_status_key in {"completed", "failed", "cancelled", "expired"}:
                    continue
                leg_row.status = leg_target_status
                if note_reason:
                    leg_row.last_error = note_reason
                leg_row.updated_at = now
                execution_leg_updates += 1
                session_id = str(leg_row.session_id or "").strip()
                if session_id:
                    touched_session_ids.add(session_id)

        if touched_session_ids and not _is_active_order_status(scope_key, target_status):
            for session_id in sorted(touched_session_ids):
                if not session_id:
                    continue
                session_row = await _refresh_execution_session_rollups(session, session_id=session_id)
                if session_row is None:
                    continue
                session_status_key = _normalize_status_key(session_row.status)
                if session_status_key not in ACTIVE_EXECUTION_SESSION_STATUSES:
                    continue
                legs_open = int(session_row.legs_open or 0)
                legs_failed = int(session_row.legs_failed or 0)
                legs_completed = int(session_row.legs_completed or 0)
                if legs_open <= 0 and legs_failed > 0 and legs_completed == 0:
                    session_row.status = "failed"
                    session_row.error_message = (
                        f"Order cleanup transitioned all active legs to {target_status} ({note_reason})."
                    )
                    session_row.completed_at = now
                    session_row.updated_at = now
                    execution_session_updates += 1
        await _commit_with_retry(session)
        await sync_trader_position_inventory(session, trader_id=trader_id)

    return {
        "trader_id": trader_id,
        "scope": scope_key,
        "max_age_hours": max_age_hours,
        "max_age_seconds": max_age_seconds,
        "source": source_filter or None,
        "require_unfilled": bool(require_unfilled),
        "dry_run": bool(dry_run),
        "target_status": target_status,
        "matched": len(candidates),
        "candidates_total": candidates_total,
        "candidates_deferred": candidates_deferred,
        "updated": updated,
        "by_mode": mode_breakdown,
        "by_status": status_breakdown,
        "provider_cancel_attempted": provider_cancel_attempted,
        "provider_cancelled": provider_cancelled,
        "provider_cancel_failed": provider_cancel_failed,
        "taker_rescue_attempted": taker_rescue_attempted,
        "taker_rescue_succeeded": taker_rescue_succeeded,
        "taker_rescue_failed": taker_rescue_failed,
        "execution_order_updates": execution_order_updates,
        "execution_leg_updates": execution_leg_updates,
        "execution_session_updates": execution_session_updates,
    }


async def get_market_exposure(session: AsyncSession, market_id: str, mode: Optional[str] = None) -> float:
    query = select(TraderOrder).where(TraderOrder.market_id == market_id).where(_visible_trader_order_query_clause())
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    rows = list((await session.execute(query)).scalars().all())
    return float(sum(_active_order_notional_for_metrics(row) for row in rows))


async def get_trader_source_exposure(
    session: AsyncSession,
    *,
    trader_id: str,
    source: str,
    mode: Optional[str] = None,
) -> float:
    source_key = str(source or "").strip().lower()
    if not source_key:
        return 0.0

    query = select(TraderOrder).where(
        TraderOrder.trader_id == trader_id,
        func.lower(func.coalesce(TraderOrder.source, "")) == source_key,
    ).where(_visible_trader_order_query_clause())
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    rows = list((await session.execute(query)).scalars().all())
    return float(sum(_active_order_notional_for_metrics(row) for row in rows))


async def get_trader_copy_leader_exposure(
    session: AsyncSession,
    *,
    trader_id: str,
    source_wallet: str,
    mode: Optional[str] = None,
) -> float:
    normalized_wallet = StrategySDK.normalize_trader_wallet(source_wallet)
    if not normalized_wallet:
        return 0.0

    query = select(TraderOrder).where(
        TraderOrder.trader_id == trader_id,
        func.lower(func.coalesce(TraderOrder.source, "")) == "traders",
    ).where(_visible_trader_order_query_clause())
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = list((await session.execute(query)).scalars().all())
    exposure = 0.0
    for row in rows:
        row_payload = dict(row.payload_json or {})
        row_source_wallet = _extract_copy_source_wallet_from_payload(row_payload)
        if row_source_wallet != normalized_wallet:
            continue
        exposure += _active_order_notional_for_metrics(row)
    return float(exposure)


async def get_trader_copy_analytics(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    limit: int = 2000,
    leader_limit: int = 20,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 2000), 10_000))
    query = (
        select(TraderOrder)
        .where(TraderOrder.trader_id == trader_id)
        .where(_visible_trader_order_query_clause())
        .order_by(desc(TraderOrder.created_at), desc(TraderOrder.id))
        .limit(capped_limit)
    )
    mode_key: str | None = None
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = list((await session.execute(query)).scalars().all())
    filtered_rows: list[tuple[TraderOrder, dict[str, Any], str]] = []
    for row in rows:
        if str(row.source or "").strip().lower() != "traders":
            continue
        row_payload = dict(row.payload_json or {})
        source_wallet = _extract_copy_source_wallet_from_payload(row_payload)
        strategy_key = str(row.strategy_key or "").strip().lower()
        if not source_wallet and strategy_key != "traders_copy_trade":
            continue
        filtered_rows.append((row, row_payload, source_wallet or "unknown"))

    total_orders = len(filtered_rows)
    total_notional_usd = 0.0
    open_exposure_usd = 0.0
    realized_orders = 0
    realized_pnl_usd = 0.0
    slippage_sum = 0.0
    slippage_count = 0
    adverse_slippage_sum = 0.0
    adverse_slippage_count = 0
    latency_sum = 0.0
    latency_count = 0

    per_leader: dict[str, dict[str, Any]] = {}
    for row, row_payload, leader_wallet in filtered_rows:
        status_key = _normalize_status_key(row.status)
        row_mode = _normalize_mode_key(row.mode)
        row_notional_usd = abs(safe_float(row.notional_usd, 0.0))
        row_active_exposure = _active_order_notional_for_metrics(row)
        total_notional_usd += row_notional_usd
        open_exposure_usd += row_active_exposure

        if status_key in REALIZED_ORDER_STATUSES:
            realized_orders += 1
            realized_pnl_usd += safe_float(row.actual_profit, 0.0)

        copy_attribution = row_payload.get("copy_attribution")
        copy_attribution = copy_attribution if isinstance(copy_attribution, dict) else {}
        row_slippage_bps = safe_float(copy_attribution.get("slippage_bps"), None)
        row_adverse_slippage_bps = safe_float(copy_attribution.get("adverse_slippage_bps"), None)
        row_copy_latency_ms = safe_float(copy_attribution.get("copy_latency_ms"), None)

        if row_slippage_bps is not None:
            slippage_sum += row_slippage_bps
            slippage_count += 1
        if row_adverse_slippage_bps is not None:
            adverse_slippage_sum += row_adverse_slippage_bps
            adverse_slippage_count += 1
        if row_copy_latency_ms is not None:
            latency_sum += row_copy_latency_ms
            latency_count += 1

        leader = per_leader.get(leader_wallet)
        if leader is None:
            leader = {
                "source_wallet": leader_wallet,
                "orders": 0,
                "active_orders": 0,
                "realized_orders": 0,
                "realized_pnl_usd": 0.0,
                "gross_notional_usd": 0.0,
                "open_exposure_usd": 0.0,
                "slippage_sum": 0.0,
                "slippage_count": 0,
                "adverse_slippage_sum": 0.0,
                "adverse_slippage_count": 0,
                "latency_sum": 0.0,
                "latency_count": 0,
            }
            per_leader[leader_wallet] = leader

        leader["orders"] += 1
        leader["gross_notional_usd"] += row_notional_usd
        leader["open_exposure_usd"] += row_active_exposure
        if _is_active_order_status(row_mode, row.status):
            leader["active_orders"] += 1
        if status_key in REALIZED_ORDER_STATUSES:
            leader["realized_orders"] += 1
            leader["realized_pnl_usd"] += safe_float(row.actual_profit, 0.0)
        if row_slippage_bps is not None:
            leader["slippage_sum"] += row_slippage_bps
            leader["slippage_count"] += 1
        if row_adverse_slippage_bps is not None:
            leader["adverse_slippage_sum"] += row_adverse_slippage_bps
            leader["adverse_slippage_count"] += 1
        if row_copy_latency_ms is not None:
            leader["latency_sum"] += row_copy_latency_ms
            leader["latency_count"] += 1

    resolved_orders = sum(1 for row, _, _ in filtered_rows if _normalize_status_key(row.status) in REALIZED_ORDER_STATUSES)
    win_orders = sum(1 for row, _, _ in filtered_rows if _normalize_status_key(row.status) in REALIZED_WIN_ORDER_STATUSES)
    loss_orders = sum(1 for row, _, _ in filtered_rows if _normalize_status_key(row.status) in REALIZED_LOSS_ORDER_STATUSES)
    win_rate_pct = (win_orders / resolved_orders * 100.0) if resolved_orders > 0 else None

    sorted_leaders = sorted(
        per_leader.values(),
        key=lambda item: (
            -float(item.get("gross_notional_usd") or 0.0),
            -int(item.get("orders") or 0),
            str(item.get("source_wallet") or ""),
        ),
    )
    leader_rows: list[dict[str, Any]] = []
    for leader in sorted_leaders[: max(1, min(int(leader_limit or 20), 500))]:
        leader_rows.append(
            {
                "source_wallet": leader["source_wallet"],
                "orders": int(leader["orders"]),
                "active_orders": int(leader["active_orders"]),
                "realized_orders": int(leader["realized_orders"]),
                "realized_pnl_usd": float(leader["realized_pnl_usd"]),
                "gross_notional_usd": float(leader["gross_notional_usd"]),
                "open_exposure_usd": float(leader["open_exposure_usd"]),
                "avg_slippage_bps": (
                    float(leader["slippage_sum"] / leader["slippage_count"])
                    if int(leader["slippage_count"] or 0) > 0
                    else None
                ),
                "avg_adverse_slippage_bps": (
                    float(leader["adverse_slippage_sum"] / leader["adverse_slippage_count"])
                    if int(leader["adverse_slippage_count"] or 0) > 0
                    else None
                ),
                "avg_copy_latency_ms": (
                    float(leader["latency_sum"] / leader["latency_count"])
                    if int(leader["latency_count"] or 0) > 0
                    else None
                ),
            }
        )

    return {
        "trader_id": trader_id,
        "mode": mode_key or "all",
        "as_of": to_iso(_now()),
        "sample_size": total_orders,
        "scanned_orders": len(rows),
        "summary": {
            "total_orders": total_orders,
            "resolved_orders": resolved_orders,
            "win_orders": win_orders,
            "loss_orders": loss_orders,
            "win_rate_pct": win_rate_pct,
            "realized_pnl_usd": float(realized_pnl_usd),
            "gross_notional_usd": float(total_notional_usd),
            "open_exposure_usd": float(open_exposure_usd),
            "avg_slippage_bps": (float(slippage_sum / slippage_count) if slippage_count > 0 else None),
            "avg_adverse_slippage_bps": (
                float(adverse_slippage_sum / adverse_slippage_count) if adverse_slippage_count > 0 else None
            ),
            "avg_copy_latency_ms": (float(latency_sum / latency_count) if latency_count > 0 else None),
            "distinct_leaders": len(per_leader),
        },
        "leaders": leader_rows,
    }


async def get_gross_exposure(session: AsyncSession, mode: Optional[str] = None) -> float:
    query = select(
        TraderOrder.mode,
        TraderOrder.status,
        TraderOrder.notional_usd,
        TraderOrder.payload_json,
    ).where(_visible_trader_order_query_clause())
    active_statuses = OPEN_ORDER_STATUSES
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        active_statuses = _active_statuses_for_mode(mode_key)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    query = query.where(TraderOrder.status.in_(tuple(sorted(active_statuses))))
    rows = (await session.execute(query)).all()
    live_order_exposure = 0.0
    other_order_exposure = 0.0
    for row in rows:
        exposure = _active_order_notional_for_metrics_fields(
            mode=row.mode,
            status=row.status,
            notional_usd=row.notional_usd,
            payload_json=row.payload_json,
        )
        if _normalize_mode_key(row.mode) == "live":
            live_order_exposure += exposure
        else:
            other_order_exposure += exposure

    wallet_live_exposure = 0.0
    if mode is None or _normalize_mode_key(mode) == "live":
        wallet_position_rows = (
            await session.execute(
                select(
                    LiveTradingPosition.size,
                    LiveTradingPosition.average_cost,
                    LiveTradingPosition.current_price,
                ).where(LiveTradingPosition.size > 0)
            )
        ).all()
        for row in wallet_position_rows:
            size = max(0.0, safe_float(row.size, 0.0) or 0.0)
            average_cost = safe_float(row.average_cost, 0.0) or 0.0
            current_price = safe_float(row.current_price, 0.0) or 0.0
            price = average_cost if average_cost > 0.0 else current_price
            if size > 0.0 and price > 0.0:
                wallet_live_exposure += size * price

    if mode is not None and _normalize_mode_key(mode) == "live":
        return float(max(live_order_exposure, wallet_live_exposure))
    return float(other_order_exposure + max(live_order_exposure, wallet_live_exposure))


async def get_realized_pnl(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    mode: Optional[str] = None,
    since: Optional[datetime] = None,
) -> float:
    # Only sum actual_profit for orders whose realized P&L was verified
    # against Polymarket. Truly-verified = wallet_activity (matched to
    # actual on-chain trade by tx_hash) or venue_fill (CLOB confirmed
    # filled with real average_fill_price). Everything else is inferred
    # — wallet_position guesses close_price from currentPrice / wallet
    # aggregate, which conflates manual user trades with bot fills and
    # produces phantom P&L (the Flash Crash +$85 the user caught).
    query = select(func.coalesce(func.sum(TraderOrder.actual_profit), 0.0)).where(
        TraderOrder.status.in_(tuple(REALIZED_ORDER_STATUSES)),
        _visible_trader_order_query_clause(),
        func.lower(func.coalesce(TraderOrder.verification_status, "")).notin_(
            (
                "summary_only",
                "local",
                "disputed",
                "wallet_position",
                "venue_order",
            )
        ),
    )
    if trader_id:
        query = query.where(TraderOrder.trader_id == trader_id)
    if since is not None:
        query = query.where(TraderOrder.updated_at >= since)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    return float((await session.execute(query)).scalar() or 0.0)


# Per-cycle TTL caches for setup-phase aggregations.  These run on
# EVERY orchestrator cycle for every trader (~6 traders × multiple
# cycles/second) and previously contributed ~1-3s of DB time to the
# 5s ``setup`` stage observed in production.  They aggregate over
# ``trader_orders`` which only changes when an order resolves; a
# 30s TTL is safe (no decision logic depends on sub-30s freshness)
# and the next-tick re-evaluation after a fill is bounded by the
# TTL window.  Invalidation is best-effort via
# ``invalidate_realized_pnl_cache`` from order-completion paths.
_DAILY_PNL_CACHE_TTL_SECONDS = 30.0
_LOSS_STREAK_CACHE_TTL_SECONDS = 30.0
_LAST_LOSS_CACHE_TTL_SECONDS = 30.0
# Stale-while-revalidate: once an entry is older than this fraction of
# its TTL, the next reader returns the cached value AND kicks off a
# background refresh.  By the time the entry would have expired the
# cache has already been replaced — readers never block on the DB.
_REALIZED_PNL_REFRESH_FRACTION = 0.70
_daily_pnl_cache: dict[tuple, tuple[float, float]] = {}
_loss_streak_cache: dict[tuple, tuple[float, int]] = {}
_last_loss_cache: dict[tuple, tuple[float, Optional[datetime]]] = {}
# In-flight bg refresh tracking — one task per (cache_name, key) so
# multiple cycles don't pile up duplicate refreshes for the same
# trader.  Keys clear themselves via a done callback.
_realized_pnl_refresh_tasks: dict[tuple, asyncio.Task] = {}


def invalidate_realized_pnl_cache(
    trader_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    """Drop cached daily PnL / loss-streak / last-loss for one trader.

    Call from any code path that flips a ``trader_order.status`` into
    a terminal/realized state so the next read sees fresh values
    rather than waiting up to ``_DAILY_PNL_CACHE_TTL_SECONDS``.
    """
    if trader_id is None and mode is None:
        _daily_pnl_cache.clear()
        _loss_streak_cache.clear()
        _last_loss_cache.clear()
        return
    for cache in (_daily_pnl_cache, _loss_streak_cache, _last_loss_cache):
        for key in [k for k in cache if (trader_id is None or k[0] == trader_id) and (mode is None or k[1] == mode)]:
            cache.pop(key, None)


def _schedule_realized_pnl_bg_refresh(
    *,
    cache_name: str,
    cache_key: tuple,
    coro_factory: Callable[[], Awaitable[Any]],
) -> None:
    """Fire-and-forget refresh — readers never await this.

    Soft-fail: if no asyncio loop is running (sync entry point / tests)
    or task creation fails, we silently skip; the cached value is still
    returned and the next read past TTL pays the DB cost normally.
    """
    task_key = (cache_name, cache_key)
    existing = _realized_pnl_refresh_tasks.get(task_key)
    if existing is not None and not existing.done():
        return  # one refresh per key in flight at a time
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _runner() -> None:
        try:
            await coro_factory()
        except Exception:
            # Refresh failures must not crash the loop; the stale value
            # remains cached and the next reader will retry.
            pass

    try:
        task = loop.create_task(_runner(), name=f"realized-pnl-bg-refresh-{cache_name}")
    except RuntimeError:
        return
    _realized_pnl_refresh_tasks[task_key] = task
    task.add_done_callback(lambda _t, k=task_key: _realized_pnl_refresh_tasks.pop(k, None))


async def _refresh_daily_pnl(
    *,
    trader_id: Optional[str],
    mode: Optional[str],
    cache_key: tuple,
) -> None:
    async with AsyncSessionLocal() as session:
        today_start = _now().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await get_realized_pnl(
            session,
            trader_id=trader_id,
            mode=mode,
            since=today_start,
        )
    _daily_pnl_cache[cache_key] = (time.monotonic(), result)


async def _query_loss_streak(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str],
    limit: int,
    since: Optional[datetime] = None,
) -> int:
    query = (
        select(TraderOrder.status, TraderOrder.updated_at)
        .where(TraderOrder.trader_id == trader_id)
        .where(TraderOrder.status.in_(tuple(REALIZED_ORDER_STATUSES)))
        .where(_visible_trader_order_query_clause())
        .order_by(desc(TraderOrder.updated_at), desc(TraderOrder.id))
        .limit(max(1, min(int(limit or 100), 1000)))
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    if since is not None:
        since_utc = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since.astimezone(timezone.utc)
        query = query.where(TraderOrder.updated_at >= since_utc)
    rows = (await session.execute(query)).all()
    losses = 0
    for row in rows:
        status = _normalize_status_key(row.status)
        if status in REALIZED_LOSS_ORDER_STATUSES:
            losses += 1
            continue
        if status in REALIZED_WIN_ORDER_STATUSES:
            break
    return losses


async def _refresh_loss_streak(
    *,
    trader_id: str,
    mode: Optional[str],
    limit: int,
    cache_key: tuple,
) -> None:
    async with AsyncSessionLocal() as session:
        losses = await _query_loss_streak(
            session,
            trader_id=trader_id,
            mode=mode,
            limit=limit,
        )
    _loss_streak_cache[cache_key] = (time.monotonic(), losses)


async def _query_last_loss(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str],
    since: Optional[datetime] = None,
) -> Optional[datetime]:
    query = select(func.max(TraderOrder.updated_at)).where(
        TraderOrder.trader_id == trader_id,
        TraderOrder.status.in_(tuple(REALIZED_LOSS_ORDER_STATUSES)),
        _visible_trader_order_query_clause(),
    )
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)
    if since is not None:
        since_utc = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since.astimezone(timezone.utc)
        query = query.where(TraderOrder.updated_at >= since_utc)
    return (await session.execute(query)).scalar_one_or_none()


async def _refresh_last_loss(
    *,
    trader_id: str,
    mode: Optional[str],
    cache_key: tuple,
) -> None:
    async with AsyncSessionLocal() as session:
        result = await _query_last_loss(
            session,
            trader_id=trader_id,
            mode=mode,
        )
    _last_loss_cache[cache_key] = (time.monotonic(), result)


async def get_daily_realized_pnl(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> float:
    cache_key = (trader_id, _normalize_mode_key(mode) if mode is not None else None)
    cached = _daily_pnl_cache.get(cache_key)
    if cached is not None:
        age = time.monotonic() - cached[0]
        if age < _DAILY_PNL_CACHE_TTL_SECONDS:
            if age >= _DAILY_PNL_CACHE_TTL_SECONDS * _REALIZED_PNL_REFRESH_FRACTION:
                _schedule_realized_pnl_bg_refresh(
                    cache_name="daily_pnl",
                    cache_key=cache_key,
                    coro_factory=lambda: _refresh_daily_pnl(
                        trader_id=trader_id,
                        mode=mode,
                        cache_key=cache_key,
                    ),
                )
            return cached[1]
    today_start = _now().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await get_realized_pnl(
        session,
        trader_id=trader_id,
        mode=mode,
        since=today_start,
    )
    _daily_pnl_cache[cache_key] = (time.monotonic(), result)
    return result


async def get_unrealized_pnl(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> float:
    """Compute mark-to-market unrealized PnL for all active orders.

    For each open position with an entry price and notional, fetches the
    current live mid-price and computes unrealized PnL as:
        quantity * (current_price - entry_price)
    where quantity = notional / entry_price.

    Returns the total unrealized PnL in USD (negative means the open
    positions are currently losing money).
    """
    from services.live_price_snapshot import get_live_mid_prices

    # Gather active orders with entry prices and token IDs
    query = select(TraderOrder).where(_visible_trader_order_query_clause())
    if trader_id:
        query = query.where(TraderOrder.trader_id == trader_id)
    if mode is not None:
        mode_key = _normalize_mode_key(mode)
        if mode_key == "other":
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == "")
        else:
            query = query.where(func.lower(func.coalesce(TraderOrder.mode, "")) == mode_key)

    rows = list((await session.execute(query)).scalars().all())

    # Filter to active orders only and collect token IDs for price lookup
    active_orders = []
    token_ids: list[str] = []
    for order in rows:
        order_mode = _normalize_mode_key(order.mode)
        if not _is_active_order_status(order_mode, order.status):
            continue

        payload = dict(order.payload_json or {})
        position_state = payload.get("position_state")
        position_state = position_state if isinstance(position_state, dict) else {}
        row_notional = safe_float(order.notional_usd, 0.0) or 0.0
        notional = _live_active_notional(order_mode, order.status, row_notional, payload, executed_at=order.executed_at)
        _, filled_shares, fill_price = _extract_live_fill_metrics(payload)
        entry_price = (
            fill_price
            if fill_price is not None and fill_price > 0
            else safe_float(order.effective_price, 0.0) or safe_float(order.entry_price, 0.0) or 0.0
        )
        if entry_price <= 0 or notional <= 0:
            continue
        quantity = (
            float(filled_shares)
            if order_mode in {"live", "shadow"} and filled_shares > 0.0
            else float(notional / entry_price if entry_price > 0 else 0.0)
        )
        if quantity <= 0:
            continue

        token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip()
        if not token_id:
            continue

        active_orders.append(
            (
                order,
                entry_price,
                quantity,
                token_id,
                safe_float(position_state.get("last_mark_price"), 0.0) or 0.0,
            )
        )
        token_ids.append(token_id)

    if not active_orders:
        return 0.0

    # Fetch current prices in batch
    try:
        async with release_conn(session):
            prices = await get_live_mid_prices(token_ids)
    except Exception:
        prices = {}

    total_unrealized = 0.0
    for order, entry_price, quantity, token_id, last_mark_price in active_orders:
        current_price = prices.get(token_id)
        if (current_price is None or current_price <= 0) and last_mark_price > 0:
            current_price = last_mark_price
        if current_price is None or current_price <= 0:
            continue
        direction = str(order.direction or "").strip().lower()
        if direction in ("sell", "no", "short"):
            # Short position: profit when price drops
            total_unrealized += quantity * (entry_price - current_price)
        else:
            # Long position (default): profit when price rises
            total_unrealized += quantity * (current_price - entry_price)

    return total_unrealized


async def get_consecutive_loss_count(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    limit: int = 100,
    since: Optional[datetime] = None,
) -> int:
    # Cached at the same 30s cadence as daily PnL — both depend on
    # terminal-state trader_orders which change only on resolution.
    # Skip cache when caller passes a custom ``since`` so paginated
    # / windowed lookups still hit the DB.
    if since is None:
        cache_key = (trader_id, _normalize_mode_key(mode) if mode is not None else None, int(limit or 100))
        cached = _loss_streak_cache.get(cache_key)
        if cached is not None:
            age = time.monotonic() - cached[0]
            if age < _LOSS_STREAK_CACHE_TTL_SECONDS:
                if age >= _LOSS_STREAK_CACHE_TTL_SECONDS * _REALIZED_PNL_REFRESH_FRACTION:
                    _schedule_realized_pnl_bg_refresh(
                        cache_name="loss_streak",
                        cache_key=cache_key,
                        coro_factory=lambda: _refresh_loss_streak(
                            trader_id=trader_id,
                            mode=mode,
                            limit=int(limit or 100),
                            cache_key=cache_key,
                        ),
                    )
                return cached[1]
    losses = await _query_loss_streak(
        session,
        trader_id=trader_id,
        mode=mode,
        limit=int(limit or 100),
        since=since,
    )
    if since is None:
        cache_key = (trader_id, _normalize_mode_key(mode) if mode is not None else None, int(limit or 100))
        _loss_streak_cache[cache_key] = (time.monotonic(), losses)
    return losses


async def get_last_resolved_loss_at(
    session: AsyncSession,
    *,
    trader_id: str,
    mode: Optional[str] = None,
    since: Optional[datetime] = None,
) -> Optional[datetime]:
    if since is None:
        cache_key = (trader_id, _normalize_mode_key(mode) if mode is not None else None)
        cached = _last_loss_cache.get(cache_key)
        if cached is not None:
            age = time.monotonic() - cached[0]
            if age < _LAST_LOSS_CACHE_TTL_SECONDS:
                if age >= _LAST_LOSS_CACHE_TTL_SECONDS * _REALIZED_PNL_REFRESH_FRACTION:
                    _schedule_realized_pnl_bg_refresh(
                        cache_name="last_loss",
                        cache_key=cache_key,
                        coro_factory=lambda: _refresh_last_loss(
                            trader_id=trader_id,
                            mode=mode,
                            cache_key=cache_key,
                        ),
                    )
                return cached[1]
    result = await _query_last_loss(
        session,
        trader_id=trader_id,
        mode=mode,
        since=since,
    )
    if since is None:
        cache_key = (trader_id, _normalize_mode_key(mode) if mode is not None else None)
        _last_loss_cache[cache_key] = (time.monotonic(), result)
    return result


async def compute_orchestrator_metrics(session: AsyncSession) -> dict[str, Any]:
    order_status_key = func.lower(func.trim(func.coalesce(TraderOrder.status, "")))
    order_mode_key = func.lower(func.trim(func.coalesce(TraderOrder.mode, "")))
    open_order_filter = or_(
        and_(order_mode_key == "shadow", order_status_key.in_(tuple(UNFILLED_ORDER_STATUSES))),
        and_(order_mode_key == "live", order_status_key.in_(tuple(UNFILLED_ORDER_STATUSES))),
        and_(
            order_mode_key.not_in(("shadow", "live")),
            order_status_key.in_(tuple(UNFILLED_ORDER_STATUSES)),
        ),
    )
    active_execution_status_key = func.lower(func.trim(func.coalesce(ExecutionSession.status, "")))
    metrics_row = (
        await session.execute(
            select(
                select(func.count(Trader.id)).scalar_subquery().label("traders_total"),
                select(func.count(Trader.id))
                .where(
                    Trader.is_enabled == True,  # noqa: E712
                    Trader.is_paused == False,  # noqa: E712
                )
                .scalar_subquery()
                .label("traders_running"),
                select(func.count(TraderDecision.id)).scalar_subquery().label("decisions_count"),
                select(func.count(TraderOrder.id))
                .where(_visible_trader_order_query_clause())
                .scalar_subquery()
                .label("orders_count"),
                select(func.count(ExecutionSession.id)).scalar_subquery().label("execution_sessions_count"),
                select(func.count(ExecutionSession.id))
                .where(active_execution_status_key.in_(tuple(ACTIVE_EXECUTION_SESSION_STATUSES)))
                .scalar_subquery()
                .label("active_execution_sessions"),
                select(func.count(TraderOrder.id))
                .where(_visible_trader_order_query_clause())
                .where(open_order_filter)
                .scalar_subquery()
                .label("open_orders"),
            )
        )
    ).one()
    daily_pnl = await get_daily_realized_pnl(session)
    latency_summary = await execution_latency_metrics.snapshot()

    return {
        "traders_total": int(metrics_row.traders_total or 0),
        "traders_running": int(metrics_row.traders_running or 0),
        "decisions_count": int(metrics_row.decisions_count or 0),
        "orders_count": int(metrics_row.orders_count or 0),
        "execution_sessions_count": int(metrics_row.execution_sessions_count or 0),
        "active_execution_sessions": int(metrics_row.active_execution_sessions or 0),
        "open_orders": int(metrics_row.open_orders or 0),
        "gross_exposure_usd": await get_gross_exposure(session),
        "daily_pnl": daily_pnl,
        "execution_latency": latency_summary,
    }



async def compose_trader_orchestrator_config(
    session: AsyncSession,
    *,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if control is None:
        control = await read_orchestrator_control(session)
    settings_json = _normalize_control_settings(control.get("settings") or {})
    global_risk = dict(settings_json.get("global_risk") or _normalize_global_risk({}))
    global_runtime = dict(settings_json.get("global_runtime") or _normalize_global_runtime_settings({}))
    pending_live_exit_guard = dict(
        global_runtime.get("pending_live_exit_guard") or _normalize_pending_live_exit_guard({})
    )
    live_risk_clamps = _normalize_live_risk_clamps(
        global_runtime.get("live_risk_clamps"),
        explicit=bool(global_runtime.get("live_risk_clamps_explicit")),
    )
    live_market_context = dict(global_runtime.get("live_market_context") or _normalize_live_market_context({}))
    live_provider_health = dict(global_runtime.get("live_provider_health") or _normalize_live_provider_health({}))
    return {
        "mode": control.get("mode", "shadow"),
        "kill_switch": bool(control.get("kill_switch", False)),
        "run_interval_seconds": int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
        "global_risk": {
            "max_gross_exposure_usd": safe_float(global_risk.get("max_gross_exposure_usd"), 5000.0),
            "max_daily_loss_usd": safe_float(global_risk.get("max_daily_loss_usd"), 500.0),
            "max_orders_per_cycle": safe_int(global_risk.get("max_orders_per_cycle"), 50),
        },
        "trading_domains": settings_json.get("trading_domains") or ["event_markets", "crypto"],
        "enabled_strategies": settings_json.get("enabled_strategies") or [],
        "llm_verify_trades": bool(settings_json.get("llm_verify_trades", False)),
        "global_runtime": {
            "pending_live_exit_guard": {
                "max_pending_exits": int(pending_live_exit_guard.get("max_pending_exits", 0) or 0),
                "identity_guard_enabled": bool(pending_live_exit_guard.get("identity_guard_enabled", True)),
                "terminal_statuses": list(
                    pending_live_exit_guard.get("terminal_statuses")
                    or DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"]
                ),
            },
            "live_risk_clamps": dict(live_risk_clamps),
            "live_market_context": {
                "enabled": bool(live_market_context.get("enabled", True)),
                "history_window_seconds": int(live_market_context.get("history_window_seconds", 7200) or 7200),
                "history_fidelity_seconds": int(live_market_context.get("history_fidelity_seconds", 300) or 300),
                "max_history_points": int(live_market_context.get("max_history_points", 120) or 120),
                "timeout_seconds": float(live_market_context.get("timeout_seconds", 4.0) or 4.0),
                "strict_ws_pricing_only": bool(
                    live_market_context.get(
                        "strict_ws_pricing_only",
                        DEFAULT_LIVE_MARKET_CONTEXT["strict_ws_pricing_only"],
                    )
                ),
                "max_market_data_age_ms": int(
                    live_market_context.get(
                        "max_market_data_age_ms",
                        DEFAULT_LIVE_MARKET_CONTEXT["max_market_data_age_ms"],
                    )
                    or DEFAULT_LIVE_MARKET_CONTEXT["max_market_data_age_ms"]
                ),
            },
            "live_provider_health": {
                "window_seconds": int(live_provider_health.get("window_seconds", 180) or 180),
                "min_errors": int(live_provider_health.get("min_errors", 2) or 2),
                "block_seconds": int(live_provider_health.get("block_seconds", 120) or 120),
            },
            "trader_cycle_timeout_seconds": (
                float(global_runtime.get("trader_cycle_timeout_seconds"))
                if global_runtime.get("trader_cycle_timeout_seconds") is not None
                else None
            ),
        },
    }


async def get_orchestrator_overview(session: AsyncSession) -> dict[str, Any]:
    control = await read_orchestrator_control(session)
    worker = await read_orchestrator_snapshot(session)
    if isinstance(worker, dict):
        worker = dict(worker)
        worker["stats"] = summarize_worker_stats(worker.get("stats"))
    metrics = dict(worker.get("stats") or {}) if isinstance(worker, dict) else {}
    if isinstance(worker, dict):
        metrics.setdefault("traders_total", int(worker.get("traders_total", 0) or 0))
        metrics.setdefault("traders_running", int(worker.get("traders_running", 0) or 0))
        metrics.setdefault("decisions_count", int(worker.get("decisions_count", 0) or 0))
        metrics.setdefault("orders_count", int(worker.get("orders_count", 0) or 0))
        metrics.setdefault("open_orders", int(worker.get("open_orders", 0) or 0))
        metrics.setdefault("gross_exposure_usd", float(worker.get("gross_exposure_usd", 0.0) or 0.0))
        metrics.setdefault("daily_pnl", float(worker.get("daily_pnl", 0.0) or 0.0))
    existing_latency = metrics.get("execution_latency")
    if not isinstance(existing_latency, dict) or not int(existing_latency.get("sample_count") or 0):
        metrics["execution_latency"] = await _latency_snapshot_from_events(session, rolling_window_seconds=86400)
    return {
        "control": control,
        "worker": worker,
        "reconciliation_worker": await read_worker_snapshot(session, "trader_reconciliation"),
        "config": await compose_trader_orchestrator_config(session, control=control),
        "metrics": metrics,
    }


async def _build_preflight_checks(
    session: AsyncSession,
    control: dict[str, Any],
    trader_count: int,
) -> list[dict[str, Any]]:
    app_settings = await session.get(AppSettings, "default")
    polymarket_ready = bool(
        app_settings is not None
        and decrypt_secret(app_settings.polymarket_api_key)
        and decrypt_secret(app_settings.polymarket_api_secret)
        and decrypt_secret(app_settings.polymarket_api_passphrase)
    )
    kalshi_ready = bool(
        app_settings is not None
        and (app_settings.kalshi_email or "").strip()
        and decrypt_secret(app_settings.kalshi_password)
        and decrypt_secret(app_settings.kalshi_api_key)
    )
    live_creds_ready = polymarket_ready or kalshi_ready

    return [
        {
            "id": "live_credentials_configured",
            "ok": live_creds_ready,
            "message": "At least one live venue credential set must be configured",
            "polymarket_ready": polymarket_ready,
            "kalshi_ready": kalshi_ready,
        },
        {
            "id": "kill_switch_clear",
            "ok": not bool(control.get("kill_switch")),
            "message": "Kill switch must be disabled",
        },
        {
            "id": "traders_configured",
            "ok": trader_count > 0,
            "message": "At least one trader must exist",
            "trader_count": trader_count,
        },
    ]


async def create_live_preflight(
    session: AsyncSession,
    *,
    requested_mode: str,
    requested_by: Optional[str],
) -> dict[str, Any]:
    control_row = await ensure_orchestrator_control(session)
    control = _serialize_control(control_row)
    trader_count = int((await session.execute(select(func.count(Trader.id)))).scalar() or 0)
    checks = await _build_preflight_checks(session, control, trader_count)
    failed = [check for check in checks if not check["ok"]]
    status = "passed" if not failed else "failed"

    preflight = {
        "preflight_id": _new_id(),
        "requested_mode": requested_mode,
        "requested_by": requested_by,
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "created_at": to_iso(_now()),
    }

    settings_json = dict(control_row.settings_json or {})
    settings_json["live_preflight"] = preflight
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await _commit_with_retry(session)

    await create_trader_event(
        session,
        event_type="live_preflight",
        severity="info" if status == "passed" else "warn",
        source="trader_orchestrator",
        operator=requested_by,
        message=f"Live preflight {status}",
        payload=preflight,
    )
    return preflight


async def arm_live_start(
    session: AsyncSession,
    *,
    preflight_id: str,
    ttl_seconds: int,
    requested_by: Optional[str],
) -> dict[str, Any]:
    control_row = await ensure_orchestrator_control(session)
    settings_json = dict(control_row.settings_json or {})
    preflight = settings_json.get("live_preflight") or {}

    if str(preflight.get("preflight_id")) != str(preflight_id):
        raise ValueError("Unknown preflight_id")
    if str(preflight.get("status")) != "passed":
        raise ValueError("Preflight did not pass")

    arm_token = _new_id()
    expires_at = _now() + timedelta(seconds=max(30, min(ttl_seconds, 1800)))
    arm_data = {
        "arm_token": arm_token,
        "expires_at": to_iso(expires_at),
        "consumed_at": None,
        "requested_by": requested_by,
    }

    settings_json["live_arm"] = arm_data
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await _commit_with_retry(session)

    await create_trader_event(
        session,
        event_type="live_arm",
        source="trader_orchestrator",
        operator=requested_by,
        message="Live start token issued",
        payload={"preflight_id": preflight_id, "expires_at": arm_data["expires_at"]},
    )

    return {
        "preflight_id": preflight_id,
        "arm_token": arm_token,
        "expires_at": arm_data["expires_at"],
    }


async def consume_live_arm_token(session: AsyncSession, arm_token: str) -> bool:
    control_row = await ensure_orchestrator_control(session)
    settings_json = dict(control_row.settings_json or {})
    arm_data = settings_json.get("live_arm") or {}

    if str(arm_data.get("arm_token")) != str(arm_token):
        return False
    if arm_data.get("consumed_at"):
        return False

    expires_at_raw = arm_data.get("expires_at")
    expires_at = None
    if isinstance(expires_at_raw, str):
        try:
            expires_at = datetime.fromisoformat(expires_at_raw.replace("Z", "+00:00"))
        except Exception:
            expires_at = None

    if expires_at is not None and _now().astimezone(timezone.utc) > expires_at.astimezone(timezone.utc):
        return False

    arm_data["consumed_at"] = to_iso(_now())
    settings_json["live_arm"] = arm_data
    control_row.settings_json = settings_json
    control_row.updated_at = _now()
    await _commit_with_retry(session)
    return True


async def list_serialized_trader_decisions(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    trader_ids: Optional[list[str]] = None,
    decision: Optional[str] = None,
    limit: int = 200,
    per_trader_limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    rows = await list_trader_decisions(
        session,
        trader_id=trader_id,
        trader_ids=trader_ids,
        decision=decision,
        limit=limit,
        per_trader_limit=per_trader_limit,
    )
    decision_ids = sorted({str(row.id).strip() for row in rows if str(row.id or "").strip()})
    signal_ids = sorted({str(row.signal_id).strip() for row in rows if str(row.signal_id or "").strip()})
    signals_by_id: dict[str, TradeSignal] = {}
    if signal_ids:
        signal_rows = (await session.execute(select(TradeSignal).where(TradeSignal.id.in_(signal_ids)))).scalars().all()
        signals_by_id = {str(signal_row.id): signal_row for signal_row in signal_rows}

    failed_checks_by_decision: dict[str, list[dict[str, Any]]] = {}
    if decision_ids:
        failed_check_rows = (
            (
                await session.execute(
                    select(TraderDecisionCheck)
                    .where(
                        TraderDecisionCheck.decision_id.in_(decision_ids),
                        TraderDecisionCheck.passed.is_(False),
                    )
                    .order_by(TraderDecisionCheck.created_at.asc())
                )
            )
            .scalars()
            .all()
        )
        for check_row in failed_check_rows:
            decision_key = str(check_row.decision_id or "").strip()
            if not decision_key:
                continue
            failed_checks_by_decision.setdefault(decision_key, []).append(
                {
                    "check_key": str(check_row.check_key or ""),
                    "check_label": str(check_row.check_label or ""),
                    "detail": check_row.detail,
                    "score": check_row.score,
                    "payload": check_row.payload_json or {},
                    "created_at": to_iso(check_row.created_at),
                }
            )

    serialized_rows: list[dict[str, Any]] = []
    for row in rows:
        serialized = _serialize_decision_with_signal(row, signals_by_id.get(str(row.signal_id or "")))
        failed_checks = failed_checks_by_decision.get(str(row.id or ""), [])
        serialized["failed_checks"] = failed_checks
        serialized["failed_check_count"] = len(failed_checks)
        serialized_rows.append(serialized)

    return serialized_rows


def _trade_bundle_leg_count_for_summary(
    signal: TradeSignal | None,
    row: TraderOrder,
    sibling_rows: list[TraderOrder] | None = None,
) -> int:
    bundle = _build_trade_bundle(signal, row, sibling_rows)
    return int(bundle.get("leg_count") or 0) if isinstance(bundle, dict) else 0


def _trade_group_key_for_summary(
    signal: TradeSignal | None,
    row: TraderOrder,
    sibling_rows: list[TraderOrder] | None = None,
) -> tuple[str, bool]:
    leg_count = _trade_bundle_leg_count_for_summary(signal, row, sibling_rows)
    if leg_count > 1:
        bundle_id = str(signal.id if signal is not None else row.signal_id or row.id).strip() or str(row.id)
        return f"bundle:{bundle_id}", True
    return f"order:{row.id}", False


def _grouped_trade_counts(
    rows: list[TraderOrder],
    signals_by_id: dict[str, TradeSignal],
    *,
    open_statuses: tuple[str, ...],
    resolved_statuses: tuple[str, ...],
    failed_statuses: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    rows_by_signal_id: dict[str, list[TraderOrder]] = defaultdict(list)
    for row in rows:
        signal_key = str(row.signal_id or "").strip()
        if signal_key:
            rows_by_signal_id[signal_key].append(row)

    grouped: dict[str, dict[str, Any]] = {}
    per_trader: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "trade_count": 0,
            "open_trades": 0,
            "resolved_trades": 0,
            "failed_trades": 0,
            "partial_open_bundles": 0,
        }
    )
    open_status_set = set(open_statuses)
    resolved_status_set = set(resolved_statuses)
    failed_status_set = set(failed_statuses)
    incomplete_bundle_statuses = {"submitted", "open"}

    for row in rows:
        if not _visible_trader_order_row(row):
            continue
        signal_key = str(row.signal_id or "").strip()
        signal = signals_by_id.get(signal_key)
        group_key, is_bundle = _trade_group_key_for_summary(
            signal,
            row,
            rows_by_signal_id.get(signal_key),
        )
        group = grouped.setdefault(
            group_key,
            {
                "trader_id": str(row.trader_id or "").strip(),
                "statuses": set(),
                "is_bundle": is_bundle,
            },
        )
        status = str(row.status or "").strip().lower()
        if status:
            group["statuses"].add(status)

    totals = {
        "total_trades": 0,
        "open_trades": 0,
        "resolved_trades": 0,
        "failed_trades": 0,
        "partial_open_bundles": 0,
    }

    for group in grouped.values():
        trader_id = str(group.get("trader_id") or "").strip()
        trader_counts = per_trader[trader_id]
        statuses = set(group.get("statuses") or ())
        totals["total_trades"] += 1
        trader_counts["trade_count"] += 1
        if statuses & open_status_set:
            totals["open_trades"] += 1
            trader_counts["open_trades"] += 1
        elif statuses & resolved_status_set:
            totals["resolved_trades"] += 1
            trader_counts["resolved_trades"] += 1
        elif statuses & failed_status_set:
            totals["failed_trades"] += 1
            trader_counts["failed_trades"] += 1
        if bool(group.get("is_bundle")) and "executed" in statuses and bool(statuses & incomplete_bundle_statuses):
            totals["partial_open_bundles"] += 1
            trader_counts["partial_open_bundles"] += 1

    return totals, per_trader


async def get_trader_orders_summary(
    session: AsyncSession,
    *,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    open_statuses = tuple(sorted(OPEN_ORDER_STATUSES))
    resolved_statuses = ('resolved', 'resolved_win', 'resolved_loss', 'closed_win', 'closed_loss', 'win', 'loss')
    failed_statuses = ('failed', 'rejected', 'error', 'cancelled')

    status_col = func.lower(TraderOrder.status)
    is_open = status_col.in_(open_statuses)
    is_resolved = status_col.in_(resolved_statuses)
    is_failed = status_col.in_(failed_statuses)
    # Only count actual_profit for orders whose realized P&L was verified
    # against Polymarket as the source of truth. The ONLY truly-verified
    # statuses are:
    #   * wallet_activity — actual on-chain trade matched by tx_hash
    #     from polymarket.get_wallet_trades (covers bot SELLs, manual
    #     user sells, and resolution payouts after Phase 4)
    #   * venue_fill     — Polymarket CLOB order status confirmed filled
    #     with an actual average_fill_price from the venue
    #
    # Everything else is inferred / pending and excluded from totals:
    #   * wallet_position — wallet shows balance=0, close_price was
    #     guessed from currentPrice / wallet aggregate (this is the
    #     phantom-PnL path the user caught — Flash Crash $85)
    #   * venue_order     — order on the book, not yet filled
    #   * local           — bot's local belief, no venue confirmation
    #   * summary_only    — wallet shows position closed but no trade
    #     match (verifier will upgrade once trades data appears)
    #   * disputed        — venue truth contradicts our internal claim
    _UNVERIFIED_VERIFICATION_STATUSES = (
        "summary_only",
        "local",
        "disputed",
        "wallet_position",
        "venue_order",
    )
    is_pnl_verified = func.lower(
        func.coalesce(TraderOrder.verification_status, "")
    ).notin_(_UNVERIFIED_VERIFICATION_STATUSES)
    profit_col = func.coalesce(TraderOrder.actual_profit, 0.0)
    # verified_profit_col is what gets summed for any "P&L" total. If the
    # row's verification_status is unverified, contribute 0 to the sum so
    # we never display a phantom number to the user. Counts (wins/losses)
    # use the raw column so the operator can still see the order count.
    verified_profit_col = case(
        (is_pnl_verified, profit_col),
        else_=0.0,
    )

    query = select(
        func.count().label("total_count"),
        func.count().filter(is_open).label("open"),
        func.count().filter(is_resolved).label("resolved"),
        func.count().filter(is_failed).label("failed"),
        func.sum(verified_profit_col).filter(is_resolved).label("resolved_pnl"),
        func.sum(verified_profit_col).filter(is_resolved & (verified_profit_col > 0)).label("wins_pnl"),
        func.count().filter(is_resolved & is_pnl_verified & (profit_col > 0)).label("wins"),
        func.count().filter(is_resolved & is_pnl_verified & (profit_col < 0)).label("losses"),
        func.sum(func.abs(func.coalesce(TraderOrder.notional_usd, 0.0))).label("total_notional"),
        func.avg(TraderOrder.edge_percent).filter(TraderOrder.edge_percent.isnot(None) & (TraderOrder.edge_percent != 0)).label("avg_edge"),
        func.avg(TraderOrder.confidence).filter(TraderOrder.confidence.isnot(None) & (TraderOrder.confidence != 0)).label("avg_confidence"),
    ).where(_visible_trader_order_query_clause())
    if mode:
        query = query.where(func.lower(TraderOrder.mode) == mode.strip().lower())

    row = (await session.execute(query)).one()
    resolved_count = int(row.resolved or 0)
    wins_count = int(row.wins or 0)

    by_trader_query = select(
        TraderOrder.trader_id,
        func.count().label("orders"),
        func.count().filter(is_open).label("open"),
        func.count().filter(is_resolved).label("resolved"),
        func.sum(verified_profit_col).filter(is_resolved).label("pnl"),
        func.sum(func.abs(func.coalesce(TraderOrder.notional_usd, 0.0))).label("notional"),
        func.count().filter(is_resolved & is_pnl_verified & (profit_col > 0)).label("wins"),
        func.count().filter(is_resolved & is_pnl_verified & (profit_col < 0)).label("losses"),
        func.max(func.coalesce(TraderOrder.updated_at, TraderOrder.created_at)).label("latest_activity"),
    ).where(_visible_trader_order_query_clause()).group_by(TraderOrder.trader_id)
    if mode:
        by_trader_query = by_trader_query.where(func.lower(TraderOrder.mode) == mode.strip().lower())
    trader_rows = (await session.execute(by_trader_query)).all()

    all_orders_query = select(TraderOrder).where(_visible_trader_order_query_clause())
    if mode:
        all_orders_query = all_orders_query.where(func.lower(TraderOrder.mode) == mode.strip().lower())
    all_order_rows = list((await session.execute(all_orders_query)).scalars().all())
    signal_ids = sorted({str(order.signal_id).strip() for order in all_order_rows if str(order.signal_id or "").strip()})
    signals_by_id: dict[str, TradeSignal] = {}
    if signal_ids:
        signal_rows = (await session.execute(select(TradeSignal).where(TradeSignal.id.in_(signal_ids)))).scalars().all()
        signals_by_id = {str(signal.id).strip(): signal for signal in signal_rows}
    trade_totals, trade_counts_by_trader = _grouped_trade_counts(
        all_order_rows,
        signals_by_id,
        open_statuses=open_statuses,
        resolved_statuses=resolved_statuses,
        failed_statuses=failed_statuses,
    )
    active_position_query = (
        select(TraderPosition.trader_id, func.count().label("open_trades"))
        .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
        .group_by(TraderPosition.trader_id)
    )
    if mode:
        active_position_query = active_position_query.where(
            func.lower(func.coalesce(TraderPosition.mode, "")) == mode.strip().lower()
        )
    active_position_rows = (await session.execute(active_position_query)).all()
    active_position_counts_by_trader = {
        str(row.trader_id or ""): int(row.open_trades or 0)
        for row in active_position_rows
    }
    active_position_open_trades = sum(active_position_counts_by_trader.values())
    if active_position_open_trades > 0:
        trade_totals["open_trades"] = active_position_open_trades
        for trader_id_key, open_trade_count in active_position_counts_by_trader.items():
            trade_counts = trade_counts_by_trader.setdefault(
                trader_id_key,
                {
                    "trade_count": 0,
                    "open_trades": 0,
                    "resolved_trades": 0,
                    "failed_trades": 0,
                    "partial_open_bundles": 0,
                },
            )
            trade_counts["open_trades"] = int(open_trade_count or 0)
    by_source_query = select(
        TraderOrder.source,
        func.count().label("orders"),
        func.count().filter(is_resolved).label("resolved"),
        func.sum(verified_profit_col).filter(is_resolved).label("pnl"),
        func.sum(func.abs(func.coalesce(TraderOrder.notional_usd, 0.0))).label("notional"),
        func.count().filter(is_resolved & is_pnl_verified & (profit_col > 0)).label("wins"),
        func.count().filter(is_resolved & is_pnl_verified & (profit_col < 0)).label("losses"),
    ).where(_visible_trader_order_query_clause()).group_by(TraderOrder.source)
    if mode:
        by_source_query = by_source_query.where(func.lower(TraderOrder.mode) == mode.strip().lower())
    by_source_query = by_source_query.order_by(func.count().desc()).limit(8)
    source_rows = (await session.execute(by_source_query)).all()

    return {
        "total_count": int(row.total_count or 0),
        "open": int(row.open or 0),
        "resolved": resolved_count,
        "failed": int(row.failed or 0),
        "resolved_pnl": float(row.resolved_pnl or 0),
        "wins": wins_count,
        "losses": int(row.losses or 0),
        "total_notional": float(row.total_notional or 0),
        "win_rate": (wins_count / resolved_count * 100) if resolved_count > 0 else 0,
        "avg_edge": float(row.avg_edge or 0),
        "avg_confidence": float(row.avg_confidence or 0),
        "total_trades": int(trade_totals["total_trades"] or 0),
        "open_trades": int(trade_totals["open_trades"] or 0),
        "resolved_trades": int(trade_totals["resolved_trades"] or 0),
        "failed_trades": int(trade_totals["failed_trades"] or 0),
        "partial_open_bundles": int(trade_totals["partial_open_bundles"] or 0),
        "by_trader": [
            {
                "trader_id": str(tr.trader_id or ""),
                "orders": int(tr.orders or 0),
                "open": int(tr.open or 0),
                "resolved": int(tr.resolved or 0),
                "trade_count": int((trade_counts_by_trader.get(str(tr.trader_id or "")) or {}).get("trade_count", tr.orders or 0)),
                "open_trades": int((trade_counts_by_trader.get(str(tr.trader_id or "")) or {}).get("open_trades", tr.open or 0)),
                "resolved_trades": int((trade_counts_by_trader.get(str(tr.trader_id or "")) or {}).get("resolved_trades", tr.resolved or 0)),
                "failed_trades": int((trade_counts_by_trader.get(str(tr.trader_id or "")) or {}).get("failed_trades", 0)),
                "partial_open_bundles": int((trade_counts_by_trader.get(str(tr.trader_id or "")) or {}).get("partial_open_bundles", 0)),
                "pnl": float(tr.pnl or 0),
                "notional": float(tr.notional or 0),
                "wins": int(tr.wins or 0),
                "losses": int(tr.losses or 0),
                "latest_activity_ts": tr.latest_activity.isoformat() + "Z" if tr.latest_activity else None,
            }
            for tr in sorted(trader_rows, key=lambda r: float(r.pnl or 0), reverse=True)
        ],
        "by_source": [
            {
                "source": str(sr.source or ""),
                "orders": int(sr.orders or 0),
                "resolved": int(sr.resolved or 0),
                "pnl": float(sr.pnl or 0),
                "notional": float(sr.notional or 0),
                "wins": int(sr.wins or 0),
                "losses": int(sr.losses or 0),
            }
            for sr in source_rows
        ],
    }


async def list_serialized_trader_orders(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    status: Optional[str] = None,
    mode: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    since_seconds: Optional[int] = None,
) -> list[dict[str, Any]]:
    rows = await list_trader_orders(
        session,
        trader_id=trader_id,
        status=status,
        mode=mode,
        limit=limit,
        offset=offset,
        since_seconds=since_seconds,
    )
    signal_ids = sorted({str(row.signal_id).strip() for row in rows if str(row.signal_id or "").strip()})
    signals_by_id: dict[str, TradeSignal] = {}
    if signal_ids:
        signal_rows = (await session.execute(select(TradeSignal).where(TradeSignal.id.in_(signal_ids)))).scalars().all()
        signals_by_id = {str(signal_row.id): signal_row for signal_row in signal_rows}
    sibling_rows_by_signal_id: dict[str, list[TraderOrder]] = defaultdict(list)
    if signal_ids:
        sibling_rows = (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.signal_id.in_(signal_ids),
                    _visible_trader_order_query_clause(),
                )
            )
        ).scalars().all()
        for sibling_row in sibling_rows:
            signal_key = str(sibling_row.signal_id or "").strip()
            if signal_key:
                sibling_rows_by_signal_id[signal_key].append(sibling_row)
    condition_ids_for_rows = [_extract_order_condition_id(row) for row in rows]
    condition_ids = sorted({condition_id for condition_id in condition_ids_for_rows if condition_id})
    cached_markets_by_condition_id: dict[str, CachedMarket] = {}
    if condition_ids:
        cached_market_rows = (
            (
                await session.execute(
                    select(CachedMarket).where(func.lower(CachedMarket.condition_id).in_(condition_ids))
                )
            )
            .scalars()
            .all()
        )
        for cached_market in cached_market_rows:
            key = _normalize_condition_id(cached_market.condition_id)
            if key:
                cached_markets_by_condition_id[key] = cached_market

    serialized_rows: list[dict[str, Any]] = []
    for row, condition_id in zip(rows, condition_ids_for_rows):
        serialized_rows.append(
            _serialize_order(
                row,
                signals_by_id.get(str(row.signal_id or "")),
                cached_markets_by_condition_id.get(condition_id),
                sibling_rows_by_signal_id.get(str(row.signal_id or "").strip()),
            )
        )
    return serialized_rows


async def list_serialized_trader_events(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    limit: int = 200,
    cursor: Optional[str] = None,
    event_types: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    rows, next_cursor = await list_trader_events(
        session,
        trader_id=trader_id,
        limit=limit,
        cursor=cursor,
        event_types=event_types,
    )
    return ([_serialize_event(row) for row in rows], next_cursor)


async def list_serialized_trader_events_bulk(
    session: AsyncSession,
    *,
    trader_ids: Optional[list[str]] = None,
    limit: int = 200,
    event_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    query = select(TraderEvent).order_by(desc(TraderEvent.created_at), desc(TraderEvent.id))
    if trader_ids:
        query = query.where(TraderEvent.trader_id.in_(trader_ids))
    if event_types:
        query = query.where(TraderEvent.event_type.in_(event_types))
    query = query.limit(max(1, min(limit, 2000)))
    rows = list((await session.execute(query)).scalars().all())
    return [_serialize_event(row) for row in rows]


async def list_serialized_execution_sessions(
    session: AsyncSession,
    *,
    trader_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    return [
        _serialize_execution_session(row)
        for row in await list_execution_sessions(
            session,
            trader_id=trader_id,
            status=status,
            limit=limit,
        )
    ]


async def get_serialized_execution_session_detail(
    session: AsyncSession,
    session_id: str,
) -> dict[str, Any] | None:
    return await get_execution_session_detail(session, session_id)

