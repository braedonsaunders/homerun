"""Dedicated trader orchestrator worker consuming normalized trade signals."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.exc import OperationalError

from config import settings
from models.database import (
    AppSettings,
    AsyncSessionLocal,
    DiscoveredWallet,
    SimulationAccount,
    TradeSignal,
    TraderSignalConsumption,
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
from services.trader_orchestrator.position_lifecycle import (
    load_market_info_for_orders,
    reconcile_paper_positions,
    reconcile_shadow_positions,
)
from services.polymarket import polymarket_client
from services.simulation import simulation_service
from services.live_execution_service import live_execution_service
from services.trader_orchestrator.risk_manager import evaluate_risk
from services.trader_orchestrator.decision_gates import (
    apply_platform_decision_gates,
    is_within_trading_schedule_utc,
)
from services.worker_state import _commit_with_retry, _is_retryable_db_error
from services.trader_orchestrator.sources.registry import (
    normalize_source_key,
)
from services.trader_orchestrator.strategies.registry import (
    get_strategy as resolve_strategy_instance,
)
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_experiments import (
    get_active_strategy_experiment,
    resolve_experiment_assignment,
    upsert_strategy_experiment_assignment,
)
from services.strategy_loader import StrategyValidationError, strategy_loader
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.strategy_sdk import StrategySDK
from services.strategy_versioning import normalize_strategy_version, resolve_strategy_version
from services.redis_streams import redis_streams as _redis_stream_client
from services.trade_signal_stream import (
    ack_trade_signal_batches,
    auto_claim_trade_signal_batches,
    ensure_trade_signal_group,
    read_trade_signal_batches,
)
from services.trader_orchestrator_state import (
    DEFAULT_LIVE_MARKET_CONTEXT,
    DEFAULT_LIVE_PROVIDER_HEALTH,
    DEFAULT_LIVE_RISK_CLAMPS,
    DEFAULT_PENDING_LIVE_EXIT_GUARD,
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
    get_open_order_count_for_trader,
    get_pending_live_exit_summary_for_trader,
    get_trader_source_exposure,
    get_open_market_ids_for_trader,
    get_open_position_count_for_trader,
    get_trader_signal_cursor,
    list_traders,
    list_unconsumed_trade_signals,
    reconcile_live_provider_orders,
    read_orchestrator_control,
    record_signal_consumption,
    set_trader_paused,
    sync_trader_position_inventory,
    update_orchestrator_control,
    upsert_trader_signal_cursor,
    write_orchestrator_snapshot,
)
from services.signal_bus import expire_stale_signals, set_trade_signal_status
from utils.utcnow import utcnow
from utils.converters import safe_float, safe_int
from utils.secrets import decrypt_secret

logger = logging.getLogger("trader_orchestrator_worker")
strategy_db_loader = strategy_loader
create_trader_order = _create_trader_order
_RESUME_POLICIES = {"resume_full", "manage_only", "flatten_then_start"}
_PAPER_ACTIVE_ORDER_STATUSES = {"submitted", "executed", "open"}
_ORPHAN_CLEANUP_MAX_TRADERS_PER_CYCLE = 32
_ORCHESTRATOR_CYCLE_LOCK_KEY = 0x54524F5243485354  # "TRORCHST"
_orchestrator_lock_session: Any | None = None
_TRADER_IDLE_MAINTENANCE_INTERVAL_SECONDS = 60
_HIGH_FREQUENCY_CRYPTO_MAINTENANCE_INTERVAL_SECONDS = 1.0
_STANDARD_MAX_SIGNALS_PER_CYCLE = 500
_STANDARD_DEFAULT_MAX_SIGNALS_PER_CYCLE = 200
_HIGH_FREQUENCY_MAX_SIGNALS_PER_CYCLE = 5000
_LOSS_STREAK_RESET_AT_KEY = "loss_streak_reset_at"
_HIGH_FREQUENCY_DEFAULT_MAX_SIGNALS_PER_CYCLE = 2000
_HIGH_FREQUENCY_DEFAULT_SCAN_BATCH_SIZE = 1000
_trader_idle_maintenance_last_run: dict[str, datetime] = {}
_TRADER_CYCLE_HEARTBEAT_EVENT_INTERVAL_SECONDS = 15
_trader_cycle_heartbeat_last_emitted: dict[str, datetime] = {}
_TRADE_SIGNAL_STREAM_DRAIN_READS = 4
_TERMINAL_STALE_ORDER_CHECK_INTERVAL_SECONDS = 30
_TERMINAL_STALE_ORDER_MIN_AGE_MINUTES = 3
_TERMINAL_STALE_ORDER_ALERT_COOLDOWN_SECONDS = 300
_OPEN_ORDER_TIMEOUT_CLEANUP_FAILURE_COOLDOWN_SECONDS = 30
_LIVE_PROVIDER_BLOCK_EVENT_COOLDOWN_SECONDS = 60
_LIVE_RISK_CLAMP_EVENT_COOLDOWN_SECONDS = 300
_CRYPTO_OPEN_ORDER_TIMEOUT_FLOOR_SECONDS = 20.0
_LIVE_PROVIDER_RECONCILE_MIN_INTERVAL_SECONDS = 5.0
_terminal_stale_order_last_checked_at: datetime | None = None
_terminal_stale_order_alert_last_emitted: dict[str, datetime] = {}
_open_order_timeout_cleanup_failure_cooldown_until: dict[str, datetime] = {}
_live_provider_entry_blocked_until: dict[str, datetime] = {}
_live_provider_block_event_cooldown_until: dict[str, datetime] = {}
_live_risk_clamp_event_cooldown_until: dict[str, datetime] = {}
_live_provider_reconcile_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}

_LIVE_PROVIDER_INFRA_ERROR_MARKERS = (
    "connection refused",
    "database system is not yet accepting connections",
    "dial tcp",
    "i/o timeout",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection reset",
    "connection closed",
    "transport error",
    "read: connection",
    "write: broken pipe",
    "upstream connect error",
    "network is unreachable",
)


async def _worker_sleep(interval_seconds: float) -> None:
    await asyncio.sleep(interval_seconds)


def _coalesce_stream_trigger(
    stream_rows: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    if not stream_rows:
        return None
    source_signal_ids: dict[str, list[str]] = {}
    source_seen_ids: dict[str, set[str]] = {}
    stream_entry_ids: list[str] = []
    for entry_id, payload in stream_rows:
        stream_entry_ids.append(str(entry_id))
        source_key = normalize_source_key(payload.get("source")) or "__all__"
        source_signal_ids.setdefault(source_key, [])
        source_seen_ids.setdefault(source_key, set())
        for raw_signal_id in payload.get("signal_ids") or []:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in source_seen_ids[source_key]:
                continue
            source_seen_ids[source_key].add(signal_id)
            source_signal_ids[source_key].append(signal_id)
    if not stream_entry_ids:
        return None
    if not source_signal_ids:
        return {
            "event_type": "trade_signal_stream",
            "source": "",
            "source_signal_ids": {},
            "stream_entry_ids": stream_entry_ids,
        }
    non_global_sources = sorted(source for source in source_signal_ids.keys() if source != "__all__")
    trigger_source = non_global_sources[0] if len(non_global_sources) == 1 and "__all__" not in source_signal_ids else ""
    return {
        "event_type": "trade_signal_stream",
        "source": trigger_source,
        "source_signal_ids": source_signal_ids,
        "stream_entry_ids": stream_entry_ids,
    }


async def _wait_for_trade_signal_stream_trigger(
    *,
    consumer: str,
    timeout_seconds: float,
    claim_cursor: str,
    last_claim_run_at: datetime | None,
) -> tuple[dict[str, Any] | None, str, datetime]:
    timeout = max(0.05, float(timeout_seconds))
    now = datetime.now(timezone.utc)
    next_claim_cursor = str(claim_cursor or "0-0")
    last_claim = last_claim_run_at or now

    # If Streams are not available (Redis < 5.0), sleep for the timeout
    # and return None so the caller falls through to scheduled DB polling.
    # This avoids the tight spin-loop of failed xreadgroup calls.
    if not await _redis_stream_client.check_streams_available():
        await asyncio.sleep(timeout)
        return None, next_claim_cursor, last_claim

    deadline = now + timedelta(seconds=timeout)
    stream_rows: list[tuple[str, dict[str, Any]]] = []
    claim_interval_seconds = max(0.1, float(settings.TRADE_SIGNAL_STREAM_CLAIM_INTERVAL_SECONDS))
    claim_due = (
        last_claim_run_at is None
        or (now - last_claim_run_at).total_seconds() >= claim_interval_seconds
    )
    if claim_due:
        next_claim_cursor, claimed_rows = await auto_claim_trade_signal_batches(
            consumer=consumer,
            min_idle_ms=int(settings.TRADE_SIGNAL_STREAM_CLAIM_IDLE_MS),
            start_id=next_claim_cursor,
            count=int(settings.TRADE_SIGNAL_STREAM_CLAIM_READ_COUNT),
        )
        if claimed_rows:
            stream_rows.extend(claimed_rows)
            return _coalesce_stream_trigger(stream_rows), next_claim_cursor, now
        last_claim = now
    while datetime.now(timezone.utc) < deadline:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            break
        block_ms = max(1, min(int(remaining * 1000), int(settings.TRADE_SIGNAL_STREAM_BLOCK_MS)))
        rows = await read_trade_signal_batches(
            consumer=consumer,
            block_ms=block_ms,
            count=int(settings.TRADE_SIGNAL_STREAM_READ_COUNT),
            include_pending=False,
        )
        if not rows:
            continue
        stream_rows.extend(rows)
        for _ in range(_TRADE_SIGNAL_STREAM_DRAIN_READS):
            tail_rows = await read_trade_signal_batches(
                consumer=consumer,
                block_ms=1,
                count=int(settings.TRADE_SIGNAL_STREAM_READ_COUNT),
                include_pending=False,
            )
            if not tail_rows:
                break
            stream_rows.extend(tail_rows)
        break
    return _coalesce_stream_trigger(stream_rows), next_claim_cursor, last_claim


async def _wait_for_runtime_trigger(
    _trigger_queue: Any,
    timeout_seconds: float,
    *,
    consumer: str,
    claim_cursor: str,
    last_claim_run_at: datetime | None,
) -> tuple[dict[str, Any] | None, str, datetime]:
    return await _wait_for_trade_signal_stream_trigger(
        consumer=consumer,
        timeout_seconds=timeout_seconds,
        claim_cursor=claim_cursor,
        last_claim_run_at=last_claim_run_at,
    )


def _session_dialect_name(session: Any) -> str:
    try:
        bind = session.get_bind()
        return str(getattr(getattr(bind, "dialect", None), "name", "") or "").lower()
    except Exception:
        return ""


async def _write_orchestrator_snapshot_best_effort(session: Any, **snapshot_kwargs: Any) -> None:
    try:
        await write_orchestrator_snapshot(session, **snapshot_kwargs)
    except OperationalError as exc:
        if not _is_retryable_db_error(exc):
            raise
        if hasattr(session, "rollback"):
            await session.rollback()
        logger.warning("Skipped orchestrator snapshot write due to transient DB error")


async def submit_order(
    *,
    session_engine: ExecutionSessionEngine,
    trader_id: str,
    signal: RuntimeTradeSignalView,
    decision_id: str,
    strategy_key: str,
    strategy_version: int | None,
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
        strategy_version=strategy_version,
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


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_strategy_type_values(raw: Any) -> set[str]:
    if isinstance(raw, (list, tuple, set)):
        values = raw
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        values = []
    normalized: set[str] = set()
    for item in values:
        value = str(item or "").strip().lower()
        if value:
            normalized.add(value)
    return normalized


def _strategy_instance_for_source_config(source_config: dict[str, Any] | None) -> Any | None:
    if not isinstance(source_config, dict):
        return None
    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
    if not strategy_key:
        return None
    loaded = strategy_loader.get_strategy(strategy_key)
    if loaded is None:
        try:
            return resolve_strategy_instance(strategy_key)
        except Exception:
            return None
    instance = getattr(loaded, "instance", None)
    if instance is not None:
        return instance
    return loaded


def _merged_strategy_params_for_source_config(source_config: dict[str, Any]) -> dict[str, Any]:
    source_key = normalize_source_key(source_config.get("source_key"))
    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
    explicit_params = dict(source_config.get("strategy_params") or {})
    strategy_defaults: dict[str, Any] = {}
    strategy_version = normalize_strategy_version(source_config.get("strategy_version"))

    if strategy_version is None:
        strategy_instance = _strategy_instance_for_source_config(source_config)
        if strategy_instance is not None:
            configured_defaults = getattr(strategy_instance, "config", None)
            if isinstance(configured_defaults, dict):
                strategy_defaults = dict(configured_defaults)
            else:
                declared_defaults = getattr(strategy_instance, "default_config", None)
                if isinstance(declared_defaults, dict):
                    strategy_defaults = dict(declared_defaults)

    merged = {**strategy_defaults, **explicit_params}

    if source_key == "traders":
        if strategy_key == "traders_copy_trade":
            merged = StrategySDK.validate_traders_copy_trade_config(merged)
        else:
            merged = StrategySDK.validate_trader_filter_config(merged)
    elif source_key == "news":
        merged = StrategySDK.validate_news_filter_config(merged)

    return StrategySDK.normalize_strategy_retention_config(merged)


def _accepted_signal_strategy_types(source_config: dict[str, Any]) -> set[str]:
    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
    if not strategy_key:
        return set()

    allowed = {strategy_key}
    strategy_params = _merged_strategy_params_for_source_config(source_config)
    allowed.update(_normalize_strategy_type_values(strategy_params.get("accepted_signal_strategy_types")))

    strategy_instance = _strategy_instance_for_source_config(source_config)
    if strategy_instance is not None:
        allowed.update(
            _normalize_strategy_type_values(getattr(strategy_instance, "accepted_signal_strategy_types", None))
        )

    return allowed


def _supports_live_market_context(
    signal: Any,
    source_config: dict[str, Any] | None = None,
) -> bool:
    """Apply live-market enrichment where strategy evaluation requires it."""
    source = str(getattr(signal, "source", "") or "").strip().lower()
    if source != "crypto":
        return True
    if not isinstance(source_config, dict):
        return False

    strategy_params = _merged_strategy_params_for_source_config(source_config)
    explicit = _coerce_optional_bool(strategy_params.get("enable_live_market_context"))
    if explicit is None:
        explicit = _coerce_optional_bool(strategy_params.get("requires_live_market_context"))
    if explicit is not None:
        return explicit

    strategy_instance = _strategy_instance_for_source_config(source_config)
    if strategy_instance is not None:
        declared = _coerce_optional_bool(getattr(strategy_instance, "requires_live_market_context", None))
        if declared is None:
            declared = _coerce_optional_bool(getattr(strategy_instance, "enable_live_market_context", None))
        if declared is not None:
            return declared

    return False


def _is_exit_only_strategy_instance(strategy_instance: Any) -> bool:
    if strategy_instance is None:
        return False
    strategy_cls = strategy_instance.__class__
    class_dict = getattr(strategy_cls, "__dict__", {})
    has_custom_detect = any(name in class_dict for name in ("detect", "detect_async", "on_event"))
    has_custom_evaluate = "evaluate" in class_dict
    has_custom_should_exit = "should_exit" in class_dict
    return bool(has_custom_should_exit and not has_custom_detect and not has_custom_evaluate)


def _source_config_allows_new_entries(source_config: dict[str, Any] | None) -> bool:
    if not isinstance(source_config, dict):
        return True

    strategy_params = _merged_strategy_params_for_source_config(source_config)
    explicit_allow = _coerce_optional_bool(strategy_params.get("allow_new_entries"))
    if explicit_allow is not None:
        return bool(explicit_allow)
    explicit_disable = _coerce_optional_bool(strategy_params.get("disable_new_entries"))
    if explicit_disable is not None:
        return not bool(explicit_disable)

    strategy_instance = _strategy_instance_for_source_config(source_config)
    if strategy_instance is None:
        return True

    declared_allow = _coerce_optional_bool(getattr(strategy_instance, "allow_new_entries", None))
    if declared_allow is not None:
        return bool(declared_allow)
    declared_disable = _coerce_optional_bool(getattr(strategy_instance, "disable_new_entries", None))
    if declared_disable is not None:
        return not bool(declared_disable)

    if _is_exit_only_strategy_instance(strategy_instance):
        return False

    return True


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


def _is_high_frequency_maintenance_due(trader: dict[str, Any], now: datetime) -> bool:
    last_run = _parse_iso(trader.get("last_run_at"))
    if last_run is None:
        return True
    elapsed_seconds = (now - last_run.astimezone(timezone.utc)).total_seconds()
    return elapsed_seconds >= _HIGH_FREQUENCY_CRYPTO_MAINTENANCE_INTERVAL_SECONDS


def _is_high_frequency_crypto_trader(trader: dict[str, Any]) -> bool:
    source_configs = _normalize_source_configs(trader)
    if not source_configs:
        return False
    return "crypto" in _query_sources_for_configs(source_configs)


def _runtime_trigger_matches_trader(
    trader: dict[str, Any],
    trigger_event: dict[str, Any] | None,
) -> bool:
    if not trigger_event:
        return False
    source_signal_ids = trigger_event.get("source_signal_ids")
    if isinstance(source_signal_ids, dict):
        if "__all__" in source_signal_ids:
            return True
        source_configs = _normalize_source_configs(trader)
        if not source_configs:
            return False
        trader_sources = set(_query_sources_for_configs(source_configs))
        if not trader_sources:
            return False
        for source_key in source_signal_ids.keys():
            normalized_source = normalize_source_key(source_key)
            if normalized_source and normalized_source in trader_sources:
                return True
        return False
    source = str(trigger_event.get("source") or "").strip().lower()
    if not source:
        return True
    source_configs = _normalize_source_configs(trader)
    if not source_configs:
        return False
    return source in _query_sources_for_configs(source_configs)


def _trigger_signal_ids_for_trader(
    trader: dict[str, Any],
    trigger_event: dict[str, Any] | None,
) -> dict[str, list[str]] | None:
    if not trigger_event:
        return None
    raw = trigger_event.get("source_signal_ids")
    if not isinstance(raw, dict) or not raw:
        return None

    source_configs = _normalize_source_configs(trader)
    trader_sources = set(_query_sources_for_configs(source_configs))
    if not trader_sources:
        return None

    filtered: dict[str, list[str]] = {}
    global_ids: list[str] = []
    global_seen: set[str] = set()
    for source_key, signal_ids in raw.items():
        source_token = str(source_key or "").strip().lower()
        normalized_source = normalize_source_key(source_token)
        if not isinstance(signal_ids, list):
            continue
        if source_token == "__all__":
            for raw_signal_id in signal_ids:
                signal_id = str(raw_signal_id or "").strip()
                if not signal_id or signal_id in global_seen:
                    continue
                global_seen.add(signal_id)
                global_ids.append(signal_id)
            continue
        if not normalized_source or normalized_source not in trader_sources:
            continue
        seen: set[str] = set()
        out: list[str] = []
        for raw_signal_id in signal_ids:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in seen:
                continue
            seen.add(signal_id)
            out.append(signal_id)
        if out:
            filtered[normalized_source] = out
    if global_ids:
        filtered["__all__"] = global_ids
    if not filtered:
        return None
    return filtered


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
    return StrategySDK.normalize_trader_wallet(value)


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
        strategy_version = normalize_strategy_version(raw.get("strategy_version"))
        if not source_key or not strategy_key:
            continue
        source_config = {
            "source_key": source_key,
            "strategy_key": strategy_key,
            "strategy_version": strategy_version,
            "strategy_params": dict(raw.get("strategy_params") or {}),
        }
        strategy_params = _merged_strategy_params_for_source_config(source_config)
        normalized[source_key] = {
            "source_key": source_key,
            "strategy_key": strategy_key,
            "strategy_version": strategy_version,
            "strategy_params": strategy_params,
        }
    return normalized


def _source_open_order_timeout_seconds(source_config: dict[str, Any]) -> float | None:
    source_key = normalize_source_key(source_config.get("source_key"))
    strategy_params = dict(source_config.get("strategy_params") or {})
    strategy_instance = _strategy_instance_for_source_config(source_config)
    default_seconds = None
    if strategy_instance is not None:
        declared_default = safe_float(getattr(strategy_instance, "default_open_order_timeout_seconds", None), None)
        if declared_default is not None and declared_default > 0.0:
            default_seconds = float(declared_default)
        else:
            strategy_defaults = getattr(strategy_instance, "default_config", None)
            if isinstance(strategy_defaults, dict):
                default_seconds = StrategySDK.resolve_open_order_timeout_seconds(
                    strategy_defaults,
                    default_seconds=None,
                )
    timeout_seconds = StrategySDK.resolve_open_order_timeout_seconds(
        strategy_params,
        default_seconds=default_seconds,
    )
    if timeout_seconds is None:
        return None
    if source_key == "crypto":
        configured_floor = safe_float(strategy_params.get("min_open_order_timeout_seconds"), None)
        if configured_floor is None:
            configured_floor = safe_float(
                getattr(strategy_instance, "default_open_order_timeout_seconds", None),
                _CRYPTO_OPEN_ORDER_TIMEOUT_FLOOR_SECONDS,
            )
        crypto_floor_seconds = max(
            _CRYPTO_OPEN_ORDER_TIMEOUT_FLOOR_SECONDS,
            max(1.0, float(configured_floor)),
        )
        timeout_seconds = max(float(timeout_seconds), crypto_floor_seconds)
    return float(timeout_seconds)


async def _enforce_source_open_order_timeouts(
    session: Any,
    *,
    trader_id: str,
    run_mode: str,
    source_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not source_configs:
        return {"configured": 0, "updated": 0, "suppressed": 0, "sources": [], "errors": []}

    scope = run_mode if run_mode in {"paper", "shadow", "live"} else "all"
    configured = 0
    updated = 0
    suppressed = 0
    source_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    now_utc = utcnow()
    provider_reconcile: dict[str, Any] = {}

    expired_keys = [
        key for key, until in _open_order_timeout_cleanup_failure_cooldown_until.items() if until <= now_utc
    ]
    for key in expired_keys:
        _open_order_timeout_cleanup_failure_cooldown_until.pop(key, None)

    if scope == "live":
        cache_key = str(trader_id)
        cache_row = _live_provider_reconcile_cache.get(cache_key)
        cache_age_seconds: float | None = None
        if cache_row is not None:
            cache_ts, cached_payload = cache_row
            cache_age_seconds = max(0.0, (now_utc - cache_ts).total_seconds())
            if (
                cache_age_seconds < _LIVE_PROVIDER_RECONCILE_MIN_INTERVAL_SECONDS
                and isinstance(cached_payload, dict)
            ):
                provider_reconcile = dict(cached_payload)
                provider_reconcile["cache_hit"] = True
                provider_reconcile["cache_age_seconds"] = round(cache_age_seconds, 3)
        if not provider_reconcile:
            try:
                provider_reconcile = await reconcile_live_provider_orders(
                    session,
                    trader_id=trader_id,
                    commit=False,
                    broadcast=False,
                )
            except Exception as exc:
                logger.warning(
                    "Live provider reconcile before timeout cleanup failed for trader=%s",
                    trader_id,
                    exc_info=exc,
                )
                provider_reconcile = {
                    "provider_ready": False,
                    "error": str(exc),
                }
            _live_provider_reconcile_cache[cache_key] = (now_utc, dict(provider_reconcile))

    provider_ready_for_cleanup = bool(provider_reconcile.get("provider_ready", True))

    for source_key, source_config in source_configs.items():
        timeout_seconds = _source_open_order_timeout_seconds(source_config)
        if timeout_seconds is None:
            continue
        configured += 1
        if scope == "live" and not provider_ready_for_cleanup:
            suppressed += 1
            source_rows.append(
                {
                    "source": source_key,
                    "timeout_seconds": timeout_seconds,
                    "matched": 0,
                    "updated": 0,
                    "suppressed": True,
                    "suppressed_reason": "provider_unavailable",
                }
            )
            continue
        cooldown_key = f"{trader_id}:{scope}:{source_key}"
        cooldown_until = _open_order_timeout_cleanup_failure_cooldown_until.get(cooldown_key)
        if cooldown_until is not None and now_utc < cooldown_until:
            suppressed += 1
            source_rows.append(
                {
                    "source": source_key,
                    "timeout_seconds": timeout_seconds,
                    "matched": 0,
                    "updated": 0,
                    "suppressed": True,
                    "suppressed_until": cooldown_until.isoformat(),
                }
            )
            continue
        try:
            cleanup = await cleanup_trader_open_orders(
                session,
                trader_id=trader_id,
                scope=scope,
                max_age_seconds=timeout_seconds,
                source=source_key,
                require_unfilled=True,
                dry_run=False,
                target_status="cancelled",
                reason=f"max_open_order_timeout:{source_key}",
            )
            _open_order_timeout_cleanup_failure_cooldown_until.pop(cooldown_key, None)
        except Exception as exc:
            logger.warning(
                "Source open-order timeout cleanup failed for trader=%s source=%s",
                trader_id,
                source_key,
                exc_info=exc,
            )
            cooldown_until = now_utc + timedelta(seconds=_OPEN_ORDER_TIMEOUT_CLEANUP_FAILURE_COOLDOWN_SECONDS)
            _open_order_timeout_cleanup_failure_cooldown_until[cooldown_key] = cooldown_until
            errors.append(
                {
                    "source": source_key,
                    "timeout_seconds": timeout_seconds,
                    "error": str(exc),
                    "suppressed_until": cooldown_until.isoformat(),
                    "cooldown_seconds": _OPEN_ORDER_TIMEOUT_CLEANUP_FAILURE_COOLDOWN_SECONDS,
                }
            )
            continue

        source_updated = int(cleanup.get("updated", 0))
        updated += source_updated
        source_rows.append(
            {
                "source": source_key,
                "timeout_seconds": timeout_seconds,
                "matched": int(cleanup.get("matched", 0)),
                "updated": source_updated,
                "suppressed": False,
            }
        )

    return {
        "configured": configured,
        "updated": updated,
        "suppressed": suppressed,
        "provider_reconcile": provider_reconcile,
        "sources": source_rows,
        "errors": errors,
    }


def _normalize_portfolio_config(risk_limits: dict[str, Any]) -> dict[str, Any]:
    raw = risk_limits.get("portfolio")
    config = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(config.get("enabled", False)),
        "target_utilization_pct": float(
            max(
                1.0,
                min(
                    100.0,
                    safe_float(config.get("target_utilization_pct"), 100.0),
                ),
            )
        ),
        "max_source_exposure_pct": float(
            max(
                1.0,
                min(
                    100.0,
                    safe_float(config.get("max_source_exposure_pct"), 100.0),
                ),
            )
        ),
        "min_order_notional_usd": float(
            max(
                1.0,
                safe_float(config.get("min_order_notional_usd"), 10.0),
            )
        ),
    }


def _allocate_portfolio_notional(
    *,
    requested_size_usd: float,
    gross_exposure_cap_usd: float,
    current_gross_exposure_usd: float,
    current_source_exposure_usd: float,
    source_key: str,
    portfolio_config: dict[str, Any],
) -> dict[str, Any]:
    requested_size = float(max(0.0, safe_float(requested_size_usd, 0.0)))
    gross_cap = float(max(0.0, safe_float(gross_exposure_cap_usd, 0.0)))
    target_utilization_pct = float(portfolio_config.get("target_utilization_pct", 100.0))
    max_source_exposure_pct = float(portfolio_config.get("max_source_exposure_pct", 100.0))
    min_order_notional = float(portfolio_config.get("min_order_notional_usd", 10.0))

    target_gross_cap = gross_cap * (target_utilization_pct / 100.0)
    current_gross = float(max(0.0, safe_float(current_gross_exposure_usd, 0.0)))
    current_source = float(max(0.0, safe_float(current_source_exposure_usd, 0.0)))
    remaining_gross = float(max(0.0, target_gross_cap - current_gross))
    source_cap = float(target_gross_cap * (max_source_exposure_pct / 100.0))
    source_remaining = float(max(0.0, source_cap - current_source))
    allocated_size = float(max(0.0, min(requested_size, remaining_gross, source_remaining)))

    reason = "Portfolio allocation accepted requested size"
    allowed = True

    if requested_size <= 0:
        allowed = False
        reason = "Portfolio blocked: non-positive requested notional"
    elif gross_cap <= 0:
        allowed = False
        reason = "Portfolio blocked: max_gross_exposure_usd must be positive"
    elif target_gross_cap <= 0:
        allowed = False
        reason = "Portfolio blocked: target utilization leaves no allocatable gross exposure"
    elif remaining_gross <= 0:
        allowed = False
        reason = "Portfolio blocked: no remaining gross exposure budget"
    elif source_remaining <= 0:
        allowed = False
        reason = f"Portfolio blocked: source exposure cap reached for '{source_key or 'unknown'}'"
    elif allocated_size < min_order_notional:
        allowed = False
        reason = (
            "Portfolio blocked: allocated notional below minimum order size "
            f"({allocated_size:.2f} < {min_order_notional:.2f})"
        )
    elif allocated_size < requested_size:
        reason = f"Portfolio capped: allocated {allocated_size:.2f} from requested {requested_size:.2f}"

    return {
        "allowed": bool(allowed),
        "reason": reason,
        "size_usd": allocated_size if allowed else 0.0,
        "requested_size_usd": requested_size,
        "target_gross_cap_usd": target_gross_cap,
        "remaining_gross_cap_usd": remaining_gross,
        "source_key": source_key,
        "source_cap_usd": source_cap,
        "source_exposure_usd": current_source,
        "source_remaining_usd": source_remaining,
        "min_order_notional_usd": min_order_notional,
        "target_utilization_pct": target_utilization_pct,
        "max_source_exposure_pct": max_source_exposure_pct,
    }


def _strategy_instance_from_loaded(candidate: Any) -> Any:
    if candidate is None:
        return None
    instance = getattr(candidate, "instance", None)
    if instance is not None:
        return instance
    return candidate


def _versioned_strategy_alias(strategy_key: str, strategy_version: int) -> str:
    return f"{str(strategy_key or '').strip().lower()}__v{int(strategy_version)}"


def _load_versioned_strategy_instance(
    *,
    strategy_key: str,
    strategy_version: int,
    source_code: str,
    config: dict[str, Any] | None,
) -> tuple[Any | None, str | None]:
    alias = _versioned_strategy_alias(strategy_key, strategy_version)
    loaded = strategy_loader.get_strategy(alias)
    if loaded is None:
        try:
            strategy_loader.load(alias, source_code, config or None)
        except StrategyValidationError as exc:
            return None, str(exc)
        except Exception as exc:
            return None, str(exc)
        loaded = strategy_loader.get_strategy(alias)
    instance = _strategy_instance_from_loaded(loaded)
    if instance is None:
        return None, "Strategy cache miss"
    try:
        instance.key = str(strategy_key or "").strip().lower()
    except Exception:
        pass
    return instance, None


async def _try_acquire_orchestrator_cycle_lock(session: Any) -> bool:
    """Acquire a cross-process lock when running on PostgreSQL.

    For non-PostgreSQL dialects, fall back to
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
    global _orchestrator_lock_session

    if _orchestrator_lock_session is not None:
        return True

    session = AsyncSessionLocal()
    keep_lock_session = False
    try:
        acquired = await _try_acquire_orchestrator_cycle_lock(session)
        if not acquired:
            return False

        if _session_dialect_name(session) == "postgresql":
            _orchestrator_lock_session = session
            keep_lock_session = True
            logger.info("Acquired orchestrator cross-process cycle lock")

        return True
    finally:
        if not keep_lock_session:
            await session.close()


async def _release_orchestrator_cycle_lock_owner() -> None:
    """Release owned lock session on shutdown/error."""
    global _orchestrator_lock_session
    if _orchestrator_lock_session is None:
        return

    session = _orchestrator_lock_session
    _orchestrator_lock_session = None
    try:
        await _release_orchestrator_cycle_lock(session)
        logger.info("Released orchestrator cross-process cycle lock")
    finally:
        await session.close()


def _query_sources_for_configs(source_configs: dict[str, dict[str, Any]]) -> list[str]:
    if not source_configs:
        return []
    return sorted(source_configs.keys())


def _query_strategy_types_for_configs(source_configs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    if not source_configs:
        return {}
    out: dict[str, list[str]] = {}
    for source_key, source_config in source_configs.items():
        allowed = _accepted_signal_strategy_types(source_config)
        if not allowed:
            continue
        out[source_key] = sorted(allowed)
    return out


def _normalize_crypto_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


def _normalize_crypto_timeframe(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    compact = raw.replace("_", "").replace("-", "").replace(" ", "")
    mapping = {
        "5m": "5m",
        "5min": "5m",
        "5minute": "5m",
        "5minutes": "5m",
        "15m": "15m",
        "15min": "15m",
        "15minute": "15m",
        "15minutes": "15m",
        "1h": "1h",
        "1hr": "1h",
        "1hour": "1h",
        "60m": "1h",
        "60min": "1h",
        "4h": "4h",
        "4hr": "4h",
        "4hour": "4h",
        "240m": "4h",
        "240min": "4h",
    }
    return mapping.get(compact, compact)


def _normalize_crypto_scope_values(raw: Any, normalizer) -> set[str]:
    if isinstance(raw, (list, tuple, set)):
        values = raw
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        values = []
    normalized: set[str] = set()
    for item in values:
        value = normalizer(item)
        if value:
            normalized.add(value)
    return normalized


def _crypto_signal_dimensions(signal: Any) -> tuple[str, str]:
    strategy_context = getattr(signal, "strategy_context_json", None)
    payload = getattr(signal, "payload_json", None)
    context = strategy_context if isinstance(strategy_context, dict) else {}
    body = payload if isinstance(payload, dict) else {}

    asset_raw = (
        context.get("asset")
        or context.get("coin")
        or context.get("symbol")
        or body.get("asset")
        or body.get("coin")
        or body.get("symbol")
    )
    timeframe_raw = (
        context.get("timeframe")
        or context.get("cadence")
        or context.get("interval")
        or body.get("timeframe")
        or body.get("cadence")
        or body.get("interval")
    )
    return _normalize_crypto_asset(asset_raw), _normalize_crypto_timeframe(timeframe_raw)


def _crypto_signal_in_scope(signal: Any, source_config: dict[str, Any]) -> bool:
    strategy_params = dict(source_config.get("strategy_params") or {})
    include_assets = _normalize_crypto_scope_values(
        strategy_params.get("include_assets"),
        _normalize_crypto_asset,
    )
    exclude_assets = _normalize_crypto_scope_values(
        strategy_params.get("exclude_assets"),
        _normalize_crypto_asset,
    )
    include_timeframes = _normalize_crypto_scope_values(
        strategy_params.get("include_timeframes"),
        _normalize_crypto_timeframe,
    )
    exclude_timeframes = _normalize_crypto_scope_values(
        strategy_params.get("exclude_timeframes"),
        _normalize_crypto_timeframe,
    )

    asset, timeframe = _crypto_signal_dimensions(signal)

    if include_assets and (not asset or asset not in include_assets):
        return False
    if asset and asset in exclude_assets:
        return False
    if include_timeframes and (not timeframe or timeframe not in include_timeframes):
        return False
    if timeframe and timeframe in exclude_timeframes:
        return False
    return True


def _signal_wallets(signal: Any) -> set[str]:
    return StrategySDK.extract_trader_signal_wallets(signal)


def _is_live_provider_infra_error(error_text: Any) -> bool:
    text = str(error_text or "").strip().lower()
    if not text:
        return False
    if "not enough balance / allowance" in text:
        return False
    return any(marker in text for marker in _LIVE_PROVIDER_INFRA_ERROR_MARKERS)


def _live_provider_health_event_due(trader_id: str, now: datetime) -> bool:
    cooldown_until = _live_provider_block_event_cooldown_until.get(trader_id)
    if cooldown_until is not None and now < cooldown_until:
        return False
    _live_provider_block_event_cooldown_until[trader_id] = now + timedelta(
        seconds=_LIVE_PROVIDER_BLOCK_EVENT_COOLDOWN_SECONDS
    )
    return True


def _live_risk_clamp_event_due(trader_id: str, now: datetime) -> bool:
    cooldown_until = _live_risk_clamp_event_cooldown_until.get(trader_id)
    if cooldown_until is not None and now < cooldown_until:
        return False
    _live_risk_clamp_event_cooldown_until[trader_id] = now + timedelta(seconds=_LIVE_RISK_CLAMP_EVENT_COOLDOWN_SECONDS)
    return True


def _provider_reconcile_has_material_changes(payload: dict[str, Any]) -> bool:
    material_keys = (
        "status_changes",
        "session_status_changes",
        "notional_updates",
        "price_updates",
        "terminal_session_cancels",
    )
    return any(int(payload.get(key, 0) or 0) > 0 for key in material_keys)


async def _live_provider_failure_snapshot(
    session: Any,
    *,
    trader_id: str,
    window_seconds: int,
) -> dict[str, Any]:
    if not hasattr(session, "execute"):
        return {"count": 0, "window_seconds": window_seconds, "errors": []}

    now = utcnow()
    cutoff = now - timedelta(seconds=max(30, int(window_seconds)))
    mode_key_expr = func.lower(func.coalesce(TraderOrder.mode, ""))
    rows = (
        await session.execute(
            select(
                TraderOrder.id,
                TraderOrder.status,
                TraderOrder.error_message,
                TraderOrder.payload_json,
                TraderOrder.updated_at,
            )
            .where(TraderOrder.trader_id == trader_id)
            .where(mode_key_expr == "live")
            .where(TraderOrder.updated_at.is_not(None))
            .where(TraderOrder.updated_at >= cutoff)
            .order_by(TraderOrder.updated_at.desc())
            .limit(200)
        )
    ).all()

    failures: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row.payload_json or {})
        pending_live_exit = payload.get("pending_live_exit")
        pending_live_exit = pending_live_exit if isinstance(pending_live_exit, dict) else {}
        provider_reconcile = payload.get("provider_reconciliation")
        provider_reconcile = provider_reconcile if isinstance(provider_reconcile, dict) else {}
        candidates = [
            row.error_message,
            payload.get("error_message"),
            payload.get("error"),
            pending_live_exit.get("last_error"),
            provider_reconcile.get("error"),
        ]
        matched_error = next((text for text in candidates if _is_live_provider_infra_error(text)), None)
        if matched_error is None:
            continue
        failures.append(
            {
                "order_id": str(row.id or "").strip() or None,
                "status": str(row.status or "").strip().lower() or None,
                "updated_at": row.updated_at.isoformat() if row.updated_at is not None else None,
                "error": str(matched_error),
            }
        )

    return {
        "count": len(failures),
        "window_seconds": int(max(30, int(window_seconds))),
        "errors": failures[:8],
    }


def _apply_live_risk_clamps(
    effective_risk_limits: dict[str, Any],
    live_risk_clamps: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}

    enforce_allow_averaging_off = bool(
        live_risk_clamps.get(
            "enforce_allow_averaging_off",
            DEFAULT_LIVE_RISK_CLAMPS["enforce_allow_averaging_off"],
        )
    )
    if enforce_allow_averaging_off:
        configured_allow_averaging = bool(effective_risk_limits.get("allow_averaging", False))
        if configured_allow_averaging:
            changes["allow_averaging"] = {"configured": configured_allow_averaging, "effective": False}
        effective_risk_limits["allow_averaging"] = False

    configured_cooldown_seconds = max(0, safe_int(effective_risk_limits.get("cooldown_seconds"), 0))
    min_cooldown_seconds = max(
        0,
        safe_int(
            live_risk_clamps.get("min_cooldown_seconds"),
            DEFAULT_LIVE_RISK_CLAMPS["min_cooldown_seconds"],
        ),
    )
    clamped_cooldown_seconds = max(configured_cooldown_seconds, min_cooldown_seconds)
    if clamped_cooldown_seconds != configured_cooldown_seconds:
        changes["cooldown_seconds"] = {
            "configured": configured_cooldown_seconds,
            "effective": clamped_cooldown_seconds,
        }
    effective_risk_limits["cooldown_seconds"] = clamped_cooldown_seconds

    configured_max_consecutive_losses = max(1, safe_int(effective_risk_limits.get("max_consecutive_losses"), 4))
    max_consecutive_losses_cap = max(
        1,
        safe_int(
            live_risk_clamps.get("max_consecutive_losses_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_consecutive_losses_cap"],
        ),
    )
    clamped_max_consecutive_losses = min(configured_max_consecutive_losses, max_consecutive_losses_cap)
    if clamped_max_consecutive_losses != configured_max_consecutive_losses:
        changes["max_consecutive_losses"] = {
            "configured": configured_max_consecutive_losses,
            "effective": clamped_max_consecutive_losses,
        }
    effective_risk_limits["max_consecutive_losses"] = clamped_max_consecutive_losses

    configured_max_open_orders = max(1, safe_int(effective_risk_limits.get("max_open_orders"), 20))
    max_open_orders_cap = max(
        1,
        safe_int(
            live_risk_clamps.get("max_open_orders_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_open_orders_cap"],
        ),
    )
    clamped_max_open_orders = min(configured_max_open_orders, max_open_orders_cap)
    if clamped_max_open_orders != configured_max_open_orders:
        changes["max_open_orders"] = {
            "configured": configured_max_open_orders,
            "effective": clamped_max_open_orders,
        }
    effective_risk_limits["max_open_orders"] = clamped_max_open_orders

    configured_max_open_positions = max(1, safe_int(effective_risk_limits.get("max_open_positions"), 12))
    max_open_positions_cap = max(
        1,
        safe_int(
            live_risk_clamps.get("max_open_positions_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_open_positions_cap"],
        ),
    )
    clamped_max_open_positions = min(configured_max_open_positions, max_open_positions_cap)
    if clamped_max_open_positions != configured_max_open_positions:
        changes["max_open_positions"] = {
            "configured": configured_max_open_positions,
            "effective": clamped_max_open_positions,
        }
    effective_risk_limits["max_open_positions"] = clamped_max_open_positions

    configured_max_trade_notional = max(1.0, safe_float(effective_risk_limits.get("max_trade_notional_usd"), 350.0))
    max_trade_notional_usd_cap = max(
        1.0,
        safe_float(
            live_risk_clamps.get("max_trade_notional_usd_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_trade_notional_usd_cap"],
        ),
    )
    clamped_max_trade_notional = min(float(configured_max_trade_notional), float(max_trade_notional_usd_cap))
    if clamped_max_trade_notional != float(configured_max_trade_notional):
        changes["max_trade_notional_usd"] = {
            "configured": float(configured_max_trade_notional),
            "effective": clamped_max_trade_notional,
        }
    effective_risk_limits["max_trade_notional_usd"] = clamped_max_trade_notional

    configured_max_orders_per_cycle = max(1, safe_int(effective_risk_limits.get("max_orders_per_cycle"), 50))
    max_orders_per_cycle_cap = max(
        1,
        safe_int(
            live_risk_clamps.get("max_orders_per_cycle_cap"),
            DEFAULT_LIVE_RISK_CLAMPS["max_orders_per_cycle_cap"],
        ),
    )
    clamped_max_orders_per_cycle = min(configured_max_orders_per_cycle, max_orders_per_cycle_cap)
    if clamped_max_orders_per_cycle != configured_max_orders_per_cycle:
        changes["max_orders_per_cycle"] = {
            "configured": configured_max_orders_per_cycle,
            "effective": clamped_max_orders_per_cycle,
        }
    effective_risk_limits["max_orders_per_cycle"] = clamped_max_orders_per_cycle

    enforce_halt_on_consecutive_losses = bool(
        live_risk_clamps.get(
            "enforce_halt_on_consecutive_losses",
            DEFAULT_LIVE_RISK_CLAMPS["enforce_halt_on_consecutive_losses"],
        )
    )
    if enforce_halt_on_consecutive_losses:
        configured_halt = bool(effective_risk_limits.get("halt_on_consecutive_losses", False))
        if not configured_halt:
            changes["halt_on_consecutive_losses"] = {
                "configured": configured_halt,
                "effective": True,
            }
        effective_risk_limits["halt_on_consecutive_losses"] = True
    return changes


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
        paper_simulation = payload.get("paper_simulation") if isinstance(payload.get("paper_simulation"), dict) else {}
        execution_fee_usd = safe_float(paper_simulation.get("estimated_fee_usd"), 0.0) or 0.0
        execution_slippage_usd = safe_float(paper_simulation.get("slippage_usd"), 0.0) or 0.0
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
                    "paper_simulation": paper_simulation,
                },
                execution_fee_usd=execution_fee_usd,
                execution_slippage_usd=execution_slippage_usd,
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
    normalized_scope = StrategySDK.validate_trader_scope_config(traders_scope)
    modes = {
        str(mode or "").strip().lower() for mode in (normalized_scope.get("modes") or []) if str(mode or "").strip()
    }
    tracked_wallets: set[str] = set()
    pool_wallets: set[str] = set()
    group_wallets: set[str] = set()

    if "tracked" in modes:
        tracked_rows = (await session.execute(select(TrackedWallet.address))).scalars().all()
        tracked_wallets = {_normalize_wallet(address) for address in tracked_rows if _normalize_wallet(address)}

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
        pool_wallets = {_normalize_wallet(address) for address in pool_rows if _normalize_wallet(address)}

    if "group" in modes and list(normalized_scope.get("group_ids") or []):
        group_rows = (
            (
                await session.execute(
                    select(TraderGroupMember.wallet_address).where(
                        TraderGroupMember.group_id.in_(list(normalized_scope.get("group_ids") or []))
                    )
                )
            )
            .scalars()
            .all()
        )
        group_wallets = {_normalize_wallet(address) for address in group_rows if _normalize_wallet(address)}

    return StrategySDK.build_trader_scope_runtime_context(
        normalized_scope,
        tracked_wallets=tracked_wallets,
        pool_wallets=pool_wallets,
        group_wallets=group_wallets,
    )


def _signal_matches_traders_scope(signal: Any, scope_context: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    return StrategySDK.match_trader_signal_scope(signal, scope_context)


async def _persist_trader_cycle_heartbeat(session: Any, trader_id: str) -> None:
    row = await session.get(Trader, trader_id)
    if row is None:
        return
    now = utcnow()
    row.last_run_at = now
    row.requested_run_at = None
    row.updated_at = now
    await _commit_with_retry(
        session,
        retry_attempts=2,
        base_delay_seconds=0.05,
        max_delay_seconds=0.1,
    )


def _signal_cursor_timestamp(signal: Any) -> Any:
    updated_at = getattr(signal, "updated_at", None)
    if updated_at is not None:
        return updated_at
    return getattr(signal, "created_at", None)


def _signal_sort_key(signal: Any) -> tuple[datetime, str]:
    ts = _signal_cursor_timestamp(signal)
    if not isinstance(ts, datetime):
        ts = datetime(1970, 1, 1)
    elif ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    signal_id = str(getattr(signal, "id", "") or "")
    return ts, signal_id


async def _list_triggered_trade_signals(
    session: Any,
    *,
    trader_id: str,
    signal_ids_by_source: dict[str, list[str]],
    sources: list[str],
    strategy_types_by_source: dict[str, list[str]],
    cursor_created_at: datetime | None,
    cursor_signal_id: str | None,
    statuses: list[str],
    limit: int,
) -> list[TradeSignal]:
    if not signal_ids_by_source or not sources:
        return []

    normalized_signal_ids: list[str] = []
    seen_ids: set[str] = set()
    global_signal_ids = signal_ids_by_source.get("__all__") or []
    for raw_signal_id in global_signal_ids:
        signal_id = str(raw_signal_id or "").strip()
        if not signal_id or signal_id in seen_ids:
            continue
        seen_ids.add(signal_id)
        normalized_signal_ids.append(signal_id)
    for source_key in sources:
        source_signal_ids = signal_ids_by_source.get(source_key) or []
        for raw_signal_id in source_signal_ids:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in seen_ids:
                continue
            seen_ids.add(signal_id)
            normalized_signal_ids.append(signal_id)

    if not normalized_signal_ids:
        return []

    signal_sort_ts = func.coalesce(TradeSignal.updated_at, TradeSignal.created_at)
    latest_consumed_at = (
        select(func.max(TraderSignalConsumption.consumed_at))
        .where(
            TraderSignalConsumption.trader_id == trader_id,
            TraderSignalConsumption.signal_id == TradeSignal.id,
        )
        .correlate(TradeSignal)
        .scalar_subquery()
    )
    query = (
        select(TradeSignal)
        .where(TradeSignal.id.in_(normalized_signal_ids))
        .where(or_(latest_consumed_at.is_(None), signal_sort_ts > latest_consumed_at))
    )
    normalized_cursor_created_at = cursor_created_at
    if isinstance(normalized_cursor_created_at, datetime) and normalized_cursor_created_at.tzinfo is not None:
        normalized_cursor_created_at = normalized_cursor_created_at.astimezone(timezone.utc).replace(tzinfo=None)
    normalized_cursor_signal_id = str(cursor_signal_id or "").strip()
    if normalized_cursor_created_at is not None:
        if normalized_cursor_signal_id:
            query = query.where(
                or_(
                    signal_sort_ts > normalized_cursor_created_at,
                    and_(signal_sort_ts == normalized_cursor_created_at, TradeSignal.id > normalized_cursor_signal_id),
                )
            )
        else:
            query = query.where(signal_sort_ts > normalized_cursor_created_at)
    now_naive = utcnow().replace(tzinfo=None)
    query = query.where(or_(TradeSignal.expires_at.is_(None), TradeSignal.expires_at >= now_naive))

    normalized_statuses = [str(status or "").strip().lower() for status in statuses if str(status or "").strip()]
    if normalized_statuses:
        query = query.where(func.lower(func.coalesce(TradeSignal.status, "")).in_(normalized_statuses))

    normalized_sources = [normalize_source_key(source) for source in sources if normalize_source_key(source)]
    if normalized_sources:
        query = query.where(func.lower(func.coalesce(TradeSignal.source, "")).in_(normalized_sources))

    source_strategy_clauses = []
    for source_key, strategy_types in strategy_types_by_source.items():
        if not strategy_types:
            continue
        normalized_strategy_types = [str(item or "").strip().lower() for item in strategy_types if str(item or "").strip()]
        if not normalized_strategy_types:
            continue
        source_strategy_clauses.append(
            and_(
                func.lower(func.coalesce(TradeSignal.source, "")) == normalize_source_key(source_key),
                func.lower(func.coalesce(TradeSignal.strategy_type, "")).in_(normalized_strategy_types),
            )
        )
    if source_strategy_clauses:
        query = query.where(or_(*source_strategy_clauses))

    rows = list((await session.execute(query)).scalars().all())
    rows.sort(key=_signal_sort_key)
    return rows[: max(1, min(limit, 5000))]


async def _emit_cycle_heartbeat_if_due(
    session: Any,
    *,
    trader_id: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    now = utcnow()
    last_emitted = _trader_cycle_heartbeat_last_emitted.get(trader_id)
    if last_emitted is not None:
        elapsed_seconds = (now - last_emitted).total_seconds()
        if elapsed_seconds < _TRADER_CYCLE_HEARTBEAT_EVENT_INTERVAL_SECONDS:
            return

    await create_trader_event(
        session,
        trader_id=trader_id,
        event_type="cycle_heartbeat",
        source="worker",
        message=message,
        payload=payload or {},
        commit=False,
    )
    _trader_cycle_heartbeat_last_emitted[trader_id] = now


def _market_winner_value(market_info: Any) -> str:
    if not isinstance(market_info, dict):
        return ""
    winner_raw = (
        market_info.get("winning_outcome")
        if market_info.get("winning_outcome") not in (None, "")
        else market_info.get("winner")
    )
    return str(winner_raw or "").strip()


def _market_outcome_prices(market_info: Any) -> list[float]:
    if not isinstance(market_info, dict):
        return []
    prices = market_info.get("outcome_prices")
    if not isinstance(prices, list):
        prices = market_info.get("outcomePrices")
    if not isinstance(prices, list):
        return []
    out: list[float] = []
    for raw in prices[:2]:
        parsed = safe_float(raw)
        if parsed is None:
            out.append(-1.0)
            continue
        out.append(min(1.0, max(0.0, parsed)))
    return out


def _is_terminal_market_state(
    market_info: Any,
    *,
    now: datetime,
    settle_floor: float = 0.98,
) -> bool:
    if not isinstance(market_info, dict):
        return False
    if bool(market_info.get("closed", False)):
        return True
    if _market_winner_value(market_info):
        return True

    is_tradable = polymarket_client.is_market_tradable(market_info, now=now)
    settle_floor = min(1.0, max(0.5, settle_floor))
    settle_ceiling = max(0.0, 1.0 - settle_floor)
    prices = _market_outcome_prices(market_info)
    if not is_tradable and len(prices) >= 2:
        yes_price = prices[0]
        no_price = prices[1]
        if yes_price >= settle_floor and no_price <= settle_ceiling:
            return True
        if no_price >= settle_floor and yes_price <= settle_ceiling:
            return True
    return False


async def _run_terminal_stale_order_watchdog(session: Any, *, now: datetime | None = None) -> dict[str, Any]:
    global _terminal_stale_order_last_checked_at

    now_utc = now or utcnow()
    if _terminal_stale_order_last_checked_at is not None:
        elapsed_seconds = (now_utc - _terminal_stale_order_last_checked_at).total_seconds()
        if elapsed_seconds < _TERMINAL_STALE_ORDER_CHECK_INTERVAL_SECONDS:
            return {"checked": False, "stale": 0, "alerted": 0}
    _terminal_stale_order_last_checked_at = now_utc

    status_key_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    candidates = (
        (
            await session.execute(
                select(TraderOrder).where(status_key_expr.in_(tuple(_PAPER_ACTIVE_ORDER_STATUSES))).limit(2000)
            )
        )
        .scalars()
        .all()
    )
    if not candidates:
        return {"checked": True, "stale": 0, "alerted": 0}

    market_info_by_id = await load_market_info_for_orders(list(candidates))
    stale_rows: list[dict[str, Any]] = []
    for row in candidates:
        market_id = str(row.market_id or "").strip()
        if not market_id:
            continue
        market_info = market_info_by_id.get(market_id)
        if not _is_terminal_market_state(market_info, now=now_utc):
            continue
        age_anchor = row.executed_at or row.updated_at or row.created_at
        if age_anchor is None:
            continue
        age_minutes = max(0.0, (now_utc - age_anchor).total_seconds() / 60.0)
        if age_minutes < float(_TERMINAL_STALE_ORDER_MIN_AGE_MINUTES):
            continue
        stale_rows.append(
            {
                "order_id": str(row.id),
                "trader_id": str(row.trader_id or ""),
                "mode": str(row.mode or ""),
                "status": str(row.status or ""),
                "market_id": market_id,
                "age_minutes": round(age_minutes, 2),
            }
        )

    if not stale_rows:
        return {"checked": True, "stale": 0, "alerted": 0}

    stale_cutoff = now_utc - timedelta(hours=24)
    for order_id, emitted_at in list(_terminal_stale_order_alert_last_emitted.items()):
        if emitted_at < stale_cutoff:
            _terminal_stale_order_alert_last_emitted.pop(order_id, None)

    alert_rows: list[dict[str, Any]] = []
    for detail in stale_rows:
        order_id = detail["order_id"]
        last_emitted = _terminal_stale_order_alert_last_emitted.get(order_id)
        if last_emitted is not None:
            elapsed = (now_utc - last_emitted).total_seconds()
            if elapsed < _TERMINAL_STALE_ORDER_ALERT_COOLDOWN_SECONDS:
                continue
        _terminal_stale_order_alert_last_emitted[order_id] = now_utc
        alert_rows.append(detail)

    if not alert_rows:
        return {"checked": True, "stale": len(stale_rows), "alerted": 0}

    affected_traders = sorted({row["trader_id"] for row in alert_rows if row["trader_id"]})
    await create_trader_event(
        session,
        trader_id=None,
        event_type="terminal_stale_orders",
        severity="warn",
        source="worker",
        message=f"Detected {len(alert_rows)} stale active order(s) on terminal markets.",
        payload={
            "stale_count": len(stale_rows),
            "alerted_count": len(alert_rows),
            "affected_traders": affected_traders,
            "orders": alert_rows[:50],
            "min_age_minutes": _TERMINAL_STALE_ORDER_MIN_AGE_MINUTES,
            "check_interval_seconds": _TERMINAL_STALE_ORDER_CHECK_INTERVAL_SECONDS,
        },
        commit=True,
    )

    return {
        "checked": True,
        "stale": len(stale_rows),
        "alerted": len(alert_rows),
        "affected_traders": len(affected_traders),
    }


async def _run_trader_once(
    trader: dict[str, Any],
    control: dict[str, Any],
    *,
    process_signals: bool = True,
    trigger_signal_ids_by_source: dict[str, list[str]] | None = None,
) -> tuple[int, int, int]:
    decisions_written = 0
    orders_written = 0
    processed_signals = 0
    prefiltered_signals = 0
    prefiltered_by_reason: dict[str, int] = {}
    crypto_scope_prefiltered_dimensions: dict[str, int] = {}

    async with AsyncSessionLocal() as session:
        trader_id = str(trader["id"])
        source_configs = _normalize_source_configs(trader)
        effective_process_signals = bool(process_signals)
        default_strategy_params: dict[str, Any] = {}
        sources: list[str] = []
        strategy_types_by_source: dict[str, list[str]] = {}
        source_entry_modes: dict[str, bool] = {}
        if source_configs:
            default_source_config = next(iter(source_configs.values()))
            default_strategy_params = dict(default_source_config.get("strategy_params") or {})
            sources = _query_sources_for_configs(source_configs)
            strategy_types_by_source = _query_strategy_types_for_configs(source_configs)
            source_entry_modes = {
                source_key: _source_config_allows_new_entries(source_config)
                for source_key, source_config in source_configs.items()
            }
        elif effective_process_signals:
            await _emit_cycle_heartbeat_if_due(
                session,
                trader_id=trader_id,
                message="Idle cycle: no source configs configured.",
            )
            effective_process_signals = False
        risk_limits_raw = trader.get("risk_limits")
        risk_limits = dict(risk_limits_raw or {})
        if isinstance(risk_limits_raw, dict):
            normalized_limits = StrategySDK.validate_trader_risk_config(risk_limits_raw)
            for key in list(risk_limits.keys()):
                key_name = str(key)
                if not key_name:
                    continue
                if key_name in normalized_limits:
                    risk_limits[key_name] = normalized_limits[key_name]
        position_cap_scope = str(risk_limits.get("position_cap_scope") or "market_direction").strip().lower()
        if position_cap_scope not in {"market_direction", "market", "asset_timeframe"}:
            position_cap_scope = "market_direction"
        metadata = StrategySDK.validate_trader_runtime_metadata(trader.get("metadata"))
        loss_streak_reset_at = _parse_iso(str(metadata.get(_LOSS_STREAK_RESET_AT_KEY) or "").strip())
        run_mode = str(control.get("mode") or "paper").strip().lower()
        resume_policy = _normalize_resume_policy(metadata.get("resume_policy"))
        cursor_created_at, cursor_signal_id = await get_trader_signal_cursor(
            session,
            trader_id=trader_id,
        )
        prefetched_signals: list[Any] | None = None
        stream_trigger_mode = bool(trigger_signal_ids_by_source)
        if effective_process_signals and sources:
            if stream_trigger_mode:
                prefetched_signals = await _list_triggered_trade_signals(
                    session,
                    trader_id=trader_id,
                    signal_ids_by_source=trigger_signal_ids_by_source or {},
                    sources=sources,
                    strategy_types_by_source=strategy_types_by_source,
                    cursor_created_at=cursor_created_at,
                    cursor_signal_id=cursor_signal_id,
                    statuses=["pending", "selected"],
                    limit=max(
                        _STANDARD_MAX_SIGNALS_PER_CYCLE,
                        _HIGH_FREQUENCY_MAX_SIGNALS_PER_CYCLE,
                    ),
                )
            else:
                pending_preview = await list_unconsumed_trade_signals(
                    session,
                    trader_id=trader_id,
                    sources=sources,
                    statuses=["pending", "selected"],
                    strategy_types_by_source=strategy_types_by_source,
                    cursor_created_at=cursor_created_at,
                    cursor_signal_id=cursor_signal_id,
                    limit=1,
                )
                if pending_preview:
                    prefetched_signals = pending_preview

            if not prefetched_signals:
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
                        await _emit_cycle_heartbeat_if_due(
                            session,
                            trader_id=trader_id,
                            message="Idle cycle: no pending signals.",
                            payload={
                                "idle_maintenance_interval_seconds": idle_maintenance_interval_seconds,
                                "elapsed_seconds": elapsed_seconds,
                                "triggered_cycle": stream_trigger_mode,
                            },
                        )
                        effective_process_signals = False
                if effective_process_signals:
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
        elif run_mode == "shadow":
            force_flatten = resume_policy == "flatten_then_start"
            lifecycle_result = await reconcile_shadow_positions(
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
                    event_type="shadow_positions_closed",
                    source="worker",
                    message=f"Closed {closed_positions} shadow position(s)",
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
            # Live order/position reconciliation is handled by
            # workers.trader_reconciliation_worker so this loop stays focused
            # on strategy selection and execution.
            pass
        if run_mode != "live":
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
        timeout_cleanup = await _enforce_source_open_order_timeouts(
            session,
            trader_id=trader_id,
            run_mode=run_mode,
            source_configs=source_configs,
        )
        provider_reconcile_payload = timeout_cleanup.get("provider_reconcile")
        provider_reconcile_payload = (
            dict(provider_reconcile_payload) if isinstance(provider_reconcile_payload, dict) else {}
        )
        if run_mode == "live" and provider_reconcile_payload:
            provider_active_seen = int(provider_reconcile_payload.get("active_seen", 0) or 0)
            provider_status_changes = int(provider_reconcile_payload.get("status_changes", 0) or 0)
            provider_updates = int(provider_reconcile_payload.get("updated_orders", 0) or 0)
            provider_ready = bool(provider_reconcile_payload.get("provider_ready", True))
            provider_material_changes = _provider_reconcile_has_material_changes(provider_reconcile_payload)
            if (not provider_ready) or provider_material_changes:
                provider_severity = "warn" if not provider_ready else "info"
                provider_message = (
                    "Provider reconcile unavailable before timeout cleanup"
                    if not provider_ready
                    else (
                        f"Provider reconcile active_seen={provider_active_seen} "
                        f"status_changes={provider_status_changes} updated_orders={provider_updates}"
                    )
                )
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="provider_reconcile",
                    severity=provider_severity,
                    source="worker",
                    message=provider_message,
                    payload=provider_reconcile_payload,
                    commit=False,
                )
        if timeout_cleanup.get("updated", 0) > 0:
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="open_order_timeout_cleanup",
                source="worker",
                message=f"Cancelled {int(timeout_cleanup['updated'])} stale unfilled order(s)",
                payload=timeout_cleanup,
            )
        if timeout_cleanup.get("errors"):
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="open_order_timeout_cleanup_failed",
                severity="warn",
                source="worker",
                    message="One or more source open-order timeout cleanups failed",
                    payload=timeout_cleanup,
                )
        suppressed_sources = [
            dict(row)
            for row in (timeout_cleanup.get("sources") or [])
            if isinstance(row, dict) and bool(row.get("suppressed"))
        ]
        if suppressed_sources:
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="timeout_cleanup_suppressed",
                severity="warn",
                source="worker",
                message=f"Suppressed timeout cleanup for {len(suppressed_sources)} source(s)",
                payload={
                    "suppressed": int(timeout_cleanup.get("suppressed", len(suppressed_sources)) or 0),
                    "sources": suppressed_sources,
                    "provider_reconcile": provider_reconcile_payload,
                },
                commit=False,
            )
        open_positions = await get_open_position_count_for_trader(
            session,
            trader_id,
            mode=run_mode,
            position_cap_scope=position_cap_scope,
        )
        open_order_count = await get_open_order_count_for_trader(
            session,
            trader_id,
            mode=run_mode,
        )
        control_settings = dict(control.get("settings") or {})
        global_runtime_settings = dict(control_settings.get("global_runtime") or {})
        pending_live_exit_guard_settings = dict(global_runtime_settings.get("pending_live_exit_guard") or {})
        pending_live_exit_max_allowed = max(
            0,
            safe_int(
                pending_live_exit_guard_settings.get("max_pending_exits"),
                int(DEFAULT_PENDING_LIVE_EXIT_GUARD["max_pending_exits"]),
            ),
        )
        pending_live_exit_identity_guard_enabled = bool(
            pending_live_exit_guard_settings.get(
                "identity_guard_enabled",
                DEFAULT_PENDING_LIVE_EXIT_GUARD["identity_guard_enabled"],
            )
        )
        pending_live_exit_terminal_statuses = pending_live_exit_guard_settings.get("terminal_statuses")
        if not isinstance(pending_live_exit_terminal_statuses, list) or not pending_live_exit_terminal_statuses:
            pending_live_exit_terminal_statuses = list(DEFAULT_PENDING_LIVE_EXIT_GUARD["terminal_statuses"])
        live_provider_health_settings = dict(
            global_runtime_settings.get("live_provider_health") or DEFAULT_LIVE_PROVIDER_HEALTH
        )
        live_market_context_settings = dict(
            global_runtime_settings.get("live_market_context") or DEFAULT_LIVE_MARKET_CONTEXT
        )
        pending_live_exit_summary = {
            "count": 0,
            "order_ids": [],
            "market_ids": [],
            "signal_ids": [],
            "statuses": {},
            "identities": [],
            "identity_keys": [],
        }
        if run_mode == "live":
            pending_live_exit_summary = await get_pending_live_exit_summary_for_trader(
                session,
                trader_id,
                mode=run_mode,
                terminal_statuses=pending_live_exit_terminal_statuses,
            )
        pending_live_exit_count = int(pending_live_exit_summary.get("count", 0) or 0)
        effective_open_positions = max(open_positions, open_order_count)
        open_market_ids = await get_open_market_ids_for_trader(
            session,
            trader_id,
            mode=run_mode,
        )

        block_entries_reason = None
        block_entries_payload: dict[str, Any] = {
            "resume_policy": resume_policy,
            "mode": run_mode,
            "open_positions": open_positions,
        }
        block_entries_event_type = "trader_resume_policy"
        block_entries_event_severity = "warn" if run_mode in {"live", "shadow"} else "info"
        if resume_policy == "manage_only":
            block_entries_reason = "Resume policy manage_only blocks new entries"
        elif resume_policy == "flatten_then_start" and open_positions > 0:
            if run_mode == "paper":
                block_entries_reason = (
                    f"Resume policy flatten_then_start waiting to flatten {open_positions} open paper position(s)"
                )
            elif run_mode == "shadow":
                block_entries_reason = (
                    f"Resume policy flatten_then_start waiting to flatten {open_positions} open shadow position(s)"
                )
            else:
                block_entries_reason = f"Resume policy flatten_then_start blocked: {open_positions} open live position(s) require manual flattening"
        elif source_configs and source_entry_modes and not any(source_entry_modes.values()):
            block_entries_reason = "Configured strategies disable new entries (manage-existing-only mode)"
            block_entries_event_type = "trader_strategy_entry_mode"
            block_entries_event_severity = "info"
            block_entries_payload["source_entry_modes"] = {
                source_key: ("entries_enabled" if allows_entries else "manage_existing_only")
                for source_key, allows_entries in source_entry_modes.items()
            }

        if run_mode == "live":
            provider_health_window_seconds = int(
                max(
                    30,
                    min(
                        900,
                        safe_int(
                            live_provider_health_settings.get("window_seconds"),
                            int(DEFAULT_LIVE_PROVIDER_HEALTH["window_seconds"]),
                        ),
                    ),
                )
            )
            provider_health_min_errors = int(
                max(
                    1,
                    min(
                        20,
                        safe_int(
                            live_provider_health_settings.get("min_errors"),
                            int(DEFAULT_LIVE_PROVIDER_HEALTH["min_errors"]),
                        ),
                    ),
                )
            )
            provider_health_block_seconds = int(
                max(
                    15,
                    min(
                        3600,
                        safe_int(
                            live_provider_health_settings.get("block_seconds"),
                            int(DEFAULT_LIVE_PROVIDER_HEALTH["block_seconds"]),
                        ),
                    ),
                )
            )
            provider_health_now = utcnow()
            blocked_until = _live_provider_entry_blocked_until.get(trader_id)
            if blocked_until is not None and provider_health_now >= blocked_until:
                _live_provider_entry_blocked_until.pop(trader_id, None)
                blocked_until = None

            provider_health_snapshot: dict[str, Any]
            new_provider_block = False
            if blocked_until is None:
                provider_health_snapshot = await _live_provider_failure_snapshot(
                    session,
                    trader_id=trader_id,
                    window_seconds=provider_health_window_seconds,
                )
                if int(provider_health_snapshot.get("count", 0) or 0) >= provider_health_min_errors:
                    blocked_until = provider_health_now + timedelta(seconds=provider_health_block_seconds)
                    _live_provider_entry_blocked_until[trader_id] = blocked_until
                    new_provider_block = True
            else:
                provider_health_snapshot = {
                    "count": 0,
                    "window_seconds": provider_health_window_seconds,
                    "errors": [],
                }

            if blocked_until is not None and provider_health_now < blocked_until:
                if block_entries_reason is None:
                    recent_error_count = int(provider_health_snapshot.get("count", 0) or 0)
                    blocked_until_iso = blocked_until.isoformat().replace("+00:00", "Z")
                    block_entries_reason = (
                        "Live provider health guard: "
                        f"entries paused until {blocked_until_iso} after {recent_error_count} provider infra "
                        f"failure(s) in {provider_health_window_seconds}s (threshold={provider_health_min_errors})"
                    )
                    block_entries_event_type = "live_provider_health_block"
                    block_entries_payload = {
                        **block_entries_payload,
                        "provider_health": {
                            "window_seconds": provider_health_window_seconds,
                            "min_errors": provider_health_min_errors,
                            "block_seconds": provider_health_block_seconds,
                            "blocked_until": blocked_until_iso,
                            "recent_failure_count": recent_error_count,
                            "recent_failures": provider_health_snapshot.get("errors") or [],
                            "new_block": new_provider_block,
                        },
                    }

        if block_entries_reason is not None and effective_process_signals:
            effective_process_signals = False
            should_emit_block_event = True
            if block_entries_event_type == "live_provider_health_block":
                should_emit_block_event = _live_provider_health_event_due(trader_id, utcnow())
            if should_emit_block_event:
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type=block_entries_event_type,
                    severity=block_entries_event_severity,
                    source="worker",
                    message=block_entries_reason,
                    payload=block_entries_payload,
                )

        if not effective_process_signals:
            await _persist_trader_cycle_heartbeat(session, trader_id)
            return 0, 0, processed_signals

        is_high_frequency_trader = _is_high_frequency_crypto_trader(trader)
        max_signals_cap = (
            _HIGH_FREQUENCY_MAX_SIGNALS_PER_CYCLE if is_high_frequency_trader else _STANDARD_MAX_SIGNALS_PER_CYCLE
        )
        default_max_signals = (
            _HIGH_FREQUENCY_DEFAULT_MAX_SIGNALS_PER_CYCLE
            if is_high_frequency_trader
            else _STANDARD_DEFAULT_MAX_SIGNALS_PER_CYCLE
        )
        default_scan_batch_size = (
            _HIGH_FREQUENCY_DEFAULT_SCAN_BATCH_SIZE if is_high_frequency_trader else default_max_signals
        )
        max_signals_per_cycle = int(
            max(
                1,
                min(
                    max_signals_cap,
                    safe_int(default_strategy_params.get("max_signals_per_cycle"), default_max_signals),
                ),
            )
        )
        scan_batch_size = int(
            max(
                1,
                min(
                    max_signals_cap,
                    safe_int(
                        default_strategy_params.get("scan_batch_size"),
                        min(default_scan_batch_size, max_signals_per_cycle),
                    ),
                ),
            )
        )
        scan_batch_size = min(scan_batch_size, max_signals_per_cycle)
        enable_live_market_context = bool(live_market_context_settings.get("enabled", True))
        history_window_seconds = int(
            max(
                300,
                min(
                    21600,
                    safe_int(
                        live_market_context_settings.get("history_window_seconds"),
                        int(DEFAULT_LIVE_MARKET_CONTEXT["history_window_seconds"]),
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
                        live_market_context_settings.get("history_fidelity_seconds"),
                        int(DEFAULT_LIVE_MARKET_CONTEXT["history_fidelity_seconds"]),
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
                        live_market_context_settings.get("max_history_points"),
                        int(DEFAULT_LIVE_MARKET_CONTEXT["max_history_points"]),
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
                        live_market_context_settings.get("timeout_seconds"),
                        float(DEFAULT_LIVE_MARKET_CONTEXT["timeout_seconds"]),
                    )
                    or float(DEFAULT_LIVE_MARKET_CONTEXT["timeout_seconds"]),
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
        trading_schedule_ok = is_within_trading_schedule_utc(metadata, now_utc)
        global_limits = dict(control_settings.get("global_risk") or {})
        effective_risk_limits = dict(risk_limits)
        if "max_orders_per_cycle" not in effective_risk_limits:
            fallback_cycle_limit = safe_int(global_limits.get("max_orders_per_cycle"), 50)
            if fallback_cycle_limit > 0:
                effective_risk_limits["max_orders_per_cycle"] = fallback_cycle_limit
        if "max_daily_loss_usd" not in effective_risk_limits:
            fallback_daily_loss = safe_float(global_limits.get("max_daily_loss_usd"), 0.0) or 0.0
            if fallback_daily_loss > 0:
                effective_risk_limits["max_daily_loss_usd"] = fallback_daily_loss
        live_risk_clamp_changes: dict[str, dict[str, Any]] = {}
        if run_mode in {"live", "shadow"}:
            live_risk_clamp_settings = dict(global_runtime_settings.get("live_risk_clamps") or DEFAULT_LIVE_RISK_CLAMPS)
            live_risk_clamp_changes = _apply_live_risk_clamps(
                effective_risk_limits,
                live_risk_clamp_settings,
            )
            if live_risk_clamp_changes and _live_risk_clamp_event_due(trader_id, utcnow()):
                await create_trader_event(
                    session,
                    trader_id=trader_id,
                    event_type="live_risk_clamped",
                    severity="warn",
                    source="worker",
                    message="Applied live risk safety clamps for directional execution stability.",
                    payload={"changes": live_risk_clamp_changes},
                )
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
            since=loss_streak_reset_at,
        )
        last_loss_at = await get_last_resolved_loss_at(
            session,
            trader_id=trader_id,
            mode=run_mode,
            since=loss_streak_reset_at,
        )
        cooldown_seconds = max(0, safe_int(effective_risk_limits.get("cooldown_seconds"), 0))
        cooldown_active = False
        cooldown_remaining_seconds = 0
        if cooldown_seconds > 0 and last_loss_at is not None and trader_loss_streak > 0:
            if last_loss_at.tzinfo is None:
                loss_anchor = last_loss_at.replace(tzinfo=timezone.utc)
            else:
                loss_anchor = last_loss_at.astimezone(timezone.utc)
            cooldown_until = loss_anchor + timedelta(seconds=cooldown_seconds)
            if cooldown_until > now_utc:
                cooldown_active = True
                cooldown_remaining_seconds = int((cooldown_until - now_utc).total_seconds())

        # Circuit breaker: auto-pause trader and trigger safe exit of open positions
        halt_on_losses = bool(effective_risk_limits.get("halt_on_consecutive_losses", False))
        max_consecutive_losses_limit = max(1, safe_int(effective_risk_limits.get("max_consecutive_losses"), 4))
        if halt_on_losses and trader_loss_streak >= max_consecutive_losses_limit:
            await set_trader_paused(session, trader_id, True)
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="circuit_breaker_pause",
                source="worker",
                message=(
                    f"Circuit breaker: auto-paused after {trader_loss_streak} consecutive losses "
                    f"(limit={max_consecutive_losses_limit})"
                ),
                payload={
                    "trader_loss_streak": trader_loss_streak,
                    "max_consecutive_losses": max_consecutive_losses_limit,
                    "action": "auto_pause",
                },
            )
            logger.warning(
                "Circuit breaker auto-paused trader %s after %d consecutive losses (limit=%d)",
                trader_id,
                trader_loss_streak,
                max_consecutive_losses_limit,
            )

            if run_mode == "live":
                from services.trader_orchestrator.position_lifecycle import reconcile_live_positions

                try:
                    safe_exit_result = await reconcile_live_positions(
                        session,
                        trader_id=trader_id,
                        trader_params=default_strategy_params,
                        dry_run=False,
                        force_mark_to_market=True,
                        reason="circuit_breaker_safe_exit",
                    )
                    safe_exit_closed = int(safe_exit_result.get("closed", 0))
                    if safe_exit_closed > 0:
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="circuit_breaker_safe_exit",
                            source="worker",
                            message=f"Circuit breaker safe exit: closed {safe_exit_closed} position(s)",
                            payload=safe_exit_result,
                        )
                except Exception as exc:
                    logger.warning(
                        "Circuit breaker safe exit failed for trader %s: %s",
                        trader_id,
                        exc,
                        exc_info=exc,
                    )
            elif run_mode == "shadow":
                try:
                    safe_exit_result = await reconcile_shadow_positions(
                        session,
                        trader_id=trader_id,
                        trader_params=default_strategy_params,
                        dry_run=False,
                        force_mark_to_market=True,
                        reason="circuit_breaker_safe_exit",
                    )
                    safe_exit_closed = int(safe_exit_result.get("closed", 0))
                    if safe_exit_closed > 0:
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="circuit_breaker_safe_exit",
                            source="worker",
                            message=f"Circuit breaker safe exit: closed {safe_exit_closed} shadow position(s)",
                            payload=safe_exit_result,
                        )
                except Exception as exc:
                    logger.warning(
                        "Circuit breaker safe exit failed for trader %s: %s",
                        trader_id,
                        exc,
                        exc_info=exc,
                    )
            elif run_mode == "paper":
                try:
                    safe_exit_result = await reconcile_paper_positions(
                        session,
                        trader_id=trader_id,
                        trader_params=default_strategy_params,
                        dry_run=False,
                        force_mark_to_market=True,
                        reason="circuit_breaker_safe_exit",
                    )
                    safe_exit_closed = int(safe_exit_result.get("closed", 0))
                    if safe_exit_closed > 0:
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="circuit_breaker_safe_exit",
                            source="worker",
                            message=f"Circuit breaker safe exit: closed {safe_exit_closed} paper position(s)",
                            payload=safe_exit_result,
                        )
                except Exception as exc:
                    logger.warning(
                        "Circuit breaker safe exit failed for trader %s: %s",
                        trader_id,
                        exc,
                        exc_info=exc,
                    )

            await _persist_trader_cycle_heartbeat(session, trader_id)
            return 0, 0, processed_signals

        traders_scope_context: dict[str, Any] | None = None
        traders_source_config = source_configs.get("traders")
        if traders_source_config is not None:
            traders_params = dict(traders_source_config.get("strategy_params") or {})
            traders_scope_context = await _build_traders_scope_context(
                session,
                traders_params.get("traders_scope"),
            )

        # Kill switch is the very first gate: short-circuit before any signal
        # fetching, live-market context building, or risk evaluation.
        if bool(control.get("kill_switch")):
            logger.info(
                "Kill switch active for trader %s — skipping all signal processing",
                trader_id,
            )
            await _persist_trader_cycle_heartbeat(session, trader_id)
            return 0, 0, processed_signals

        while processed_signals < max_signals_per_cycle:
            batch_limit = min(scan_batch_size, max_signals_per_cycle - processed_signals)
            if prefetched_signals is not None:
                signals = prefetched_signals[:batch_limit]
                prefetched_signals = prefetched_signals[batch_limit:] if len(prefetched_signals) > batch_limit else None
            else:
                if stream_trigger_mode:
                    break
                signals = await list_unconsumed_trade_signals(
                    session,
                    trader_id=trader_id,
                    sources=sources,
                    statuses=["pending", "selected"],
                    strategy_types_by_source=strategy_types_by_source,
                    cursor_created_at=cursor_created_at,
                    cursor_signal_id=cursor_signal_id,
                    limit=batch_limit,
                )
            if not signals:
                break

            scoped_signals: list[Any] = []
            prefiltered = 0
            for signal in signals:
                signal_source = normalize_source_key(getattr(signal, "source", ""))
                source_config = source_configs.get(signal_source)
                signal_id = str(signal.id)
                signal_ts = _signal_cursor_timestamp(signal)
                if source_config is not None:
                    if not source_entry_modes.get(signal_source, True):
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            outcome="skipped",
                            reason="Signal excluded: strategy is manage-existing-only (new entries disabled)",
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=signal_ts,
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        cursor_created_at = signal_ts
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        prefiltered_signals += 1
                        prefiltered_by_reason["strategy_entry_disabled"] = (
                            prefiltered_by_reason.get("strategy_entry_disabled", 0) + 1
                        )
                        prefiltered += 1
                        continue
                    expected_strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
                    signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
                    allowed_strategy_types = set(strategy_types_by_source.get(signal_source) or [])
                    strategy_mismatch = bool(
                        expected_strategy_key
                        and allowed_strategy_types
                        and signal_strategy_type not in allowed_strategy_types
                    )
                    if strategy_mismatch:
                        allowed_label = ", ".join(sorted(allowed_strategy_types)) or expected_strategy_key
                        await record_signal_consumption(
                            session,
                            trader_id=trader_id,
                            signal_id=signal_id,
                            outcome="skipped",
                            reason=(
                                "Signal excluded by source strategy filter "
                                f"(signal={signal_strategy_type or 'unknown'}, allowed={allowed_label})"
                            ),
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=signal_ts,
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        cursor_created_at = signal_ts
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        prefiltered_signals += 1
                        prefiltered_by_reason["source_strategy_filter"] = (
                            prefiltered_by_reason.get("source_strategy_filter", 0) + 1
                        )
                        prefiltered += 1
                        continue
                if signal_source != "crypto" or source_config is None or _crypto_signal_in_scope(signal, source_config):
                    scoped_signals.append(signal)
                    continue
                await record_signal_consumption(
                    session,
                    trader_id=trader_id,
                    signal_id=signal_id,
                    outcome="skipped",
                    reason="Signal excluded by crypto scope filters",
                    commit=False,
                )
                await upsert_trader_signal_cursor(
                    session,
                    trader_id=trader_id,
                    last_signal_created_at=signal_ts,
                    last_signal_id=signal_id,
                    commit=False,
                )
                cursor_created_at = signal_ts
                cursor_signal_id = signal_id
                processed_signals += 1
                prefiltered_signals += 1
                prefiltered_by_reason["crypto_scope_filter"] = prefiltered_by_reason.get("crypto_scope_filter", 0) + 1
                signal_asset, signal_timeframe = _crypto_signal_dimensions(signal)
                dimension_key = f"{signal_asset or 'unknown'}:{signal_timeframe or 'unknown'}"
                crypto_scope_prefiltered_dimensions[dimension_key] = (
                    crypto_scope_prefiltered_dimensions.get(dimension_key, 0) + 1
                )
                prefiltered += 1
            if prefiltered > 0:
                await _commit_with_retry(session)
            if not scoped_signals:
                continue
            signals = scoped_signals

            live_contexts: dict[str, dict[str, Any]] = {}
            if enable_live_market_context:
                context_candidates: list[Any] = []
                for sig in signals:
                    source_key = normalize_source_key(getattr(sig, "source", ""))
                    source_config = source_configs.get(source_key)
                    if _supports_live_market_context(sig, source_config):
                        context_candidates.append(sig)
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
                resolved_strategy_version: int | None = None
                requested_strategy_version: int | None = None
                experiment_row = None
                assignment_group: str | None = None
                assignment_sample_pct: float | None = None
                assignment_source = "pinned_config"

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
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await _commit_with_retry(session)
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        continue

                    strategy_key = str(source_config.get("strategy_key") or "").strip().lower()
                    strategy_params = dict(source_config.get("strategy_params") or {})
                    requested_strategy_version = normalize_strategy_version(source_config.get("strategy_version"))
                    live_context = live_contexts.get(signal_id, {})
                    runtime_signal = RuntimeTradeSignalView(signal, live_context=live_context)
                    runtime_signal.source = signal_source
                    traders_scope_payload: dict[str, Any] | None = None
                    experiment_row = await get_active_strategy_experiment(
                        session,
                        source_key=signal_source,
                        strategy_key=strategy_key,
                    )
                    assignment_source = "pinned_config"
                    resolved_version_request = requested_strategy_version
                    if experiment_row is not None:
                        assignment_group, assigned_version, assignment_sample_pct = resolve_experiment_assignment(
                            experiment=experiment_row,
                            trader_id=trader_id,
                            signal_id=signal_id,
                        )
                        resolved_version_request = int(assigned_version)
                        assignment_source = "experiment"

                    # ── Strategy/version resolution (unified loader + immutable versions) ─────────────
                    version_resolution = None
                    try:
                        version_resolution = await resolve_strategy_version(
                            session,
                            strategy_key=strategy_key,
                            requested_version=resolved_version_request,
                        )
                    except ValueError as exc:
                        blocked_reason = "strategy_version_unavailable"
                        strategy_detail = str(exc)
                        checks_payload = [
                            {
                                "check_key": "strategy_version_available",
                                "check_label": "Strategy version available",
                                "passed": False,
                                "score": None,
                                "detail": strategy_detail,
                                "payload": {
                                    "requested_strategy_key": strategy_key,
                                    "requested_strategy_version": requested_strategy_version,
                                    "resolved_strategy_version": resolved_version_request,
                                },
                            }
                        ]
                        decision_row = await create_trader_decision(
                            session,
                            trader_id=trader_id,
                            signal=runtime_signal,
                            strategy_key=strategy_key,
                            strategy_version=resolved_version_request,
                            decision="blocked",
                            reason=blocked_reason,
                            score=0.0,
                            checks_summary={"count": len(checks_payload)},
                            risk_snapshot={},
                            payload={
                                "source_key": signal_source,
                                "source_config": source_config,
                                "strategy_runtime_error": strategy_detail,
                                "experiment": {
                                    "id": str(experiment_row.id) if experiment_row is not None else None,
                                    "assignment_group": assignment_group,
                                    "assignment_source": assignment_source,
                                    "assignment_sample_pct": assignment_sample_pct,
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
                        if experiment_row is not None:
                            await upsert_strategy_experiment_assignment(
                                session,
                                experiment_id=str(experiment_row.id),
                                trader_id=trader_id,
                                signal_id=signal_id,
                                source_key=signal_source,
                                strategy_key=strategy_key,
                                strategy_version=int(resolved_version_request or 1),
                                assignment_group=str(assignment_group or "control"),
                                decision_id=decision_row.id,
                                payload={
                                    "sample_pct": assignment_sample_pct,
                                    "assignment_source": assignment_source,
                                },
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
                                "requested_strategy_version": requested_strategy_version,
                                "resolved_strategy_version": resolved_version_request,
                                "error": strategy_detail,
                            },
                            commit=False,
                        )
                        await upsert_trader_signal_cursor(
                            session,
                            trader_id=trader_id,
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await _commit_with_retry(session)
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        continue

                    resolved_strategy_key = (
                        str(version_resolution.strategy.slug or strategy_key).strip().lower() or strategy_key
                    )
                    resolved_strategy_version = int(
                        version_resolution.version_row.version or version_resolution.latest_version or 1
                    )
                    strategy = None
                    strategy_detail: str | None = None

                    if int(resolved_strategy_version) == int(version_resolution.latest_version):
                        strategy_status = strategy_loader.get_availability(resolved_strategy_key)
                        if strategy_status.available:
                            loaded = strategy_loader.get_strategy(strategy_status.resolved_key or resolved_strategy_key)
                            strategy = _strategy_instance_from_loaded(loaded)

                        if strategy is None and strategy_status.available:
                            signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
                            if signal_strategy_type:
                                loaded = strategy_loader.get_strategy(signal_strategy_type)
                                candidate = _strategy_instance_from_loaded(loaded)
                                if candidate is not None and hasattr(candidate, "evaluate"):
                                    strategy = candidate

                        if strategy is None and strategy_status.available:
                            loaded = strategy_loader.get_strategy(signal_source)
                            candidate = _strategy_instance_from_loaded(loaded)
                            if candidate is not None and hasattr(candidate, "evaluate"):
                                strategy = candidate

                        if strategy is None:
                            strategy_detail = str(strategy_status.reason or "Strategy cache miss")
                    else:
                        strategy, strategy_detail = _load_versioned_strategy_instance(
                            strategy_key=resolved_strategy_key,
                            strategy_version=resolved_strategy_version,
                            source_code=str(version_resolution.version_row.source_code or ""),
                            config=dict(version_resolution.version_row.config or {}),
                        )

                    if strategy is None:
                        blocked_reason = f"strategy_unavailable:{resolved_strategy_key}:v{resolved_strategy_version}"
                        strategy_detail = str(strategy_detail or blocked_reason)
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
                                    "requested_strategy_version": requested_strategy_version,
                                    "resolved_strategy_version": resolved_strategy_version,
                                },
                            }
                        ]
                        decision_row = await create_trader_decision(
                            session,
                            trader_id=trader_id,
                            signal=runtime_signal,
                            strategy_key=resolved_strategy_key,
                            strategy_version=resolved_strategy_version,
                            decision="blocked",
                            reason=blocked_reason,
                            score=0.0,
                            checks_summary={"count": len(checks_payload)},
                            risk_snapshot={},
                            payload={
                                "source_key": signal_source,
                                "source_config": source_config,
                                "strategy_runtime_error": strategy_detail,
                                "experiment": {
                                    "id": str(experiment_row.id) if experiment_row is not None else None,
                                    "assignment_group": assignment_group,
                                    "assignment_source": assignment_source,
                                    "assignment_sample_pct": assignment_sample_pct,
                                },
                            },
                            commit=False,
                        )
                        decisions_written += 1
                        if experiment_row is not None:
                            await upsert_strategy_experiment_assignment(
                                session,
                                experiment_id=str(experiment_row.id),
                                trader_id=trader_id,
                                signal_id=signal_id,
                                source_key=signal_source,
                                strategy_key=resolved_strategy_key,
                                strategy_version=resolved_strategy_version,
                                assignment_group=str(assignment_group or "control"),
                                decision_id=decision_row.id,
                                payload={
                                    "sample_pct": assignment_sample_pct,
                                    "assignment_source": assignment_source,
                                },
                                commit=False,
                            )
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
                            last_signal_created_at=_signal_cursor_timestamp(signal),
                            last_signal_id=signal_id,
                            commit=False,
                        )
                        await _commit_with_retry(session)
                        cursor_created_at = _signal_cursor_timestamp(signal)
                        cursor_signal_id = signal_id
                        processed_signals += 1
                        continue

                    if signal_source == "traders":
                        if traders_scope_context is None:
                            traders_params = dict(source_config.get("strategy_params") or {})
                            traders_scope_context = await _build_traders_scope_context(
                                session,
                                traders_params.get("traders_scope"),
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
                                strategy_version=resolved_strategy_version,
                                decision="skipped",
                                reason="Signal excluded by traders_scope",
                                score=0.0,
                                checks_summary={"count": len(checks_payload)},
                                risk_snapshot={},
                                payload={
                                    "source_key": signal_source,
                                    "source_config": source_config,
                                    "traders_scope": scope_payload,
                                    "experiment": {
                                        "id": str(experiment_row.id) if experiment_row is not None else None,
                                        "assignment_group": assignment_group,
                                        "assignment_source": assignment_source,
                                        "assignment_sample_pct": assignment_sample_pct,
                                    },
                                },
                                commit=False,
                            )
                            decisions_written += 1
                            if experiment_row is not None:
                                await upsert_strategy_experiment_assignment(
                                    session,
                                    experiment_id=str(experiment_row.id),
                                    trader_id=trader_id,
                                    signal_id=signal_id,
                                    source_key=signal_source,
                                    strategy_key=resolved_strategy_key,
                                    strategy_version=int(resolved_strategy_version or 1),
                                    assignment_group=str(assignment_group or "control"),
                                    decision_id=decision_row.id,
                                    payload={
                                        "sample_pct": assignment_sample_pct,
                                        "assignment_source": assignment_source,
                                    },
                                    commit=False,
                                )
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
                                last_signal_created_at=_signal_cursor_timestamp(signal),
                                last_signal_id=signal_id,
                                commit=False,
                            )
                            await _commit_with_retry(session)
                            cursor_created_at = _signal_cursor_timestamp(signal)
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
                            "traders_scope_context": (
                                traders_scope_context if signal_source == "traders" else None
                            ),
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
                        "trader_open_positions": effective_open_positions,
                        "trader_open_orders": open_order_count,
                        "pending_live_exit_count": pending_live_exit_count,
                        "pending_live_exit_statuses": pending_live_exit_summary.get("statuses") or {},
                        "live_risk_clamps": live_risk_clamp_changes,
                        "trading_schedule_ok": trading_schedule_ok,
                    }

                    risk_evaluator = None
                    portfolio_allocator = None
                    portfolio_runtime_payload: dict[str, Any] = {}
                    if decision_obj.decision == "selected" and trading_schedule_ok:
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
                            "trader_open_positions": effective_open_positions,
                            "trader_open_orders": open_order_count,
                            "pending_live_exit_count": pending_live_exit_count,
                            "pending_live_exit_statuses": pending_live_exit_summary.get("statuses") or {},
                        }

                        def _evaluate_runtime_risk(size_for_eval: float):
                            risk_result = evaluate_risk(
                                size_usd=size_for_eval,
                                gross_exposure_usd=gross_exposure,
                                trader_open_positions=effective_open_positions,
                                trader_open_orders=open_order_count,
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

                        portfolio_config = _normalize_portfolio_config(effective_risk_limits)
                        if bool(portfolio_config.get("enabled", False)):
                            source_exposure = await get_trader_source_exposure(
                                session,
                                trader_id=trader_id,
                                source=signal_source,
                                mode=run_mode,
                            )
                            gross_exposure_cap = safe_float(
                                effective_risk_limits.get("max_gross_exposure_usd"),
                                safe_float(global_limits.get("max_gross_exposure_usd"), 5000.0),
                            )
                            portfolio_runtime_payload = {
                                "source_key": signal_source,
                                "gross_exposure_cap_usd": gross_exposure_cap,
                                "current_gross_exposure_usd": gross_exposure,
                                "current_source_exposure_usd": source_exposure,
                                "target_utilization_pct": portfolio_config["target_utilization_pct"],
                                "max_source_exposure_pct": portfolio_config["max_source_exposure_pct"],
                                "min_order_notional_usd": portfolio_config["min_order_notional_usd"],
                            }

                            def _evaluate_portfolio(size_for_eval: float):
                                return _allocate_portfolio_notional(
                                    requested_size_usd=size_for_eval,
                                    gross_exposure_cap_usd=gross_exposure_cap,
                                    current_gross_exposure_usd=gross_exposure,
                                    current_source_exposure_usd=source_exposure,
                                    source_key=signal_source,
                                    portfolio_config=portfolio_config,
                                )

                            portfolio_allocator = _evaluate_portfolio

                    gate_result = apply_platform_decision_gates(
                        decision_obj=decision_obj,
                        runtime_signal=runtime_signal,
                        strategy=strategy,
                        checks_payload=checks_payload,
                        trading_schedule_ok=trading_schedule_ok,
                        trading_schedule_config=metadata.get("trading_schedule_utc"),
                        global_limits=global_limits,
                        effective_risk_limits=effective_risk_limits,
                        allow_averaging=allow_averaging,
                        open_market_ids=open_market_ids,
                        pending_live_exit_count=pending_live_exit_count,
                        pending_live_exit_summary=pending_live_exit_summary,
                        pending_live_exit_max_allowed=pending_live_exit_max_allowed,
                        pending_live_exit_identity_guard_enabled=pending_live_exit_identity_guard_enabled,
                        portfolio_allocator=portfolio_allocator,
                        risk_evaluator=risk_evaluator,
                        invoke_hooks=True,
                        strategy_params=strategy_params,
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
                        strategy_version=resolved_strategy_version,
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
                                "trader_open_orders": risk_runtime_payload["trader_open_orders"],
                                "pending_live_exit_count": risk_runtime_payload["pending_live_exit_count"],
                                "pending_live_exit_statuses": risk_runtime_payload["pending_live_exit_statuses"],
                                "live_risk_clamps": risk_runtime_payload["live_risk_clamps"],
                                "trading_schedule_ok": risk_runtime_payload["trading_schedule_ok"],
                            },
                            "portfolio_runtime": portfolio_runtime_payload,
                            "experiment": {
                                "id": str(experiment_row.id) if experiment_row is not None else None,
                                "assignment_group": assignment_group,
                                "assignment_source": assignment_source,
                                "assignment_sample_pct": assignment_sample_pct,
                                "resolved_strategy_version": resolved_strategy_version,
                            },
                        },
                        commit=False,
                    )
                    decisions_written += 1
                    if experiment_row is not None:
                        await upsert_strategy_experiment_assignment(
                            session,
                            experiment_id=str(experiment_row.id),
                            trader_id=trader_id,
                            signal_id=signal_id,
                            source_key=signal_source,
                            strategy_key=resolved_strategy_key,
                            strategy_version=int(resolved_strategy_version or 1),
                            assignment_group=str(assignment_group or "control"),
                            decision_id=decision_row.id,
                            payload={
                                "sample_pct": assignment_sample_pct,
                                "assignment_source": assignment_source,
                                "decision": final_decision,
                            },
                            commit=False,
                        )

                    await create_trader_decision_checks(
                        session,
                        decision_id=decision_row.id,
                        checks=checks_payload,
                        commit=False,
                    )
                    freshness_gate = next(
                        (
                            gate
                            for gate in gate_result["platform_gates"]
                            if isinstance(gate, dict) and str(gate.get("gate") or "") == "market_data_freshness"
                        ),
                        None,
                    )
                    live_revalidation_gate = next(
                        (
                            gate
                            for gate in gate_result["platform_gates"]
                            if isinstance(gate, dict) and str(gate.get("gate") or "") == "live_market_revalidation"
                        ),
                        None,
                    )
                    freshness_payload = (
                        dict(freshness_gate.get("payload") or {})
                        if isinstance(freshness_gate, dict) and isinstance(freshness_gate.get("payload"), dict)
                        else {}
                    )
                    freshness_status = str((freshness_gate or {}).get("status") or "").strip().lower()
                    if freshness_status not in {"passed", "blocked"}:
                        live_revalidation_status = str((live_revalidation_gate or {}).get("status") or "").strip().lower()
                        if live_revalidation_status == "blocked":
                            freshness_status = "blocked"
                            if not freshness_payload:
                                freshness_payload = (
                                    dict(live_revalidation_gate.get("payload") or {})
                                    if isinstance(live_revalidation_gate, dict)
                                    and isinstance(live_revalidation_gate.get("payload"), dict)
                                    else {}
                                )
                    freshness_source = str(freshness_payload.get("source") or signal_source or "").strip().lower()
                    if freshness_status in {"passed", "blocked"} and freshness_source in {"scanner", "crypto"}:
                        freshness_age_ms = safe_float(freshness_payload.get("age_ms"), None)
                        freshness_max_age_ms = safe_float(freshness_payload.get("max_age_ms"), None)
                        freshness_severity = "warn" if freshness_status == "blocked" else "info"
                        freshness_message = (
                            f"Market data freshness {freshness_status}: source={freshness_source or 'unknown'} "
                            f"age_ms={freshness_age_ms if freshness_age_ms is not None else 'unknown'} "
                            f"max={freshness_max_age_ms if freshness_max_age_ms is not None else 'unknown'}"
                        )
                        await create_trader_event(
                            session,
                            trader_id=trader_id,
                            event_type="market_data_freshness_source",
                            severity=freshness_severity,
                            source=freshness_source or signal_source,
                            message=freshness_message,
                            payload={
                                "decision_id": decision_row.id,
                                "signal_id": signal_id,
                                "status": freshness_status,
                                "source": freshness_source or None,
                                "timeframe": freshness_payload.get("timeframe"),
                                "age_ms": freshness_age_ms,
                                "max_age_ms": freshness_max_age_ms,
                                "market_data_source": freshness_payload.get("market_data_source"),
                                "required_sources": freshness_payload.get("required_sources"),
                                "freshness_enforced": freshness_payload.get("freshness_enforced"),
                                "final_decision": final_decision,
                                "final_reason": final_reason,
                            },
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
                            strategy_version=resolved_strategy_version,
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
                            if normalized_order_status in {
                                "executed",
                                "completed",
                                "working",
                                "hedging",
                                "partial",
                                "open",
                                "submitted",
                            }:
                                open_positions = await get_open_position_count_for_trader(
                                    session,
                                    trader_id,
                                    mode=run_mode,
                                    position_cap_scope=position_cap_scope,
                                )
                                open_order_count = await get_open_order_count_for_trader(
                                    session,
                                    trader_id,
                                    mode=run_mode,
                                )
                                effective_open_positions = max(open_positions, open_order_count)
                                opened_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
                                if opened_market_id:
                                    open_market_ids.add(opened_market_id)
                                intra_cycle_committed_usd += size_usd
                        else:
                            normalized_order_status = str(submit_result.status or "").strip().lower()
                            order_status = normalized_order_status
                            orders_written += int(submit_result.orders_written or 0)

                            if normalized_order_status in {
                                "executed",
                                "completed",
                                "working",
                                "hedging",
                                "partial",
                                "open",
                                "submitted",
                            }:
                                open_positions = await get_open_position_count_for_trader(
                                    session,
                                    trader_id,
                                    mode=run_mode,
                                    position_cap_scope=position_cap_scope,
                                )
                                open_order_count = await get_open_order_count_for_trader(
                                    session,
                                    trader_id,
                                    mode=run_mode,
                                )
                                effective_open_positions = max(open_positions, open_order_count)
                                opened_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
                                if opened_market_id:
                                    open_market_ids.add(opened_market_id)
                                intra_cycle_committed_usd += size_usd
                    else:
                        signal_status = "failed" if final_decision == "failed" else "skipped"
                        await set_trade_signal_status(
                            session,
                            signal_id=signal_id,
                            status=signal_status,
                            commit=False,
                        )

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
                            "market_id": getattr(runtime_signal, "market_id", None),
                            "market_question": getattr(runtime_signal, "market_question", None),
                            "direction": getattr(runtime_signal, "direction", None),
                        },
                        commit=False,
                    )
                    await upsert_trader_signal_cursor(
                        session,
                        trader_id=trader_id,
                        last_signal_created_at=_signal_cursor_timestamp(signal),
                        last_signal_id=signal_id,
                        commit=False,
                    )
                    await _commit_with_retry(session)
                    cursor_created_at = _signal_cursor_timestamp(signal)
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
                    strategy_version_for_error = resolved_strategy_version
                    failure_reason = f"Signal processing failed ({error_type})"
                    decision_row = await create_trader_decision(
                        session,
                        trader_id=trader_id,
                        signal=signal,
                        strategy_key=strategy_for_error,
                        strategy_version=strategy_version_for_error,
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
                    if experiment_row is not None:
                        await upsert_strategy_experiment_assignment(
                            session,
                            experiment_id=str(experiment_row.id),
                            trader_id=trader_id,
                            signal_id=signal_id,
                            source_key=signal_source,
                            strategy_key=strategy_for_error,
                            strategy_version=int(strategy_version_for_error or experiment_row.control_version or 1),
                            assignment_group=str(assignment_group or "control"),
                            decision_id=decision_row.id,
                            payload={
                                "sample_pct": assignment_sample_pct,
                                "assignment_source": assignment_source,
                                "error_type": error_type,
                            },
                            commit=False,
                        )
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
                        last_signal_created_at=_signal_cursor_timestamp(signal),
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
                    cursor_created_at = _signal_cursor_timestamp(signal)
                    cursor_signal_id = signal_id
                    processed_signals += 1

        if process_signals:
            if decisions_written == 0 and orders_written == 0:
                if prefiltered_signals > 0:
                    payload = {
                        "processed_signals": processed_signals,
                        "decisions_written": decisions_written,
                        "orders_written": orders_written,
                        "prefiltered_signals": prefiltered_signals,
                        "prefiltered_by_reason": prefiltered_by_reason,
                    }
                    if crypto_scope_prefiltered_dimensions:
                        payload["crypto_scope_prefiltered_dimensions"] = crypto_scope_prefiltered_dimensions
                    await _emit_cycle_heartbeat_if_due(
                        session,
                        trader_id=trader_id,
                        message="Idle cycle: pending signals filtered before strategy evaluation.",
                        payload=payload,
                    )
                elif processed_signals == 0:
                    await _emit_cycle_heartbeat_if_due(
                        session,
                        trader_id=trader_id,
                        message="Idle cycle: no pending signals.",
                        payload={
                            "processed_signals": processed_signals,
                            "decisions_written": decisions_written,
                            "orders_written": orders_written,
                        },
                    )
            await _persist_trader_cycle_heartbeat(session, trader_id)

    return decisions_written, orders_written, processed_signals


async def _reconcile_orphan_open_orders(session: Any) -> dict[str, int]:
    """Close/cancel open orders for trader_ids that no longer exist."""
    mode_key_expr = func.lower(func.coalesce(TraderOrder.mode, ""))
    status_key_expr = func.lower(func.coalesce(TraderOrder.status, ""))
    rows = (
        await session.execute(
            select(
                TraderOrder.trader_id,
                mode_key_expr.label("mode_key"),
                func.count(TraderOrder.id).label("count"),
            )
            .select_from(TraderOrder)
            .outerjoin(Trader, Trader.id == TraderOrder.trader_id)
            .where(Trader.id.is_(None))
            .where(status_key_expr.in_(tuple(_PAPER_ACTIVE_ORDER_STATUSES)))
            .group_by(TraderOrder.trader_id, mode_key_expr)
            .limit(_ORPHAN_CLEANUP_MAX_TRADERS_PER_CYCLE)
        )
    ).all()

    if not rows:
        return {
            "traders_seen": 0,
            "rows_seen": 0,
            "paper_closed": 0,
            "shadow_closed": 0,
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
    shadow_closed = 0
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
        if mode_key == "shadow":
            result = await reconcile_shadow_positions(
                session,
                trader_id=trader_id,
                trader_params={},
                dry_run=False,
                force_mark_to_market=False,
                reason="orphan_trader_lifecycle",
            )
            shadow_closed += int(result.get("closed", 0))
            await sync_trader_position_inventory(session, trader_id=trader_id, mode="shadow")
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
            f"(paper_closed={paper_closed}, shadow_closed={shadow_closed}, non_paper_cancelled={non_paper_cancelled})"
        ),
        payload={
            "traders_seen": len({tid for tid, _ in per_trader_mode.keys()}),
            "rows_seen": len(per_trader_mode),
            "paper_closed": paper_closed,
            "shadow_closed": shadow_closed,
            "non_paper_cancelled": non_paper_cancelled,
        },
        commit=True,
    )

    return {
        "traders_seen": len({tid for tid, _ in per_trader_mode.keys()}),
        "rows_seen": len(per_trader_mode),
        "paper_closed": paper_closed,
        "shadow_closed": shadow_closed,
        "non_paper_cancelled": non_paper_cancelled,
    }


async def _run_trader_once_with_timeout(
    trader: dict[str, Any],
    control: dict[str, Any],
    *,
    process_signals: bool,
    trigger_signal_ids_by_source: dict[str, list[str]] | None,
    timeout_seconds: float,
) -> tuple[int, int, int]:
    timeout = max(1.0, float(timeout_seconds))
    trader_id = str(trader.get("id") or "")
    try:
        return await asyncio.wait_for(
            _run_trader_once(
                trader,
                control,
                process_signals=process_signals,
                trigger_signal_ids_by_source=trigger_signal_ids_by_source,
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
        return 0, 0, 0
    except OperationalError as exc:
        if not _is_retryable_db_error(exc):
            raise
        logger.warning(
            "Trader cycle skipped due to transient DB error for trader=%s process_signals=%s",
            trader_id,
            process_signals,
        )
        return 0, 0, 0


async def run_worker_loop() -> None:
    logger.info("Starting trader orchestrator worker loop")

    try:
        async with AsyncSessionLocal() as session:
            await ensure_all_strategies_seeded(session)
            await refresh_strategy_runtime_if_needed(session, source_keys=None, force=True)
    except Exception as exc:
        logger.warning("Orchestrator strategy startup sync failed: %s", exc)

    runtime_trigger_event: dict[str, Any] | None = None
    stream_consumer_name = f"{os.getpid()}-{utcnow().strftime('%Y%m%d%H%M%S%f')}"
    stream_claim_cursor = "0-0"
    stream_last_claim_run_at: datetime | None = None
    if not await ensure_trade_signal_group():
        logger.warning("Trade signal stream group ensure failed; worker will continue with scheduled polling fallback")

    try:
        while True:
            cycle_interval = ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS
            sleep_seconds = 2
            skip_cycle = False
            manual_force_cycle = False
            cycle_trigger = runtime_trigger_event
            runtime_trigger_event = None
            cycle_stream_entry_ids: list[str] = []
            if isinstance(cycle_trigger, dict):
                cycle_stream_entry_ids = [
                    str(entry_id).strip()
                    for entry_id in (cycle_trigger.get("stream_entry_ids") or [])
                    if str(entry_id).strip()
                ]
            manage_only_cycle = False
            manage_only_reasons: list[str] = []
            mode = "paper"
            control: dict[str, Any] = {}
            traders: list[dict[str, Any]] = []
            try:
                if not await _ensure_orchestrator_cycle_lock_owner():
                    logger.debug(
                        "Orchestrator cycle lock held by another worker instance; waiting to retry",
                    )
                    await _worker_sleep(2)
                    continue

                async with AsyncSessionLocal() as session:
                    try:
                        await expire_stale_signals(session)
                    except OperationalError as exc:
                        if _is_retryable_db_error(exc):
                            if hasattr(session, "rollback"):
                                await session.rollback()
                            logger.warning("Skipped stale-signal expiry due to transient DB error")
                        else:
                            raise
                    except Exception as exc:
                        if hasattr(session, "rollback"):
                            await session.rollback()
                        logger.warning("Failed stale-signal expiry pass: %s", exc)
                    try:
                        await refresh_strategy_runtime_if_needed(session, source_keys=None)
                    except Exception as exc:
                        if hasattr(session, "rollback"):
                            await session.rollback()
                        logger.warning("Failed strategy runtime refresh pass: %s", exc)

                    try:
                        orphan_cleanup = await _reconcile_orphan_open_orders(session)
                        if orphan_cleanup.get("rows_seen", 0) > 0:
                            logger.warning(
                                "Reconciled orphan trader orders",
                                extra={"orphan_cleanup": orphan_cleanup},
                            )
                    except Exception as exc:
                        if hasattr(session, "rollback"):
                            await session.rollback()
                        logger.warning("Failed orphan-order reconciliation pass: %s", exc)

                    try:
                        stale_watchdog = await _run_terminal_stale_order_watchdog(session)
                        if stale_watchdog.get("alerted", 0) > 0:
                            logger.warning(
                                "Detected stale active orders on terminal markets",
                                extra={"stale_watchdog": stale_watchdog},
                            )
                    except Exception as exc:
                        if hasattr(session, "rollback"):
                            await session.rollback()
                        logger.warning("Failed stale-terminal-order watchdog pass: %s", exc)

                    control = await read_orchestrator_control(session)
                    mode = str(control.get("mode") or "paper").strip().lower()
                    cycle_interval = max(
                        1,
                        int(control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS),
                    )
                    sleep_seconds = cycle_interval
                    default_trader_cycle_timeout = float(max(8, cycle_interval * 2))
                    control_settings = dict(control.get("settings") or {})
                    global_runtime_settings = dict(control_settings.get("global_runtime") or {})
                    configured_trader_cycle_timeout = safe_float(
                        global_runtime_settings.get("trader_cycle_timeout_seconds"),
                        0.0,
                    )
                    effective_trader_cycle_timeout = (
                        configured_trader_cycle_timeout
                        if configured_trader_cycle_timeout > 0
                        else default_trader_cycle_timeout
                    )
                    trader_cycle_timeout_seconds = float(
                        max(
                            3.0,
                            min(
                                120.0,
                                effective_trader_cycle_timeout,
                            ),
                        )
                    )
                    manual_force_cycle = _parse_iso(control.get("requested_run_at")) is not None

                    if not control.get("is_enabled"):
                        traders = await list_traders(
                            session,
                            mode=mode if mode in {"paper", "shadow", "live"} else None,
                        )
                        has_active_traders = any(
                            bool(trader.get("is_enabled", True)) and not bool(trader.get("is_paused", False))
                            for trader in traders
                        )
                        if has_active_traders:
                            manage_only_cycle = True
                            manage_only_reasons.append("global_disabled")
                        else:
                            await _write_orchestrator_snapshot_best_effort(
                                session,
                                running=False,
                                enabled=False,
                                current_activity="Disabled",
                                interval_seconds=cycle_interval,
                                stats=await compute_orchestrator_metrics(session),
                            )
                            skip_cycle = True

                    if not skip_cycle and bool(control.get("is_paused")):
                        manage_only_cycle = True
                        manage_only_reasons.append("global_pause")

                    if not skip_cycle and bool(control.get("kill_switch")):
                        manage_only_cycle = True
                        manage_only_reasons.append("kill_switch")

                    if not skip_cycle:
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
                            if not _is_live_credentials_configured(app_settings):
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
                            elif not await live_execution_service.ensure_initialized():
                                await _write_orchestrator_snapshot_best_effort(
                                    session,
                                    running=False,
                                    enabled=True,
                                    current_activity="Blocked: live trading service failed to initialize",
                                    interval_seconds=cycle_interval,
                                    last_error=None,
                                    stats=await compute_orchestrator_metrics(session),
                                )
                                skip_cycle = True

                    if not skip_cycle and not traders:
                        traders = await list_traders(
                            session,
                            mode=mode if mode in {"paper", "shadow", "live"} else None,
                        )

                if skip_cycle:
                    if cycle_stream_entry_ids:
                        await ack_trade_signal_batches(cycle_stream_entry_ids)
                    (
                        runtime_trigger_event,
                        stream_claim_cursor,
                        stream_last_claim_run_at,
                    ) = await _wait_for_runtime_trigger(
                        None,
                        sleep_seconds,
                        consumer=stream_consumer_name,
                        claim_cursor=stream_claim_cursor,
                        last_claim_run_at=stream_last_claim_run_at,
                    )
                    continue

                total_decisions = 0
                total_orders = 0
                total_processed_signals = 0
                now = datetime.now(timezone.utc)
                high_frequency_trader_ids: set[str] = set()
                high_frequency_crypto_active = False
                for trader in traders:
                    trader_id = str(trader.get("id") or "")
                    if not trader_id:
                        continue
                    if not _is_high_frequency_crypto_trader(trader):
                        continue
                    high_frequency_trader_ids.add(trader_id)
                    if bool(trader.get("is_enabled", True)) and not bool(trader.get("is_paused", False)):
                        high_frequency_crypto_active = True

                if high_frequency_crypto_active and not manage_only_cycle:
                    sleep_seconds = min(
                        float(sleep_seconds),
                        _HIGH_FREQUENCY_CRYPTO_MAINTENANCE_INTERVAL_SECONDS,
                    )
                for trader in traders:
                    trader_id = str(trader.get("id") or "")
                    if not trader.get("is_enabled", True):
                        continue
                    trader_mode = str(trader.get("mode") or "paper").strip().lower()
                    if trader_mode not in {"paper", "shadow", "live"}:
                        trader_mode = "paper"
                    if trader_mode != mode:
                        continue

                    is_paused = bool(trader.get("is_paused", False))
                    runtime_trigger_for_trader = _runtime_trigger_matches_trader(trader, cycle_trigger)
                    due = manual_force_cycle or runtime_trigger_for_trader or _is_due(trader, now)
                    if trader_id in high_frequency_trader_ids and not manage_only_cycle and not is_paused:
                        due = manual_force_cycle or runtime_trigger_for_trader or _is_high_frequency_maintenance_due(
                            trader,
                            now,
                        )
                    process_signals_for_trader = True
                    if manage_only_cycle or is_paused or not due:
                        process_signals_for_trader = False
                    trigger_signal_ids_by_source = (
                        _trigger_signal_ids_for_trader(trader, cycle_trigger) if runtime_trigger_for_trader else None
                    )
                    if (
                        process_signals_for_trader
                        and runtime_trigger_for_trader
                        and isinstance(cycle_trigger, dict)
                        and str(cycle_trigger.get("event_type") or "").strip().lower() == "trade_signal_stream"
                        and trigger_signal_ids_by_source is None
                    ):
                        process_signals_for_trader = False

                    if mode not in ("paper", "shadow", "live"):
                        continue
                    decisions, orders, processed_signals = await _run_trader_once_with_timeout(
                        trader,
                        control,
                        process_signals=process_signals_for_trader,
                        trigger_signal_ids_by_source=trigger_signal_ids_by_source,
                        timeout_seconds=trader_cycle_timeout_seconds,
                    )
                    total_decisions += decisions
                    total_orders += orders
                    total_processed_signals += processed_signals

                async with AsyncSessionLocal() as session:
                    if manual_force_cycle and not manage_only_cycle:
                        await update_orchestrator_control(session, requested_run_at=None)
                    metrics = await compute_orchestrator_metrics(session)
                    metrics["decisions_last_cycle"] = total_decisions
                    metrics["orders_last_cycle"] = total_orders
                    metrics["signals_processed_last_cycle"] = total_processed_signals
                    trigger_label = (
                        str(cycle_trigger.get("event_type") or "").strip().lower()
                        if isinstance(cycle_trigger, dict)
                        else ""
                    ) or ("requested_run" if manual_force_cycle else "scheduled")
                    metrics["cycle_trigger"] = trigger_label
                    open_orders = int(metrics.get("open_orders", 0) or 0)
                    activity = (
                        f"Cycle[{trigger_label}] signals={total_processed_signals} "
                        f"decisions={total_decisions} orders={total_orders}"
                    )
                    if manage_only_cycle:
                        reasons = ",".join(manage_only_reasons) if manage_only_reasons else "gated"
                        activity = (
                            f"Manage-only[{trigger_label}] ({reasons}) signals={total_processed_signals} "
                            f"decisions={total_decisions} orders={total_orders}"
                        )
                    elif (
                        total_processed_signals == 0 and total_decisions == 0 and total_orders == 0 and open_orders > 0
                    ):
                        activity = (
                            f"Cycle[{trigger_label}] monitoring open orders={open_orders} (no new pending signals)"
                        )
                    elif high_frequency_crypto_active and total_processed_signals == 0:
                        activity = (
                            f"Cycle[{trigger_label}] high-frequency crypto monitor active (no new pending signals)"
                        )
                    await _write_orchestrator_snapshot_best_effort(
                        session,
                        running=True,
                        enabled=True,
                        current_activity=activity,
                        interval_seconds=cycle_interval,
                        last_run_at=utcnow(),
                        stats=metrics,
                    )

                if cycle_stream_entry_ids:
                    acknowledged = await ack_trade_signal_batches(cycle_stream_entry_ids)
                    if acknowledged < len(cycle_stream_entry_ids):
                        logger.warning(
                            "Trade signal stream ack partial",
                            extra={
                                "requested": len(cycle_stream_entry_ids),
                                "acknowledged": int(acknowledged),
                            },
                        )
                (
                    runtime_trigger_event,
                    stream_claim_cursor,
                    stream_last_claim_run_at,
                ) = await _wait_for_runtime_trigger(
                    None,
                    sleep_seconds,
                    consumer=stream_consumer_name,
                    claim_cursor=stream_claim_cursor,
                    last_claim_run_at=stream_last_claim_run_at,
                )
            except Exception as exc:
                logger.exception("Trader orchestrator worker cycle failed: %s", exc)
                try:
                    async with AsyncSessionLocal() as session:
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
                (
                    runtime_trigger_event,
                    stream_claim_cursor,
                    stream_last_claim_run_at,
                ) = await _wait_for_runtime_trigger(
                    None,
                    2,
                    consumer=stream_consumer_name,
                    claim_cursor=stream_claim_cursor,
                    last_claim_run_at=stream_last_claim_run_at,
                )
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
