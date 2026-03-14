from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable

from config import settings
from services.data_events import BlockReason
from services.strategy_sdk import StrategySDK
from utils.converters import coerce_bool as _coerce_bool, safe_float


def _parse_hhmm_utc(value: Any) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except Exception:
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute


def _parse_date_utc(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if "T" in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc)
            return parsed.date()
        return date.fromisoformat(text)
    except Exception:
        return None


def _parse_datetime_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_nonnegative_seconds(value: Any) -> float | None:
    parsed = safe_float(value, None)
    if parsed is None:
        return None
    if parsed < 0:
        return None
    return float(parsed)


def _seconds_until_utc(end_time_value: Any) -> float | None:
    end_time = _parse_datetime_utc(end_time_value)
    if end_time is None:
        return None
    now = datetime.now(timezone.utc)
    return max(0.0, (end_time - now).total_seconds())


def _runtime_signal_seconds_left(runtime_payload: Any) -> float | None:
    payload = runtime_payload if isinstance(runtime_payload, dict) else {}
    for key in ("seconds_left",):
        parsed = _parse_nonnegative_seconds(payload.get(key))
        if parsed is not None:
            return parsed

    strategy_context = payload.get("strategy_context")
    if isinstance(strategy_context, dict):
        parsed = _parse_nonnegative_seconds(strategy_context.get("seconds_left"))
        if parsed is not None:
            return parsed

    live_market_payload = payload.get("live_market")
    if isinstance(live_market_payload, dict):
        parsed = _parse_nonnegative_seconds(live_market_payload.get("seconds_left"))
        if parsed is not None:
            return parsed

    for key in ("end_time",):
        parsed = _seconds_until_utc(payload.get(key))
        if parsed is not None:
            return parsed

    if isinstance(strategy_context, dict):
        parsed = _seconds_until_utc(strategy_context.get("end_time"))
        if parsed is not None:
            return parsed

    if isinstance(live_market_payload, dict):
        parsed = _seconds_until_utc(live_market_payload.get("end_time"))
        if parsed is not None:
            return parsed

    return None


_TIMEFRAME_PARAM_SUFFIXES: dict[str, tuple[str, ...]] = {
    "1m": ("1m", "1min"),
    "3m": ("3m", "3min"),
    "5m": ("5m", "5min"),
    "15m": ("15m", "15min"),
    "1h": ("1h", "1hr", "60m"),
    "4h": ("4h", "4hr", "240m"),
}


def _normalize_timeframe(value: Any) -> str:
    raw = str(value or "").strip().lower()
    compact = raw.replace("_", "").replace("-", "").replace(" ", "")
    if compact in {"1m", "1min", "1minute", "1minutes"}:
        return "1m"
    if compact in {"3m", "3min", "3minute", "3minutes"}:
        return "3m"
    if compact in {"5m", "5min", "5minute", "5minutes"}:
        return "5m"
    if compact in {"15m", "15min", "15minute", "15minutes"}:
        return "15m"
    if compact in {"1h", "1hr", "1hour", "60m", "60min"}:
        return "1h"
    if compact in {"4h", "4hr", "4hour", "240m", "240min"}:
        return "4h"
    return raw


def _parse_timeframe_minutes(value: Any) -> float | None:
    normalized = _normalize_timeframe(value)
    if not normalized:
        return None
    if normalized.endswith("m"):
        parsed = safe_float(normalized[:-1], None)
        if parsed is None or parsed <= 0:
            return None
        return float(parsed)
    if normalized.endswith("h"):
        parsed = safe_float(normalized[:-1], None)
        if parsed is None or parsed <= 0:
            return None
        return float(parsed) * 60.0
    return None


def _timeframe_param_value(params: dict[str, Any], base_key: str, timeframe: str) -> Any:
    normalized = _normalize_timeframe(timeframe)
    if not normalized:
        return None
    for suffix in _TIMEFRAME_PARAM_SUFFIXES.get(normalized, (normalized,)):
        key = f"{base_key}_{suffix}"
        if key in params:
            return params.get(key)
    return None


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _resolve_market_data_age_budget_ms(strategy_params: dict[str, Any], timeframe: str) -> int:
    default_budget = max(50, int(safe_float(getattr(settings, "EXECUTION_MARKET_DATA_MAX_AGE_MS", 1200), 1200.0)))
    candidate = _timeframe_param_value(strategy_params, "max_market_data_age_ms", timeframe)
    if candidate is None:
        candidate = strategy_params.get("max_market_data_age_ms")
    parsed = safe_float(candidate, float(default_budget))
    if parsed is None:
        return default_budget
    return max(50, min(300_000, int(parsed)))


def _runtime_signal_market_data_context(runtime_signal: Any) -> dict[str, Any]:
    payload = getattr(runtime_signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}
    strategy_context = payload.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    live_market = payload.get("live_market")
    live_market = live_market if isinstance(live_market, dict) else {}

    timeframe = _normalize_timeframe(
        payload.get("timeframe")
        or strategy_context.get("timeframe")
        or live_market.get("timeframe")
        or live_market.get("cadence")
        or live_market.get("interval")
    )
    source = (
        str(getattr(runtime_signal, "source", None) or payload.get("source") or strategy_context.get("source") or "")
        .strip()
        .lower()
    )

    age_candidates: list[float] = []
    for candidate in (
        live_market.get("market_data_age_ms"),
        live_market.get("age_ms"),
        payload.get("market_data_age_ms"),
        strategy_context.get("market_data_age_ms"),
    ):
        parsed_age = safe_float(candidate, None)
        if parsed_age is None or parsed_age < 0.0:
            continue
        age_candidates.append(float(parsed_age))
    age_ms = min(age_candidates) if age_candidates else None

    observed_candidates: list[datetime] = []
    for candidate in (
        live_market.get("source_observed_at"),
        live_market.get("fetched_at"),
        live_market.get("live_market_fetched_at"),
        live_market.get("signal_updated_at"),
        payload.get("source_observed_at"),
        payload.get("live_market_fetched_at"),
        payload.get("signal_updated_at"),
        strategy_context.get("source_observed_at"),
        strategy_context.get("live_market_fetched_at"),
        getattr(runtime_signal, "updated_at", None),
        getattr(runtime_signal, "created_at", None),
    ):
        parsed_observed_at = _parse_datetime_utc(candidate)
        if parsed_observed_at is None:
            continue
        observed_candidates.append(parsed_observed_at)
    observed_at = max(observed_candidates) if observed_candidates else None

    if age_ms is None and observed_at is not None:
        age_ms = max(
            0.0,
            (datetime.now(timezone.utc) - observed_at.astimezone(timezone.utc)).total_seconds() * 1000.0,
        )

    return {
        "source": source,
        "timeframe": timeframe,
        "age_ms": age_ms,
        "observed_at": observed_at.isoformat().replace("+00:00", "Z") if observed_at is not None else None,
    }


def _runtime_signal_risk_score(runtime_signal: Any) -> float | None:
    direct = safe_float(getattr(runtime_signal, "risk_score", None), None)
    if direct is not None:
        return max(0.0, min(1.0, float(direct)))

    payload = getattr(runtime_signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}
    strategy_context = payload.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
    live_market = payload.get("live_market")
    live_market = live_market if isinstance(live_market, dict) else {}

    for candidate in (
        payload.get("risk_score"),
        strategy_context.get("risk_score"),
        live_market.get("risk_score"),
    ):
        parsed = safe_float(candidate, None)
        if parsed is not None:
            return max(0.0, min(1.0, float(parsed)))
    return None


def _normalize_schedule_days(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    aliases = {
        "mon": "mon",
        "monday": "mon",
        "tue": "tue",
        "tuesday": "tue",
        "wed": "wed",
        "wednesday": "wed",
        "thu": "thu",
        "thursday": "thu",
        "fri": "fri",
        "friday": "fri",
        "sat": "sat",
        "saturday": "sat",
        "sun": "sun",
        "sunday": "sun",
    }
    out: list[str] = []
    seen: set[str] = set()
    for raw in value:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        day = aliases.get(token)
        if day is None and len(token) >= 3:
            day = aliases.get(token[:3])
        if day is None or day in seen:
            continue
        seen.add(day)
        out.append(day)
    return out


def is_within_trading_schedule_utc(metadata: dict[str, Any], now_utc: datetime) -> bool:
    schedule = metadata.get("trading_schedule_utc")
    if not isinstance(schedule, dict):
        return True

    if not bool(schedule.get("enabled", False)):
        return True

    now = now_utc
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    end_at = _parse_datetime_utc(schedule.get("end_at"))
    if end_at is not None and now >= end_at:
        return False

    now_date = now.date()
    start_date = _parse_date_utc(schedule.get("start_date"))
    if start_date is not None and now_date < start_date:
        return False
    end_date = _parse_date_utc(schedule.get("end_date"))
    if end_date is not None and now_date > end_date:
        return False

    days = _normalize_schedule_days(schedule.get("days"))
    if days:
        weekday_map = {
            0: "mon",
            1: "tue",
            2: "wed",
            3: "thu",
            4: "fri",
            5: "sat",
            6: "sun",
        }
        if weekday_map.get(now.weekday()) not in set(days):
            return False

    start = _parse_hhmm_utc(schedule.get("start_time"))
    end = _parse_hhmm_utc(schedule.get("end_time"))
    if start is None or end is None:
        return True

    start_minutes = (start[0] * 60) + start[1]
    end_minutes = (end[0] * 60) + end[1]
    now_minutes = (now.hour * 60) + now.minute

    if start_minutes == end_minutes:
        return True
    if start_minutes < end_minutes:
        return start_minutes <= now_minutes < end_minutes
    return now_minutes >= start_minutes or now_minutes < end_minutes


_RISK_CHECK_KEY_TO_BLOCK_REASON: dict[str, str] = {
    "global_daily_loss": BlockReason.RISK_DAILY_LOSS,
    "trader_daily_loss": BlockReason.RISK_DAILY_LOSS,
    "global_daily_total_loss": BlockReason.RISK_DAILY_LOSS,
    "trader_daily_total_loss": BlockReason.RISK_DAILY_LOSS,
    "global_gross_exposure": BlockReason.RISK_GROSS_EXPOSURE,
    "trader_loss_streak": BlockReason.RISK_CONSECUTIVE_LOSS,
    "trader_cooldown": BlockReason.RISK_CONSECUTIVE_LOSS,
    "trader_trade_notional": BlockReason.RISK_TRADE_NOTIONAL,
    "trader_orders_per_cycle": BlockReason.RISK_OPEN_POSITIONS,
    "trader_open_orders": BlockReason.RISK_OPEN_POSITIONS,
    "trader_open_positions": BlockReason.RISK_OPEN_POSITIONS,
    "trader_market_exposure": BlockReason.RISK_MARKET_EXPOSURE,
}
_GENERIC_FILTER_REASONS: tuple[str, ...] = (
    "crypto worker filters not met",
    "filters not met",
    "strategy skipped",
    "signal skipped",
)


def _risk_block_reason(risk_result: Any) -> str:
    for check in getattr(risk_result, "checks", []) or []:
        if not getattr(check, "passed", True):
            mapped = _RISK_CHECK_KEY_TO_BLOCK_REASON.get(str(getattr(check, "key", "") or ""))
            if mapped:
                return mapped
    return BlockReason.RISK_DAILY_LOSS


def _risk_checks_payload(risk_result: Any) -> list[dict[str, Any]]:
    return [
        {
            "check_key": check.key,
            "check_label": check.key,
            "passed": check.passed,
            "score": check.score,
            "detail": check.detail,
        }
        for check in getattr(risk_result, "checks", []) or []
    ]


def _failed_check_fragments(checks_payload: list[dict[str, Any]], *, max_items: int = 3) -> list[str]:
    fragments: list[str] = []
    for raw_check in checks_payload:
        if not isinstance(raw_check, dict):
            continue
        if bool(raw_check.get("passed", False)):
            continue
        label = str(raw_check.get("check_label") or raw_check.get("check_key") or "check").strip()
        detail = str(raw_check.get("detail") or "").strip()
        if detail:
            compact_detail = " ".join(detail.split())
            if len(compact_detail) > 120:
                compact_detail = compact_detail[:117].rstrip() + "..."
            fragments.append(f"{label}: {compact_detail}")
        else:
            fragments.append(label)
        if len(fragments) >= max_items:
            break
    return fragments


def _enrich_final_reason(
    *,
    final_decision: str,
    final_reason: str,
    checks_payload: list[dict[str, Any]],
) -> str:
    reason = str(final_reason or "").strip()
    failed_fragments = _failed_check_fragments(checks_payload)
    if not failed_fragments:
        return reason

    if not reason:
        if final_decision == "skipped":
            return f"Skipped by checks: {' | '.join(failed_fragments)}"
        if final_decision == "blocked":
            return f"Blocked by checks: {' | '.join(failed_fragments)}"
        return " | ".join(failed_fragments)

    normalized = reason.lower()
    if any(token in normalized for token in _GENERIC_FILTER_REASONS):
        return f"{reason} | failed checks: {' | '.join(failed_fragments)}"

    if normalized.startswith("risk blocked:") and "(" not in reason:
        return f"{reason} ({failed_fragments[0]})"

    return reason


def apply_platform_decision_gates(
    *,
    decision_obj: Any,
    runtime_signal: Any,
    strategy: Any | None,
    checks_payload: list[dict[str, Any]],
    trading_schedule_ok: bool,
    trading_schedule_config: dict[str, Any] | None,
    global_limits: dict[str, Any],
    effective_risk_limits: dict[str, Any],
    allow_averaging: bool,
    open_market_ids: set[str],
    portfolio_allocator: Callable[[float], dict[str, Any]] | None,
    risk_evaluator: Callable[[float], tuple[Any, dict[str, Any]]] | None,
    invoke_hooks: bool,
    pending_live_exit_count: int = 0,
    pending_live_exit_summary: dict[str, Any] | None = None,
    pending_live_exit_max_allowed: int = 0,
    pending_live_exit_identity_guard_enabled: bool = True,
    strategy_params: dict[str, Any] | None = None,
    execution_mode: str = "live",
) -> dict[str, Any]:
    final_decision = str(getattr(decision_obj, "decision", "failed") or "failed")
    final_reason = str(getattr(decision_obj, "reason", "") or "")
    score = getattr(decision_obj, "score", None)
    params = dict(strategy_params or {})
    execution_mode = str(execution_mode or "live").strip().lower() or "live"
    live_execution_gates_enabled = execution_mode != "backtest"
    min_order_floor = StrategySDK.resolve_min_order_size_usd(params, fallback=0.01)
    size_usd = float(max(min_order_floor, safe_float(getattr(decision_obj, "size_usd", None), 10.0)))
    pending_exit_count = max(0, int(pending_live_exit_count or 0))
    pending_exit_max_allowed = max(0, int(pending_live_exit_max_allowed or 0))
    pending_exit_summary = dict(pending_live_exit_summary or {})
    risk_snapshot: dict[str, Any] = {}
    platform_gates: list[dict[str, Any]] = []
    market_data_context = _runtime_signal_market_data_context(runtime_signal)
    strict_ws_gate_recorded = False
    live_revalidation_gate_recorded = False

    if final_decision == "selected":
        if trading_schedule_ok:
            platform_gates.append(
                {
                    "gate": "trading_schedule",
                    "status": "passed",
                    "detail": "Inside configured UTC trading schedule",
                }
            )
        else:
            final_decision = "blocked"
            final_reason = "Outside configured trading schedule (UTC)"
            platform_gates.append(
                {
                    "gate": "trading_schedule",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_blocked"):
                    strategy.on_blocked(
                        runtime_signal,
                        BlockReason.TRADING_WINDOW,
                        {"trading_schedule": trading_schedule_config},
                    )
    else:
        platform_gates.append(
            {
                "gate": "trading_schedule",
                "status": "skipped",
                "detail": f"Skipped because strategy decision is '{final_decision}'",
            }
        )

    if final_decision == "selected":
        gross_cap = safe_float(global_limits.get("max_gross_exposure_usd"), 5000.0)
        notional_default = max(50.0, gross_cap * 0.10)
        max_trade_notional = max(
            1.0,
            safe_float(
                effective_risk_limits.get("max_trade_notional_usd"),
                notional_default,
            ),
        )
        if size_usd > max_trade_notional:
            original_size = size_usd
            size_usd = max_trade_notional
            platform_gates.append(
                {
                    "gate": "size_cap",
                    "status": "capped",
                    "detail": f"Capped to max_trade_notional_usd={max_trade_notional:.2f}",
                    "payload": {
                        "original_size_usd": original_size,
                        "capped_size_usd": size_usd,
                    },
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_size_capped"):
                    strategy.on_size_capped(original_size, size_usd, "Max position size exceeded")
        else:
            platform_gates.append(
                {
                    "gate": "size_cap",
                    "status": "passed",
                    "detail": f"Size {size_usd:.2f} within max_trade_notional_usd={max_trade_notional:.2f}",
                }
            )
    else:
        platform_gates.append(
            {
                "gate": "size_cap",
                "status": "skipped",
                "detail": f"Skipped because decision is '{final_decision}'",
            }
        )

    if final_decision == "selected":
        runtime_payload = getattr(runtime_signal, "payload_json", None)
        runtime_payload = runtime_payload if isinstance(runtime_payload, dict) else {}
        live_market_payload = runtime_payload.get("live_market")
        live_market_payload = live_market_payload if isinstance(live_market_payload, dict) else {}

        source = str(market_data_context.get("source") or "")
        timeframe = str(market_data_context.get("timeframe") or "")
        max_age_ms = _resolve_market_data_age_budget_ms(params, timeframe)
        market_data_source = str(
            live_market_payload.get("market_data_source")
            or live_market_payload.get("live_selected_price_source")
            or "unknown"
        ).strip().lower()
        if not market_data_source:
            market_data_source = "unknown"
        strict_ws_pricing_required = _coerce_bool(params.get("require_strict_ws_pricing"), False)
        strict_ws_price_sources = _normalize_text_list(params.get("strict_ws_price_sources"))
        if not strict_ws_price_sources:
            strict_ws_price_sources = ["ws_strict"]

        live_revalidation_enforced = _coerce_bool(params.get("require_live_market_revalidation"), True)
        live_revalidation_sources = _normalize_text_list(params.get("require_live_revalidation_for_sources"))
        if not live_revalidation_sources:
            live_revalidation_sources = _normalize_text_list(
                params.get("require_market_data_age_for_sources", ["crypto", "scanner"])
            )
        live_revalidation_required = False

        live_selected_price = safe_float(live_market_payload.get("live_selected_price"), None)
        live_fetched_at = _parse_datetime_utc(
            live_market_payload.get("live_market_fetched_at") or live_market_payload.get("fetched_at")
        )
        live_observed_at = _parse_datetime_utc(
            live_market_payload.get("source_observed_at") or live_market_payload.get("signal_updated_at")
        )
        live_age_raw = live_market_payload.get("market_data_age_ms")
        if live_age_raw is None:
            live_age_raw = live_market_payload.get("age_ms")
        live_age_ms = safe_float(live_age_raw, None)
        trusted_live_age_sources = {
            "ws_strict",
            "http_batch",
        }
        if live_age_ms is not None and live_age_ms < 0.0:
            live_age_ms = None
        if live_age_ms is None and live_observed_at is not None:
            live_age_ms = max(
                0.0,
                (datetime.now(timezone.utc) - live_observed_at.astimezone(timezone.utc)).total_seconds() * 1000.0,
            )
        if (
            live_age_ms is None
            and live_fetched_at is not None
            and market_data_source in trusted_live_age_sources
        ):
            live_age_ms = max(
                0.0,
                (datetime.now(timezone.utc) - live_fetched_at.astimezone(timezone.utc)).total_seconds() * 1000.0,
            )

        if not live_execution_gates_enabled:
            checks_payload.append(
                {
                    "check_key": "strict_ws_pricing",
                    "check_label": "Strict WS pricing source",
                    "passed": True,
                    "score": live_age_ms,
                    "detail": "Skipped in backtest mode; strict WS pricing applies only to live execution",
                    "payload": {
                        "required": strict_ws_pricing_required,
                        "market_data_source": market_data_source,
                        "strict_ws_price_sources": strict_ws_price_sources,
                        "live_selected_price": live_selected_price,
                        "live_age_ms": live_age_ms,
                        "execution_mode": execution_mode,
                    },
                }
            )
            platform_gates.append(
                {
                    "gate": "strict_ws_pricing",
                    "status": "skipped",
                    "detail": "Skipped in backtest mode; strict WS pricing applies only to live execution",
                    "payload": {
                        "required": strict_ws_pricing_required,
                        "market_data_source": market_data_source,
                        "strict_ws_price_sources": strict_ws_price_sources,
                        "live_selected_price": live_selected_price,
                        "live_age_ms": live_age_ms,
                        "execution_mode": execution_mode,
                    },
                }
            )
            strict_ws_gate_recorded = True

            checks_payload.append(
                {
                    "check_key": "live_market_revalidation",
                    "check_label": "Execution-time live market revalidation",
                    "passed": True,
                    "score": live_age_ms,
                    "detail": "Skipped in backtest mode; live revalidation applies only to live execution",
                    "payload": {
                        "source": source or None,
                        "timeframe": timeframe or None,
                        "required_sources": live_revalidation_sources,
                        "revalidation_required": False,
                        "live_selected_price": live_selected_price,
                        "live_market_fetched_at": (
                            live_fetched_at.isoformat().replace("+00:00", "Z") if live_fetched_at is not None else None
                        ),
                        "live_observed_at": (
                            live_observed_at.isoformat().replace("+00:00", "Z") if live_observed_at is not None else None
                        ),
                        "live_age_ms": live_age_ms,
                        "max_age_ms": max_age_ms,
                        "market_data_source": market_data_source,
                        "execution_mode": execution_mode,
                    },
                }
            )
            platform_gates.append(
                {
                    "gate": "live_market_revalidation",
                    "status": "skipped",
                    "detail": "Skipped in backtest mode; live revalidation applies only to live execution",
                    "payload": {
                        "source": source or None,
                        "timeframe": timeframe or None,
                        "required_sources": live_revalidation_sources,
                        "live_selected_price": live_selected_price,
                        "live_market_fetched_at": (
                            live_fetched_at.isoformat().replace("+00:00", "Z") if live_fetched_at is not None else None
                        ),
                        "live_age_ms": live_age_ms,
                        "max_age_ms": max_age_ms,
                        "market_data_source": market_data_source,
                        "execution_mode": execution_mode,
                    },
                }
            )
            live_revalidation_gate_recorded = True
        else:
            strict_ws_price_passed = (
                not strict_ws_pricing_required
                or (
                    live_selected_price is not None
                    and live_selected_price > 0.0
                    and market_data_source in set(strict_ws_price_sources)
                    and live_age_ms is not None
                    and live_age_ms <= max_age_ms
                )
            )
            checks_payload.append(
                {
                    "check_key": "strict_ws_pricing",
                    "check_label": "Strict WS pricing source",
                    "passed": strict_ws_price_passed,
                    "score": live_age_ms,
                    "detail": (
                        (
                            f"source={market_data_source} age_ms={live_age_ms:.0f} "
                            f"max={max_age_ms} required_sources={','.join(strict_ws_price_sources)}"
                            if live_age_ms is not None
                            else (
                                f"source={market_data_source} age_ms=unknown "
                                f"required_sources={','.join(strict_ws_price_sources)}"
                            )
                        )
                        if strict_ws_pricing_required
                        else "Strict WS pricing disabled by strategy config"
                    ),
                    "payload": {
                        "required": strict_ws_pricing_required,
                        "market_data_source": market_data_source,
                        "strict_ws_price_sources": strict_ws_price_sources,
                        "live_selected_price": live_selected_price,
                        "live_age_ms": live_age_ms,
                        "execution_mode": execution_mode,
                    },
                }
            )
            if strict_ws_price_passed:
                platform_gates.append(
                    {
                        "gate": "strict_ws_pricing",
                        "status": "passed" if strict_ws_pricing_required else "skipped",
                        "detail": (
                            (
                                f"source={market_data_source} strict sources={','.join(strict_ws_price_sources)} "
                                f"age_ms={live_age_ms:.0f} max={max_age_ms}"
                                if live_age_ms is not None
                                else (
                                    f"source={market_data_source} strict sources={','.join(strict_ws_price_sources)} "
                                    "age_ms=unknown"
                                )
                            )
                            if strict_ws_pricing_required
                            else "Strict WS pricing disabled by strategy config"
                        ),
                        "payload": {
                            "required": strict_ws_pricing_required,
                            "market_data_source": market_data_source,
                            "strict_ws_price_sources": strict_ws_price_sources,
                            "live_selected_price": live_selected_price,
                            "live_age_ms": live_age_ms,
                            "execution_mode": execution_mode,
                        },
                    }
                )
                strict_ws_gate_recorded = True
            else:
                final_decision = "blocked"
                final_reason = (
                    "Strict WS pricing required before execution: "
                    f"source={market_data_source or 'unknown'} "
                    f"age_ms={live_age_ms if live_age_ms is not None else 'unknown'} "
                    f"max={max_age_ms} required={strict_ws_price_sources}"
                )
                platform_gates.append(
                    {
                        "gate": "strict_ws_pricing",
                        "status": "blocked",
                        "detail": final_reason,
                        "payload": {
                            "required": strict_ws_pricing_required,
                            "market_data_source": market_data_source,
                            "strict_ws_price_sources": strict_ws_price_sources,
                            "live_selected_price": live_selected_price,
                            "live_age_ms": live_age_ms,
                            "execution_mode": execution_mode,
                        },
                    }
                )
                strict_ws_gate_recorded = True

            live_revalidation_required = bool(
                final_decision == "selected"
                and live_revalidation_enforced
                and source
                and source in set(live_revalidation_sources)
            )
            live_revalidation_passed = (
                not live_revalidation_required
                or (
                    live_selected_price is not None
                    and live_selected_price > 0.0
                    and live_fetched_at is not None
                    and live_age_ms is not None
                    and live_age_ms <= max_age_ms
                )
            )
            checks_payload.append(
                {
                    "check_key": "live_market_revalidation",
                    "check_label": "Execution-time live market revalidation",
                    "passed": live_revalidation_passed,
                    "score": live_age_ms,
                    "detail": (
                        f"live_age_ms={live_age_ms:.0f} max={max_age_ms} source={source or 'unknown'}"
                        if live_revalidation_required and live_age_ms is not None
                        else (
                            "Live revalidation required but live market context unavailable"
                            if live_revalidation_required
                            else "Live revalidation optional for source"
                        )
                    ),
                    "payload": {
                        "source": source or None,
                        "timeframe": timeframe or None,
                        "required_sources": live_revalidation_sources,
                        "revalidation_required": live_revalidation_required,
                        "live_selected_price": live_selected_price,
                        "live_market_fetched_at": (
                            live_fetched_at.isoformat().replace("+00:00", "Z") if live_fetched_at is not None else None
                        ),
                        "live_observed_at": (
                            live_observed_at.isoformat().replace("+00:00", "Z") if live_observed_at is not None else None
                        ),
                        "live_age_ms": live_age_ms,
                        "max_age_ms": max_age_ms,
                        "market_data_source": market_data_source,
                        "execution_mode": execution_mode,
                    },
                }
            )
            if live_revalidation_passed:
                platform_gates.append(
                    {
                        "gate": "live_market_revalidation",
                        "status": "passed" if live_revalidation_required else "skipped",
                        "detail": (
                            f"live_age_ms={live_age_ms:.0f} max={max_age_ms}"
                            if live_revalidation_required and live_age_ms is not None
                            else "Live revalidation optional for this source"
                        ),
                        "payload": {
                            "source": source or None,
                            "timeframe": timeframe or None,
                            "required_sources": live_revalidation_sources,
                            "live_selected_price": live_selected_price,
                            "live_market_fetched_at": (
                                live_fetched_at.isoformat().replace("+00:00", "Z")
                                if live_fetched_at is not None
                                else None
                            ),
                            "live_age_ms": live_age_ms,
                            "max_age_ms": max_age_ms,
                            "market_data_source": market_data_source,
                            "execution_mode": execution_mode,
                        },
                    }
                )
                live_revalidation_gate_recorded = True
            else:
                final_decision = "blocked"
                final_reason = (
                    "Live market revalidation required before execution: "
                    f"source={source or 'unknown'} age_ms={live_age_ms if live_age_ms is not None else 'unknown'} "
                    f"max={max_age_ms}"
                )
                platform_gates.append(
                    {
                        "gate": "live_market_revalidation",
                        "status": "blocked",
                        "detail": final_reason,
                        "payload": {
                            "source": source or None,
                            "timeframe": timeframe or None,
                            "required_sources": live_revalidation_sources,
                            "live_selected_price": live_selected_price,
                            "live_market_fetched_at": (
                                live_fetched_at.isoformat().replace("+00:00", "Z")
                                if live_fetched_at is not None
                                else None
                            ),
                            "live_age_ms": live_age_ms,
                            "max_age_ms": max_age_ms,
                            "market_data_source": market_data_source,
                            "execution_mode": execution_mode,
                        },
                    }
                )
                live_revalidation_gate_recorded = True
                if invoke_hooks and strategy is not None:
                    if hasattr(strategy, "on_blocked"):
                        strategy.on_blocked(
                            runtime_signal,
                            BlockReason.SIGNAL_EXPIRED,
                            {
                                "source": source or None,
                                "timeframe": timeframe or None,
                                "required_sources": live_revalidation_sources,
                                "live_age_ms": live_age_ms,
                                "max_age_ms": max_age_ms,
                                "market_data_source": market_data_source,
                            },
                        )

    if final_decision == "selected":
        freshness_enforced = _coerce_bool(params.get("enforce_market_data_freshness"), True)
        required_sources = _normalize_text_list(params.get("require_market_data_age_for_sources", ["crypto", "scanner"]))
        source = str(market_data_context.get("source") or "")
        timeframe = str(market_data_context.get("timeframe") or "")
        age_ms = safe_float(market_data_context.get("age_ms"), None)
        observed_at = market_data_context.get("observed_at")
        max_age_ms = _resolve_market_data_age_budget_ms(params, timeframe)
        age_required = bool(source and source in set(required_sources))

        if not live_execution_gates_enabled:
            checks_payload.append(
                {
                    "check_key": "market_data_freshness",
                    "check_label": "Market data freshness",
                    "passed": True,
                    "score": age_ms,
                    "detail": "Skipped in backtest mode; freshness is enforced only for live execution readiness",
                    "payload": {
                        "source": source or None,
                        "timeframe": timeframe or None,
                        "age_ms": age_ms,
                        "max_age_ms": max_age_ms,
                        "observed_at": observed_at,
                        "age_required": age_required,
                        "required_sources": required_sources,
                        "freshness_enforced": freshness_enforced,
                        "market_data_source": market_data_source,
                        "execution_mode": execution_mode,
                    },
                }
            )
            platform_gates.append(
                {
                    "gate": "market_data_freshness",
                    "status": "skipped",
                    "detail": "Skipped in backtest mode; freshness is enforced only for live execution readiness",
                    "payload": {
                        "source": source or None,
                        "timeframe": timeframe or None,
                        "age_ms": age_ms,
                        "max_age_ms": max_age_ms,
                        "observed_at": observed_at,
                        "age_required": age_required,
                        "required_sources": required_sources,
                        "freshness_enforced": freshness_enforced,
                        "market_data_source": market_data_source,
                        "execution_mode": execution_mode,
                    },
                }
            )
        else:
            checks_payload.append(
                {
                    "check_key": "market_data_freshness",
                    "check_label": "Market data freshness",
                    "passed": (
                        not freshness_enforced
                        or (age_ms is not None and age_ms <= max_age_ms)
                        or (age_ms is None and not age_required)
                    ),
                    "score": age_ms,
                    "detail": (
                        "Freshness gate disabled by strategy config"
                        if not freshness_enforced
                        else (
                            f"age_ms={age_ms:.0f} max={max_age_ms} source={source or 'unknown'} "
                            f"timeframe={timeframe or 'unknown'}"
                            if age_ms is not None
                            else (f"age unavailable; source={source or 'unknown'} required={age_required}")
                        )
                    ),
                    "payload": {
                        "source": source or None,
                        "timeframe": timeframe or None,
                        "age_ms": age_ms,
                        "max_age_ms": max_age_ms,
                        "observed_at": observed_at,
                        "age_required": age_required,
                        "required_sources": required_sources,
                        "freshness_enforced": freshness_enforced,
                        "market_data_source": market_data_source,
                        "execution_mode": execution_mode,
                    },
                }
            )

            freshness_passed = (
                (not freshness_enforced)
                or (age_ms is not None and age_ms <= max_age_ms)
                or (age_ms is None and not age_required)
            )
            if freshness_passed:
                platform_gates.append(
                    {
                        "gate": "market_data_freshness",
                        "status": "passed" if freshness_enforced else "skipped",
                        "detail": (
                            f"age_ms={age_ms:.0f} max={max_age_ms}"
                            if freshness_enforced and age_ms is not None
                            else (
                                "Freshness gate disabled by strategy config"
                                if not freshness_enforced
                                else "Age unavailable but optional for this source"
                            )
                        ),
                        "payload": {
                            "source": source or None,
                            "timeframe": timeframe or None,
                            "age_ms": age_ms,
                            "max_age_ms": max_age_ms,
                            "observed_at": observed_at,
                            "age_required": age_required,
                            "required_sources": required_sources,
                            "freshness_enforced": freshness_enforced,
                            "market_data_source": market_data_source,
                            "execution_mode": execution_mode,
                        },
                    }
                )
            else:
                final_decision = "blocked"
                final_reason = (
                    f"Market data freshness gate blocked: source={source or 'unknown'} "
                    f"age_ms={age_ms if age_ms is not None else 'unknown'} max={max_age_ms}"
                )
                platform_gates.append(
                    {
                        "gate": "market_data_freshness",
                        "status": "blocked",
                        "detail": final_reason,
                        "payload": {
                            "source": source or None,
                            "timeframe": timeframe or None,
                            "age_ms": age_ms,
                            "max_age_ms": max_age_ms,
                            "observed_at": observed_at,
                            "age_required": age_required,
                            "required_sources": required_sources,
                            "freshness_enforced": freshness_enforced,
                            "market_data_source": market_data_source,
                            "execution_mode": execution_mode,
                        },
                    }
                )
                if invoke_hooks and strategy is not None:
                    if hasattr(strategy, "on_blocked"):
                        strategy.on_blocked(
                            runtime_signal,
                            BlockReason.SIGNAL_EXPIRED,
                            {
                                "market_data_context": market_data_context,
                                "max_age_ms": max_age_ms,
                                "required_sources": required_sources,
                                "market_data_source": market_data_source,
                            },
                        )
    else:
        if not strict_ws_gate_recorded:
            platform_gates.append(
                {
                    "gate": "strict_ws_pricing",
                    "status": "skipped",
                    "detail": f"Skipped because decision is '{final_decision}'",
                }
            )
        if not live_revalidation_gate_recorded:
            platform_gates.append(
                {
                    "gate": "live_market_revalidation",
                    "status": "skipped",
                    "detail": f"Skipped because decision is '{final_decision}'",
                }
            )
        platform_gates.append(
            {
                "gate": "market_data_freshness",
                "status": "skipped",
                "detail": f"Skipped because decision is '{final_decision}'",
            }
        )

    if final_decision == "selected":
        signal_direction = str(getattr(runtime_signal, "direction", "") or "").strip().lower()
        source = str(market_data_context.get("source") or "").strip().lower()
        timeframe = str(market_data_context.get("timeframe") or "").strip().lower()
        timeframe_minutes = _parse_timeframe_minutes(timeframe)
        min_timeframe_minutes = safe_float(
            params.get("live_directional_min_timeframe_minutes")
            if params.get("live_directional_min_timeframe_minutes") is not None
            else params.get("directional_min_timeframe_minutes"),
            None,
        )
        if min_timeframe_minutes is None:
            min_timeframe_minutes = _parse_timeframe_minutes(
                params.get("live_directional_min_timeframe")
                if params.get("live_directional_min_timeframe") is not None
                else params.get("directional_min_timeframe", "5m")
            )
        if min_timeframe_minutes is None:
            min_timeframe_minutes = 5.0
        min_timeframe_minutes = max(1.0, float(min_timeframe_minutes))

        directional_gate_enabled = _coerce_bool(params.get("enforce_live_directional_timeframe"), True)
        if params.get("enforce_directional_timeframe") is not None:
            directional_gate_enabled = _coerce_bool(
                params.get("enforce_directional_timeframe"),
                directional_gate_enabled,
            )
        is_directional_signal = signal_direction in {
            "buy_yes",
            "buy_no",
            "buy",
            "sell",
            "yes",
            "no",
            "long",
            "short",
            "up",
            "down",
        }
        should_enforce_directional_gate = directional_gate_enabled and is_directional_signal and source == "crypto"
        directional_gate_passed = (not should_enforce_directional_gate) or (
            timeframe_minutes is not None and timeframe_minutes + 1e-9 >= min_timeframe_minutes
        )
        checks_payload.append(
            {
                "check_key": "directional_min_timeframe",
                "check_label": "Directional minimum timeframe",
                "passed": directional_gate_passed,
                "score": timeframe_minutes,
                "detail": (
                    (f"timeframe={timeframe} ({timeframe_minutes:.0f}m) >= required {min_timeframe_minutes:.0f}m")
                    if should_enforce_directional_gate and timeframe_minutes is not None and directional_gate_passed
                    else (
                        f"timeframe={timeframe or 'unknown'} below required {min_timeframe_minutes:.0f}m"
                        if should_enforce_directional_gate
                        else "Gate not applicable for this signal"
                    )
                ),
                "payload": {
                    "enabled": directional_gate_enabled,
                    "applied": should_enforce_directional_gate,
                    "source": source or None,
                    "direction": signal_direction or None,
                    "timeframe": timeframe or None,
                    "timeframe_minutes": timeframe_minutes,
                    "min_timeframe_minutes": min_timeframe_minutes,
                },
            }
        )
        if should_enforce_directional_gate and not directional_gate_passed:
            final_decision = "blocked"
            final_reason = (
                f"Directional timeframe guard blocked: timeframe={timeframe or 'unknown'} "
                f"requires >= {min_timeframe_minutes:.0f}m"
            )
            platform_gates.append(
                {
                    "gate": "directional_min_timeframe",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_blocked"):
                    strategy.on_blocked(
                        runtime_signal,
                        BlockReason.SIGNAL_EXPIRED,
                        {
                            "source": source,
                            "direction": signal_direction,
                            "timeframe": timeframe,
                            "timeframe_minutes": timeframe_minutes,
                            "min_timeframe_minutes": min_timeframe_minutes,
                        },
                    )
        else:
            platform_gates.append(
                {
                    "gate": "directional_min_timeframe",
                    "status": ("passed" if should_enforce_directional_gate else "skipped"),
                    "detail": (
                        f"Directional signal timeframe satisfied ({timeframe or 'unknown'})"
                        if should_enforce_directional_gate
                        else "Skipped for non-directional/non-crypto signal"
                    ),
                }
            )
    else:
        platform_gates.append(
            {
                "gate": "directional_min_timeframe",
                "status": "skipped",
                "detail": f"Skipped because decision is '{final_decision}'",
            }
        )

    if final_decision == "selected":
        max_risk_score = safe_float(params.get("max_risk_score"), None)
        signal_risk_score = _runtime_signal_risk_score(runtime_signal)
        risk_gate_enabled = max_risk_score is not None
        risk_gate_passed = (
            (not risk_gate_enabled) or signal_risk_score is None or signal_risk_score <= float(max_risk_score)
        )
        checks_payload.append(
            {
                "check_key": "max_risk_score_guard",
                "check_label": "Maximum risk score",
                "passed": risk_gate_passed,
                "score": signal_risk_score,
                "detail": (
                    "Guard disabled (max_risk_score unset)"
                    if not risk_gate_enabled
                    else (
                        f"risk_score={signal_risk_score:.3f} <= max={float(max_risk_score):.3f}"
                        if signal_risk_score is not None and risk_gate_passed
                        else (
                            "Signal risk unavailable; guard skipped"
                            if signal_risk_score is None
                            else f"risk_score={signal_risk_score:.3f} exceeds max={float(max_risk_score):.3f}"
                        )
                    )
                ),
                "payload": {
                    "enabled": risk_gate_enabled,
                    "signal_risk_score": signal_risk_score,
                    "max_risk_score": float(max_risk_score) if max_risk_score is not None else None,
                },
            }
        )
        if risk_gate_enabled and signal_risk_score is not None and not risk_gate_passed:
            final_decision = "blocked"
            final_reason = (
                f"Max-risk guard blocked: risk_score={signal_risk_score:.3f} "
                f"> max_risk_score={float(max_risk_score):.3f}"
            )
            platform_gates.append(
                {
                    "gate": "max_risk_score",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None and hasattr(strategy, "on_blocked"):
                strategy.on_blocked(
                    runtime_signal,
                    BlockReason.RISK_TRADE_NOTIONAL,
                    {
                        "signal_risk_score": signal_risk_score,
                        "max_risk_score": float(max_risk_score),
                    },
                )
        else:
            platform_gates.append(
                {
                    "gate": "max_risk_score",
                    "status": "passed" if risk_gate_enabled else "skipped",
                    "detail": (
                        f"risk_score={signal_risk_score:.3f} max={float(max_risk_score):.3f}"
                        if risk_gate_enabled and signal_risk_score is not None
                        else ("Signal risk unavailable; guard skipped" if risk_gate_enabled else "Guard not configured")
                    ),
                }
            )
    else:
        platform_gates.append(
            {
                "gate": "max_risk_score",
                "status": "skipped",
                "detail": f"Skipped because decision is '{final_decision}'",
            }
        )

    if final_decision == "selected":
        min_order_size_usd = StrategySDK.resolve_min_order_size_usd(params, fallback=1.0)
        entry_price = safe_float(getattr(runtime_signal, "entry_price", None), None)
        runtime_payload = getattr(runtime_signal, "payload_json", None)
        if (entry_price is None or entry_price <= 0.0) and isinstance(runtime_payload, dict):
            live_market_payload = runtime_payload.get("live_market")
            if isinstance(live_market_payload, dict):
                entry_price = safe_float(live_market_payload.get("live_selected_price"), None)
                if entry_price is None or entry_price <= 0.0:
                    entry_price = safe_float(live_market_payload.get("signal_entry_price"), None)
            if entry_price is None or entry_price <= 0.0:
                entry_price = safe_float(runtime_payload.get("entry_price"), None)

        min_exit_guard_enabled = _coerce_bool(params.get("enforce_min_exit_notional"), True)
        stop_loss_pct = safe_float(params.get("live_stop_loss_pct"), None)
        if stop_loss_pct is None:
            stop_loss_pct = safe_float(params.get("stop_loss_pct"), None)
        stop_loss_policy_raw = params.get("live_stop_loss_policy")
        if stop_loss_policy_raw is None:
            stop_loss_policy_raw = params.get("stop_loss_policy")
        if stop_loss_policy_raw is None:
            stop_loss_policy_raw = params.get("stop_loss_mode")
        stop_loss_policy = str(stop_loss_policy_raw or "always").strip().lower()
        stop_loss_near_close_only = stop_loss_policy in {"near_close", "near_close_only", "close_window"}
        stop_loss_activation_seconds = safe_float(params.get("live_stop_loss_activation_seconds"), None)
        if stop_loss_activation_seconds is None:
            stop_loss_activation_seconds = safe_float(params.get("stop_loss_activation_seconds"), None)
        if stop_loss_activation_seconds is None:
            stop_loss_activation_seconds = safe_float(params.get("live_stop_loss_near_close_seconds"), None)
        if stop_loss_activation_seconds is None:
            stop_loss_activation_seconds = safe_float(params.get("stop_loss_near_close_seconds"), None)
        if stop_loss_activation_seconds is None:
            stop_loss_activation_seconds = 120.0
        stop_loss_activation_seconds = max(0.0, float(stop_loss_activation_seconds))
        signal_seconds_left = _runtime_signal_seconds_left(runtime_payload)
        stop_loss_armed = (not stop_loss_near_close_only) or (
            signal_seconds_left is not None and signal_seconds_left <= stop_loss_activation_seconds
        )
        configured_exit_price_ratio = safe_float(params.get("live_exit_price_ratio_floor"), None)
        if configured_exit_price_ratio is None:
            configured_exit_price_ratio = safe_float(params.get("exit_price_ratio_floor"), None)
        if configured_exit_price_ratio is not None and (
            configured_exit_price_ratio <= 0.0 or configured_exit_price_ratio >= 1.0
        ):
            configured_exit_price_ratio = None
        fallback_exit_price_ratio = 0.5
        exit_price_floor = safe_float(params.get("live_exit_price_floor"), None)
        if exit_price_floor is None:
            exit_price_floor = safe_float(params.get("exit_price_floor"), None)
        if exit_price_floor is None or exit_price_floor <= 0.0:
            exit_price_floor = 0.01

        required_size_usd = min_order_size_usd
        conservative_exit_price = None
        conservative_exit_price_ratio = None
        conservative_exit_source = ""
        min_exit_notional_passed = True
        if min_exit_guard_enabled:
            if entry_price is not None and entry_price > 0.0:
                if stop_loss_pct is not None and 0.0 < stop_loss_pct < 100.0 and stop_loss_armed:
                    stop_loss_price = entry_price * (1.0 - (stop_loss_pct / 100.0))
                    conservative_exit_price = max(exit_price_floor, stop_loss_price)
                    conservative_exit_source = "stop_loss_pct"
                else:
                    ratio_to_use = configured_exit_price_ratio
                    conservative_exit_source = "configured_ratio_floor"
                    if ratio_to_use is None:
                        ratio_to_use = fallback_exit_price_ratio
                        conservative_exit_source = "default_ratio_floor"
                    conservative_exit_price = max(exit_price_floor, entry_price * ratio_to_use)
                conservative_exit_ratio = conservative_exit_price / entry_price if entry_price > 0.0 else 0.0
                conservative_exit_price_ratio = conservative_exit_ratio if conservative_exit_ratio > 0.0 else None
                if conservative_exit_ratio > 0.0:
                    required_size_usd = max(required_size_usd, min_order_size_usd / conservative_exit_ratio)

            min_exit_notional_passed = size_usd + 1e-9 >= required_size_usd
        else:
            conservative_exit_source = "guard_disabled"
        checks_payload.append(
            {
                "check_key": "min_exit_notional_guard",
                "check_label": "Minimum exit notional feasibility",
                "passed": min_exit_notional_passed,
                "score": size_usd,
                "detail": (
                    "Guard disabled by strategy config"
                    if not min_exit_guard_enabled
                    else (
                        f"size {size_usd:.2f} supports min exit notional at conservative_exit_price={conservative_exit_price:.4f}"
                        if min_exit_notional_passed and conservative_exit_price is not None
                        else (
                            f"size {size_usd:.2f} meets min_order_size_usd={min_order_size_usd:.2f} (entry price unavailable)"
                            if min_exit_notional_passed
                            else (f"size {size_usd:.2f} is below required min feasible size {required_size_usd:.2f}")
                        )
                    )
                ),
                "payload": {
                    "enabled": min_exit_guard_enabled,
                    "entry_price": entry_price,
                    "stop_loss_pct": stop_loss_pct,
                    "stop_loss_policy": stop_loss_policy,
                    "stop_loss_activation_seconds": stop_loss_activation_seconds,
                    "signal_seconds_left": signal_seconds_left,
                    "stop_loss_armed": stop_loss_armed,
                    "min_order_size_usd": min_order_size_usd,
                    "required_size_usd": required_size_usd,
                    "conservative_exit_price": conservative_exit_price,
                    "conservative_exit_price_ratio": conservative_exit_price_ratio,
                    "conservative_exit_source": conservative_exit_source,
                    "exit_price_floor": exit_price_floor,
                },
            }
        )

        if min_exit_guard_enabled and not min_exit_notional_passed:
            final_decision = "blocked"
            final_reason = (
                f"Min-exit-notional guard blocked: required size >= {required_size_usd:.2f} "
                f"for min exit ${min_order_size_usd:.2f}"
            )
            platform_gates.append(
                {
                    "gate": "min_exit_notional",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_blocked"):
                    strategy.on_blocked(
                        runtime_signal,
                        BlockReason.RISK_TRADE_NOTIONAL,
                        {
                            "required_size_usd": required_size_usd,
                            "min_order_size_usd": min_order_size_usd,
                            "entry_price": entry_price,
                            "conservative_exit_price": conservative_exit_price,
                        },
                    )
        elif min_exit_guard_enabled:
            platform_gates.append(
                {
                    "gate": "min_exit_notional",
                    "status": "passed",
                    "detail": (f"Size supports min exit notional with required_size_usd={required_size_usd:.2f}"),
                }
            )
        else:
            platform_gates.append(
                {
                    "gate": "min_exit_notional",
                    "status": "skipped",
                    "detail": "Skipped because enforce_min_exit_notional=false",
                }
            )
    else:
        platform_gates.append(
            {
                "gate": "min_exit_notional",
                "status": "skipped",
                "detail": f"Skipped because decision is '{final_decision}'",
            }
        )

    if final_decision == "selected" and portfolio_allocator is not None:
        portfolio_result = portfolio_allocator(size_usd) or {}
        requested_size_usd = float(max(0.0, size_usd))
        allocated_size_usd = float(
            max(
                0.0,
                safe_float(portfolio_result.get("size_usd"), requested_size_usd),
            )
        )
        allocation_allowed = bool(portfolio_result.get("allowed", False))
        allocation_reason = str(portfolio_result.get("reason", "") or "")
        portfolio_snapshot = {
            "allowed": allocation_allowed,
            "reason": allocation_reason,
            "requested_size_usd": requested_size_usd,
            "allocated_size_usd": allocated_size_usd,
            "target_gross_cap_usd": safe_float(portfolio_result.get("target_gross_cap_usd"), None),
            "remaining_gross_cap_usd": safe_float(portfolio_result.get("remaining_gross_cap_usd"), None),
            "source_key": str(portfolio_result.get("source_key", "") or ""),
            "source_cap_usd": safe_float(portfolio_result.get("source_cap_usd"), None),
            "source_exposure_usd": safe_float(portfolio_result.get("source_exposure_usd"), None),
            "source_remaining_usd": safe_float(portfolio_result.get("source_remaining_usd"), None),
            "min_order_notional_usd": safe_float(portfolio_result.get("min_order_notional_usd"), None),
            "target_utilization_pct": safe_float(portfolio_result.get("target_utilization_pct"), None),
            "max_source_exposure_pct": safe_float(portfolio_result.get("max_source_exposure_pct"), None),
        }
        risk_snapshot["portfolio"] = portfolio_snapshot
        checks_payload.append(
            {
                "check_key": "portfolio_allocator",
                "check_label": "Portfolio allocation",
                "passed": allocation_allowed and allocated_size_usd > 0.0,
                "score": allocated_size_usd,
                "detail": allocation_reason
                or (
                    f"allocated {allocated_size_usd:.2f} from requested {requested_size_usd:.2f}"
                    if allocation_allowed
                    else "Allocation blocked"
                ),
                "payload": portfolio_snapshot,
            }
        )

        if not allocation_allowed or allocated_size_usd <= 0.0:
            final_decision = "blocked"
            final_reason = allocation_reason or "Portfolio allocator blocked signal"
            platform_gates.append(
                {
                    "gate": "portfolio",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_blocked"):
                    strategy.on_blocked(
                        runtime_signal, BlockReason.RISK_GROSS_EXPOSURE, {"portfolio": portfolio_snapshot}
                    )
        elif allocated_size_usd < size_usd:
            original_size = size_usd
            size_usd = allocated_size_usd
            platform_gates.append(
                {
                    "gate": "portfolio",
                    "status": "capped",
                    "detail": allocation_reason or "Portfolio allocator reduced position size",
                    "payload": {
                        "original_size_usd": original_size,
                        "capped_size_usd": size_usd,
                    },
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_size_capped"):
                    strategy.on_size_capped(original_size, size_usd, "Portfolio allocation cap")
        else:
            platform_gates.append(
                {
                    "gate": "portfolio",
                    "status": "passed",
                    "detail": allocation_reason or "Portfolio allocation accepted requested size",
                }
            )
    else:
        platform_gates.append(
            {
                "gate": "portfolio",
                "status": "skipped",
                "detail": (
                    "Skipped because no portfolio allocator was provided"
                    if portfolio_allocator is None
                    else f"Skipped because decision is '{final_decision}'"
                ),
            }
        )

    if final_decision == "selected" and risk_evaluator is not None:
        risk_result, risk_snapshot_base = risk_evaluator(size_usd)
        if isinstance(risk_snapshot_base, dict):
            risk_snapshot.update(risk_snapshot_base)
        risk_checks = _risk_checks_payload(risk_result)
        checks_payload.extend(risk_checks)
        risk_snapshot.update(
            {
                "allowed": bool(getattr(risk_result, "allowed", False)),
                "reason": str(getattr(risk_result, "reason", "") or ""),
                "checks": risk_checks,
            }
        )

        if bool(getattr(risk_result, "allowed", False)):
            platform_gates.append(
                {
                    "gate": "risk",
                    "status": "passed",
                    "detail": str(getattr(risk_result, "reason", "") or "Risk checks passed"),
                }
            )
        else:
            final_decision = "blocked"
            final_reason = str(getattr(risk_result, "reason", "") or "Risk blocked")
            platform_gates.append(
                {
                    "gate": "risk",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_blocked"):
                    strategy.on_blocked(
                        runtime_signal,
                        _risk_block_reason(risk_result),
                        {"risk_snapshot": risk_snapshot},
                    )
    else:
        platform_gates.append(
            {
                "gate": "risk",
                "status": "skipped",
                "detail": (
                    "Skipped because no risk evaluator was provided"
                    if risk_evaluator is None
                    else f"Skipped because decision is '{final_decision}'"
                ),
            }
        )

    if final_decision == "selected":
        pending_exit_guard_passed = pending_exit_count <= pending_exit_max_allowed
        pending_exit_detail = (
            f"Pending live exits <= {pending_exit_max_allowed} (current={pending_exit_count})"
            if pending_exit_max_allowed > 0
            else (
                "No non-terminal pending live exits detected"
                if pending_exit_guard_passed
                else f"{pending_exit_count} non-terminal pending live exit(s) detected"
            )
        )
        checks_payload.append(
            {
                "check_key": "pending_live_exit_guard",
                "check_label": "Pending live exits clear",
                "passed": pending_exit_guard_passed,
                "score": float(pending_exit_count),
                "detail": pending_exit_detail,
                "payload": {
                    "count": pending_exit_count,
                    "max_allowed": pending_exit_max_allowed,
                    "statuses": dict(pending_exit_summary.get("statuses") or {}),
                    "order_ids": list(pending_exit_summary.get("order_ids") or []),
                    "market_ids": list(pending_exit_summary.get("market_ids") or []),
                    "signal_ids": list(pending_exit_summary.get("signal_ids") or []),
                    "identities": list(pending_exit_summary.get("identities") or []),
                    "identity_keys": list(pending_exit_summary.get("identity_keys") or []),
                },
            }
        )
        if pending_exit_guard_passed:
            platform_gates.append(
                {
                    "gate": "pending_live_exit_guard",
                    "status": "passed",
                    "detail": pending_exit_detail,
                }
            )
        else:
            final_decision = "blocked"
            final_reason = (
                "Pending live exit guard blocked: "
                f"{pending_exit_count} pending close order(s) in-flight (max_allowed={pending_exit_max_allowed})"
            )
            platform_gates.append(
                {
                    "gate": "pending_live_exit_guard",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_blocked"):
                    strategy.on_blocked(
                        runtime_signal,
                        BlockReason.RISK_OPEN_POSITIONS,
                        {"pending_live_exit": pending_exit_summary},
                    )
    else:
        platform_gates.append(
            {
                "gate": "pending_live_exit_guard",
                "status": "skipped",
                "detail": f"Skipped because decision is '{final_decision}'",
            }
        )

    if final_decision == "selected" and pending_live_exit_identity_guard_enabled:
        signal_market_id = str(getattr(runtime_signal, "market_id", "") or "").strip()
        signal_direction = str(getattr(runtime_signal, "direction", "") or "").strip().lower()
        signal_id = str(getattr(runtime_signal, "id", "") or "").strip()
        matching_pending_identity: dict[str, Any] | None = None
        identities = pending_exit_summary.get("identities")
        if isinstance(identities, list):
            for item in identities:
                if not isinstance(item, dict):
                    continue
                item_market_id = str(item.get("market_id") or "").strip()
                item_direction = str(item.get("direction") or "").strip().lower()
                item_signal_id = str(item.get("signal_id") or "").strip()
                if not item_market_id or not item_direction:
                    continue
                if item_market_id != signal_market_id or item_direction != signal_direction:
                    continue
                if item_signal_id and signal_id and item_signal_id != signal_id:
                    continue
                matching_pending_identity = {
                    "order_id": str(item.get("order_id") or "").strip() or None,
                    "market_id": item_market_id,
                    "direction": item_direction,
                    "signal_id": item_signal_id or None,
                    "status": str(item.get("status") or "").strip().lower() or None,
                }
                break
        identity_guard_passed = matching_pending_identity is None
        checks_payload.append(
            {
                "check_key": "pending_live_exit_identity_guard",
                "check_label": "Pending live exit identity clear",
                "passed": identity_guard_passed,
                "score": None,
                "detail": (
                    "No matching non-terminal pending live exit identity"
                    if identity_guard_passed
                    else "Matching pending live exit identity is still in-flight"
                ),
                "payload": {
                    "signal_market_id": signal_market_id or None,
                    "signal_direction": signal_direction or None,
                    "signal_id": signal_id or None,
                    "match": matching_pending_identity,
                },
            }
        )
        if identity_guard_passed:
            platform_gates.append(
                {
                    "gate": "pending_live_exit_identity_guard",
                    "status": "passed",
                    "detail": "No matching pending live exit identity",
                }
            )
        else:
            final_decision = "blocked"
            final_reason = "Pending live exit identity guard: matching market/direction exit still pending"
            platform_gates.append(
                {
                    "gate": "pending_live_exit_identity_guard",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_blocked"):
                    strategy.on_blocked(
                        runtime_signal,
                        BlockReason.RISK_OPEN_POSITIONS,
                        {"pending_live_exit_identity": matching_pending_identity},
                    )
    elif final_decision == "selected":
        checks_payload.append(
            {
                "check_key": "pending_live_exit_identity_guard",
                "check_label": "Pending live exit identity clear",
                "passed": True,
                "score": None,
                "detail": "Disabled by global runtime setting",
                "payload": {
                    "enabled": False,
                },
            }
        )
        platform_gates.append(
            {
                "gate": "pending_live_exit_identity_guard",
                "status": "skipped",
                "detail": "Disabled by global runtime setting",
            }
        )
    else:
        platform_gates.append(
            {
                "gate": "pending_live_exit_identity_guard",
                "status": "skipped",
                "detail": f"Skipped because decision is '{final_decision}'",
            }
        )

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
            platform_gates.append(
                {
                    "gate": "stacking_guard",
                    "status": "blocked",
                    "detail": final_reason,
                }
            )
            if invoke_hooks and strategy is not None:
                if hasattr(strategy, "on_blocked"):
                    strategy.on_blocked(runtime_signal, BlockReason.STACKING_GUARD, {"market_id": signal_market_id})
        else:
            platform_gates.append(
                {
                    "gate": "stacking_guard",
                    "status": "passed",
                    "detail": "No existing open position for market",
                }
            )
    else:
        platform_gates.append(
            {
                "gate": "stacking_guard",
                "status": "skipped",
                "detail": (
                    "Skipped because allow_averaging=true"
                    if allow_averaging
                    else f"Skipped because decision is '{final_decision}'"
                ),
            }
        )

    final_reason = _enrich_final_reason(
        final_decision=final_decision,
        final_reason=final_reason,
        checks_payload=checks_payload,
    )

    return {
        "strategy_decision": str(getattr(decision_obj, "decision", "failed") or "failed"),
        "strategy_reason": str(getattr(decision_obj, "reason", "") or ""),
        "final_decision": final_decision,
        "final_reason": final_reason,
        "score": score,
        "size_usd": size_usd,
        "checks_payload": checks_payload,
        "risk_snapshot": risk_snapshot,
        "platform_gates": platform_gates,
    }
