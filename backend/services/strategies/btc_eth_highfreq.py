"""
Strategy: BTC/ETH High-Frequency Market Making

Specialized strategy for Bitcoin and Ethereum binary markets on Polymarket,
targeting the highly liquid 5-minute, 15-minute, 1-hour and 4-hour
"up or down" markets.

Three-layer architecture:

  Layer 1: Maker Quote (PRIMARY)
    Two-sided post_only limit orders on YES and NO sides.
    Earns bid-ask spread + maker rebates (0% taker fee).
    Oracle-gated inventory skew tilts quotes toward the predicted winner.

  Layer 2: Directional Edge
    Oracle-gated directional bet with fee-aware entry gating.
    Edge must exceed 2x taker fee to enter, preventing the historical
    pattern of sub-fee-threshold entries that caused 100% loss records.

  Layer 3: Convergence
    Near-expiry (5-45 seconds) maker order on the oracle-predicted
    winning side at $0.85-$0.95.  The market converges to $1.00/$0.00
    at resolution, so any fill is profitable.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import math

from models import Market, Event, Opportunity
from config import settings as _cfg
from .base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.data_events import DataEvent
from services.strategy_sdk import StrategySDK
from utils.converters import to_float, to_confidence, to_bool, clamp
from utils.signal_helpers import signal_payload
from services.quality_filter import QualityFilterOverrides
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evaluate-method constants (ported from BaseCryptoTimeframeStrategy)
# ---------------------------------------------------------------------------

_ALLOWED_MODES = {"auto", "directional", "maker_quote", "convergence"}
_REGIMES = {"opening", "mid", "closing"}

_EDGE_MODE_FACTORS: dict[str, dict[str, float]] = {
    "opening": {"auto": 1.0, "directional": 1.05, "maker_quote": 0.90, "convergence": 0.95},
    "mid": {"auto": 1.0, "directional": 1.00, "maker_quote": 0.85, "convergence": 0.90},
    "closing": {"auto": 0.9, "directional": 0.90, "maker_quote": 0.80, "convergence": 0.85},
}

_CONF_MODE_FACTORS: dict[str, float] = {
    "auto": 1.0,
    "directional": 1.0,
    "maker_quote": 0.9,
    "convergence": 0.95,
}

_REGIME_CONF_FACTORS: dict[str, float] = {
    "opening": 1.0,
    "mid": 1.0,
    "closing": 0.95,
}

_MODE_SIZE_FACTORS: dict[str, float] = {
    "auto": 1.0,
    "directional": 1.0,
    "maker_quote": 0.85,
    "convergence": 0.9,
}

_REGIME_SIZE_FACTORS: dict[str, float] = {
    "opening": 0.95,
    "mid": 1.0,
    "closing": 1.1,
}


# ---------------------------------------------------------------------------
# Evaluate-method helpers (ported from BaseCryptoTimeframeStrategy)
# ---------------------------------------------------------------------------


def _normalize_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in _ALLOWED_MODES:
        return "auto"
    return mode


def _normalize_regime(value: Any) -> str:
    regime = str(value or "mid").strip().lower()
    if regime not in _REGIMES:
        return "mid"
    return regime


def _normalize_asset(value: Any) -> str:
    asset = str(value or "").strip().upper()
    if asset == "XBT":
        return "BTC"
    return asset


def _normalize_timeframe(value: Any) -> str:
    tf = str(value or "").strip().lower()
    if tf in {"5m", "5min", "5"}:
        return "5m"
    if tf in {"15m", "15min", "15"}:
        return "15m"
    if tf in {"1h", "1hr", "60m", "60min"}:
        return "1h"
    if tf in {"4h", "4hr", "240m", "240min"}:
        return "4h"
    return tf


_TIMEFRAME_PARAM_SUFFIXES: dict[str, tuple[str, ...]] = {
    "5m": ("5m", "5min"),
    "15m": ("15m", "15min"),
    "1h": ("1h", "1hr", "60m"),
    "4h": ("4h", "4hr", "240m"),
}


def _timeframe_override(params: dict[str, Any], base_key: str, timeframe: str) -> Any:
    normalized_tf = _normalize_timeframe(timeframe)
    if not normalized_tf:
        return None
    for suffix in _TIMEFRAME_PARAM_SUFFIXES.get(normalized_tf, (normalized_tf,)):
        key = f"{base_key}_{suffix}"
        if key in params:
            return params.get(key)
    return None


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        parsed = default
    return max(lo, min(hi, parsed))


def _crypto_hf_param_value(config: dict[str, Any], base_key: str, timeframe: Any) -> Any:
    timeframe_value = _normalize_timeframe(timeframe)
    tf_value = _timeframe_override(config, base_key, timeframe_value)
    if tf_value is not None:
        return tf_value
    return config.get(base_key)


CRYPTO_HF_SCOPE_DEFAULTS: dict[str, Any] = {
    "min_edge_percent": 2.0,
    "min_confidence": 0.40,
    "max_risk_score": 0.80,
    "base_size_usd": 20.0,
    "max_size_usd": 150.0,
    "include_assets": ["BTC", "ETH", "SOL", "XRP"],
    "exclude_assets": [],
    "include_timeframes": ["5m", "15m", "1h", "4h"],
    "exclude_timeframes": [],
    "enabled_sub_strategies": ["maker_quote", "directional_edge", "convergence"],
    "live_window_required": True,
    "min_liquidity_usd": 250.0,
    "min_liquidity_usd_opening": 4000.0,
    "max_spread_pct": 0.08,
    "max_signal_age_seconds": 35.0,
    "max_open_order_seconds": 14.0,
    "max_signal_age_seconds_5m": 1.25,
    "max_signal_age_seconds_15m": 2.5,
    "max_signal_age_seconds_1h": 4.0,
    "max_signal_age_seconds_4h": 6.0,
    "max_market_data_age_ms": 1200,
    "max_market_data_age_ms_5m": 700,
    "max_market_data_age_ms_15m": 900,
    "max_market_data_age_ms_1h": 1200,
    "max_market_data_age_ms_4h": 1500,
    "enforce_market_data_freshness": True,
    "require_market_data_age_for_sources": ["crypto"],
    "max_live_context_age_seconds": 3.0,
    "max_oracle_age_seconds": 20.0,
    "max_oracle_age_ms": 20_000.0,
    "require_oracle_for_directional": True,
    "oracle_source_policy": "degrade",
    "oracle_fallback_degrade_edge_multiplier": 1.35,
    "oracle_fallback_degrade_confidence_multiplier": 1.08,
    "oracle_fallback_degrade_size_multiplier": 0.45,
    "oracle_fallback_degrade_edge_multiplier_direct": 1.15,
    "oracle_fallback_degrade_confidence_multiplier_direct": 1.04,
    "oracle_fallback_degrade_size_multiplier_direct": 0.70,
    "min_edge_persistence_ms": 1400,
    "max_recent_move_zscore_for_entry": 2.25,
    "max_spread_widening_bps": 28.0,
    "max_orderbook_imbalance": 0.92,
    "reentry_cooldown_seconds_per_market": 15,
    "min_seconds_left_for_entry_5m": 60.0,
    "min_seconds_left_for_entry_15m": 180.0,
    "min_seconds_left_for_entry_1h": 360.0,
    "min_seconds_left_for_entry_4h": 600.0,
    "opening_directional_buy_yes_enabled": False,
    "opening_directional_buy_yes_block_elapsed_pct": 0.10,
    "opening_directional_buy_yes_block_elapsed_pct_5m": 0.45,
    "opening_directional_buy_yes_block_elapsed_pct_15m": 0.25,
    "opening_directional_buy_yes_block_elapsed_pct_1h": 0.10,
    "opening_directional_buy_yes_block_elapsed_pct_4h": 0.05,
    "opening_directional_buy_no_enabled": True,
    "entry_executable_exit_ratio_floor": 0.28,
    "entry_executable_exit_ratio_floor_closing": 0.24,
    "directional_min_entry_price_floor": 0.25,
    "maker_min_entry_price_floor": 0.16,
    "directional_max_entry_price_ceiling": 0.75,
    "maker_max_entry_price_ceiling": 0.70,
    "rapid_take_profit_pct": 10.0,
    "rapid_take_profit_pct_5m": 10.0,
    "rapid_take_profit_pct_15m": 10.0,
    "rapid_take_profit_pct_1h": 12.0,
    "rapid_take_profit_pct_4h": 15.0,
    "rapid_exit_window_minutes": 2.0,
    "rapid_exit_window_minutes_5m": 1.0,
    "rapid_exit_window_minutes_15m": 2.0,
    "rapid_exit_window_minutes_1h": 6.0,
    "rapid_exit_window_minutes_4h": 15.0,
    "rapid_exit_min_increase_pct": 0.0,
    "rapid_exit_breakeven_buffer_pct": 0.0,
    "reverse_on_adverse_velocity_enabled": False,
    "reverse_min_loss_pct": 2.0,
    "reverse_min_adverse_velocity_score": 0.55,
    "reverse_flow_imbalance_threshold": -0.2,
    "reverse_momentum_short_pct_threshold": -0.25,
    "reverse_min_seconds_left": 90.0,
    "reverse_min_price_headroom": 0.18,
    "reverse_min_edge_percent": 0.8,
    "reverse_confidence": 0.62,
    "reverse_size_multiplier": 1.0,
    "reverse_signal_ttl_seconds": 45.0,
    "reverse_cooldown_seconds": 20.0,
    "reverse_max_reentries_per_position": 1,
    "underwater_rebound_exit_enabled": True,
    "underwater_dwell_minutes": 2.5,
    "underwater_dwell_minutes_5m": 0.75,
    "underwater_dwell_minutes_15m": 2.0,
    "underwater_dwell_minutes_1h": 6.0,
    "underwater_dwell_minutes_4h": 18.0,
    "underwater_recovery_ratio_min": 0.35,
    "underwater_recovery_ratio_min_5m": 0.30,
    "underwater_recovery_ratio_min_15m": 0.35,
    "underwater_recovery_ratio_min_1h": 0.40,
    "underwater_recovery_ratio_min_4h": 0.45,
    "underwater_rebound_pct_min": 1.2,
    "underwater_rebound_pct_min_5m": 0.8,
    "underwater_rebound_pct_min_15m": 1.2,
    "underwater_rebound_pct_min_1h": 1.8,
    "underwater_rebound_pct_min_4h": 2.4,
    "underwater_exit_fade_pct": 0.45,
    "underwater_exit_fade_pct_5m": 0.35,
    "underwater_exit_fade_pct_15m": 0.45,
    "underwater_exit_fade_pct_1h": 0.6,
    "underwater_exit_fade_pct_4h": 0.8,
    "underwater_timeout_minutes": 10.0,
    "underwater_timeout_minutes_5m": 2.0,
    "underwater_timeout_minutes_15m": 6.0,
    "underwater_timeout_minutes_1h": 18.0,
    "underwater_timeout_minutes_4h": 45.0,
    "underwater_timeout_loss_pct": 8.0,
    "take_profit_pct": 8.0,
    "stop_loss_pct": 3.0,
    "stop_loss_policy": "always",
    "stop_loss_policy_5m": "always",
    "stop_loss_policy_15m": "always",
    "stop_loss_policy_1h": "always",
    "stop_loss_policy_4h": "always",
    "immediate_stop_loss_pct": 2.0,
    "immediate_stop_loss_pct_5m": 1.5,
    "immediate_stop_loss_pct_15m": 2.0,
    "immediate_stop_loss_pct_1h": 3.0,
    "immediate_stop_loss_pct_4h": 4.0,
    "stop_loss_activation_seconds": 90,
    "stop_loss_activation_seconds_5m": 45.0,
    "stop_loss_activation_seconds_15m": 120.0,
    "stop_loss_activation_seconds_1h": 300.0,
    "stop_loss_activation_seconds_4h": 900.0,
    "trailing_stop_pct": 3.0,
    "trailing_stop_activation_profit_pct": 4.0,
    "trailing_stop_activation_profit_pct_5m": 4.0,
    "trailing_stop_activation_profit_pct_15m": 6.0,
    "trailing_stop_activation_profit_pct_1h": 8.0,
    "trailing_stop_activation_profit_pct_4h": 10.0,
    "min_hold_minutes": 1.0,
    "max_hold_minutes": 60,
    "force_flatten_seconds_left": 120.0,
    "force_flatten_seconds_left_5m": 90.0,
    "force_flatten_seconds_left_15m": 210.0,
    "force_flatten_seconds_left_1h": 480.0,
    "force_flatten_seconds_left_4h": 900.0,
    "force_flatten_max_profit_pct": 3.0,
    "force_flatten_headroom_floor": 1.0,
    "force_flatten_min_loss_pct": 2.0,
    "resolution_risk_flatten_enabled": True,
    "resolution_risk_seconds_left": 180.0,
    "resolution_risk_seconds_left_5m": 105.0,
    "resolution_risk_seconds_left_15m": 240.0,
    "resolution_risk_seconds_left_1h": 540.0,
    "resolution_risk_seconds_left_4h": 1200.0,
    "resolution_risk_max_profit_pct": 6.0,
    "resolution_risk_max_profit_pct_5m": 4.0,
    "resolution_risk_min_loss_pct": 2.0,
    "resolution_risk_min_headroom_ratio": 0.9,
    "resolution_risk_disable_when_take_profit_armed": True,
    "resolve_only": False,
    "close_on_inactive_market": False,
    "preplace_take_profit_exit": False,
    "enforce_min_exit_notional": True,
}


def crypto_highfreq_scope_defaults() -> dict[str, Any]:
    return dict(CRYPTO_HF_SCOPE_DEFAULTS)


def crypto_highfreq_direction_allowed(
    params: Any,
    *,
    regime: Any,
    active_mode: Any,
    direction: Any,
    timeframe: Any = None,
    seconds_left: Optional[float] = None,
) -> tuple[bool, str]:
    cfg = params if isinstance(params, dict) else {}
    defaults = crypto_highfreq_scope_defaults()
    normalized_regime = str(regime or "").strip().lower()
    mode = str(active_mode or "").strip().lower()
    normalized_direction = str(direction or "").strip().lower()
    normalized_timeframe = _normalize_timeframe(timeframe)

    if normalized_direction not in {"buy_yes", "buy_no"}:
        return True, "direction_not_supported"

    if mode == "directional":
        yes_enabled = _coerce_bool(
            cfg.get("opening_directional_buy_yes_enabled"),
            _coerce_bool(defaults.get("opening_directional_buy_yes_enabled"), False),
        )
        no_enabled = _coerce_bool(
            cfg.get("opening_directional_buy_no_enabled"),
            _coerce_bool(defaults.get("opening_directional_buy_no_enabled"), True),
        )
    else:
        return True, "mode_not_gated"

    if normalized_direction == "buy_yes":
        if yes_enabled:
            return True, f"opening_{mode}_buy_yes_enabled={yes_enabled}"
        if mode != "directional":
            if normalized_regime == "opening":
                return False, f"opening_{mode}_buy_yes_enabled={yes_enabled}"
            return True, "regime_not_opening"

        elapsed_ratio: Optional[float] = None
        timeframe_seconds: Optional[float] = None
        if normalized_timeframe in {"5m", "15m", "1h", "4h"}:
            timeframe_seconds = float(
                {
                    "5m": 300.0,
                    "15m": 900.0,
                    "1h": 3600.0,
                    "4h": 14400.0,
                }[normalized_timeframe]
            )
        if timeframe_seconds is not None and seconds_left is not None and seconds_left >= 0.0:
            elapsed_ratio = clamp(1.0 - (float(seconds_left) / timeframe_seconds), 0.0, 1.0)

        gate_ratio_raw = _timeframe_override(cfg, "opening_directional_buy_yes_block_elapsed_pct", normalized_timeframe)
        if gate_ratio_raw is None:
            gate_ratio_raw = cfg.get("opening_directional_buy_yes_block_elapsed_pct")
        if gate_ratio_raw is None:
            gate_ratio_raw = _timeframe_override(
                defaults, "opening_directional_buy_yes_block_elapsed_pct", normalized_timeframe
            )
        if gate_ratio_raw is None:
            gate_ratio_raw = defaults.get("opening_directional_buy_yes_block_elapsed_pct")
        gate_ratio = _coerce_float(gate_ratio_raw, 0.10, 0.0, 1.0)

        if elapsed_ratio is None:
            if normalized_regime == "opening":
                return False, f"opening_{mode}_buy_yes_enabled={yes_enabled}"
            return True, "elapsed_unavailable_regime_not_opening"
        if elapsed_ratio < gate_ratio:
            return (
                False,
                f"opening_{mode}_buy_yes_enabled={yes_enabled} elapsed={elapsed_ratio:.3f} "
                f"< min_elapsed={gate_ratio:.3f} timeframe={normalized_timeframe or 'unknown'}",
            )
        return (
            True,
            f"opening_{mode}_buy_yes_enabled={yes_enabled} elapsed={elapsed_ratio:.3f} "
            f">= min_elapsed={gate_ratio:.3f} timeframe={normalized_timeframe or 'unknown'}",
        )
    if no_enabled:
        return True, f"opening_{mode}_buy_no_enabled={no_enabled}"
    if normalized_regime == "opening":
        return False, f"opening_{mode}_buy_no_enabled={no_enabled}"
    return True, "regime_not_opening"


def crypto_highfreq_should_flatten_resolution_risk(
    params: Any,
    *,
    timeframe: Any = None,
    seconds_left: Optional[float] = None,
    pnl_percent: Optional[float] = None,
    exit_headroom_ratio: Optional[float] = None,
    take_profit_armed: bool = False,
) -> tuple[bool, str]:
    cfg = params if isinstance(params, dict) else {}
    enabled = _coerce_bool(cfg.get("resolution_risk_flatten_enabled"), True)
    if not enabled:
        return False, "disabled"

    if take_profit_armed and _coerce_bool(cfg.get("resolution_risk_disable_when_take_profit_armed"), True):
        return False, "take_profit_armed"

    if seconds_left is None or seconds_left < 0.0:
        return False, "seconds_left_unavailable"

    seconds_budget_raw = _crypto_hf_param_value(cfg, "resolution_risk_seconds_left", timeframe)
    if seconds_budget_raw is None:
        seconds_budget_raw = _crypto_hf_param_value(cfg, "force_flatten_seconds_left", timeframe)
    seconds_budget = _coerce_float(seconds_budget_raw, 120.0, 0.0, 86_400.0)
    if seconds_left > seconds_budget:
        return False, f"seconds_left={seconds_left:.1f} > budget={seconds_budget:.1f}"

    max_profit_raw = _crypto_hf_param_value(cfg, "resolution_risk_max_profit_pct", timeframe)
    max_profit_pct = _coerce_float(max_profit_raw, 6.0, 0.0, 100.0)
    min_loss_raw = _crypto_hf_param_value(cfg, "resolution_risk_min_loss_pct", timeframe)
    min_loss_pct = _coerce_float(min_loss_raw, 2.0, 0.0, 100.0)
    min_headroom_raw = _crypto_hf_param_value(cfg, "resolution_risk_min_headroom_ratio", timeframe)
    min_headroom_ratio = _coerce_float(min_headroom_raw, 0.0, 0.0, 100.0)

    if pnl_percent is not None:
        if pnl_percent > max_profit_pct:
            return False, f"pnl={pnl_percent:.2f}% > max_profit={max_profit_pct:.2f}%"
        if pnl_percent < -abs(min_loss_pct):
            return False, f"pnl={pnl_percent:.2f}% < -max_loss={min_loss_pct:.2f}%"

    if exit_headroom_ratio is not None and exit_headroom_ratio < min_headroom_ratio:
        return False, f"headroom={exit_headroom_ratio:.2f}x < min={min_headroom_ratio:.2f}x"

    pnl_text = f"{pnl_percent:.2f}%" if pnl_percent is not None else "unknown"
    headroom_text = f"{exit_headroom_ratio:.2f}x" if exit_headroom_ratio is not None else "unknown"
    detail = f"seconds_left={seconds_left:.1f}s <= {seconds_budget:.1f}s, pnl={pnl_text}, headroom={headroom_text}"
    return True, detail


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",")]
    return []


def _normalize_scope(value: Any, normalizer: Callable[[Any], str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in _as_list(value):
        normalized = normalizer(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _normalize_regime_scope(value: Any) -> set[str]:
    allowed = set(_REGIMES)
    normalized: set[str] = set()
    for raw in _as_list(value):
        regime = _normalize_regime(raw)
        if regime in allowed:
            normalized.add(regime)
    return normalized


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _parse_datetime_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts <= 0:
            return None
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    numeric = to_float(text)
    if numeric is not None and numeric > 0:
        return _parse_datetime_utc(numeric)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _to_iso_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_component_edge(payload: dict[str, Any], direction: str, mode: str) -> float:
    component_edges = payload.get("component_edges")
    if isinstance(component_edges, dict):
        side_edges = component_edges.get(direction)
        if isinstance(side_edges, dict):
            return max(0.0, to_float(side_edges.get(mode), 0.0))
        return max(0.0, to_float(component_edges.get(mode), 0.0))

    mode_edges = payload.get("mode_edges")
    if isinstance(mode_edges, dict):
        side_mode_edges = mode_edges.get(direction)
        if isinstance(side_mode_edges, dict):
            return max(0.0, to_float(side_mode_edges.get(mode), 0.0))
        return max(0.0, to_float(mode_edges.get(mode), 0.0))

    per_mode_key = {
        "directional": "directional_edge",
        "maker_quote": "maker_quote_edge",
        "convergence": "convergence_edge",
    }.get(mode)
    if per_mode_key is None:
        return 0.0
    return max(0.0, to_float(payload.get(per_mode_key), 0.0))


def _get_net_edge(payload: dict[str, Any], direction: str, fallback: float) -> float:
    net_edges = payload.get("net_edges")
    if not isinstance(net_edges, dict):
        return fallback
    return to_float(net_edges.get(direction), fallback)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return _to_iso_utc(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = _json_safe(item)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _normalize_oracle_source(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "chainlink" in text:
        return "chainlink"
    if "binance_direct" in text:
        return "binance_direct"
    if "binance" in text:
        return "binance"
    return text


def _epoch_ms(value: Any) -> int | None:
    parsed = to_float(value, None)
    if parsed is None or parsed <= 0:
        return None
    if parsed < 1_000_000_000_000:
        parsed *= 1000.0
    return int(parsed)


def _age_ms(*, age_ms: Any = None, age_seconds: Any = None, updated_at_ms: Any = None, now_ms: int) -> float | None:
    updated = _epoch_ms(updated_at_ms)
    if updated is not None:
        return max(0.0, float(now_ms - updated))
    direct_ms = to_float(age_ms, None)
    if direct_ms is not None and direct_ms >= 0.0:
        return float(direct_ms)
    direct_seconds = to_float(age_seconds, None)
    if direct_seconds is not None and direct_seconds >= 0.0:
        return float(direct_seconds) * 1000.0
    return None


def _resolve_oracle_availability(
    *,
    price: float | None,
    price_to_beat: float | None,
    age_ms: float | None,
    updated_at_ms: int | None,
) -> dict[str, Any]:
    has_price = price is not None and price > 0.0
    has_price_to_beat = price_to_beat is not None and price_to_beat > 0.0
    has_time = age_ms is not None or updated_at_ms is not None
    freshness_reasons: list[str] = []
    if not has_price:
        freshness_reasons.append("missing_price")
    if not has_time:
        freshness_reasons.append("missing_timestamp")
    fresh_available = not freshness_reasons

    reasons = list(freshness_reasons)
    if not has_price_to_beat:
        reasons.append("missing_price_to_beat")
    directional_available = not reasons
    if directional_available:
        state = "available"
    elif age_ms is not None:
        state = "age_present_but_unavailable"
    else:
        state = reasons[0]
    if fresh_available:
        freshness_state = "fresh"
    elif age_ms is not None:
        freshness_state = "age_present_but_unavailable"
    else:
        freshness_state = freshness_reasons[0]
    return {
        "available": bool(directional_available),
        "fresh_available": bool(fresh_available),
        "freshness_state": freshness_state,
        "freshness_reasons": freshness_reasons,
        "directional_available": bool(directional_available),
        "directional_state": state,
        "directional_reasons": reasons,
        "availability_state": state,
        "availability_reasons": reasons,
        "has_price": bool(has_price),
        "has_price_to_beat": bool(has_price_to_beat),
        "has_timestamp": bool(has_time),
    }


def _oracle_point(raw: Any, *, source_hint: str | None, now_ms: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    source = _normalize_oracle_source(raw.get("source")) or source_hint
    price = to_float(raw.get("price"), None)
    updated_at_ms = _epoch_ms(raw.get("updated_at_ms"))
    point_age_ms = _age_ms(
        age_ms=raw.get("age_ms"),
        age_seconds=raw.get("age_seconds"),
        updated_at_ms=updated_at_ms,
        now_ms=now_ms,
    )
    if updated_at_ms is None and point_age_ms is not None:
        updated_at_ms = max(0, now_ms - int(point_age_ms))
    return {
        "source": source,
        "price": price,
        "updated_at_ms": updated_at_ms,
        "age_ms": point_age_ms,
    }


def _extract_oracle_status(
    *,
    live_market: dict[str, Any],
    payload: dict[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    live_oracle_status = live_market.get("oracle_status")
    payload_oracle_status = payload.get("oracle_status")
    if isinstance(live_oracle_status, dict):
        candidates.append(("live_market.oracle_status", dict(live_oracle_status)))
    if isinstance(payload_oracle_status, dict):
        candidates.append(("payload.oracle_status", dict(payload_oracle_status)))
    if isinstance(live_market, dict):
        candidates.append(("live_market", dict(live_market)))
    if isinstance(payload, dict):
        candidates.append(("payload", dict(payload)))

    def _candidate_score(candidate: dict[str, Any]) -> tuple[int, int]:
        has_price = to_float(_first_present(candidate.get("oracle_price"), candidate.get("price")), None) is not None
        candidate_price_to_beat = to_float(
            _first_present(
                candidate.get("price_to_beat"),
                (
                    candidate.get("oracle_status", {}).get("price_to_beat")
                    if isinstance(candidate.get("oracle_status"), dict)
                    else None
                ),
                (
                    candidate.get("strategy_context", {}).get("price_to_beat")
                    if isinstance(candidate.get("strategy_context"), dict)
                    else None
                ),
                (
                    candidate.get("strategy_context_json", {}).get("price_to_beat")
                    if isinstance(candidate.get("strategy_context_json"), dict)
                    else None
                ),
            ),
            None,
        )
        has_price_to_beat = candidate_price_to_beat is not None and candidate_price_to_beat > 0.0
        has_time = (
            _epoch_ms(_first_present(candidate.get("oracle_updated_at_ms"), candidate.get("updated_at_ms"))) is not None
            or _age_ms(
                age_ms=_first_present(candidate.get("oracle_age_ms"), candidate.get("age_ms")),
                age_seconds=_first_present(candidate.get("oracle_age_seconds"), candidate.get("age_seconds")),
                updated_at_ms=_first_present(candidate.get("oracle_updated_at_ms"), candidate.get("updated_at_ms")),
                now_ms=now_ms,
            )
            is not None
        )
        by_source = candidate.get("oracle_prices_by_source") or candidate.get("by_source")
        has_by_source = isinstance(by_source, dict) and bool(by_source)
        score = int(has_price) + int(has_price_to_beat) + int(has_time) + int(has_by_source)
        preferred = int(isinstance(candidate.get("oracle_status"), dict))
        return score, preferred

    selected_label = "payload"
    selected = {}
    best_score: tuple[int, int] = (-1, -1)
    for label, candidate in candidates:
        score = _candidate_score(candidate)
        if score > best_score:
            best_score = score
            selected_label = label
            selected = candidate

    by_source_raw = selected.get("oracle_prices_by_source")
    if not isinstance(by_source_raw, dict):
        by_source_raw = selected.get("by_source")
    by_source: dict[str, dict[str, Any]] = {}
    if isinstance(by_source_raw, dict):
        for raw_key, raw_value in by_source_raw.items():
            point = _oracle_point(raw_value, source_hint=_normalize_oracle_source(raw_key), now_ms=now_ms)
            if point is None:
                continue
            source = point.get("source")
            if not source:
                continue
            by_source[source] = point

    selected_source = _normalize_oracle_source(
        _first_present(
            selected.get("oracle_source"),
            selected.get("source"),
        )
    )
    freshest_source: str | None = None
    freshest_age_ms: float | None = None
    freshest_rank = 99
    for source, point in by_source.items():
        if not isinstance(point, dict):
            continue
        point_price = to_float(point.get("price"), None)
        point_age_ms = to_float(point.get("age_ms"), None)
        if point_price is None or point_price <= 0.0 or point_age_ms is None or point_age_ms < 0.0:
            continue
        if source == "binance_direct":
            source_rank = 0
        elif source == "chainlink":
            source_rank = 1
        elif source == "binance":
            source_rank = 2
        else:
            source_rank = 3
        if (
            freshest_age_ms is None
            or point_age_ms < freshest_age_ms
            or (point_age_ms == freshest_age_ms and source_rank < freshest_rank)
        ):
            freshest_source = source
            freshest_age_ms = point_age_ms
            freshest_rank = source_rank

    if freshest_source is not None:
        selected_source = freshest_source
    elif selected_source not in by_source:
        if "chainlink" in by_source:
            selected_source = "chainlink"
        elif "binance_direct" in by_source:
            selected_source = "binance_direct"
        elif "binance" in by_source:
            selected_source = "binance"
        elif by_source:
            selected_source = next(iter(by_source))

    selected_point = by_source.get(selected_source) if selected_source else None
    top_level_price = to_float(_first_present(selected.get("oracle_price"), selected.get("price")), None)
    top_level_updated_at_ms = _epoch_ms(
        _first_present(selected.get("oracle_updated_at_ms"), selected.get("updated_at_ms"))
    )
    top_level_age_ms = _age_ms(
        age_ms=_first_present(selected.get("oracle_age_ms"), selected.get("age_ms")),
        age_seconds=_first_present(selected.get("oracle_age_seconds"), selected.get("age_seconds")),
        updated_at_ms=top_level_updated_at_ms,
        now_ms=now_ms,
    )

    selected_price = to_float(selected_point.get("price"), None) if isinstance(selected_point, dict) else None
    selected_updated_at_ms = (
        _epoch_ms(selected_point.get("updated_at_ms")) if isinstance(selected_point, dict) else None
    )
    selected_age_ms = (
        _age_ms(
            age_ms=selected_point.get("age_ms"),
            updated_at_ms=selected_updated_at_ms,
            now_ms=now_ms,
        )
        if isinstance(selected_point, dict)
        else None
    )
    if selected_price is None:
        selected_price = top_level_price
    if selected_updated_at_ms is None:
        selected_updated_at_ms = top_level_updated_at_ms
    if selected_age_ms is None:
        selected_age_ms = top_level_age_ms
    if selected_source is None:
        selected_source = _normalize_oracle_source(selected.get("source"))

    price_to_beat = to_float(selected.get("price_to_beat"), None)
    if price_to_beat is None:
        for _label, candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_price_to_beat = to_float(
                _first_present(
                    candidate.get("price_to_beat"),
                    (
                        candidate.get("oracle_status", {}).get("price_to_beat")
                        if isinstance(candidate.get("oracle_status"), dict)
                        else None
                    ),
                    (
                        candidate.get("strategy_context", {}).get("price_to_beat")
                        if isinstance(candidate.get("strategy_context"), dict)
                        else None
                    ),
                    (
                        candidate.get("strategy_context_json", {}).get("price_to_beat")
                        if isinstance(candidate.get("strategy_context_json"), dict)
                        else None
                    ),
                ),
                None,
            )
            if candidate_price_to_beat is None:
                markets = candidate.get("markets")
                if isinstance(markets, list):
                    for market in markets:
                        if not isinstance(market, dict):
                            continue
                        candidate_price_to_beat = to_float(market.get("price_to_beat"), None)
                        if candidate_price_to_beat is not None:
                            break
            if candidate_price_to_beat is not None and candidate_price_to_beat > 0.0:
                price_to_beat = candidate_price_to_beat
                break
    availability = _resolve_oracle_availability(
        price=selected_price,
        price_to_beat=price_to_beat,
        age_ms=selected_age_ms,
        updated_at_ms=selected_updated_at_ms,
    )

    return {
        "price": selected_price,
        "price_to_beat": price_to_beat,
        "source": selected_source,
        "updated_at_ms": selected_updated_at_ms,
        "age_ms": selected_age_ms,
        "fresh_available": availability["fresh_available"],
        "freshness_state": availability["freshness_state"],
        "freshness_reasons": availability["freshness_reasons"],
        "directional_available": availability["directional_available"],
        "directional_state": availability["directional_state"],
        "directional_reasons": availability["directional_reasons"],
        "availability_state": availability["availability_state"],
        "availability_reasons": availability["availability_reasons"],
        "available": availability["available"],
        "has_price": availability["has_price"],
        "has_price_to_beat": availability["has_price_to_beat"],
        "has_timestamp": availability["has_timestamp"],
        "context": selected_label,
        "by_source": by_source,
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Question / slug patterns used to identify BTC/ETH high-frequency markets
_ASSET_PATTERNS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["ripple", "xrp"],
}

_TIMEFRAME_PATTERNS: dict[str, list[str]] = {
    "5min": [
        "updown-5m",
        "5m-",
        "5m",
        "5 min",
        "5-minute",
    ],
    "15min": [
        "updown-15m",  # actual Polymarket slug pattern
        "updown-15m-",  # with trailing timestamp
        "15m-",  # short form in slugs (e.g. "btc…15m-17707…")
        "15 min",
        "15-min",
        "15min",
        "15m",  # bare short form
        "fifteen min",
        "15 minute",
        "15-minute",
        "15 minutes",
        "15-minutes",
        "quarter hour",
        "quarter-hour",
    ],
    "1hr": [
        "updown-1h",  # actual Polymarket slug pattern
        "updown-1h-",  # with trailing timestamp
        "1 hour",
        "1-hour",
        "1hr",
        "1h-",  # short form in slugs
        "1h",  # bare short form
        "one hour",
        "60 min",
        "60-min",
        "60m",  # short form
        "60 minute",
        "60-minute",
        "60 minutes",
        "60-minutes",
        "hourly",
        "next hour",
    ],
    "4hr": [
        "updown-4h",  # actual Polymarket slug pattern
        "updown-4h-",  # with trailing timestamp
        "4 hour",
        "4-hour",
        "4hr",
        "4h-",  # short form in slugs
        "4h",  # bare short form
        "four hour",
        "four-hour",
        "240 min",
        "240-min",
        "240m",  # short form
        "240 minute",
        "240-minute",
        "240 minutes",
        "240-minutes",
    ],
}

_DIRECTION_KEYWORDS: list[str] = [
    "up or down",
    "higher or lower",
    "go up",
    "go down",
    "above or below",
    "increase or decrease",
    "up",
    "down",
    "higher",
    "lower",
    "price",
    "beat",
    "price to beat",
]

# Slug regex: matches slugs where asset and timeframe may be separated by
# other words.  Allows "bitcoin-15-minute-up-or-down", "btc-price-15min",
# "ethereum-1-hour-up-down", etc.
_SLUG_REGEX = re.compile(
    r"(btc|eth|sol|xrp|bitcoin|ethereum|solana|ripple)"
    r".*?"  # allow intervening words (non-greedy)
    r"(5[\s_-]?m(?:in(?:ute)?s?)?"
    r"|15[\s_-]?m(?:in(?:ute)?s?)?"  # "15m", "15min", "15-minute", …
    r"|1[\s_-]?h(?:(?:ou)?r)?"  # "1h", "1hr", "1hour", "1-h", …
    r"|4[\s_-]?h(?:(?:ou)?r)?"  # "4h", "4hr", "4hour", "4-h", …
    r"|240[\s_-]?m(?:in(?:ute)?s?)?"  # "240m", "240min", "240 minutes", etc.
    r"|60[\s_-]?m(?:in(?:ute)?s?)?"
    r"|quarter[\s_-]?hour|hourly)",
    re.IGNORECASE,
)


# Polymarket crypto series definitions.
# Each series has a unique ID on the Gamma API that returns all active events
# in the series.  Querying /events?series_id=X&active=true&closed=false
# reliably returns the currently-live and upcoming 15-minute markets.
# Series IDs are configurable via the Settings UI (persisted in DB).
def _get_crypto_series() -> list[tuple[str, str, str]]:
    """Return crypto series configs, reading IDs from the live config singleton."""
    return [
        # (series_id, asset, timeframe)
        (_cfg.BTC_ETH_HF_SERIES_BTC_15M, "BTC", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_15M, "ETH", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_15M, "SOL", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_15M, "XRP", "15min"),
        (_cfg.BTC_ETH_HF_SERIES_BTC_5M, "BTC", "5min"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_5M, "ETH", "5min"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_5M, "SOL", "5min"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_5M, "XRP", "5min"),
        (_cfg.BTC_ETH_HF_SERIES_BTC_1H, "BTC", "1hr"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_1H, "ETH", "1hr"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_1H, "SOL", "1hr"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_1H, "XRP", "1hr"),
        (_cfg.BTC_ETH_HF_SERIES_BTC_4H, "BTC", "4hr"),
        (_cfg.BTC_ETH_HF_SERIES_ETH_4H, "ETH", "4hr"),
        (_cfg.BTC_ETH_HF_SERIES_SOL_4H, "SOL", "4hr"),
        (_cfg.BTC_ETH_HF_SERIES_XRP_4H, "XRP", "4hr"),
    ]


# ---------------------------------------------------------------------------
# Fee curve — official Polymarket taker fee for 15-minute crypto markets
# ---------------------------------------------------------------------------


def polymarket_fee_curve(price: float) -> float:
    """Compute the taker fee per share at a given price.

    Official formula from Polymarket docs:
        fee = price * 0.25 * (price * (1 - price))^2

    At $0.50: fee = 0.50 * 0.25 * (0.50 * 0.50)^2 = $0.0078 → 1.56%
    At $0.10: fee = 0.10 * 0.25 * (0.10 * 0.90)^2 = $0.0002 → 0.20%
    At $0.90: fee = 0.90 * 0.25 * (0.90 * 0.10)^2 = $0.0018 → 0.20%
    """
    p = max(0.0, min(1.0, price))
    return p * 0.25 * (p * (1.0 - p)) ** 2


def polymarket_fee_pct(price: float) -> float:
    """Fee as a fraction of the share price (0.0 – 0.0156)."""
    if price <= 0:
        return 0.0
    return polymarket_fee_curve(price) / price


# ---------------------------------------------------------------------------
# Sub-strategy scoring constants
# ---------------------------------------------------------------------------

# -- Directional Edge scoring (oracle-based) --
_DIRECTIONAL_EARLY_PHASE_MINUTES = 10.0  # Remaining minutes for early/mid boundary
_DIRECTIONAL_EARLY_MIN_EDGE = 0.08  # Required edge in early phase
_DIRECTIONAL_EARLY_SCORE_MULT = 1.0
_DIRECTIONAL_MID_PHASE_MINUTES = 5.0  # Remaining minutes for mid/late boundary
_DIRECTIONAL_MID_MIN_EDGE = 0.05  # Required edge in mid phase
_DIRECTIONAL_MID_SCORE_MULT = 1.5
_DIRECTIONAL_LATE_MIN_EDGE = 0.03  # Required edge in late phase
_DIRECTIONAL_LATE_SCORE_MULT = 2.0
_DIRECTIONAL_EDGE_SCORE_SCALE = 500.0  # Score scale for edge magnitude
_DIRECTIONAL_MAX_SCORE = 80.0  # Cap on directional score
_DIRECTIONAL_LATE_PHASE_BONUS = 20.0  # Extra score in late phase
_DIRECTIONAL_ORACLE_PROB_MIN = 0.03  # Min/max model probability clamp (sigmoid)
_DIRECTIONAL_ORACLE_PROB_MAX = 0.97
_DIRECTIONAL_BASE_SCALE = 0.50  # Base sigmoid scale (diff_pct / scale)
_DIRECTIONAL_MIN_SCALE = 0.08  # Min sigmoid scale (late in window)

# -- Maker Quote (two-sided market-making) scoring --
_MAKER_QUOTE_MIN_SPREAD = 0.01  # Minimum bid-ask spread ($0.01 — makers get 0% fee)
_MAKER_QUOTE_MIN_LIQUIDITY = 200.0  # Minimum market liquidity (thin books are BETTER)
_MAKER_QUOTE_MIN_SECONDS_LEFT = 30.0  # Can quote until near expiry
_MAKER_QUOTE_TICK_SIZE = 0.01  # 1 tick = $0.01
_MAKER_QUOTE_BASE_SCORE = 25.0  # PRIMARY strategy — highest base score
_MAKER_QUOTE_SPREAD_SCORE_SCALE = 400.0  # Score scale for spread width
_MAKER_QUOTE_MAX_SCORE = 90.0  # Highest possible score
_MAKER_QUOTE_THIN_BOOK_USD = 1000.0  # Below this, higher fill probability
_MAKER_QUOTE_THIN_BOOK_BONUS = 15.0
_MAKER_QUOTE_MIN_SIZE_USD = 5.0  # Minimum position size per side
_MAKER_QUOTE_MAX_SIZE_USD = 25.0  # Maximum position size per side
_MAKER_QUOTE_RESOLUTION_RISK_SECONDS = 45.0  # Extra risk near resolution
_MAKER_QUOTE_SKEW_MAX = 0.03  # Max skew per side from oracle signal ($0.03)

# -- Convergence (near-expiry) scoring --
_CONVERGENCE_MIN_SECONDS_LEFT = 5.0
_CONVERGENCE_MAX_SECONDS_LEFT = 45.0
_CONVERGENCE_MIN_ORACLE_DIFF_PCT = 0.30
_CONVERGENCE_BASE_SCORE = 20.0
_CONVERGENCE_MAX_SCORE = 85.0
_CONVERGENCE_MIN_PRICE = 0.85
_CONVERGENCE_MAX_ENTRY_PRICE = 0.95

# Price history defaults
_DEFAULT_HISTORY_WINDOW_SEC = 300  # 5 minutes for 15-min markets
_1HR_HISTORY_WINDOW_SEC = 600  # 10 minutes for 1-hr markets
_4HR_HISTORY_WINDOW_SEC = 1800  # 30 minutes for 4-hr markets
_MAX_HISTORY_ENTRIES = 200  # Maximum price snapshots per market


# ---------------------------------------------------------------------------
# Gamma API crypto market fetcher
# ---------------------------------------------------------------------------


class _CryptoMarketFetcher:
    """Sync HTTP fetcher that queries Polymarket's Gamma API for crypto markets
    using series_id-based discovery (the same approach used by
    PolymarketBTC15mAssistant and other production bots).

    Each crypto asset/timeframe has a stable ``series_id`` on the Gamma API.
    Querying ``GET /events?series_id=X&active=true&closed=false`` reliably
    returns the currently-live and upcoming 15-minute (or hourly) markets
    with correct ``endDate`` values, real-time ``bestBid``/``bestAsk``
    pricing, CLOB token IDs, and liquidity data.

    Results are cached for ``ttl_seconds`` to avoid hammering the API.
    """

    def __init__(self, gamma_url: str = "", ttl_seconds: int = 15):
        self._gamma_url = gamma_url or _cfg.GAMMA_API_URL
        self._ttl = ttl_seconds
        self._markets: list[Market] = []
        self._last_fetch: float = 0.0

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self._last_fetch) > self._ttl

    def get_markets(self) -> list[Market]:
        """Return cached crypto markets, refreshing if stale."""
        if self.is_stale:
            fetched = self._fetch()
            if fetched is not None:
                self._markets = fetched
                self._last_fetch = time.monotonic()
                # Subscribe new market tokens to the WS feed for real-time prices
                self._subscribe_tokens_to_ws(fetched)
        return self._markets

    @staticmethod
    def _subscribe_tokens_to_ws(markets: list[Market]) -> None:
        """Fire-and-forget: subscribe crypto market CLOB tokens to the
        WebSocket price feed so we get real-time bid/ask updates instead
        of relying on stale HTTP polling."""
        import asyncio

        token_ids = []
        for m in markets:
            token_ids.extend(t for t in m.clob_token_ids if len(t) > 20)
        if not token_ids:
            return

        try:
            from services.ws_feeds import get_feed_manager

            feed_mgr = get_feed_manager()
            if not feed_mgr._started:
                return

            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(feed_mgr.polymarket_feed.subscribe(token_ids=token_ids))
            else:
                loop.run_until_complete(feed_mgr.polymarket_feed.subscribe(token_ids=token_ids))
            logger.debug(
                "BtcEthHighFreq: subscribed %d crypto tokens to WS feed",
                len(token_ids),
            )
        except Exception as e:
            logger.debug("BtcEthHighFreq: WS subscription failed (non-critical): %s", e)

    @staticmethod
    def _is_currently_live(event: dict, now_ms: float) -> bool:
        """Check if an event's market window is currently live.

        A 15-minute market is live when:
          startTime <= now < endDate
        The event-level ``startTime`` is when the 15-min window opens;
        ``endDate`` is when it resolves.
        """
        start_str = event.get("startTime") or event.get("startDate")
        end_str = event.get("endDate")
        if not end_str:
            return False

        try:
            end_ms = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp() * 1000
        except (ValueError, AttributeError):
            return False

        if now_ms >= end_ms:
            return False  # Already resolved

        if start_str:
            try:
                start_ms = datetime.fromisoformat(start_str.replace("Z", "+00:00")).timestamp() * 1000
                if now_ms < start_ms:
                    return False  # Not started yet
            except (ValueError, AttributeError):
                pass

        return True

    @staticmethod
    def _pick_live_and_upcoming(events: list[dict], max_upcoming: int = 2) -> list[dict]:
        """From a list of events, return the currently-live one plus next upcoming.

        This mirrors the reference bot's ``pickLatestLiveMarket`` logic but
        returns multiple events so we can show upcoming opportunities too.
        """
        now_ms = time.time() * 1000
        live: list[dict] = []
        upcoming: list[dict] = []

        for evt in events:
            if evt.get("closed"):
                continue
            start_str = evt.get("startTime") or evt.get("startDate")
            end_str = evt.get("endDate")
            if not end_str:
                continue
            try:
                end_ms = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp() * 1000
            except (ValueError, AttributeError):
                continue
            if end_ms <= now_ms:
                continue  # Already resolved

            start_ms = None
            if start_str:
                try:
                    start_ms = datetime.fromisoformat(start_str.replace("Z", "+00:00")).timestamp() * 1000
                except (ValueError, AttributeError):
                    pass

            if start_ms is not None and start_ms <= now_ms:
                live.append((end_ms, evt))
            else:
                upcoming.append((end_ms, evt))

        # Sort by end time (soonest first)
        live.sort(key=lambda x: x[0])
        upcoming.sort(key=lambda x: x[0])

        result = [e for _, e in live]
        result.extend(e for _, e in upcoming[:max_upcoming])
        return result

    def _fetch(self) -> list[Market]:
        """Fetch live crypto markets from Gamma API using series_id.

        For each crypto series (BTC 15m, ETH 15m, etc.), queries the
        events endpoint to get active events, then picks the currently-live
        and next-upcoming markets.
        """
        import httpx

        all_markets: list[Market] = []
        seen_ids: set[str] = set()

        def _market_id(mkt: dict) -> str:
            return str(mkt.get("conditionId") or mkt.get("condition_id") or mkt.get("id", ""))

        try:
            series = _get_crypto_series()
            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with httpx.Client(timeout=10.0) as client:
                for series_id, asset, timeframe in series:
                    try:
                        resp = client.get(
                            f"{self._gamma_url}/events",
                            params={
                                "series_id": series_id,
                                "active": "true",
                                "closed": "false",
                                # Exclude stale unresolved history and walk forward
                                # from now for live/nearest-upcoming selection.
                                "end_date_min": now_iso,
                                "order": "endDate",
                                "ascending": "true",
                                "limit": 10,
                            },
                        )
                        if resp.status_code != 200:
                            logger.debug(
                                "BtcEthHighFreq: Gamma series_id=%s returned %s",
                                series_id,
                                resp.status_code,
                            )
                            continue

                        events = resp.json()
                        if not isinstance(events, list):
                            continue

                        # Pick live + upcoming events
                        selected = self._pick_live_and_upcoming(events)

                        for event_data in selected:
                            for mkt_data in event_data.get("markets", []):
                                mid = _market_id(mkt_data)
                                if mid and mid not in seen_ids:
                                    try:
                                        m = Market.from_gamma_response(mkt_data)
                                        all_markets.append(m)
                                        seen_ids.add(mid)
                                    except Exception as e:
                                        logger.debug(
                                            "BtcEthHighFreq: failed to parse market %s: %s",
                                            mid,
                                            e,
                                        )

                        time.sleep(0.05)  # Rate limit between series
                    except Exception as e:
                        logger.debug(
                            "BtcEthHighFreq: series_id=%s fetch failed: %s",
                            series_id,
                            e,
                        )

        except Exception as exc:
            logger.warning(
                "Crypto market fetch failed: %s",
                str(exc),
                exc_info=True,
            )

        if all_markets:
            logger.info(
                "BtcEthHighFreq: fetched %d live crypto markets via Gamma series API (%s)",
                len(all_markets),
                ", ".join(f"{a} {tf}" for _, a, tf in series),
            )
        else:
            logger.debug(
                "BtcEthHighFreq: no live crypto markets found across %d series",
                len(series),
            )
        return all_markets


# Module-level crypto market fetcher (lazy-initialized)
_crypto_fetcher: Optional[_CryptoMarketFetcher] = None


def _get_crypto_fetcher() -> _CryptoMarketFetcher:
    """Get or create the singleton crypto market fetcher."""
    global _crypto_fetcher
    if _crypto_fetcher is None:
        _crypto_fetcher = _CryptoMarketFetcher()
    return _crypto_fetcher


# ---------------------------------------------------------------------------
# Sub-strategy enum
# ---------------------------------------------------------------------------


class SubStrategy(str, Enum):
    MAKER_QUOTE = "maker_quote"           # Two-sided maker quoting (Layer 1 - PRIMARY)
    DIRECTIONAL_EDGE = "directional_edge"  # Oracle-gated directional (Layer 2 - skew amplifier)
    CONVERGENCE = "convergence"            # Near-expiry convergence (Layer 3)


_SUB_STRATEGY_ALIASES: dict[str, SubStrategy] = {
    "maker_quote": SubStrategy.MAKER_QUOTE,
    "makerquote": SubStrategy.MAKER_QUOTE,
    "maker": SubStrategy.MAKER_QUOTE,
    "market_make": SubStrategy.MAKER_QUOTE,
    "passive_quote": SubStrategy.MAKER_QUOTE,
    "passivequote": SubStrategy.MAKER_QUOTE,
    "passive": SubStrategy.MAKER_QUOTE,
    "directional_edge": SubStrategy.DIRECTIONAL_EDGE,
    "directional": SubStrategy.DIRECTIONAL_EDGE,
    "edge_directional": SubStrategy.DIRECTIONAL_EDGE,
    "convergence": SubStrategy.CONVERGENCE,
    "converge": SubStrategy.CONVERGENCE,
    "near_expiry": SubStrategy.CONVERGENCE,
}


def _normalize_sub_strategy(value: Any) -> Optional[SubStrategy]:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return None
    return _SUB_STRATEGY_ALIASES.get(token)


def _resolve_enabled_sub_strategies(config: Any) -> set[SubStrategy]:
    cfg = config if isinstance(config, dict) else {}
    raw = _first_present(
        cfg.get("enabled_sub_strategies"),
        cfg.get("sub_strategy_allowlist"),
        cfg.get("sub_strategies"),
    )
    if raw is None:
        return set(SubStrategy)

    enabled: set[SubStrategy] = set()
    for item in _as_list(raw):
        normalized = _normalize_sub_strategy(item)
        if normalized is not None:
            enabled.add(normalized)
    return enabled


def _resolve_enabled_active_modes(config: Any) -> set[str]:
    enabled_sub_strategies = _resolve_enabled_sub_strategies(config)
    if not enabled_sub_strategies:
        return {"directional", "maker_quote", "convergence"}

    enabled_modes: set[str] = set()
    if SubStrategy.DIRECTIONAL_EDGE in enabled_sub_strategies:
        enabled_modes.add("directional")
    if SubStrategy.MAKER_QUOTE in enabled_sub_strategies:
        enabled_modes.add("maker_quote")
    if SubStrategy.CONVERGENCE in enabled_sub_strategies:
        enabled_modes.add("convergence")

    if not enabled_modes:
        return {"directional", "maker_quote", "convergence"}
    return enabled_modes


# ---------------------------------------------------------------------------
# Price history tracker
# ---------------------------------------------------------------------------


@dataclass
class PriceSnapshot:
    """A single price observation at a point in time."""

    timestamp: float  # time.monotonic()
    yes_price: float
    no_price: float


@dataclass
class MarketPriceHistory:
    """Rolling window of price snapshots for a single market."""

    window_seconds: float = _DEFAULT_HISTORY_WINDOW_SEC
    snapshots: deque[PriceSnapshot] = field(default_factory=deque)

    def record(self, yes_price: float, no_price: float) -> None:
        """Append a snapshot and evict stale entries."""
        now = time.monotonic()
        self.snapshots.append(
            PriceSnapshot(
                timestamp=now,
                yes_price=yes_price,
                no_price=no_price,
            )
        )
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.snapshots and self.snapshots[0].timestamp < cutoff:
            self.snapshots.popleft()

    @property
    def has_data(self) -> bool:
        return len(self.snapshots) >= 2

    def max_drop_yes(self) -> float:
        """Return the largest drop (positive value) in YES price over the window."""
        if not self.has_data:
            return 0.0
        peak = max(s.yes_price for s in self.snapshots)
        current = self.snapshots[-1].yes_price
        return max(peak - current, 0.0)

    def max_drop_no(self) -> float:
        """Return the largest drop (positive value) in NO price over the window."""
        if not self.has_data:
            return 0.0
        peak = max(s.no_price for s in self.snapshots)
        current = self.snapshots[-1].no_price
        return max(peak - current, 0.0)

    def recent_volatility(self) -> float:
        """Simple volatility proxy: max price range over the window (YES side)."""
        if not self.has_data:
            return 0.0
        prices = [s.yes_price for s in self.snapshots]
        return max(prices) - min(prices)


# ---------------------------------------------------------------------------
# Candidate detection helper
# ---------------------------------------------------------------------------


@dataclass
class HighFreqCandidate:
    """A market identified as a BTC/ETH high-frequency binary market."""

    market: Market
    asset: str  # "BTC", "ETH", "SOL", or "XRP"
    timeframe: str  # "5min", "15min", "1hr", or "4hr"
    yes_price: float
    no_price: float
    oracle_price: Optional[float] = None
    price_to_beat: Optional[float] = None


# ---------------------------------------------------------------------------
# Sub-strategy scoring
# ---------------------------------------------------------------------------


@dataclass
class SubStrategyScore:
    """Score and metadata for a candidate sub-strategy."""

    strategy: SubStrategy
    score: float  # Higher is better (0-100 scale)
    reason: str
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------


class BtcEthHighFreqStrategy(BaseStrategy):
    """
    High-frequency market-making strategy for BTC/ETH binary markets.

    Three-layer architecture:
      Layer 1: Maker Quote   -- two-sided post_only limit orders (PRIMARY)
      Layer 2: Directional   -- oracle-gated directional with fee-aware entry
      Layer 3: Convergence   -- near-expiry maker order on predicted winner

    Designed for Polymarket's 5m/15m/1h/4h BTC/ETH/SOL/XRP up-or-down markets.
    """

    strategy_type = "btc_eth_highfreq"
    name = "BTC/ETH High-Frequency"
    description = "Dynamic high-frequency arbitrage on BTC/ETH 15-min and 1-hr binary markets"
    mispricing_type = "within_market"
    source_key = "crypto"
    worker_affinity = "crypto"
    market_categories = ["crypto"]
    requires_historical_prices = True
    subscriptions = ["crypto_update"]
    supports_entry_take_profit_exit = True
    default_open_order_timeout_seconds = 20.0

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=1.0,
        max_resolution_months=0.1,
    )

    default_config = crypto_highfreq_scope_defaults()

    def __init__(self) -> None:
        super().__init__()
        # 15-minute crypto markets have taker-only fees using a price-curve:
        #   fee_per_share = price * 0.25 * (price * (1 - price))^2
        # At 50% (where up/down markets sit), this is ~1.56%.
        # We set self.fee to the midpoint estimate; the scoring methods
        # use polymarket_fee_curve() for price-specific calculations.
        # See: https://docs.polymarket.com/polymarket-learn/trading/maker-rebates-program
        self.fee = _cfg.BTC_ETH_HF_FEE_ESTIMATE  # default ~1.56% at 50% probability
        # Per-market price history keyed by market ID
        self._price_histories: dict[str, MarketPriceHistory] = {}
        # Runtime anti-churn controls used by evaluate().
        self._edge_first_seen_ms: dict[str, int] = {}
        self._last_selected_at_ms_by_market: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        """Detect arbitrage opportunities across BTC/ETH high-freq markets.

        1. Filter markets to find BTC/ETH high-freq candidates.
        2. Update price history for each candidate.
        3. Run the dynamic strategy selector on each candidate.
        4. Return all detected opportunities.
        """
        if not _cfg.BTC_ETH_HF_ENABLED:
            return []

        opportunities: list[Opportunity] = []

        candidates = self._find_candidates(markets, prices)
        if not candidates:
            logger.debug("BtcEthHighFreq: no BTC/ETH high-freq candidates found")
            return opportunities

        enabled_sub_strategies = _resolve_enabled_sub_strategies(getattr(self, "config", {}))
        if not enabled_sub_strategies:
            logger.info("BtcEthHighFreq: no sub-strategies enabled in config; skipping detection")
            return opportunities

        logger.info(f"BtcEthHighFreq: found {len(candidates)} candidate market(s) — evaluating sub-strategies")

        for candidate in candidates:
            # Update price history
            self._update_price_history(candidate)

            # Dynamic strategy selection
            selected, all_scores = self._select_sub_strategy(candidate, enabled_sub_strategies)
            if selected is None:
                reasons = " | ".join(f"{s.strategy.value}: {s.reason}" for s in all_scores)
                logger.debug(
                    f"BtcEthHighFreq: no viable sub-strategy for market "
                    f"{candidate.market.id} ({candidate.asset} {candidate.timeframe}, "
                    f"yes={candidate.yes_price:.3f} no={candidate.no_price:.3f} "
                    f"liq=${candidate.market.liquidity:.0f}) — {reasons}"
                )
                continue

            scores_str = ", ".join(f"{s.strategy.value}={s.score:.1f}" for s in all_scores)
            logger.info(
                f"BtcEthHighFreq: market {candidate.market.id} "
                f"({candidate.asset} {candidate.timeframe}) — "
                f"selected {selected.strategy.value} (score={selected.score:.1f}). "
                f"All scores: {scores_str}"
            )

            # Generate opportunity from the selected sub-strategy
            opp = self._generate_opportunity(candidate, selected)
            if opp is not None:
                opportunities.append(opp)
                logger.info(
                    f"BtcEthHighFreq: opportunity detected — {opp.title} | "
                    f"ROI {opp.roi_percent:.2f}% | sub-strategy={selected.strategy.value} | "
                    f"market={candidate.market.id}"
                )
            else:
                logger.debug(
                    f"BtcEthHighFreq: create_opportunity rejected market "
                    f"{candidate.market.id} ({candidate.asset} {candidate.timeframe}, "
                    f"sub={selected.strategy.value}, score={selected.score:.1f}) — "
                    f"hard filters in base strategy blocked it "
                    f"(yes={candidate.yes_price:.3f} no={candidate.no_price:.3f} "
                    f"liq=${candidate.market.liquidity:.0f})"
                )

        logger.info(f"BtcEthHighFreq: scan complete — {len(opportunities)} opportunity(ies) found")
        return opportunities

    # ------------------------------------------------------------------
    # Market identification
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[HighFreqCandidate]:
        """Filter the full market list to BTC/ETH high-freq binary markets.

        Also queries the Gamma API directly for crypto markets by tag/slug
        to catch BTC/ETH 15-min markets that may not be in the top 500.
        """
        candidates: list[HighFreqCandidate] = []
        seen_ids: set[str] = set()

        # Combine scanner markets with directly-fetched crypto markets
        all_markets = list(markets)
        try:
            fetcher = _get_crypto_fetcher()
            extra = fetcher.get_markets()
            for m in extra:
                if m.id not in seen_ids:
                    all_markets.append(m)
            # Also mark scanner-provided markets so we don't double-count
            for m in markets:
                seen_ids.add(m.id)
        except Exception:
            pass  # Non-fatal: fall back to scanner markets only

        logger.debug(
            f"BtcEthHighFreq: scanning {len(all_markets)} markets "
            f"({len(markets)} from scanner, {len(all_markets) - len(markets)} from Gamma)"
        )

        asset_hit_no_tf = 0  # track markets that pass asset but fail timeframe
        for market in all_markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) != 2:
                continue
            if market.id in seen_ids and market not in markets:
                continue  # Already processed from scanner list
            seen_ids.add(market.id)

            asset = self._detect_asset(market)
            if asset is None:
                continue

            timeframe = self._detect_timeframe(market)
            if timeframe is None:
                asset_hit_no_tf += 1
                if asset_hit_no_tf <= 5:
                    logger.debug(
                        f"BtcEthHighFreq: asset={asset} but no timeframe — "
                        f"slug={market.slug} question={market.question[:80]}"
                    )
                continue

            # Resolve live prices
            yes_price, no_price = self._resolve_prices(market, prices)

            candidates.append(
                HighFreqCandidate(
                    market=market,
                    asset=asset,
                    timeframe=timeframe,
                    yes_price=yes_price,
                    no_price=no_price,
                )
            )

        return candidates

    @staticmethod
    def _detect_asset(market: Market) -> Optional[str]:
        """Return 'BTC' or 'ETH' if the market targets one of those assets."""
        text = f"{market.question} {market.slug}".lower()
        for asset, keywords in _ASSET_PATTERNS.items():
            if any(kw in text for kw in keywords):
                return asset
        return None

    @staticmethod
    def _detect_timeframe(market: Market) -> Optional[str]:
        """Return a supported timeframe if a match is detected."""
        text = f"{market.question} {market.slug}".lower()

        # Try slug regex first — now allows words between asset and timeframe
        slug_text = f"{market.slug} {market.question}".lower()
        slug_match = _SLUG_REGEX.search(slug_text)
        if slug_match:
            raw_tf = slug_match.group(2).lower().replace("-", "").replace("_", "")
            # Check 15m before 5m (15m contains "5m" substring)
            if "15" in raw_tf or "quarter" in raw_tf:
                return "15min"
            if "4h" in raw_tf or "240" in raw_tf:
                return "4hr"
            if "5m" in raw_tf or (raw_tf.startswith("5") and "15" not in raw_tf):
                return "5min"
            if "1h" in raw_tf or "60" in raw_tf or "hourly" in raw_tf:
                return "1hr"

        # Fallback: question-text keyword matching (broadened patterns)
        # Check 15m before 5m (15m contains "5m" substring)
        for tf_key in ("15min", "5min", "1hr", "4hr"):
            patterns = _TIMEFRAME_PATTERNS.get(tf_key, [])
            if any(p in text for p in patterns):
                return tf_key

        return None

    @staticmethod
    def _is_direction_market(market: Market) -> bool:
        """Check if the market is a directional up/down style question."""
        text = market.question.lower()
        return any(kw in text for kw in _DIRECTION_KEYWORDS)

    @staticmethod
    def _resolve_prices(
        market: Market,
        prices: dict[str, dict],
    ) -> tuple[float, float]:
        """Return (yes_price, no_price) using live CLOB prices when available."""
        yes_price = market.yes_price
        no_price = market.no_price

        if market.clob_token_ids:
            if len(market.clob_token_ids) > 0:
                token = market.clob_token_ids[0]
                if token in prices:
                    yes_price = prices[token].get("mid", yes_price)
            if len(market.clob_token_ids) > 1:
                token = market.clob_token_ids[1]
                if token in prices:
                    no_price = prices[token].get("mid", no_price)

        return yes_price, no_price

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def _update_price_history(self, candidate: HighFreqCandidate) -> None:
        """Record the latest prices into the rolling window for this market."""
        mid = candidate.market.id
        if mid not in self._price_histories:
            if candidate.timeframe == "4hr":
                window = _4HR_HISTORY_WINDOW_SEC
            elif candidate.timeframe == "1hr":
                window = _1HR_HISTORY_WINDOW_SEC
            elif candidate.timeframe == "5min":
                window = 120  # 2 min for 5-min markets
            else:
                window = _DEFAULT_HISTORY_WINDOW_SEC
            self._price_histories[mid] = MarketPriceHistory(window_seconds=window)

        self._price_histories[mid].record(
            candidate.yes_price,
            candidate.no_price,
        )

    def _get_history(self, market_id: str) -> Optional[MarketPriceHistory]:
        return self._price_histories.get(market_id)

    # ------------------------------------------------------------------
    # Dynamic strategy selector
    # ------------------------------------------------------------------

    def _select_sub_strategy(
        self,
        candidate: HighFreqCandidate,
        enabled_sub_strategies: set[SubStrategy],
    ) -> tuple[Optional[SubStrategyScore], list[SubStrategyScore]]:
        """Score all sub-strategies and return the best enabled one.

        Returns (best_score_or_None, all_scores).
        A sub-strategy with score <= 0 is considered non-viable.
        """
        candidate_scores: list[SubStrategyScore] = [
            self._score_maker_quote(candidate),
            self._score_directional_edge(candidate),
            self._score_convergence(candidate),
        ]
        scores: list[SubStrategyScore] = []
        for score in candidate_scores:
            if score.strategy in enabled_sub_strategies:
                scores.append(score)
                continue
            scores.append(
                SubStrategyScore(
                    strategy=score.strategy,
                    score=0.0,
                    reason="disabled_by_config",
                    params=score.params,
                )
            )

        # Sort descending by score
        scores.sort(key=lambda s: s.score, reverse=True)

        best = scores[0] if scores[0].score > 0 else None
        return best, scores

    # -- Sub-strategy: Directional Edge scoring (Layer 2) --

    def _score_directional_edge(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score directional edge using oracle price vs price-to-beat.

        Compares the real-time Chainlink oracle price against the oracle
        price recorded at market open (the "price to beat") to compute
        a sigmoid model probability of Up vs Down.  When this diverges
        from the market-implied probability, there is a directional edge.

        This replaces the old momentum-chasing approach that used
        Polymarket order-book trend, which was the root cause of
        money-losing trades (trading the market's own noise).
        """
        try:
            from services.chainlink_feed import get_chainlink_feed
        except ImportError:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason="Chainlink feed not available",
            )

        feed = get_chainlink_feed()
        oracle = feed.get_price(c.asset)
        if not oracle or not oracle.price:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=f"No oracle price for {c.asset}",
            )

        # Check oracle freshness (must be <60 seconds old)
        age_ms = (time.time() * 1000) - (oracle.updated_at_ms or 0)
        if age_ms > 60_000:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=f"Oracle price stale ({age_ms / 1000:.0f}s old)",
            )

        oracle_price = oracle.price

        # --- Derive price_to_beat from Chainlink history at market open ---
        # The market's start time is end_date minus the timeframe duration.
        # We look up the Chainlink price at that start timestamp.
        _TF_SECONDS: dict[str, int] = {
            "5min": 300, "15min": 900, "1hr": 3600, "4hr": 14400,
        }
        timeframe_seconds = _TF_SECONDS.get(c.timeframe, 900)

        remaining_secs: Optional[float] = None
        start_ts: Optional[float] = None
        if c.market.end_date:
            try:
                if hasattr(c.market.end_date, "timestamp"):
                    end_ts = c.market.end_date.timestamp()
                else:
                    end_str = str(c.market.end_date)
                    end_ts = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp()
                remaining_secs = max(0.0, end_ts - time.time())
                start_ts = end_ts - float(timeframe_seconds)
            except (ValueError, AttributeError):
                pass

        price_to_beat: Optional[float] = None
        if start_ts is not None:
            price_to_beat = feed.get_price_at_time(c.asset, start_ts)

        if price_to_beat is None or price_to_beat <= 0.0:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=(
                    f"No price_to_beat for {c.asset} "
                    f"(start_ts={'none' if start_ts is None else f'{start_ts:.0f}'})"
                ),
            )

        # --- Oracle direction: diff_pct sign determines side ---
        diff_pct = ((oracle_price - price_to_beat) / price_to_beat) * 100.0

        # Minimum oracle diff to have ANY directional edge
        _MIN_DIFF_FOR_DIRECTIONAL = 0.15  # %
        if abs(diff_pct) < _MIN_DIFF_FOR_DIRECTIONAL:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=(
                    f"Oracle diff too small for directional: |{diff_pct:.4f}%| < "
                    f"{_MIN_DIFF_FOR_DIRECTIONAL}%"
                ),
            )

        # Direction from oracle SIGN only -- never buy against oracle
        best_side = "UP" if diff_pct > 0 else "DOWN"
        buy_price = c.yes_price if best_side == "UP" else c.no_price

        # Edge = oracle signal magnitude scaled by time urgency
        if remaining_secs is not None:
            time_ratio = clamp(remaining_secs / float(max(1, timeframe_seconds)), 0.01, 1.0)
        else:
            time_ratio = 1.0
        time_multiplier = 1.0 + 2.0 * (1.0 - time_ratio)
        best_edge = abs(diff_pct) * time_multiplier / 100.0  # as fraction

        # Sigmoid for model_up (used by generate for expected payout calc)
        directional_scale = max(_DIRECTIONAL_MIN_SCALE, _DIRECTIONAL_BASE_SCALE * time_ratio)
        directional_z = clamp(diff_pct / directional_scale, -60.0, 60.0)
        model_up = clamp(
            1.0 / (1.0 + math.exp(-directional_z)),
            _DIRECTIONAL_ORACLE_PROB_MIN,
            _DIRECTIONAL_ORACLE_PROB_MAX,
        )

        # Fee-aware entry gate: edge must exceed 2x taker fee
        taker_fee_pct = polymarket_fee_pct(buy_price) if buy_price > 0 else 0.0
        min_fee_edge = taker_fee_pct * 2.0
        if best_edge < min_fee_edge:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=(
                    f"Edge below fee threshold: {best_side} edge={best_edge:.4f} < "
                    f"2xfee={min_fee_edge:.4f} (fee_pct={taker_fee_pct:.4f} at price={buy_price:.3f})"
                ),
            )

        # Time-phase awareness
        remaining_min = (remaining_secs / 60.0) if remaining_secs is not None else 15.0

        if remaining_min > _DIRECTIONAL_EARLY_PHASE_MINUTES:
            phase = "EARLY"
            min_edge = _DIRECTIONAL_EARLY_MIN_EDGE
            score_multiplier = _DIRECTIONAL_EARLY_SCORE_MULT
        elif remaining_min > _DIRECTIONAL_MID_PHASE_MINUTES:
            phase = "MID"
            min_edge = _DIRECTIONAL_MID_MIN_EDGE
            score_multiplier = _DIRECTIONAL_MID_SCORE_MULT
        else:
            phase = "LATE"
            min_edge = _DIRECTIONAL_LATE_MIN_EDGE
            score_multiplier = _DIRECTIONAL_LATE_SCORE_MULT

        if best_edge < min_edge:
            return SubStrategyScore(
                strategy=SubStrategy.DIRECTIONAL_EDGE,
                score=0.0,
                reason=(
                    f"Edge too small ({phase}): {best_side} edge={best_edge:.3f} "
                    f"(need >{min_edge:.2f}), model_up={model_up:.2f} "
                    f"vs market_up={market_up_prob:.3f}, "
                    f"diff_pct={diff_pct:+.4f}%, "
                    f"{remaining_min:.1f}min left"
                ),
            )

        # Score based on edge size, amplified by time phase
        score = min(best_edge * _DIRECTIONAL_EDGE_SCORE_SCALE * score_multiplier, _DIRECTIONAL_MAX_SCORE)

        # LATE phase bonus: oracle signal is most predictive near resolution
        if phase == "LATE":
            score += _DIRECTIONAL_LATE_PHASE_BONUS

        buy_fee = polymarket_fee_curve(buy_price)

        return SubStrategyScore(
            strategy=SubStrategy.DIRECTIONAL_EDGE,
            score=score,
            reason=(
                f"Directional {best_side} ({phase}, {remaining_min:.0f}m left): "
                f"edge={best_edge:.3f}, model_up={model_up:.2f}, "
                f"market_yes={c.yes_price:.3f}, market_no={c.no_price:.3f}, "
                f"oracle={oracle_price:.2f}, ptb={price_to_beat:.2f}, "
                f"diff_pct={diff_pct:+.4f}%, fee={buy_fee:.4f}"
            ),
            params={
                "side": best_side,
                "edge": best_edge,
                "model_up": model_up,
                "model_down": 1.0 - model_up,
                "market_up": c.yes_price,
                "market_down": c.no_price,
                "buy_price": buy_price,
                "oracle_price": oracle_price,
                "price_to_beat": price_to_beat,
                "diff_pct": diff_pct,
                "phase": phase,
                "remaining_minutes": remaining_min,
            },
        )

    # -- Helper: remaining seconds until resolution --

    def _remaining_seconds(self, c: HighFreqCandidate) -> Optional[float]:
        if c.market.end_date:
            try:
                if hasattr(c.market.end_date, "timestamp"):
                    end_ts = c.market.end_date.timestamp()
                else:
                    end_str = str(c.market.end_date)
                    end_ts = datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp()
                return max(0.0, end_ts - time.time())
            except (ValueError, AttributeError):
                pass
        return None

    # -- Sub-strategy: Maker Quote scoring (Layer 1 - PRIMARY) --

    def _score_maker_quote(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score two-sided maker quoting opportunity (Layer 1).

        Places post_only limit buy orders on BOTH YES and NO sides.
        Primary strategy: earns bid-ask spread + maker rebates.
        When oracle data available, skews quotes for directional edge.
        """
        spread = abs(1.0 - c.yes_price - c.no_price)
        if spread < _MAKER_QUOTE_MIN_SPREAD:
            return SubStrategyScore(
                strategy=SubStrategy.MAKER_QUOTE,
                score=0.0,
                reason=f"Spread too narrow ({spread:.4f} < {_MAKER_QUOTE_MIN_SPREAD})",
            )

        liquidity = c.market.liquidity
        if liquidity < _MAKER_QUOTE_MIN_LIQUIDITY:
            return SubStrategyScore(
                strategy=SubStrategy.MAKER_QUOTE,
                score=0.0,
                reason=f"Liquidity too low (${liquidity:.0f} < ${_MAKER_QUOTE_MIN_LIQUIDITY:.0f})",
            )

        # Calculate seconds remaining
        remaining_secs = self._remaining_seconds(c)

        if remaining_secs is not None and remaining_secs < _MAKER_QUOTE_MIN_SECONDS_LEFT:
            return SubStrategyScore(
                strategy=SubStrategy.MAKER_QUOTE,
                score=0.0,
                reason=f"Too close to resolution ({remaining_secs:.0f}s < {_MAKER_QUOTE_MIN_SECONDS_LEFT:.0f}s)",
            )

        # Base quote prices: 1 tick below each side's current ask
        base_quote_yes = c.yes_price - _MAKER_QUOTE_TICK_SIZE
        base_quote_no = c.no_price - _MAKER_QUOTE_TICK_SIZE

        if base_quote_yes <= 0.01 or base_quote_no <= 0.01:
            return SubStrategyScore(
                strategy=SubStrategy.MAKER_QUOTE,
                score=0.0,
                reason="Quote price too low after tick adjustment",
            )

        # Oracle-gated inventory skew (Layer 2 integration)
        oracle_skew = 0.0
        oracle_direction = None
        oracle_price = c.oracle_price
        price_to_beat = c.price_to_beat
        if oracle_price is not None and price_to_beat is not None and price_to_beat > 0:
            diff_pct = ((oracle_price - price_to_beat) / price_to_beat) * 100.0
            oracle_skew = clamp(diff_pct * 0.015, -_MAKER_QUOTE_SKEW_MAX, _MAKER_QUOTE_SKEW_MAX)
            oracle_direction = "up" if diff_pct > 0 else "down"

        # Apply skew: if oracle says UP, tighten YES quote (more aggressive), widen NO quote
        quote_yes = clamp(base_quote_yes - oracle_skew, 0.01, 0.99)
        quote_no = clamp(base_quote_no + oracle_skew, 0.01, 0.99)

        combined_cost = quote_yes + quote_no
        spread_capture = max(0.0, 1.0 - combined_cost)

        # Fill probability
        if liquidity < _MAKER_QUOTE_THIN_BOOK_USD:
            fill_prob = clamp(0.6 + 0.3 * (1.0 - liquidity / _MAKER_QUOTE_THIN_BOOK_USD), 0.3, 0.9)
        else:
            fill_prob = clamp(0.5 - 0.2 * (liquidity / 10000.0), 0.1, 0.5)

        # Resolution risk
        if remaining_secs is not None and remaining_secs < _MAKER_QUOTE_RESOLUTION_RISK_SECONDS:
            resolution_risk = 0.10 * (1.0 - remaining_secs / _MAKER_QUOTE_RESOLUTION_RISK_SECONDS)
        else:
            resolution_risk = 0.0

        expected_profit = spread_capture * fill_prob - resolution_risk
        if expected_profit <= 0.0:
            return SubStrategyScore(
                strategy=SubStrategy.MAKER_QUOTE,
                score=0.0,
                reason=f"Negative EV: spread_capture={spread_capture:.4f}, fill_prob={fill_prob:.2f}, resolution_risk={resolution_risk:.4f}",
            )

        score = _MAKER_QUOTE_BASE_SCORE + spread_capture * _MAKER_QUOTE_SPREAD_SCORE_SCALE
        if liquidity < _MAKER_QUOTE_THIN_BOOK_USD:
            score += _MAKER_QUOTE_THIN_BOOK_BONUS
        # Oracle skew bonus: having directional edge on top of spread is extra valuable
        if abs(oracle_skew) > 0.005:
            score += 10.0
        score = min(score, _MAKER_QUOTE_MAX_SCORE)

        size_ratio = clamp(spread_capture / 0.06, 0.0, 1.0)
        size_usd = _MAKER_QUOTE_MIN_SIZE_USD + size_ratio * (_MAKER_QUOTE_MAX_SIZE_USD - _MAKER_QUOTE_MIN_SIZE_USD)

        return SubStrategyScore(
            strategy=SubStrategy.MAKER_QUOTE,
            score=score,
            reason=(
                f"Maker quote: spread={spread:.4f}, combined=${combined_cost:.4f}, "
                f"spread_capture=${spread_capture:.4f}, fill_prob={fill_prob:.2f}, "
                f"EV=${expected_profit:.4f}, liq=${liquidity:.0f}, "
                f"oracle_skew={oracle_skew:+.4f}, size=${size_usd:.1f}/side"
            ),
            params={
                "quote_yes": quote_yes,
                "quote_no": quote_no,
                "combined_cost": combined_cost,
                "spread_capture": spread_capture,
                "fill_probability": fill_prob,
                "expected_profit": expected_profit,
                "resolution_risk": resolution_risk,
                "size_usd": size_usd,
                "liquidity": liquidity,
                "remaining_seconds": remaining_secs,
                "oracle_skew": oracle_skew,
                "oracle_direction": oracle_direction,
            },
        )

    # -- Sub-strategy: Convergence scoring (Layer 3) --

    def _score_convergence(self, c: HighFreqCandidate) -> SubStrategyScore:
        """Score near-expiry convergence opportunity (Layer 3).

        In the final 5-45 seconds before resolution, if the oracle strongly
        indicates a direction (diff > 0.3%), place a maker order at $0.85-$0.95
        on the winning side. The market converges to $1.00/$0.00 at resolution,
        so any fill below $0.95 is profitable.
        """
        remaining_secs = self._remaining_seconds(c)
        if remaining_secs is None:
            return SubStrategyScore(
                strategy=SubStrategy.CONVERGENCE,
                score=0.0,
                reason="Cannot determine seconds remaining",
            )

        if remaining_secs > _CONVERGENCE_MAX_SECONDS_LEFT or remaining_secs < _CONVERGENCE_MIN_SECONDS_LEFT:
            return SubStrategyScore(
                strategy=SubStrategy.CONVERGENCE,
                score=0.0,
                reason=f"Not in convergence window ({remaining_secs:.0f}s, need {_CONVERGENCE_MIN_SECONDS_LEFT}-{_CONVERGENCE_MAX_SECONDS_LEFT}s)",
            )

        # Need oracle data to determine winning side
        oracle_price = c.oracle_price
        price_to_beat = c.price_to_beat
        if oracle_price is None or price_to_beat is None or price_to_beat <= 0:
            return SubStrategyScore(
                strategy=SubStrategy.CONVERGENCE,
                score=0.0,
                reason="No oracle data for convergence",
            )

        diff_pct = abs((oracle_price - price_to_beat) / price_to_beat) * 100.0
        if diff_pct < _CONVERGENCE_MIN_ORACLE_DIFF_PCT:
            return SubStrategyScore(
                strategy=SubStrategy.CONVERGENCE,
                score=0.0,
                reason=f"Oracle diff too small ({diff_pct:.3f}% < {_CONVERGENCE_MIN_ORACLE_DIFF_PCT}%)",
            )

        # Determine winning side
        oracle_says_up = oracle_price > price_to_beat
        winning_side = "YES" if oracle_says_up else "NO"
        winning_price = c.yes_price if oracle_says_up else c.no_price

        # Entry price must be reasonable: between $0.85 and $0.95
        if winning_price < _CONVERGENCE_MIN_PRICE:
            return SubStrategyScore(
                strategy=SubStrategy.CONVERGENCE,
                score=0.0,
                reason=f"Winning side price too low ({winning_price:.3f} < {_CONVERGENCE_MIN_PRICE})",
            )
        if winning_price > _CONVERGENCE_MAX_ENTRY_PRICE:
            return SubStrategyScore(
                strategy=SubStrategy.CONVERGENCE,
                score=0.0,
                reason=f"Winning side already converged ({winning_price:.3f} > {_CONVERGENCE_MAX_ENTRY_PRICE})",
            )

        # Edge: expected $1.00 payout minus entry price
        edge = 1.0 - winning_price

        # Score based on edge and time to expiry (less time = more certainty)
        time_certainty = clamp(1.0 - remaining_secs / _CONVERGENCE_MAX_SECONDS_LEFT, 0.0, 1.0)
        oracle_strength = clamp(diff_pct / 1.0, 0.0, 1.0)
        score = _CONVERGENCE_BASE_SCORE + edge * 200.0 * time_certainty + oracle_strength * 20.0
        score = min(score, _CONVERGENCE_MAX_SCORE)

        # Quote price: 1 tick below current winning side (maker order)
        entry_price = winning_price - _MAKER_QUOTE_TICK_SIZE
        entry_price = clamp(entry_price, 0.01, 0.99)

        return SubStrategyScore(
            strategy=SubStrategy.CONVERGENCE,
            score=score,
            reason=(
                f"Convergence: {winning_side} @ ${winning_price:.3f}, "
                f"edge={edge:.3f}, diff={diff_pct:.3f}%, "
                f"{remaining_secs:.0f}s left, oracle_strength={oracle_strength:.2f}"
            ),
            params={
                "winning_side": winning_side,
                "winning_price": winning_price,
                "entry_price": entry_price,
                "edge": edge,
                "diff_pct": diff_pct,
                "remaining_seconds": remaining_secs,
                "oracle_says_up": oracle_says_up,
                "time_certainty": time_certainty,
                "oracle_strength": oracle_strength,
            },
        )

    # ------------------------------------------------------------------
    # Opportunity generation
    # ------------------------------------------------------------------

    def _generate_opportunity(
        self,
        candidate: HighFreqCandidate,
        selected: SubStrategyScore,
    ) -> Optional[Opportunity]:
        """Turn a scored sub-strategy into an Opportunity via the base
        class ``create_opportunity`` (which applies all hard filters)."""

        market = candidate.market
        sub = selected.strategy
        params = selected.params

        if sub == SubStrategy.MAKER_QUOTE:
            return self._generate_maker_quote(candidate, params)
        elif sub == SubStrategy.DIRECTIONAL_EDGE:
            return self._generate_directional_edge(candidate, params)
        elif sub == SubStrategy.CONVERGENCE:
            return self._generate_convergence(candidate, params)

        logger.warning(
            "BtcEthHighFreq: unknown sub-strategy %s for market %s",
            sub,
            market.id,
        )
        return None

    def _generate_directional_edge(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[Opportunity]:
        """Generate opportunity for sub-strategy D: Directional Edge.

        Buys only the predicted winning side (directional bet, not arb).
        Uses maker orders to avoid taker fees and earn rebates.
        """
        market = c.market
        side = params["side"]  # "UP" or "DOWN"
        edge = params["edge"]
        buy_price = params["buy_price"]
        model_up = params["model_up"]

        # Build single-side position
        maker_mode = _cfg.BTC_ETH_HF_MAKER_MODE
        positions = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            if side == "UP":
                token_id = market.clob_token_ids[0]
                outcome = "YES"
            else:
                token_id = market.clob_token_ids[1]
                outcome = "NO"

            positions = [
                {
                    "action": "BUY",
                    "outcome": outcome,
                    "price": buy_price,
                    "token_id": token_id,
                    "_maker_mode": maker_mode,
                    "_maker_price": buy_price,
                    "post_only": maker_mode,
                }
            ]

        # Fair value for directional bet: model probability * $1.00 payout
        expected_payout = model_up if side == "UP" else (1.0 - model_up)

        opp = self.create_opportunity(
            title=(f"BTC/ETH HF Directional: {c.asset} {c.timeframe} ({side} edge {edge:.1%})"),
            description=(
                f"Directional {side} bet on {c.asset} {c.timeframe} market. "
                f"Model: {model_up:.0%} Up / {1 - model_up:.0%} Down. "
                f"Market: {params['market_up']:.1%} Up. "
                f"Edge: {edge:.1%}. "
                f"{'Maker order (0% fee + rebates).' if maker_mode else ''}"
            ),
            total_cost=buy_price,
            expected_payout=expected_payout,
            markets=[market],
            positions=positions,
            is_guaranteed=False,  # Directional, not guaranteed
            min_liquidity_hard=200.0,
            min_position_size=5.0,
            min_absolute_profit=0.5,
        )

        if opp is not None:
            self._attach_highfreq_metadata(
                opp,
                c,
                SubStrategy.DIRECTIONAL_EDGE,
                params,
            )
            opp.risk_factors.insert(
                0,
                f"Directional bet: profit depends on {c.asset} going {side}",
            )
        return opp

    def _generate_maker_quote(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[Opportunity]:
        """Generate opportunity for Layer 1: Maker Quote.

        Places post_only limit buy orders on BOTH YES and NO sides,
        1 tick below each side's current ask.  Earns the bid-ask spread
        plus maker rebates when both sides fill.  Oracle skew tilts quotes
        toward the predicted winner for additional directional edge.
        """
        market = c.market
        quote_yes = params["quote_yes"]
        quote_no = params["quote_no"]
        combined_cost = params["combined_cost"]
        size_usd = params["size_usd"]

        positions: list[dict] = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            positions = [
                {
                    "action": "LIMIT_BUY",
                    "outcome": "YES",
                    "price": quote_yes,
                    "token_id": market.clob_token_ids[0],
                    "post_only": True,
                    "_maker_mode": True,
                    "_maker_price": quote_yes,
                    "note": f"Maker quote YES @ ${quote_yes:.2f}",
                },
                {
                    "action": "LIMIT_BUY",
                    "outcome": "NO",
                    "price": quote_no,
                    "token_id": market.clob_token_ids[1],
                    "post_only": True,
                    "_maker_mode": True,
                    "_maker_price": quote_no,
                    "note": f"Maker quote NO @ ${quote_no:.2f}",
                },
            ]

        spread_capture = params["spread_capture"]
        oracle_skew = params.get("oracle_skew", 0.0)

        opp = self.create_opportunity(
            title=(
                f"BTC/ETH HF Maker Quote: {c.asset} {c.timeframe} "
                f"(spread ${spread_capture:.3f})"
            ),
            description=(
                f"Two-sided maker quoting on {c.asset} {c.timeframe} market. "
                f"Quote YES@${quote_yes:.2f} + NO@${quote_no:.2f} = "
                f"${combined_cost:.4f} for $1.00 payout. "
                f"Spread capture: ${spread_capture:.4f}. "
                f"Oracle skew: {oracle_skew:+.4f}. "
                f"Post-only maker orders (0% taker fee + rebates)."
            ),
            total_cost=combined_cost,
            expected_payout=1.0,
            markets=[market],
            positions=positions,
            is_guaranteed=False,
            min_liquidity_hard=0.0,
            min_position_size=0.0,
            min_absolute_profit=0.0,
        )

        if opp is not None:
            opp.max_position_size = max(opp.max_position_size, size_usd * 2.0)
            self._attach_highfreq_metadata(
                opp,
                c,
                SubStrategy.MAKER_QUOTE,
                params,
            )
            opp.risk_factors.insert(
                0,
                "Maker quote: profit requires BOTH sides filling; "
                "single-side fill creates directional exposure",
            )
        return opp

    def _generate_convergence(
        self,
        c: HighFreqCandidate,
        params: dict,
    ) -> Optional[Opportunity]:
        """Generate opportunity for Layer 3: Near-expiry convergence."""
        market = c.market
        winning_side = params["winning_side"]
        entry_price = params["entry_price"]
        edge = params["edge"]

        positions: list[dict] = []
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            token_idx = 0 if winning_side == "YES" else 1
            positions = [
                {
                    "action": "LIMIT_BUY",
                    "outcome": winning_side,
                    "price": entry_price,
                    "token_id": market.clob_token_ids[token_idx],
                    "post_only": True,
                    "_maker_mode": True,
                    "_maker_price": entry_price,
                    "note": f"Convergence {winning_side} @ ${entry_price:.3f} ({params['remaining_seconds']:.0f}s left)",
                }
            ]

        opp = self.create_opportunity(
            title=f"BTC/ETH HF Convergence: {c.asset} {c.timeframe} ({winning_side} @ ${entry_price:.2f})",
            description=(
                f"Near-expiry convergence on {c.asset} {c.timeframe}. "
                f"Oracle diff {params['diff_pct']:.2f}%, {winning_side} @ ${entry_price:.2f}, "
                f"{params['remaining_seconds']:.0f}s to resolution. "
                f"Edge: {edge:.1%}. Post-only maker order."
            ),
            total_cost=entry_price,
            expected_payout=1.0,
            markets=[market],
            positions=positions,
            is_guaranteed=False,
            min_liquidity_hard=0.0,
            min_position_size=0.0,
            min_absolute_profit=0.0,
        )

        if opp is not None:
            self._attach_highfreq_metadata(opp, c, SubStrategy.CONVERGENCE, params)
            opp.risk_factors.insert(0, f"Convergence bet: {params['remaining_seconds']:.0f}s to resolution")
        return opp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_highfreq_metadata(
        opp: Opportunity,
        candidate: HighFreqCandidate,
        sub_strategy: SubStrategy,
        params: dict,
    ) -> None:
        """Attach BTC/ETH high-freq metadata to the opportunity for
        downstream consumers (execution engine, dashboard, logging)."""
        # Store in the existing positions_to_take metadata (which is a list
        # of dicts). We append a metadata entry at the end.
        opp.positions_to_take.append(
            {
                "_highfreq_metadata": True,
                "asset": candidate.asset,
                "timeframe": candidate.timeframe,
                "sub_strategy": sub_strategy.value,
                "sub_strategy_params": params,
            }
        )

    # ------------------------------------------------------------------
    # Evaluate / Should-Exit  (unified strategy interface)
    # ------------------------------------------------------------------

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        """Full crypto high-frequency evaluation with multi-mode regime system,
        direction guardrails, component edges, and asset/timeframe filtering.

        Ported from BaseCryptoTimeframeStrategy.evaluate().
        """
        params = context.get("params") or {}
        payload = signal_payload(signal)
        live_market = context.get("live_market")
        if not isinstance(live_market, dict):
            live_market = payload.get("live_market")
        if not isinstance(live_market, dict):
            live_market = {}

        # --- Core thresholds ---
        min_edge = to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = to_confidence(params.get("min_confidence", 0.45), 0.45)
        base_size = to_float(params.get("base_size_usd", 25.0), 25.0)
        max_size = max(1.0, to_float(params.get("max_size_usd", base_size * 3.0), base_size * 3.0))

        # --- Direction guardrail parameters ---
        guardrail_enabled = to_bool(params.get("direction_guardrail_enabled"), True)
        guardrail_prob_floor = max(
            0.5,
            min(1.0, to_float(params.get("direction_guardrail_prob_floor", 0.55), 0.55)),
        )
        guardrail_price_floor = max(
            0.5,
            min(1.0, to_float(params.get("direction_guardrail_price_floor", 0.80), 0.80)),
        )
        guardrail_regimes = _normalize_regime_scope(params.get("direction_guardrail_regimes", ["mid", "closing"]))
        if not guardrail_regimes:
            guardrail_regimes = {"mid", "closing"}

        # --- Mode selection ---
        requested_mode = _normalize_mode(params.get("strategy_mode") or params.get("mode"))
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        regime = _normalize_regime(payload.get("regime"))

        # --- Asset / timeframe extraction ---
        signal_asset = _normalize_asset(
            _first_present(
                live_market.get("asset"),
                live_market.get("coin"),
                live_market.get("symbol"),
                payload.get("asset"),
                payload.get("coin"),
                payload.get("symbol"),
            )
        )
        signal_timeframe = _normalize_timeframe(
            _first_present(
                live_market.get("timeframe"),
                live_market.get("cadence"),
                live_market.get("interval"),
                payload.get("timeframe"),
                payload.get("cadence"),
                payload.get("interval"),
            )
        )

        # --- Asset/timeframe include+exclude filtering ---
        include_assets = _normalize_scope(
            params.get("include_assets"),
            _normalize_asset,
        )
        exclude_assets = _normalize_scope(
            _first_present(
                params.get("exclude_assets"),
            ),
            _normalize_asset,
        )
        include_timeframes = _normalize_scope(
            params.get("include_timeframes"),
            _normalize_timeframe,
        )
        exclude_timeframes = _normalize_scope(
            _first_present(
                params.get("exclude_timeframes"),
            ),
            _normalize_timeframe,
        )
        asset_in_scope = (not include_assets) or (bool(signal_asset) and signal_asset in include_assets)
        asset_not_excluded = not (bool(signal_asset) and signal_asset in exclude_assets)
        asset_scope_ok = asset_in_scope and asset_not_excluded
        # Unified strategy handles all timeframes — no fixed expected_timeframe.
        # The strategy_timeframe check passes when no single timeframe is enforced.
        strategy_timeframe_ok = True
        timeframe_in_scope = (not include_timeframes) or (
            bool(signal_timeframe) and signal_timeframe in include_timeframes
        )
        timeframe_not_excluded = not (bool(signal_timeframe) and signal_timeframe in exclude_timeframes)
        timeframe_scope_ok = timeframe_in_scope and timeframe_not_excluded

        # --- Active mode resolution ---
        dominant_mode = _normalize_mode(payload.get("dominant_strategy"))
        active_mode = dominant_mode if requested_mode == "auto" and dominant_mode != "auto" else requested_mode
        if active_mode == "auto":
            active_mode = "maker_quote"
        enabled_active_modes = _resolve_enabled_active_modes(params)
        mode_allowlist_ok = active_mode in enabled_active_modes

        # --- Source / origin checks ---
        source_ok = str(getattr(signal, "source", "")) == "crypto"
        signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()
        origin_ok = str(
            payload.get("strategy_origin") or ""
        ).strip().lower() == "crypto_worker" or signal_type.startswith("crypto_worker")

        live_window_required = to_bool(params.get("live_window_required"), True)
        signal_is_live_raw = _first_present(live_market.get("is_live"), payload.get("is_live"))
        signal_is_live = signal_is_live_raw if isinstance(signal_is_live_raw, bool) else None
        signal_is_current_raw = _first_present(live_market.get("is_current"), payload.get("is_current"))
        signal_is_current = signal_is_current_raw if isinstance(signal_is_current_raw, bool) else None
        signal_seconds_left = to_float(
            _first_present(
                live_market.get("seconds_left"),
                payload.get("seconds_left"),
            ),
            -1.0,
        )
        signal_end_time = str(
            _first_present(
                live_market.get("market_end_time"),
                live_market.get("end_time"),
                payload.get("end_time"),
            )
            or ""
        ).strip()
        if signal_is_live is None and signal_end_time:
            try:
                parsed_end = datetime.fromisoformat(signal_end_time.replace("Z", "+00:00"))
                signal_is_live = parsed_end.timestamp() > time.time()
            except Exception:
                signal_is_live = None
        if signal_is_live is None and signal_seconds_left >= 0:
            signal_is_live = signal_seconds_left > 0
        if signal_is_current is None:
            signal_is_current = signal_is_live
        live_window_ok = (not live_window_required) or bool(signal_is_live and signal_is_current)

        max_live_context_age_seconds = max(
            0.1,
            to_float(params.get("max_live_context_age_seconds", 3.0), 3.0),
        )
        live_context_fetched_at = _parse_datetime_utc(
            _first_present(
                live_market.get("fetched_at"),
                payload.get("live_market_fetched_at"),
            )
        )
        live_context_age_seconds: Optional[float] = None
        if live_context_fetched_at is not None:
            live_context_age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - live_context_fetched_at.astimezone(timezone.utc)).total_seconds(),
            )
        live_context_fresh_ok = (
            live_context_age_seconds is None or live_context_age_seconds <= max_live_context_age_seconds
        )

        signal_timestamp_used = _parse_datetime_utc(
            _first_present(
                live_market.get("signal_updated_at"),
                live_market.get("updated_at"),
                live_market.get("fetched_at"),
                payload.get("signal_updated_at"),
                payload.get("signal_emitted_at"),
                payload.get("live_market_fetched_at"),
                getattr(signal, "updated_at", None),
                getattr(signal, "created_at", None),
            )
        )
        signal_created_at = _parse_datetime_utc(getattr(signal, "created_at", None))
        signal_updated_at = _parse_datetime_utc(getattr(signal, "updated_at", None))
        signal_age_seconds: Optional[float] = None
        if isinstance(signal_timestamp_used, datetime):
            signal_ts_utc = (
                signal_timestamp_used
                if signal_timestamp_used.tzinfo
                else signal_timestamp_used.replace(tzinfo=timezone.utc)
            )
            signal_age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - signal_ts_utc.astimezone(timezone.utc)).total_seconds(),
            )
        max_signal_age_seconds_cfg = self._float(
            _timeframe_override(params, "max_signal_age_seconds", signal_timeframe)
        )
        if max_signal_age_seconds_cfg is None:
            max_signal_age_seconds_cfg = to_float(params.get("max_signal_age_seconds", 20.0), 20.0)
        max_signal_age_seconds = max(0.1, float(max_signal_age_seconds_cfg))
        signal_fresh_ok = signal_age_seconds is None or signal_age_seconds <= max_signal_age_seconds

        market_data_age_ms = self._float(
            _first_present(
                live_market.get("market_data_age_ms"),
                payload.get("market_data_age_ms"),
            )
        )
        if market_data_age_ms is None:
            observed_at = _parse_datetime_utc(
                _first_present(
                    live_market.get("source_observed_at"),
                    payload.get("source_observed_at"),
                    live_market.get("fetched_at"),
                    payload.get("live_market_fetched_at"),
                    payload.get("signal_updated_at"),
                    getattr(signal, "updated_at", None),
                    getattr(signal, "created_at", None),
                )
            )
            if observed_at is not None:
                market_data_age_ms = max(
                    0.0,
                    (datetime.now(timezone.utc) - observed_at.astimezone(timezone.utc)).total_seconds() * 1000.0,
                )
        max_market_data_age_ms_cfg = self._float(
            _timeframe_override(params, "max_market_data_age_ms", signal_timeframe)
        )
        if max_market_data_age_ms_cfg is None:
            max_market_data_age_ms_cfg = to_float(params.get("max_market_data_age_ms", 900.0), 900.0)
        max_market_data_age_ms = max(50.0, float(max_market_data_age_ms_cfg))
        require_market_data_age_for_sources = {
            str(item or "").strip().lower()
            for item in _as_list(_first_present(params.get("require_market_data_age_for_sources"), ["crypto"]))
            if str(item or "").strip()
        }
        require_market_data_age = (
            str(getattr(signal, "source", "") or "").strip().lower() in require_market_data_age_for_sources
        )
        market_data_freshness_enforced = to_bool(params.get("enforce_market_data_freshness"), True)
        market_data_fresh_ok = (
            (not market_data_freshness_enforced)
            or (market_data_age_ms is not None and market_data_age_ms <= max_market_data_age_ms)
            or (market_data_age_ms is None and not require_market_data_age)
        )

        default_min_seconds_by_timeframe: dict[str, float] = {
            "5m": 45.0,
            "15m": 180.0,
            "1h": 360.0,
            "4h": 900.0,
        }
        timeframe_specific_floor = self._float(
            _timeframe_override(params, "min_seconds_left_for_entry", signal_timeframe)
        )
        global_min_seconds = self._float(params.get("min_seconds_left_for_entry"))
        min_seconds_left_for_entry = (
            max(0.0, timeframe_specific_floor)
            if timeframe_specific_floor is not None
            else (
                max(0.0, global_min_seconds)
                if global_min_seconds is not None
                else default_min_seconds_by_timeframe.get(signal_timeframe, 0.0)
            )
        )
        entry_window_ok = signal_seconds_left < 0 or signal_seconds_left >= float(min_seconds_left_for_entry)

        if signal_seconds_left >= 0 and signal_timeframe:
            regime = self._crypto_regime(signal_seconds_left, self._timeframe_seconds(signal_timeframe))

        live_window_detail = (
            f"required={live_window_required} "
            f"is_live={signal_is_live if signal_is_live is not None else 'unknown'} "
            f"is_current={signal_is_current if signal_is_current is not None else 'unknown'} "
            f"seconds_left={signal_seconds_left if signal_seconds_left >= 0 else 'unknown'}"
        )
        signal_freshness_detail = (
            f"age={signal_age_seconds:.1f}s max={max_signal_age_seconds:.1f}s "
            f"timestamp={_to_iso_utc(signal_timestamp_used)}"
            if signal_age_seconds is not None
            else "signal timestamp unavailable"
        )
        market_data_freshness_detail = (
            (
                f"age_ms={market_data_age_ms:.0f} max={max_market_data_age_ms:.0f} "
                f"source={str(getattr(signal, 'source', '') or '').strip().lower() or 'unknown'}"
            )
            if market_data_age_ms is not None
            else (
                "market_data_age unavailable but optional"
                if not require_market_data_age
                else "market_data_age unavailable and required"
            )
        )
        entry_window_detail = (
            f"seconds_left={signal_seconds_left:.1f} required>={min_seconds_left_for_entry:.1f}"
            if signal_seconds_left >= 0
            else "seconds_left unavailable"
        )
        live_context_freshness_detail = (
            f"age={live_context_age_seconds:.1f}s max={max_live_context_age_seconds:.1f}s"
            if live_context_age_seconds is not None
            else "live context timestamp unavailable"
        )

        # --- Edge / confidence ---
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        mode_edge = _get_component_edge(payload, direction, active_mode)
        if mode_edge <= 0.0 and active_mode == dominant_mode and edge > 0.0:
            mode_edge = edge
        net_edge = _get_net_edge(payload, direction, edge)

        signal_liquidity_usd = self._float(
            _first_present(
                live_market.get("liquidity_usd"),
                live_market.get("liquidity"),
                payload.get("liquidity_usd"),
                payload.get("liquidity"),
                getattr(signal, "liquidity", None),
            )
        )
        min_liquidity_usd = max(0.0, to_float(params.get("min_liquidity_usd", 300.0), 300.0))
        if regime == "opening":
            min_liquidity_opening = self._float(
                _timeframe_override(params, "min_liquidity_usd_opening", signal_timeframe)
            )
            if min_liquidity_opening is None:
                min_liquidity_opening = self._float(params.get("min_liquidity_usd_opening"))
            if min_liquidity_opening is not None:
                min_liquidity_usd = max(min_liquidity_usd, max(0.0, float(min_liquidity_opening)))
        liquidity_ok = signal_liquidity_usd is None or signal_liquidity_usd >= min_liquidity_usd
        liquidity_detail = (
            f"liquidity={signal_liquidity_usd:.0f} min={min_liquidity_usd:.0f}"
            if signal_liquidity_usd is not None
            else "liquidity unavailable"
        )

        signal_volume_usd = self._float(
            _first_present(
                live_market.get("volume_usd"),
                live_market.get("volume"),
                payload.get("volume_usd"),
                payload.get("volume"),
                getattr(signal, "volume", None),
            )
        )

        direction_policy_ok, direction_policy_detail = crypto_highfreq_direction_allowed(
            params,
            regime=regime,
            active_mode=active_mode,
            direction=direction,
            timeframe=signal_timeframe,
            seconds_left=signal_seconds_left if signal_seconds_left >= 0.0 else None,
        )

        signal_spread = self._float(
            _first_present(
                live_market.get("spread"),
                payload.get("spread"),
                payload.get("market_spread"),
            )
        )
        if signal_spread is not None:
            signal_spread = max(0.0, min(1.0, signal_spread))
        max_spread_pct = max(0.0, min(1.0, to_float(params.get("max_spread_pct", 0.06), 0.06)))
        spread_ok = signal_spread is None or signal_spread <= max_spread_pct
        spread_detail = (
            f"spread={signal_spread:.4f} max={max_spread_pct:.4f}"
            if signal_spread is not None
            else "spread unavailable"
        )

        history_summary = live_market.get("history_summary")
        if not isinstance(history_summary, dict):
            history_summary = payload.get("history_summary")
        if not isinstance(history_summary, dict):
            history_summary = {}

        move_5m_pct = self._float(
            _first_present(
                (
                    (history_summary.get("move_5m") or {}).get("percent")
                    if isinstance(history_summary.get("move_5m"), dict)
                    else None
                ),
                payload.get("move_5m_percent"),
                payload.get("move_5m_pct"),
            )
        )
        move_30m_pct = self._float(
            _first_present(
                (
                    (history_summary.get("move_30m") or {}).get("percent")
                    if isinstance(history_summary.get("move_30m"), dict)
                    else None
                ),
                payload.get("move_30m_percent"),
                payload.get("move_30m_pct"),
            )
        )
        move_2h_pct = self._float(
            _first_present(
                (
                    (history_summary.get("move_2h") or {}).get("percent")
                    if isinstance(history_summary.get("move_2h"), dict)
                    else None
                ),
                payload.get("move_2h_percent"),
                payload.get("move_2h_pct"),
            )
        )

        recent_move_zscore: Optional[float] = None
        if move_5m_pct is not None:
            move_scale = max(
                0.15,
                abs(move_30m_pct or 0.0) / 2.0,
                abs(move_2h_pct or 0.0) / 3.0,
            )
            recent_move_zscore = abs(move_5m_pct) / move_scale if move_scale > 0 else None
        max_recent_move_zscore_for_entry = max(
            0.2,
            to_float(params.get("max_recent_move_zscore_for_entry", 2.0), 2.0),
        )
        recent_move_ok = recent_move_zscore is None or recent_move_zscore <= max_recent_move_zscore_for_entry
        recent_move_detail = (
            f"z={recent_move_zscore:.2f} max={max_recent_move_zscore_for_entry:.2f} "
            f"move_5m={move_5m_pct if move_5m_pct is not None else 'n/a'} "
            f"move_30m={move_30m_pct if move_30m_pct is not None else 'n/a'} "
            f"move_2h={move_2h_pct if move_2h_pct is not None else 'n/a'}"
            if recent_move_zscore is not None
            else "recent move z-score unavailable"
        )

        spread_widening_bps = self._float(
            _first_present(
                live_market.get("spread_widening_bps"),
                live_market.get("spread_delta_bps"),
                payload.get("spread_widening_bps"),
                payload.get("spread_delta_bps"),
            )
        )
        max_spread_widening_bps = max(
            0.0,
            to_float(params.get("max_spread_widening_bps", 20.0), 20.0),
        )
        spread_widening_ok = spread_widening_bps is None or spread_widening_bps <= max_spread_widening_bps
        spread_widening_detail = (
            f"widening_bps={spread_widening_bps:.2f} max={max_spread_widening_bps:.2f}"
            if spread_widening_bps is not None
            else "spread widening unavailable"
        )

        raw_orderbook_imbalance = self._float(
            _first_present(
                live_market.get("orderbook_imbalance"),
                live_market.get("book_imbalance"),
                live_market.get("imbalance"),
                payload.get("orderbook_imbalance"),
                payload.get("book_imbalance"),
                payload.get("imbalance"),
            )
        )
        orderbook_imbalance = None
        if raw_orderbook_imbalance is not None:
            normalized = abs(raw_orderbook_imbalance)
            if normalized > 1.0 and normalized <= 100.0:
                normalized /= 100.0
            orderbook_imbalance = min(1.0, max(0.0, normalized))
        max_orderbook_imbalance = min(
            1.0,
            max(0.5, to_float(params.get("max_orderbook_imbalance", 0.88), 0.88)),
        )
        orderbook_imbalance_ok = orderbook_imbalance is None or orderbook_imbalance <= max_orderbook_imbalance
        orderbook_imbalance_detail = (
            f"imbalance={orderbook_imbalance:.3f} max={max_orderbook_imbalance:.3f}"
            if orderbook_imbalance is not None
            else "orderbook imbalance unavailable"
        )

        # --- Oracle / guardrail data ---
        model_prob_yes = max(0.0, min(1.0, to_float(payload.get("model_prob_yes"), 0.5)))
        model_prob_no = max(0.0, min(1.0, to_float(payload.get("model_prob_no"), 0.5)))
        up_price = max(0.0, min(1.0, to_float(payload.get("up_price"), 0.5)))
        down_price = max(0.0, min(1.0, to_float(payload.get("down_price"), 0.5)))
        now_epoch_ms = int(time.time() * 1000.0)
        oracle_status = _extract_oracle_status(
            live_market=live_market,
            payload=payload,
            now_ms=now_epoch_ms,
        )
        oracle_age_ms = self._float(oracle_status.get("age_ms"))
        oracle_age_seconds = (oracle_age_ms / 1000.0) if oracle_age_ms is not None else None
        max_oracle_age_seconds_cfg = to_float(params.get("max_oracle_age_seconds", 12.0), 12.0)
        max_oracle_age_seconds_cfg = max(0.1, float(max_oracle_age_seconds_cfg))
        max_oracle_age_ms_cfg = self._float(params.get("max_oracle_age_ms"))
        max_oracle_age_ms = (
            max(100.0, float(max_oracle_age_ms_cfg))
            if max_oracle_age_ms_cfg is not None
            else max(100.0, max_oracle_age_seconds_cfg * 1000.0)
        )
        max_oracle_age_seconds = max_oracle_age_ms / 1000.0
        require_oracle_for_directional = to_bool(params.get("require_oracle_for_directional"), True)
        oracle_required = require_oracle_for_directional and active_mode == "directional"
        oracle_source_policy = str(params.get("oracle_source_policy") or "degrade").strip().lower()
        if oracle_source_policy not in {"degrade", "hard_skip", "allow_fallback"}:
            oracle_source_policy = "degrade"

        oracle_by_source = (
            dict(oracle_status.get("by_source")) if isinstance(oracle_status.get("by_source"), dict) else {}
        )
        chainlink_point = oracle_by_source.get("chainlink")
        binance_point = oracle_by_source.get("binance")
        binance_direct_point = oracle_by_source.get("binance_direct")
        chainlink_age_ms = (
            self._float(chainlink_point.get("age_ms"))
            if isinstance(chainlink_point, dict)
            else (oracle_age_ms if str(oracle_status.get("source") or "").strip().lower() == "chainlink" else None)
        )
        binance_age_ms = (
            self._float(binance_point.get("age_ms"))
            if isinstance(binance_point, dict)
            else (oracle_age_ms if str(oracle_status.get("source") or "").strip().lower() == "binance" else None)
        )
        binance_direct_age_ms = (
            self._float(binance_direct_point.get("age_ms"))
            if isinstance(binance_direct_point, dict)
            else None
        )
        binance_rtds_fresh = binance_age_ms is not None and binance_age_ms <= max_oracle_age_ms
        binance_direct_fresh = binance_direct_age_ms is not None and binance_direct_age_ms <= max_oracle_age_ms
        binance_fresh = binance_rtds_fresh or binance_direct_fresh
        # Prefer direct Binance (lower latency) over RTDS-relayed Binance.
        binance_fallback_source = "binance_direct" if binance_direct_fresh else "binance"
        binance_fallback_point = binance_direct_point if binance_direct_fresh else binance_point
        chainlink_stale_binance_fresh = (
            chainlink_age_ms is not None and chainlink_age_ms > max_oracle_age_ms and bool(binance_fresh)
        )
        oracle_fresh_available = bool(
            oracle_status.get("fresh_available")
            if oracle_status.get("fresh_available") is not None
            else (oracle_status.get("has_price") and oracle_status.get("has_timestamp"))
        )
        oracle_available = bool(oracle_status.get("available"))
        oracle_directional_available = bool(
            oracle_status.get("directional_available")
            if oracle_status.get("directional_available") is not None
            else oracle_available
        )
        oracle_fallback_used = False
        oracle_fallback_degraded = False
        oracle_source_policy_ok = True
        oracle_effective_source = str(oracle_status.get("source") or "").strip().lower() or None
        oracle_fallback_reason = "not_needed"
        if oracle_required and not oracle_directional_available:
            oracle_fallback_reason = "required_but_unavailable"
        if oracle_required and chainlink_stale_binance_fresh:
            if oracle_source_policy == "hard_skip":
                oracle_source_policy_ok = False
                oracle_fallback_reason = "hard_skip_policy"
            else:
                oracle_fallback_used = True
                oracle_fallback_degraded = oracle_source_policy == "degrade"
                oracle_effective_source = binance_fallback_source
                fallback_price_to_beat = self._float(oracle_status.get("price_to_beat"))
                if isinstance(binance_fallback_point, dict):
                    fallback_price = self._float(binance_fallback_point.get("price"))
                    fallback_updated_at_ms = self._float(binance_fallback_point.get("updated_at_ms"))
                    fallback_age_ms = self._float(binance_fallback_point.get("age_ms"))
                    if fallback_price is not None:
                        oracle_status["price"] = fallback_price
                    if fallback_updated_at_ms is not None:
                        oracle_status["updated_at_ms"] = int(fallback_updated_at_ms)
                    if fallback_age_ms is not None:
                        oracle_status["age_ms"] = fallback_age_ms
                    oracle_status["source"] = binance_fallback_source
                    oracle_age_ms = self._float(oracle_status.get("age_ms"))
                    oracle_age_seconds = (oracle_age_ms / 1000.0) if oracle_age_ms is not None else None
                fallback_price = self._float(oracle_status.get("price"))
                fallback_updated_at_ms = self._float(oracle_status.get("updated_at_ms"))
                fallback_availability = _resolve_oracle_availability(
                    price=fallback_price,
                    price_to_beat=fallback_price_to_beat,
                    age_ms=self._float(oracle_status.get("age_ms")),
                    updated_at_ms=(int(fallback_updated_at_ms) if fallback_updated_at_ms is not None else None),
                )
                fallback_available = bool(fallback_availability["available"])
                fallback_fresh_available = bool(fallback_availability["fresh_available"])
                fallback_directional_available = bool(fallback_availability["directional_available"])
                oracle_status["availability_reasons"] = list(fallback_availability["availability_reasons"])
                oracle_status["fresh_available"] = bool(fallback_availability["fresh_available"])
                oracle_status["freshness_state"] = str(fallback_availability["freshness_state"])
                oracle_status["freshness_reasons"] = list(fallback_availability["freshness_reasons"])
                oracle_status["directional_available"] = bool(fallback_availability["directional_available"])
                oracle_status["directional_state"] = str(fallback_availability["directional_state"])
                oracle_status["directional_reasons"] = list(fallback_availability["directional_reasons"])
                oracle_status["has_price"] = bool(fallback_availability["has_price"])
                oracle_status["has_price_to_beat"] = bool(fallback_availability["has_price_to_beat"])
                oracle_status["has_timestamp"] = bool(fallback_availability["has_timestamp"])
                if fallback_available:
                    oracle_status["availability_state"] = (
                        "available_degraded_binance_fallback"
                        if oracle_fallback_degraded
                        else "available_binance_fallback"
                    )
                else:
                    oracle_status["availability_state"] = str(fallback_availability["availability_state"])
                oracle_status["available"] = fallback_available
                oracle_fresh_available = fallback_fresh_available
                oracle_available = fallback_available
                oracle_directional_available = fallback_directional_available
                oracle_fallback_reason = "degraded_binance_fallback" if oracle_fallback_degraded else "binance_fallback"

        oracle_threshold_edge_multiplier = 1.0
        oracle_threshold_conf_multiplier = 1.0
        oracle_size_multiplier = 1.0
        if oracle_fallback_degraded:
            # Direct Binance WS has lower latency than RTDS relay — apply
            # reduced degradation penalties when it is the fallback source.
            if oracle_effective_source == "binance_direct":
                oracle_threshold_edge_multiplier = clamp(
                    to_float(params.get("oracle_fallback_degrade_edge_multiplier_direct"), 1.15),
                    1.0,
                    3.0,
                )
                oracle_threshold_conf_multiplier = clamp(
                    to_float(params.get("oracle_fallback_degrade_confidence_multiplier_direct"), 1.04),
                    1.0,
                    2.0,
                )
                oracle_size_multiplier = clamp(
                    to_float(params.get("oracle_fallback_degrade_size_multiplier_direct"), 0.70),
                    0.05,
                    1.0,
                )
            else:
                oracle_threshold_edge_multiplier = clamp(
                    to_float(params.get("oracle_fallback_degrade_edge_multiplier"), 1.35),
                    1.0,
                    3.0,
                )
                oracle_threshold_conf_multiplier = clamp(
                    to_float(params.get("oracle_fallback_degrade_confidence_multiplier"), 1.08),
                    1.0,
                    2.0,
                )
                oracle_size_multiplier = clamp(
                    to_float(params.get("oracle_fallback_degrade_size_multiplier"), 0.45),
                    0.05,
                    1.0,
                )

        oracle_fresh_base = oracle_fresh_available and oracle_age_ms is not None and oracle_age_ms <= max_oracle_age_ms
        if oracle_required:
            oracle_fresh_ok = oracle_fresh_base and oracle_source_policy_ok and oracle_directional_available
            if oracle_fallback_used:
                oracle_fresh_ok = oracle_source_policy_ok and (
                    oracle_fresh_available
                    and oracle_directional_available
                    and oracle_age_ms is not None
                    and oracle_age_ms <= max_oracle_age_ms
                )
        else:
            oracle_fresh_ok = (not oracle_fresh_available) or oracle_age_ms is None or oracle_age_ms <= max_oracle_age_ms
            if oracle_fallback_used and not oracle_source_policy_ok:
                oracle_fresh_ok = False

        oracle_policy_detail = (
            f"policy={oracle_source_policy} fallback_used={oracle_fallback_used} "
            f"degraded={oracle_fallback_degraded} reason={oracle_fallback_reason} "
            f"chainlink_age_ms={chainlink_age_ms if chainlink_age_ms is not None else 'n/a'} "
            f"binance_age_ms={binance_age_ms if binance_age_ms is not None else 'n/a'}"
        )
        oracle_reasons_text = ",".join(str(reason) for reason in (oracle_status.get("availability_reasons") or []))
        oracle_freshness_reasons_text = ",".join(str(reason) for reason in (oracle_status.get("freshness_reasons") or []))
        oracle_directional_reasons_text = ",".join(
            str(reason) for reason in (oracle_status.get("directional_reasons") or [])
        )
        if not oracle_reasons_text:
            oracle_reasons_text = "none"
        if not oracle_freshness_reasons_text:
            oracle_freshness_reasons_text = "none"
        if not oracle_directional_reasons_text:
            oracle_directional_reasons_text = "none"
        oracle_freshness_detail = (
            (
                f"fresh_available={oracle_fresh_available} directional_available={oracle_directional_available} "
                f"state={oracle_status.get('availability_state')} fresh_state={oracle_status.get('freshness_state')} "
                f"source={oracle_effective_source or 'unknown'} age_ms={oracle_age_ms:.0f} "
                f"max_ms={max_oracle_age_ms:.0f} required={oracle_required} "
                f"policy={oracle_source_policy} fallback={oracle_fallback_reason} "
                f"reasons={oracle_reasons_text} fresh_reasons={oracle_freshness_reasons_text} "
                f"directional_reasons={oracle_directional_reasons_text}"
            )
            if oracle_age_ms is not None
            else (
                f"fresh_available={oracle_fresh_available} directional_available={oracle_directional_available} "
                f"state={oracle_status.get('availability_state')} fresh_state={oracle_status.get('freshness_state')} "
                f"source={oracle_effective_source or 'unknown'} age=unknown required={oracle_required} "
                f"policy={oracle_source_policy} fallback={oracle_fallback_reason} "
                f"reasons={oracle_reasons_text} fresh_reasons={oracle_freshness_reasons_text} "
                f"directional_reasons={oracle_directional_reasons_text}"
            )
        )

        # --- Regime-aware required thresholds ---
        required_edge = (
            min_edge * _EDGE_MODE_FACTORS.get(regime, {}).get(active_mode, 1.0) * oracle_threshold_edge_multiplier
        )
        required_conf = (
            min_conf
            * _CONF_MODE_FACTORS.get(active_mode, 1.0)
            * _REGIME_CONF_FACTORS.get(regime, 1.0)
            * oracle_threshold_conf_multiplier
        )

        entry_price_for_execution = self._float(
            _first_present(
                getattr(signal, "entry_price", None),
                payload.get("entry_price"),
                payload.get("selected_price"),
                live_market.get("live_selected_price"),
                live_market.get("signal_entry_price"),
            )
        )
        if entry_price_for_execution is None or entry_price_for_execution <= 0.0:
            if direction == "buy_yes":
                entry_price_for_execution = self._float(
                    _first_present(
                        live_market.get("yes_price"),
                        payload.get("up_price"),
                        payload.get("yes_price"),
                    )
                )
            elif direction == "buy_no":
                entry_price_for_execution = self._float(
                    _first_present(
                        live_market.get("no_price"),
                        payload.get("down_price"),
                        payload.get("no_price"),
                    )
                )
        if entry_price_for_execution is not None and entry_price_for_execution > 0.0:
            entry_price_for_execution = clamp(entry_price_for_execution, 0.0, 1.0)
        else:
            entry_price_for_execution = None

        # --- Direction guardrail ---
        guardrail_blocked = False
        guardrail_detail = "disabled"
        if guardrail_enabled:
            guardrail_detail = "guardrail conditions not met"
            if oracle_available and regime in guardrail_regimes:
                if (
                    direction == "buy_no"
                    and model_prob_yes >= guardrail_prob_floor
                    and up_price >= guardrail_price_floor
                ):
                    guardrail_blocked = True
                    guardrail_detail = (
                        f"blocked contrarian buy_no: model_prob_yes={model_prob_yes:.3f} up_price={up_price:.3f}"
                    )
                elif (
                    direction == "buy_yes"
                    and model_prob_no >= guardrail_prob_floor
                    and down_price >= guardrail_price_floor
                ):
                    guardrail_blocked = True
                    guardrail_detail = (
                        f"blocked contrarian buy_yes: model_prob_no={model_prob_no:.3f} down_price={down_price:.3f}"
                    )

        # --- Oracle direction agreement (hard gate, ALL modes/regimes) ---
        # If the oracle says UP (diff > 0), ONLY buy_yes is allowed.
        # If the oracle says DOWN (diff < 0), ONLY buy_no is allowed.
        # Prevents rebalance-in-disguise where edge comes from model-vs-market
        # deviation rather than oracle signal strength.
        oracle_diff_pct_for_gate = to_float(payload.get("oracle_diff_pct"), 0.0)
        oracle_direction_ok = True
        oracle_direction_detail = "no oracle diff data"
        if oracle_available and oracle_diff_pct_for_gate is not None and abs(oracle_diff_pct_for_gate) > 0.01:
            oracle_says_up = oracle_diff_pct_for_gate > 0
            if oracle_says_up and direction == "buy_no":
                oracle_direction_ok = False
                oracle_direction_detail = (
                    f"BLOCKED: oracle says UP (diff={oracle_diff_pct_for_gate:+.4f}%) "
                    f"but direction=buy_no"
                )
            elif not oracle_says_up and direction == "buy_yes":
                oracle_direction_ok = False
                oracle_direction_detail = (
                    f"BLOCKED: oracle says DOWN (diff={oracle_diff_pct_for_gate:+.4f}%) "
                    f"but direction=buy_yes"
                )
            else:
                oracle_direction_detail = (
                    f"oracle agrees: diff={oracle_diff_pct_for_gate:+.4f}% direction={direction}"
                )

        # --- Adaptive edge gating ---
        edge_for_gate = min(edge, mode_edge) if mode_edge > 0.0 else edge
        now_ms = int(time.time() * 1000.0)
        market_id = str(getattr(signal, "market_id", "") or "").strip()
        edge_tracker_key = f"{market_id}|{direction}|{active_mode}" if market_id else ""
        min_edge_persistence_ms = max(
            0,
            int(to_float(params.get("min_edge_persistence_ms", 1400), 1400.0)),
        )
        edge_persistence_elapsed_ms: Optional[int] = None
        edge_persistence_ok = True
        if edge_tracker_key and edge_for_gate >= required_edge and min_edge_persistence_ms > 0:
            first_seen = self._edge_first_seen_ms.get(edge_tracker_key)
            if first_seen is None:
                self._edge_first_seen_ms[edge_tracker_key] = now_ms
                first_seen = now_ms
            edge_persistence_elapsed_ms = max(0, int(now_ms - first_seen))
            edge_persistence_ok = edge_persistence_elapsed_ms >= min_edge_persistence_ms
        elif edge_tracker_key and edge_for_gate < required_edge:
            self._edge_first_seen_ms.pop(edge_tracker_key, None)
        edge_persistence_detail = (
            f"elapsed_ms={edge_persistence_elapsed_ms if edge_persistence_elapsed_ms is not None else 0} "
            f"required_ms={min_edge_persistence_ms}"
            if min_edge_persistence_ms > 0
            else "edge persistence disabled"
        )

        force_disable_reentry_cooldown = to_bool(params.get("force_disable_reentry_cooldown"), False)
        if force_disable_reentry_cooldown:
            reentry_cooldown_seconds = 0.0
        else:
            default_reentry_cooldown_seconds = max(
                0.0,
                to_float(getattr(self, "config", {}).get("reentry_cooldown_seconds_per_market"), 0.0),
            )
            reentry_cooldown_seconds = max(
                0.0,
                to_float(
                    params.get("reentry_cooldown_seconds_per_market", default_reentry_cooldown_seconds),
                    default_reentry_cooldown_seconds,
                ),
            )
        reentry_cooldown_ms = int(reentry_cooldown_seconds * 1000.0)
        reentry_cooldown_elapsed_ms: Optional[int] = None
        reentry_cooldown_ok = True
        if market_id and reentry_cooldown_ms > 0:
            last_selected_at_ms = self._last_selected_at_ms_by_market.get(market_id)
            if last_selected_at_ms is not None:
                reentry_cooldown_elapsed_ms = max(0, int(now_ms - last_selected_at_ms))
                reentry_cooldown_ok = reentry_cooldown_elapsed_ms >= reentry_cooldown_ms
        reentry_cooldown_detail = (
            f"elapsed_ms={reentry_cooldown_elapsed_ms if reentry_cooldown_elapsed_ms is not None else 'none'} "
            f"required_ms={reentry_cooldown_ms}"
            if reentry_cooldown_ms > 0
            else "re-entry cooldown disabled"
        )

        trader_context = context.get("trader")
        risk_limits_context = (
            trader_context.get("risk_limits")
            if isinstance(trader_context, dict) and isinstance(trader_context.get("risk_limits"), dict)
            else {}
        )
        max_trade_notional_hint = self._float(risk_limits_context.get("max_trade_notional_usd"))
        edge_boost_for_checks = 1.0 + max(0.0, edge_for_gate - required_edge) / 30.0
        conf_boost_for_checks = 0.8 + (confidence * 0.8)
        estimated_size_for_checks = (
            base_size
            * _MODE_SIZE_FACTORS.get(active_mode, 1.0)
            * _REGIME_SIZE_FACTORS.get(regime, 1.0)
            * edge_boost_for_checks
            * conf_boost_for_checks
            * oracle_size_multiplier
        )
        estimated_size_for_checks = max(1.0, min(max_size, estimated_size_for_checks))
        if max_trade_notional_hint is not None and max_trade_notional_hint > 0.0:
            estimated_size_for_checks = min(estimated_size_for_checks, max_trade_notional_hint)

        min_order_size_usd = StrategySDK.resolve_min_order_size_usd(
            params,
            mode=context.get("mode"),
            fallback=1.0,
        )
        default_exit_ratio_floor_by_timeframe = {
            "5m": 0.38,
            "15m": 0.34,
            "1h": 0.30,
            "4h": 0.26,
        }
        default_entry_exit_ratio_floor = default_exit_ratio_floor_by_timeframe.get(signal_timeframe, 0.38)
        if active_mode == "maker_quote":
            default_entry_exit_ratio_floor -= 0.08
        elif active_mode == "directional":
            default_entry_exit_ratio_floor -= 0.03
        if regime == "closing":
            default_entry_exit_ratio_floor -= 0.05
        default_entry_exit_ratio_floor = clamp(default_entry_exit_ratio_floor, 0.16, 0.90)
        configured_entry_exit_ratio_floor = self._float(
            _first_present(
                _timeframe_override(params, "entry_executable_exit_ratio_floor", signal_timeframe),
                params.get("entry_executable_exit_ratio_floor_closing") if regime == "closing" else None,
                params.get("entry_executable_exit_ratio_floor"),
            )
        )
        executable_exit_ratio_floor = (
            clamp(configured_entry_exit_ratio_floor, 0.05, 0.95)
            if configured_entry_exit_ratio_floor is not None
            else default_entry_exit_ratio_floor
        )
        required_size_for_exitability = min_order_size_usd / max(0.01, executable_exit_ratio_floor)
        entry_exitability_ok = estimated_size_for_checks + 1e-9 >= required_size_for_exitability
        entry_exitability_detail = (
            f"est_size={estimated_size_for_checks:.2f} required>={required_size_for_exitability:.2f} "
            f"ratio_floor={executable_exit_ratio_floor:.2f}"
        )

        directional_entry_price_floor = max(
            0.0,
            to_float(params.get("directional_min_entry_price_floor", 0.25), 0.25),
        )
        maker_entry_price_floor = max(
            directional_entry_price_floor,
            to_float(params.get("maker_min_entry_price_floor", 0.16), 0.16),
        )
        default_entry_price_floor = (
            maker_entry_price_floor if active_mode == "maker_quote" else directional_entry_price_floor
        )
        if regime == "closing":
            default_entry_price_floor = max(default_entry_price_floor, 0.05)
        entry_price_floor = clamp(
            to_float(params.get("min_entry_price_floor"), default_entry_price_floor),
            0.0,
            0.99,
        )
        entry_price_floor_ok = entry_price_for_execution is None or entry_price_for_execution >= entry_price_floor
        entry_price_floor_detail = (
            f"entry_price={entry_price_for_execution:.4f} floor={entry_price_floor:.4f}"
            if entry_price_for_execution is not None
            else "entry_price unavailable"
        )
        directional_entry_price_ceiling = clamp(
            to_float(params.get("directional_max_entry_price_ceiling", 0.75), 0.75),
            0.01,
            1.0,
        )
        maker_entry_price_ceiling = clamp(
            to_float(params.get("maker_max_entry_price_ceiling", 0.70), 0.70),
            0.01,
            1.0,
        )
        default_entry_price_ceiling = (
            maker_entry_price_ceiling if active_mode == "maker_quote" else directional_entry_price_ceiling
        )
        if regime == "opening":
            default_entry_price_ceiling = min(
                default_entry_price_ceiling,
                0.68 if active_mode == "maker_quote" else 0.72,
            )
        entry_price_ceiling = clamp(
            to_float(params.get("max_entry_price_ceiling"), default_entry_price_ceiling),
            0.01,
            1.0,
        )
        if entry_price_ceiling < entry_price_floor:
            entry_price_ceiling = entry_price_floor
        entry_price_ceiling_ok = entry_price_for_execution is None or entry_price_for_execution <= entry_price_ceiling
        entry_price_ceiling_detail = (
            f"entry_price={entry_price_for_execution:.4f} ceiling={entry_price_ceiling:.4f}"
            if entry_price_for_execution is not None
            else "entry_price unavailable"
        )

        # --- Decision checks ---
        checks = [
            DecisionCheck("source", "Crypto source", source_ok, detail="Requires crypto worker signals."),
            DecisionCheck(
                "signal_origin",
                "Dedicated crypto worker signal",
                origin_ok,
                detail="Legacy scanner crypto opportunities are unsupported.",
            ),
            DecisionCheck(
                "live_window",
                "Current live window only",
                live_window_ok,
                detail=live_window_detail,
            ),
            DecisionCheck(
                "live_context_freshness",
                "Live context freshness",
                live_context_fresh_ok,
                score=live_context_age_seconds,
                detail=live_context_freshness_detail,
            ),
            DecisionCheck(
                "signal_freshness",
                "Signal freshness",
                signal_fresh_ok,
                score=signal_age_seconds,
                detail=signal_freshness_detail,
            ),
            DecisionCheck(
                "market_data_freshness",
                "Market data freshness",
                market_data_fresh_ok,
                score=market_data_age_ms,
                detail=market_data_freshness_detail,
            ),
            DecisionCheck(
                "entry_window",
                "Minimum seconds-left entry window",
                entry_window_ok,
                score=signal_seconds_left if signal_seconds_left >= 0 else None,
                detail=entry_window_detail,
            ),
            DecisionCheck(
                "asset_scope",
                "Asset include/exclude scope",
                asset_scope_ok,
                detail=(
                    f"asset={signal_asset or 'unknown'} "
                    f"include={','.join(include_assets) or 'all'} "
                    f"exclude={','.join(exclude_assets) or 'none'}"
                ),
            ),
            DecisionCheck(
                "timeframe_scope",
                "Cadence include/exclude scope",
                timeframe_scope_ok,
                detail=(
                    f"timeframe={signal_timeframe or 'unknown'} "
                    f"include={','.join(include_timeframes) or 'all'} "
                    f"exclude={','.join(exclude_timeframes) or 'none'}"
                ),
            ),
            DecisionCheck(
                "mode_allowlist",
                "Enabled strategy modes",
                mode_allowlist_ok,
                detail=(
                    f"active_mode={active_mode} "
                    f"enabled={','.join(sorted(enabled_active_modes)) or 'none'}"
                ),
            ),
            DecisionCheck(
                "liquidity",
                "Minimum liquidity",
                liquidity_ok,
                score=signal_liquidity_usd,
                detail=liquidity_detail,
            ),
            DecisionCheck(
                "spread",
                "Maximum spread",
                spread_ok,
                score=signal_spread,
                detail=spread_detail,
            ),
            DecisionCheck(
                "spread_widening",
                "Spread widening guard",
                spread_widening_ok,
                score=spread_widening_bps,
                detail=spread_widening_detail,
            ),
            DecisionCheck(
                "orderbook_imbalance",
                "Orderbook imbalance guard",
                orderbook_imbalance_ok,
                score=orderbook_imbalance,
                detail=orderbook_imbalance_detail,
            ),
            DecisionCheck(
                "recent_move_zscore",
                "Recent move z-score guard",
                recent_move_ok,
                score=recent_move_zscore,
                detail=recent_move_detail,
            ),
            DecisionCheck(
                "oracle_freshness",
                "Oracle readiness",
                oracle_fresh_ok,
                score=oracle_age_ms,
                detail=oracle_freshness_detail,
                payload={
                    "oracle_status": dict(oracle_status),
                    "max_oracle_age_ms": float(max_oracle_age_ms),
                    "max_oracle_age_seconds": float(max_oracle_age_seconds),
                },
            ),
            DecisionCheck(
                "oracle_source_policy",
                "Oracle source policy",
                oracle_source_policy_ok,
                detail=oracle_policy_detail,
                payload={
                    "policy": oracle_source_policy,
                    "fallback_used": bool(oracle_fallback_used),
                    "fallback_degraded": bool(oracle_fallback_degraded),
                },
            ),
            DecisionCheck(
                "strategy_timeframe",
                "Strategy timeframe",
                strategy_timeframe_ok,
                detail=(f"observed={signal_timeframe or 'unknown'} (unified strategy accepts all timeframes)"),
            ),
            DecisionCheck(
                "direction_guardrail",
                "Direction guardrail",
                not guardrail_blocked,
                detail=guardrail_detail,
            ),
            DecisionCheck(
                "direction_policy",
                "Direction policy",
                direction_policy_ok,
                detail=direction_policy_detail,
            ),
            DecisionCheck(
                "oracle_direction_agreement",
                "Oracle direction agreement",
                oracle_direction_ok,
                detail=oracle_direction_detail,
            ),
            DecisionCheck(
                "edge",
                "Edge threshold",
                edge_for_gate >= required_edge,
                score=edge_for_gate,
                detail=f"mode={active_mode} regime={regime} min={required_edge:.2f}",
            ),
            DecisionCheck(
                "edge_persistence",
                "Edge persistence",
                edge_persistence_ok,
                score=edge_persistence_elapsed_ms,
                detail=edge_persistence_detail,
            ),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= required_conf,
                score=confidence,
                detail=f"min={required_conf:.2f}",
            ),
            DecisionCheck(
                "entry_price_floor",
                "Entry price floor",
                entry_price_floor_ok,
                score=entry_price_for_execution,
                detail=entry_price_floor_detail,
            ),
            DecisionCheck(
                "entry_price_ceiling",
                "Entry price ceiling",
                entry_price_ceiling_ok,
                score=entry_price_for_execution,
                detail=entry_price_ceiling_detail,
            ),
            DecisionCheck(
                "entry_exitability",
                "Entry exitability under stress",
                entry_exitability_ok,
                score=estimated_size_for_checks,
                detail=entry_exitability_detail,
            ),
            DecisionCheck(
                "reentry_cooldown",
                "Re-entry cooldown",
                reentry_cooldown_ok,
                score=reentry_cooldown_elapsed_ms,
                detail=reentry_cooldown_detail,
            ),
            DecisionCheck(
                "execution_edge",
                "Execution-adjusted edge",
                net_edge > 0.0,
                score=net_edge,
                detail="Requires positive post-penalty edge.",
            ),
        ]

        if requested_mode != "auto":
            checks.append(
                DecisionCheck(
                    "mode_signal",
                    "Requested strategy mode has signal",
                    mode_edge > 0.0,
                    score=mode_edge,
                    detail=f"requested={requested_mode}",
                )
            )

        failed_check_keys = [
            str(getattr(check, "key", "") or "").strip()
            for check in checks
            if not bool(getattr(check, "passed", False))
        ]
        failed_check_keys = [key for key in failed_check_keys if key]
        missed_opportunity_size_usd = max(0.0, float(estimated_size_for_checks))
        missed_opportunity_edge_percent = max(0.0, float(net_edge))
        missed_opportunity_total_pnl_usd = (missed_opportunity_size_usd * missed_opportunity_edge_percent) / 100.0
        missed_opportunity_by_check: dict[str, float] = {}
        if failed_check_keys and missed_opportunity_total_pnl_usd > 0.0:
            per_check_pnl = missed_opportunity_total_pnl_usd / float(len(failed_check_keys))
            for check_key in failed_check_keys:
                missed_opportunity_by_check[check_key] = round(per_check_pnl, 6)
        skip_by_check_counter = {check_key: 1 for check_key in failed_check_keys}
        age_present_but_unavailable = int(oracle_status.get("availability_state") == "age_present_but_unavailable")

        # --- Build shared payload dict ---
        decision_payload: dict[str, Any] = {
            "requested_mode": requested_mode,
            "active_mode": active_mode,
            "dominant_mode": dominant_mode,
            "regime": regime,
            "edge": edge,
            "mode_edge": mode_edge,
            "net_edge": net_edge,
            "confidence": confidence,
            "required_edge": required_edge,
            "required_confidence": required_conf,
            "asset": signal_asset,
            "timeframe": signal_timeframe,
            "live_window_required": live_window_required,
            "is_live": signal_is_live,
            "is_current": signal_is_current,
            "seconds_left": signal_seconds_left if signal_seconds_left >= 0 else None,
            "end_time": signal_end_time or None,
            "live_context_age_seconds": live_context_age_seconds,
            "max_live_context_age_seconds": float(max_live_context_age_seconds),
            "min_seconds_left_for_entry": float(min_seconds_left_for_entry),
            "signal_age_seconds": signal_age_seconds,
            "max_signal_age_seconds": float(max_signal_age_seconds),
            "market_data_age_ms": market_data_age_ms,
            "max_market_data_age_ms": float(max_market_data_age_ms),
            "enforce_market_data_freshness": bool(market_data_freshness_enforced),
            "require_market_data_age_for_sources": sorted(require_market_data_age_for_sources),
            "signal_timestamp_used": _to_iso_utc(signal_timestamp_used),
            "signal_created_at": _to_iso_utc(signal_created_at),
            "signal_updated_at": _to_iso_utc(signal_updated_at),
            "liquidity_usd": signal_liquidity_usd,
            "min_liquidity_usd": float(min_liquidity_usd),
            "volume_usd": signal_volume_usd,
            "spread": signal_spread,
            "max_spread_pct": float(max_spread_pct),
            "spread_widening_bps": spread_widening_bps,
            "max_spread_widening_bps": float(max_spread_widening_bps),
            "orderbook_imbalance": orderbook_imbalance,
            "max_orderbook_imbalance": float(max_orderbook_imbalance),
            "move_5m_percent": move_5m_pct,
            "move_30m_percent": move_30m_pct,
            "move_2h_percent": move_2h_pct,
            "recent_move_zscore": recent_move_zscore,
            "max_recent_move_zscore_for_entry": float(max_recent_move_zscore_for_entry),
            "oracle_age_seconds": oracle_age_seconds,
            "oracle_age_ms": oracle_age_ms,
            "max_oracle_age_seconds": float(max_oracle_age_seconds),
            "max_oracle_age_ms": float(max_oracle_age_ms),
            "require_oracle_for_directional": bool(require_oracle_for_directional),
            "oracle_status": dict(oracle_status),
            "oracle_source_policy": oracle_source_policy,
            "oracle_fallback_used": bool(oracle_fallback_used),
            "oracle_fallback_degraded": bool(oracle_fallback_degraded),
            "oracle_fallback_reason": oracle_fallback_reason,
            "oracle_effective_source": oracle_effective_source,
            "oracle_fresh_available": bool(oracle_fresh_available),
            "oracle_directional_available": bool(oracle_directional_available),
            "oracle_policy_detail": oracle_policy_detail,
            "oracle_chainlink_age_ms": chainlink_age_ms,
            "oracle_binance_age_ms": binance_age_ms,
            "oracle_threshold_edge_multiplier": float(oracle_threshold_edge_multiplier),
            "oracle_threshold_confidence_multiplier": float(oracle_threshold_conf_multiplier),
            "oracle_size_multiplier": float(oracle_size_multiplier),
            "edge_persistence_elapsed_ms": edge_persistence_elapsed_ms,
            "min_edge_persistence_ms": int(min_edge_persistence_ms),
            "force_disable_reentry_cooldown": bool(force_disable_reentry_cooldown),
            "reentry_cooldown_elapsed_ms": reentry_cooldown_elapsed_ms,
            "reentry_cooldown_seconds_per_market": float(reentry_cooldown_seconds),
            "entry_price": entry_price_for_execution,
            "entry_price_floor": float(entry_price_floor),
            "entry_price_ceiling": float(entry_price_ceiling),
            "entry_exitability_ratio_floor": float(executable_exit_ratio_floor),
            "entry_exitability_required_size_usd": float(required_size_for_exitability),
            "entry_exitability_estimated_size_usd": float(estimated_size_for_checks),
            "entry_exitability_min_order_size_usd": float(min_order_size_usd),
            "max_trade_notional_hint_usd": max_trade_notional_hint,
            "direction_guardrail": {
                "enabled": guardrail_enabled,
                "blocked": guardrail_blocked,
                "oracle_available": oracle_available,
                "oracle_fresh_available": bool(oracle_fresh_available),
                "oracle_directional_available": bool(oracle_directional_available),
                "regime": regime,
                "regimes": sorted(guardrail_regimes),
                "prob_floor": guardrail_prob_floor,
                "price_floor": guardrail_price_floor,
                "model_prob_yes": model_prob_yes,
                "model_prob_no": model_prob_no,
                "up_price": up_price,
                "down_price": down_price,
            },
            "freshness_clocks": {
                "market_microprice": {
                    "age_ms": market_data_age_ms,
                    "max_age_ms": float(max_market_data_age_ms),
                    "required": bool(require_market_data_age),
                    "enforced": bool(market_data_freshness_enforced),
                    "passed": bool(market_data_fresh_ok),
                },
                "oracle_reference": {
                    "age_ms": oracle_age_ms,
                    "age_seconds": oracle_age_seconds,
                    "max_age_ms": float(max_oracle_age_ms),
                    "max_age_seconds": float(max_oracle_age_seconds),
                    "required": bool(oracle_required),
                    "passed": bool(oracle_fresh_ok),
                    "source": oracle_effective_source,
                    "availability_state": oracle_status.get("availability_state"),
                    "fresh_available": bool(oracle_fresh_available),
                    "directional_available": bool(oracle_directional_available),
                },
                "signal_context": {
                    "signal_age_seconds": signal_age_seconds,
                    "max_signal_age_seconds": float(max_signal_age_seconds),
                    "signal_passed": bool(signal_fresh_ok),
                    "live_context_age_seconds": live_context_age_seconds,
                    "max_live_context_age_seconds": float(max_live_context_age_seconds),
                    "live_context_passed": bool(live_context_fresh_ok),
                },
            },
            "telemetry_counters": {
                "age_present_but_unavailable": age_present_but_unavailable,
                "skip_by_check": skip_by_check_counter,
                "missed_opportunity": {
                    "estimated_edge_percent": round(missed_opportunity_edge_percent, 6),
                    "estimated_size_usd": round(missed_opportunity_size_usd, 6),
                    "estimated_pnl_usd_total": round(missed_opportunity_total_pnl_usd, 6),
                    "attribution_by_check": missed_opportunity_by_check,
                },
            },
            "direction_policy": {
                "allowed": bool(direction_policy_ok),
                "detail": direction_policy_detail,
            },
            "include_assets": include_assets,
            "exclude_assets": exclude_assets,
            "include_timeframes": include_timeframes,
            "exclude_timeframes": exclude_timeframes,
            "live_market_context": {
                "available": bool(live_market.get("available")),
                "fetched_at": live_market.get("fetched_at"),
                "selected_token_id": live_market.get("selected_token_id"),
                "market_end_time": live_market.get("market_end_time"),
                "seconds_left": live_market.get("seconds_left"),
                "is_live": live_market.get("is_live"),
                "is_current": live_market.get("is_current"),
            },
            "decision_snapshot": {
                "signal": {
                    "id": str(getattr(signal, "id", "") or ""),
                    "source": str(getattr(signal, "source", "") or ""),
                    "signal_type": str(getattr(signal, "signal_type", "") or ""),
                    "market_id": str(getattr(signal, "market_id", "") or ""),
                    "direction": str(getattr(signal, "direction", "") or ""),
                    "entry_price": self._float(getattr(signal, "entry_price", None)),
                    "edge_percent": self._float(getattr(signal, "edge_percent", None)),
                    "confidence": self._float(getattr(signal, "confidence", None)),
                    "created_at": _to_iso_utc(signal_created_at),
                    "updated_at": _to_iso_utc(signal_updated_at),
                },
                "params": _json_safe(params),
                "payload": _json_safe(payload),
                "live_market": _json_safe(live_market),
                "oracle_status": _json_safe(oracle_status),
            },
        }

        score = (edge_for_gate * 0.7) + (confidence * 30.0)

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Crypto worker filters not met",
                score=score,
                checks=checks,
                payload=decision_payload,
            )

        # --- Position sizing ---
        # Historical data shows edge calibration is inverted: higher reported
        # edge correlates with worse outcomes. Use a conservative, capped
        # edge boost that penalises suspiciously large edges.
        edge_excess = max(0.0, edge_for_gate - required_edge)
        if edge_excess > 15.0:
            # Suspiciously large edge — size DOWN (inverted calibration)
            edge_boost = max(0.5, 1.0 - (edge_excess - 15.0) / 60.0)
        else:
            # Moderate edge — small linear boost, capped
            edge_boost = 1.0 + min(edge_excess / 50.0, 0.3)
        conf_boost = 0.8 + (confidence * 0.8)
        size = (
            base_size
            * _MODE_SIZE_FACTORS.get(active_mode, 1.0)
            * _REGIME_SIZE_FACTORS.get(regime, 1.0)
            * edge_boost
            * conf_boost
            * oracle_size_multiplier
        )
        size = max(1.0, min(max_size, size))
        if market_id:
            self._last_selected_at_ms_by_market[market_id] = now_ms
        if edge_tracker_key:
            self._edge_first_seen_ms.pop(edge_tracker_key, None)

        return StrategyDecision(
            decision="selected",
            reason=f"Crypto {active_mode} setup validated ({regime})",
            score=score,
            size_usd=size,
            checks=checks,
            payload=decision_payload,
        )

    def _evaluate_local_exit(self, position: Any, market_state: dict) -> dict[str, Any]:
        state = market_state if isinstance(market_state, dict) else {}
        base_config = getattr(position, "config", None)
        config = dict(base_config) if isinstance(base_config, dict) else {}
        defaults = crypto_highfreq_scope_defaults()

        strategy_context = getattr(position, "strategy_context", None)
        if isinstance(strategy_context, dict):
            context_payload = strategy_context
        else:
            context_payload = {}
            try:
                setattr(position, "strategy_context", context_payload)
            except Exception:
                pass

        timeframe = _normalize_timeframe(
            context_payload.get("timeframe") or context_payload.get("cadence") or context_payload.get("interval")
        )

        default_stop_loss_activation_by_timeframe = {
            "5m": 45.0,
            "15m": 120.0,
            "1h": 300.0,
            "4h": 900.0,
        }
        default_trailing_activation_profit_by_timeframe = {
            "5m": 4.0,
            "15m": 6.0,
            "1h": 8.0,
            "4h": 10.0,
        }
        default_min_hold_by_timeframe = {
            "5m": 0.25,
            "15m": 0.5,
            "1h": 1.0,
            "4h": 2.0,
        }
        default_max_hold_by_timeframe = {
            "5m": 15.0,
            "15m": 45.0,
            "1h": 120.0,
            "4h": 360.0,
        }
        default_rapid_exit_window_by_timeframe = {
            "5m": 1.0,
            "15m": 2.0,
            "1h": 6.0,
            "4h": 15.0,
        }
        default_underwater_dwell_minutes_by_timeframe = {
            "5m": 0.75,
            "15m": 2.0,
            "1h": 6.0,
            "4h": 18.0,
        }
        default_underwater_recovery_ratio_by_timeframe = {
            "5m": 0.30,
            "15m": 0.35,
            "1h": 0.40,
            "4h": 0.45,
        }
        default_underwater_rebound_pct_by_timeframe = {
            "5m": 0.8,
            "15m": 1.2,
            "1h": 1.8,
            "4h": 2.4,
        }
        default_underwater_exit_fade_pct_by_timeframe = {
            "5m": 0.35,
            "15m": 0.45,
            "1h": 0.6,
            "4h": 0.8,
        }
        default_force_flatten_seconds_left_by_timeframe = {
            "5m": to_float(defaults.get("force_flatten_seconds_left_5m"), 30.0),
            "15m": to_float(defaults.get("force_flatten_seconds_left_15m"), 75.0),
            "1h": to_float(defaults.get("force_flatten_seconds_left_1h"), 240.0),
            "4h": to_float(defaults.get("force_flatten_seconds_left_4h"), 600.0),
        }
        default_underwater_timeout_minutes_by_timeframe = {
            "5m": 2.0,
            "15m": 6.0,
            "1h": 18.0,
            "4h": 45.0,
        }
        default_executable_time_pressure_seconds_by_timeframe = {
            "5m": 40.0,
            "15m": 120.0,
            "1h": 300.0,
            "4h": 900.0,
        }

        config.setdefault("rapid_take_profit_pct", defaults.get("rapid_take_profit_pct", 10.0))
        config.setdefault("take_profit_pct", defaults.get("take_profit_pct", 8.0))
        config.setdefault("stop_loss_pct", to_float(defaults.get("stop_loss_pct"), 4.0))
        default_stop_loss_policy = str(defaults.get("stop_loss_policy") or "always").strip().lower()
        if default_stop_loss_policy in {"near_close", "close_window"}:
            default_stop_loss_policy = "near_close_only"
        if default_stop_loss_policy not in {"always", "near_close_only", "volatility_adaptive"}:
            default_stop_loss_policy = "always"
        config.setdefault("stop_loss_policy", default_stop_loss_policy)
        config.setdefault(
            "stop_loss_activation_seconds",
            default_stop_loss_activation_by_timeframe.get(
                timeframe,
                to_float(defaults.get("stop_loss_activation_seconds"), 90.0),
            ),
        )
        config.setdefault("trailing_stop_pct", defaults.get("trailing_stop_pct", 3.0))
        config.setdefault(
            "trailing_stop_activation_profit_pct",
            default_trailing_activation_profit_by_timeframe.get(
                timeframe,
                to_float(defaults.get("trailing_stop_activation_profit_pct"), 4.0),
            ),
        )
        config.setdefault(
            "min_hold_minutes",
            default_min_hold_by_timeframe.get(timeframe, to_float(defaults.get("min_hold_minutes"), 1.0)),
        )
        config.setdefault(
            "max_hold_minutes",
            default_max_hold_by_timeframe.get(timeframe, to_float(defaults.get("max_hold_minutes"), 60.0)),
        )
        config.setdefault(
            "rapid_exit_window_minutes",
            default_rapid_exit_window_by_timeframe.get(
                timeframe,
                to_float(defaults.get("rapid_exit_window_minutes"), 2.0),
            ),
        )
        config.setdefault("rapid_exit_min_increase_pct", defaults.get("rapid_exit_min_increase_pct", 0.0))
        config.setdefault("rapid_exit_breakeven_buffer_pct", defaults.get("rapid_exit_breakeven_buffer_pct", 0.0))
        config.setdefault(
            "underwater_rebound_exit_enabled",
            to_bool(defaults.get("underwater_rebound_exit_enabled"), True),
        )
        config.setdefault(
            "underwater_dwell_minutes",
            default_underwater_dwell_minutes_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_dwell_minutes"), 2.5),
            ),
        )
        config.setdefault(
            "underwater_recovery_ratio_min",
            default_underwater_recovery_ratio_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_recovery_ratio_min"), 0.35),
            ),
        )
        config.setdefault(
            "underwater_rebound_pct_min",
            default_underwater_rebound_pct_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_rebound_pct_min"), 1.2),
            ),
        )
        config.setdefault(
            "underwater_exit_fade_pct",
            default_underwater_exit_fade_pct_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_exit_fade_pct"), 0.45),
            ),
        )
        config.setdefault(
            "underwater_timeout_minutes",
            default_underwater_timeout_minutes_by_timeframe.get(
                timeframe,
                to_float(defaults.get("underwater_timeout_minutes"), 10.0),
            ),
        )
        config.setdefault("underwater_timeout_loss_pct", defaults.get("underwater_timeout_loss_pct", 8.0))
        config.setdefault(
            "force_flatten_seconds_left",
            default_force_flatten_seconds_left_by_timeframe.get(
                timeframe,
                to_float(defaults.get("force_flatten_seconds_left"), 120.0),
            ),
        )
        config.setdefault("force_flatten_max_profit_pct", to_float(defaults.get("force_flatten_max_profit_pct"), 1.0))
        config.setdefault("force_flatten_headroom_floor", to_float(defaults.get("force_flatten_headroom_floor"), 1.15))
        config.setdefault("force_flatten_min_loss_pct", to_float(defaults.get("force_flatten_min_loss_pct"), 0.0))
        config.setdefault(
            "resolution_risk_flatten_enabled", to_bool(defaults.get("resolution_risk_flatten_enabled"), True)
        )
        config.setdefault("resolution_risk_seconds_left", to_float(defaults.get("resolution_risk_seconds_left"), 180.0))
        config.setdefault(
            "resolution_risk_max_profit_pct", to_float(defaults.get("resolution_risk_max_profit_pct"), 6.0)
        )
        config.setdefault("resolution_risk_min_loss_pct", to_float(defaults.get("resolution_risk_min_loss_pct"), 2.0))
        config.setdefault(
            "resolution_risk_min_headroom_ratio",
            to_float(defaults.get("resolution_risk_min_headroom_ratio"), 0.9),
        )
        config.setdefault(
            "resolution_risk_disable_when_take_profit_armed",
            to_bool(defaults.get("resolution_risk_disable_when_take_profit_armed"), True),
        )
        config.setdefault("executable_exit_headroom_urgent", 1.35)
        config.setdefault("executable_exit_headroom_warn", 2.0)
        config.setdefault("executable_exit_hazard_threshold", 0.62)
        config.setdefault(
            "executable_exit_time_pressure_seconds",
            default_executable_time_pressure_seconds_by_timeframe.get(timeframe, 120.0),
        )

        if timeframe:
            for key in (
                "rapid_take_profit_pct",
                "rapid_exit_window_minutes",
                "take_profit_pct",
                "stop_loss_pct",
                "stop_loss_policy",
                "stop_loss_activation_seconds",
                "trailing_stop_pct",
                "trailing_stop_activation_profit_pct",
                "min_hold_minutes",
                "max_hold_minutes",
                "underwater_dwell_minutes",
                "underwater_recovery_ratio_min",
                "underwater_rebound_pct_min",
                "underwater_exit_fade_pct",
                "underwater_timeout_minutes",
                "underwater_timeout_loss_pct",
                "force_flatten_seconds_left",
                "force_flatten_max_profit_pct",
                "force_flatten_min_loss_pct",
                "resolution_risk_seconds_left",
                "resolution_risk_max_profit_pct",
                "resolution_risk_min_loss_pct",
                "resolution_risk_min_headroom_ratio",
                "executable_exit_headroom_urgent",
                "executable_exit_headroom_warn",
                "executable_exit_hazard_threshold",
                "executable_exit_time_pressure_seconds",
            ):
                override = _timeframe_override(config, key, timeframe)
                if override is not None:
                    config[key] = override
            config["timeframe"] = timeframe

        entry_price = self._float(getattr(position, "entry_price", None))
        current_price = self._float(state.get("current_price"))
        if entry_price is not None and entry_price > 0.0 and current_price is not None and current_price > 0.0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            # --- IMMEDIATE stop-loss (always active, regardless of policy) ---
            immediate_stop_loss_pct = max(0.5, to_float(config.get("immediate_stop_loss_pct"), 2.0))
            immediate_stop_loss_pct_override = _timeframe_override(config, "immediate_stop_loss_pct", timeframe)
            if immediate_stop_loss_pct_override is not None:
                immediate_stop_loss_pct = max(0.5, to_float(immediate_stop_loss_pct_override, 2.0))
            if pnl_pct <= -immediate_stop_loss_pct:
                context_payload["_crypto_hf_rapid_exit_state"] = context_payload.get("_crypto_hf_rapid_exit_state", {})
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": (
                            f"Immediate stop-loss triggered "
                            f"(pnl={pnl_pct:.2f}% <= -{immediate_stop_loss_pct:.1f}%)"
                        ),
                        "close_price": current_price,
                    },
                }

            take_profit_pct = max(0.1, to_float(config.get("take_profit_pct"), 8.0))
            rapid_take_profit_pct = max(0.1, to_float(config.get("rapid_take_profit_pct"), 10.0))
            rapid_arm_pct = max(
                0.1,
                min(
                    rapid_take_profit_pct,
                    max(take_profit_pct, rapid_take_profit_pct * 0.85),
                ),
            )
            rapid_hard_cap_pct = max(rapid_take_profit_pct + 6.0, rapid_take_profit_pct * 1.8)
            winner_target_profit_pct = max(rapid_hard_cap_pct + 8.0, rapid_take_profit_pct * 3.0, 30.0)
            winner_target_override = _timeframe_override(config, "winner_ride_target_profit_pct", timeframe)
            if winner_target_override is None:
                winner_target_override = config.get("winner_ride_target_profit_pct")
            winner_target_profit_pct = max(
                rapid_arm_pct + 1.0,
                to_float(winner_target_override, winner_target_profit_pct),
            )
            winner_take_profit_floor_pct = max(winner_target_profit_pct + 6.0, rapid_hard_cap_pct + 10.0)
            winner_take_profit_floor_override = _timeframe_override(
                config,
                "winner_ride_take_profit_floor_pct",
                timeframe,
            )
            if winner_take_profit_floor_override is None:
                winner_take_profit_floor_override = config.get("winner_ride_take_profit_floor_pct")
            winner_take_profit_floor_pct = max(
                winner_target_profit_pct + 1.0,
                to_float(winner_take_profit_floor_override, winner_take_profit_floor_pct),
            )
            winner_peak_buffer_pct_override = _timeframe_override(config, "winner_ride_peak_buffer_pct", timeframe)
            if winner_peak_buffer_pct_override is None:
                winner_peak_buffer_pct_override = config.get("winner_ride_peak_buffer_pct")
            winner_peak_buffer_pct = clamp(to_float(winner_peak_buffer_pct_override, 0.65), 0.25, 2.5)
            trailing_stop_pct = max(0.5, to_float(config.get("trailing_stop_pct"), 3.0))

            timeframe_backside_factor = {
                "5m": 0.85,
                "15m": 1.0,
                "1h": 1.2,
                "4h": 1.35,
            }.get(timeframe, 1.0)
            base_backside_pct = clamp(
                trailing_stop_pct * 0.58 * timeframe_backside_factor,
                0.45,
                max(0.9, trailing_stop_pct * 0.98),
            )

            rapid_window_minutes = max(0.0, to_float(config.get("rapid_exit_window_minutes"), 0.0))
            min_increase_pct = max(0.0, to_float(config.get("rapid_exit_min_increase_pct"), 0.0))
            breakeven_buffer_pct = max(0.0, to_float(config.get("rapid_exit_breakeven_buffer_pct"), 0.0))

            rapid_state_key = "_crypto_hf_rapid_exit_state"
            raw_rapid_state = context_payload.get(rapid_state_key)
            rapid_state = dict(raw_rapid_state) if isinstance(raw_rapid_state, dict) else {}
            highest_seen = self._float(getattr(position, "highest_price", None))
            tracked_peak = self._float(rapid_state.get("peak_price"))
            peak_price = entry_price
            if highest_seen is not None:
                peak_price = max(peak_price, highest_seen)
            if tracked_peak is not None:
                peak_price = max(peak_price, tracked_peak)
            peak_price = max(peak_price, current_price)
            rapid_state["peak_price"] = peak_price
            rapid_state["peak_pnl_pct"] = ((peak_price - entry_price) / entry_price) * 100.0
            rapid_state["arm_pct"] = rapid_arm_pct
            rapid_state["hard_cap_pct"] = rapid_hard_cap_pct

            age_minutes = self._float(getattr(position, "age_minutes", None))
            evaluated = bool(rapid_state.get("evaluated"))
            if not evaluated and (
                rapid_window_minutes <= 0.0 or (age_minutes is not None and age_minutes >= rapid_window_minutes)
            ):
                required_peak = entry_price * (1.0 + (min_increase_pct / 100.0))
                rapid_state["evaluated"] = True
                rapid_state["stalled"] = peak_price <= (required_peak + 1e-9)
                rapid_state["evaluated_at_minutes"] = age_minutes
                rapid_state["required_peak_price"] = required_peak
                rapid_state["window_minutes"] = rapid_window_minutes

            token_id = str(
                state.get("token_id") or context_payload.get("token_id") or rapid_state.get("token_id") or ""
            ).strip()
            if not token_id:
                execution_plan = context_payload.get("execution_plan")
                if isinstance(execution_plan, dict):
                    legs = execution_plan.get("legs")
                    if isinstance(legs, list):
                        for leg in legs:
                            if not isinstance(leg, dict):
                                continue
                            candidate = str(leg.get("token_id") or "").strip()
                            if candidate:
                                token_id = candidate
                                break

            flow_imbalance = self._float(rapid_state.get("flow_imbalance"))
            momentum_short_pct = self._float(rapid_state.get("momentum_short_pct"))
            recent_range_pct = self._float(rapid_state.get("recent_range_pct"))
            if token_id:
                price_history = StrategySDK.get_price_history(token_id, max_snapshots=24)
                mids: list[float] = []
                for snapshot in price_history:
                    if not isinstance(snapshot, dict):
                        continue
                    mid = self._float(snapshot.get("mid"))
                    if mid is None or mid <= 0.0:
                        continue
                    mids.append(float(mid))
                    if len(mids) >= 18:
                        break
                if len(mids) >= 3:
                    lookback_idx = min(len(mids) - 1, 6)
                    anchor_price = mids[lookback_idx]
                    if anchor_price > 0.0:
                        momentum_short_pct = ((mids[0] - anchor_price) / anchor_price) * 100.0
                    window = mids[: min(len(mids), 8)]
                    window_max = max(window)
                    window_min = min(window)
                    if window_max > 0.0:
                        recent_range_pct = ((window_max - window_min) / window_max) * 100.0
                raw_imbalance = self._float(StrategySDK.get_buy_sell_imbalance(token_id, lookback_seconds=20.0))
                if raw_imbalance is not None:
                    flow_imbalance = clamp(raw_imbalance, -1.0, 1.0)

            dynamic_backside_pct = base_backside_pct
            if recent_range_pct is not None:
                dynamic_backside_pct *= clamp(1.0 + (recent_range_pct / 16.0), 0.85, 1.35)
            if flow_imbalance is not None:
                dynamic_backside_pct *= clamp(1.0 + (flow_imbalance * 0.30), 0.65, 1.35)
            if momentum_short_pct is not None:
                dynamic_backside_pct *= clamp(1.0 + (momentum_short_pct / 5.0), 0.60, 1.25)
            if pnl_pct > rapid_arm_pct:
                winner_progress = clamp(
                    (pnl_pct - rapid_arm_pct) / max(1.0, winner_target_profit_pct - rapid_arm_pct),
                    0.0,
                    1.0,
                )
                dynamic_backside_pct *= clamp(1.0 - (winner_progress * 0.40), 0.40, 1.0)
            dynamic_backside_pct = clamp(
                dynamic_backside_pct,
                0.45,
                max(1.2, trailing_stop_pct * 0.95),
            )

            drawdown_from_peak_pct = 0.0
            if peak_price > 0.0:
                drawdown_from_peak_pct = max(0.0, ((peak_price - current_price) / peak_price) * 100.0)

            min_order_size_usd = max(0.01, to_float(state.get("min_order_size_usd"), 1.0))
            filled_size = self._float(getattr(position, "filled_size", None))
            if filled_size is None or filled_size <= 0.0:
                position_notional = self._float(getattr(position, "notional_usd", None))
                if position_notional is None:
                    position_notional = self._float(state.get("notional_usd"))
                if position_notional is not None and position_notional > 0.0:
                    filled_size = position_notional / entry_price

            min_exit_price = None
            panic_exit_price = None
            exit_notional_estimate = None
            exit_headroom_ratio = None
            if filled_size is not None and filled_size > 0.0:
                min_exit_price = min_order_size_usd / filled_size
                panic_exit_price = min_exit_price * 1.35
                exit_notional_estimate = filled_size * current_price
                if min_order_size_usd > 0.0:
                    exit_headroom_ratio = exit_notional_estimate / min_order_size_usd

            urgent_headroom_ratio = max(1.0, to_float(config.get("executable_exit_headroom_urgent"), 1.35))
            warn_headroom_ratio = max(
                urgent_headroom_ratio + 0.05, to_float(config.get("executable_exit_headroom_warn"), 2.0)
            )
            hazard_threshold = clamp(to_float(config.get("executable_exit_hazard_threshold"), 0.62), 0.0, 1.0)
            time_pressure_seconds = max(1.0, to_float(config.get("executable_exit_time_pressure_seconds"), 120.0))

            seconds_left = self._float(
                _first_present(
                    state.get("seconds_left"),
                    context_payload.get("seconds_left"),
                    context_payload.get("live_market_context", {}).get("seconds_left")
                    if isinstance(context_payload.get("live_market_context"), dict)
                    else None,
                )
            )
            time_pressure = 0.0
            if seconds_left is not None and seconds_left >= 0.0:
                time_pressure = clamp((time_pressure_seconds - seconds_left) / time_pressure_seconds, 0.0, 1.0)
            headroom_stress = 0.0
            if exit_headroom_ratio is not None:
                denom = max(0.05, warn_headroom_ratio - 1.0)
                headroom_stress = clamp((warn_headroom_ratio - exit_headroom_ratio) / denom, 0.0, 1.0)
            flow_neg = clamp(-(flow_imbalance or 0.0), 0.0, 1.0)
            momentum_neg = clamp(-((momentum_short_pct or 0.0) / 2.0), 0.0, 1.0)
            hazard_score = clamp(
                (headroom_stress * 0.55) + (flow_neg * 0.25) + (momentum_neg * 0.15) + (time_pressure * 0.05),
                0.0,
                1.0,
            )

            if exit_headroom_ratio is not None:
                dynamic_backside_pct *= clamp(exit_headroom_ratio / 3.0, 0.30, 1.0)
                dynamic_backside_pct = clamp(dynamic_backside_pct, 0.30, max(1.2, trailing_stop_pct * 0.95))

            rapid_state["token_id"] = token_id or None
            rapid_state["flow_imbalance"] = flow_imbalance
            rapid_state["momentum_short_pct"] = momentum_short_pct
            rapid_state["recent_range_pct"] = recent_range_pct
            rapid_state["dynamic_backside_pct"] = dynamic_backside_pct
            rapid_state["drawdown_from_peak_pct"] = drawdown_from_peak_pct
            rapid_state["min_exit_price"] = min_exit_price
            rapid_state["panic_exit_price"] = panic_exit_price
            rapid_state["exit_notional_estimate"] = exit_notional_estimate
            rapid_state["exit_headroom_ratio"] = exit_headroom_ratio
            rapid_state["executable_hazard_score"] = hazard_score
            rapid_state["time_pressure"] = time_pressure
            rapid_state["winner_target_profit_pct"] = winner_target_profit_pct
            rapid_state["winner_take_profit_floor_pct"] = winner_take_profit_floor_pct
            rapid_state["winner_peak_buffer_pct"] = winner_peak_buffer_pct

            if exit_headroom_ratio is not None and exit_headroom_ratio <= urgent_headroom_ratio and pnl_pct <= 0.5:
                context_payload[rapid_state_key] = rapid_state
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": (
                            "Executable-notional guard (headroom emergency) "
                            f"(headroom={exit_headroom_ratio:.2f}x <= {urgent_headroom_ratio:.2f}x)"
                        ),
                        "close_price": current_price,
                    },
                }

            if panic_exit_price is not None and current_price <= panic_exit_price and pnl_pct <= 0.0:
                context_payload[rapid_state_key] = rapid_state
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": (
                            "Executable-notional guard "
                            f"({current_price:.4f} <= {panic_exit_price:.4f}, min_exit={min_exit_price:.4f})"
                        ),
                        "close_price": current_price,
                    },
                }

            if (
                exit_headroom_ratio is not None
                and hazard_score >= hazard_threshold
                and pnl_pct <= max(1.0, rapid_take_profit_pct * 0.20)
            ):
                context_payload[rapid_state_key] = rapid_state
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": (
                            "Executable hazard exit "
                            f"(hazard={hazard_score:.2f} >= {hazard_threshold:.2f}, "
                            f"headroom={exit_headroom_ratio:.2f}x)"
                        ),
                        "close_price": current_price,
                    },
                }

            if pnl_pct >= rapid_arm_pct:
                rapid_state["armed"] = True
                if rapid_state.get("armed_at_minutes") is None:
                    rapid_state["armed_at_minutes"] = age_minutes
                if rapid_state.get("armed_price") is None:
                    rapid_state["armed_price"] = current_price
                rapid_state["armed_peak_price"] = peak_price

            should_flatten_resolution_risk, resolution_risk_detail = crypto_highfreq_should_flatten_resolution_risk(
                config,
                timeframe=timeframe,
                seconds_left=seconds_left,
                pnl_percent=pnl_pct,
                exit_headroom_ratio=exit_headroom_ratio,
                take_profit_armed=bool(rapid_state.get("armed")),
            )
            if should_flatten_resolution_risk:
                context_payload[rapid_state_key] = rapid_state
                return {
                    "config": config,
                    "decision": {
                        "action": "close",
                        "reason": f"Resolution-risk flatten ({resolution_risk_detail})",
                        "close_price": current_price,
                    },
                }

            if bool(rapid_state.get("armed")):
                config["take_profit_pct"] = max(take_profit_pct, winner_take_profit_floor_pct)
                if pnl_pct >= winner_target_profit_pct:
                    rapid_state["winner_target_hit"] = True
                    rapid_state["winner_target_hit_pnl_pct"] = pnl_pct
                if bool(rapid_state.get("winner_target_hit")) and drawdown_from_peak_pct >= winner_peak_buffer_pct:
                    context_payload[rapid_state_key] = rapid_state
                    return {
                        "config": config,
                        "decision": {
                            "action": "close",
                            "reason": (
                                "Winner ride peak-lock exit "
                                f"(drawdown={drawdown_from_peak_pct:.2f}% >= {winner_peak_buffer_pct:.2f}%, "
                                f"target={winner_target_profit_pct:.1f}%)"
                            ),
                            "close_price": current_price,
                        },
                    }

                reversal_signal = drawdown_from_peak_pct >= (dynamic_backside_pct * 0.55) and (
                    (momentum_short_pct is not None and momentum_short_pct <= -0.25)
                    or (flow_imbalance is not None and flow_imbalance <= -0.2)
                )
                if drawdown_from_peak_pct >= dynamic_backside_pct or reversal_signal:
                    context_payload[rapid_state_key] = rapid_state
                    return {
                        "config": config,
                        "decision": {
                            "action": "close",
                            "reason": (
                                "Adaptive backside peak exit "
                                f"(drawdown={drawdown_from_peak_pct:.2f}% "
                                f"threshold={dynamic_backside_pct:.2f}%, "
                                f"flow={flow_imbalance if flow_imbalance is not None else 0.0:.2f}, "
                                f"momentum={momentum_short_pct if momentum_short_pct is not None else 0.0:.2f}%)"
                            ),
                            "close_price": current_price,
                        },
                    }

            context_payload[rapid_state_key] = rapid_state
            if bool(rapid_state.get("stalled")):
                breakeven_floor = entry_price * (1.0 + (breakeven_buffer_pct / 100.0))
                if current_price >= (breakeven_floor - 1e-9):
                    return {
                        "config": config,
                        "decision": {
                            "action": "close",
                            "reason": (
                                "Rapid window stalled without upside; exiting at breakeven+ "
                                f"({current_price:.4f} >= {breakeven_floor:.4f})"
                            ),
                            "close_price": current_price,
                        },
                    }

            underwater_state_key = "_crypto_hf_loss_recovery_state"
            if to_bool(config.get("underwater_rebound_exit_enabled"), True):
                raw_underwater_state = context_payload.get(underwater_state_key)
                underwater_state = dict(raw_underwater_state) if isinstance(raw_underwater_state, dict) else {}
                underwater_dwell_minutes = max(0.0, to_float(config.get("underwater_dwell_minutes"), 0.0))
                underwater_recovery_ratio_min = clamp(
                    to_float(config.get("underwater_recovery_ratio_min"), 0.35),
                    0.0,
                    1.0,
                )
                underwater_rebound_pct_min = max(0.0, to_float(config.get("underwater_rebound_pct_min"), 1.2))
                underwater_exit_fade_pct = max(0.0, to_float(config.get("underwater_exit_fade_pct"), 0.45))
                underwater_timeout_minutes = max(0.0, to_float(config.get("underwater_timeout_minutes"), 0.0))
                underwater_timeout_loss_pct = max(0.0, to_float(config.get("underwater_timeout_loss_pct"), 0.0))

                if current_price < (entry_price - 1e-9):
                    lowest_seen = self._float(getattr(position, "lowest_price", None))
                    prior_trough = self._float(underwater_state.get("trough_price"))
                    trough_price = current_price
                    if lowest_seen is not None and lowest_seen > 0.0:
                        trough_price = min(trough_price, lowest_seen)
                    if prior_trough is not None and prior_trough > 0.0:
                        trough_price = min(trough_price, prior_trough)

                    underwater_since_minutes = self._float(underwater_state.get("underwater_since_age_minutes"))
                    if underwater_since_minutes is None and age_minutes is not None:
                        underwater_since_minutes = age_minutes
                    if (
                        age_minutes is not None
                        and underwater_since_minutes is not None
                        and underwater_since_minutes > age_minutes
                    ):
                        underwater_since_minutes = age_minutes

                    dwell_elapsed_minutes = None
                    if age_minutes is not None and underwater_since_minutes is not None:
                        dwell_elapsed_minutes = max(0.0, age_minutes - underwater_since_minutes)

                    rebound_distance = max(0.0, current_price - trough_price)
                    drawdown_distance = max(0.0, entry_price - trough_price)
                    recovery_ratio = 0.0
                    if drawdown_distance > 0.0:
                        recovery_ratio = clamp(rebound_distance / drawdown_distance, 0.0, 2.5)
                    rebound_pct = 0.0
                    if trough_price > 0.0:
                        rebound_pct = max(0.0, ((current_price - trough_price) / trough_price) * 100.0)

                    rebound_peak_price = current_price
                    previous_rebound_peak = self._float(underwater_state.get("rebound_peak_price"))
                    if previous_rebound_peak is not None and previous_rebound_peak > 0.0:
                        rebound_peak_price = max(rebound_peak_price, previous_rebound_peak)
                    fade_from_rebound_peak_pct = 0.0
                    if rebound_peak_price > 0.0:
                        fade_from_rebound_peak_pct = max(
                            0.0,
                            ((rebound_peak_price - current_price) / rebound_peak_price) * 100.0,
                        )

                    reversal_pressure = (
                        (momentum_short_pct is not None and momentum_short_pct <= -0.20)
                        or (flow_imbalance is not None and flow_imbalance <= -0.15)
                        or (drawdown_from_peak_pct >= max(0.30, dynamic_backside_pct * 0.45))
                    )
                    rebound_fading = fade_from_rebound_peak_pct >= underwater_exit_fade_pct
                    soft_fade = fade_from_rebound_peak_pct >= (underwater_exit_fade_pct * 0.60)
                    dwell_ready = (
                        dwell_elapsed_minutes is not None and dwell_elapsed_minutes >= underwater_dwell_minutes
                    )
                    recovery_ready = (
                        rebound_pct >= underwater_rebound_pct_min and recovery_ratio >= underwater_recovery_ratio_min
                    )

                    underwater_state["underwater_since_age_minutes"] = underwater_since_minutes
                    underwater_state["dwell_elapsed_minutes"] = dwell_elapsed_minutes
                    underwater_state["trough_price"] = trough_price
                    underwater_state["rebound_peak_price"] = rebound_peak_price
                    underwater_state["rebound_pct"] = rebound_pct
                    underwater_state["recovery_ratio"] = recovery_ratio
                    underwater_state["fade_from_rebound_peak_pct"] = fade_from_rebound_peak_pct
                    underwater_state["reversal_pressure"] = reversal_pressure
                    underwater_state["timeout_loss_pct"] = underwater_timeout_loss_pct

                    if (
                        exit_headroom_ratio is not None
                        and recovery_ready
                        and soft_fade
                        and exit_headroom_ratio <= warn_headroom_ratio * 1.1
                    ):
                        context_payload[underwater_state_key] = underwater_state
                        return {
                            "config": config,
                            "decision": {
                                "action": "close",
                                "reason": (
                                    "Underwater executable salvage exit "
                                    f"(headroom={exit_headroom_ratio:.2f}x, "
                                    f"recovery={recovery_ratio:.2f}, fade={fade_from_rebound_peak_pct:.2f}%)"
                                ),
                                "close_price": current_price,
                            },
                        }

                    if dwell_ready and recovery_ready and (rebound_fading or (soft_fade and reversal_pressure)):
                        context_payload[underwater_state_key] = underwater_state
                        return {
                            "config": config,
                            "decision": {
                                "action": "close",
                                "reason": (
                                    "Underwater rebound salvage exit "
                                    f"(dwell={dwell_elapsed_minutes if dwell_elapsed_minutes is not None else 0.0:.2f}m, "
                                    f"recovery={recovery_ratio:.2f}, rebound={rebound_pct:.2f}%, "
                                    f"fade={fade_from_rebound_peak_pct:.2f}%)"
                                ),
                                "close_price": current_price,
                            },
                        }

                    if (
                        underwater_timeout_minutes > 0.0
                        and underwater_timeout_loss_pct > 0.0
                        and dwell_elapsed_minutes is not None
                        and dwell_elapsed_minutes >= underwater_timeout_minutes
                        and pnl_pct <= -abs(underwater_timeout_loss_pct)
                    ):
                        context_payload[underwater_state_key] = underwater_state
                        return {
                            "config": config,
                            "decision": {
                                "action": "close",
                                "reason": (
                                    "Underwater timeout stop "
                                    f"(dwell={dwell_elapsed_minutes:.2f}m >= {underwater_timeout_minutes:.2f}m, "
                                    f"loss={pnl_pct:.2f}% <= -{underwater_timeout_loss_pct:.2f}%)"
                                ),
                                "close_price": current_price,
                            },
                        }
                    context_payload[underwater_state_key] = underwater_state
                else:
                    context_payload.pop(underwater_state_key, None)
            else:
                context_payload.pop(underwater_state_key, None)

        stop_loss_policy = str(config.get("stop_loss_policy") or "near_close_only").strip().lower()
        if stop_loss_policy == "volatility_adaptive":
            spread = self._float(context_payload.get("spread"))
            if spread is None:
                spread = self._float(state.get("spread"))
            liquidity = self._float(context_payload.get("liquidity"))
            if liquidity is None:
                liquidity = self._float(state.get("liquidity"))
            seconds_left = self._float(state.get("seconds_left"))
            timeframe_seconds = self._timeframe_seconds(timeframe)

            volatility_factor = 1.0
            if spread is not None:
                volatility_factor += min(0.8, max(0.0, (spread * 100.0) / 6.0))
            if liquidity is not None and liquidity > 0.0:
                volatility_factor += min(0.7, 6000.0 / max(liquidity, 1.0))
            if seconds_left is not None and seconds_left >= 0.0:
                close_ratio = 1.0 - clamp(seconds_left / float(max(1, timeframe_seconds)), 0.0, 1.0)
                volatility_factor += close_ratio * 0.4

            base_stop_loss_pct = max(0.5, to_float(config.get("stop_loss_pct"), 5.0))
            adaptive_stop_loss_pct = clamp(
                base_stop_loss_pct * volatility_factor,
                max(0.5, base_stop_loss_pct * 0.8),
                max(1.0, base_stop_loss_pct * 1.9),
            )
            config["stop_loss_pct"] = adaptive_stop_loss_pct

            trailing_activation = max(0.1, to_float(config.get("trailing_stop_activation_profit_pct"), 4.0))
            config["trailing_stop_activation_profit_pct"] = clamp(
                trailing_activation / max(1.0, volatility_factor * 0.85),
                0.1,
                100.0,
            )

        return {"config": config, "decision": None}

    def _build_reverse_intent_for_close(
        self,
        *,
        position: Any,
        market_state: dict[str, Any],
        close_reason: str,
    ) -> dict[str, Any] | None:
        config = getattr(position, "config", None)
        if not isinstance(config, dict):
            return None
        if not to_bool(config.get("reverse_on_adverse_velocity_enabled"), False):
            return None

        entry_price = self._float(getattr(position, "entry_price", None))
        current_price = self._float(market_state.get("current_price"))
        if entry_price is None or entry_price <= 0.0 or current_price is None or current_price <= 0.0:
            return None

        pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
        reverse_min_loss_pct = max(0.0, to_float(config.get("reverse_min_loss_pct"), 2.0))
        if pnl_pct > -abs(reverse_min_loss_pct):
            return None

        strategy_context = getattr(position, "strategy_context", None)
        strategy_context = strategy_context if isinstance(strategy_context, dict) else {}
        rapid_state_raw = strategy_context.get("_crypto_hf_rapid_exit_state")
        rapid_state = rapid_state_raw if isinstance(rapid_state_raw, dict) else {}

        flow_imbalance = self._float(rapid_state.get("flow_imbalance"))
        momentum_short_pct = self._float(rapid_state.get("momentum_short_pct"))
        hazard_score = clamp(to_float(rapid_state.get("executable_hazard_score"), 0.0), 0.0, 1.0)

        flow_neg = clamp(-(flow_imbalance or 0.0), 0.0, 1.0)
        momentum_neg = clamp(-((momentum_short_pct or 0.0) / 2.0), 0.0, 1.0)
        adverse_velocity_score = clamp((hazard_score * 0.5) + (flow_neg * 0.3) + (momentum_neg * 0.2), 0.0, 1.0)
        reverse_min_velocity_score = clamp(to_float(config.get("reverse_min_adverse_velocity_score"), 0.55), 0.0, 1.0)
        if adverse_velocity_score + 1e-9 < reverse_min_velocity_score:
            return None

        reverse_flow_threshold = to_float(config.get("reverse_flow_imbalance_threshold"), -0.2)
        reverse_momentum_threshold = to_float(config.get("reverse_momentum_short_pct_threshold"), -0.25)
        directional_confirmation = bool(
            (flow_imbalance is not None and flow_imbalance <= reverse_flow_threshold)
            or (momentum_short_pct is not None and momentum_short_pct <= reverse_momentum_threshold)
        )
        if not directional_confirmation:
            return None

        seconds_left = self._float(
            _first_present(
                market_state.get("seconds_left"),
                strategy_context.get("seconds_left"),
                strategy_context.get("live_market_context", {}).get("seconds_left")
                if isinstance(strategy_context.get("live_market_context"), dict)
                else None,
            )
        )
        reverse_min_seconds_left = max(0.0, to_float(config.get("reverse_min_seconds_left"), 0.0))
        if seconds_left is not None and seconds_left < reverse_min_seconds_left:
            return None

        current_direction = (
            str(
                _first_present(
                    getattr(position, "direction", None),
                    strategy_context.get("direction"),
                    market_state.get("direction"),
                )
                or ""
            )
            .strip()
            .lower()
        )
        reverse_direction = StrategySDK.opposite_direction(current_direction, default="")
        if reverse_direction not in {"buy_yes", "buy_no"}:
            return None

        reverse_confidence = clamp(to_float(config.get("reverse_confidence"), 0.62), 0.0, 1.0)
        reverse_min_edge_percent = max(0.0, to_float(config.get("reverse_min_edge_percent"), 0.8))
        derived_edge_percent = max(
            reverse_min_edge_percent,
            abs(float(momentum_short_pct or 0.0)),
            abs(float(flow_imbalance or 0.0)) * 100.0,
            max(0.0, -pnl_pct),
        )
        reverse_size_multiplier = max(0.05, to_float(config.get("reverse_size_multiplier"), 1.0))
        reverse_ttl_seconds = max(5.0, to_float(config.get("reverse_signal_ttl_seconds"), 45.0))
        reverse_cooldown_seconds = max(0.0, to_float(config.get("reverse_cooldown_seconds"), 0.0))
        reverse_max_reentries = int(max(0.0, to_float(config.get("reverse_max_reentries_per_position"), 1.0)))
        reverse_min_price_headroom = clamp(to_float(config.get("reverse_min_price_headroom"), 0.18), 0.0, 1.0)

        token_id = str(
            _first_present(
                rapid_state.get("token_id"),
                market_state.get("token_id"),
                strategy_context.get("token_id"),
            )
            or ""
        ).strip()

        reverse_intent = StrategySDK.build_reverse_intent(
            direction=reverse_direction,
            signal_type="crypto_worker_reverse",
            edge_percent=derived_edge_percent,
            confidence=max(reverse_confidence, adverse_velocity_score),
            size_multiplier=reverse_size_multiplier,
            min_seconds_left=reverse_min_seconds_left,
            min_price_headroom=reverse_min_price_headroom,
            expires_in_seconds=reverse_ttl_seconds,
            strategy_type=self.strategy_type,
            token_id=token_id or None,
            reason=f"Auto reverse after close: {close_reason}",
            cooldown_seconds=reverse_cooldown_seconds,
            max_reentries_per_position=reverse_max_reentries,
            metadata={
                "trigger": "btc_hf_adverse_velocity",
                "pnl_percent": pnl_pct,
                "flow_imbalance": flow_imbalance,
                "momentum_short_pct": momentum_short_pct,
                "hazard_score": hazard_score,
                "adverse_velocity_score": adverse_velocity_score,
            },
        )
        return reverse_intent or None

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        evaluation = self._evaluate_local_exit(position, market_state)
        config = evaluation.get("config")
        if isinstance(config, dict):
            position.config = config
        decision = evaluation.get("decision")
        if isinstance(decision, dict) and str(decision.get("action") or "").strip().lower() == "close":
            close_price = self._float(decision.get("close_price"))
            close_reason = str(decision.get("reason") or "High-frequency exit")
            payload: dict[str, Any] = {}
            reverse_intent = self._build_reverse_intent_for_close(
                position=position,
                market_state=market_state,
                close_reason=close_reason,
            )
            if isinstance(reverse_intent, dict):
                payload["reverse_intent"] = reverse_intent
            return ExitDecision(
                "close",
                close_reason,
                close_price=close_price,
                payload=payload,
            )
        default_decision = self.default_exit_check(position, market_state)
        if str(default_decision.action or "").strip().lower() != "close":
            return default_decision
        payload = dict(default_decision.payload) if isinstance(default_decision.payload, dict) else {}
        reverse_intent = self._build_reverse_intent_for_close(
            position=position,
            market_state=market_state,
            close_reason=str(default_decision.reason or "High-frequency exit"),
        )
        if isinstance(reverse_intent, dict):
            payload["reverse_intent"] = reverse_intent
            return ExitDecision(
                "close",
                default_decision.reason,
                close_price=default_decision.close_price,
                reduce_fraction=default_decision.reduce_fraction,
                payload=payload,
            )
        return default_decision

    # ------------------------------------------------------------------
    # Event-driven detection (crypto_update from crypto worker)
    # ------------------------------------------------------------------

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        if event.event_type != "crypto_update":
            return []

        markets = event.payload.get("markets") or []
        if not markets:
            return []

        return self._detect_from_crypto_markets(markets)

    @staticmethod
    def _market_from_crypto_dict(d: dict) -> Market:
        """Deserialize a crypto worker market dict into a typed Market object.

        Crypto worker dicts use non-standard keys (condition_id, up_price,
        down_price, end_time). We map them to Market fields immediately at
        the DataEvent boundary so every downstream code path sees only typed
        objects — never raw dicts.
        """
        market_id = str(d.get("condition_id") or d.get("id") or "")
        up_price = float(d.get("up_price") or 0.0)
        down_price = float(d.get("down_price") or 0.0)
        liquidity = max(0.0, float(d.get("liquidity") or 0.0))
        slug = d.get("slug") or market_id
        question = d.get("question") or slug

        end_date = None
        end_time_raw = d.get("end_time")
        if isinstance(end_time_raw, str) and end_time_raw.strip():
            try:
                from datetime import datetime as _dt

                end_date = _dt.fromisoformat(end_time_raw.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        raw_token_ids = d.get("clob_token_ids") or []
        clob_token_ids = [str(t).strip() for t in raw_token_ids if str(t).strip() and len(str(t).strip()) > 20]

        return Market(
            id=market_id,
            condition_id=market_id,
            question=question,
            slug=slug,
            # yes_price = up_price (market-implied prob of going Up)
            outcome_prices=[up_price, down_price],
            liquidity=liquidity,
            end_date=end_date,
            platform="polymarket",
            clob_token_ids=clob_token_ids,
        )

    def _detect_from_crypto_markets(self, markets: list[dict]) -> list[Opportunity]:
        """Oracle-only directional signal logic for the crypto worker hot path.

        For each crypto market dict, compute oracle-based directional edge
        with fee-aware gating. No rebalance, no pure_arb -- direction is
        determined SOLELY by the oracle signal when oracle is available.
        """
        opportunities: list[Opportunity] = []

        for market in markets:
            market_id = str(market.get("condition_id") or market.get("id") or "")
            if not market_id:
                continue
            typed_market = self._market_from_crypto_dict(market)
            asset = self._detect_asset(typed_market) or _normalize_asset(
                market.get("asset") or market.get("symbol") or market.get("coin")
            )
            timeframe = _normalize_timeframe(market.get("timeframe"))
            if not timeframe:
                timeframe = _normalize_timeframe(self._detect_timeframe(typed_market))

            up_price = self._float(market.get("up_price"))
            down_price = self._float(market.get("down_price"))
            if up_price is None or down_price is None:
                continue
            if not (0.0 <= up_price <= 1.0 and 0.0 <= down_price <= 1.0):
                continue

            price_to_beat = self._float(market.get("price_to_beat"))
            oracle_price = self._float(market.get("oracle_price"))
            oracle_status = _extract_oracle_status(
                live_market={},
                payload={
                    "oracle_price": oracle_price,
                    "oracle_source": market.get("oracle_source"),
                    "oracle_updated_at_ms": market.get("oracle_updated_at_ms"),
                    "oracle_age_seconds": market.get("oracle_age_seconds"),
                    "price_to_beat": price_to_beat,
                    "oracle_prices_by_source": market.get("oracle_prices_by_source"),
                },
                now_ms=int(time.time() * 1000.0),
            )
            oracle_price = self._float(oracle_status.get("price"))
            price_to_beat = self._float(oracle_status.get("price_to_beat"))
            has_oracle = bool(oracle_status.get("available"))

            timeframe_seconds = self._timeframe_seconds(market.get("timeframe"))
            seconds_left = self._float(market.get("seconds_left"))
            if seconds_left is None:
                end_time = market.get("end_time")
                if isinstance(end_time, str) and end_time.strip():
                    try:
                        from datetime import datetime as _dt, timezone as _tz

                        parsed = _dt.fromisoformat(end_time.replace("Z", "+00:00"))
                        seconds_left = max(0.0, (parsed - _dt.now(_tz.utc)).total_seconds())
                    except Exception:
                        seconds_left = float(timeframe_seconds)
                else:
                    seconds_left = float(timeframe_seconds)
            is_live = bool(market.get("is_live")) if isinstance(market.get("is_live"), bool) else (seconds_left > 0.0)
            is_current = bool(market.get("is_current")) if isinstance(market.get("is_current"), bool) else is_live
            start_time = str(market.get("start_time") or "").strip() or None
            end_time = str(market.get("end_time") or "").strip() or None
            live_market_fetched_at = (
                str(market.get("fetched_at") or market.get("snapshot_fetched_at") or "").strip() or None
            )
            market_data_age_ms = self._float(market.get("market_data_age_ms"))
            if market_data_age_ms is None and live_market_fetched_at:
                parsed_fetched_at = _parse_datetime_utc(live_market_fetched_at)
                if parsed_fetched_at is not None:
                    market_data_age_ms = max(
                        0.0,
                        (datetime.now(timezone.utc) - parsed_fetched_at.astimezone(timezone.utc)).total_seconds()
                        * 1000.0,
                    )

            regime = self._crypto_regime(seconds_left, timeframe_seconds)

            # --- Oracle-only directional edge ---
            # Direction is determined SOLELY by oracle sign.  Edge is the
            # oracle signal strength scaled by time remaining -- NOT
            # model_prob vs market_price (that is rebalance in disguise).
            _MIN_ORACLE_DIFF_FOR_EDGE = 0.15  # % -- below this, no signal
            if has_oracle and price_to_beat is not None and oracle_price is not None:
                diff_pct = ((oracle_price - price_to_beat) / price_to_beat) * 100.0
            else:
                diff_pct = 0.0

            if abs(diff_pct) < _MIN_ORACLE_DIFF_FOR_EDGE:
                # Oracle diff too small to be meaningful -- skip
                continue

            # Direction from oracle SIGN only -- never buy against oracle
            if diff_pct > 0:
                direction = "buy_yes"
                entry_price = up_price
            else:
                direction = "buy_no"
                entry_price = down_price

            # Edge = oracle magnitude scaled by time urgency.
            # Late in the window the oracle is more predictive.
            time_ratio = clamp(seconds_left / float(max(1, timeframe_seconds)), 0.01, 1.0)
            time_multiplier = 1.0 + 2.0 * (1.0 - time_ratio)  # 1x early, 3x at expiry
            edge_percent = abs(diff_pct) * time_multiplier

            # Fee-aware entry gate
            taker_fee_pct = polymarket_fee_pct(entry_price) if entry_price > 0 else 0.0
            min_required_edge = taker_fee_pct * 200.0  # 2x fee as percentage points
            if edge_percent < max(1.0, min_required_edge):
                continue

            # Execution penalties (simplified -- no rebalance/pure_arb)
            spread = clamp(self._float(market.get("spread")) or 0.0, 0.0, 0.10)
            liquidity = max(0.0, self._float(market.get("liquidity")) or 0.0)
            spread_penalty = spread * 100.0 * 0.35
            liquidity_scale = clamp(liquidity / 250000.0, 0.0, 1.0)
            slippage_penalty = (1.35 - (0.95 * liquidity_scale))
            net_edge = edge_percent - spread_penalty - slippage_penalty
            if net_edge < 1.0:
                continue
            edge_percent = net_edge

            # Confidence -- based on oracle diff strength and edge magnitude
            confidence = clamp(
                0.32
                + clamp(edge_percent / 20.0, 0.0, 0.35)
                + clamp(abs(diff_pct) / 1.0, 0.0, 0.20),
                0.05,
                0.97,
            )
            if not has_oracle:
                confidence = clamp(confidence * 0.75, 0.05, 0.85)

            side = "YES" if direction == "buy_yes" else "NO"
            slug = market.get("slug") or market_id

            token_idx = 0 if direction == "buy_yes" else 1
            token_ids = typed_market.clob_token_ids or []
            position_token_id = token_ids[token_idx] if len(token_ids) > token_idx else None

            opp = self.create_opportunity(
                title=f"Crypto HF: {slug} {side}",
                description=(
                    f"{regime} regime, oracle directional | edge={edge_percent:.1f}%, conf={confidence:.0%}"
                ),
                total_cost=entry_price,
                expected_payout=entry_price + (edge_percent / 100.0),
                markets=[typed_market],
                positions=[
                    {
                        "action": "BUY",
                        "outcome": side,
                        "price": entry_price,
                        "token_id": position_token_id,
                        "post_only": True,
                        "_maker_mode": True,
                        "_maker_price": entry_price,
                        "_crypto_context": {
                            "signal_version": "crypto_worker_v3",
                            "signal_family": "crypto_maker",
                            "strategy_origin": "crypto_worker",
                            "selected_direction": direction,
                            "asset": asset,
                            "timeframe": timeframe,
                            "regime": regime,
                            "start_time": start_time,
                            "end_time": end_time,
                            "seconds_left": float(seconds_left),
                            "is_live": is_live,
                            "is_current": is_current,
                            "oracle_available": has_oracle,
                            "oracle_age_seconds": self._float(market.get("oracle_age_seconds")),
                            "oracle_updated_at_ms": self._float(market.get("oracle_updated_at_ms")),
                            "oracle_status": dict(oracle_status),
                            "oracle_source": oracle_status.get("source"),
                            "oracle_prices_by_source": _json_safe(market.get("oracle_prices_by_source") or {}),
                            "oracle_diff_pct": diff_pct,
                            "taker_fee_gate": min_required_edge,
                            "edge_percent": edge_percent,
                            "spread": spread,
                            "spread_widening_bps": self._float(market.get("spread_widening_bps")),
                            "orderbook_imbalance": self._float(
                                _first_present(
                                    market.get("orderbook_imbalance"),
                                    market.get("book_imbalance"),
                                    market.get("imbalance"),
                                )
                            ),
                            "liquidity": liquidity,
                            "volume": self._float(market.get("volume")) or 0.0,
                            "price_to_beat": price_to_beat,
                            "oracle_price": oracle_price,
                            "live_market_fetched_at": live_market_fetched_at,
                            "market_data_age_ms": market_data_age_ms,
                        },
                    }
                ],
                is_guaranteed=False,
                skip_fee_model=True,
                custom_roi_percent=edge_percent,
                custom_risk_score=1.0 - confidence,
                confidence=confidence,
            )
            if opp is not None:
                opp.risk_factors = [
                    f"Crypto {regime} regime",
                    f"Oracle directional (diff={diff_pct:+.3f}%)",
                    f"Oracle: {'available' if has_oracle else 'unavailable'}",
                    f"Fee gate: edge {edge_percent:.1f}% > 2x fee {min_required_edge:.1f}%",
                ]
                opp.strategy_context = {
                    "source_key": "crypto",
                    "strategy_slug": self.strategy_type,
                    "strategy_origin": "crypto_worker",
                    "asset": asset,
                    "timeframe": timeframe,
                    "regime": regime,
                    "start_time": start_time,
                    "end_time": end_time,
                    "seconds_left": float(seconds_left),
                    "is_live": is_live,
                    "is_current": is_current,
                    "selected_direction": direction,
                    "oracle_available": has_oracle,
                    "oracle_age_seconds": self._float(market.get("oracle_age_seconds")),
                    "oracle_updated_at_ms": self._float(market.get("oracle_updated_at_ms")),
                    "oracle_status": dict(oracle_status),
                    "oracle_source": oracle_status.get("source"),
                    "oracle_prices_by_source": _json_safe(market.get("oracle_prices_by_source") or {}),
                    "oracle_diff_pct": diff_pct,
                    "taker_fee_gate": min_required_edge,
                    "edge_percent": edge_percent,
                    "spread": spread,
                    "spread_widening_bps": self._float(market.get("spread_widening_bps")),
                    "orderbook_imbalance": self._float(
                        _first_present(
                            market.get("orderbook_imbalance"),
                            market.get("book_imbalance"),
                            market.get("imbalance"),
                        )
                    ),
                    "liquidity": liquidity,
                    "volume": self._float(market.get("volume")) or 0.0,
                    "price_to_beat": price_to_beat,
                    "oracle_price": oracle_price,
                    "live_market_fetched_at": live_market_fetched_at,
                    "market_data_age_ms": market_data_age_ms,
                }
                opportunities.append(opp)

        return opportunities

    @staticmethod
    def _float(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _timeframe_seconds(value: object) -> int:
        tf = str(value or "").strip().lower()
        if tf in {"5m", "5min"}:
            return 300
        if tf in {"15m", "15min"}:
            return 900
        if tf in {"1h", "1hr", "60m"}:
            return 3600
        if tf in {"4h", "4hr", "240m"}:
            return 14400
        return 900

    @staticmethod
    def _crypto_regime(seconds_left: float, timeframe_seconds: int) -> str:
        denom = float(max(1, timeframe_seconds))
        ratio = clamp(seconds_left / denom, 0.0, 1.0)
        if ratio > 0.67:
            return "opening"
        if ratio < 0.33:
            return "closing"
        return "mid"

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
