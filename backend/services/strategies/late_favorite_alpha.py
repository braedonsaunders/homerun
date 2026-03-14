"""Late favorite alpha strategy.

Finds markets close to resolution where one side is favored but not yet
definitive, and there is still directional upside before expiry.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import timezone
from typing import Any, Optional

from config import settings
from models import Event, Market, Opportunity
from models.opportunity import MispricingType
from services.quality_filter import QualityFilterOverrides
from services.strategies.base import (
    BaseStrategy,
    DecisionCheck,
    ExitDecision,
    ScoringWeights,
    SizingConfig,
    make_aware,
    utcnow,
)
from services.strategy_sdk import StrategySDK
from utils.converters import clamp, coerce_bool as _coerce_bool, safe_float, to_float
from utils.signal_helpers import days_to_resolution


LATE_FAVORITE_ALPHA_DEFAULTS: dict[str, Any] = {
    "min_edge_percent": 1.0,
    "min_confidence": 0.42,
    "max_risk_score": 0.78,
    "base_size_usd": 20.0,
    "max_size_usd": 140.0,
    "min_favorite_prob": 0.58,
    "max_favorite_prob": 0.88,
    "min_hours_to_resolution": 0.25,
    "max_hours_to_resolution": 6.0,
    "min_liquidity": 3000.0,
    "max_spread_bps": 120.0,
    "require_spread_data": True,
    "flow_lookback_seconds": 900.0,
    "min_flow_volume_usd": 150.0,
    "max_recent_move_pct": 8.0,
    "time_alpha_scale": 0.26,
    "flow_alpha_scale": 0.18,
    "momentum_alpha_scale": 0.06,
    "spread_friction_scale": 0.45,
    "min_alpha_prob": 0.014,
    "max_target_exit_price": 0.97,
    "alpha_exhaustion_ratio": 0.85,
    "force_flatten_seconds_left": 120.0,
    "force_flatten_min_pnl_pct": -2.0,
    "take_profit_pct": 12.0,
    "stop_loss_pct": 8.0,
    "max_hold_minutes": 360.0,
    "price_policy": "taker_limit",
    "time_in_force": "IOC",
    "session_timeout_seconds": 90,
}

LATE_FAVORITE_ALPHA_CONFIG_SCHEMA: dict[str, Any] = {
    "param_fields": [
        {"key": "min_favorite_prob", "label": "Min Favorite Probability", "type": "number", "min": 0.5, "max": 0.99},
        {"key": "max_favorite_prob", "label": "Max Favorite Probability", "type": "number", "min": 0.5, "max": 0.995},
        {"key": "min_hours_to_resolution", "label": "Min Hours To Resolution", "type": "number", "min": 0.0, "max": 24.0},
        {"key": "max_hours_to_resolution", "label": "Max Hours To Resolution", "type": "number", "min": 0.1, "max": 72.0},
        {"key": "min_liquidity", "label": "Min Liquidity (USD)", "type": "number", "min": 0},
        {"key": "max_spread_bps", "label": "Max Spread (bps)", "type": "number", "min": 0, "max": 5000},
        {"key": "require_spread_data", "label": "Require Spread Data", "type": "boolean"},
        {"key": "flow_lookback_seconds", "label": "Flow Lookback (sec)", "type": "number", "min": 30, "max": 7200},
        {"key": "min_flow_volume_usd", "label": "Min Flow Volume (USD)", "type": "number", "min": 0},
        {"key": "max_recent_move_pct", "label": "Max Recent Move (%)", "type": "number", "min": 0, "max": 100},
        {"key": "time_alpha_scale", "label": "Time Alpha Scale", "type": "number", "min": 0.01, "max": 1.0},
        {"key": "flow_alpha_scale", "label": "Flow Alpha Scale", "type": "number", "min": 0.0, "max": 1.0},
        {"key": "momentum_alpha_scale", "label": "Momentum Alpha Scale", "type": "number", "min": 0.0, "max": 1.0},
        {"key": "spread_friction_scale", "label": "Spread Friction Scale", "type": "number", "min": 0.0, "max": 2.0},
        {"key": "min_alpha_prob", "label": "Min Alpha Probability", "type": "number", "min": 0.0005, "max": 0.25},
        {"key": "max_target_exit_price", "label": "Max Target Exit Price", "type": "number", "min": 0.55, "max": 0.999},
        {"key": "alpha_exhaustion_ratio", "label": "Alpha Exhaustion Ratio", "type": "number", "min": 0.0, "max": 1.0},
        {"key": "force_flatten_seconds_left", "label": "Force Flatten Seconds Left", "type": "number", "min": 0, "max": 14400},
        {"key": "force_flatten_min_pnl_pct", "label": "Force Flatten Min PnL (%)", "type": "number", "min": -100, "max": 100},
        {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
        {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
        {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
        {"key": "base_size_usd", "label": "Base Size (USD)", "type": "number", "min": 1, "max": 1000000},
        {"key": "max_size_usd", "label": "Max Size (USD)", "type": "number", "min": 1, "max": 1000000},
        {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 0, "max": 100},
        {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 0, "max": 100},
        {"key": "max_hold_minutes", "label": "Max Hold (Minutes)", "type": "number", "min": 1, "max": 10080},
    ]
}


def late_favorite_alpha_defaults() -> dict[str, Any]:
    return dict(LATE_FAVORITE_ALPHA_DEFAULTS)


def late_favorite_alpha_config_schema() -> dict[str, Any]:
    return dict(LATE_FAVORITE_ALPHA_CONFIG_SCHEMA)


def _coerce_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(lo, min(hi, parsed))


def _coerce_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = default
    return max(lo, min(hi, parsed))


def validate_late_favorite_alpha_config(config: Any) -> dict[str, Any]:
    cfg = late_favorite_alpha_defaults()
    raw = config if isinstance(config, dict) else {}
    cfg.update({str(k): v for k, v in raw.items() if str(k) != "_schema"})

    cfg["min_edge_percent"] = _coerce_float(cfg.get("min_edge_percent"), 1.0, 0.0, 100.0)
    cfg["min_confidence"] = _coerce_float(cfg.get("min_confidence"), 0.42, 0.0, 1.0)
    cfg["max_risk_score"] = _coerce_float(cfg.get("max_risk_score"), 0.78, 0.0, 1.0)
    cfg["base_size_usd"] = _coerce_float(cfg.get("base_size_usd"), 20.0, 1.0, 1_000_000.0)
    cfg["max_size_usd"] = _coerce_float(cfg.get("max_size_usd"), cfg["base_size_usd"], 1.0, 1_000_000.0)
    if cfg["max_size_usd"] < cfg["base_size_usd"]:
        cfg["max_size_usd"] = cfg["base_size_usd"]

    cfg["min_favorite_prob"] = _coerce_float(cfg.get("min_favorite_prob"), 0.58, 0.5, 0.99)
    cfg["max_favorite_prob"] = _coerce_float(cfg.get("max_favorite_prob"), 0.88, cfg["min_favorite_prob"] + 0.01, 0.995)
    if cfg["max_favorite_prob"] <= cfg["min_favorite_prob"]:
        cfg["max_favorite_prob"] = min(0.995, cfg["min_favorite_prob"] + 0.01)

    cfg["min_hours_to_resolution"] = _coerce_float(cfg.get("min_hours_to_resolution"), 0.25, 0.0, 24.0)
    cfg["max_hours_to_resolution"] = _coerce_float(
        cfg.get("max_hours_to_resolution"),
        6.0,
        cfg["min_hours_to_resolution"] + 0.01,
        72.0,
    )
    if cfg["max_hours_to_resolution"] <= cfg["min_hours_to_resolution"]:
        cfg["max_hours_to_resolution"] = min(72.0, cfg["min_hours_to_resolution"] + 0.01)

    cfg["min_liquidity"] = _coerce_float(cfg.get("min_liquidity"), 3000.0, 0.0, 1_000_000_000.0)
    cfg["max_spread_bps"] = _coerce_float(cfg.get("max_spread_bps"), 120.0, 0.0, 5000.0)
    cfg["require_spread_data"] = _coerce_bool(cfg.get("require_spread_data"), True)
    cfg["flow_lookback_seconds"] = _coerce_float(cfg.get("flow_lookback_seconds"), 900.0, 30.0, 7200.0)
    cfg["min_flow_volume_usd"] = _coerce_float(cfg.get("min_flow_volume_usd"), 150.0, 0.0, 1_000_000_000.0)
    cfg["max_recent_move_pct"] = _coerce_float(cfg.get("max_recent_move_pct"), 8.0, 0.0, 100.0)
    cfg["time_alpha_scale"] = _coerce_float(cfg.get("time_alpha_scale"), 0.26, 0.01, 1.0)
    cfg["flow_alpha_scale"] = _coerce_float(cfg.get("flow_alpha_scale"), 0.18, 0.0, 1.0)
    cfg["momentum_alpha_scale"] = _coerce_float(cfg.get("momentum_alpha_scale"), 0.06, 0.0, 1.0)
    cfg["spread_friction_scale"] = _coerce_float(cfg.get("spread_friction_scale"), 0.45, 0.0, 2.0)
    cfg["min_alpha_prob"] = _coerce_float(cfg.get("min_alpha_prob"), 0.014, 0.0005, 0.25)
    cfg["max_target_exit_price"] = _coerce_float(cfg.get("max_target_exit_price"), 0.97, 0.55, 0.999)
    cfg["alpha_exhaustion_ratio"] = _coerce_float(cfg.get("alpha_exhaustion_ratio"), 0.85, 0.0, 1.0)
    cfg["force_flatten_seconds_left"] = _coerce_float(cfg.get("force_flatten_seconds_left"), 120.0, 0.0, 14_400.0)
    cfg["force_flatten_min_pnl_pct"] = _coerce_float(cfg.get("force_flatten_min_pnl_pct"), -2.0, -100.0, 100.0)
    cfg["take_profit_pct"] = _coerce_float(cfg.get("take_profit_pct"), 12.0, 0.0, 100.0)
    cfg["stop_loss_pct"] = _coerce_float(cfg.get("stop_loss_pct"), 8.0, 0.0, 100.0)
    cfg["max_hold_minutes"] = _coerce_int(cfg.get("max_hold_minutes"), 360, 1, 10_080)
    cfg["session_timeout_seconds"] = _coerce_int(cfg.get("session_timeout_seconds"), 90, 1, 86_400)
    cfg["price_policy"] = str(cfg.get("price_policy") or "taker_limit").strip().lower() or "taker_limit"
    cfg["time_in_force"] = str(cfg.get("time_in_force") or "IOC").strip().upper() or "IOC"

    return StrategySDK.normalize_strategy_retention_config(cfg)


class LateFavoriteAlphaStrategy(BaseStrategy):
    """Near-expiry favorite continuation strategy."""

    strategy_type = "late_favorite_alpha"
    name = "Late Favorite Alpha"
    description = "Near-expiry favorites with unresolved headroom and measurable continuation alpha"
    mispricing_type = "within_market"
    source_key = "scanner"
    requires_resolution_date = True
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.8,
        max_resolution_months=0.4,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.60,
        confidence_weight=30.0,
        risk_penalty=8.5,
    )
    sizing_config = SizingConfig()

    default_config = late_favorite_alpha_defaults()
    pipeline_defaults = {
        "min_edge_percent": 1.0,
        "min_confidence": 0.42,
        "max_risk_score": 0.78,
        "base_size_usd": 20.0,
        "max_size_usd": 140.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = validate_late_favorite_alpha_config(self.default_config)

    def configure(self, config: dict) -> None:
        self.config = validate_late_favorite_alpha_config(config)

    @staticmethod
    def _token_for_side(market: Market, side: str) -> Optional[str]:
        token_ids = list(getattr(market, "clob_token_ids", []) or [])
        if side == "YES":
            return token_ids[0] if len(token_ids) > 0 else None
        return token_ids[1] if len(token_ids) > 1 else None

    @staticmethod
    def _selected_entry_price(signal: Any, payload: dict[str, Any]) -> float:
        entry = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        if entry > 0:
            return entry
        positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
        if not positions:
            return 0.0
        return to_float((positions[0] or {}).get("price"), 0.0)

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        cfg = validate_late_favorite_alpha_config(getattr(self, "config", {}) or {})

        min_favorite_prob = to_float(cfg.get("min_favorite_prob"), 0.58)
        max_favorite_prob = to_float(cfg.get("max_favorite_prob"), 0.88)
        min_hours_to_resolution = to_float(cfg.get("min_hours_to_resolution"), 0.25)
        max_hours_to_resolution = to_float(cfg.get("max_hours_to_resolution"), 6.0)
        min_liquidity = to_float(cfg.get("min_liquidity"), 3000.0)
        max_spread_bps = to_float(cfg.get("max_spread_bps"), 120.0)
        require_spread_data = bool(cfg.get("require_spread_data", True))
        flow_lookback_seconds = max(30.0, to_float(cfg.get("flow_lookback_seconds"), 900.0))
        min_flow_volume_usd = max(0.0, to_float(cfg.get("min_flow_volume_usd"), 150.0))
        max_recent_move_pct = max(0.0, to_float(cfg.get("max_recent_move_pct"), 8.0))
        time_alpha_scale = clamp(to_float(cfg.get("time_alpha_scale"), 0.26), 0.01, 1.0)
        flow_alpha_scale = clamp(to_float(cfg.get("flow_alpha_scale"), 0.18), 0.0, 1.0)
        momentum_alpha_scale = clamp(to_float(cfg.get("momentum_alpha_scale"), 0.06), 0.0, 1.0)
        spread_friction_scale = clamp(to_float(cfg.get("spread_friction_scale"), 0.45), 0.0, 2.0)
        min_alpha_prob = max(0.0005, to_float(cfg.get("min_alpha_prob"), 0.014))
        max_target_exit_price = clamp(to_float(cfg.get("max_target_exit_price"), 0.97), 0.55, 0.999)
        max_opportunities = max(1, int(to_float(cfg.get("max_opportunities"), 100)))

        event_by_market: dict[str, Event] = {}
        for event in events:
            for market in event.markets:
                event_by_market[str(market.id)] = event

        now = utcnow().astimezone(timezone.utc)
        trade_tape_available = StrategySDK.is_ws_feed_started()
        history_by_key = self.state.setdefault("late_favorite_price_history", {})
        candidates: list[tuple[float, Opportunity]] = []

        for market in markets:
            if market.closed or not market.active:
                continue
            if to_float(getattr(market, "liquidity", 0.0), 0.0) < min_liquidity:
                continue

            token_ids = list(getattr(market, "clob_token_ids", []) or [])
            outcome_prices = list(getattr(market, "outcome_prices", []) or [])
            if len(token_ids) < 2 and len(outcome_prices) < 2:
                continue

            end_date = make_aware(getattr(market, "end_date", None))
            if end_date is None:
                continue
            hours_to_resolution = (end_date - now).total_seconds() / 3600.0
            if hours_to_resolution < min_hours_to_resolution or hours_to_resolution > max_hours_to_resolution:
                continue

            for outcome in ("YES", "NO"):
                entry_price = to_float(StrategySDK.get_live_price(market, prices, side=outcome), 0.0)
                if entry_price <= 0.0 or entry_price >= 1.0:
                    continue
                if entry_price < min_favorite_prob or entry_price > max_favorite_prob:
                    continue

                spread_bps = StrategySDK.get_spread_bps(market, prices, side=outcome)
                if spread_bps is None and require_spread_data and trade_tape_available:
                    continue
                if spread_bps is not None and spread_bps > max_spread_bps:
                    continue

                token_id = self._token_for_side(market, outcome)
                history_key = f"{market.id}:{outcome}"
                history_points = max(8, min(64, int(flow_lookback_seconds / 15.0) + 4))
                history = history_by_key.get(history_key)
                if not isinstance(history, deque):
                    history = deque(maxlen=history_points)
                    history_by_key[history_key] = history
                history.append((now, float(entry_price)))
                history_horizon_seconds = max(flow_lookback_seconds * 1.2, 60.0)
                while history and isinstance(history[0], tuple):
                    point_ts = history[0][0]
                    if not hasattr(point_ts, "astimezone"):
                        history.popleft()
                        continue
                    if (now - point_ts).total_seconds() <= history_horizon_seconds:
                        break
                    history.popleft()

                flow_imbalance = 0.0
                recent_volume = 0.0
                recent_move_pct = 0.0
                flow_data_available = bool(token_id) and trade_tape_available

                if flow_data_available:
                    flow_imbalance = clamp(
                        to_float(
                            StrategySDK.get_buy_sell_imbalance(token_id, lookback_seconds=flow_lookback_seconds),
                            0.0,
                        ),
                        -1.0,
                        1.0,
                    )
                    volume_payload = StrategySDK.get_trade_volume(token_id, lookback_seconds=flow_lookback_seconds)
                    if isinstance(volume_payload, dict):
                        recent_volume = max(0.0, to_float(volume_payload.get("total"), 0.0))
                    price_delta = StrategySDK.get_price_change(token_id, lookback_seconds=int(flow_lookback_seconds))
                    if isinstance(price_delta, dict):
                        recent_move_pct = abs(to_float(price_delta.get("change_pct"), 0.0))
                if recent_move_pct <= 0.0 and len(history) >= 2:
                    baseline_price = to_float(history[0][1], 0.0)
                    if baseline_price > 0.0:
                        recent_move_pct = abs(((entry_price - baseline_price) / baseline_price) * 100.0)

                if flow_data_available and recent_volume < min_flow_volume_usd:
                    continue
                if recent_move_pct > max_recent_move_pct:
                    continue

                unresolved_headroom = max(0.0, 1.0 - entry_price)
                time_pressure = 1.0 - (hours_to_resolution / max(max_hours_to_resolution, 0.0001))
                time_pressure = clamp(time_pressure, 0.0, 1.0)
                positive_flow = max(0.0, flow_imbalance)
                momentum_tailwind = max(0.0, recent_move_pct / 100.0)
                spread_cost_ratio = (spread_bps / 10000.0) if isinstance(spread_bps, (int, float)) else 0.0

                alpha_prob = (
                    (unresolved_headroom * time_pressure * time_alpha_scale)
                    + (unresolved_headroom * positive_flow * flow_alpha_scale)
                    + (unresolved_headroom * momentum_tailwind * momentum_alpha_scale)
                    - (spread_cost_ratio * spread_friction_scale)
                )
                if alpha_prob < min_alpha_prob:
                    continue

                target_exit_price = min(max_target_exit_price, entry_price + alpha_prob)
                if target_exit_price <= (entry_price + 1e-6):
                    continue

                if flow_data_available:
                    volume_score = clamp(
                        recent_volume / max(min_flow_volume_usd * 3.0, 1.0),
                        0.0,
                        1.0,
                    )
                else:
                    liquidity_anchor = max(1.0, min_liquidity * 4.0)
                    volume_score = clamp(
                        to_float(getattr(market, "liquidity", 0.0), 0.0) / liquidity_anchor,
                        0.25,
                        1.0,
                    )
                spread_score = (
                    0.55
                    if spread_bps is None
                    else 1.0 - clamp(spread_bps / max(max_spread_bps, 1.0), 0.0, 1.0)
                )
                confidence = clamp(
                    0.28
                    + (time_pressure * 0.28)
                    + (positive_flow * 0.18)
                    + (volume_score * 0.16)
                    + (spread_score * 0.10),
                    0.12,
                    0.96,
                )

                positions = [
                    {
                        "action": "BUY",
                        "outcome": outcome,
                        "price": entry_price,
                        "token_id": token_id,
                        "entry_style": "late_favorite_alpha",
                        "_late_favorite": {
                            "hours_to_resolution": hours_to_resolution,
                            "target_exit_price": target_exit_price,
                            "alpha_prob": alpha_prob,
                            "flow_imbalance": flow_imbalance,
                            "recent_volume": recent_volume,
                            "recent_move_pct": recent_move_pct,
                            "spread_bps": spread_bps,
                        },
                    }
                ]

                opportunity = self.create_opportunity(
                    title=f"Late Favorite Alpha: {outcome} {entry_price:.1%}",
                    description=(
                        f"{outcome} favored at {entry_price:.3f} with {hours_to_resolution:.2f}h to resolution; "
                        f"estimated continuation alpha {alpha_prob:.2%}."
                    ),
                    total_cost=entry_price,
                    expected_payout=target_exit_price,
                    markets=[market],
                    positions=positions,
                    event=event_by_market.get(str(market.id)),
                    is_guaranteed=False,
                    min_liquidity_hard=min_liquidity,
                    min_position_size=max(settings.MIN_POSITION_SIZE, 5.0),
                    confidence=confidence,
                )
                if opportunity is None:
                    continue

                favorite_strength = clamp(
                    (entry_price - min_favorite_prob) / max(max_favorite_prob - min_favorite_prob, 1e-6),
                    0.0,
                    1.0,
                )
                spread_risk = (
                    0.45
                    if spread_bps is None
                    else clamp(spread_bps / max(max_spread_bps, 1.0), 0.0, 1.0)
                )
                volume_risk = 1.0 - volume_score
                risk_score = (
                    0.62
                    - (time_pressure * 0.16)
                    - (favorite_strength * 0.10)
                    + (spread_risk * 0.15)
                    + (volume_risk * 0.14)
                )
                opportunity.risk_score = clamp(risk_score, 0.20, 0.88)
                opportunity.risk_factors = [
                    f"Market resolves in {hours_to_resolution:.2f} hours",
                    f"Favorite probability {entry_price:.1%} with remaining headroom {(1.0 - entry_price):.1%}",
                    (
                        f"Order-flow imbalance {flow_imbalance:+.2f}, spread {spread_bps:.0f} bps"
                        if spread_bps is not None
                        else f"Order-flow imbalance {flow_imbalance:+.2f}, spread unavailable"
                    ),
                ]
                opportunity.mispricing_type = MispricingType.WITHIN_MARKET
                opportunity.strategy_context = {
                    "entry_price": entry_price,
                    "target_exit_price": target_exit_price,
                    "alpha_prob": alpha_prob,
                    "hours_to_resolution": hours_to_resolution,
                    "flow_imbalance": flow_imbalance,
                    "recent_volume_usd": recent_volume,
                    "recent_move_pct": recent_move_pct,
                    "spread_bps": spread_bps,
                    "favorite_outcome": outcome,
                    "flow_data_available": flow_data_available,
                    "trade_tape_available": trade_tape_available,
                }

                strength = alpha_prob * confidence * max(0.01, 1.0 - opportunity.risk_score)
                candidates.append((strength, opportunity))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[Opportunity] = []
        seen_markets: set[str] = set()
        for _, opportunity in candidates:
            market_payload = (opportunity.markets or [{}])[0]
            market_id = str((market_payload or {}).get("id") or "")
            if not market_id or market_id in seen_markets:
                continue
            seen_markets.add(market_id)
            selected.append(opportunity)
            if len(selected) >= max_opportunities:
                break
        return selected

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        min_entry = clamp(to_float(params.get("min_favorite_prob", 0.58), 0.58), 0.5, 0.98)
        max_entry = clamp(to_float(params.get("max_favorite_prob", 0.88), 0.88), min_entry + 0.01, 0.995)
        min_alpha = max(0.0005, to_float(params.get("min_alpha_prob", 0.014), 0.014))
        min_hours = max(0.0, to_float(params.get("min_hours_to_resolution", 0.25), 0.25))
        max_hours = max(min_hours + 0.01, to_float(params.get("max_hours_to_resolution", 6.0), 6.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        strategy_type = (
            str(payload.get("strategy") or payload.get("strategy_type") or getattr(signal, "strategy_type", "") or "")
            .strip()
            .lower()
        )
        strategy_ok = strategy_type == self.strategy_type

        entry_price = self._selected_entry_price(signal, payload)
        dtr = days_to_resolution(payload)
        hours_to_resolution = (dtr * 24.0) if dtr is not None else None
        alpha_prob = to_float(
            payload.get("alpha_prob") or payload.get("strategy_context", {}).get("alpha_prob"),
            0.0,
        )
        if alpha_prob <= 0.0:
            target_exit_price = to_float(
                payload.get("target_exit_price") or payload.get("strategy_context", {}).get("target_exit_price"),
                0.0,
            )
            if target_exit_price > entry_price > 0.0:
                alpha_prob = target_exit_price - entry_price

        payload["_alpha_prob"] = alpha_prob
        payload["_hours_to_resolution"] = hours_to_resolution

        return [
            DecisionCheck("source", "Scanner source", source == "scanner", detail="Requires source=scanner"),
            DecisionCheck(
                "strategy",
                "Strategy type match",
                strategy_ok,
                detail=f"strategy_type={self.strategy_type}",
            ),
            DecisionCheck(
                "favorite_band",
                "Favorite probability band",
                min_entry <= entry_price <= max_entry,
                score=entry_price,
                detail=f"[{min_entry:.3f}, {max_entry:.3f}]",
            ),
            DecisionCheck(
                "alpha_prob",
                "Minimum alpha probability",
                alpha_prob >= min_alpha,
                score=alpha_prob,
                detail=f"min={min_alpha:.4f}",
            ),
            DecisionCheck(
                "resolution_window",
                "Resolution window (hours)",
                hours_to_resolution is not None and min_hours <= hours_to_resolution <= max_hours,
                score=hours_to_resolution,
                detail=f"[{min_hours:.2f}, {max_hours:.2f}]",
            ),
        ]

    def compute_score(
        self,
        edge: float,
        confidence: float,
        risk_score: float,
        market_count: int,
        payload: dict,
    ) -> float:
        base = super().compute_score(edge, confidence, risk_score, market_count, payload)
        alpha_prob = max(0.0, to_float(payload.get("_alpha_prob"), 0.0))
        hours_to_resolution = safe_float(payload.get("_hours_to_resolution"), default=None)
        time_bonus = 0.0
        if hours_to_resolution is not None:
            time_bonus = clamp((24.0 - max(0.0, float(hours_to_resolution))) / 24.0, 0.0, 1.0) * 4.0
        return base + (alpha_prob * 130.0) + time_bonus

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        strategy_context = getattr(position, "strategy_context", None) or {}
        current_price = safe_float(market_state.get("current_price"), default=None)
        if current_price is None:
            return ExitDecision("hold", "No current price available")

        target_exit_price = to_float(strategy_context.get("target_exit_price"), 0.0)
        if target_exit_price > 0.0 and current_price >= target_exit_price:
            return ExitDecision(
                "close",
                f"Alpha target reached ({current_price:.4f} >= {target_exit_price:.4f})",
                close_price=current_price,
            )

        cfg = validate_late_favorite_alpha_config(getattr(position, "config", None) or {})
        entry_price = safe_float(getattr(position, "entry_price", None), default=None)
        if entry_price is None:
            entry_price = safe_float(strategy_context.get("entry_price"), default=0.0)
        pnl_percent = ((current_price - entry_price) / entry_price * 100.0) if entry_price and entry_price > 0.0 else 0.0

        alpha_exhaustion_ratio = clamp(to_float(cfg.get("alpha_exhaustion_ratio"), 0.85), 0.0, 1.0)
        if target_exit_price > entry_price > 0.0:
            exhaustion_price = entry_price + ((target_exit_price - entry_price) * alpha_exhaustion_ratio)
            if current_price >= exhaustion_price:
                return ExitDecision(
                    "close",
                    f"Alpha mostly realized ({alpha_exhaustion_ratio:.0%} threshold reached)",
                    close_price=current_price,
                )

        force_flatten_seconds_left = max(0.0, to_float(cfg.get("force_flatten_seconds_left"), 120.0))
        force_flatten_min_pnl_pct = to_float(cfg.get("force_flatten_min_pnl_pct"), -2.0)
        seconds_left = self._seconds_left_for_position(position, market_state)
        if (
            seconds_left is not None
            and force_flatten_seconds_left > 0.0
            and seconds_left <= force_flatten_seconds_left
            and pnl_percent >= force_flatten_min_pnl_pct
        ):
            return ExitDecision(
                "close",
                f"Resolution window flatten ({seconds_left:.0f}s left, pnl={pnl_percent:.2f}%)",
                close_price=current_price,
            )

        return self.default_exit_check(position, market_state)
