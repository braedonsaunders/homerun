# SEED TEMPLATE — not the live strategy. The DB `strategies.source_code` is the
# runtime master (loaded by the orchestrator + backtester); edits here only seed
# fresh installs / reset-to-factory. See services/opportunity_strategy_catalog.py.
"""Sports overreaction fader strategy.

Detects rapid price movements in live sports markets and fades the
overreaction. Academic research (Moskowitz 2021, Choi & Hui 2014) shows
that ~50 % of open-to-close line movement in sports markets is reversed,
and that *surprising* in-game events cause systematic overreaction.

Entry: after a rapid price swing (>= ``min_move_pct`` within
``move_window_seconds``) on a high-favorite sports market, buy the side
that just dropped — betting the move will partially revert.

Exit: take-profit when the reversion target is reached, stop-loss if the
move continues further against us, or time-based max hold.

Risk: directional single-leg position.  Low per-trade risk because:
 * entry requires objective evidence of overreaction (speed + magnitude)
 * tight stop-loss caps downside to ``stop_loss_pct`` per trade
 * 0.25× Kelly sizing keeps variance manageable
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
from utils.converters import clamp, safe_float, to_float

SPORT_KEYWORDS: frozenset[str] = frozenset(
    [
        "nba",
        "nfl",
        "mlb",
        "nhl",
        "soccer",
        "football",
        "basketball",
        "tennis",
        "atp",
        "wta",
        "ufc",
        "mma",
        "fifa",
        "golf",
        "cricket",
        "baseball",
        "hockey",
        "f1",
        "formula",
        "boxing",
        "rugby",
    ]
)

SPORTS_OVERREACTION_DEFAULTS: dict[str, Any] = {
    # ── Detection filters ──────────────────────────────────
    "min_move_pct": 5.0,
    "max_move_pct": 40.0,
    "move_window_seconds": 300.0,
    "min_favorite_prob": 0.60,
    "max_favorite_prob": 0.94,
    "min_hours_to_resolution": 0.05,
    "max_hours_to_resolution": 72.0,
    "min_liquidity": 2000.0,
    "max_spread_bps": 200.0,
    "require_spread_data": False,
    # ── Reversion model ────────────────────────────────────
    "reversion_fraction": 0.50,
    "min_reversion_edge": 0.015,
    # ── Flow / momentum confirmation ───────────────────────
    "flow_lookback_seconds": 600.0,
    "min_flow_volume_usd": 50.0,
    # ── Risk / sizing ──────────────────────────────────────
    "min_edge_percent": 0.5,
    "min_confidence": 0.30,
    "max_risk_score": 0.85,
    # ── Exit rules ─────────────────────────────────────────
    "take_profit_pct": 10.0,
    "stop_loss_pct": 6.0,
    "max_hold_minutes": 120.0,
    "trailing_stop_pct": 0.0,
    # ── Execution ──────────────────────────────────────────
    "price_policy": "taker_limit",
    "time_in_force": "IOC",
    "session_timeout_seconds": 60,
}

SPORTS_OVERREACTION_CONFIG_SCHEMA: dict[str, Any] = {
    "param_fields": [
        {"key": "min_move_pct", "label": "Min Move (%)", "type": "number", "min": 1.0, "max": 50.0, "phase": "signal"},
        {"key": "max_move_pct", "label": "Max Move (%)", "type": "number", "min": 2.0, "max": 80.0, "phase": "signal"},
        {"key": "move_window_seconds", "label": "Move Window (sec)", "type": "number", "min": 30, "max": 3600, "phase": "signal"},
        {"key": "min_favorite_prob", "label": "Min Favorite Prob", "type": "number", "min": 0.5, "max": 0.99, "phase": "signal"},
        {"key": "max_favorite_prob", "label": "Max Favorite Prob", "type": "number", "min": 0.5, "max": 0.999, "phase": "signal"},
        {"key": "min_hours_to_resolution", "label": "Min Hours To Resolve", "type": "number", "min": 0, "max": 24, "phase": "signal"},
        {"key": "max_hours_to_resolution", "label": "Max Hours To Resolve", "type": "number", "min": 0.1, "max": 168, "phase": "signal"},
        {"key": "min_liquidity", "label": "Min Liquidity (USD)", "type": "number", "min": 0, "phase": "signal"},
        {"key": "max_spread_bps", "label": "Max Spread (bps)", "type": "number", "min": 0, "max": 5000, "phase": "signal"},
        {"key": "reversion_fraction", "label": "Reversion Fraction", "type": "number", "min": 0.1, "max": 1.0, "phase": "signal"},
        {"key": "min_reversion_edge", "label": "Min Reversion Edge", "type": "number", "min": 0.001, "max": 0.50, "phase": "signal"},
        {"key": "flow_lookback_seconds", "label": "Flow Lookback (sec)", "type": "number", "min": 30, "max": 7200, "phase": "signal"},
        {"key": "min_flow_volume_usd", "label": "Min Flow Volume (USD)", "type": "number", "min": 0, "phase": "signal"},
        {"key": "min_edge_percent", "label": "Min Edge (%)", "type": "number", "min": 0, "max": 100},
        {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
        {"key": "max_risk_score", "label": "Max Risk Score", "type": "number", "min": 0, "max": 1},
        {"key": "take_profit_pct", "label": "Take Profit (%)", "type": "number", "min": 0, "max": 100},
        {"key": "stop_loss_pct", "label": "Stop Loss (%)", "type": "number", "min": 0, "max": 100},
        {"key": "max_hold_minutes", "label": "Max Hold (min)", "type": "number", "min": 1, "max": 10080},
    ]
}


def sports_overreaction_defaults() -> dict[str, Any]:
    return dict(SPORTS_OVERREACTION_DEFAULTS)


def sports_overreaction_config_schema() -> dict[str, Any]:
    return dict(SPORTS_OVERREACTION_CONFIG_SCHEMA)


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


def validate_sports_overreaction_config(config: Any) -> dict[str, Any]:
    cfg = sports_overreaction_defaults()
    raw = config if isinstance(config, dict) else {}
    cfg.update({str(k): v for k, v in raw.items() if str(k) != "_schema"})

    cfg["min_move_pct"] = _coerce_float(cfg.get("min_move_pct"), 5.0, 1.0, 50.0)
    cfg["max_move_pct"] = _coerce_float(cfg.get("max_move_pct"), 40.0, cfg["min_move_pct"] + 0.5, 80.0)
    cfg["move_window_seconds"] = _coerce_float(cfg.get("move_window_seconds"), 300.0, 30.0, 3600.0)
    cfg["min_favorite_prob"] = _coerce_float(cfg.get("min_favorite_prob"), 0.60, 0.5, 0.99)
    cfg["max_favorite_prob"] = _coerce_float(cfg.get("max_favorite_prob"), 0.94, cfg["min_favorite_prob"] + 0.01, 0.999)
    cfg["min_hours_to_resolution"] = _coerce_float(cfg.get("min_hours_to_resolution"), 0.05, 0.0, 24.0)
    cfg["max_hours_to_resolution"] = _coerce_float(
        cfg.get("max_hours_to_resolution"), 72.0, cfg["min_hours_to_resolution"] + 0.01, 168.0,
    )
    cfg["min_liquidity"] = _coerce_float(cfg.get("min_liquidity"), 2000.0, 0.0, 1_000_000_000.0)
    cfg["max_spread_bps"] = _coerce_float(cfg.get("max_spread_bps"), 200.0, 0.0, 5000.0)
    cfg["require_spread_data"] = bool(cfg.get("require_spread_data", False))
    cfg["reversion_fraction"] = _coerce_float(cfg.get("reversion_fraction"), 0.50, 0.1, 1.0)
    cfg["min_reversion_edge"] = _coerce_float(cfg.get("min_reversion_edge"), 0.015, 0.001, 0.50)
    cfg["flow_lookback_seconds"] = _coerce_float(cfg.get("flow_lookback_seconds"), 600.0, 30.0, 7200.0)
    cfg["min_flow_volume_usd"] = _coerce_float(cfg.get("min_flow_volume_usd"), 50.0, 0.0, 1_000_000_000.0)
    cfg["min_edge_percent"] = _coerce_float(cfg.get("min_edge_percent"), 0.5, 0.0, 100.0)
    cfg["min_confidence"] = _coerce_float(cfg.get("min_confidence"), 0.30, 0.0, 1.0)
    cfg["max_risk_score"] = _coerce_float(cfg.get("max_risk_score"), 0.85, 0.0, 1.0)
    cfg["take_profit_pct"] = _coerce_float(cfg.get("take_profit_pct"), 10.0, 0.0, 100.0)
    cfg["stop_loss_pct"] = _coerce_float(cfg.get("stop_loss_pct"), 6.0, 0.0, 100.0)
    cfg["max_hold_minutes"] = _coerce_int(cfg.get("max_hold_minutes"), 120, 1, 10_080)
    cfg["session_timeout_seconds"] = _coerce_int(cfg.get("session_timeout_seconds"), 60, 1, 86_400)
    cfg["price_policy"] = str(cfg.get("price_policy") or "taker_limit").strip().lower() or "taker_limit"
    cfg["time_in_force"] = str(cfg.get("time_in_force") or "IOC").strip().upper() or "IOC"

    return StrategySDK.normalize_strategy_retention_config(cfg)


def _is_sports_market(market: Market) -> bool:
    """Check whether a market looks like a sports event."""
    text_parts: list[str] = []
    for attr in ("question", "slug", "category", "event_slug", "group_item_title"):
        val = getattr(market, attr, None)
        if val:
            text_parts.append(str(val))
    tags = getattr(market, "tags", None)
    if isinstance(tags, (list, tuple)):
        text_parts.extend(str(t) for t in tags)
    text = " ".join(text_parts).lower()
    return any(kw in text for kw in SPORT_KEYWORDS)


class SportsOverreactionFaderStrategy(BaseStrategy):
    """Fades overreactions in live sports markets.

    Monitors sports markets for rapid price swings that exceed the
    ``min_move_pct`` threshold within ``move_window_seconds``.  When a
    swing is detected, enters a position against the move — betting on
    mean-reversion of the overreaction (empirically ~50 % reversal per
    Moskowitz 2021).
    """

    strategy_type = "sports_overreaction_fader"
    name = "Sports Overreaction Fader"
    description = (
        "Fades rapid price overreactions in live sports markets. "
        "Enters against sharp moves and exits on partial mean-reversion."
    )
    mispricing_type = "within_market"
    source_key = "sports"
    requires_resolution_date = True
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.5,
        max_resolution_months=3.0,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=25.0,
        risk_penalty=8.0,
    )
    sizing_config = SizingConfig()

    default_config = sports_overreaction_defaults()
    pipeline_defaults = {
        "min_edge_percent": 0.5,
        "min_confidence": 0.30,
        "max_risk_score": 0.85,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = validate_sports_overreaction_config(self.default_config)

    def configure(self, config: dict) -> None:
        self.config = validate_sports_overreaction_config(config)

    @staticmethod
    def _token_for_side(market: Market, side: str) -> Optional[str]:
        token_ids = list(getattr(market, "clob_token_ids", []) or [])
        if side == "YES":
            return token_ids[0] if len(token_ids) > 0 else None
        return token_ids[1] if len(token_ids) > 1 else None

    def _measure_move(
        self,
        history: deque,
        current_price: float,
        window_seconds: float,
        now,
    ) -> tuple[float, float]:
        """Return (move_pct, baseline_price) over the window.

        move_pct is signed: positive means price went UP, negative means
        price went DOWN relative to the oldest point in the window.
        """
        if len(history) < 2:
            return 0.0, current_price
        baseline_price = current_price
        for ts, price in history:
            if not hasattr(ts, "astimezone"):
                continue
            age = (now - ts).total_seconds()
            if age <= window_seconds * 1.2:
                baseline_price = price
                break
        if baseline_price <= 0.0 or baseline_price >= 1.0:
            return 0.0, current_price
        move_pct = ((current_price - baseline_price) / baseline_price) * 100.0
        return move_pct, baseline_price

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        cfg = validate_sports_overreaction_config(getattr(self, "config", {}) or {})

        min_move_pct = to_float(cfg.get("min_move_pct"), 5.0)
        max_move_pct = to_float(cfg.get("max_move_pct"), 40.0)
        move_window_seconds = max(30.0, to_float(cfg.get("move_window_seconds"), 300.0))
        min_favorite_prob = to_float(cfg.get("min_favorite_prob"), 0.60)
        max_favorite_prob = to_float(cfg.get("max_favorite_prob"), 0.94)
        min_hours = to_float(cfg.get("min_hours_to_resolution"), 0.05)
        max_hours = to_float(cfg.get("max_hours_to_resolution"), 72.0)
        min_liquidity = to_float(cfg.get("min_liquidity"), 2000.0)
        max_spread_bps = to_float(cfg.get("max_spread_bps"), 200.0)
        require_spread = bool(cfg.get("require_spread_data", False))
        reversion_fraction = clamp(to_float(cfg.get("reversion_fraction"), 0.50), 0.1, 1.0)
        min_reversion_edge = max(0.001, to_float(cfg.get("min_reversion_edge"), 0.015))
        flow_lookback = max(30.0, to_float(cfg.get("flow_lookback_seconds"), 600.0))
        min_flow_volume = max(0.0, to_float(cfg.get("min_flow_volume_usd"), 50.0))
        max_opportunities = max(1, int(to_float(cfg.get("max_opportunities"), 50)))

        event_by_market: dict[str, Event] = {}
        for event in events:
            for mkt in event.markets:
                event_by_market[str(mkt.id)] = event

        now = utcnow().astimezone(timezone.utc)
        trade_tape_available = StrategySDK.is_ws_feed_started()
        history_by_key = self.state.setdefault("sports_price_history", {})
        candidates: list[tuple[float, Opportunity]] = []

        for market in markets:
            if market.closed or not market.active:
                continue
            if not _is_sports_market(market):
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
            if hours_to_resolution < min_hours or hours_to_resolution > max_hours:
                continue

            for outcome in ("YES", "NO"):
                current_price = to_float(StrategySDK.get_live_price(market, prices, side=outcome), 0.0)
                if current_price <= 0.01 or current_price >= 0.99:
                    continue

                # We care about the market-level favorite probability
                opposite_price = to_float(
                    StrategySDK.get_live_price(market, prices, side=("NO" if outcome == "YES" else "YES")),
                    0.0,
                )
                fav_prob = max(current_price, opposite_price)
                if fav_prob < min_favorite_prob or fav_prob > max_favorite_prob:
                    continue

                spread_bps = StrategySDK.get_spread_bps(market, prices, side=outcome)
                if spread_bps is None and require_spread and trade_tape_available:
                    continue
                if spread_bps is not None and spread_bps > max_spread_bps:
                    continue

                token_id = self._token_for_side(market, outcome)
                history_key = f"{market.id}:{outcome}"
                history_points = max(12, min(120, int(move_window_seconds / 10.0) + 8))
                history = history_by_key.get(history_key)
                if not isinstance(history, deque):
                    history = deque(maxlen=history_points)
                    history_by_key[history_key] = history
                history.append((now, float(current_price)))

                # Trim old points
                horizon = max(move_window_seconds * 1.5, 60.0)
                while history and isinstance(history[0], tuple):
                    pt_ts = history[0][0]
                    if not hasattr(pt_ts, "astimezone"):
                        history.popleft()
                        continue
                    if (now - pt_ts).total_seconds() <= horizon:
                        break
                    history.popleft()

                move_pct, baseline_price = self._measure_move(
                    history, current_price, move_window_seconds, now,
                )

                # Also try SDK price change if we have trade tape
                if abs(move_pct) < min_move_pct and token_id and trade_tape_available:
                    sdk_change = StrategySDK.get_price_change(token_id, lookback_seconds=int(move_window_seconds))
                    if isinstance(sdk_change, dict):
                        sdk_pct = to_float(sdk_change.get("change_pct"), 0.0)
                        if abs(sdk_pct) > abs(move_pct):
                            move_pct = sdk_pct
                            old_p = to_float(sdk_change.get("old_price"), 0.0)
                            if old_p > 0:
                                baseline_price = old_p

                abs_move = abs(move_pct)
                if abs_move < min_move_pct or abs_move > max_move_pct:
                    continue

                # Determine fade direction: we buy the side that DROPPED
                # If this outcome's price dropped (move_pct < 0) → buy this side
                # If this outcome's price rose  (move_pct > 0) → skip (the OTHER side dropped)
                if move_pct >= 0:
                    continue  # Price went up for this side — the overreaction to fade is on the other side

                # Price dropped for this outcome → this is the fading opportunity
                move_magnitude = abs(current_price - baseline_price)
                reversion_target = current_price + (move_magnitude * reversion_fraction)
                reversion_edge = reversion_target - current_price
                if reversion_edge < min_reversion_edge:
                    continue

                # Flow data (confirmatory, not required)
                flow_imbalance = 0.0
                recent_volume = 0.0
                flow_data_available = bool(token_id) and trade_tape_available

                if flow_data_available:
                    flow_imbalance = clamp(
                        to_float(StrategySDK.get_buy_sell_imbalance(token_id, lookback_seconds=flow_lookback), 0.0),
                        -1.0,
                        1.0,
                    )
                    vol_payload = StrategySDK.get_trade_volume(token_id, lookback_seconds=flow_lookback)
                    if isinstance(vol_payload, dict):
                        recent_volume = max(0.0, to_float(vol_payload.get("total"), 0.0))

                if flow_data_available and recent_volume < min_flow_volume:
                    continue

                # Surprise metric (Choi & Hui 2014): how surprising is this move?
                # If the *underdog's* price dropped, that's expected → less edge.
                # If the *favorite's* price dropped, that's surprising → more edge.
                is_favorite_dropping = current_price == fav_prob or (
                    baseline_price > 0.5 and current_price > 0.5
                )
                surprise_bonus = 0.12 if is_favorite_dropping else 0.0

                # Confidence model
                move_strength = clamp(abs_move / max(max_move_pct, 1.0), 0.0, 1.0)
                liquidity_score = clamp(
                    to_float(getattr(market, "liquidity", 0.0), 0.0) / max(min_liquidity * 5.0, 1.0),
                    0.2,
                    1.0,
                )
                spread_score = (
                    0.50
                    if spread_bps is None
                    else 1.0 - clamp(spread_bps / max(max_spread_bps, 1.0), 0.0, 1.0)
                )
                confidence = clamp(
                    0.25
                    + (move_strength * 0.30)
                    + (liquidity_score * 0.15)
                    + (spread_score * 0.10)
                    + surprise_bonus
                    + (max(0.0, flow_imbalance) * 0.08),
                    0.15,
                    0.92,
                )

                # Build position
                positions = [
                    {
                        "action": "BUY",
                        "outcome": outcome,
                        "price": current_price,
                        "token_id": token_id,
                        "entry_style": "sports_overreaction_fader",
                        "_sports_fader": {
                            "baseline_price": baseline_price,
                            "move_pct": move_pct,
                            "reversion_target": reversion_target,
                            "reversion_edge": reversion_edge,
                            "is_favorite_dropping": is_favorite_dropping,
                            "flow_imbalance": flow_imbalance,
                            "recent_volume": recent_volume,
                            "spread_bps": spread_bps,
                        },
                    }
                ]

                opportunity = self.create_opportunity(
                    title=f"Sports Fade: {outcome} {current_price:.1%} (moved {move_pct:+.1f}%)",
                    description=(
                        f"Sports overreaction detected: {outcome} dropped {abs_move:.1f}% "
                        f"from {baseline_price:.3f} to {current_price:.3f} within "
                        f"{move_window_seconds:.0f}s. Targeting {reversion_fraction:.0%} "
                        f"reversion to {reversion_target:.3f}."
                    ),
                    total_cost=current_price,
                    expected_payout=reversion_target,
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

                # Risk score
                time_safety = clamp(hours_to_resolution / max(max_hours, 1.0), 0.0, 1.0)
                spread_risk = (
                    0.40
                    if spread_bps is None
                    else clamp(spread_bps / max(max_spread_bps, 1.0), 0.0, 1.0)
                )
                move_risk = clamp(abs_move / max(max_move_pct, 1.0), 0.0, 1.0)
                risk_score = (
                    0.55
                    - (time_safety * 0.08)
                    + (spread_risk * 0.12)
                    + (move_risk * 0.15)
                    - (liquidity_score * 0.08)
                )
                opportunity.risk_score = clamp(risk_score, 0.25, 0.90)
                opportunity.risk_factors = [
                    f"Price moved {move_pct:+.1f}% in {move_window_seconds:.0f}s",
                    f"Reversion target {reversion_target:.3f} ({reversion_fraction:.0%} of move)",
                    f"Market resolves in {hours_to_resolution:.1f}h",
                    f"{'Favorite' if is_favorite_dropping else 'Underdog'} side dropped (surprise {'high' if is_favorite_dropping else 'low'})",
                ]
                opportunity.mispricing_type = MispricingType.WITHIN_MARKET
                opportunity.strategy_context = {
                    "entry_price": current_price,
                    "baseline_price": baseline_price,
                    "move_pct": move_pct,
                    "reversion_target": reversion_target,
                    "reversion_edge": reversion_edge,
                    "reversion_fraction": reversion_fraction,
                    "is_favorite_dropping": is_favorite_dropping,
                    "hours_to_resolution": hours_to_resolution,
                    "flow_imbalance": flow_imbalance,
                    "recent_volume_usd": recent_volume,
                    "spread_bps": spread_bps,
                    "fade_outcome": outcome,
                    "flow_data_available": flow_data_available,
                    "trade_tape_available": trade_tape_available,
                    # Sports-specific metadata from Gamma API
                    "game_start_time": getattr(market, "game_start_time", None),
                    "sports_market_type": getattr(market, "sports_market_type", None),
                    "line": getattr(market, "line", None),
                }

                strength = reversion_edge * confidence * max(0.01, 1.0 - opportunity.risk_score)
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
        strategy_type = (
            str(payload.get("strategy") or payload.get("strategy_type") or getattr(signal, "strategy_type", "") or "")
            .strip()
            .lower()
        )
        strategy_ok = strategy_type == self.strategy_type

        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        if entry_price <= 0:
            positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
            if positions:
                entry_price = to_float((positions[0] or {}).get("price"), 0.0)

        reversion_edge = to_float(
            payload.get("reversion_edge") or payload.get("strategy_context", {}).get("reversion_edge"),
            0.0,
        )
        min_rev = max(0.001, to_float(params.get("min_reversion_edge", 0.015), 0.015))

        move_pct = to_float(
            payload.get("move_pct") or payload.get("strategy_context", {}).get("move_pct"),
            0.0,
        )
        min_move = max(1.0, to_float(params.get("min_move_pct", 5.0), 5.0))

        payload["_reversion_edge"] = reversion_edge
        payload["_move_pct"] = move_pct

        return [
            DecisionCheck(
                "strategy",
                "Strategy type match",
                strategy_ok,
                detail=f"strategy_type={self.strategy_type}",
            ),
            DecisionCheck(
                "reversion_edge",
                "Minimum reversion edge",
                reversion_edge >= min_rev,
                score=reversion_edge,
                detail=f"min={min_rev:.4f}",
            ),
            DecisionCheck(
                "move_magnitude",
                "Sufficient price move",
                abs(move_pct) >= min_move,
                score=abs(move_pct),
                detail=f"min={min_move:.1f}%",
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
        reversion_edge = max(0.0, to_float(payload.get("_reversion_edge"), 0.0))
        move_pct = abs(to_float(payload.get("_move_pct"), 0.0))
        move_bonus = clamp(move_pct / 20.0, 0.0, 1.0) * 5.0
        return base + (reversion_edge * 100.0) + move_bonus

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        strategy_context = getattr(position, "strategy_context", None) or {}
        current_price = safe_float(market_state.get("current_price"), default=None)
        if current_price is None:
            return ExitDecision("hold", "No current price available")

        reversion_target = to_float(strategy_context.get("reversion_target"), 0.0)
        if reversion_target > 0.0 and current_price >= reversion_target:
            return ExitDecision(
                "close",
                f"Reversion target reached ({current_price:.4f} >= {reversion_target:.4f})",
                close_price=current_price,
            )

        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Phase 3: declarative pre-submit gates
    # ------------------------------------------------------------------
    #
    # Inline detect-time filters already enforce these; the gates run
    # the same predicates at signal-submit time so a stale/updated
    # liquidity or spread reading short-circuits before DB writes.

    def get_pre_submit_gates(self):
        from services.strategy_sdk import Gate, CostClass, GateContext, GateResult

        def _payload(ctx: GateContext) -> dict:
            payload = (ctx.extras or {}).get("signal_payload")
            if isinstance(payload, dict):
                return payload
            payload = getattr(ctx.runtime_signal, "payload_json", None)
            return payload if isinstance(payload, dict) else {}

        def _merged_params(ctx: GateContext) -> dict:
            merged = dict(self.default_config or {})
            merged.update(ctx.strategy_params or {})
            return merged

        def liquidity_floor_predicate(ctx: GateContext) -> GateResult:
            cfg = _merged_params(ctx)
            min_liq = float(safe_float(cfg.get("min_liquidity"), 0.0) or 0.0)
            if min_liq <= 0.0:
                return GateResult(passed=True)
            liq = safe_float(getattr(ctx.runtime_signal, "liquidity", None), None)
            if liq is None:
                liq = safe_float(_payload(ctx).get("liquidity"), None)
            if liq is None:
                return GateResult(passed=True)
            if float(liq) < min_liq:
                return GateResult(
                    passed=False,
                    reason="sports_fader_min_liquidity",
                    error_message=f"Liquidity {liq:.0f} < min {min_liq:.0f}",
                    detail={"liquidity": float(liq), "min_liquidity": min_liq},
                )
            return GateResult(passed=True)

        def max_spread_bps_predicate(ctx: GateContext) -> GateResult:
            cfg = _merged_params(ctx)
            cap_bps = float(safe_float(cfg.get("max_spread_bps"), 0.0) or 0.0)
            if cap_bps <= 0.0:
                return GateResult(passed=True)
            payload = _payload(ctx)
            book_spread_bps = safe_float(payload.get("book_spread_bps"), None)
            if book_spread_bps is None:
                # Fall through to L1 venue gate which has live book.
                return GateResult(passed=True)
            if float(book_spread_bps) > cap_bps:
                return GateResult(
                    passed=False,
                    reason="sports_fader_max_spread_bps",
                    error_message=(
                        f"Book spread {book_spread_bps:.0f}bps > cap {cap_bps:.0f}bps"
                    ),
                    detail={
                        "book_spread_bps": float(book_spread_bps),
                        "max_spread_bps": cap_bps,
                    },
                )
            return GateResult(passed=True)

        def favorite_probability_predicate(ctx: GateContext) -> GateResult:
            cfg = _merged_params(ctx)
            min_prob = float(safe_float(cfg.get("min_favorite_prob"), 0.0) or 0.0)
            if min_prob <= 0.0:
                return GateResult(passed=True)
            payload = _payload(ctx)
            fav_prob = safe_float(payload.get("_favorite_probability"), None)
            if fav_prob is None:
                fav_prob = safe_float(payload.get("favorite_probability"), None)
            if fav_prob is None:
                return GateResult(passed=True)
            if float(fav_prob) < min_prob:
                return GateResult(
                    passed=False,
                    reason="sports_fader_min_favorite_prob",
                    error_message=(
                        f"Favorite probability {fav_prob:.3f} < min {min_prob:.3f}"
                    ),
                    detail={
                        "favorite_probability": float(fav_prob),
                        "min_favorite_prob": min_prob,
                    },
                )
            return GateResult(passed=True)

        return [
            Gate(
                name="sports_fader_min_liquidity",
                cost_class=CostClass.L0_MEMORY,
                predicate=liquidity_floor_predicate,
            ),
            Gate(
                name="sports_fader_max_spread_bps",
                cost_class=CostClass.L0_MEMORY,
                predicate=max_spread_bps_predicate,
            ),
            Gate(
                name="sports_fader_min_favorite_prob",
                cost_class=CostClass.L0_MEMORY,
                predicate=favorite_probability_predicate,
            ),
        ]
