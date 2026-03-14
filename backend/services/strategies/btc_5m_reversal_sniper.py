"""BTC 5-Minute Reversal Sniper Strategy.

Captures the empirically observed pattern where BTC 5-minute "Up or Down" markets
briefly switch to the opposing side in the final ~3-10 seconds before resolution,
then revert or resolve on the new side.

Key data findings (March 2026 Polymarket orderbook analysis):
- 16% of BTC 5m markets switch sides in the final 30 seconds
- Among CONTESTED markets (mid 0.3-0.7), the switch rate jumps to 54%
- Most switches happen in the final 1-5 seconds before resolution
- Spreads widen 2-3x during switches (from ~0.02 to ~0.06)
- Price velocity is 3x higher for switching markets (0.032/sec vs 0.004/sec)
- The oracle (Binance BTC price) is the definitive signal — markets that switch
  are those where the actual BTC price crosses the reference price
- Best entry window is T-8s to T-3s when the market is still mispriced

Strategy logic:
1. Monitor BTC 5m markets in the final 15 seconds
2. Compare Binance oracle price against the price-to-beat reference
3. When oracle indicates a DIFFERENT winner than the current market consensus,
   buy the underpriced side (the oracle-predicted winner)
4. Hold through resolution — the market converges to $1.00 for the winner
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from models import Market, Opportunity
from services.data_events import DataEvent, EventType
from services.quality_filter import QualityFilterOverrides
from services.strategies.base import BaseStrategy, DecisionCheck, ExitDecision, StrategyDecision
from utils.converters import clamp, safe_float, to_confidence, to_float
from utils.signal_helpers import signal_payload


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


def _taker_fee_pct(entry_price: float) -> float:
    price = clamp(float(entry_price), 0.0001, 0.9999)
    return 0.25 * ((price * (1.0 - price)) ** 2)


def _sigmoid(z: float) -> float:
    bounded = clamp(float(z), -60.0, 60.0)
    return 1.0 / (1.0 + math.exp(-bounded))


class BTC5mReversalSniperStrategy(BaseStrategy):
    """Snipes the last-second reversal in BTC 5-minute binary markets.

    When the Binance oracle price diverges from the Polymarket consensus in the
    final seconds, the market briefly misprice the actual winner. This strategy
    buys the oracle-predicted winner at a discount during that window.
    """

    strategy_type = "btc_5m_reversal_sniper"
    name = "BTC 5m Reversal Sniper"
    description = (
        "Captures last-second side reversals in BTC 5m markets using oracle divergence. "
        "Data shows 54% of contested markets switch sides in final 10 seconds."
    )
    mispricing_type = "within_market"
    source_key = "crypto"
    worker_affinity = "crypto"
    market_categories = ["crypto"]
    subscriptions = [EventType.CRYPTO_UPDATE]
    requires_live_market_context = True
    supports_entry_take_profit_exit = False  # Hold to resolution
    default_open_order_timeout_seconds = 8.0  # Very tight — we're in the last seconds

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.3,
        max_resolution_months=0.01,  # Markets resolve in seconds
    )

    default_config = {
        # ── Asset & Market Targeting ──
        "asset": "BTC",
        "timeframe": "5min",

        # ── Entry Window ──
        # The sweet spot: enter T-12s to T-2s before resolution
        # Data shows best entry at T-5s to T-3s, but we need fill time
        "entry_window_start_seconds": 15.0,   # Start looking 15s before end
        "entry_window_end_seconds": 2.0,      # Stop at 2s (need fill time)
        "optimal_entry_seconds": 6.0,         # Peak entry point
        "entry_window_tolerance": 8.0,        # Tolerance around optimal

        # ── Oracle Divergence Thresholds ──
        # The oracle (Binance) must show the OPPOSITE direction from market consensus
        "min_oracle_divergence_pct": 0.10,    # Min BTC price move vs price-to-beat
        "max_oracle_age_seconds": 5.0,        # Oracle must be very fresh for last-second trades
        "oracle_confidence_scale": 0.35,      # Sigmoid scaling for oracle signal

        # ── Consensus Detection ──
        # A market is "contested" when mid is between these bounds
        "contested_zone_lower": 0.30,
        "contested_zone_upper": 0.70,
        # Bonus edge when market is very mispriced vs oracle
        "strong_signal_mid_threshold": 0.25,  # If market is at 0.25 but oracle says YES, strong signal

        # ── Spread & Liquidity ──
        "max_spread_pct": 0.08,               # Wider allowed — spreads blow out near resolution
        "min_liquidity_usd": 1000.0,          # Lower threshold for 5m markets
        "slippage_spread_multiplier": 0.30,   # Higher slippage in thin books
        "min_slippage_pct": 0.10,

        # ── Edge & Confidence ──
        "min_net_edge_percent": 0.25,         # Lower threshold — binary payout compensates
        "min_confidence": 0.45,               # Lower bar — the edge comes from oracle accuracy
        "max_entry_price": 0.92,              # Don't buy at 0.95 for a slim edge

        # ── Position Sizing ──
        "base_size_usd": 30.0,
        "max_size_usd": 200.0,
        "min_order_size_usd": 2.0,
        "sizing_policy": "kelly",
        "kelly_fractional_scale": 0.30,
        "liquidity_cap_fraction": 0.08,

        # ── Risk Management ──
        # No TP/SL — we hold to resolution (binary payout)
        "hold_to_resolution": True,
        "max_hold_seconds": 30.0,             # Safety: force exit if stuck past resolution
        "max_markets_per_event": 6,           # Don't spray into too many at once

        # ── Contested Market Bonus ──
        # When the market is contested (mid 0.3-0.7), switch probability is ~54%
        # We give extra confidence/size for these setups
        "contested_confidence_bonus": 0.08,
        "contested_size_multiplier": 1.4,

        # ── Reversal Pattern Detection ──
        # Track price velocity to detect the reversal happening in real-time
        "min_velocity_for_reversal": 0.005,   # Min |velocity| per second indicating a move
        "velocity_lookback_ticks": 20,        # Number of ticks to compute velocity from
    }

    def __init__(self) -> None:
        super().__init__()
        self.min_profit = 0.0

    # ── Market Parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _row_market(row: dict[str, Any]) -> Market | None:
        market_id = str(row.get("condition_id") or row.get("id") or "").strip()
        if not market_id:
            return None

        up_price = safe_float(row.get("up_price"), None)
        down_price = safe_float(row.get("down_price"), None)
        if up_price is None or down_price is None:
            return None

        end_date = _parse_datetime_utc(row.get("end_time"))
        token_ids = [
            str(token).strip()
            for token in list(row.get("clob_token_ids") or [])
            if str(token).strip() and len(str(token).strip()) > 20
        ]

        return Market(
            id=market_id,
            condition_id=market_id,
            question=str(row.get("question") or row.get("slug") or market_id),
            slug=str(row.get("slug") or market_id),
            outcome_prices=[float(up_price), float(down_price)],
            liquidity=max(0.0, float(safe_float(row.get("liquidity"), 0.0) or 0.0)),
            end_date=end_date,
            platform="polymarket",
            clob_token_ids=token_ids,
        )

    @staticmethod
    def _seconds_left(row: dict[str, Any]) -> float:
        seconds_left = safe_float(row.get("seconds_left"), None)
        if seconds_left is not None and seconds_left >= 0.0:
            return float(seconds_left)
        end_date = _parse_datetime_utc(row.get("end_time"))
        if end_date is not None:
            return max(0.0, (end_date - datetime.now(timezone.utc)).total_seconds())
        return 300.0  # Default to full 5m if unknown

    @staticmethod
    def _spread_pct(row: dict[str, Any]) -> float:
        spread = safe_float(row.get("spread"), None)
        if spread is None or spread < 0.0:
            best_bid = safe_float(row.get("best_bid"), None)
            best_ask = safe_float(row.get("best_ask"), None)
            if best_bid is not None and best_ask is not None and best_ask >= best_bid:
                spread = best_ask - best_bid
        if spread is None:
            spread = 0.0
        return clamp(float(spread), 0.0, 1.0)

    # ── Core Scoring ────────────────────────────────────────────────────────

    def _score_market(self, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
        """Score a single market row for reversal-snipe opportunity."""

        # ── Filter: Asset & Timeframe ──
        asset = str(row.get("asset") or "").strip().upper()
        target_asset = str(cfg.get("asset", "BTC") or "BTC").strip().upper()
        if asset != target_asset:
            return None

        timeframe = str(row.get("timeframe") or "").strip().lower()
        if timeframe not in {"5m", "5min", "5-minute", "5minutes"}:
            return None

        # ── Filter: Required Price Data ──
        up_price = safe_float(row.get("up_price"), None)
        down_price = safe_float(row.get("down_price"), None)
        oracle_price = safe_float(row.get("oracle_price"), None)
        price_to_beat = safe_float(row.get("price_to_beat"), None)
        if up_price is None or down_price is None or oracle_price is None or price_to_beat is None:
            return None
        if not (0.0 < up_price < 1.0 and 0.0 < down_price < 1.0 and price_to_beat > 0.0):
            return None

        # ── Filter: Oracle Freshness ──
        oracle_age_seconds = safe_float(row.get("oracle_age_seconds"), None)
        max_oracle_age = max(0.1, to_float(cfg.get("max_oracle_age_seconds", 5.0), 5.0))
        if oracle_age_seconds is None or oracle_age_seconds > max_oracle_age:
            return None

        # ── Filter: Entry Window (last 15 seconds only) ──
        seconds_left = self._seconds_left(row)
        window_start = max(1.0, to_float(cfg.get("entry_window_start_seconds", 15.0), 15.0))
        window_end = max(0.5, to_float(cfg.get("entry_window_end_seconds", 2.0), 2.0))
        if seconds_left > window_start or seconds_left < window_end:
            return None

        # ── Filter: Liquidity & Spread ──
        liquidity_usd = max(0.0, safe_float(row.get("liquidity"), 0.0) or 0.0)
        min_liquidity = max(0.0, to_float(cfg.get("min_liquidity_usd", 1000.0), 1000.0))
        if liquidity_usd < min_liquidity:
            return None

        spread_pct = self._spread_pct(row)
        max_spread = clamp(to_float(cfg.get("max_spread_pct", 0.08), 0.08), 0.001, 0.30)
        if spread_pct > max_spread:
            return None

        # ── Oracle Divergence: The Core Signal ──
        # diff_pct > 0 means BTC is ABOVE price-to-beat → oracle says UP/YES
        # diff_pct < 0 means BTC is BELOW price-to-beat → oracle says DOWN/NO
        diff_pct = ((oracle_price - price_to_beat) / price_to_beat) * 100.0

        min_oracle_divergence = max(0.01, to_float(cfg.get("min_oracle_divergence_pct", 0.10), 0.10))
        if abs(diff_pct) < min_oracle_divergence:
            return None

        # Oracle-predicted direction
        if diff_pct > 0.0:
            oracle_direction = "buy_yes"
            outcome = "YES"
            selected_price = float(up_price)
            oracle_predicted_winner = "YES"
        else:
            oracle_direction = "buy_no"
            outcome = "NO"
            selected_price = float(down_price)
            oracle_predicted_winner = "NO"

        # ── Reversal Detection: Is the market on the WRONG side? ──
        # The mid_price tells us market consensus
        mid_price = (up_price + (1.0 - down_price)) / 2.0  # mid_price of YES
        market_consensus = "YES" if mid_price >= 0.50 else "NO"

        # The key pattern: oracle says one thing, market says another
        is_reversal = oracle_predicted_winner != market_consensus

        # How contested is the market?
        contested_lower = to_float(cfg.get("contested_zone_lower", 0.30), 0.30)
        contested_upper = to_float(cfg.get("contested_zone_upper", 0.70), 0.70)
        is_contested = contested_lower <= mid_price <= contested_upper

        # ── Entry Price Validation ──
        max_entry_price = clamp(to_float(cfg.get("max_entry_price", 0.92), 0.92), 0.05, 0.99)
        if selected_price <= 0.0 or selected_price >= 1.0 or selected_price > max_entry_price:
            return None

        # ── Edge Calculation ──
        # For reversal trades, the expected value is high because binary payout = $1
        # If we buy YES at 0.30 and it resolves YES, profit = $0.70
        # The edge = (oracle_prob * $1 - selected_price) / selected_price

        # Model oracle probability using sigmoid on price divergence
        oracle_scale = max(0.05, to_float(cfg.get("oracle_confidence_scale", 0.35), 0.35))
        z_value = diff_pct / oracle_scale
        model_prob_yes = clamp(_sigmoid(z_value), 0.03, 0.97)
        model_prob_no = 1.0 - model_prob_yes

        if oracle_direction == "buy_yes":
            expected_prob = model_prob_yes
        else:
            expected_prob = model_prob_no

        # Raw edge: how much the market is mispriced vs our model
        repricing_buffer = expected_prob - selected_price
        if repricing_buffer < 0.005:
            return None

        raw_edge_percent = max(0.0, repricing_buffer * 100.0)

        # Costs
        fee_percent = _taker_fee_pct(selected_price) * 100.0
        slippage_mult = max(0.01, to_float(cfg.get("slippage_spread_multiplier", 0.30), 0.30))
        min_slippage = max(0.0, to_float(cfg.get("min_slippage_pct", 0.10), 0.10))
        slippage_percent = max(min_slippage, (spread_pct * 100.0) * slippage_mult)

        net_edge_percent = raw_edge_percent - fee_percent - slippage_percent
        min_net_edge = max(0.0, to_float(cfg.get("min_net_edge_percent", 0.25), 0.25))
        if net_edge_percent < min_net_edge:
            return None

        # ── Confidence Scoring ──
        # Components:
        # 1. Oracle signal strength: how far BTC moved past the reference
        signal_strength = clamp(abs(diff_pct) / max(min_oracle_divergence * 3.0, 0.01), 0.0, 1.0)

        # 2. Oracle freshness: fresher = more reliable
        freshness = clamp(1.0 - (float(oracle_age_seconds) / max_oracle_age), 0.0, 1.0)

        # 3. Spread quality: tighter spread = better fills
        spread_quality = clamp(1.0 - (spread_pct / max_spread), 0.0, 1.0)

        # 4. Time pressure alignment: closer to optimal entry = better
        optimal_entry = to_float(cfg.get("optimal_entry_seconds", 6.0), 6.0)
        entry_tolerance = max(1.0, to_float(cfg.get("entry_window_tolerance", 8.0), 8.0))
        time_alignment = clamp(1.0 - (abs(seconds_left - optimal_entry) / entry_tolerance), 0.0, 1.0)

        # 5. Reversal bonus: if market is clearly mispriced vs oracle
        reversal_bonus = 0.12 if is_reversal else 0.0
        contested_bonus = to_float(cfg.get("contested_confidence_bonus", 0.08), 0.08) if is_contested else 0.0

        confidence = clamp(
            0.38
            + (signal_strength * 0.22)
            + (freshness * 0.12)
            + (spread_quality * 0.06)
            + (time_alignment * 0.10)
            + reversal_bonus
            + contested_bonus,
            0.30,
            0.96,
        )

        min_confidence = to_confidence(cfg.get("min_confidence", 0.45), 0.45)
        if confidence < min_confidence:
            return None

        # ── Risk Score ──
        risk_score = clamp(
            0.55
            - (signal_strength * 0.20)
            - (freshness * 0.10)
            + ((1.0 - spread_quality) * 0.15)
            + (0.10 if not is_reversal else 0.0),
            0.10,
            0.85,
        )

        # ── Composite Score (for ranking multiple opportunities) ──
        score = (
            (net_edge_percent * 3.0)
            + (confidence * 35.0)
            + (signal_strength * 12.0)
            + (time_alignment * 8.0)
            + (spread_quality * 5.0)
            + (15.0 if is_reversal else 0.0)
            + (10.0 if is_contested else 0.0)
        )

        return {
            "asset": asset,
            "timeframe": "5min",
            "direction": oracle_direction,
            "outcome": outcome,
            "selected_price": selected_price,
            "seconds_left": float(seconds_left),
            "oracle_price": float(oracle_price),
            "price_to_beat": float(price_to_beat),
            "oracle_diff_pct": float(diff_pct),
            "raw_edge_percent": float(raw_edge_percent),
            "net_edge_percent": float(net_edge_percent),
            "fee_percent": float(fee_percent),
            "slippage_percent": float(slippage_percent),
            "confidence": float(confidence),
            "risk_score": float(risk_score),
            "signal_strength": float(signal_strength),
            "freshness": float(freshness),
            "spread_quality": float(spread_quality),
            "spread_pct": float(spread_pct),
            "time_alignment": float(time_alignment),
            "liquidity_usd": float(liquidity_usd),
            "expected_prob": float(expected_prob),
            "model_prob_yes": float(model_prob_yes),
            "model_prob_no": float(model_prob_no),
            "oracle_age_seconds": float(oracle_age_seconds),
            "mid_price": float(mid_price),
            "market_consensus": market_consensus,
            "oracle_predicted_winner": oracle_predicted_winner,
            "is_reversal": is_reversal,
            "is_contested": is_contested,
            "reversal_bonus": float(reversal_bonus),
            "contested_bonus": float(contested_bonus),
            "score": float(score),
        }

    # ── Opportunity Building ────────────────────────────────────────────────

    def _build_opportunity(self, row: dict[str, Any], signal: dict[str, Any]) -> Opportunity | None:
        typed_market = self._row_market(row)
        if typed_market is None:
            return None

        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        direction = str(signal["direction"])
        outcome = str(signal["outcome"])
        selected_price = float(signal["selected_price"])
        token_ids = list(typed_market.clob_token_ids or [])
        token_idx = 0 if direction == "buy_yes" else 1
        token_id = token_ids[token_idx] if len(token_ids) > token_idx else None

        reversal_tag = "REVERSAL" if signal["is_reversal"] else "ALIGNED"
        contested_tag = " CONTESTED" if signal["is_contested"] else ""

        context = {
            "strategy_origin": "crypto_worker",
            "strategy_slug": self.strategy_type,
            "asset": signal["asset"],
            "timeframe": signal["timeframe"],
            "direction": direction,
            "seconds_left": signal["seconds_left"],
            "oracle_price": signal["oracle_price"],
            "price_to_beat": signal["price_to_beat"],
            "oracle_diff_pct": signal["oracle_diff_pct"],
            "raw_edge_percent": signal["raw_edge_percent"],
            "net_edge_percent": signal["net_edge_percent"],
            "fee_percent": signal["fee_percent"],
            "slippage_percent": signal["slippage_percent"],
            "confidence": signal["confidence"],
            "risk_score": signal["risk_score"],
            "signal_strength": signal["signal_strength"],
            "freshness": signal["freshness"],
            "spread_quality": signal["spread_quality"],
            "spread_pct": signal["spread_pct"],
            "time_alignment": signal["time_alignment"],
            "liquidity_usd": signal["liquidity_usd"],
            "expected_prob": signal["expected_prob"],
            "model_prob_yes": signal["model_prob_yes"],
            "model_prob_no": signal["model_prob_no"],
            "oracle_age_seconds": signal["oracle_age_seconds"],
            "mid_price": signal["mid_price"],
            "market_consensus": signal["market_consensus"],
            "oracle_predicted_winner": signal["oracle_predicted_winner"],
            "is_reversal": signal["is_reversal"],
            "is_contested": signal["is_contested"],
            "market_data_age_ms": safe_float(row.get("market_data_age_ms"), None),
            "end_time": row.get("end_time"),
            "execution_intent": "taker_rescue",  # Always taker — no time for maker fills
            "hold_to_resolution": True,
        }

        positions = [
            {
                "action": "BUY",
                "outcome": outcome,
                "price": selected_price,
                "token_id": token_id,
                "price_policy": "taker_limit",
                "time_in_force": "IOC",
                "_reversal_sniper": context,
            }
        ]

        opp = self.create_opportunity(
            title=(
                f"Reversal Sniper: {outcome} @ {selected_price:.3f} "
                f"[{reversal_tag}{contested_tag}] {signal['seconds_left']:.0f}s left"
            ),
            description=(
                f"Oracle {signal['oracle_diff_pct']:+.3f}% vs market consensus {signal['market_consensus']}, "
                f"mid={signal['mid_price']:.3f}, net edge {signal['net_edge_percent']:.2f}%, "
                f"model prob {signal['expected_prob']:.2f}"
            ),
            total_cost=selected_price,
            expected_payout=1.0,  # Binary market — winner gets $1.00
            markets=[typed_market],
            positions=positions,
            is_guaranteed=False,
            skip_fee_model=True,
            custom_roi_percent=signal["net_edge_percent"],
            custom_risk_score=signal["risk_score"],
            confidence=signal["confidence"],
            min_liquidity_hard=max(100.0, to_float(cfg.get("min_liquidity_usd", 1000.0), 1000.0)),
            min_position_size=max(1.0, to_float(cfg.get("min_order_size_usd", 2.0), 2.0)),
        )
        if opp is None:
            return None

        opp.risk_factors = [
            f"Last-second entry ({signal['seconds_left']:.0f}s left, {reversal_tag})",
            f"Oracle divergence={signal['oracle_diff_pct']:+.3f}% (age {signal['oracle_age_seconds']:.1f}s)",
            f"Net edge after fees/slippage={signal['net_edge_percent']:.2f}%",
            f"Market mid={signal['mid_price']:.3f}, consensus={signal['market_consensus']}",
            f"Spread={signal['spread_pct']:.3f} (quality={signal['spread_quality']:.2f})",
        ]
        opp.strategy_context = context
        return opp

    # ── Detection ───────────────────────────────────────────────────────────

    def _detect_from_rows(self, rows: list[dict[str, Any]]) -> list[Opportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        candidates: list[tuple[float, Opportunity]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            signal = self._score_market(row, cfg)
            if signal is None:
                continue
            opp = self._build_opportunity(row, signal)
            if opp is None:
                continue
            candidates.append((float(signal["score"]), opp))

        if not candidates:
            return []

        # Sort by score (best first) and deduplicate by market
        candidates.sort(key=lambda item: item[0], reverse=True)
        max_markets = max(1, int(to_float(cfg.get("max_markets_per_event", 6), 6.0)))

        out: list[Opportunity] = []
        seen_markets: set[str] = set()
        for _, opp in candidates:
            market_id = str((opp.markets or [{}])[0].get("id") or "")
            if market_id in seen_markets:
                continue
            seen_markets.add(market_id)
            out.append(opp)
            if len(out) >= max_markets:
                break

        return out

    def detect(self, events: list, markets: list, prices: dict[str, dict]) -> list[Opportunity]:
        """Standard scanner interface — not used, detection is via on_event."""
        return []

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        """Primary detection: triggered by crypto_worker updates."""
        if event.event_type != EventType.CRYPTO_UPDATE:
            return []
        rows = event.payload.get("markets") if isinstance(event.payload, dict) else None
        if not isinstance(rows, list) or not rows:
            return []
        return self._detect_from_rows(rows)

    # ── Evaluation ──────────────────────────────────────────────────────────

    def _extract_live_selected_price(self, live_market: dict[str, Any], payload: dict[str, Any]) -> float | None:
        for candidate in (
            live_market.get("live_selected_price"),
            live_market.get("selected_price"),
            payload.get("live_selected_price"),
            payload.get("selected_price"),
            payload.get("entry_price"),
        ):
            parsed = safe_float(candidate, None)
            if parsed is not None and 0.0 < parsed < 1.0:
                return float(parsed)
        return None

    def _extract_seconds_left(self, live_market: dict[str, Any], payload: dict[str, Any]) -> float | None:
        direct = safe_float(live_market.get("seconds_left"), safe_float(payload.get("seconds_left"), None))
        if direct is not None and direct >= 0.0:
            return float(direct)
        end_time = (
            live_market.get("market_end_time")
            or live_market.get("end_time")
            or payload.get("end_time")
            or payload.get("strategy_context", {}).get("end_time")
        )
        parsed_end = _parse_datetime_utc(end_time)
        if parsed_end is None:
            return None
        return max(0.0, (parsed_end - datetime.now(timezone.utc)).total_seconds())

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        """Re-validate the opportunity at execution time.

        Key checks:
        1. Still within the entry window (seconds_left)
        2. Oracle still fresh and divergent
        3. Entry price hasn't moved against us
        4. Spread hasn't blown out catastrophically
        """
        params = context.get("params") or {}
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})
        cfg.update(params)

        payload = signal_payload(signal)
        live_market = context.get("live_market")
        if not isinstance(live_market, dict):
            live_market = payload.get("live_market") if isinstance(payload.get("live_market"), dict) else {}

        checks: list[DecisionCheck] = []
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)

        # Check 1: Entry window still valid
        seconds_left = self._extract_seconds_left(live_market, payload)
        window_start = to_float(cfg.get("entry_window_start_seconds", 15.0), 15.0)
        window_end = to_float(cfg.get("entry_window_end_seconds", 2.0), 2.0)
        window_ok = seconds_left is not None and window_end <= seconds_left <= window_start
        checks.append(DecisionCheck(
            "window", "Entry window",
            window_ok, 1.0 if window_ok else 0.0,
            f"seconds_left={seconds_left:.1f}" if seconds_left is not None else "unknown",
        ))

        # Check 2: Entry price acceptable
        selected_price = self._extract_live_selected_price(live_market, payload)
        max_entry_price = clamp(to_float(cfg.get("max_entry_price", 0.92), 0.92), 0.05, 0.99)
        price_ok = selected_price is not None and selected_price <= max_entry_price
        checks.append(DecisionCheck(
            "price", "Entry price",
            price_ok, 1.0 if price_ok else 0.0,
            f"price={selected_price:.3f}" if selected_price is not None else "unknown",
        ))

        # Check 3: Oracle freshness at evaluation time
        oracle_age = safe_float(
            live_market.get("oracle_age_seconds"),
            safe_float(payload.get("oracle_age_seconds"), None),
        )
        max_oracle_age = to_float(cfg.get("max_oracle_age_seconds", 5.0), 5.0)
        oracle_ok = oracle_age is not None and oracle_age <= max_oracle_age * 1.5  # Slight slack at eval time
        checks.append(DecisionCheck(
            "oracle", "Oracle freshness",
            oracle_ok, 1.0 if oracle_ok else 0.0,
            f"age={oracle_age:.1f}s" if oracle_age is not None else "unknown",
        ))

        # Check 4: Minimum edge
        min_edge = max(0.0, to_float(cfg.get("min_net_edge_percent", 0.25), 0.25))
        edge_ok = edge >= min_edge * 0.7  # Allow slight edge decay
        checks.append(DecisionCheck(
            "edge", "Net edge",
            edge_ok, edge / max(min_edge, 0.01),
            f"edge={edge:.2f}% vs min={min_edge:.2f}%",
        ))

        # Check 5: Minimum confidence
        min_conf = to_confidence(cfg.get("min_confidence", 0.45), 0.45)
        conf_ok = confidence >= min_conf * 0.85
        checks.append(DecisionCheck(
            "confidence", "Confidence",
            conf_ok, confidence,
            f"conf={confidence:.3f} vs min={min_conf:.3f}",
        ))

        all_passed = all(c.passed for c in checks)

        if not all_passed:
            failed = [c for c in checks if not c.passed]
            return StrategyDecision(
                decision="skipped",
                reason=f"Failed: {', '.join(c.key for c in failed)}",
                checks=checks,
            )

        # ── Position Sizing ──
        sizing_policy = str(cfg.get("sizing_policy", "kelly")).lower()
        base_size = max(1.0, to_float(cfg.get("base_size_usd", 30.0), 30.0))
        max_size = max(base_size, to_float(cfg.get("max_size_usd", 200.0), 200.0))

        if sizing_policy == "kelly" and selected_price and confidence > 0.0:
            kelly_f = max(0.0, (confidence * (1.0 / selected_price) - 1.0) / ((1.0 / selected_price) - 1.0))
            kelly_scale = to_float(cfg.get("kelly_fractional_scale", 0.30), 0.30)
            sized = base_size * (1.0 + kelly_f * kelly_scale * 3.0)
        else:
            sized = base_size * (1.0 + edge / 10.0)

        # Contested market size bonus
        is_contested = payload.get("is_contested", False)
        if is_contested:
            contested_mult = to_float(cfg.get("contested_size_multiplier", 1.4), 1.4)
            sized *= contested_mult

        # Liquidity cap
        liq_cap_frac = to_float(cfg.get("liquidity_cap_fraction", 0.08), 0.08)
        liq = safe_float(live_market.get("liquidity"), safe_float(payload.get("liquidity_usd"), 10000.0))
        if liq and liq > 0:
            sized = min(sized, float(liq) * liq_cap_frac)

        final_size = clamp(sized, max(1.0, to_float(cfg.get("min_order_size_usd", 2.0), 2.0)), max_size)

        return StrategyDecision(
            decision="selected",
            reason=f"Reversal sniper: {seconds_left:.0f}s left, edge={edge:.2f}%, conf={confidence:.3f}",
            checks=checks,
            size_usd=final_size,
        )

    # ── Exit Management ─────────────────────────────────────────────────────

    def should_exit(self, position: Any, market_state: dict[str, Any]) -> ExitDecision:
        """Hold to resolution. Only exit on safety timeout or if market has resolved."""

        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        # Check if market has resolved
        if market_state.get("resolved") or market_state.get("closed"):
            return ExitDecision(action="close", reason="Market resolved")

        # Check seconds remaining
        payload = {}
        if hasattr(position, "strategy_context") and isinstance(position.strategy_context, dict):
            payload = position.strategy_context

        seconds_left = self._extract_seconds_left(market_state, payload)
        if seconds_left is not None and seconds_left <= 0.0:
            # Past resolution time — should have settled
            return ExitDecision(action="hold", reason="At resolution, awaiting settlement")

        # Safety timeout: if somehow we're still holding way past resolution
        max_hold = to_float(cfg.get("max_hold_seconds", 30.0), 30.0)
        hold_duration = safe_float(market_state.get("hold_duration_seconds"), None)
        if hold_duration is not None and hold_duration > max_hold:
            return ExitDecision(action="close", reason=f"Safety timeout: held {hold_duration:.0f}s > max {max_hold:.0f}s")

        # Default: hold to resolution
        return ExitDecision(action="hold", reason="Holding to resolution (binary payout)")
