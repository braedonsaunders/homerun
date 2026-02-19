"""
Strategy: Behavioral Bias Exploitation (Bias Fader)

Exploits three well-documented behavioral biases in prediction markets:

1. **Anchoring at Round Numbers** -- Prices stuck at 0.25 / 0.50 / 0.75
   for many scan cycles indicate human anchoring. Near resolution, these
   prices should be moving toward 0 or 1. Fade the anchor.

2. **Recency Overreaction** -- After a large price move, partial retrace
   signals mean-reversion opportunity. The remaining excess is the edge.

3. **Disposition Effect** -- After a market rallies significantly from
   neutral (0.50), holders take profits causing a dip. The dip is
   temporary and represents an entry opportunity.

NOT risk-free. These are statistical tendencies, not structural arbitrage.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from models import Market, Event, Opportunity
from .base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig, utcnow, make_aware
from services.quality_filter import QualityFilterOverrides
from utils.kelly import kelly_fraction
from utils.logger import get_logger

logger = get_logger(__name__)

# Round numbers where human anchoring bias is strongest
ROUND_NUMBERS = [0.25, 0.50, 0.75]


class BiasFaderStrategy(BaseStrategy):
    """Fade behavioral biases: anchoring, overreaction, and disposition effect."""

    strategy_type = "bias_fader"
    name = "Bias Fader"
    description = "Fade behavioral biases: anchoring, overreaction, and disposition effect"
    mispricing_type = "within_market"
    realtime_processing_mode = "incremental"
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.0,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.50,
        confidence_weight=22.0,
        risk_penalty=8.0,
    )
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.70,
        risk_scale_factor=0.35,
        risk_floor=0.55,
    )

    default_config = {
        "min_edge_percent": 2.0,
        "min_confidence": 0.45,
        "max_risk_score": 0.78,
        "anchor_proximity_cents": 0.02,
        "anchor_min_scans_stuck": 5,
        "overreaction_min_move_pct": 10.0,
        "overreaction_min_retrace_pct": 30.0,
        "overreaction_lookback_seconds": 21600,
        "disposition_gain_threshold_pct": 10.0,
        "disposition_dip_threshold_pct": 3.0,
        "min_liquidity": 500.0,
        "base_size_usd": 15.0,
        "max_size_usd": 140.0,
        "max_opportunities": 20,
    }

    pipeline_defaults = {
        "min_edge_percent": 2.0,
        "min_confidence": 0.45,
        "max_risk_score": 0.78,
        "base_size_usd": 15.0,
        "max_size_usd": 140.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _live_yes_price(market: Market, prices: dict[str, dict]) -> float:
        """Return the best available YES price (live > static)."""
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            yes_token = market.clob_token_ids[0]
            if yes_token in prices:
                yes_price = prices[yes_token].get("mid", yes_price)
        return yes_price

    @staticmethod
    def _live_no_price(market: Market, prices: dict[str, dict]) -> float:
        """Return the best available NO price (live > static)."""
        no_price = market.no_price
        if market.clob_token_ids and len(market.clob_token_ids) > 1:
            no_token = market.clob_token_ids[1]
            if no_token in prices:
                no_price = prices[no_token].get("mid", no_price)
        return no_price

    # ------------------------------------------------------------------
    # Sub-detector 1: Anchoring at Round Numbers
    # ------------------------------------------------------------------

    def _detect_anchoring(
        self,
        market: Market,
        yes_price: float,
        no_price: float,
        prices: dict[str, dict],
        cfg: dict,
    ) -> Optional[dict]:
        """Detect prices stuck near round numbers.

        When a price has been stuck at a round number (0.25, 0.50, 0.75)
        for several consecutive scans and the market is approaching
        resolution, the price should be moving toward 0 or 1. Fade the
        anchor toward the implied direction.

        Returns:
            Dict with opportunity metadata or None.
        """
        proximity = float(cfg.get("anchor_proximity_cents", 0.02))
        min_scans = int(cfg.get("anchor_min_scans_stuck", 5))

        anchoring = self.state.setdefault("anchoring_count", {})

        matched_anchor = None
        for rn in ROUND_NUMBERS:
            if abs(yes_price - rn) <= proximity:
                matched_anchor = rn
                break

        if matched_anchor is None:
            # Not near a round number -- reset counter
            anchoring.pop(market.id, None)
            return None

        anchoring[market.id] = anchoring.get(market.id, 0) + 1

        if anchoring[market.id] < min_scans:
            return None

        # Price has been stuck at this round number for N scans.
        # Estimate direction: if near resolution, price should move
        # away from 0.50 toward 0 or 1.
        resolution_date = getattr(market, "end_date", None)
        if not resolution_date:
            return None

        days_left = (make_aware(resolution_date) - utcnow()).total_seconds() / 86400.0
        if days_left <= 0 or days_left > 60:
            # Too far from resolution for anchoring to matter
            return None

        # Stronger signal closer to resolution
        if days_left > 30:
            return None

        # Use price history to estimate direction (momentum)
        price_history = self.state.setdefault("price_history", {})
        history = price_history.get(market.id, [])

        # Estimate direction from recent momentum
        if len(history) >= 3:
            recent_prices = [p for _, p in history[-5:]]
            avg_recent = sum(recent_prices) / len(recent_prices)
            if avg_recent > matched_anchor:
                # Trending up, but stuck at anchor -- push toward 1.0
                direction = "up"
            elif avg_recent < matched_anchor:
                # Trending down, but stuck at anchor -- push toward 0.0
                direction = "down"
            else:
                # No clear direction
                return None
        else:
            # Not enough history for direction; skip
            return None

        # Calculate edge: distance from anchor to expected fair value
        # Near resolution, fair value should be moving away from 0.50
        time_pressure = max(0.0, 1.0 - (days_left / 30.0))  # 0..1

        if direction == "up":
            # Fair value is higher than anchor
            fair_estimate = matched_anchor + 0.05 * time_pressure
            edge = fair_estimate - yes_price
            outcome = "YES"
            buy_price = yes_price
            token_id = market.clob_token_ids[0] if market.clob_token_ids and len(market.clob_token_ids) > 0 else None
        else:
            # Fair value is lower than anchor -- buy NO
            fair_estimate = matched_anchor - 0.05 * time_pressure
            edge = yes_price - fair_estimate
            outcome = "NO"
            buy_price = no_price
            token_id = market.clob_token_ids[1] if market.clob_token_ids and len(market.clob_token_ids) > 1 else None

        if edge <= 0.005:
            return None

        edge_pct = (edge / max(buy_price, 0.01)) * 100.0
        # Lower confidence for anchoring (weakest signal)
        confidence = min(0.70, 0.45 + time_pressure * 0.15)

        return {
            "bias_type": "anchoring",
            "market": market,
            "outcome": outcome,
            "buy_price": buy_price,
            "token_id": token_id,
            "edge_pct": edge_pct,
            "confidence": confidence,
            "anchor_value": matched_anchor,
            "scans_stuck": anchoring[market.id],
            "days_left": days_left,
            "direction": direction,
        }

    # ------------------------------------------------------------------
    # Sub-detector 2: Recency Overreaction
    # ------------------------------------------------------------------

    def _detect_overreaction(
        self,
        market: Market,
        yes_price: float,
        prices: dict[str, dict],
        cfg: dict,
        scan_time: float,
    ) -> Optional[dict]:
        """Detect overreaction after large moves -- fade the excess.

        After a large price move, a partial retrace indicates the market
        overshot. The remaining excess from the peak is the tradeable edge.

        Returns:
            Dict with opportunity metadata or None.
        """
        min_move_pct = float(cfg.get("overreaction_min_move_pct", 10.0))
        min_retrace_pct = float(cfg.get("overreaction_min_retrace_pct", 30.0))
        lookback_seconds = float(cfg.get("overreaction_lookback_seconds", 21600))

        price_history = self.state.setdefault("price_history", {})

        if market.id not in price_history:
            price_history[market.id] = []
        price_history[market.id].append((scan_time, yes_price))

        # Keep bounded
        if len(price_history[market.id]) > 200:
            price_history[market.id] = price_history[market.id][-200:]

        history = price_history[market.id]
        if len(history) < 3:
            return None

        lookback_cutoff = scan_time - lookback_seconds
        # Find the price at the lookback point
        baseline = None
        for ts, p in history:
            if ts >= lookback_cutoff:
                baseline = p
                break
        if baseline is None:
            return None

        # Calculate move
        move = yes_price - baseline
        move_pct = abs(move) / max(baseline, 0.01) * 100.0

        if move_pct < min_move_pct:
            return None

        # Check for retrace: after a big move, has there been a partial reversal?
        if move > 0:
            peak = max(p for _, p in history)
        else:
            peak = min(p for _, p in history)

        peak_to_baseline = abs(peak - baseline)
        if peak_to_baseline < 0.01:
            return None

        retrace_from_peak = abs(peak - yes_price) / max(peak_to_baseline, 0.01) * 100.0

        if retrace_from_peak < min_retrace_pct:
            return None

        # Overreaction with partial retrace detected -- fade the remaining excess
        # Buy in the direction of the retrace (mean reversion)
        remaining_excess = abs(peak - yes_price)
        edge_pct = remaining_excess / max(yes_price, 0.01) * 100.0

        if edge_pct < 2.0:
            return None

        no_price = self._live_no_price(market, prices)

        if move > 0:
            # Price moved up then retraced -- it may continue down (buy NO)
            outcome = "NO"
            buy_price = no_price
            token_id = market.clob_token_ids[1] if market.clob_token_ids and len(market.clob_token_ids) > 1 else None
        else:
            # Price moved down then retraced up -- it may continue up (buy YES)
            outcome = "YES"
            buy_price = yes_price
            token_id = market.clob_token_ids[0] if market.clob_token_ids and len(market.clob_token_ids) > 0 else None

        # Higher confidence for overreaction (stronger signal)
        confidence = min(0.80, 0.55 + (retrace_from_peak - min_retrace_pct) / 100.0 * 0.3)

        return {
            "bias_type": "overreaction",
            "market": market,
            "outcome": outcome,
            "buy_price": buy_price,
            "token_id": token_id,
            "edge_pct": edge_pct,
            "confidence": confidence,
            "move_pct": move_pct,
            "retrace_pct": retrace_from_peak,
            "baseline": baseline,
            "peak": peak,
            "direction": "down" if move > 0 else "up",
        }

    # ------------------------------------------------------------------
    # Sub-detector 3: Disposition Effect
    # ------------------------------------------------------------------

    def _detect_disposition(
        self,
        market: Market,
        yes_price: float,
        prices: dict[str, dict],
        cfg: dict,
        scan_time: float,
    ) -> Optional[dict]:
        """Detect disposition selling: price dip after gains.

        After a market rallies from neutral (0.50), holders take profits
        (disposition effect) causing a dip. The dip is temporary.

        Returns:
            Dict with opportunity metadata or None.
        """
        gain_threshold = float(cfg.get("disposition_gain_threshold_pct", 10.0))
        dip_threshold = float(cfg.get("disposition_dip_threshold_pct", 3.0))

        peak_prices = self.state.setdefault("peak_prices", {})

        # Track running peak
        if market.id not in peak_prices:
            peak_prices[market.id] = (yes_price, scan_time)
        else:
            stored_peak, stored_time = peak_prices[market.id]
            if yes_price > stored_peak:
                peak_prices[market.id] = (yes_price, scan_time)

        peak, peak_time = peak_prices[market.id]
        if peak <= 0.20:
            # Not interesting for low-priced markets
            return None

        gain_from_50c = (peak - 0.50) / 0.50 * 100.0  # gain from neutral
        if gain_from_50c < gain_threshold:
            return None

        # Check for dip from peak
        dip_pct = (peak - yes_price) / max(peak, 0.01) * 100.0
        if dip_pct < dip_threshold or dip_pct > 15.0:
            # Below threshold or too large (may be fundamental, not disposition)
            return None

        # Dip from peak after gains -- likely disposition selling
        edge_pct = dip_pct  # the dip IS the edge (expect reversion)

        token_id = market.clob_token_ids[0] if market.clob_token_ids and len(market.clob_token_ids) > 0 else None

        # Moderate confidence
        confidence = min(0.75, 0.50 + (dip_pct - dip_threshold) / 12.0 * 0.2)

        return {
            "bias_type": "disposition",
            "market": market,
            "outcome": "YES",
            "buy_price": yes_price,
            "token_id": token_id,
            "edge_pct": edge_pct,
            "confidence": confidence,
            "peak": peak,
            "gain_from_neutral_pct": gain_from_50c,
            "dip_pct": dip_pct,
        }

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        """Detect behavioral bias opportunities across all three sub-detectors."""
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        min_liquidity = float(cfg.get("min_liquidity", 500.0))
        max_opportunities = int(cfg.get("max_opportunities", 20))
        scan_time = time.time()

        # Map market_id -> Event for opportunity creation
        market_to_event: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                market_to_event[m.id] = event

        # Collect candidates from all sub-detectors
        candidates: dict[str, dict] = {}  # market_id -> best candidate

        for market in markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) < 2:
                continue
            if market.liquidity < min_liquidity:
                continue

            yes_price = self._live_yes_price(market, prices)
            no_price = self._live_no_price(market, prices)

            # Skip extreme prices
            if yes_price < 0.03 or yes_price > 0.97:
                continue

            # Run all three sub-detectors
            results = []

            anchoring = self._detect_anchoring(market, yes_price, no_price, prices, cfg)
            if anchoring:
                results.append(anchoring)

            overreaction = self._detect_overreaction(market, yes_price, prices, cfg, scan_time)
            if overreaction:
                results.append(overreaction)

            disposition = self._detect_disposition(market, yes_price, prices, cfg, scan_time)
            if disposition:
                results.append(disposition)

            if not results:
                continue

            # Keep the highest-edge candidate for this market
            best = max(results, key=lambda r: r["edge_pct"])
            existing = candidates.get(market.id)
            if existing is None or best["edge_pct"] > existing["edge_pct"]:
                candidates[market.id] = best

        # Convert candidates to opportunities
        opportunities: list[Opportunity] = []

        for market_id, candidate in candidates.items():
            market = candidate["market"]
            bias_type = candidate["bias_type"]
            outcome = candidate["outcome"]
            buy_price = candidate["buy_price"]
            token_id = candidate["token_id"]
            edge_pct = candidate["edge_pct"]
            confidence = candidate["confidence"]

            if buy_price <= 0.01:
                continue

            # Build position
            positions = [
                {
                    "action": "BUY",
                    "outcome": outcome,
                    "market_id": market.id,
                    "price": buy_price,
                    "token_id": token_id,
                    "bias_type": bias_type,
                    "edge_pct": round(edge_pct, 2),
                }
            ]

            # Build description per bias type
            if bias_type == "anchoring":
                desc = (
                    f"Anchoring at {candidate['anchor_value']:.2f} "
                    f"({candidate['scans_stuck']} scans), "
                    f"momentum {candidate['direction']}, "
                    f"{candidate['days_left']:.0f}d to resolution. "
                    f"Buy {outcome} @ ${buy_price:.3f}, edge {edge_pct:.1f}%."
                )
            elif bias_type == "overreaction":
                desc = (
                    f"Overreaction: {candidate['move_pct']:.1f}% move, "
                    f"{candidate['retrace_pct']:.1f}% retrace. "
                    f"Fade remaining excess {candidate['direction']}. "
                    f"Buy {outcome} @ ${buy_price:.3f}, edge {edge_pct:.1f}%."
                )
            else:  # disposition
                desc = (
                    f"Disposition dip: {candidate['dip_pct']:.1f}% from peak "
                    f"(gained {candidate['gain_from_neutral_pct']:.1f}% from 0.50). "
                    f"Buy {outcome} @ ${buy_price:.3f}, edge {edge_pct:.1f}%."
                )

            # Risk assessment
            risk_base = 0.55
            if bias_type == "anchoring":
                risk_base = 0.60  # Weakest signal
            elif bias_type == "overreaction":
                risk_base = 0.50  # Stronger signal
            elif bias_type == "disposition":
                risk_base = 0.52

            risk_score = min(0.78, risk_base)
            risk_factors = [
                f"Behavioral bias ({bias_type}): statistical tendency, not guaranteed",
                f"Buy {outcome} @ ${buy_price:.3f}, estimated edge {edge_pct:.1f}%",
            ]

            if bias_type == "anchoring":
                risk_factors.append(f"Anchored at {candidate['anchor_value']:.2f} for {candidate['scans_stuck']} scans")
            elif bias_type == "overreaction":
                risk_factors.append(
                    f"Large move ({candidate['move_pct']:.1f}%) with partial retrace ({candidate['retrace_pct']:.1f}%)"
                )
            elif bias_type == "disposition":
                risk_factors.append(
                    f"Peak {candidate['peak']:.3f}, dip {candidate['dip_pct']:.1f}% "
                    f"(gain from neutral: {candidate['gain_from_neutral_pct']:.1f}%)"
                )

            event_obj = market_to_event.get(market.id)

            opp = self.create_opportunity(
                title=f"Bias Fader ({bias_type}): {market.question[:50]}",
                description=desc,
                total_cost=buy_price,
                expected_payout=1.0,
                markets=[market],
                positions=positions,
                event=event_obj,
                is_guaranteed=False,
                custom_roi_percent=edge_pct,
                custom_risk_score=risk_score,
                confidence=confidence,
            )

            if opp is not None:
                opp.risk_factors = risk_factors
                opportunities.append(opp)

        # Sort by edge descending
        opportunities.sort(
            key=lambda o: o.roi_percent if o.roi_percent else 0.0,
            reverse=True,
        )

        return opportunities[:max_opportunities]

    # ------------------------------------------------------------------
    # Composable evaluate pipeline overrides
    # ------------------------------------------------------------------

    def compute_size(
        self,
        base_size: float,
        max_size: float,
        edge: float,
        confidence: float,
        risk_score: float,
        market_count: int,
    ) -> float:
        """Kelly sizing with bias-type aware scaling.

        Anchoring gets lower confidence (smaller size), overreaction gets
        higher (larger size), disposition is moderate.
        """
        p_estimated = 0.5 + (edge / 200.0)
        p_market = 0.5
        kelly_f = kelly_fraction(p_estimated, p_market, fraction=0.25)
        kelly_sz = base_size * (1.0 + kelly_f * 8.0)

        risk_scale = max(0.55, 1.0 - risk_score * 0.35)
        size = kelly_sz * (0.70 + confidence * 0.6) * risk_scale
        return max(1.0, min(max_size, size))

    def custom_checks(
        self,
        signal: Any,
        context: Any,
        params: dict,
        payload: dict,
    ) -> list[DecisionCheck]:
        """Verify signal source is scanner."""
        source = str(getattr(signal, "source", "") or "").strip().lower()
        return [
            DecisionCheck("source", "Scanner source", source == "scanner", detail=f"got={source}"),
        ]

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Exit when bias is corrected (price moves past expected value), or default TP/SL."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)

        ctx = getattr(position, "strategy_context", None) or {}
        bias_type = ctx.get("bias_type")
        current_price = market_state.get("current_price")
        entry_price = float(getattr(position, "entry_price", 0) or 0)

        if current_price and entry_price > 0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            # Bias-specific exit logic
            if bias_type == "anchoring":
                # Anchoring corrected: price moved >5% from entry
                if pnl_pct > 5.0:
                    return ExitDecision(
                        "close",
                        f"Anchoring corrected: +{pnl_pct:.1f}% from entry",
                        close_price=current_price,
                    )
            elif bias_type == "overreaction":
                # Overreaction fade successful: edge captured
                edge_pct = float(ctx.get("edge_pct", 0) or 0)
                if edge_pct > 0 and pnl_pct >= edge_pct * 0.6:
                    return ExitDecision(
                        "close",
                        f"Overreaction fade captured {pnl_pct:.1f}% (target {edge_pct:.1f}%)",
                        close_price=current_price,
                    )
            elif bias_type == "disposition":
                # Disposition dip recovered
                dip_pct = float(ctx.get("dip_pct", 0) or 0)
                if dip_pct > 0 and pnl_pct >= dip_pct * 0.7:
                    return ExitDecision(
                        "close",
                        f"Disposition dip recovered: +{pnl_pct:.1f}% (dip was {dip_pct:.1f}%)",
                        close_price=current_price,
                    )

        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal: Any, reason: str, context: Any) -> None:
        logger.info(
            "%s: signal blocked - %s (market=%s)",
            self.name,
            reason,
            getattr(signal, "market_id", "?"),
        )

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info(
            "%s: size capped $%.0f -> $%.0f - %s",
            self.name,
            original_size,
            capped_size,
            reason,
        )
