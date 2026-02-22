"""
Strategy: Implied Probability Surface Arbitrage

When multiple markets on the same event have different thresholds
(e.g., "Will BTC exceed $50K / $60K / $70K / $80K"), their implied
probabilities must be monotonically decreasing for "above"-style
questions: P(X > low) >= P(X > high).

Deviations from monotonicity are mispricings. This strategy detects
them by:
1. Grouping markets into "contract families" by event and numeric
   threshold extraction.
2. Fitting a monotone non-increasing surface via the Pool Adjacent
   Violators Algorithm (isotonic regression).
3. Trading the deviations: buy underpriced contracts, sell/fade
   overpriced contracts within the same family.

NOT risk-free. The fitted surface is an estimate. Market-makers may
have information justifying the deviation. Use with position limits.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Optional

from models import Market, Event, Opportunity
from .base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig
from services.quality_filter import QualityFilterOverrides
from utils.kelly import kelly_fraction
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Threshold extraction regexes
# ---------------------------------------------------------------------------
_THRESHOLD_PATTERNS = [
    re.compile(r"(?:above|exceed|over|reach|at least|more than)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:below|under|less than|at most)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:between)\s+\$?([\d,]+\.?\d*)\s+and\s+\$?([\d,]+\.?\d*)", re.IGNORECASE),
    re.compile(r"(\d+\.?\d*)\s*(?:\u00b0[CF]|degrees|percent|%|mph|mm)", re.IGNORECASE),
]

# Direction patterns: "above"-style means monotone decreasing, "below"-style
# means monotone increasing.
_ABOVE_PATTERN = re.compile(r"\b(?:above|exceed|over|reach|at least|more than)\b", re.IGNORECASE)
_BELOW_PATTERN = re.compile(r"\b(?:below|under|less than|at most)\b", re.IGNORECASE)


def _parse_threshold(question: str) -> Optional[float]:
    """Extract a single numeric threshold from a market question.

    Returns the parsed float or None if no threshold is found.
    """
    for pattern in _THRESHOLD_PATTERNS:
        match = pattern.search(question)
        if match:
            raw = match.group(1).replace(",", "")
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def _detect_direction(question: str) -> str:
    """Detect whether the question is 'above'-style or 'below'-style.

    Returns 'above' (default), 'below', or 'above' as fallback.
    """
    if _BELOW_PATTERN.search(question):
        return "below"
    return "above"


class ProbSurfaceArbStrategy(BaseStrategy):
    """Arbitrage deviations from monotonic probability surfaces across
    contract families sharing a common event and numeric threshold axis."""

    strategy_type = "prob_surface_arb"
    name = "Probability Surface Arb"
    description = "Arbitrage deviations from monotonic probability surfaces across contract families"
    mispricing_type = "cross_market"
    realtime_processing_mode = "full_snapshot"
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.0,
        max_legs=8,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=28.0,
        risk_penalty=7.0,
        market_count_bonus=1.5,
    )
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.70,
        risk_scale_factor=0.35,
        risk_floor=0.55,
        market_scale_factor=0.08,
        market_scale_cap=6,
    )

    default_config = {
        "min_edge_percent": 2.0,
        "min_confidence": 0.50,
        "max_risk_score": 0.75,
        "min_family_size": 3,
        "min_deviation_cents": 0.03,
        "min_liquidity": 500.0,
        "base_size_usd": 15.0,
        "max_size_usd": 150.0,
        "max_opportunities": 20,
        "take_profit_pct": 12.0,
    }

    pipeline_defaults = {
        "min_edge_percent": 2.0,
        "min_confidence": 0.50,
        "max_risk_score": 0.75,
        "base_size_usd": 15.0,
        "max_size_usd": 150.0,
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
    # Isotonic regression (Pool Adjacent Violators Algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def _isotonic_fit(prices: list[float], decreasing: bool = True) -> list[float]:
        """Fit a monotone function to *prices* via PAVA.

        Args:
            prices: Observed prices sorted by threshold (ascending).
            decreasing: If True, enforce monotone non-increasing (above-style).
                        If False, enforce monotone non-decreasing (below-style).

        Returns:
            List of fitted (adjusted) prices preserving the specified monotonicity.
        """
        n = len(prices)
        if n <= 1:
            return list(prices)

        fitted = list(prices)

        if decreasing:
            # Enforce non-increasing: fitted[i] >= fitted[i+1]
            # When a violation occurs (fitted[i] < fitted[i+1]), pool the
            # violating block and replace with their average.
            changed = True
            while changed:
                changed = False
                i = 0
                while i < n - 1:
                    if fitted[i] < fitted[i + 1]:
                        # Find the extent of the violation block
                        j = i + 1
                        while j < n - 1 and fitted[j] < fitted[j + 1]:
                            j += 1
                        # Pool [i..j] and average
                        avg = sum(fitted[i : j + 1]) / (j - i + 1)
                        for k in range(i, j + 1):
                            fitted[k] = avg
                        changed = True
                        i = j + 1
                    else:
                        i += 1
        else:
            # Enforce non-decreasing: fitted[i] <= fitted[i+1]
            changed = True
            while changed:
                changed = False
                i = 0
                while i < n - 1:
                    if fitted[i] > fitted[i + 1]:
                        j = i + 1
                        while j < n - 1 and fitted[j] > fitted[j + 1]:
                            j += 1
                        avg = sum(fitted[i : j + 1]) / (j - i + 1)
                        for k in range(i, j + 1):
                            fitted[k] = avg
                        changed = True
                        i = j + 1
                    else:
                        i += 1

        return fitted

    # ------------------------------------------------------------------
    # Family building
    # ------------------------------------------------------------------

    def _build_families(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
        cfg: dict,
    ) -> dict[str, list[tuple[float, Market, float]]]:
        """Group markets into contract families by event_id + threshold.

        Returns:
            Dict mapping event_id -> sorted list of (threshold, Market, yes_price).
            Only families with >= min_family_size members are included.
        """
        min_family_size = int(cfg.get("min_family_size", 3))
        min_liquidity = float(cfg.get("min_liquidity", 500.0))

        # Map market_id -> event for fast lookup
        market_to_event: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                market_to_event[m.id] = event

        # Group by event_id, extracting thresholds
        raw_families: dict[str, list[tuple[float, Market, float]]] = defaultdict(list)

        for market in markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) < 2:
                continue
            if market.liquidity < min_liquidity:
                continue

            threshold = _parse_threshold(market.question or "")
            if threshold is None:
                continue

            event = market_to_event.get(market.id)
            if event is None:
                continue

            yes_price = self._live_yes_price(market, prices)
            if yes_price <= 0.01 or yes_price >= 0.99:
                continue

            raw_families[event.id].append((threshold, market, yes_price))

        # Filter to families meeting min size and sort by threshold
        families: dict[str, list[tuple[float, Market, float]]] = {}
        for event_id, members in raw_families.items():
            if len(members) < min_family_size:
                continue
            # Sort ascending by threshold
            members.sort(key=lambda x: x[0])
            families[event_id] = members

        return families

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        """Detect probability surface arbitrage opportunities."""
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        min_deviation = float(cfg.get("min_deviation_cents", 0.03))
        max_opportunities = int(cfg.get("max_opportunities", 20))

        # Use self.state to cache families; invalidate every 5 minutes
        now = time.time()
        family_cache_time = self.state.get("family_cache_time", 0.0)
        if now - family_cache_time > 300:
            self.state["cached_families"] = None
            self.state["family_cache_time"] = now

        families = self._build_families(events, markets, prices, cfg)
        if not families:
            return []

        # Map market_id -> Event for opportunity creation
        market_to_event: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                market_to_event[m.id] = event

        opportunities: list[Opportunity] = []

        for event_id, members in families.items():
            thresholds = [t for t, _, _ in members]
            actual_prices = [p for _, _, p in members]
            family_markets = [m for _, m, _ in members]

            # Detect direction from the first market's question
            direction = _detect_direction(family_markets[0].question or "")
            decreasing = direction == "above"

            # Fit isotonic surface
            fitted = self._isotonic_fit(actual_prices, decreasing=decreasing)

            # Find deviations
            deviations: list[tuple[int, float]] = []  # (index, deviation)
            for i in range(len(actual_prices)):
                dev = fitted[i] - actual_prices[i]
                if abs(dev) >= min_deviation:
                    deviations.append((i, dev))

            if not deviations:
                continue

            # Build multi-leg opportunity from deviations
            total_deviation = sum(abs(d) for _, d in deviations)
            positions = []
            opp_markets = []
            total_cost = 0.0

            for idx, dev in deviations:
                market = family_markets[idx]
                yes_price = actual_prices[idx]
                no_price = self._live_no_price(market, prices)
                threshold_val = thresholds[idx]

                if dev > 0:
                    # Fitted > actual: market is underpriced, BUY YES
                    token_id = (
                        market.clob_token_ids[0] if market.clob_token_ids and len(market.clob_token_ids) > 0 else None
                    )
                    positions.append(
                        {
                            "action": "BUY",
                            "outcome": "YES",
                            "market_id": market.id,
                            "market_question": market.question,
                            "price": yes_price,
                            "token_id": token_id,
                            "threshold": threshold_val,
                            "deviation": round(dev, 4),
                            "fitted_price": round(fitted[idx], 4),
                        }
                    )
                    total_cost += yes_price
                    opp_markets.append(market)
                else:
                    # Fitted < actual: market is overpriced, BUY NO (fade)
                    token_id = (
                        market.clob_token_ids[1] if market.clob_token_ids and len(market.clob_token_ids) > 1 else None
                    )
                    positions.append(
                        {
                            "action": "BUY",
                            "outcome": "NO",
                            "market_id": market.id,
                            "market_question": market.question,
                            "price": no_price,
                            "token_id": token_id,
                            "threshold": threshold_val,
                            "deviation": round(dev, 4),
                            "fitted_price": round(fitted[idx], 4),
                        }
                    )
                    total_cost += no_price
                    opp_markets.append(market)

            if total_cost <= 0 or not positions:
                continue

            # Edge = total deviation / total cost (as percent)
            edge_pct = (total_deviation / total_cost) * 100.0

            # Confidence scales with family size and deviation consistency
            family_size = len(members)
            deviation_count = len(deviations)
            confidence = min(0.90, 0.50 + (family_size - 3) * 0.05 + deviation_count * 0.03)

            # Risk assessment
            risk_score, risk_factors = self.calculate_risk_score(
                opp_markets,
                opp_markets[0].end_date if opp_markets else None,
            )
            risk_factors.insert(0, "Cross-market surface arb: not guaranteed profit")
            risk_factors.append(f"Family size: {family_size} contracts, {deviation_count} deviations")
            risk_factors.append(f"Direction: {direction}, total deviation: {total_deviation:.3f}")

            # Build description
            threshold_range = f"{thresholds[0]:.0f}-{thresholds[-1]:.0f}"
            event_obj = market_to_event.get(opp_markets[0].id)
            event_title = event_obj.title[:40] if event_obj else "Unknown"

            opp = self.create_opportunity(
                title=f"Surface Arb: {event_title} ({threshold_range})",
                description=(
                    f"Monotonicity violation in {family_size}-contract family "
                    f"({direction} {threshold_range}). "
                    f"{deviation_count} legs, total deviation {total_deviation:.3f}, "
                    f"edge {edge_pct:.1f}%."
                ),
                total_cost=total_cost,
                markets=opp_markets,
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
        """Kelly-informed sizing with edge = weighted deviation."""
        p_estimated = 0.5 + (edge / 200.0)
        p_market = 0.5
        kelly_f = kelly_fraction(p_estimated, p_market, fraction=0.25)
        kelly_sz = base_size * (1.0 + kelly_f * 8.0)

        # Scale up for multi-leg (more diversification within family)
        market_scale = 1.0 + min(6, max(0, market_count - 1)) * 0.08
        risk_scale = max(0.55, 1.0 - risk_score * 0.35)

        size = kelly_sz * (0.70 + confidence * 0.6) * market_scale * risk_scale
        return max(1.0, min(max_size, size))

    def custom_checks(
        self,
        signal: Any,
        context: Any,
        params: dict,
        payload: dict,
    ) -> list[DecisionCheck]:
        """Verify signal source and family size."""
        source = str(getattr(signal, "source", "") or "").strip().lower()
        # Extract family size from payload positions
        positions = payload.get("positions_to_take") or payload.get("positions") or []
        num_legs = len(positions) if isinstance(positions, list) else 0
        if num_legs == 0:
            num_legs = len(payload.get("markets") or [])

        return [
            DecisionCheck("source", "Scanner source", source == "scanner", detail=f"got={source}"),
            DecisionCheck(
                "family_size",
                "Minimum family legs",
                num_legs >= 2,
                score=float(num_legs),
                detail=f"legs={num_legs}, min=2",
            ),
        ]

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Hold to resolution or until surface normalizes. Default TP/SL applies."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)

        # Check if the deviation has been corrected
        ctx = getattr(position, "strategy_context", None) or {}
        original_deviation = ctx.get("deviation")
        current_price = market_state.get("current_price")
        fitted_price = ctx.get("fitted_price")

        if original_deviation and current_price and fitted_price:
            try:
                current_dev = abs(float(fitted_price) - float(current_price))
                original_dev = abs(float(original_deviation))
                # Surface has normalized: deviation reduced by >70%
                if original_dev > 0 and current_dev / original_dev < 0.30:
                    return ExitDecision(
                        "close",
                        f"Surface normalized: deviation reduced from {original_dev:.3f} to {current_dev:.3f}",
                        close_price=current_price,
                    )
            except (TypeError, ValueError):
                pass

        config = getattr(position, "config", None) or {}
        config = dict(config)
        configured_tp = (getattr(self, "config", None) or {}).get("take_profit_pct", 12.0)
        try:
            default_tp = float(configured_tp)
        except (TypeError, ValueError):
            default_tp = 12.0
        config.setdefault("take_profit_pct", default_tp)
        position.config = config
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
