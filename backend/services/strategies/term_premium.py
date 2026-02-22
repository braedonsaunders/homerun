"""
Strategy: Term Premium (Capital Lockup Discount)

Far-dated high-probability contracts trade at a discount because traders demand
compensation for locking up capital. This is the prediction market equivalent
of the bond term premium.

The strategy identifies contracts where:
  1. YES probability is high (>= 80c) indicating a likely outcome
  2. Resolution is far away (30-365 days)
  3. The market price is below the discount-model fair value

As time passes and the resolution date approaches, the discount shrinks and the
price converges toward the true probability, generating carry-like returns.

Fair value model:
  fair_price = true_prob / (1 + discount_rate * years_to_resolution)

The true probability is estimated using FLB-calibrated priors for the price
bucket. When the market price is below fair_price, the excess discount
represents our edge.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from models import Market, Event, Opportunity
from models.opportunity import MispricingType
from .base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig, utcnow, make_aware
from services.quality_filter import QualityFilterOverrides
from utils.kelly import kelly_fraction, polymarket_taker_fee
from utils.logger import get_logger

logger = get_logger(__name__)


# Calibration priors reused from FLB research (Snowberg & Wolfers).
# For the term premium strategy we focus on the favorite range (>= 0.80).
_CALIBRATION_PRIORS: dict[tuple[float, float], float] = {
    (0.80, 0.85): 0.84,  # 80-85c favorites win ~84%
    (0.85, 0.90): 0.88,  # 85-90c favorites win ~88%
    (0.90, 0.95): 0.93,  # 90-95c favorites win ~93%
    (0.95, 1.00): 0.97,  # 95-100c favorites win ~97%
}


def _get_true_prob(yes_price: float) -> Optional[float]:
    """Estimate true probability from calibration priors for a price bucket."""
    for (lo, hi), rate in _CALIBRATION_PRIORS.items():
        if lo <= yes_price < hi:
            return rate
    return None


class TermPremiumStrategy(BaseStrategy):
    """Capture capital lockup discount on far-dated favorites."""

    strategy_type = "term_premium"
    name = "Term Premium"
    description = "Capture capital lockup discount on far-dated favorites"
    mispricing_type = "within_market"
    requires_resolution_date = True
    realtime_processing_mode = "incremental"
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=1.0,
        max_resolution_months=12.0,
        min_annualized_roi=3.0,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.45,
        confidence_weight=20.0,
        risk_penalty=5.0,
    )
    sizing_config = SizingConfig(
        base_divisor=120.0,
        confidence_offset=0.65,
        risk_scale_factor=0.25,
        risk_floor=0.65,
    )

    default_config = {
        "min_edge_percent": 1.0,
        "min_confidence": 0.45,
        "max_risk_score": 0.65,
        "min_probability": 0.80,
        "min_days_to_resolution": 30,
        "max_days_to_resolution": 365,
        "discount_rate": 0.08,  # annual discount rate (8%)
        "min_liquidity": 1000.0,
        "base_size_usd": 15.0,
        "max_size_usd": 150.0,
        "convergence_exit_threshold": 0.02,  # exit when within 2% of true_prob
        "drawdown_exit_threshold": 0.10,  # exit on 10% price drop from entry
        "take_profit_pct": 10.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def _get_live_yes_price(self, market: Market, prices: dict[str, dict]) -> float:
        """Get the best available YES price from CLOB or HTTP fallback."""
        yes_price = market.yes_price
        tokens = list(getattr(market, "clob_token_ids", []) or [])
        if tokens:
            token = tokens[0]
            if token in prices:
                yes_price = prices[token].get("mid", yes_price)
        return yes_price

    # ------------------------------------------------------------------
    # Discount model
    # ------------------------------------------------------------------

    def _compute_fair_price(self, true_prob: float, days_to_resolution: float, discount_rate: float) -> float:
        """Compute the fair discounted price for a far-dated contract.

        fair_price = true_prob / (1 + discount_rate * years)

        A contract worth true_prob at resolution currently trades at a discount
        because capital is locked up until resolution.
        """
        years = days_to_resolution / 365.0
        fair = true_prob / (1.0 + discount_rate * years)
        return max(0.01, min(0.99, fair))

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        min_prob = float(cfg.get("min_probability", 0.80))
        min_days = float(cfg.get("min_days_to_resolution", 30))
        max_days = float(cfg.get("max_days_to_resolution", 365))
        min_edge_pct = float(cfg.get("min_edge_percent", 1.0))
        min_liquidity = float(cfg.get("min_liquidity", 1000.0))
        discount_rate = float(cfg.get("discount_rate", 0.08))

        # Build event lookup
        event_by_market: dict[str, Event] = {}
        for ev in events:
            for em in ev.markets:
                event_by_market[em.id] = ev

        now = utcnow()
        opportunities: list[Opportunity] = []
        entry_prices = self.state.setdefault("entry_prices", {})

        for market in markets:
            if market.closed or not market.active:
                continue
            if len(list(getattr(market, "outcome_prices", []) or [])) < 2:
                continue
            if float(getattr(market, "liquidity", 0.0) or 0.0) < min_liquidity:
                continue

            # Require a resolution date
            end_date = make_aware(getattr(market, "end_date", None))
            if end_date is None:
                continue

            days_to_resolution = (end_date - now).total_seconds() / 86400.0
            if days_to_resolution < min_days or days_to_resolution > max_days:
                continue

            yes_price = self._get_live_yes_price(market, prices)

            # Must be a high-probability outcome
            if yes_price < min_prob:
                continue

            # Get calibrated true probability
            true_prob = _get_true_prob(yes_price)
            if true_prob is None:
                continue

            # Compute discount-adjusted fair price
            fair_price = self._compute_fair_price(true_prob, days_to_resolution, discount_rate)

            # The edge: if market price < fair price, the discount is excessive
            if yes_price >= fair_price:
                continue  # No excess discount

            edge = fair_price - yes_price
            edge_pct = (edge / yes_price) * 100.0

            if edge_pct < min_edge_pct:
                continue

            # Fee-adjusted edge
            fee = polymarket_taker_fee(yes_price)
            net_edge = edge - fee
            net_edge_pct = (net_edge / yes_price) * 100.0 if yes_price > 0 else 0.0

            if net_edge_pct < min_edge_pct * 0.5:
                continue  # Fees consume too much edge

            # Confidence: higher for closer resolution, higher liquidity, larger edge
            time_factor = 1.0 - (days_to_resolution / max_days)  # closer = more confident
            liq_factor = min(1.0, float(getattr(market, "liquidity", 0.0) or 0.0) / 10000.0)
            edge_factor = min(1.0, edge_pct / 5.0)
            confidence = 0.35 * time_factor + 0.30 * liq_factor + 0.35 * edge_factor
            confidence = max(0.30, min(0.90, confidence))

            tokens = list(getattr(market, "clob_token_ids", []) or [])
            token_id = tokens[0] if tokens else None

            question_short = market.question[:60]
            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "price": yes_price,
                    "token_id": token_id,
                    "market_id": market.id,
                    "rationale": (
                        f"Term premium: YES at {yes_price:.3f}, "
                        f"true prob {true_prob:.3f}, fair price {fair_price:.3f}, "
                        f"edge {edge_pct:.2f}% (net {net_edge_pct:.2f}%), "
                        f"{days_to_resolution:.0f}d to resolution"
                    ),
                    "_term_premium": {
                        "true_prob": true_prob,
                        "fair_price": fair_price,
                        "discount_rate": discount_rate,
                        "days_to_resolution": days_to_resolution,
                    },
                },
            ]

            years = days_to_resolution / 365.0
            opp = self.create_opportunity(
                title=f"Term Premium: YES {yes_price:.1%} in {days_to_resolution:.0f}d ({question_short}...)",
                description=(
                    f"Capital lockup discount: YES at {yes_price:.3f} is below fair value {fair_price:.3f}. "
                    f"True probability ~{true_prob:.3f} discounted at {discount_rate:.0%}/yr over "
                    f"{years:.1f} years. Edge {edge_pct:.2f}% (fee-adjusted {net_edge_pct:.2f}%). "
                    f"Price should converge as resolution approaches."
                ),
                total_cost=yes_price,
                expected_payout=fair_price,
                markets=[market],
                positions=positions,
                event=event_by_market.get(market.id),
                is_guaranteed=False,
                confidence=confidence,
                skip_fee_model=True,
                custom_roi_percent=net_edge_pct,
            )

            if opp is None:
                continue

            # Risk score: longer duration = higher risk
            risk_score, risk_factors = self.calculate_risk_score([market], end_date)
            # Add duration-specific risk
            if days_to_resolution > 180:
                risk_score = min(0.85, risk_score + 0.10)
                risk_factors.append(f"Long duration: {days_to_resolution:.0f} days of capital lockup")
            elif days_to_resolution > 90:
                risk_score = min(0.80, risk_score + 0.05)
                risk_factors.append(f"Medium duration: {days_to_resolution:.0f} days of capital lockup")

            opp.risk_score = risk_score
            opp.risk_factors = risk_factors + [
                "DIRECTIONAL BET - based on term premium discount model",
                f"Discount rate assumption: {discount_rate:.0%} annualized",
                f"Fair value: {fair_price:.3f} (true prob {true_prob:.3f} discounted over {years:.1f}yr)",
                f"Fee-adjusted edge: {net_edge_pct:.2f}%",
            ]
            opp.mispricing_type = MispricingType.WITHIN_MARKET

            # Track entry for convergence monitoring
            entry_prices[market.id] = {
                "entry_price": yes_price,
                "entry_time": time.time(),
                "true_prob": true_prob,
                "fair_price": fair_price,
            }

            opportunities.append(opp)

        if opportunities:
            logger.info(
                "Term Premium: found %d opportunity(ies), avg edge %.2f%%",
                len(opportunities),
                sum(o.roi_percent for o in opportunities) / len(opportunities),
            )

        return opportunities

    # ------------------------------------------------------------------
    # Composable evaluate pipeline overrides
    # ------------------------------------------------------------------

    def custom_checks(self, signal: Any, context: Any, params: dict, payload: dict) -> list[DecisionCheck]:
        """Term Premium: verify source and minimum liquidity."""
        source = str(getattr(signal, "source", "") or "").strip().lower()
        min_liq = float(params.get("min_liquidity", 1000.0) or 1000.0)
        liquidity = float(payload.get("min_liquidity", 0) or 0)
        if liquidity <= 0:
            mkts = payload.get("markets") or []
            if mkts:
                liquidity = min((float(m.get("liquidity", 0) or 0) for m in mkts), default=0)

        return [
            DecisionCheck(
                "source",
                "Scanner source",
                source == "scanner",
                detail=f"got={source}",
            ),
            DecisionCheck(
                "liquidity",
                "Minimum liquidity",
                liquidity >= min_liq,
                score=liquidity,
                detail=f"min=${min_liq:.0f}",
            ),
        ]

    def compute_size(
        self,
        base_size: float,
        max_size: float,
        edge: float,
        confidence: float,
        risk_score: float,
        market_count: int,
    ) -> float:
        """Conservative Kelly sizing for longer-hold term premium trades."""
        # Use conservative 0.15 fraction due to longer hold periods
        p_estimated = 0.5 + (edge / 200.0)
        p_market = 0.5
        kelly_f = kelly_fraction(p_estimated, p_market, fraction=0.15)
        kelly_sz = base_size * (1.0 + kelly_f * 6.0)
        risk_scale = max(0.55, 1.0 - risk_score * 0.35)
        size = kelly_sz * (0.65 + confidence * 0.50) * risk_scale
        return max(1.0, min(max_size, size))

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Term Premium: exit on convergence, resolution, or drawdown."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)

        current_price = market_state.get("current_price")
        if current_price is None:
            return ExitDecision("hold", "No current price available")

        entry_price = float(getattr(position, "entry_price", 0) or 0)
        config = getattr(position, "config", None) or {}
        config = dict(config)
        configured_tp = (getattr(self, "config", None) or {}).get("take_profit_pct", 10.0)
        try:
            default_tp = float(configured_tp)
        except (TypeError, ValueError):
            default_tp = 10.0
        config.setdefault("take_profit_pct", default_tp)
        position.config = config
        convergence_threshold = float(config.get("convergence_exit_threshold", 0.02) or 0.02)
        drawdown_threshold = float(config.get("drawdown_exit_threshold", 0.10) or 0.10)

        # Check convergence: price has risen close to true_prob
        strategy_ctx = getattr(position, "strategy_context", None) or {}
        true_prob = float(strategy_ctx.get("true_prob", 0) or 0)
        if true_prob > 0 and current_price >= (true_prob - convergence_threshold):
            return ExitDecision(
                "close",
                f"Price converged to true prob ({current_price:.3f} >= {true_prob:.3f} - {convergence_threshold})",
                close_price=current_price,
            )

        # Check drawdown: price dropped significantly from entry
        if entry_price > 0:
            drawdown = (entry_price - current_price) / entry_price
            if drawdown >= drawdown_threshold:
                return ExitDecision(
                    "close",
                    f"Drawdown exit: price dropped {drawdown:.1%} from entry ({entry_price:.3f} -> {current_price:.3f})",
                    close_price=current_price,
                )

        # Fallback to standard TP/SL/trailing/max-hold
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
