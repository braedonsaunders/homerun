from __future__ import annotations

import logging
from typing import Any

from models import Market, Event, Opportunity
from .base import BaseStrategy, DecisionCheck, ExitDecision, ScoringWeights, SizingConfig
from services.quality_filter import QualityFilterOverrides
from utils.converters import to_float

logger = logging.getLogger(__name__)


class BasicArbStrategy(BaseStrategy):
    """
    Strategy 1: Basic Arbitrage

    Buy YES + NO on the same binary market when total cost < $1.00
    Guaranteed profit since one of them must win.

    Example:
    - YES price: $0.48
    - NO price: $0.48
    - Total cost: $0.96
    - Payout: $1.00
    - Profit: $0.04 (before fees)
    """

    strategy_type = "basic"
    name = "Basic Arbitrage"
    description = "Buy YES and NO on same market when total < $1"
    mispricing_type = "within_market"
    subscriptions = ["market_data_refresh"]
    realtime_processing_mode = "incremental"

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.3,
        min_annualized_roi=0.0,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=30.0,
        risk_penalty=8.0,
        liquidity_weight=8.0,
        liquidity_divisor=5000.0,
        market_count_bonus=1.5,
        structural_bonus=2.0,
    )
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.75,
        risk_scale_factor=0.35,
        risk_floor=0.55,
        market_scale_factor=0.08,
        market_scale_cap=4,
    )
    default_config = {
        "min_edge_percent": 4.0,
        "min_confidence": 0.45,
        "max_risk_score": 0.78,
        "min_liquidity": 25.0,
        "min_markets": 1,
        "base_size_usd": 18.0,
        "max_size_usd": 150.0,
    }

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:
        opportunities = []

        for market in markets:
            # Skip if not a binary market
            if len(market.outcome_prices) != 2:
                continue

            # Skip inactive or closed markets
            if market.closed or not market.active:
                continue

            # Get prices (use live prices if available)
            yes_price = market.yes_price
            no_price = market.no_price

            # Update with live prices if we have them
            if market.clob_token_ids:
                yes_token = market.clob_token_ids[0] if len(market.clob_token_ids) > 0 else None
                no_token = market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None

                if yes_token and yes_token in prices:
                    yes_price = prices[yes_token].get("mid", yes_price)
                if no_token and no_token in prices:
                    no_price = prices[no_token].get("mid", no_price)

            # Calculate total cost
            total_cost = yes_price + no_price

            # Check for arbitrage (need room for fees)
            # Total must be less than (1 - fee) to profit
            if total_cost >= 1.0:
                continue

            # Create opportunity
            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "price": yes_price,
                    "token_id": market.clob_token_ids[0] if market.clob_token_ids else None,
                },
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "price": no_price,
                    "token_id": market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None,
                },
            ]

            opp = self.create_opportunity(
                title=f"Basic Arb: {market.question[:50]}...",
                description=f"Buy YES (${yes_price:.3f}) + NO (${no_price:.3f}) = ${total_cost:.3f} for guaranteed $1 payout",
                total_cost=total_cost,
                markets=[market],
                positions=positions,
            )

            if opp:
                opportunities.append(opp)

        return opportunities

    SOURCES = {"scanner"}

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in self.SOURCES
        min_liquidity = max(0.0, to_float(params.get("min_liquidity", 25.0), 25.0))
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        min_markets = max(1, int(to_float(params.get("min_markets", 1), 1)))
        market_count = len(payload.get("markets") or [])
        position_count = len(payload.get("positions_to_take") or [])

        # Stash signal liquidity so the base compute_score can pick it up via
        # payload["liquidity"] — overrides any opportunity-level liquidity value
        # with the live signal's liquidity for more accurate scoring.
        payload["liquidity"] = liquidity

        return [
            DecisionCheck("source", "Scanner source", source_ok, detail="Requires source=scanner."),
            DecisionCheck(
                "liquidity",
                "Liquidity floor",
                liquidity >= min_liquidity,
                score=liquidity,
                detail=f"min={min_liquidity:.0f}",
            ),
            DecisionCheck(
                "markets",
                "Market leg coverage",
                market_count >= min_markets and position_count > 0,
                score=float(market_count),
                detail=f"markets={market_count}, positions={position_count}",
            ),
        ]

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Guaranteed-spread: hold to resolution for maximum value."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        if not config.get("resolve_only", True):
            return self.default_exit_check(position, market_state)
        return ExitDecision("hold", "Guaranteed spread — holding to resolution")

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
