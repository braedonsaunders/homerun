from typing import Any

from models import Market, Event, ArbitrageOpportunity
from .base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.strategies._evaluate_helpers import to_float, to_confidence, signal_payload


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

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:
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

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        params = context.get("params") or {}
        payload = signal_payload(signal)

        min_edge = to_float(params.get("min_edge_percent", 4.0), 4.0)
        min_conf = to_confidence(params.get("min_confidence", 0.45), 0.45)
        max_risk = to_confidence(params.get("max_risk_score", 0.78), 0.78)
        min_liquidity = max(0.0, to_float(params.get("min_liquidity", 25.0), 25.0))
        min_markets = max(1, int(to_float(params.get("min_markets", 1), 1)))
        base_size = max(1.0, to_float(params.get("base_size_usd", 18.0), 18.0))
        max_size = max(base_size, to_float(params.get("max_size_usd", 150.0), 150.0))

        source = str(getattr(signal, "source", "") or "").strip().lower()
        source_ok = source in self.SOURCES

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])
        position_count = len(payload.get("positions_to_take") or [])
        is_guaranteed = bool(payload.get("is_guaranteed", False))

        checks = [
            DecisionCheck("source", "Scanner source", source_ok, detail="Requires source=scanner."),
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "risk_score",
                "Risk score ceiling",
                risk_score <= max_risk,
                score=risk_score,
                detail=f"max={max_risk:.2f}",
            ),
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

        score = (
            (edge * 0.55)
            + (confidence * 30.0)
            + (min(1.0, liquidity / 5000.0) * 8.0)
            + (min(4, market_count) * 1.5)
            + (2.0 if is_guaranteed else 0.0)
            - (risk_score * 8.0)
        )

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="General opportunity filters not met",
                score=score,
                checks=checks,
                payload={
                    "source": source,
                    "edge": edge,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "liquidity": liquidity,
                    "market_count": market_count,
                    "position_count": position_count,
                    "is_guaranteed": is_guaranteed,
                },
            )

        market_scale = 1.0 + (min(4, max(0, market_count - 1)) * 0.08)
        risk_scale = max(0.55, 1.0 - (risk_score * 0.35))
        size = base_size * (1.0 + (edge / 100.0)) * (0.75 + confidence) * market_scale * risk_scale
        size = max(1.0, min(max_size, size))

        return StrategyDecision(
            decision="selected",
            reason="General opportunity signal selected",
            score=score,
            size_usd=size,
            checks=checks,
            payload={
                "source": source,
                "edge": edge,
                "confidence": confidence,
                "risk_score": risk_score,
                "liquidity": liquidity,
                "market_count": market_count,
                "position_count": position_count,
                "is_guaranteed": is_guaranteed,
                "size_usd": size,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Guaranteed-spread: hold to resolution for maximum value."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        config = getattr(position, "config", None) or {}
        if not config.get("resolve_only", True):
            return self.default_exit_check(position, market_state)
        return ExitDecision("hold", "Guaranteed spread — holding to resolution")
