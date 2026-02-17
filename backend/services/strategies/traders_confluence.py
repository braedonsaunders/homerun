"""
Traders Confluence Strategy

Evaluates tracked trader confluence signals to detect actionable opportunities
where multiple smart wallets are converging on the same market position.

Unlike scanner strategies that detect structural mispricings, this strategy
detects BEHAVIORAL mispricings based on smart money flow patterns.

Pipeline:
  1. Wallet tracker monitors known profitable wallets
  2. Confluence engine detects multi-wallet convergence
  3. Signal bus emits trader flow signals
  4. This strategy filters signals and converts to ArbitrageOpportunity
"""

from __future__ import annotations

import logging
from typing import Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class TradersConfluenceStrategy(BaseStrategy):
    """
    Traders Confluence Strategy

    Detects opportunities where multiple tracked profitable wallets
    are converging on the same market position, filtered by
    configurable quality gates.
    """

    strategy_type = "traders_confluence"
    name = "Traders Confluence"
    description = "Detect smart money convergence via tracked wallet confluence analysis"

    # Default config thresholds (overridden by DB config column)
    DEFAULT_CONFIG = {
        "min_edge_percent": 3.0,
        "min_confidence": 0.45,
        "min_confluence_strength": 0.50,
        "min_tier": "high",
        "min_wallet_count": 2,
        "max_entry_price": 0.85,
        "risk_base_score": 0.40,
    }

    TIER_ORDER = {"low": 0, "medium": 1, "high": 2, "extreme": 3}

    def __init__(self):
        super().__init__()
        self._config = dict(self.DEFAULT_CONFIG)

    def configure(self, config: dict) -> None:
        """Apply user config overrides from the DB config column."""
        if config:
            for key in self.DEFAULT_CONFIG:
                if key in config:
                    self._config[key] = config[key]

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Sync detect - returns empty.

        TradersConfluenceStrategy is fed by the tracked trader pipeline,
        not the main scanner loop. Confluence signals are converted to
        opportunities by detect_from_signals().
        """
        return []

    def detect_from_signals(
        self,
        signals: list[dict],
        markets: list[Market],
        events: list[Event],
    ) -> list[ArbitrageOpportunity]:
        """Convert trader confluence signals into ArbitrageOpportunity objects.

        Each signal dict should contain:
          - market_id: str
          - direction: "buy_yes" | "buy_no"
          - edge_percent: float
          - confidence: float
          - confluence_strength: float (0-1)
          - tier: "low" | "medium" | "high" | "extreme"
          - wallet_count: int
          - total_volume_usd: float
          - wallets: list[str] (wallet addresses/labels)
          - entry_price: float
          - target_price: float
        """
        if not signals:
            return []

        cfg = self._config
        opportunities: list[ArbitrageOpportunity] = []
        market_map = {m.id: m for m in markets}
        event_map: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                event_map[m.id] = event

        for signal in signals:
            try:
                opp = self._evaluate_signal(signal, market_map, event_map, cfg)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug("Traders Confluence: skipped signal: %s", e)

        if opportunities:
            logger.info(
                "Traders Confluence: %d opportunities from %d signals",
                len(opportunities),
                len(signals),
            )
        return opportunities

    def _evaluate_signal(
        self,
        signal: dict,
        market_map: dict[str, Market],
        event_map: dict[str, Event],
        cfg: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Evaluate a single confluence signal against config thresholds."""
        edge = float(signal.get("edge_percent", 0))
        confidence = float(signal.get("confidence", 0))
        strength = float(signal.get("confluence_strength", 0))
        tier = str(signal.get("tier", "low")).lower()
        wallet_count = int(signal.get("wallet_count", 0))
        entry_price = float(signal.get("entry_price", 1))

        # Quality gates
        if edge < cfg["min_edge_percent"]:
            return None
        if confidence < cfg["min_confidence"]:
            return None
        if strength < cfg["min_confluence_strength"]:
            return None
        min_tier_val = self.TIER_ORDER.get(cfg["min_tier"], 2)
        signal_tier_val = self.TIER_ORDER.get(tier, 0)
        if signal_tier_val < min_tier_val:
            return None
        if wallet_count < cfg.get("min_wallet_count", 2):
            return None
        if entry_price > cfg["max_entry_price"]:
            return None

        market_id = signal.get("market_id")
        market = market_map.get(market_id) if market_id else None
        if not market:
            return None

        event = event_map.get(market_id)
        direction = signal.get("direction", "buy_yes")
        target_price = float(signal.get("target_price", entry_price))
        total_volume = float(signal.get("total_volume_usd", 0))
        wallets = signal.get("wallets", [])

        # Position details
        side = "YES" if direction == "buy_yes" else "NO"
        token_id = None
        if market.clob_token_ids:
            idx = 0 if direction == "buy_yes" else (1 if len(market.clob_token_ids) > 1 else 0)
            token_id = market.clob_token_ids[idx]

        expected_payout = target_price
        total_cost = entry_price
        gross_profit = expected_payout - total_cost
        fee_amount = expected_payout * self.fee
        net_profit = gross_profit - fee_amount
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        if roi < cfg["min_edge_percent"] / 2:
            return None

        min_liquidity = market.liquidity
        max_position = min(min_liquidity * 0.05, 500.0)

        if max_position < settings.MIN_POSITION_SIZE:
            return None

        # Risk scoring
        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Smart money convergence bet (behavioral edge)",
            f"Confluence: {strength:.0%} strength, {wallet_count} wallets, tier {tier}",
            f"Total tracked volume: ${total_volume:,.0f}",
        ]
        if confidence < 0.6:
            risk_score += 0.15
            risk_factors.append("Moderate confidence")
        if wallet_count <= 2:
            risk_score += 0.1
            risk_factors.append("Minimal wallet convergence")
        if tier == "high":
            risk_score -= 0.05
        elif tier == "extreme":
            risk_score -= 0.1
        risk_score = max(min(risk_score, 1.0), 0.0)

        positions = [
            {
                "action": "BUY",
                "outcome": side,
                "price": entry_price,
                "token_id": token_id,
                "_traders_confluence": {
                    "confluence_strength": strength,
                    "tier": tier,
                    "wallet_count": wallet_count,
                    "total_volume_usd": total_volume,
                    "wallets": wallets[:10],  # Cap for payload size
                    "edge_percent": edge,
                    "confidence": confidence,
                    "direction": direction,
                },
            }
        ]

        market_dict = {
            "id": market.id,
            "slug": market.slug,
            "question": market.question,
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "liquidity": market.liquidity,
        }

        return ArbitrageOpportunity(
            strategy=self.strategy_type,
            title=f"Trader Flow: {wallet_count} wallets → {market.question[:40]}",
            description=(
                f"{wallet_count} tracked wallets ({tier} tier, {strength:.0%} confluence) "
                f"converging on {side} at ${entry_price:.2f} "
                f"(edge: {edge:.1f}%, volume: ${total_volume:,.0f})"
            ),
            total_cost=total_cost,
            expected_payout=expected_payout,
            gross_profit=gross_profit,
            fee=fee_amount,
            net_profit=net_profit,
            roi_percent=roi,
            is_guaranteed=False,
            roi_type="directional_payout",
            risk_score=risk_score,
            risk_factors=risk_factors,
            markets=[market_dict],
            event_id=event.id if event else None,
            event_slug=event.slug if event else None,
            event_title=event.title if event else None,
            category=event.category if event else None,
            min_liquidity=min_liquidity,
            max_position_size=max_position,
            resolution_date=market.end_date,
            mispricing_type=MispricingType.NEWS_INFORMATION,
            positions_to_take=positions,
        )
