"""
Intent Generator -- Converts high-conviction findings into trade intents.

Applies sizing policy, cooldown windows, and duplicate suppression before
creating NewsTradeIntent records that the auto-trader can consume.

Pattern from: Quant-tool (signal-to-trade conversion with confidence scoring).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from services.news.edge_estimator import WorkflowFinding

logger = logging.getLogger(__name__)


class IntentGenerator:
    """Generates trade intents from high-conviction workflow findings."""

    # Cooldown: don't create duplicate intents for same market within this window
    _COOLDOWN_MINUTES = 30
    # Max suggested position size
    _MAX_SIZE_USD = 500.0
    # Position size as fraction of market liquidity
    _LIQUIDITY_FRACTION = 0.05

    def __init__(self) -> None:
        # In-memory cooldown tracker: market_id -> last intent created_at
        self._cooldowns: dict[str, datetime] = {}

    async def generate(
        self,
        findings: list[WorkflowFinding],
        min_edge: float = 10.0,
        min_confidence: float = 0.6,
    ) -> list[dict]:
        """Generate trade intent records from actionable findings.

        Args:
            findings: Workflow findings (already filtered for actionable).
            min_edge: Minimum edge % to create intent.
            min_confidence: Minimum confidence to create intent.

        Returns:
            List of dicts ready for DB insertion as NewsTradeIntent rows.
        """
        intents: list[dict] = []
        now = datetime.now(timezone.utc)

        for finding in findings:
            if not finding.actionable:
                continue
            if finding.edge_percent < min_edge:
                continue
            if finding.confidence < min_confidence:
                continue

            # Cooldown check
            if self._is_cooled_down(finding.market_id, now):
                logger.debug(
                    "Skipping intent for %s (cooldown active)", finding.market_id
                )
                continue

            # Compute suggested size
            suggested_size = self._compute_size(finding)

            intent = {
                "id": uuid.uuid4().hex[:16],
                "finding_id": finding.id,
                "market_id": finding.market_id,
                "market_question": finding.market_question,
                "direction": finding.direction,
                "entry_price": finding.market_price
                if finding.direction == "buy_yes"
                else (1.0 - finding.market_price),
                "model_probability": finding.model_probability,
                "edge_percent": finding.edge_percent,
                "confidence": finding.confidence,
                "suggested_size_usd": suggested_size,
                "status": "pending",
                "created_at": now,
            }

            intents.append(intent)
            self._cooldowns[finding.market_id] = now

        logger.info(
            "Intent generator: %d intents from %d findings",
            len(intents),
            len(findings),
        )
        return intents

    def _is_cooled_down(self, market_id: str, now: datetime) -> bool:
        """Check if this market is in cooldown."""
        last = self._cooldowns.get(market_id)
        if last is None:
            return False
        return (now - last) < timedelta(minutes=self._COOLDOWN_MINUTES)

    def _compute_size(self, finding: WorkflowFinding) -> float:
        """Compute suggested position size.

        Conservative sizing for directional news-driven bets:
        - Base: 5% of market liquidity
        - Max: $500
        - Scaled by confidence
        """
        # Get liquidity from evidence if available
        liquidity = finding.evidence.get("retrieval", {}).get("liquidity", 5000.0)
        if not liquidity or liquidity <= 0:
            liquidity = 5000.0

        base_size = min(liquidity * self._LIQUIDITY_FRACTION, self._MAX_SIZE_USD)

        # Scale by confidence (0.6 confidence = 60% of base size)
        size = base_size * finding.confidence

        # Minimum viable size
        size = max(size, 1.0)

        return round(size, 2)

    def clear_cooldowns(self) -> int:
        """Clear all cooldowns. Returns count cleared."""
        count = len(self._cooldowns)
        self._cooldowns.clear()
        return count


# Singleton
intent_generator = IntentGenerator()
