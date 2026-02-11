"""
Intent Generator -- Converts high-conviction findings into trade intents.

Applies sizing policy and deterministic keys before creating NewsTradeIntent
records that the auto-trader can consume.

Pattern from: Quant-tool (signal-to-trade conversion with confidence scoring).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from services.news.edge_estimator import WorkflowFinding

logger = logging.getLogger(__name__)


class IntentGenerator:
    """Generates trade intents from high-conviction workflow findings."""

    # Max suggested position size
    _MAX_SIZE_USD = 500.0
    # Position size as fraction of market liquidity
    _LIQUIDITY_FRACTION = 0.05

    async def generate(
        self,
        findings: list[WorkflowFinding],
        min_edge: float = 10.0,
        min_confidence: float = 0.6,
        market_metadata_by_id: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[dict]:
        """Generate trade intent records from actionable findings.

        Args:
            findings: Workflow findings (already filtered for actionable).
            min_edge: Minimum edge % to create intent.
            min_confidence: Minimum confidence to create intent.
            market_metadata_by_id: Optional metadata map keyed by market_id.

        Returns:
            List of dicts ready for DB insertion as NewsTradeIntent rows.
        """
        intents: list[dict] = []
        now = datetime.now(timezone.utc)
        market_metadata_by_id = market_metadata_by_id or {}

        for finding in findings:
            if not finding.actionable:
                continue
            if finding.edge_percent < min_edge:
                continue
            if finding.confidence < min_confidence:
                continue

            # Compute suggested size
            market_meta = market_metadata_by_id.get(finding.market_id, {})
            suggested_size = self._compute_size(finding, market_meta)
            signal_key = self._signal_key(finding)
            token_ids = market_meta.get("token_ids") or []
            if not isinstance(token_ids, list):
                token_ids = []

            metadata = {
                "market": {
                    "id": finding.market_id,
                    "slug": market_meta.get("slug"),
                    "event_slug": market_meta.get("event_slug"),
                    "event_title": market_meta.get("event_title"),
                    "liquidity": market_meta.get("liquidity"),
                    "yes_price": market_meta.get("yes_price"),
                    "no_price": market_meta.get("no_price"),
                    "token_ids": token_ids,
                },
                "finding": {
                    "article_id": finding.article_id,
                    "signal_key": getattr(finding, "signal_key", None),
                    "cache_key": getattr(finding, "cache_key", None),
                },
            }

            intent = {
                "id": signal_key[:16],
                "signal_key": signal_key,
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
                "metadata_json": metadata,
                "status": "pending",
                "created_at": now,
            }

            intents.append(intent)

        logger.info(
            "Intent generator: %d intents from %d findings",
            len(intents),
            len(findings),
        )
        return intents

    @staticmethod
    def _signal_key(finding: WorkflowFinding) -> str:
        raw = (
            f"{getattr(finding, 'signal_key', '')}:{finding.article_id}:"
            f"{finding.market_id}:{finding.direction}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _compute_size(
        self,
        finding: WorkflowFinding,
        market_meta: Optional[dict[str, Any]] = None,
    ) -> float:
        """Compute suggested position size.

        Conservative sizing for directional news-driven bets:
        - Base: 5% of market liquidity
        - Max: $500
        - Scaled by confidence
        """
        market_meta = market_meta or {}
        liquidity = market_meta.get("liquidity")
        if liquidity is None:
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
        """Back-compat no-op (cooldowns replaced by DB idempotency)."""
        return 0


# Singleton
intent_generator = IntentGenerator()
