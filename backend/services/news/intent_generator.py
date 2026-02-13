"""
Intent Generator -- Converts high-conviction findings into trade intents.

Applies sizing policy and deterministic keys before creating NewsTradeIntent
records that the trader orchestrator can consume.

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
            market_link_payload = {
                "id": finding.market_id,
                "market_id": finding.market_id,
                "slug": market_meta.get("slug"),
                "event_slug": market_meta.get("event_slug"),
                "event_ticker": market_meta.get("event_ticker"),
                "platform": market_meta.get("platform"),
            }
            market_url = self._build_market_url(market_link_payload)

            metadata = {
                "market": {
                    "id": finding.market_id,
                    "slug": market_meta.get("slug"),
                    "event_slug": market_meta.get("event_slug"),
                    "event_ticker": market_meta.get("event_ticker"),
                    "platform": market_meta.get("platform"),
                    "event_title": market_meta.get("event_title"),
                    "liquidity": market_meta.get("liquidity"),
                    "yes_price": market_meta.get("yes_price"),
                    "no_price": market_meta.get("no_price"),
                    "token_ids": token_ids,
                    "market_url": market_url,
                    "url": market_url,
                },
                "finding": {
                    "article_id": finding.article_id,
                    "signal_key": getattr(finding, "signal_key", None),
                    "cache_key": getattr(finding, "cache_key", None),
                    "reasoning": finding.reasoning or "",
                    "evidence": finding.evidence or {},
                },
            }
            supporting_articles = self._extract_supporting_articles(finding)
            metadata["supporting_articles"] = supporting_articles
            metadata["supporting_article_count"] = len(supporting_articles)

            intent = {
                "id": signal_key[:16],
                "signal_key": signal_key,
                "finding_id": finding.id,
                "market_id": finding.market_id,
                "market_question": finding.market_question,
                "direction": finding.direction,
                "entry_price": finding.market_price if finding.direction == "buy_yes" else (1.0 - finding.market_price),
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
    def _extract_supporting_articles(finding: WorkflowFinding) -> list[dict[str, Any]]:
        evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
        cluster = evidence.get("cluster") if isinstance(evidence, dict) else None
        refs: list[dict[str, Any]] = []

        if isinstance(cluster, dict):
            raw_refs = cluster.get("article_refs")
            if isinstance(raw_refs, list):
                for raw in raw_refs:
                    if not isinstance(raw, dict):
                        continue
                    refs.append(
                        {
                            "article_id": str(raw.get("article_id") or ""),
                            "title": str(raw.get("title") or ""),
                            "url": str(raw.get("url") or ""),
                            "source": str(raw.get("source") or ""),
                            "published": raw.get("published"),
                            "fetched_at": raw.get("fetched_at"),
                        }
                    )

        if not refs:
            refs = [
                {
                    "article_id": finding.article_id,
                    "title": finding.article_title,
                    "url": finding.article_url,
                    "source": finding.article_source,
                    "published": None,
                    "fetched_at": finding.created_at.isoformat() if finding.created_at else None,
                }
            ]

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ref in refs:
            key = (
                str(ref.get("article_id") or "").strip()
                or str(ref.get("url") or "").strip()
                or str(ref.get("title") or "").strip().lower()
            )
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped[:8]

    @staticmethod
    def _signal_key(finding: WorkflowFinding) -> str:
        raw = f"{getattr(finding, 'signal_key', '')}:{finding.article_id}:{finding.market_id}:{finding.direction}"
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

    @staticmethod
    def _build_market_url(market_payload: dict[str, Any]) -> str | None:
        from utils.market_urls import build_market_url

        url = build_market_url(
            market_payload,
            opportunity_event_slug=market_payload.get("event_slug"),
        )
        return str(url).strip() if isinstance(url, str) and url.strip() else None

    def clear_cooldowns(self) -> int:
        """Back-compat no-op (cooldowns replaced by DB idempotency)."""
        return 0


# Singleton
intent_generator = IntentGenerator()
