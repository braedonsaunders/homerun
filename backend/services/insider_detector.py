"""Insider detection and intent generation for discovered wallets."""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    AsyncSessionLocal,
    DiscoveredWallet,
    InsiderTradeIntent,
    NewsWorkflowFinding,
    WalletActivityRollup,
)
from services.polymarket import polymarket_client
from utils.logger import get_logger

logger = get_logger("insider_detector")


METRIC_WEIGHTS: dict[str, float] = {
    "win_rate_component": 0.10,
    "timing_alpha_component": 0.16,
    "roi_component": 0.08,
    "brier_component": 0.12,
    "entry_resolution_edge_component": 0.10,
    "position_concentration_component": 0.07,
    "pre_news_timing_component": 0.12,
    "market_selection_edge_component": 0.08,
    "drawdown_behavior_component": 0.05,
    "cluster_correlation_component": 0.08,
    "funding_overlap_proxy_component": 0.04,
}

FLAGGED_THRESHOLD = 0.72
WATCH_THRESHOLD = 0.60
FLAGGED_CONFIDENCE = 0.60
WATCH_CONFIDENCE = 0.50
FLAGGED_SAMPLE = 25
WATCH_SAMPLE = 15

MIN_TOTAL_TRADES = 30
MIN_SAMPLE_FOR_DISPLAY = 15

RECENT_TRADE_WINDOW_MINUTES = 90
AGGREGATION_WINDOW_MINUTES = 15
INTENT_MAX_AGE_MINUTES = 240


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _floor_to_minutes(ts: datetime, bucket_minutes: int) -> datetime:
    minute = (ts.minute // bucket_minutes) * bucket_minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


def _normalize_side_to_direction(raw_side: Optional[str]) -> Optional[str]:
    side = (raw_side or "").strip().upper()
    if side in {"BUY", "YES"}:
        return "buy_yes"
    if side in {"SELL", "NO"}:
        return "buy_no"
    return None


def _cohort_signature(wallets: list[str]) -> str:
    joined = "|".join(sorted({w.lower() for w in wallets if w}))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _intent_signal_key(
    market_id: str,
    direction: str,
    bucket_10m: datetime,
    wallets: list[str],
) -> str:
    payload = f"{market_id}|{direction}|{bucket_10m.isoformat()}|{_cohort_signature(wallets)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]


class InsiderDetectorService:
    """Wallet-level insider scoring + trade intent generation."""

    def __init__(self) -> None:
        self.client = polymarket_client
        self._market_context_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    async def rescore_wallets(
        self,
        *,
        stale_minutes: int = 15,
        max_wallets: Optional[int] = None,
    ) -> dict[str, Any]:
        """Rescore eligible wallets and persist insider metrics."""
        now = utcnow()
        stale_cutoff = now - timedelta(minutes=max(1, stale_minutes))

        async with AsyncSessionLocal() as session:
            cluster_counts_rows = await session.execute(
                select(
                    DiscoveredWallet.cluster_id,
                    func.count(DiscoveredWallet.address),
                )
                .where(DiscoveredWallet.cluster_id.is_not(None))
                .group_by(DiscoveredWallet.cluster_id)
            )
            cluster_sizes = {
                str(cluster_id): int(count or 0)
                for cluster_id, count in cluster_counts_rows.all()
                if cluster_id
            }

            query = (
                select(DiscoveredWallet)
                .where(DiscoveredWallet.total_trades >= MIN_TOTAL_TRADES)
                .where(
                    or_(
                        DiscoveredWallet.insider_last_scored_at.is_(None),
                        DiscoveredWallet.insider_last_scored_at < stale_cutoff,
                    )
                )
                .order_by(DiscoveredWallet.last_analyzed_at.desc().nullslast())
            )
            if max_wallets:
                query = query.limit(max(1, max_wallets))

            wallets = list((await session.execute(query)).scalars().all())
            if not wallets:
                return {
                    "scored_wallets": 0,
                    "flagged_insiders": 0,
                    "watch_insiders": 0,
                }

            flagged = 0
            watch = 0

            for wallet in wallets:
                score_result = await self._score_wallet(
                    session=session,
                    wallet=wallet,
                    cluster_sizes=cluster_sizes,
                    now=now,
                )
                wallet.insider_score = score_result["insider_score"]
                wallet.insider_confidence = score_result["insider_confidence"]
                wallet.insider_sample_size = score_result["sample_size"]
                wallet.insider_last_scored_at = now
                wallet.insider_metrics_json = score_result["metrics"]
                wallet.insider_reasons_json = score_result["reasons"]

                if score_result["classification"] == "flagged_insider":
                    flagged += 1
                elif score_result["classification"] == "watch_insider":
                    watch += 1

            await session.commit()

        logger.info(
            "Insider wallet rescoring complete",
            scored_wallets=len(wallets),
            flagged_insiders=flagged,
            watch_insiders=watch,
        )
        return {
            "scored_wallets": len(wallets),
            "flagged_insiders": flagged,
            "watch_insiders": watch,
        }

    async def generate_intents(self) -> dict[str, Any]:
        """Generate/upsert insider trade intents from recent flagged-wallet activity."""
        now = utcnow()
        cutoff_recent = now - timedelta(minutes=RECENT_TRADE_WINDOW_MINUTES)

        async with AsyncSessionLocal() as session:
            await self._expire_stale_intents(session, max_age_minutes=INTENT_MAX_AGE_MINUTES)

            flagged_wallets_result = await session.execute(
                select(DiscoveredWallet).where(
                    DiscoveredWallet.insider_score >= FLAGGED_THRESHOLD,
                    DiscoveredWallet.insider_confidence >= FLAGGED_CONFIDENCE,
                    DiscoveredWallet.insider_sample_size >= FLAGGED_SAMPLE,
                )
            )
            flagged_wallets = list(flagged_wallets_result.scalars().all())
            if not flagged_wallets:
                return {"created_or_updated": 0, "candidates_seen": 0}

            wallet_map = {w.address.lower(): w for w in flagged_wallets}
            addresses = list(wallet_map.keys())

            events_result = await session.execute(
                select(WalletActivityRollup)
                .where(
                    WalletActivityRollup.wallet_address.in_(addresses),
                    WalletActivityRollup.traded_at >= cutoff_recent,
                )
                .order_by(WalletActivityRollup.traded_at.asc())
            )
            events = list(events_result.scalars().all())

            grouped: dict[tuple[str, str, datetime], list[WalletActivityRollup]] = defaultdict(list)
            for event in events:
                direction = _normalize_side_to_direction(event.side)
                if direction is None:
                    continue
                if event.price is None or event.price <= 0:
                    continue
                bucket = _floor_to_minutes(event.traded_at, AGGREGATION_WINDOW_MINUTES)
                grouped[(event.market_id, direction, bucket)].append(event)

            created_or_updated = 0
            candidates_seen = 0

            for (market_id, direction, bucket), group_events in grouped.items():
                candidates_seen += 1

                wallets = sorted({e.wallet_address.lower() for e in group_events if e.wallet_address})
                if not wallets:
                    continue

                wallet_rows = [wallet_map[w] for w in wallets if w in wallet_map]
                if not wallet_rows:
                    continue

                wallet_count = len(wallet_rows)
                cluster_count = len(
                    {
                        w.cluster_id
                        for w in wallet_rows
                        if w.cluster_id
                    }
                )
                avg_insider_score = sum(_safe_float(w.insider_score) for w in wallet_rows) / wallet_count
                avg_insider_conf = sum(_safe_float(w.insider_confidence) for w in wallet_rows) / wallet_count

                entry_prices = [
                    _safe_float(e.price)
                    for e in group_events
                    if e.price is not None and e.price > 0
                ]
                entry_price = (sum(entry_prices) / len(entry_prices)) if entry_prices else None

                pre_news_lead_values: list[float] = []
                pre_news_components: list[float] = []
                timing_short_values: list[float] = []

                for wallet in wallet_rows:
                    metrics = wallet.insider_metrics_json or {}
                    raw = metrics.get("raw", {}) if isinstance(metrics, dict) else {}
                    components = metrics.get("components", {}) if isinstance(metrics, dict) else {}

                    lead = raw.get("pre_news_lead_minutes")
                    if isinstance(lead, (int, float)):
                        pre_news_lead_values.append(float(lead))

                    pre_news_component = components.get("pre_news_timing_component")
                    if isinstance(pre_news_component, (int, float)):
                        pre_news_components.append(float(pre_news_component))

                    timing_short = raw.get("timing_alpha_short")
                    if isinstance(timing_short, (int, float)):
                        timing_short_values.append(float(timing_short))

                pre_news_lead = (
                    sum(pre_news_lead_values) / len(pre_news_lead_values)
                    if pre_news_lead_values
                    else 0.0
                )
                pre_news_component = (
                    sum(pre_news_components) / len(pre_news_components)
                    if pre_news_components
                    else 0.0
                )
                timing_alpha_short = (
                    sum(timing_short_values) / len(timing_short_values)
                    if timing_short_values
                    else 0.5
                )

                market_context = await self._get_market_context(market_id)
                liquidity = _safe_float(market_context.get("liquidity"), 0.0)
                market_question = (
                    str(market_context.get("question") or "").strip() or f"Market {market_id}"
                )

                confidence = _clamp(
                    0.40 * avg_insider_score
                    + 0.35 * avg_insider_conf
                    + 0.15 * _clamp(wallet_count / 4.0)
                    + 0.10 * _clamp(timing_alpha_short),
                    0.0,
                    0.99,
                )

                if entry_price is None:
                    continue
                side_market_prob = entry_price if direction == "buy_yes" else (1.0 - entry_price)
                model_side_prob = _clamp(
                    side_market_prob
                    + (avg_insider_score - 0.5) * 0.22
                    + (timing_alpha_short - 0.5) * 0.14
                    + (pre_news_component - 0.5) * 0.12,
                    0.02,
                    0.98,
                )
                edge_percent = (model_side_prob - side_market_prob) * 100.0

                strong_single_wallet_override = (
                    wallet_count == 1
                    and avg_insider_score >= 0.82
                    and pre_news_component >= 0.70
                )

                if liquidity < 300:
                    continue
                if confidence < 0.62:
                    continue
                if not (wallet_count >= 2 or strong_single_wallet_override):
                    continue
                if edge_percent <= 0.5:
                    continue

                wallet_sig = _cohort_signature(wallets)
                bucket_10m = _floor_to_minutes(bucket, 10)
                signal_key = _intent_signal_key(market_id, direction, bucket_10m, wallets)

                suggested_size = _clamp(
                    18.0
                    + confidence * 45.0
                    + min(wallet_count, 4) * 6.0
                    + _clamp(edge_percent / 6.0, 0.0, 12.0),
                    10.0,
                    180.0,
                )

                metadata = {
                    "wallet_count": wallet_count,
                    "cluster_count": cluster_count,
                    "pre_news_lead_minutes": round(pre_news_lead, 2),
                    "timing_alpha_short": round(timing_alpha_short, 4),
                    "market_liquidity": liquidity,
                    "market_slug": market_context.get("market_slug"),
                    "wallet_cohort_signature": wallet_sig,
                    "funding_overlap_method": "proxy_cluster_timing",
                    "generated_at": now.isoformat(),
                }

                intent_created = await self._upsert_intent(
                    session=session,
                    signal_key=signal_key,
                    market_id=market_id,
                    market_question=market_question,
                    direction=direction,
                    entry_price=entry_price,
                    edge_percent=edge_percent,
                    confidence=confidence,
                    insider_score=avg_insider_score,
                    wallet_addresses=wallets,
                    suggested_size_usd=suggested_size,
                    metadata=metadata,
                )
                if intent_created:
                    created_or_updated += 1

            await session.commit()

        logger.info(
            "Insider intent generation complete",
            candidates_seen=candidates_seen,
            created_or_updated=created_or_updated,
        )
        return {
            "created_or_updated": created_or_updated,
            "candidates_seen": candidates_seen,
        }

    async def list_intents(
        self,
        *,
        status_filter: Optional[str] = None,
        limit: int = 100,
    ) -> list[InsiderTradeIntent]:
        async with AsyncSessionLocal() as session:
            query = select(InsiderTradeIntent).order_by(InsiderTradeIntent.created_at.desc())
            if status_filter:
                query = query.where(InsiderTradeIntent.status == status_filter)
            query = query.limit(max(1, min(limit, 2000)))
            return list((await session.execute(query)).scalars().all())

    async def list_opportunities(
        self,
        *,
        min_confidence: float = 0.0,
        direction: Optional[str] = None,
        max_age_minutes: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            query = select(InsiderTradeIntent).where(
                InsiderTradeIntent.status.in_(("pending", "submitted"))
            )

            if min_confidence > 0:
                query = query.where(InsiderTradeIntent.confidence >= min_confidence)
            if direction:
                query = query.where(InsiderTradeIntent.direction == direction)
            if max_age_minutes:
                cutoff = utcnow() - timedelta(minutes=max(1, max_age_minutes))
                query = query.where(InsiderTradeIntent.created_at >= cutoff)

            count_q = select(func.count()).select_from(query.subquery())
            total = int((await session.execute(count_q)).scalar() or 0)

            query = (
                query.order_by(
                    InsiderTradeIntent.confidence.desc(),
                    InsiderTradeIntent.created_at.desc(),
                )
                .offset(max(0, offset))
                .limit(max(1, min(limit, 500)))
            )

            intents = list((await session.execute(query)).scalars().all())
            address_set: set[str] = set()
            for row in intents:
                for addr in (row.wallet_addresses_json or []):
                    if isinstance(addr, str):
                        address_set.add(addr.lower())

            wallet_map: dict[str, DiscoveredWallet] = {}
            if address_set:
                wallets_res = await session.execute(
                    select(DiscoveredWallet).where(DiscoveredWallet.address.in_(list(address_set)))
                )
                wallet_map = {w.address.lower(): w for w in wallets_res.scalars().all()}

            opportunities: list[dict[str, Any]] = []
            now = utcnow()
            for row in intents:
                wallet_addresses = [
                    str(addr).lower()
                    for addr in (row.wallet_addresses_json or [])
                    if isinstance(addr, str)
                ]
                wallets_payload = []
                for addr in wallet_addresses:
                    profile = wallet_map.get(addr)
                    wallets_payload.append(
                        {
                            "address": addr,
                            "username": profile.username if profile else None,
                            "insider_score": _safe_float(profile.insider_score if profile else 0.0),
                            "insider_confidence": _safe_float(
                                profile.insider_confidence if profile else 0.0
                            ),
                        }
                    )

                wallets_payload.sort(
                    key=lambda item: (
                        _safe_float(item.get("insider_score"), 0.0),
                        _safe_float(item.get("insider_confidence"), 0.0),
                    ),
                    reverse=True,
                )
                top_wallet = wallets_payload[0] if wallets_payload else None

                metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
                created_at = row.created_at or now
                opportunities.append(
                    {
                        "id": row.id,
                        "signal_key": row.signal_key,
                        "market_id": row.market_id,
                        "market_question": row.market_question,
                        "direction": row.direction,
                        "entry_price": row.entry_price,
                        "edge_percent": row.edge_percent,
                        "confidence": row.confidence,
                        "insider_score": row.insider_score,
                        "wallet_addresses": wallet_addresses,
                        "wallet_count": int(metadata.get("wallet_count", len(wallet_addresses))),
                        "cluster_count": int(metadata.get("cluster_count", 0)),
                        "pre_news_lead_minutes": _safe_float(
                            metadata.get("pre_news_lead_minutes"),
                            0.0,
                        ),
                        "timing_alpha_short": _safe_float(
                            metadata.get("timing_alpha_short"),
                            0.5,
                        ),
                        "market_liquidity": _safe_float(metadata.get("market_liquidity"), 0.0),
                        "suggested_size_usd": row.suggested_size_usd,
                        "status": row.status,
                        "freshness_minutes": max(
                            0.0,
                            (now - created_at).total_seconds() / 60.0,
                        ),
                        "top_wallet": top_wallet,
                        "wallets": wallets_payload,
                        "created_at": created_at.isoformat() if created_at else None,
                    }
                )

            return {
                "total": total,
                "offset": max(0, offset),
                "limit": max(1, min(limit, 500)),
                "opportunities": opportunities,
            }

    # ------------------------------------------------------------------
    # Wallet scoring internals
    # ------------------------------------------------------------------

    async def _score_wallet(
        self,
        *,
        session: AsyncSession,
        wallet: DiscoveredWallet,
        cluster_sizes: dict[str, int],
        now: datetime,
    ) -> dict[str, Any]:
        resolved_sample = int((wallet.wins or 0) + (wallet.losses or 0))

        events = await self._load_wallet_events(session, wallet.address, now=now)
        market_ids = sorted({e.market_id for e in events if e.market_id})
        market_events = await self._load_market_events(
            session,
            market_ids=market_ids,
            start=(events[0].traded_at - timedelta(hours=1)) if events else now,
            end=(events[-1].traded_at + timedelta(hours=25)) if events else now,
        )

        (
            timing_component,
            timing_short_raw,
            entry_resolution_component,
            market_selection_component,
            concentration_component,
            cluster_component,
        ) = self._compute_trade_pattern_components(
            wallet=wallet,
            events=events,
            market_events=market_events,
        )

        pre_news_component, pre_news_lead_minutes = await self._compute_pre_news_component(
            session=session,
            events=events,
        )

        cluster_size = (
            cluster_sizes.get(wallet.cluster_id, 0)
            if wallet.cluster_id
            else 0
        )
        funding_overlap_component = self._funding_overlap_proxy_component(
            cluster_correlation_component=cluster_component,
            cluster_size=cluster_size,
        )

        win_rate_component = _clamp(_safe_float(wallet.win_rate, 0.0))
        roi_component = self._roi_component(_safe_float(wallet.avg_roi, 0.0))
        brier_component = self._brier_component(
            win_rate=_safe_float(wallet.win_rate, 0.0),
            events=events,
        )
        drawdown_component = self._drawdown_behavior_component(
            max_drawdown=_safe_float(wallet.max_drawdown, 0.0),
            timing_component=timing_component,
        )

        components: dict[str, Optional[float]] = {
            "win_rate_component": win_rate_component if resolved_sample >= MIN_SAMPLE_FOR_DISPLAY else None,
            "timing_alpha_component": timing_component,
            "roi_component": roi_component if resolved_sample >= MIN_SAMPLE_FOR_DISPLAY else None,
            "brier_component": brier_component if resolved_sample >= MIN_SAMPLE_FOR_DISPLAY else None,
            "entry_resolution_edge_component": entry_resolution_component,
            "position_concentration_component": concentration_component,
            "pre_news_timing_component": pre_news_component,
            "market_selection_edge_component": market_selection_component,
            "drawdown_behavior_component": drawdown_component,
            "cluster_correlation_component": cluster_component,
            "funding_overlap_proxy_component": funding_overlap_component,
        }

        weighted_sum = 0.0
        available_weight_sum = 0.0
        for metric_key, weight in METRIC_WEIGHTS.items():
            value = components.get(metric_key)
            if value is None:
                continue
            weighted_sum += weight * _clamp(value)
            available_weight_sum += weight

        sample_factor = _clamp(resolved_sample / 40.0)
        insider_confidence = available_weight_sum * sample_factor

        base_score = (weighted_sum / available_weight_sum) if available_weight_sum > 0 else 0.0
        confidence_penalty = self._confidence_penalty(insider_confidence)
        insider_score = _clamp(base_score * confidence_penalty)

        classification = self._classify_wallet(
            insider_score=insider_score,
            insider_confidence=insider_confidence,
            sample_size=resolved_sample,
        )

        sorted_components = sorted(
            [
                (k, v)
                for k, v in components.items()
                if isinstance(v, (int, float))
            ],
            key=lambda item: float(item[1]),
            reverse=True,
        )
        reasons = [
            f"{key.replace('_component', '').replace('_', ' ')}={float(value):.3f}"
            for key, value in sorted_components[:5]
        ]
        if classification == "flagged_insider":
            reasons.insert(0, "flagged_insider_thresholds_met")
        elif classification == "watch_insider":
            reasons.insert(0, "watch_insider_thresholds_met")

        metrics = {
            "components": {
                key: (_clamp(float(value)) if isinstance(value, (int, float)) else None)
                for key, value in components.items()
            },
            "weights": METRIC_WEIGHTS,
            "available_weight_sum": round(available_weight_sum, 6),
            "weighted_base_score": round(base_score, 6),
            "confidence_penalty": round(confidence_penalty, 6),
            "confidence": round(insider_confidence, 6),
            "classification": classification,
            "raw": {
                "resolved_sample": resolved_sample,
                "timing_alpha_short": round(timing_short_raw, 6) if timing_short_raw is not None else None,
                "pre_news_lead_minutes": round(pre_news_lead_minutes, 3)
                if pre_news_lead_minutes is not None
                else None,
                "cluster_size": cluster_size,
                "funding_overlap_method": "proxy_cluster_timing",
            },
        }

        return {
            "insider_score": round(insider_score, 6),
            "insider_confidence": round(insider_confidence, 6),
            "sample_size": resolved_sample,
            "classification": classification,
            "metrics": metrics,
            "reasons": reasons,
        }

    async def _load_wallet_events(
        self,
        session: AsyncSession,
        wallet_address: str,
        *,
        now: datetime,
        lookback_days: int = 45,
        limit: int = 1400,
    ) -> list[WalletActivityRollup]:
        cutoff = now - timedelta(days=max(7, lookback_days))
        rows = await session.execute(
            select(WalletActivityRollup)
            .where(
                WalletActivityRollup.wallet_address == wallet_address.lower(),
                WalletActivityRollup.traded_at >= cutoff,
            )
            .order_by(WalletActivityRollup.traded_at.asc())
            .limit(max(100, limit))
        )
        return list(rows.scalars().all())

    async def _load_market_events(
        self,
        session: AsyncSession,
        *,
        market_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, list[WalletActivityRollup]]:
        if not market_ids:
            return {}

        rows = await session.execute(
            select(WalletActivityRollup)
            .where(
                WalletActivityRollup.market_id.in_(market_ids),
                WalletActivityRollup.traded_at >= start,
                WalletActivityRollup.traded_at <= end,
            )
            .order_by(WalletActivityRollup.market_id.asc(), WalletActivityRollup.traded_at.asc())
            .limit(30000)
        )
        out: dict[str, list[WalletActivityRollup]] = defaultdict(list)
        for row in rows.scalars().all():
            out[str(row.market_id)].append(row)
        return out

    def _compute_trade_pattern_components(
        self,
        *,
        wallet: DiscoveredWallet,
        events: list[WalletActivityRollup],
        market_events: dict[str, list[WalletActivityRollup]],
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        if not events or not market_events:
            return (None, None, None, None, None, None)

        one_hour_moves: list[float] = []
        six_hour_moves: list[float] = []
        day_moves: list[float] = []
        chosen_side_probs: list[float] = []
        concentration_ratios: list[float] = []
        low_market_moves: list[float] = []
        high_market_moves: list[float] = []
        coincidence_scores: list[float] = []

        market_total_notional: dict[str, float] = {}
        for market_id, rows in market_events.items():
            market_total_notional[market_id] = sum(abs(_safe_float(r.notional, 0.0)) for r in rows)

        market_notionals = sorted(v for v in market_total_notional.values() if v > 0)
        low_threshold = market_notionals[len(market_notionals) // 2] if market_notionals else 0.0

        # Build fast index for (market, direction, 15m bucket) => unique wallets
        bucket_index: dict[tuple[str, str, datetime], set[str]] = defaultdict(set)
        for market_id, rows in market_events.items():
            for row in rows:
                direction = _normalize_side_to_direction(row.side)
                if direction is None or not row.wallet_address:
                    continue
                b = _floor_to_minutes(row.traded_at, 15)
                bucket_index[(market_id, direction, b)].add(row.wallet_address.lower())

        for event in events:
            market_id = str(event.market_id)
            direction = _normalize_side_to_direction(event.side)
            entry_price = _safe_float(event.price, -1.0)
            if direction is None or entry_price <= 0 or entry_price >= 1:
                continue

            rows = market_events.get(market_id) or []
            if not rows:
                continue

            sign = 1.0 if direction == "buy_yes" else -1.0
            selected_prob = entry_price if direction == "buy_yes" else (1.0 - entry_price)
            chosen_side_probs.append(_clamp(selected_prob))

            # Find nearest future prices at 1h, 6h, 24h.
            targets = (
                ("1h", event.traded_at + timedelta(hours=1)),
                ("6h", event.traded_at + timedelta(hours=6)),
                ("24h", event.traded_at + timedelta(hours=24)),
            )
            for label, target_ts in targets:
                future_price = self._find_first_future_price(rows, target_ts)
                if future_price is None:
                    continue
                move = sign * (future_price - entry_price)
                if label == "1h":
                    one_hour_moves.append(move)
                elif label == "6h":
                    six_hour_moves.append(move)
                else:
                    day_moves.append(move)

            # Concentration proxy: wallet trade notional vs market notional.
            market_notional = max(1e-6, market_total_notional.get(market_id, 0.0))
            ratio = abs(_safe_float(event.notional, 0.0)) / market_notional
            concentration_ratios.append(_clamp(ratio, 0.0, 1.0))

            # Market-selection edge proxy: low-liquidity markets vs high-liquidity.
            six_h_future = self._find_first_future_price(rows, event.traded_at + timedelta(hours=6))
            if six_h_future is not None:
                move_6h = sign * (six_h_future - entry_price)
                if market_total_notional.get(market_id, 0.0) <= low_threshold:
                    low_market_moves.append(move_6h)
                else:
                    high_market_moves.append(move_6h)

            # Cluster-correlation proxy from synchronized entries.
            bucket = _floor_to_minutes(event.traded_at, 15)
            peers = bucket_index.get((market_id, direction, bucket), set())
            peer_count = len([addr for addr in peers if addr != wallet.address.lower()])
            coincidence_scores.append(_clamp(peer_count / 5.0))

        if not one_hour_moves and not six_hour_moves and not day_moves:
            timing_component = None
            timing_short = None
        else:
            avg_1h = (sum(one_hour_moves) / len(one_hour_moves)) if one_hour_moves else 0.0
            avg_6h = (sum(six_hour_moves) / len(six_hour_moves)) if six_hour_moves else avg_1h
            avg_24h = (sum(day_moves) / len(day_moves)) if day_moves else avg_6h
            alpha_mix = (0.50 * avg_1h) + (0.30 * avg_6h) + (0.20 * avg_24h)
            timing_component = _clamp(0.5 + math.tanh(alpha_mix * 8.0) * 0.5)
            timing_short = _clamp(0.5 + math.tanh(avg_1h * 10.0) * 0.5)

        if chosen_side_probs:
            win_rate = _clamp(_safe_float(wallet.win_rate, 0.0))
            avg_implied = sum(chosen_side_probs) / len(chosen_side_probs)
            edge = win_rate - avg_implied
            entry_resolution_component = _clamp((edge + 0.12) / 0.30)
        else:
            entry_resolution_component = None

        if low_market_moves:
            low_avg = sum(low_market_moves) / len(low_market_moves)
            high_avg = sum(high_market_moves) / len(high_market_moves) if high_market_moves else 0.0
            delta = low_avg - high_avg
            market_selection_component = _clamp(0.5 + math.tanh(delta * 8.0) * 0.5)
        else:
            market_selection_component = None

        if concentration_ratios:
            avg_ratio = sum(concentration_ratios) / len(concentration_ratios)
            concentration_component = _clamp(avg_ratio / 0.12)
        else:
            concentration_component = None

        if coincidence_scores:
            cluster_component = _clamp(sum(coincidence_scores) / len(coincidence_scores))
        else:
            cluster_component = None

        return (
            timing_component,
            timing_short,
            entry_resolution_component,
            market_selection_component,
            concentration_component,
            cluster_component,
        )

    async def _compute_pre_news_component(
        self,
        *,
        session: AsyncSession,
        events: list[WalletActivityRollup],
    ) -> tuple[Optional[float], Optional[float]]:
        if not events:
            return None, None

        market_ids = sorted({e.market_id for e in events if e.market_id})
        if not market_ids:
            return None, None

        start = events[0].traded_at
        end = events[-1].traded_at + timedelta(hours=24)

        rows = await session.execute(
            select(NewsWorkflowFinding)
            .where(
                NewsWorkflowFinding.market_id.in_(market_ids),
                NewsWorkflowFinding.created_at >= start,
                NewsWorkflowFinding.created_at <= end,
            )
            .order_by(NewsWorkflowFinding.created_at.asc())
            .limit(20000)
        )
        findings = list(rows.scalars().all())
        if not findings:
            return None, None

        by_market: dict[str, list[datetime]] = defaultdict(list)
        for finding in findings:
            if finding.market_id and finding.created_at:
                by_market[str(finding.market_id)].append(finding.created_at)

        lead_minutes_values: list[float] = []
        for event in events:
            market_times = by_market.get(str(event.market_id)) or []
            if not market_times:
                continue
            # Nearest relevant release *after* the trade.
            lead = None
            for ts in market_times:
                if ts <= event.traded_at:
                    continue
                delta_m = (ts - event.traded_at).total_seconds() / 60.0
                if delta_m < 0 or delta_m > 24 * 60:
                    continue
                lead = delta_m
                break
            if lead is not None:
                lead_minutes_values.append(lead)

        if not lead_minutes_values:
            return None, None

        avg_lead = sum(lead_minutes_values) / len(lead_minutes_values)
        component = _clamp(math.log1p(avg_lead) / math.log1p(240.0))
        return component, avg_lead

    def _funding_overlap_proxy_component(
        self,
        *,
        cluster_correlation_component: Optional[float],
        cluster_size: int,
    ) -> Optional[float]:
        if cluster_correlation_component is None and cluster_size <= 1:
            return None
        sync = _clamp(_safe_float(cluster_correlation_component, 0.0))
        cluster_norm = _clamp((max(0, cluster_size - 1)) / 6.0)
        return _clamp((0.65 * sync) + (0.35 * cluster_norm))

    def _confidence_penalty(self, insider_confidence: float) -> float:
        return _clamp(insider_confidence / 0.75)

    def _classify_wallet(
        self,
        *,
        insider_score: float,
        insider_confidence: float,
        sample_size: int,
    ) -> str:
        if (
            insider_score >= FLAGGED_THRESHOLD
            and insider_confidence >= FLAGGED_CONFIDENCE
            and sample_size >= FLAGGED_SAMPLE
        ):
            return "flagged_insider"
        if (
            insider_score >= WATCH_THRESHOLD
            and insider_confidence >= WATCH_CONFIDENCE
            and sample_size >= WATCH_SAMPLE
        ):
            return "watch_insider"
        return "none"

    def _roi_component(self, avg_roi: float) -> float:
        # -20% => 0.0, +60% => 1.0
        return _clamp((avg_roi + 20.0) / 80.0)

    def _brier_component(
        self,
        *,
        win_rate: float,
        events: list[WalletActivityRollup],
    ) -> Optional[float]:
        probs: list[float] = []
        for event in events:
            direction = _normalize_side_to_direction(event.side)
            entry_price = _safe_float(event.price, -1.0)
            if direction is None or entry_price <= 0 or entry_price >= 1:
                continue
            implied = entry_price if direction == "buy_yes" else (1.0 - entry_price)
            probs.append(_clamp(implied))

        if not probs:
            return None

        avg_prob = sum(probs) / len(probs)
        # Proxy brier against aggregate hit-rate in absence of per-market outcomes.
        brier = (avg_prob - win_rate) ** 2 + (win_rate * (1.0 - win_rate))
        return _clamp(1.0 - _clamp(brier / 0.35))

    def _drawdown_behavior_component(
        self,
        *,
        max_drawdown: float,
        timing_component: Optional[float],
    ) -> Optional[float]:
        dd_inv = _clamp(1.0 - _clamp(max_drawdown, 0.0, 1.0))
        if timing_component is None:
            return dd_inv
        return _clamp((0.60 * dd_inv) + (0.40 * _clamp(timing_component)))

    @staticmethod
    def _find_first_future_price(
        rows: list[WalletActivityRollup],
        target: datetime,
    ) -> Optional[float]:
        for row in rows:
            if row.traded_at < target:
                continue
            price = _safe_float(row.price, -1.0)
            if 0 < price < 1:
                return price
        return None

    # ------------------------------------------------------------------
    # Intent internals
    # ------------------------------------------------------------------

    async def _upsert_intent(
        self,
        *,
        session: AsyncSession,
        signal_key: str,
        market_id: str,
        market_question: str,
        direction: str,
        entry_price: float,
        edge_percent: float,
        confidence: float,
        insider_score: float,
        wallet_addresses: list[str],
        suggested_size_usd: float,
        metadata: dict[str, Any],
    ) -> bool:
        result = await session.execute(
            select(InsiderTradeIntent).where(InsiderTradeIntent.signal_key == signal_key)
        )
        existing = result.scalar_one_or_none()
        if existing is None:
            session.add(
                InsiderTradeIntent(
                    id=uuid.uuid4().hex,
                    signal_key=signal_key,
                    market_id=market_id,
                    market_question=market_question,
                    direction=direction,
                    entry_price=entry_price,
                    edge_percent=edge_percent,
                    confidence=confidence,
                    insider_score=insider_score,
                    wallet_addresses_json=wallet_addresses,
                    suggested_size_usd=suggested_size_usd,
                    metadata_json=metadata,
                    status="pending",
                    created_at=utcnow(),
                )
            )
            return True

        if existing.status in {"pending", "submitted"}:
            existing.market_question = market_question
            existing.entry_price = entry_price
            existing.edge_percent = edge_percent
            existing.confidence = confidence
            existing.insider_score = insider_score
            existing.wallet_addresses_json = wallet_addresses
            existing.suggested_size_usd = suggested_size_usd
            existing.metadata_json = metadata
            if existing.status == "submitted":
                # Keep submitted unless signal becomes stale elsewhere.
                existing.status = "submitted"
            return True

        return False

    async def _expire_stale_intents(
        self,
        session: AsyncSession,
        *,
        max_age_minutes: int,
    ) -> int:
        cutoff = utcnow() - timedelta(minutes=max(1, max_age_minutes))
        result = await session.execute(
            update(InsiderTradeIntent)
            .where(
                InsiderTradeIntent.status == "pending",
                InsiderTradeIntent.created_at < cutoff,
            )
            .values(status="expired", consumed_at=utcnow())
        )
        return int(result.rowcount or 0)

    async def _get_market_context(self, market_id: str) -> dict[str, Any]:
        now = utcnow()
        cached = self._market_context_cache.get(market_id)
        if cached:
            cached_at, payload = cached
            if now - cached_at <= timedelta(minutes=10):
                return payload

        payload = {
            "question": "",
            "liquidity": 0.0,
            "market_slug": None,
        }

        try:
            if str(market_id).startswith("0x"):
                info = await self.client.get_market_by_condition_id(str(market_id))
            else:
                info = await self.client.get_market_by_token_id(str(market_id))
            if info:
                payload["question"] = str(info.get("question") or "")
                payload["liquidity"] = _safe_float(info.get("liquidity"), 0.0)
                payload["market_slug"] = info.get("event_slug") or info.get("slug")
        except Exception:
            # Use fallback payload if market metadata fetch fails.
            pass

        self._market_context_cache[market_id] = (now, payload)
        return payload


insider_detector = InsiderDetectorService()
