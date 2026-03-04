"""Bridge: convert Opportunity objects into TradeSignal DB rows.

Workers and the scanner emit Opportunity objects.  This module converts
those opportunities into normalized TradeSignal rows using a single upsert
pattern shared by all signal sources.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Any

from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal
from models.opportunity import Opportunity
from services.event_bus import event_bus
from services.signal_bus import (
    build_signal_contract_from_opportunity,
    expire_source_signals_except,
    make_dedupe_key,
    refresh_trade_signal_snapshots,
    upsert_trade_signal,
)
from services.trade_signal_stream import publish_trade_signal_batch as publish_trade_signal_stream_batch
from services.worker_state import _commit_with_retry
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)
_snapshot_refresh_task: asyncio.Task[None] | None = None
_snapshot_refresh_last_started_mono = 0.0
_SNAPSHOT_REFRESH_MIN_INTERVAL_SECONDS = 0.25


def _parse_datetime_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts <= 0:
            return None
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def _refresh_trade_signal_snapshots_background() -> None:
    try:
        async with AsyncSessionLocal() as refresh_session:
            await refresh_trade_signal_snapshots(refresh_session)
    except Exception as exc:
        logger.warning("Deferred trade signal snapshot refresh failed", exc_info=exc)


def _schedule_trade_signal_snapshot_refresh() -> None:
    global _snapshot_refresh_task
    global _snapshot_refresh_last_started_mono
    if _snapshot_refresh_task is not None and not _snapshot_refresh_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    now_mono = loop.time()
    if (now_mono - _snapshot_refresh_last_started_mono) < _SNAPSHOT_REFRESH_MIN_INTERVAL_SECONDS:
        return
    _snapshot_refresh_last_started_mono = now_mono
    _snapshot_refresh_task = loop.create_task(
        _refresh_trade_signal_snapshots_background(),
        name="trade-signal-snapshot-refresh",
    )


async def bridge_opportunities_to_signals(
    session: AsyncSession,
    opportunities: list[Opportunity],
    source: str,
    *,
    signal_type_override: str | None = None,
    default_ttl_minutes: int = 120,
    quality_filter_pipeline: Optional[Any] = None,
    quality_reports: Optional[dict] = None,
    sweep_missing: bool = False,
    refresh_prices: bool = True,
) -> int:
    """Convert Opportunity objects into TradeSignal rows.

    Each opportunity is upserted by (source, dedupe_key) so repeated
    detections update rather than duplicate.

    Quality information can be supplied in two ways:

    *   ``quality_filter_pipeline`` — a ``QualityFilterPipeline`` instance.
        Each opportunity is evaluated inline before upserting.
    *   ``quality_reports`` — a dict mapping ``opp.stable_id`` (or ``opp.id``)
        to a pre-computed ``QualityReport``.  Used when the caller already ran
        the quality filter (e.g. the scanner) and wants to propagate results
        without re-evaluating.

    Opportunities that fail quality are still written to the DB (with
    ``quality_passed=False``) so the orchestrator can skip them using the
    stored result rather than re-evaluating.

    Returns the number of signals upserted.
    """
    now = utcnow()
    signal_type = str(signal_type_override or "").strip().lower()
    if not signal_type:
        signal_type = f"{source}_opportunity"
    if opportunities and refresh_prices:
        try:
            from services.scanner import scanner as market_scanner

            opportunities = await market_scanner.refresh_opportunity_prices(
                opportunities,
                now=now,
                drop_stale=False,
            )
        except Exception as exc:
            logger.warning(
                "Failed to refresh opportunity prices before bridging signals",
                source=str(source),
                opportunities=len(opportunities),
                exc_info=exc,
            )
    emitted = 0
    keep_dedupe_keys: set[str] = set()
    signal_ids: list[str] = []
    market_data_ages_ms: list[int] = []

    for opp in opportunities:
        market_id, direction, entry_price, market_question, payload_json, strategy_context_json = (
            build_signal_contract_from_opportunity(opp)
        )
        if not market_id:
            continue

        dedupe_key = make_dedupe_key(
            opp.stable_id,
            opp.strategy,
            market_id,
        )
        keep_dedupe_keys.add(dedupe_key)
        expires = opp.resolution_date or (now + timedelta(minutes=default_ttl_minutes))
        ingested_at = now.astimezone(timezone.utc)
        ingested_at_iso = _to_iso_utc(ingested_at)

        payload = dict(payload_json or {})
        strategy_context = dict(strategy_context_json or {}) if isinstance(strategy_context_json, dict) else {}

        source_observed_at = _parse_datetime_utc(
            payload.get("source_observed_at")
            or payload.get("live_market_fetched_at")
            or payload.get("signal_updated_at")
            or payload.get("signal_emitted_at")
            or payload.get("last_priced_at")
            or payload.get("last_detected_at")
            or payload.get("first_detected_at")
            or payload.get("detected_at")
            or strategy_context.get("source_observed_at")
            or strategy_context.get("live_market_fetched_at")
            or strategy_context.get("fetched_at")
            or strategy_context.get("last_priced_at")
            or strategy_context.get("last_detected_at")
        )
        source_observed_iso = _to_iso_utc(source_observed_at)
        market_data_age_ms: int | None = None
        if source_observed_at is not None:
            market_data_age_ms = max(
                0,
                int((ingested_at - source_observed_at.astimezone(timezone.utc)).total_seconds() * 1000),
            )
            market_data_ages_ms.append(market_data_age_ms)

        payload["ingested_at"] = ingested_at_iso
        payload["signal_emitted_at"] = payload.get("signal_emitted_at") or ingested_at_iso
        payload["source_observed_at"] = source_observed_iso
        payload["market_data_age_ms"] = market_data_age_ms
        payload["bridge_source"] = str(source)
        payload["bridge_run_at"] = ingested_at_iso

        strategy_context["ingested_at"] = ingested_at_iso
        strategy_context["source_observed_at"] = source_observed_iso
        strategy_context["market_data_age_ms"] = market_data_age_ms
        strategy_context["bridge_source"] = str(source)
        strategy_context["bridge_run_at"] = ingested_at_iso

        opp_quality_passed: Optional[bool] = None
        opp_rejection_reasons: Optional[list] = None
        if quality_filter_pipeline is not None:
            report = quality_filter_pipeline.evaluate(opp)
            opp_quality_passed = bool(report.passed)
            opp_rejection_reasons = list(report.rejection_reasons) if not report.passed else None
        elif quality_reports is not None:
            opp_key = opp.stable_id or opp.id
            report = quality_reports.get(opp_key)
            if report is not None:
                opp_quality_passed = bool(report.passed)
                opp_rejection_reasons = list(report.rejection_reasons) if not report.passed else None
            else:
                opp_quality_passed = True
                opp_rejection_reasons = None

        signal_row = await upsert_trade_signal(
            session,
            source=source,
            source_item_id=opp.stable_id,
            signal_type=signal_type,
            strategy_type=opp.strategy,
            market_id=market_id,
            market_question=market_question,
            direction=direction,
            entry_price=entry_price,
            edge_percent=float(opp.roi_percent or 0.0),
            confidence=float(opp.confidence),
            liquidity=float(opp.min_liquidity or 0.0),
            expires_at=expires,
            payload_json=payload,
            strategy_context_json=strategy_context or None,
            quality_passed=opp_quality_passed,
            quality_rejection_reasons=opp_rejection_reasons,
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1
        signal_ids.append(str(signal_row.id))

    if sweep_missing:
        await expire_source_signals_except(
            session,
            source=source,
            keep_dedupe_keys=keep_dedupe_keys,
            signal_types=[signal_type],
            commit=False,
        )
    await _commit_with_retry(session)
    _schedule_trade_signal_snapshot_refresh()
    if signal_ids:
        emitted_at_iso = now.isoformat()
        try:
            await event_bus.publish(
                "trade_signal_batch",
                {
                    "event_type": "upsert_insert",
                    "source": str(source),
                    "signal_count": int(len(signal_ids)),
                    "signal_ids": signal_ids[:500],
                    "emitted_at": emitted_at_iso,
                    "trigger": "strategy_signal_bridge",
                    "market_data_age_ms_avg": (
                        round(sum(market_data_ages_ms) / len(market_data_ages_ms), 2) if market_data_ages_ms else None
                    ),
                    "market_data_age_ms_max": max(market_data_ages_ms) if market_data_ages_ms else None,
                },
            )
        except Exception as exc:
            logger.warning(
                "Failed to publish trade_signal_batch event",
                source=str(source),
                signal_count=len(signal_ids),
                exc_info=exc,
            )
        try:
            await publish_trade_signal_stream_batch(
                event_type="upsert_insert",
                source=str(source),
                signal_ids=signal_ids,
                trigger="strategy_signal_bridge",
                emitted_at=emitted_at_iso,
                metadata={
                    "market_data_age_ms_avg": (
                        round(sum(market_data_ages_ms) / len(market_data_ages_ms), 2) if market_data_ages_ms else None
                    ),
                    "market_data_age_ms_max": max(market_data_ages_ms) if market_data_ages_ms else None,
                },
            )
        except Exception as exc:
            logger.warning(
                "Failed to append trade_signal_batch stream event",
                source=str(source),
                signal_count=len(signal_ids),
                exc_info=exc,
            )
    return emitted
