"""Bridge: convert Opportunity objects into TradeSignal DB rows.

Workers and the scanner emit Opportunity objects.  This module converts
those opportunities into normalized TradeSignal rows using a single upsert
pattern shared by all signal sources.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional, Any

from sqlalchemy.ext.asyncio import AsyncSession

from models.opportunity import Opportunity
from services.signal_bus import (
    build_signal_contract_from_opportunity,
    expire_source_signals_except,
    make_dedupe_key,
    refresh_trade_signal_snapshots,
    upsert_trade_signal,
)
from services.worker_state import _commit_with_retry
from utils.utcnow import utcnow


async def bridge_opportunities_to_signals(
    session: AsyncSession,
    opportunities: list[Opportunity],
    source: str,
    *,
    default_ttl_minutes: int = 120,
    quality_filter_pipeline: Optional[Any] = None,
    quality_reports: Optional[dict] = None,
    sweep_missing: bool = False,
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
    emitted = 0
    keep_dedupe_keys: set[str] = set()

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

        await upsert_trade_signal(
            session,
            source=source,
            source_item_id=opp.stable_id,
            signal_type=f"{source}_opportunity",
            strategy_type=opp.strategy,
            market_id=market_id,
            market_question=market_question,
            direction=direction,
            entry_price=entry_price,
            edge_percent=float(opp.roi_percent or 0.0),
            confidence=float(opp.confidence),
            liquidity=float(opp.min_liquidity or 0.0),
            expires_at=expires,
            payload_json=payload_json,
            strategy_context_json=strategy_context_json,
            quality_passed=opp_quality_passed,
            quality_rejection_reasons=opp_rejection_reasons,
            dedupe_key=dedupe_key,
            commit=False,
        )
        emitted += 1

    if sweep_missing:
        await expire_source_signals_except(
            session,
            source=source,
            keep_dedupe_keys=keep_dedupe_keys,
            signal_types=[f"{source}_opportunity"],
            commit=False,
        )
    await _commit_with_retry(session)
    await refresh_trade_signal_snapshots(session)
    return emitted
