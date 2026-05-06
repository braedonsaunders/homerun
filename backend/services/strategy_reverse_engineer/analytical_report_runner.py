"""End-to-end runner for the analytical (report-mode) wallet pipeline.

Pipeline:
  1. Pull wallet trades (Polymarket /trades, capped by upstream).
  2. Resolve markets (polybacktest catalog + live + polymarket fallback).
  3. Run deterministic ``wallet_analytics.analyze`` to produce every
     statistical table.
  4. Pick a spotlight market (highest-trade-count resolved market).
  5. Run ``report_writer.draft_sections`` to produce LLM narratives
     for every section, falling back to deterministic stubs on any
     LLM failure.
  6. Persist the result on the job row + return a structured payload
     the API + PDF renderer can consume.

This is the parallel of ``agent.run_reverse_engineer_agent`` but
optimized for the *report* deliverable rather than the *strategy seed*
deliverable.  Both modes coexist — the operator chooses via the job's
``report_mode`` field.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import select

from models.database import (
    AsyncSessionLocal,
    StrategyReverseEngineerJob,
)
from utils.utcnow import utcnow

logger = logging.getLogger(__name__)


async def run_analytical_report(job_id: str) -> dict[str, Any]:
    """Build the analytical wallet report for a job.

    Persists:
      * ``wallet_profile_json`` (analytics.to_dict() — all tables)
      * ``best_strategy_code`` reused as the LLM-narrative payload
        (sections + spotlight) so the API surface keeps a single
        "best result" slot regardless of report_mode.
      * ``total_input_tokens`` / ``total_output_tokens`` / ``total_cost_usd``
        aggregated from ``LLMUsageLog`` rows whose purpose matches and
        whose timestamp falls within the run's wall-clock window.
        Without this aggregation step, the UI shows "$0 spent · 0
        tokens" because the per-section drafters discard the
        ``LLMResponse.usage`` field.
      * One ``StrategyReverseEngineerIteration`` row tagged 'report'
        carrying the same payload for audit / re-render.
    """
    from datetime import datetime, timezone
    run_start = datetime.now(timezone.utc)

    job = await _load_job(job_id)
    if job is None:
        raise ValueError(f"job '{job_id}' not found")
    address = job.wallet_address

    await _update(job_id, status="profiling", activity="Fetching wallet trades")
    from services.polymarket import polymarket_client
    from services.strategy_reverse_engineer.wallet_profile import _normalize_trade

    raw = await polymarket_client.get_wallet_trades_paginated(
        address,
        max_trades=int(job.max_wallet_trades or 50_000),
        page_size=500,
    )
    trades = [t for t in (_normalize_trade(x) for x in (raw or [])) if t]
    if not trades:
        await _mark_done(
            job_id, status="failed", activity="Wallet has no trades on Polymarket"
        )
        return {"job_id": job_id, "status": "failed", "error": "no trades"}

    await _update(
        job_id,
        wallet_trade_count=len(trades),
        wallet_window_start=min(t["timestamp"] for t in trades),
        wallet_window_end=max(t["timestamp"] for t in trades),
        activity=f"Resolving {len(trades):,} trades' markets",
    )

    from services.strategy_reverse_engineer.market_resolution import (
        resolve_markets_for_trades,
    )

    resolutions = await resolve_markets_for_trades(trades)
    await _update(
        job_id,
        activity=f"{len(resolutions):,} markets resolved · running analytics",
    )

    from services.strategy_reverse_engineer.wallet_analytics import (
        analyze,
        render_spotlight_market,
    )

    analytics = analyze(address=address, trades=trades, resolutions=resolutions)
    spotlight = render_spotlight_market(analytics=analytics, trades=trades)

    await _update(
        job_id,
        wallet_profile_json=_safe_dict(analytics.to_dict()),
        activity="Drafting LLM section narratives",
    )

    from services.strategy_reverse_engineer.report_writer import draft_sections

    sections = await draft_sections(
        analytics=analytics,
        spotlight=spotlight,
        model=job.llm_model,
    )

    # Persist the LLM payload for re-render / API surface.  We reuse
    # ``best_strategy_code`` as the narrative slot so the existing
    # API + UI shapes keep working without a schema change.
    payload = {
        "mode": "report",
        "sections": sections.to_dict(),
        "spotlight": spotlight,
    }

    # Aggregate token + cost from ``LLMUsageLog`` rows generated during
    # this run.  Each call inside ``draft_sections`` writes a row with
    # ``purpose='strategy_reverse_engineer_report'``; we sum them by
    # the run's wall-clock window so we attribute exactly the calls
    # this job made (the worker also runs other jobs concurrently —
    # the time-window filter scopes us correctly).
    usage_totals = await _aggregate_llm_usage(
        purpose="strategy_reverse_engineer_report",
        since=run_start,
    )

    # Score handling for report mode: previously hardcoded to 1.0 which
    # surfaced as a meaningless "100%" composite in the UI.  For a
    # deterministic-analytics + LLM-narrative deliverable there's no
    # iterative optimization happening, so a "score" doesn't apply.
    # Set to None — the UI special-cases ``mode=='report'`` to show
    # "Sections drafted: X/Y" instead of a fake score gauge.
    await _update(
        job_id,
        best_strategy_code=_serialize(payload),
        best_strategy_class="AnalyticalReport",
        best_score=None,
        total_input_tokens=usage_totals["input_tokens"],
        total_output_tokens=usage_totals["output_tokens"],
        total_cost_usd=usage_totals["cost_usd"],
        current_iteration=1,  # one analytical pass — for parity with strategy-seed mode
    )
    await _mark_done(job_id, status="completed", activity="Analytical report complete")

    # Single iteration row for audit consistency with strategy-seed mode.
    await _persist_iteration(job_id, payload, usage_totals)

    return {
        "job_id": job_id,
        "status": "completed",
        "trade_count": len(trades),
        "resolved_markets": len(resolutions),
        "sections_written": sum(1 for v in sections.to_dict().values() if v),
        "headline_pl_usdc": analytics.headline.realized_pl_usdc,
        "headline_roi": analytics.headline.roi_on_deployed,
        "input_tokens": usage_totals["input_tokens"],
        "output_tokens": usage_totals["output_tokens"],
        "cost_usd": usage_totals["cost_usd"],
    }


async def _aggregate_llm_usage(
    *,
    purpose: str,
    since: Any,
) -> dict[str, Any]:
    """Sum tokens + cost from LLMUsageLog rows since ``since``.

    Filtered by ``purpose`` so we only attribute calls made by the
    report drafter (not other concurrent agents on the same worker).
    Defensive: any DB / schema error returns zeros rather than crashing
    the whole run — the LLM section text already landed, the metrics
    are decorative.
    """
    try:
        from sqlalchemy import func, select as sa_select
        from models.database import LLMUsageLog

        async with AsyncSessionLocal() as session:
            q = await session.execute(
                sa_select(
                    func.coalesce(func.sum(LLMUsageLog.input_tokens), 0),
                    func.coalesce(func.sum(LLMUsageLog.output_tokens), 0),
                    func.coalesce(func.sum(LLMUsageLog.cost_usd), 0.0),
                ).where(
                    LLMUsageLog.purpose == purpose,
                    LLMUsageLog.requested_at >= since,
                    LLMUsageLog.success.is_(True),
                )
            )
            row = q.first()
        if row is None:
            return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        return {
            "input_tokens": int(row[0] or 0),
            "output_tokens": int(row[1] or 0),
            "cost_usd": float(row[2] or 0.0),
        }
    except Exception as exc:
        logger.warning("LLM usage aggregation failed: %s", exc)
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


async def _load_job(job_id: str) -> Optional[StrategyReverseEngineerJob]:
    async with AsyncSessionLocal() as session:
        return (
            await session.execute(
                select(StrategyReverseEngineerJob).where(
                    StrategyReverseEngineerJob.id == job_id
                )
            )
        ).scalar_one_or_none()


async def _update(job_id: str, **fields: Any) -> None:
    async with AsyncSessionLocal() as session:
        row = await _load_job(job_id)
        if row is None:
            return
        for k, v in fields.items():
            setattr(row, k, v)
        # Use the same session for persistence
        merged = await session.merge(row)
        _ = merged
        await session.commit()


async def _mark_done(job_id: str, *, status: str, activity: str) -> None:
    await _update(
        job_id,
        status=status,
        progress=1.0,
        activity=activity,
        finished_at=utcnow(),
    )


async def _persist_iteration(
    job_id: str,
    payload: dict[str, Any],
    usage_totals: Optional[dict[str, Any]] = None,
) -> None:
    import time
    import uuid as _uuid
    from models.database import StrategyReverseEngineerIteration

    usage = usage_totals or {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    async with AsyncSessionLocal() as session:
        row = StrategyReverseEngineerIteration(
            id=f"reit-{int(time.time() * 1000)}-{_uuid.uuid4().hex[:8]}",
            job_id=job_id,
            iteration=1,
            status="completed",
            strategy_class="AnalyticalReport",
            notes="Analytical report mode (deterministic analytics + LLM narrative).",
            # No meaningful per-iteration score in report mode — this is
            # a deterministic-analytics + section-drafter pipeline, not
            # an iterative strategy optimizer.  Use None so the UI
            # doesn't show a misleading "100%" gauge.
            score=None,
            score_breakdown_json={"mode": "report", "score_applies": False},
            llm_critique=_serialize(payload),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=usage["cost_usd"],
            completed_at=utcnow(),
        )
        session.add(row)
        await session.commit()


def _safe_dict(d: Any) -> Any:
    """Convert datetimes / dataclasses inside the analytics dict to JSON-safe.

    Postgres' JSON column type rejects ``NaN`` and ``Infinity`` — both
    of which can sneak in via open-ended bucket ranges (e.g. the top
    dominance band's upper bound is ``+inf``).  We coerce those to
    ``None`` recursively so the column write succeeds.
    """
    import json
    import math

    def _scrub(value: Any) -> Any:
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, dict):
            return {k: _scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_scrub(v) for v in value]
        if isinstance(value, tuple):
            return [_scrub(v) for v in value]
        return value

    return json.loads(json.dumps(_scrub(d), default=str))


def _serialize(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, default=str)


__all__ = ["run_analytical_report"]
