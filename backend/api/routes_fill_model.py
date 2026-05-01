"""HTTP API for the shadow-trading fill simulator.

Powers the ``Strategies → ML Models → Fill Model`` UI surface and the
ensemble / triangulation panels in ``Strategies → Research → Backtest
Suite``.

Endpoints:
    GET    /api/fill-model/active                 — active Cox/KM
    GET    /api/fill-model/history                — recent trained
    POST   /api/fill-model/retrain                — trigger a fit
    POST   /api/fill-model/promote/{model_id}     — manual promote
    GET    /api/fill-model/empirical-constants    — measured + overrides
    PUT    /api/fill-model/empirical-constants    — set overrides
    GET    /api/fill-model/latency                — measured latency dist
    GET    /api/fill-model/triangulation/{slug}   — bt vs shadow vs live
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select

from models.database import (
    AsyncSessionLocal,
    BookDeltaEvent,
    FillProbabilityModel,
    TraderOrder,
)


logger = logging.getLogger("routes.fill_model")
router = APIRouter(prefix="/fill-model", tags=["Fill Model"])


# ---------------------------------------------------------------------
# Active model + history
# ---------------------------------------------------------------------


def _serialize_model_row(row: FillProbabilityModel) -> dict[str, Any]:
    return {
        "id": row.id,
        "family": row.family,
        "strata_key": row.strata_key,
        "trained_at": row.trained_at.replace(tzinfo=timezone.utc).isoformat() if row.trained_at else None,
        "training_window_start": (
            row.training_window_start.replace(tzinfo=timezone.utc).isoformat()
            if row.training_window_start
            else None
        ),
        "training_window_end": (
            row.training_window_end.replace(tzinfo=timezone.utc).isoformat()
            if row.training_window_end
            else None
        ),
        "n_events": int(row.n_events or 0),
        "n_observations": int(row.n_observations or 0),
        "concordance_index": row.concordance_index,
        "brier_score": row.brier_score,
        "log_likelihood": row.log_likelihood,
        "coefficients": dict(row.coefficients_json or {}),
        "baseline_survival": dict(row.baseline_survival_json or {}),
        "feature_means": dict(row.feature_means_json or {}),
        "feature_stds": dict(row.feature_stds_json or {}),
        "config": dict(row.config_json or {}),
        "promoted_at": row.promoted_at.replace(tzinfo=timezone.utc).isoformat() if row.promoted_at else None,
        "active": bool(row.active),
        "notes": str(row.notes or ""),
    }


@router.get("/active")
async def get_active_fill_model(strata_key: str = Query(default="pooled")):
    """Return the active model row for the given strata, or 404."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(FillProbabilityModel)
            .where(
                FillProbabilityModel.strata_key == strata_key,
                FillProbabilityModel.active.is_(True),
            )
            .order_by(desc(FillProbabilityModel.trained_at))
        )
        rows = list(result.scalars().all())
    if not rows:
        raise HTTPException(status_code=404, detail=f"No active fill model for strata '{strata_key}'")
    rows.sort(key=lambda r: (0 if r.family == "cox_ph" else 1, -(r.n_events or 0)))
    return _serialize_model_row(rows[0])


@router.get("/history")
async def get_fill_model_history(
    strata_key: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
):
    async with AsyncSessionLocal() as session:
        stmt = select(FillProbabilityModel).order_by(desc(FillProbabilityModel.trained_at)).limit(limit)
        if strata_key is not None:
            stmt = stmt.where(FillProbabilityModel.strata_key == strata_key)
        rows = list((await session.execute(stmt)).scalars().all())
    return {"models": [_serialize_model_row(r) for r in rows]}


@router.post("/retrain")
async def trigger_retrain(window_days: int = Query(default=30, ge=1, le=365)):
    """Run one Cox training cycle synchronously.  Returns the result."""
    from services.fill_simulator.cox_trainer import train_and_persist

    async with AsyncSessionLocal() as session:
        try:
            results = await train_and_persist(session, window_days=window_days)
        except Exception as exc:
            logger.exception("Manual retrain failed")
            raise HTTPException(status_code=500, detail=f"retrain failed: {exc}") from exc
    return {
        "trained": [
            {
                "family": r.family,
                "strata_key": r.strata_key,
                "n_events": r.n_events,
                "n_observations": r.n_observations,
                "concordance_index": r.concordance_index,
                "notes": r.notes,
            }
            for r in results
        ],
        "window_days": window_days,
    }


@router.post("/promote/{model_id}")
async def promote_model(model_id: str = Path(..., min_length=1)):
    async with AsyncSessionLocal() as session:
        target = await session.get(FillProbabilityModel, model_id)
        if target is None:
            raise HTTPException(status_code=404, detail="Model not found")
        # Demote any currently-active model in the same strata + family.
        result = await session.execute(
            select(FillProbabilityModel).where(
                FillProbabilityModel.strata_key == target.strata_key,
                FillProbabilityModel.family == target.family,
                FillProbabilityModel.active.is_(True),
            )
        )
        for row in result.scalars().all():
            row.active = False
        target.active = True
        target.promoted_at = datetime.now(timezone.utc)
        await session.commit()
    # Bump the inference cache so the new model takes effect immediately.
    from services.fill_simulator.cox_inference import invalidate_cache

    invalidate_cache()
    return {"promoted": True, "model_id": model_id}


# ---------------------------------------------------------------------
# Empirical constants (operator overrides)
# ---------------------------------------------------------------------


class EmpiricalOverridesIn(BaseModel):
    displayed_depth_factor: float | None = Field(default=None, ge=0.05, le=1.0)
    maker_queue_ahead_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    maker_trade_flow_multiplier: float | None = Field(default=None, ge=0.1, le=5.0)
    adverse_selection_multiplier: float | None = Field(default=None, ge=0.05, le=2.0)
    stale_depth_decay: float | None = Field(default=None, ge=0.05, le=1.0)
    min_depth_factor: float | None = Field(default=None, ge=0.01, le=1.0)


@router.get("/empirical-constants")
async def get_empirical_constants_route():
    from services.fill_simulator.empirical_constants import (
        get_empirical_constants,
        get_overrides,
    )

    constants = get_empirical_constants()
    return {
        "measured": constants.measured,
        "sample_count": constants.sample_count,
        "measured_at_epoch": constants.measured_at_epoch,
        "notes": constants.notes,
        "values": {
            "displayed_depth_factor": constants.displayed_depth_factor,
            "maker_queue_ahead_fraction": constants.maker_queue_ahead_fraction,
            "maker_trade_flow_multiplier": constants.maker_trade_flow_multiplier,
            "adverse_selection_multiplier": constants.adverse_selection_multiplier,
            "stale_depth_decay": constants.stale_depth_decay,
            "min_depth_factor": constants.min_depth_factor,
        },
        "overrides": get_overrides(),
    }


@router.put("/empirical-constants")
async def put_empirical_constants_route(body: EmpiricalOverridesIn):
    from services.fill_simulator.empirical_constants import (
        get_empirical_constants,
        get_overrides,
        set_override,
    )

    incoming = body.model_dump(exclude_unset=True)
    for key, value in incoming.items():
        try:
            set_override(key, value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    constants = get_empirical_constants()
    return {
        "applied": True,
        "overrides": get_overrides(),
        "values": {
            "displayed_depth_factor": constants.displayed_depth_factor,
            "maker_queue_ahead_fraction": constants.maker_queue_ahead_fraction,
            "maker_trade_flow_multiplier": constants.maker_trade_flow_multiplier,
            "adverse_selection_multiplier": constants.adverse_selection_multiplier,
            "stale_depth_decay": constants.stale_depth_decay,
            "min_depth_factor": constants.min_depth_factor,
        },
    }


# ---------------------------------------------------------------------
# Latency distribution (read)
# ---------------------------------------------------------------------


@router.get("/latency")
async def get_latency_route():
    from services.fill_simulator.latency import measured_latency_async

    dist = await measured_latency_async()
    # Surface the active fallback values so the UI can render them
    # alongside the (possibly fallback) p50/p95/p99 it just received.
    from services.fill_simulator.latency import _current_fallbacks

    fp50, fp95, fp99 = _current_fallbacks()
    return {
        "p50_ms": dist.p50_ms,
        "p95_ms": dist.p95_ms,
        "p99_ms": dist.p99_ms,
        "sample_count": dist.sample_count,
        "pessimistic_ms": dist.pessimistic_ms,
        "realistic_ms": dist.realistic_ms,
        "optimistic_ms": dist.optimistic_ms,
        "fallback_p50_ms": fp50,
        "fallback_p95_ms": fp95,
        "fallback_p99_ms": fp99,
    }


@router.put("/latency/fallbacks")
async def update_latency_fallbacks(
    p50_ms: float | None = None,
    p95_ms: float | None = None,
    p99_ms: float | None = None,
):
    """Operator-tunable latency fallback envelope.

    Pass any subset; null fields preserve the existing value.  Pass 0
    to reset that quantile to the module default (200/600/1500 ms).
    These values are used by the fill simulator + BacktestStudio
    "Latency (defaults)" panel whenever no real submit/cancel
    latencies have been measured in the last 15 minutes.
    """
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, AppSettings
    from services.fill_simulator.latency import refresh_fallback_overrides

    def _coerce(v):
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f <= 0:
            return None  # reset to default
        return min(60_000.0, f)  # 60s hard cap — anything bigger is a typo

    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
        if row is None:
            row = AppSettings(id="default")
            session.add(row)
        if p50_ms is not None:
            row.latency_fallback_p50_ms = _coerce(p50_ms)
        if p95_ms is not None:
            row.latency_fallback_p95_ms = _coerce(p95_ms)
        if p99_ms is not None:
            row.latency_fallback_p99_ms = _coerce(p99_ms)
        await session.commit()
    await refresh_fallback_overrides()
    from services.fill_simulator.latency import _current_fallbacks

    fp50, fp95, fp99 = _current_fallbacks()
    return {"p50_ms": fp50, "p95_ms": fp95, "p99_ms": fp99}


# ---------------------------------------------------------------------
# Trade-vs-cancel decomposition feed (UI panel)
# ---------------------------------------------------------------------


@router.get("/decomposition-summary")
async def get_decomposition_summary(hours: int = Query(default=24, ge=1, le=168)):
    """Recent trade-vs-cancel decomposition stats per token."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                BookDeltaEvent.event_type,
                func.count(BookDeltaEvent.id).label("count"),
                func.coalesce(func.sum(BookDeltaEvent.trade_size), 0.0).label("trade_total"),
                func.coalesce(func.sum(BookDeltaEvent.cancel_size), 0.0).label("cancel_total"),
            )
            .where(BookDeltaEvent.observed_at >= cutoff)
            .group_by(BookDeltaEvent.event_type)
        )
        rows = result.all()
    by_type = {row.event_type: row for row in rows}
    trade_count = int(by_type["trade"].count) if "trade" in by_type else 0
    cancel_count = int(by_type["cancel"].count) if "cancel" in by_type else 0
    trade_size = float(by_type["trade"].trade_total) if "trade" in by_type else 0.0
    cancel_size = float(by_type["cancel"].cancel_total) if "cancel" in by_type else 0.0
    total_count = trade_count + cancel_count
    total_size = trade_size + cancel_size
    return {
        "window_hours": hours,
        "trade_count": trade_count,
        "cancel_count": cancel_count,
        "trade_size": trade_size,
        "cancel_size": cancel_size,
        "trade_count_pct": (trade_count / total_count * 100.0) if total_count > 0 else None,
        "trade_size_pct": (trade_size / total_size * 100.0) if total_size > 0 else None,
    }


# ---------------------------------------------------------------------
# Triangulation: backtest vs shadow vs live PnL per strategy
# ---------------------------------------------------------------------


class CounterfactualRequest(BaseModel):
    token_id: str
    side: str = Field(pattern="^(buy|sell)$")
    price: float = Field(gt=0.0, lt=1.0)
    size_shares: float = Field(gt=0.0)
    placed_at: datetime
    time_in_force_seconds: float = Field(default=60.0, gt=0.0, le=3600.0)


@router.post("/counterfactual")
async def run_counterfactual(req: CounterfactualRequest):
    """Replay a hypothetical order against historical book + tape.

    Asks: if I had placed (token, side, price, size) at time T with TIF
    seconds, would it have filled?  Returns realized fills, time to
    fill, and queue behavior.  Used by the UI's "what if" panel and
    by the Cox trainer when bootstrapping synthetic labels.
    """
    from services.fill_simulator import (
        CounterfactualOrder,
        replay_counterfactual_order,
    )

    placed_at = req.placed_at
    if placed_at.tzinfo is None:
        placed_at = placed_at.replace(tzinfo=timezone.utc)
    result = await replay_counterfactual_order(
        CounterfactualOrder(
            token_id=req.token_id,
            side=req.side,
            price=req.price,
            size_shares=req.size_shares,
            placed_at=placed_at,
            time_in_force_seconds=req.time_in_force_seconds,
        )
    )
    return result.to_dict()


@router.get("/triangulation/{strategy_slug}")
async def get_triangulation(
    strategy_slug: str = Path(..., min_length=1),
    days: int = Query(default=30, ge=1, le=180),
):
    """Compare PnL distributions for the same strategy across backtest,
    shadow, and live.  Big divergence between any two → fill model is
    suspect.

    Pulls TraderOrder rows by ``strategy_key`` for the given window,
    splits by mode, and aggregates realized PnL plus fill ratio.
    Backtest pnl is sourced from the strategy_backtester run history if
    a recent run exists for the slug.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                TraderOrder.mode,
                TraderOrder.status,
                TraderOrder.notional_usd,
                TraderOrder.payload_json,
                TraderOrder.created_at,
            )
            .where(TraderOrder.strategy_key == strategy_slug)
            .where(TraderOrder.created_at >= cutoff)
        )
        rows = result.all()

    # Bucket by mode; sum filled notional + realized pnl.
    summary: dict[str, dict[str, Any]] = {
        "shadow": {"orders": 0, "filled": 0, "filled_notional_usd": 0.0, "realized_pnl_usd": 0.0},
        "live": {"orders": 0, "filled": 0, "filled_notional_usd": 0.0, "realized_pnl_usd": 0.0},
    }
    for row in rows:
        mode = (str(row.mode or "").strip().lower()) or "shadow"
        if mode not in summary:
            continue
        bucket = summary[mode]
        bucket["orders"] += 1
        payload = row.payload_json or {}
        sim = (
            payload.get("shadow_simulation")
            or payload.get("paper_simulation")
            or payload.get("live_execution")
            or {}
        )
        if sim.get("filled") or row.status in {"executed", "closed_win", "closed_loss", "resolved", "resolved_win", "resolved_loss"}:
            bucket["filled"] += 1
        notional = sim.get("filled_notional_usd")
        if isinstance(notional, (int, float)):
            bucket["filled_notional_usd"] += float(notional)
        else:
            bucket["filled_notional_usd"] += float(row.notional_usd or 0.0) if row.status in {"executed", "closed_win", "resolved_win"} else 0.0
        # PnL: if payload has a "realized_pnl_usd" or strategy_close result, use it.
        for key in ("realized_pnl_usd", "pnl_usd"):
            v = payload.get(key)
            if isinstance(v, (int, float)):
                bucket["realized_pnl_usd"] += float(v)
                break

    return {
        "strategy_slug": strategy_slug,
        "window_days": days,
        "modes": summary,
    }
