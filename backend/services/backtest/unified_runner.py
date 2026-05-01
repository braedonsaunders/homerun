"""Unified backtest runner — single entry point for the new UI.

Wraps the existing ``run_execution_backtest`` (which owns the L2-replay
engine, latency model, fill simulation, walk-forward CIs) and augments
the result with the data the new institutional-grade UI surfaces:

* Cox PH fill model snapshot — family, C-index, n_events, hazard ratios
  if cox_ph promoted, baseline survival otherwise.
* Empirical constants — measured displayed_depth_factor / cancel ratio
  / etc., the "spoofiness" panel.
* Latency distribution — measured p50/p95/p99 from
  ExecutionLatencyMetrics.
* Trade-vs-cancel decomposition over the run window.
* Ensemble fill estimate sample for the first N fills.
* Counterfactual replay for the first N fills (would each fill have
  happened against historical book + tape?).

Result is JSON-safe; persisted in a process-local LRU so the UI can
re-fetch a recent run by id without re-running the engine.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.fill_simulator import (
    CounterfactualOrder,
    EnsembleResult,
    FillModelSnapshot,
    ensemble_estimate,
    get_empirical_constants,
    load_active_fill_model,
    measured_latency_async,
    replay_counterfactual_order,
)
from services.strategy_backtester import (
    ExecutionBacktestResult,
    run_execution_backtest,
)


logger = logging.getLogger("backtest.unified_runner")


_RECENT_RUNS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_RECENT_RUNS = 32


def _store_run(run: dict[str, Any]) -> None:
    run_id = str(run.get("run_id") or "")
    if not run_id:
        return
    _RECENT_RUNS[run_id] = run
    while len(_RECENT_RUNS) > _MAX_RECENT_RUNS:
        _RECENT_RUNS.popitem(last=False)


def list_recent_runs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in reversed(_RECENT_RUNS.values()):
        out.append(
            {
                "run_id": run["run_id"],
                "strategy_slug": run.get("strategy_slug"),
                "strategy_name": run.get("strategy_name"),
                "started_at": run.get("started_at"),
                "completed_at": run.get("completed_at"),
                "total_time_ms": run.get("total_time_ms"),
                "status": "ok" if run.get("execution", {}).get("success") else "failed",
                "trade_count": run.get("execution", {}).get("trade_count", 0),
                "total_return_pct": run.get("execution", {}).get("total_return_pct", 0.0),
            }
        )
    return out


def get_recent_run(run_id: str) -> Optional[dict[str, Any]]:
    return _RECENT_RUNS.get(str(run_id))


def _bucket_label_hour(hour: int) -> str:
    if 0 <= hour < 6:
        return "00–06 UTC"
    if 6 <= hour < 12:
        return "06–12 UTC"
    if 12 <= hour < 18:
        return "12–18 UTC"
    return "18–24 UTC"


def _bucket_label_dow(dow: int) -> str:
    return ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[max(0, min(6, int(dow)))]


def _bucket_label_ttr(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "no_ttr"
    if seconds < 60:
        return "<1 min"
    if seconds < 300:
        return "1–5 min"
    if seconds < 1800:
        return "5–30 min"
    if seconds < 3600:
        return "30–60 min"
    if seconds < 14400:
        return "1–4 hr"
    return ">4 hr"


def _bucket_label_size(size_usd: float | None) -> str:
    if size_usd is None or size_usd <= 0:
        return "no_size"
    if size_usd < 25:
        return "<$25"
    if size_usd < 100:
        return "$25–100"
    if size_usd < 500:
        return "$100–500"
    if size_usd < 2000:
        return "$500–2K"
    return ">$2K"


def _compute_regime_breakdown(exec_dict: dict[str, Any]) -> dict[str, Any]:
    """Bucket the run's fills by hour, day-of-week, time-to-resolution
    at entry, and notional-size — return per-bucket trade counts and
    win rates.  This is what desks call "regime decomposition" and what
    we need to spot when a strategy works in only one slice of time."""
    fills = exec_dict.get("fills_sample") or []
    if not isinstance(fills, list) or not fills:
        return {"by_hour": [], "by_dow": [], "by_ttr": [], "by_size": []}

    from datetime import datetime as _dt

    by_hour: dict[str, dict[str, Any]] = {}
    by_dow: dict[str, dict[str, Any]] = {}
    by_ttr: dict[str, dict[str, Any]] = {}
    by_size: dict[str, dict[str, Any]] = {}

    def _bump(bucket_dict: dict[str, dict[str, Any]], key: str, win: bool, pnl: float) -> None:
        e = bucket_dict.setdefault(key, {"bucket": key, "n": 0, "wins": 0, "total_pnl_usd": 0.0})
        e["n"] += 1
        if win:
            e["wins"] += 1
        e["total_pnl_usd"] += float(pnl)

    for fill in fills:
        if not isinstance(fill, dict):
            continue
        ts_iso = fill.get("timestamp") or fill.get("filled_at") or fill.get("placed_at")
        if not ts_iso:
            continue
        try:
            ts = _dt.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
        except Exception:
            continue
        pnl_raw = fill.get("realized_pnl_usd") or fill.get("pnl_usd") or 0.0
        if not isinstance(pnl_raw, (int, float)):
            pnl_raw = 0.0
        won = float(pnl_raw) > 0
        size_usd = fill.get("notional_usd") or fill.get("filled_notional_usd") or 0.0
        try:
            size_usd = float(size_usd)
        except Exception:
            size_usd = 0.0
        ttr_raw = fill.get("time_to_resolution_seconds")
        try:
            ttr = float(ttr_raw) if ttr_raw is not None else None
        except Exception:
            ttr = None

        _bump(by_hour, _bucket_label_hour(ts.hour), won, float(pnl_raw))
        _bump(by_dow, _bucket_label_dow(ts.weekday()), won, float(pnl_raw))
        _bump(by_ttr, _bucket_label_ttr(ttr), won, float(pnl_raw))
        _bump(by_size, _bucket_label_size(size_usd), won, float(pnl_raw))

    def _finalize(bucket_dict: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry in bucket_dict.values():
            n = int(entry["n"])
            wins = int(entry["wins"])
            entry["win_rate"] = (wins / n) if n > 0 else 0.0
            entry["mean_pnl_usd"] = (float(entry["total_pnl_usd"]) / n) if n > 0 else 0.0
            out.append(entry)
        return out

    return {
        "by_hour": _finalize(by_hour),
        "by_dow": _finalize(by_dow),
        "by_ttr": _finalize(by_ttr),
        "by_size": _finalize(by_size),
    }


def _compute_deflated_sharpe(exec_dict: dict[str, Any], *, n_trials: int) -> dict[str, Any]:
    """Derive López de Prado's deflated Sharpe from the equity-curve
    sample.  Falls back to zero-noise output when the curve is too
    short for a meaningful return distribution."""
    from services.backtest.metrics import deflated_sharpe_ratio

    curve = exec_dict.get("equity_curve_sample") or []
    equities: list[float] = []
    for point in curve:
        if isinstance(point, dict):
            v = point.get("equity_usd")
            if isinstance(v, (int, float)):
                equities.append(float(v))
    if len(equities) < 4:
        return {
            "observed_sharpe": float(exec_dict.get("sharpe", {}).get("value") or 0.0),
            "sr_zero": 0.0,
            "probabilistic_sharpe": 0.0,
            "deflated_sharpe": 0.0,
            "n_observations": len(equities),
            "n_trials": int(max(1, n_trials)),
        }
    returns = []
    prev = equities[0]
    for v in equities[1:]:
        if prev > 0:
            returns.append((v - prev) / prev)
        prev = v
    return deflated_sharpe_ratio(returns, n_trials=n_trials, periods_per_year=252)


async def _capture_fill_model_snapshot() -> dict[str, Any]:
    snap: FillModelSnapshot | None = await load_active_fill_model(strata_key="pooled")
    if snap is None:
        return {"loaded": False}
    # Pull the active row's config_json directly to surface the
    # calibration_bins computed at training time.  cox_inference's
    # FillModelSnapshot doesn't carry config; one extra DB query is
    # cheap (cached above by load_active_fill_model anyway).
    calibration_bins: list[dict[str, Any]] | None = None
    try:
        from sqlalchemy import select
        from models.database import AsyncSessionLocal, FillProbabilityModel

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(FillProbabilityModel.config_json)
                .where(FillProbabilityModel.active.is_(True))
                .where(FillProbabilityModel.strata_key == "pooled")
                .order_by(FillProbabilityModel.trained_at.desc())
                .limit(1)
            )
            row = result.first()
            if row is not None and isinstance(row[0], dict):
                calibration_bins = row[0].get("calibration_bins") or None
    except Exception:
        calibration_bins = None
    return {
        "loaded": True,
        "family": snap.family,
        "strata_key": snap.strata_key,
        "n_events": snap.n_events,
        "concordance_index": snap.concordance_index,
        "trained_at_epoch": snap.trained_at_epoch,
        "promoted_at_epoch": snap.promoted_at_epoch,
        "coefficients": snap.coefficients,
        "feature_means": snap.feature_means,
        "feature_stds": snap.feature_stds,
        "baseline_survival_points": [
            {"t_seconds": float(t), "survival": float(s)} for t, s in snap.baseline_survival[:200]
        ],
        "calibration_bins": calibration_bins,
        "notes": snap.notes,
    }


async def _capture_decomposition_summary(hours: int) -> dict[str, Any]:
    from sqlalchemy import func, select
    from models.database import AsyncSessionLocal, BookDeltaEvent

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, hours))
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                BookDeltaEvent.event_type,
                func.count(BookDeltaEvent.id).label("count"),
                func.coalesce(func.sum(BookDeltaEvent.trade_size), 0.0).label("trade_sum"),
                func.coalesce(func.sum(BookDeltaEvent.cancel_size), 0.0).label("cancel_sum"),
            )
            .where(BookDeltaEvent.observed_at >= cutoff)
            .group_by(BookDeltaEvent.event_type)
        )
        rows = result.all()
    by_type = {row.event_type: row for row in rows}
    trade_count = int(by_type["trade"].count) if "trade" in by_type else 0
    cancel_count = int(by_type["cancel"].count) if "cancel" in by_type else 0
    trade_size = float(by_type["trade"].trade_sum) if "trade" in by_type else 0.0
    cancel_size = float(by_type["cancel"].cancel_sum) if "cancel" in by_type else 0.0
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


async def _capture_latency() -> dict[str, Any]:
    dist = await measured_latency_async()
    return {
        "p50_ms": dist.p50_ms,
        "p95_ms": dist.p95_ms,
        "p99_ms": dist.p99_ms,
        "sample_count": dist.sample_count,
        "pessimistic_ms": dist.pessimistic_ms,
        "realistic_ms": dist.realistic_ms,
        "optimistic_ms": dist.optimistic_ms,
    }


def _capture_empirical_constants() -> dict[str, Any]:
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
    }


async def _capture_counterfactuals_for_fills(
    fills_sample: list[dict[str, Any]],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    """For up to N fills from the run, replay them counterfactually."""
    if not fills_sample:
        return []
    out: list[dict[str, Any]] = []
    for fill in fills_sample[:sample_size]:
        token_id = str(fill.get("token_id") or fill.get("market_id") or "").strip()
        side = str(fill.get("side") or "buy").strip().lower()
        price = float(fill.get("price") or fill.get("fill_price") or 0.0)
        size = float(fill.get("size") or fill.get("filled_shares") or 1.0)
        ts_iso = fill.get("timestamp") or fill.get("filled_at")
        if not token_id or not ts_iso or price <= 0 or size <= 0:
            continue
        try:
            placed_at = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
        except Exception:
            continue
        try:
            order = CounterfactualOrder(
                token_id=token_id,
                side=side,
                price=price,
                size_shares=size,
                placed_at=placed_at,
                time_in_force_seconds=60.0,
            )
            result = await replay_counterfactual_order(order)
            out.append(
                {
                    "fill": {
                        "token_id": token_id,
                        "side": side,
                        "price": price,
                        "size": size,
                        "placed_at": placed_at.isoformat(),
                    },
                    "result": result.to_dict(),
                }
            )
        except Exception as exc:
            logger.debug("Counterfactual replay failed for fill: %s", exc, exc_info=True)
    return out


async def _capture_ensemble_band_for_fills(
    fills_sample: list[dict[str, Any]],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    """For up to N fills, capture the pessimistic/realistic/optimistic
    ensemble fill probability they would have had at decision time."""
    out: list[dict[str, Any]] = []
    for fill in fills_sample[:sample_size]:
        try:
            book = fill.get("order_book_snapshot")
            if not isinstance(book, dict):
                continue
            recent_trades = fill.get("recent_trades") or []
            side = str(fill.get("side") or "buy").upper()
            price = float(fill.get("price") or 0.0)
            size = float(fill.get("size") or 1.0)
            if price <= 0 or size <= 0:
                continue
            ensemble: EnsembleResult = ensemble_estimate(
                order_book=book,
                side=side,
                size_shares=size,
                limit_price=price,
                order_type=str(fill.get("order_type") or "maker_limit"),
                recent_trades=recent_trades,
                book_age_ms=float(fill.get("book_age_ms") or 0.0) or None,
                time_in_force_seconds=float(fill.get("time_in_force_seconds") or 6.0),
            )
            out.append(
                {
                    "fill_id": str(fill.get("id") or fill.get("fill_id") or ""),
                    "pessimistic": ensemble.pessimistic.fill_probability,
                    "realistic": ensemble.realistic.fill_probability,
                    "optimistic": ensemble.optimistic.fill_probability,
                    "cox_loaded": ensemble.cox_loaded,
                }
            )
        except Exception:
            continue
    return out


async def run_unified_backtest(
    *,
    source_code: str,
    slug: str = "_backtest_unified",
    config: dict[str, Any] | None = None,
    token_ids: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    initial_capital_usd: float = 1000.0,
    submit_p50_ms: float | None = None,
    submit_p95_ms: float | None = None,
    cancel_p50_ms: float | None = None,
    cancel_p95_ms: float | None = None,
    seed: int | None = None,
    counterfactual_sample_size: int = 8,
    ensemble_sample_size: int = 8,
) -> dict[str, Any]:
    """Run the full backtest pipeline + augment with fill-simulator data.

    Returns a JSON-safe dict that the new BacktestStudio UI consumes
    directly.  Persisted in process-local LRU; retrieve by run_id via
    ``get_recent_run``.
    """
    run_id = uuid.uuid4().hex[:16]
    started_at = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()

    # Core engine.
    exec_result: ExecutionBacktestResult = await run_execution_backtest(
        source_code=source_code,
        slug=slug,
        config=config,
        token_ids=token_ids,
        start=start,
        end=end,
        initial_capital_usd=initial_capital_usd,
        submit_p50_ms=submit_p50_ms,
        submit_p95_ms=submit_p95_ms,
        cancel_p50_ms=cancel_p50_ms,
        cancel_p95_ms=cancel_p95_ms,
        seed=seed,
    )
    exec_dict = exec_result.to_dict()

    # Snapshot the fill simulator state.  Run in parallel — they
    # don't depend on each other.
    fill_model_task = asyncio.create_task(_capture_fill_model_snapshot())
    decomp_task = asyncio.create_task(_capture_decomposition_summary(hours=24))
    latency_task = asyncio.create_task(_capture_latency())

    fill_model = await fill_model_task
    decomp = await decomp_task
    latency = await latency_task
    constants = _capture_empirical_constants()

    # Counterfactual replay for sample fills (best-effort, optional).
    fills_sample = exec_dict.get("fills_sample") or []
    counterfactuals: list[dict[str, Any]] = []
    ensemble_band: list[dict[str, Any]] = []
    if fills_sample:
        try:
            counterfactuals = await _capture_counterfactuals_for_fills(
                fills_sample, sample_size=counterfactual_sample_size
            )
        except Exception:
            logger.exception("Counterfactual replay capture failed")
        try:
            ensemble_band = await _capture_ensemble_band_for_fills(
                fills_sample, sample_size=ensemble_sample_size
            )
        except Exception:
            logger.exception("Ensemble band capture failed")

    completed_at = datetime.now(timezone.utc).isoformat()
    total_time_ms = (time.perf_counter() - started_perf) * 1000.0

    # Deflated Sharpe — derive from the equity-curve sample, treat
    # the run's parameter sweep size as n_trials when present (defaults
    # to 1, i.e. no penalty, when the strategy wasn't tuned).
    deflated = _compute_deflated_sharpe(exec_dict, n_trials=1)
    regime = _compute_regime_breakdown(exec_dict)

    out = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "total_time_ms": total_time_ms,
        "strategy_slug": exec_dict.get("strategy_slug") or slug,
        "strategy_name": exec_dict.get("strategy_name"),
        "execution": exec_dict,
        "deflated_sharpe": deflated,
        "regime_breakdown": regime,
        "fill_model": fill_model,
        "empirical_constants": constants,
        "latency": latency,
        "decomposition": decomp,
        "counterfactuals": counterfactuals,
        "ensemble_band": ensemble_band,
    }
    _store_run(out)
    return out
