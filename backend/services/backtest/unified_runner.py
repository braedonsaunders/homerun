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


# Process-local hot cache.  Real persistence lives in the
# ``backtest_runs`` table (Alembic 202604300005).  The cache fronts
# the table for the very-recent reads the run-history poll generates
# every 5 seconds in the Studio UI.
_RECENT_RUNS_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_HOT_CACHE_MAX = 32


def _cache_run(run: dict[str, Any]) -> None:
    run_id = str(run.get("run_id") or "")
    if not run_id:
        return
    _RECENT_RUNS_CACHE[run_id] = run
    while len(_RECENT_RUNS_CACHE) > _HOT_CACHE_MAX:
        _RECENT_RUNS_CACHE.popitem(last=False)


def _coverage_warning(
    *, requested: int, covered: int, fraction: float
) -> Optional[str]:
    """Build the prominent no-/low-coverage validation warning, or None when
    coverage is adequate (>= 50% of requested tokens have parquet data).

    A backtest that runs against a window with NO recorded book data produces
    zero fills *silently* — the operator sees "0 trades" and mistakes a data
    gap for a strategy result. This anchors the warning on the fraction of
    requested tokens that have ANY parquet coverage in the window (distinct
    from the existing snapshot-DENSITY fidelity rating), so the gap is loud.
    Pure string-building — no I/O, never raises.
    """
    if requested <= 0:
        return None
    if fraction <= 0.0:
        return (
            f"NO DATA: 0/{requested} requested tokens have parquet coverage "
            f"in the window — backtest will produce no fills "
            f"(record/import data for this window first)."
        )
    if fraction < 0.5:
        pct = fraction * 100.0
        return (
            f"LOW COVERAGE: {covered}/{requested} tokens ({pct:.0f}%) "
            f"have parquet coverage in the window — results may be "
            f"unrepresentative."
        )
    return None


async def _resolve_coverage_summary(
    *,
    token_ids: list[str] | None,
    start: Optional[datetime],
    end: Optional[datetime],
) -> tuple[dict[str, Any], Optional[str]]:
    """Resolve parquet coverage for ``token_ids`` over ``[start, end]`` and
    return ``(summary, warning)``.

    ``summary`` is the JSON-safe shape the UI banner consumes:
    ``{requested_tokens, covered_tokens, coverage_fraction, window_start,
    window_end}``. ``warning`` is the prominent no-/low-coverage string (or
    None). Best-effort: any failure (bad window, resolver error) yields an
    empty summary and no warning rather than crashing the run — a backtest
    must never abort because the *diagnostic* failed.
    """
    if not token_ids or start is None or end is None:
        return {}, None
    requested_ids = [str(t) for t in token_ids if t]
    requested_n = len(requested_ids)
    summary: dict[str, Any] = {
        "requested_tokens": requested_n,
        "covered_tokens": 0,
        "coverage_fraction": 0.0,
        "window_start": start.isoformat() if hasattr(start, "isoformat") else str(start),
        "window_end": end.isoformat() if hasattr(end, "isoformat") else str(end),
    }
    if requested_n == 0:
        return summary, None
    try:
        from services.marketdata.coverage import resolve_coverage

        cov_map = await resolve_coverage(token_ids=requested_ids, start=start, end=end)
        covered_n = len(cov_map.covered_tokens)
        fraction = cov_map.coverage_fraction
        summary["covered_tokens"] = covered_n
        summary["coverage_fraction"] = fraction
    except Exception:  # noqa: BLE001
        logger.debug("unified_runner: coverage summary resolution failed", exc_info=True)
        return summary, None
    warning = _coverage_warning(
        requested=requested_n, covered=covered_n, fraction=fraction
    )
    return summary, warning


async def _persist_run_to_db(run: dict[str, Any]) -> None:
    """Upsert the run row into ``backtest_runs``.  Best-effort —
    failures log but don't propagate (the cache still serves the run
    in this process even if persistence fails).

    Uses INSERT ... ON CONFLICT (id) DO UPDATE so the worker pre-queue
    path (which inserts a row at status='queued' BEFORE this function
    runs) and the unified-runner completion path don't fight over the
    primary key.  Plain INSERT used to throw
    ``duplicate key value violates unique constraint`` on every run
    that was queued via the worker, leaving the cache and DB out of
    sync until the worker's own UPDATE landed.
    """
    from datetime import datetime as _dt
    from models.database import BacktestAsyncSessionLocal, BacktestRun
    from sqlalchemy.dialects.postgresql import insert as _pg_insert

    def _parse_iso(value: Any) -> _dt | None:
        if not value:
            return None
        try:
            return _dt.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    exec_dict = run.get("execution") or {}
    sparkline = _build_sparkline_pct(exec_dict)
    run_id = str(run.get("run_id") or "")
    if not run_id:
        return
    row_values = {
        "id": run_id,
        "strategy_slug": run.get("strategy_slug"),
        "strategy_name": run.get("strategy_name"),
        "started_at": _parse_iso(run.get("started_at")) or datetime.now(timezone.utc),
        "completed_at": _parse_iso(run.get("completed_at")),
        "total_time_ms": float(run.get("total_time_ms") or 0.0),
        "status": "ok" if exec_dict.get("success") else "failed",
        "trade_count": int(exec_dict.get("trade_count") or 0),
        "total_return_pct": float(exec_dict.get("total_return_pct") or 0.0),
        "sparkline_pct_json": sparkline,
        "result_json": run,
    }
    # Skip the worker-managed cols (progress / claimed_at / payload_json
    # / cancel_requested / worker_id) — they're owned by the queue
    # lifecycle, not the runner.  The DO UPDATE only refreshes columns
    # the runner actually computes.
    update_cols = {k: v for k, v in row_values.items() if k != "id"}
    try:
        async with BacktestAsyncSessionLocal() as session:
            stmt = _pg_insert(BacktestRun).values(**row_values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_=update_cols,
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as exc:
        logger.warning("Failed to persist backtest run %s: %s", run_id, exc)


def _build_sparkline_pct(exec_dict: dict[str, Any]) -> list[float]:
    """16-point %-drift series from the run's equity curve."""
    curve = exec_dict.get("equity_curve_sample") or []
    try:
        equities = [
            float(p.get("equity_usd"))
            for p in curve
            if isinstance(p, dict) and isinstance(p.get("equity_usd"), (int, float))
        ]
        if len(equities) < 2:
            return []
        target_n = 16
        if len(equities) > target_n:
            step = max(1, len(equities) // target_n)
            sampled = equities[::step][:target_n]
        else:
            sampled = equities
        base = sampled[0] or 1.0
        return [(v - base) / base * 100.0 if base else 0.0 for v in sampled]
    except Exception:
        return []


def _store_run(run: dict[str, Any]) -> None:
    """Insert the run into both the hot cache + the DB.  DB write is
    fire-and-forget (scheduled on the event loop) so the API caller
    isn't blocked on the ~few-ms write."""
    _cache_run(run)
    try:
        asyncio.create_task(_persist_run_to_db(run))
    except RuntimeError:
        # No running loop (test contexts) — fall back to direct await.
        asyncio.get_event_loop().run_until_complete(_persist_run_to_db(run))


async def list_recent_runs(*, limit: int = 32) -> list[dict[str, Any]]:
    """Read the run-history list from the DB (newest first), with
    the hot cache as a write-through that catches the most-recent
    additions before they've been read back from the table."""
    from sqlalchemy import select
    from models.database import BacktestAsyncSessionLocal, BacktestRun

    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        async with BacktestAsyncSessionLocal() as session:
            result = await session.execute(
                select(BacktestRun)
                .order_by(BacktestRun.started_at.desc())
                .limit(int(max(1, limit)))
            )
            for row in result.scalars().all():
                seen_ids.add(str(row.id))
                out.append(
                    {
                        "run_id": str(row.id),
                        "strategy_slug": row.strategy_slug,
                        "strategy_name": row.strategy_name,
                        "started_at": (
                            row.started_at.replace(tzinfo=timezone.utc).isoformat()
                            if row.started_at and row.started_at.tzinfo is None
                            else (row.started_at.isoformat() if row.started_at else None)
                        ),
                        "completed_at": (
                            row.completed_at.replace(tzinfo=timezone.utc).isoformat()
                            if row.completed_at and row.completed_at.tzinfo is None
                            else (row.completed_at.isoformat() if row.completed_at else None)
                        ),
                        "total_time_ms": float(row.total_time_ms or 0.0),
                        "status": str(row.status or "ok"),
                        "trade_count": int(row.trade_count or 0),
                        "total_return_pct": float(row.total_return_pct or 0.0),
                        "sparkline_pct": list(row.sparkline_pct_json or []),
                    }
                )
    except Exception as exc:
        logger.debug("DB run-history read failed (using cache only): %s", exc)

    # Merge in any cache rows the DB didn't return yet (write race).
    for run in reversed(_RECENT_RUNS_CACHE.values()):
        run_id = str(run.get("run_id") or "")
        if run_id in seen_ids:
            continue
        seen_ids.add(run_id)
        out.append(
            {
                "run_id": run_id,
                "strategy_slug": run.get("strategy_slug"),
                "strategy_name": run.get("strategy_name"),
                "started_at": run.get("started_at"),
                "completed_at": run.get("completed_at"),
                "total_time_ms": run.get("total_time_ms"),
                "status": "ok" if run.get("execution", {}).get("success") else "failed",
                "trade_count": run.get("execution", {}).get("trade_count", 0),
                "total_return_pct": run.get("execution", {}).get("total_return_pct", 0.0),
                "sparkline_pct": _build_sparkline_pct(run.get("execution") or {}),
            }
        )
    out.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)
    return out[:limit]


async def get_recent_run(run_id: str) -> Optional[dict[str, Any]]:
    """Hot-cache → DB fallback for a single run by id."""
    cached = _RECENT_RUNS_CACHE.get(str(run_id))
    if cached is not None:
        return cached
    from sqlalchemy import select
    from models.database import BacktestAsyncSessionLocal, BacktestRun

    try:
        async with BacktestAsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(BacktestRun).where(BacktestRun.id == str(run_id))
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            blob = row.result_json
            return blob if isinstance(blob, dict) else None
    except Exception as exc:
        logger.debug("DB single-run read failed for %s: %s", run_id, exc)
        return None


def _compute_partial_fill_aggregates(exec_dict: dict[str, Any]) -> dict[str, Any]:
    """Group child Fills by parent order_id and report aggregate
    metrics: how many ticks each order took to fill, VWAP variance
    across children, instant-fill rate, average intra-order delay.

    The matching engine already tracks fill_index per parent order,
    but the existing fills_sample lists each child atomic.  This view
    answers the operator question "do my orders typically fill in
    one tick or get walked across the book?"
    """
    fills = exec_dict.get("fills_sample") or []
    if not isinstance(fills, list) or not fills:
        return {
            "n_orders": 0,
            "n_instant_fills": 0,
            "n_partial_fills": 0,
            "instant_fill_rate": 0.0,
            "mean_children_per_order": 0.0,
            "max_children_per_order": 0,
            "mean_intra_order_seconds": 0.0,
            "mean_vwap_dispersion_bps": 0.0,
            "child_count_distribution": [],
        }

    from datetime import datetime as _dt

    by_parent: dict[str, list[dict[str, Any]]] = {}
    for f in fills:
        if not isinstance(f, dict):
            continue
        oid = str(f.get("order_id") or "")
        if not oid:
            continue
        by_parent.setdefault(oid, []).append(f)

    n_orders = len(by_parent)
    if n_orders == 0:
        return {
            "n_orders": 0,
            "n_instant_fills": 0,
            "n_partial_fills": 0,
            "instant_fill_rate": 0.0,
            "mean_children_per_order": 0.0,
            "max_children_per_order": 0,
            "mean_intra_order_seconds": 0.0,
            "mean_vwap_dispersion_bps": 0.0,
            "child_count_distribution": [],
        }

    instant = 0
    partials = 0
    children_counts: list[int] = []
    intra_durations: list[float] = []
    vwap_dispersions_bps: list[float] = []
    for oid, children in by_parent.items():
        c = len(children)
        children_counts.append(c)
        if c == 1:
            instant += 1
            continue
        partials += 1
        # Time between first and last fill.
        try:
            ts = []
            for ch in children:
                t = ch.get("occurred_at")
                if t:
                    ts.append(_dt.fromisoformat(str(t).replace("Z", "+00:00")))
            if len(ts) >= 2:
                ts.sort()
                intra_durations.append((ts[-1] - ts[0]).total_seconds())
        except Exception:
            pass
        # VWAP dispersion: 1e4 * std(prices) / VWAP.
        try:
            sizes = [float(ch.get("size") or 0.0) for ch in children]
            prices = [float(ch.get("price") or 0.0) for ch in children]
            total_size = sum(sizes)
            if total_size > 0:
                vwap = sum(s * p for s, p in zip(sizes, prices)) / total_size
                if vwap > 0 and len(prices) >= 2:
                    mean_p = sum(prices) / len(prices)
                    variance = sum((p - mean_p) ** 2 for p in prices) / (len(prices) - 1)
                    std = variance ** 0.5
                    vwap_dispersions_bps.append(std / vwap * 10_000.0)
        except Exception:
            pass

    # Distribution histogram of child counts (top 6 distinct values).
    dist_map: dict[int, int] = {}
    for c in children_counts:
        dist_map[c] = dist_map.get(c, 0) + 1
    distribution = sorted(
        [{"children": k, "n_orders": v} for k, v in dist_map.items()],
        key=lambda x: x["children"],
    )[:8]

    return {
        "n_orders": n_orders,
        "n_instant_fills": instant,
        "n_partial_fills": partials,
        "instant_fill_rate": instant / n_orders if n_orders else 0.0,
        "mean_children_per_order": float(sum(children_counts) / len(children_counts)) if children_counts else 0.0,
        "max_children_per_order": int(max(children_counts)) if children_counts else 0,
        "mean_intra_order_seconds": float(sum(intra_durations) / len(intra_durations)) if intra_durations else 0.0,
        "mean_vwap_dispersion_bps": float(sum(vwap_dispersions_bps) / len(vwap_dispersions_bps)) if vwap_dispersions_bps else 0.0,
        "child_count_distribution": distribution,
    }


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
        ts_iso = (
            fill.get("occurred_at")
            or fill.get("timestamp")
            or fill.get("filled_at")
            or fill.get("placed_at")
        )
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
        from models.database import BacktestAsyncSessionLocal, FillProbabilityModel

        async with BacktestAsyncSessionLocal() as session:
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


async def _capture_decomposition_summary(
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    fallback_hours: int = 24,
) -> dict[str, Any]:
    # Book deltas live in the canonical parquet plane (not SQL); aggregate
    # trade-vs-cancel from parquet off the event loop.  Scope to the backtest
    # window when given — the panel is about THIS run, and a historical window
    # would otherwise scan irrelevant now-minus-24h data (slow AND wrong).
    import asyncio

    from services.marketdata.deltas import aggregate_delta_events

    if start is not None and end is not None:
        lo, hi = start, end
        window_hours = round((end - start).total_seconds() / 3600.0, 4)
    else:
        hi = datetime.now(timezone.utc)
        lo = hi - timedelta(hours=max(1, fallback_hours))
        window_hours = float(fallback_hours)
    agg = await asyncio.to_thread(aggregate_delta_events, start=lo, end=hi)
    trade_count = int(agg.get("n_trade", 0) or 0)
    cancel_count = int(agg.get("n_cancel", 0) or 0)
    trade_size = float(agg.get("trade_size_sum", 0.0) or 0.0)
    cancel_size = float(agg.get("cancel_size_sum", 0.0) or 0.0)
    total_count = trade_count + cancel_count
    total_size = trade_size + cancel_size
    return {
        "window_hours": window_hours,
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


async def _capture_outcome_netting(execution: dict[str, Any]) -> dict[str, Any]:
    """Outcome-aware netting + capital-lockup report.

    Polymarket markets are not all binary YES/NO — many are
    multi-outcome (3+ tokens whose prices sum to ~1.0).  When a
    portfolio holds positions across multiple sibling tokens of the
    same parent market, the worst-case loss is bounded by
    ``sum(cost_basis) - 1.0`` (the redemption guarantee), not the
    gross sum of cost bases.  This report computes both the gross
    and the netted view so the operator can see how much capital
    they could free with outcome-aware sizing — without changing
    the matching engine itself.

    Capital lockup: USDC parked in unresolved markets isn't
    redeployable.  We report the time-weighted average lockup
    duration across closed positions plus the still-locked capital
    of currently-open positions.
    """
    from datetime import datetime as _dt

    from services.backtest.outcome_resolver import get_outcome_resolver

    positions = list(execution.get("positions_summary") or [])
    if not positions:
        return {
            "gross_exposure_usd": 0.0,
            "net_exposure_usd": 0.0,
            "capital_efficiency_pct": None,
            "locked_capital_usd": 0.0,
            "open_positions": 0,
            "outcome_groups": {"full_coverage": 0, "partial": 0, "single": 0},
            "avg_lockup_seconds": None,
            "max_lockup_seconds": None,
            "by_outcome_count": {},
        }

    resolver = get_outcome_resolver()

    # Group positions by parent market.  Tokens not in the resolver
    # index fall under a synthetic "unknown_<token_id>" group of size 1
    # (treated as single-outcome, no netting available).
    groups: dict[str, dict[str, Any]] = {}
    open_positions = 0
    total_locked_usd = 0.0
    lockup_durations: list[float] = []
    now_ts = _dt.now(tz=None).timestamp()

    for pos in positions:
        token_id = str(pos.get("token_id") or "")
        cost_basis = float(pos.get("cost_basis_usd") or 0.0)
        is_open = bool(pos.get("is_open"))
        if is_open:
            open_positions += 1
            total_locked_usd += cost_basis

        # Lockup duration: closed positions use closed - opened; open
        # positions use now - opened (proxy — the simulator's "now"
        # is end_iso, but we use wall-clock as a fallback).
        opened_iso = pos.get("opened_at")
        closed_iso = pos.get("closed_at")
        if opened_iso:
            try:
                t0 = _dt.fromisoformat(str(opened_iso).replace("Z", "+00:00")).timestamp()
                t1 = (
                    _dt.fromisoformat(str(closed_iso).replace("Z", "+00:00")).timestamp()
                    if closed_iso
                    else now_ts
                )
                if t1 > t0:
                    lockup_durations.append(t1 - t0)
            except (ValueError, TypeError):
                pass

        record = await resolver.market_for_token(token_id)
        group_key = record.market_id if record is not None else f"unknown_{token_id}"
        if group_key not in groups:
            groups[group_key] = {
                "outcome_count": record.outcome_count if record else 1,
                "siblings_total": len(record.token_ids) if record else 1,
                "siblings_held": set(),
                "gross_cost_basis": 0.0,
                "is_neg_risk": bool(record.neg_risk) if record else False,
            }
        groups[group_key]["siblings_held"].add(token_id.lower())
        # Netting reasoning relies on long positions across all
        # complementary outcomes: only BUY-side positions cap collateral
        # need at the redemption value.  Conservative: only count BUY
        # cost basis toward the redemption-rebate calculation.
        if str(pos.get("side") or "").upper() == "BUY":
            groups[group_key]["gross_cost_basis"] += cost_basis

    # Per-group netting: when all siblings are held BUY, the worst-case
    # net cost is sum(cost_basis) - $1 per share (redemption guarantee
    # ≈ 1.0 minus per-leg gas).  We approximate the rebate as
    # max(0, gross - 1.0 * shares) — but since we don't have shares
    # here without weighting by sizes, we use the heuristic
    # ``gross - count_of_outcomes_covered`` to give a conservative
    # capital-efficiency view (every fully-covered group rebates 1
    # unit of risk).
    full_coverage = 0
    partial = 0
    single = 0
    gross_exposure = 0.0
    rebate_estimate_usd = 0.0
    by_outcome_count: dict[int, int] = {}
    for g in groups.values():
        gross_exposure += g["gross_cost_basis"]
        held = len(g["siblings_held"])
        total = g["siblings_total"]
        oc = g["outcome_count"]
        by_outcome_count[oc] = by_outcome_count.get(oc, 0) + 1
        if total == 1:
            single += 1
        elif held >= total and total >= 2:
            full_coverage += 1
            # Cap the rebate at the gross — full coverage frees up to
            # 1 unit per share.  Without per-share resolution here,
            # use 50% of gross as a conservative redemption proxy
            # (real number depends on entry prices summing close to 1).
            rebate_estimate_usd += g["gross_cost_basis"] * 0.5
        elif 0 < held < total:
            partial += 1
        else:
            single += 1

    net_exposure = max(0.0, gross_exposure - rebate_estimate_usd)
    capital_efficiency_pct = (
        (1.0 - net_exposure / gross_exposure) * 100.0 if gross_exposure > 0 else None
    )

    return {
        "gross_exposure_usd": round(gross_exposure, 2),
        "net_exposure_usd": round(net_exposure, 2),
        "rebate_estimate_usd": round(rebate_estimate_usd, 2),
        "capital_efficiency_pct": (
            round(capital_efficiency_pct, 2) if capital_efficiency_pct is not None else None
        ),
        "locked_capital_usd": round(total_locked_usd, 2),
        "open_positions": open_positions,
        "outcome_groups": {
            "full_coverage": full_coverage,
            "partial": partial,
            "single": single,
            "total": len(groups),
        },
        "by_outcome_count": {str(k): v for k, v in sorted(by_outcome_count.items())},
        "avg_lockup_seconds": (
            round(sum(lockup_durations) / len(lockup_durations), 1)
            if lockup_durations
            else None
        ),
        "max_lockup_seconds": (
            round(max(lockup_durations), 1) if lockup_durations else None
        ),
        "n_lockup_samples": len(lockup_durations),
    }


def _capture_trade_order_monte_carlo(execution: dict[str, Any]) -> dict[str, Any]:
    """In-process trade-order Monte Carlo.

    Pulls the realized trade pnls from the fills sample (or, when
    closed positions are available, from realized_pnl_usd per
    closed position) and runs ``trade_order_monte_carlo`` to test
    sequence sensitivity.  Cheap — no engine re-runs — so it always
    runs as part of the unified pipeline.
    """
    from services.backtest.monte_carlo import trade_order_monte_carlo

    pnls: list[float] = []
    for pos in execution.get("positions_summary") or []:
        try:
            v = float(pos.get("realized_pnl_usd") or 0.0)
            if v != 0.0:
                pnls.append(v)
        except (TypeError, ValueError):
            continue
    if len(pnls) < 4:
        return {
            "n_resamples": 0,
            "sharpe_distribution": {},
            "observed_vs_distribution": None,
            "n_trades": len(pnls),
            "skipped_reason": "fewer than 4 closed trades — Monte Carlo not meaningful",
        }
    out = trade_order_monte_carlo(trade_pnls_usd=pnls, n_resamples=2000, seed=42)
    out["n_trades"] = len(pnls)
    return out


def _capture_data_quality() -> dict[str, Any]:
    """Snapshot the unified market-data ingestor's data-quality counters.

    Surfaces accept-rate, sequence gaps, per-reason reject counts, and
    persistence-flush latency so the BacktestStudio UI can flag data
    corruption that would otherwise silently pollute the Cox PH
    training set.  Reads from the singleton — cheap, no I/O.
    """
    try:
        from services.market_data_ingestor import get_market_data_ingestor

        return get_market_data_ingestor().get_data_quality_stats()
    except Exception:
        return {"accepted_books": 0, "total_attempts": 0, "accept_rate": None}


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
    session_id: str | None = None,
    provider_dataset_ids: list[str] | None = None,
    initial_capital_usd: float = 1000.0,
    submit_p50_ms: float | None = None,
    submit_p95_ms: float | None = None,
    cancel_p50_ms: float | None = None,
    cancel_p95_ms: float | None = None,
    seed: int | None = None,
    counterfactual_sample_size: int = 8,
    ensemble_sample_size: int = 8,
    impact_strength_bps: float | None = None,
    impact_capacity_threshold: float | None = None,
    impact_capacity_exponent: float | None = None,
    maker_rebate_bps: float | None = None,
    maker_rebate_max_spread_bps: float | None = None,
    latency_correlation_window_ms: float | None = None,
    # Number of independent parameter trials this run was selected
    # from.  Drives the López de Prado Deflated Sharpe correction:
    #   n_trials=1   → no over-fitting penalty (sr_zero = 0)
    #   n_trials=50  → meaningful penalty (sr_zero ≈ Φ⁻¹(0.98)/sqrt(T))
    # Studio's "Run backtest" button passes 1 (single-shot, no
    # search).  Studio's "Iterate params" loop passes the total
    # iteration count so the BEST run's DSR reflects the search size.
    # Hardcoded =1 default keeps legacy callers working unchanged.
    n_trials: int = 1,
    # Cap on returned fills_sample.  Default 200 is fine for UI rendering;
    # reverse-engineer scoring needs full coverage.  Capped at 100k by the
    # route-level validator so a pathological run can't blow up the JSON.
    fills_sample_size: int | None = None,
    # Discovery-replay tick grid overrides (defaults inside
    # run_execution_backtest: 1800s interval / 96 ticks).  Float so callers can
    # request sub-second (tick-grade) resolution for microstructure strategies.
    discovery_sample_interval_seconds: float | None = None,
    discovery_max_ticks: int | None = None,
    # Async-job-queue path: when the dedicated backtest worker process
    # invokes this function, it passes a pre-allocated ``run_id`` (so
    # the operator's polling already has a stable pointer) and a
    # ``progress_callback`` the engine fires every ~1k snapshots.
    # Sync callers (legacy POST /backtest/run) leave both at None.
    run_id: str | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run the full backtest pipeline + augment with fill-simulator data.

    Returns a JSON-safe dict that the new BacktestStudio UI consumes
    directly.  Persisted in process-local LRU; retrieve by run_id via
    ``get_recent_run``.

    When called from the worker, ``progress_callback`` is wired through
    to ``BacktestEngine.run`` so the UI can render a live progress bar.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:16]
    started_at = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()

    # Resolve provider_dataset_ids → (token_ids, start, end) when the
    # caller passed datasets but no explicit token universe.  The sync
    # /backtest/run route already does this resolution before calling
    # us, but the worker-queue path (/runs/enqueue → enqueue_run →
    # worker → run_unified_backtest) does NOT — it just passes the
    # raw provider_dataset_ids through.  Without this block, runs
    # scoped to a dataset would silently fall back to the live-opp
    # universe and the dataset's whole point (scope down to specific
    # tokens for parquet replay) is lost.  Empirically caught when a
    # Telonex-imported BTC dataset's 2 tokens got ignored and the
    # backtest ran against 1156 live-cache tokens with LOW fidelity.
    if provider_dataset_ids and not token_ids:
        try:
            from services.external_data.provider_import_service import resolve_dataset_scope
            scope = await resolve_dataset_scope(list(provider_dataset_ids))
            if scope is not None and scope.get("token_ids"):
                token_ids = scope["token_ids"]
                # Dataset window wins when caller didn't pin one — for
                # parquet replay we MUST scope to the file's covered
                # window or the engine reads off the end.
                if start is None and scope.get("start") is not None:
                    start = scope["start"]
                if end is None and scope.get("end") is not None:
                    end = scope["end"]
                logger.info(
                    "unified_runner: resolved %d provider_dataset_ids → %d tokens, window %s → %s",
                    len(provider_dataset_ids), len(token_ids), start, end,
                )
        except Exception:
            logger.exception(
                "unified_runner: failed to resolve provider_dataset_ids=%s — "
                "run will fall through to live-opp scope",
                provider_dataset_ids,
            )

    # Core engine.
    exec_kwargs: dict[str, Any] = {
        "source_code": source_code,
        "slug": slug,
        "config": config,
        "token_ids": token_ids,
        "start": start,
        "end": end,
        "initial_capital_usd": initial_capital_usd,
    }
    if submit_p50_ms is not None:
        exec_kwargs["submit_latency_p50_ms"] = float(submit_p50_ms)
    if submit_p95_ms is not None:
        exec_kwargs["submit_latency_p95_ms"] = float(submit_p95_ms)
    if cancel_p50_ms is not None:
        exec_kwargs["cancel_latency_p50_ms"] = float(cancel_p50_ms)
    if cancel_p95_ms is not None:
        exec_kwargs["cancel_latency_p95_ms"] = float(cancel_p95_ms)
    if seed is not None:
        exec_kwargs["seed"] = int(seed)
    if impact_strength_bps is not None:
        exec_kwargs["impact_strength_bps"] = float(impact_strength_bps)
    if impact_capacity_threshold is not None:
        exec_kwargs["impact_capacity_threshold"] = float(impact_capacity_threshold)
    if impact_capacity_exponent is not None:
        exec_kwargs["impact_capacity_exponent"] = float(impact_capacity_exponent)
    if maker_rebate_bps is not None:
        exec_kwargs["maker_rebate_bps"] = float(maker_rebate_bps)
    if maker_rebate_max_spread_bps is not None:
        exec_kwargs["maker_rebate_max_spread_bps"] = float(maker_rebate_max_spread_bps)
    if latency_correlation_window_ms is not None:
        exec_kwargs["latency_correlation_window_ms"] = float(latency_correlation_window_ms)
    # Wire the worker's progress callback through to the engine.  The
    # unified runner itself doesn't fire intermediate progress (its
    # post-engine analytics are fast); the engine's snapshot-loop is
    # where the long wall-clock lives.
    if progress_callback is not None:
        exec_kwargs["progress_callback"] = progress_callback
    if fills_sample_size is not None:
        exec_kwargs["fills_sample_size"] = int(fills_sample_size)
    if discovery_sample_interval_seconds is not None:
        # float — sub-second resolution is the whole point of a tick-grade
        # backtest; an int() cast here silently floored 0.25s → 0.
        exec_kwargs["discovery_sample_interval_seconds"] = float(discovery_sample_interval_seconds)
    if discovery_max_ticks is not None:
        exec_kwargs["discovery_max_ticks"] = int(discovery_max_ticks)

    # A backtest replays a FROZEN, pinned parquet dataset — the data plane is
    # pure parquet (MarketDataView -> load_book_series), and the only DB touch
    # is resolving the catalog (which files cover which tokens).  Re-walking the
    # filesystem + re-UPSERTing that catalog every 60s mid-run is wasted work
    # that storms the connection pool on long sub-second runs.  Scan once up
    # front, then freeze the catalog for the duration of the run.
    import services.external_data.parquet_scanner as _ps
    try:
        await _ps.ensure_recent_scan()
    except Exception:  # noqa: BLE001
        logger.debug("pre-run parquet scan skipped", exc_info=True)
    # No-silent-failure guard: resolve parquet coverage for the run's token
    # universe over [start, end] BEFORE the engine runs (catalog is fresh
    # from the scan just above).  A window with zero recorded book data
    # produces "0 fills" silently — the operator can't tell a data gap from
    # a strategy result.  We surface the covered fraction explicitly and, on
    # no/low coverage, prepend a loud validation_warning below.  Best-effort:
    # never aborts the run.
    coverage_summary, coverage_warning = await _resolve_coverage_summary(
        token_ids=token_ids, start=start, end=end
    )
    _ps.suspend_scan()
    try:
        exec_result: ExecutionBacktestResult = await run_execution_backtest(**exec_kwargs)
    finally:
        _ps.resume_scan()
    exec_dict = exec_result.to_dict()

    # Merge the run-level coverage summary onto the result's data_coverage
    # (keep the engine's per-token fidelity stats; add requested/covered/
    # coverage_fraction/window over [start,end]) and PREPEND the no-/low-
    # coverage warning so it renders at the top of the operator's list.
    if coverage_summary:
        merged_cov = dict(exec_dict.get("data_coverage") or {})
        merged_cov.update(coverage_summary)
        exec_dict["data_coverage"] = merged_cov
    if coverage_warning:
        existing_warnings = list(exec_dict.get("validation_warnings") or [])
        exec_dict["validation_warnings"] = [coverage_warning, *existing_warnings]

    # Snapshot the fill simulator state.  Run in parallel — they
    # don't depend on each other.
    fill_model_task = asyncio.create_task(_capture_fill_model_snapshot())
    decomp_task = asyncio.create_task(_capture_decomposition_summary(start=start, end=end))
    latency_task = asyncio.create_task(_capture_latency())

    fill_model = await fill_model_task
    decomp = await decomp_task
    latency = await latency_task
    constants = _capture_empirical_constants()
    data_quality = _capture_data_quality()
    outcome_netting = await _capture_outcome_netting(exec_dict)
    trade_order_mc = _capture_trade_order_monte_carlo(exec_dict)

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
    # to 1, i.e. no penalty, when the strategy wasn't tuned).  The
    # iteration loop in the studio passes the iteration count so the
    # over-fitting deflation correction actually kicks in.
    deflated = _compute_deflated_sharpe(exec_dict, n_trials=max(1, int(n_trials)))
    regime = _compute_regime_breakdown(exec_dict)
    partial_fills = _compute_partial_fill_aggregates(exec_dict)

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
        "partial_fills": partial_fills,
        "fill_model": fill_model,
        "empirical_constants": constants,
        "latency": latency,
        "decomposition": decomp,
        "counterfactuals": counterfactuals,
        "ensemble_band": ensemble_band,
        "data_quality": data_quality,
        "outcome_netting": outcome_netting,
        "trade_order_monte_carlo": trade_order_mc,
        # Hoist data_coverage to top level so the UI can render a
        # prominent banner without digging into ``execution.*``.  The
        # operator needs to see fidelity ratings BEFORE interpreting
        # trade counts — "0 trades" with low fidelity is a data
        # problem, not a strategy problem.
        "data_coverage": exec_dict.get("data_coverage", {}),
    }
    _store_run(out)
    return out
