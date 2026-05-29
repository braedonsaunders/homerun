"""
Strategy Backtester

Provides code-level backtesting for all three strategy phases:
  - DETECT: What opportunities would this code find on current and replayed snapshots?
  - EVALUATE: Given recent trade signals, which would this strategy accept/reject?
  - EXIT: Given current open positions, which would this strategy close?
"""

from __future__ import annotations

import asyncio
import itertools
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from models.database import AsyncSessionLocal
from services.strategy_loader import StrategyLoader, validate_strategy_source
from services.scanner import scanner
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_REPLAY_LOOKBACK_HOURS = 24
_DEFAULT_REPLAY_TIMEFRAME = "30m"
_DEFAULT_REPLAY_MAX_MARKETS = 80
_DEFAULT_REPLAY_MAX_STEPS = 72


@dataclass
class BacktestResult:
    """Result of running a strategy backtest against current market data."""

    success: bool = False
    # Strategy info
    strategy_slug: str = ""
    strategy_name: str = ""
    class_name: str = ""
    # Market data info
    num_events: int = 0
    num_markets: int = 0
    num_prices: int = 0
    data_source: str = ""  # "cache" or "fresh"
    replay_mode: str = "live_snapshot"
    replay_steps: int = 0
    replay_markets: int = 0
    replay_window_hours: int = 0
    replay_timeframe: str = ""
    # Results
    opportunities: list[dict[str, Any]] = field(default_factory=list)
    num_opportunities: int = 0
    quality_reports: list[dict[str, Any]] = field(default_factory=list)
    # Timing
    load_time_ms: float = 0
    data_fetch_time_ms: float = 0
    detect_time_ms: float = 0
    total_time_ms: float = 0
    # Errors
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    runtime_error: Optional[str] = None
    runtime_traceback: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayDetectRun:
    opportunities: list[Any] = field(default_factory=list)
    steps_run: int = 0
    markets_replayed: int = 0
    step_errors: int = 0


def _has_custom_detect_async(strategy) -> bool:
    """Check if strategy implements its own detect_async (not just inherited)."""
    method = getattr(type(strategy), "detect_async", None)
    if method is None:
        return False
    from services.strategies.base import BaseStrategy

    base_method = getattr(BaseStrategy, "detect_async", None)
    return method is not base_method


def _has_custom_detect_sync(strategy) -> bool:
    """Check if strategy implements its own detect_sync (not just inherited)."""
    method = getattr(type(strategy), "detect_sync", None)
    if method is None:
        return False
    from services.strategies.base import BaseStrategy

    base_method = getattr(BaseStrategy, "detect_sync", None)
    return method is not base_method


def _has_custom_detect_plain(strategy) -> bool:
    """Check if strategy implements its own ``detect()`` (the
    backwards-compatible name).  Most strategies in this codebase
    override the plain method rather than ``detect_sync``/``detect_async``,
    and ``_run_detect_once`` already knows how to route to it.  Replay
    discovery must recognise this override too — historically it didn't,
    which silently skipped 20+ scanner strategies (tail_end_carry,
    stat_arb, news_edge, every BTC/ETH variant, etc.).
    """
    method = getattr(type(strategy), "detect", None)
    if method is None:
        return False
    from services.strategies.base import BaseStrategy

    base_method = getattr(BaseStrategy, "detect", None)
    return method is not base_method


def _timeframe_to_seconds(value: str | int | None, *, default_seconds: int = 1800) -> int:
    if isinstance(value, int):
        return max(60, int(value))
    raw = str(value or "").strip().lower()
    if not raw:
        return default_seconds
    try:
        if raw.endswith("m"):
            return max(60, int(raw[:-1]) * 60)
        if raw.endswith("h"):
            return max(60, int(raw[:-1]) * 3600)
        if raw.endswith("d"):
            return max(60, int(raw[:-1]) * 86400)
        return max(60, int(raw))
    except Exception:
        return default_seconds


def _clamp_probability(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed < 0.0 or parsed > 1.01:
        return None
    return max(0.0, min(1.0, parsed))


def _bucket_ms(ts_ms: int, start_ms: int, step_ms: int) -> int:
    return start_ms + ((ts_ms - start_ms) // step_ms) * step_ms


def _serialize_opportunities(opportunities: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for opp in opportunities or []:
        try:
            if hasattr(opp, "model_dump"):
                out.append(opp.model_dump())
            elif hasattr(opp, "dict"):
                out.append(opp.dict())
            elif hasattr(opp, "__dict__"):
                out.append({k: v for k, v in opp.__dict__.items() if not k.startswith("_")})
            elif isinstance(opp, dict):
                out.append(dict(opp))
            else:
                out.append({"value": str(opp)})
        except Exception:
            out.append({"error": "Failed to serialize opportunity"})
    return out


def _build_quality_reports(opportunities: list[Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    try:
        from services.quality_filter import quality_filter as qf_pipeline
    except Exception:
        return reports

    for opp in opportunities or []:
        try:
            report = qf_pipeline.evaluate_opportunity(opp)
            reports.append(
                {
                    "opportunity_id": report.opportunity_id,
                    "passed": report.passed,
                    "rejection_reasons": report.rejection_reasons,
                    "filters": [
                        {
                            "filter_name": f.filter_name,
                            "passed": f.passed,
                            "reason": f.reason,
                            "threshold": f.threshold,
                            "actual_value": f.actual_value,
                        }
                        for f in report.filters
                    ],
                }
            )
        except Exception:
            continue
    return reports


async def _run_detect_once(
    strategy: Any,
    events: list[Any],
    markets: list[Any],
    prices: dict[str, dict[str, Any]],
    *,
    timeout_seconds: float,
    now_us: int | None = None,
) -> list[Any]:
    """Run the strategy's detect with the replay clock pinned to
    ``now_us`` (the tick's as-of time) so wall-clock reads inside detect
    resolve to simulated event time, not real time.  Both the instance
    attribute (thread-safe, for sync detect in an executor) and the
    utcnow ContextVar (for free-function ``utcnow()`` callers + child
    tasks) are set; the ContextVar is copied into the executor so a sync
    detect running off-loop still sees the simulated clock."""
    import contextvars as _cv

    from utils import utcnow as _utcnow_mod

    loop = asyncio.get_running_loop()
    prev_instance = getattr(strategy, "_replay_now_us", None)
    token = None
    if now_us is not None:
        try:
            strategy.set_replay_clock(int(now_us))
        except Exception:  # noqa: BLE001 — non-BaseStrategy duck types
            try:
                strategy._replay_now_us = int(now_us)
            except Exception:  # noqa: BLE001
                pass
        token = _utcnow_mod.set_replay_clock_us(int(now_us))
    try:
        if _has_custom_detect_async(strategy):
            return await asyncio.wait_for(
                strategy.detect_async(events, markets, prices),
                timeout=timeout_seconds,
            )
        # Sync detect runs in a thread — copy the current context (incl.
        # the replay clock ContextVar) so utcnow() inside the thread sees
        # the simulated time.
        ctx = _cv.copy_context()
        fn = strategy.detect_sync if _has_custom_detect_sync(strategy) else strategy.detect
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: ctx.run(fn, events, markets, prices)),
            timeout=timeout_seconds,
        )
    finally:
        if now_us is not None:
            try:
                strategy.set_replay_clock(prev_instance)
            except Exception:  # noqa: BLE001
                try:
                    strategy._replay_now_us = prev_instance
                except Exception:  # noqa: BLE001
                    pass
            if token is not None:
                _utcnow_mod.restore_replay_clock(token)


async def _fetch_prices_for_markets(
    markets: list[Any], *, token_cap: int = 2000, batch_size: int = 250
) -> dict[str, dict]:
    token_ids: list[str] = []
    seen: set[str] = set()
    for market in markets:
        for token_id in getattr(market, "clob_token_ids", None) or []:
            token = str(token_id or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            token_ids.append(token)
            if len(token_ids) >= token_cap:
                break
        if len(token_ids) >= token_cap:
            break
    if not token_ids:
        return {}

    from services.polymarket import polymarket_client

    prices: dict[str, dict] = {}
    for idx in range(0, len(token_ids), batch_size):
        chunk = token_ids[idx : idx + batch_size]
        try:
            batch = await polymarket_client.get_prices_batch(chunk)
            if isinstance(batch, dict):
                prices.update(batch)
        except Exception:
            continue
    return prices


def _select_replay_markets(markets: list[Any], max_markets: int) -> list[Any]:
    candidates: list[Any] = []
    for market in markets:
        if bool(getattr(market, "closed", False)) or not bool(getattr(market, "active", True)):
            continue
        token_ids = list(getattr(market, "clob_token_ids", None) or [])
        if len(token_ids) < 2:
            continue
        candidates.append(market)
    candidates.sort(
        key=lambda row: (
            float(getattr(row, "liquidity", 0.0) or 0.0),
            float(getattr(row, "volume", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return candidates[: max(1, int(max_markets))]


def _history_from_scanner_cache(
    market_id: str,
    *,
    start_ms: int,
    end_ms: int,
    step_ms: int,
) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    raw_history = getattr(scanner, "_market_price_history", {})
    points = raw_history.get(market_id, []) if isinstance(raw_history, dict) else []
    for row in points:
        if not isinstance(row, dict):
            continue
        try:
            ts_ms = int(float(row.get("t", 0)))
        except Exception:
            continue
        if ts_ms < start_ms or ts_ms > end_ms:
            continue
        yes = _clamp_probability(row.get("yes"))
        no = _clamp_probability(row.get("no"))
        if yes is None or no is None:
            continue
        out[_bucket_ms(ts_ms, start_ms, step_ms)] = (yes, no)
    return out


async def _history_from_polymarket_api(
    market: Any,
    *,
    start_ms: int,
    end_ms: int,
    step_ms: int,
) -> dict[int, tuple[float, float]]:
    token_ids = [str(token or "").strip() for token in (getattr(market, "clob_token_ids", None) or [])]
    token_ids = [token for token in token_ids if token]
    if len(token_ids) < 2:
        return {}
    yes_token = token_ids[0]
    no_token = token_ids[1]

    from services.polymarket import polymarket_client

    yes_result, no_result = await asyncio.gather(
        polymarket_client.get_prices_history(yes_token, start_ts=start_ms, end_ts=end_ms),
        polymarket_client.get_prices_history(no_token, start_ts=start_ms, end_ts=end_ms),
        return_exceptions=True,
    )
    yes_history = yes_result if isinstance(yes_result, list) else []
    no_history = no_result if isinstance(no_result, list) else []
    if not yes_history and not no_history:
        return {}

    yes_by_bucket: dict[int, float] = {}
    no_by_bucket: dict[int, float] = {}

    for row in yes_history:
        if not isinstance(row, dict):
            continue
        try:
            ts_ms = int(float(row.get("t", 0)))
        except Exception:
            continue
        if ts_ms < start_ms or ts_ms > end_ms:
            continue
        price = _clamp_probability(row.get("p"))
        if price is None:
            continue
        yes_by_bucket[_bucket_ms(ts_ms, start_ms, step_ms)] = price

    for row in no_history:
        if not isinstance(row, dict):
            continue
        try:
            ts_ms = int(float(row.get("t", 0)))
        except Exception:
            continue
        if ts_ms < start_ms or ts_ms > end_ms:
            continue
        price = _clamp_probability(row.get("p"))
        if price is None:
            continue
        no_by_bucket[_bucket_ms(ts_ms, start_ms, step_ms)] = price

    out: dict[int, tuple[float, float]] = {}
    for bucket in sorted(set(yes_by_bucket.keys()) | set(no_by_bucket.keys())):
        yes = yes_by_bucket.get(bucket)
        no = no_by_bucket.get(bucket)
        if yes is None and no is not None and 0.0 <= no <= 1.0:
            yes = 1.0 - no
        if no is None and yes is not None and 0.0 <= yes <= 1.0:
            no = 1.0 - yes
        if yes is None or no is None:
            continue
        out[bucket] = (yes, no)
    return out


def _opportunity_key(opp: Any, fallback: str) -> str:
    if isinstance(opp, dict):
        stable = str(opp.get("stable_id") or opp.get("id") or "").strip()
        return stable or fallback
    stable = str(getattr(opp, "stable_id", "") or getattr(opp, "id", "") or "").strip()
    return stable or fallback


def _opportunity_roi(opp: Any) -> float:
    if isinstance(opp, dict):
        try:
            return float(opp.get("roi_percent") or 0.0)
        except Exception:
            return 0.0
    try:
        return float(getattr(opp, "roi_percent", 0.0) or 0.0)
    except Exception:
        return 0.0


def _annotate_replay_ts(opp: Any, ts_ms: int) -> None:
    if isinstance(opp, dict):
        ctx = opp.get("strategy_context")
        if not isinstance(ctx, dict):
            ctx = {}
            opp["strategy_context"] = ctx
        ctx["backtest_replay_ts_ms"] = int(ts_ms)
        return
    ctx = getattr(opp, "strategy_context", None)
    if not isinstance(ctx, dict):
        ctx = {}
        try:
            setattr(opp, "strategy_context", ctx)
        except Exception:
            return
    ctx["backtest_replay_ts_ms"] = int(ts_ms)


async def _run_ohlc_replay_detection(
    strategy: Any,
    events: list[Any],
    markets: list[Any],
    *,
    base_prices: dict[str, dict],
    lookback_hours: int,
    timeframe: str,
    max_markets: int,
    max_steps: int,
) -> ReplayDetectRun:
    replay_markets = _select_replay_markets(markets, max_markets=max_markets)
    if not replay_markets:
        return ReplayDetectRun()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    step_ms = _timeframe_to_seconds(timeframe) * 1000
    start_ms = now_ms - (max(1, int(lookback_hours)) * 3600 * 1000)

    history_by_market: dict[str, dict[int, tuple[float, float]]] = {}
    to_fetch: list[Any] = []
    for market in replay_markets:
        market_id = str(getattr(market, "id", "") or "")
        if not market_id:
            continue
        cached = _history_from_scanner_cache(
            market_id,
            start_ms=start_ms,
            end_ms=now_ms,
            step_ms=step_ms,
        )
        if len(cached) >= 2:
            history_by_market[market_id] = cached
            continue
        to_fetch.append(market)

    if to_fetch:
        semaphore = asyncio.Semaphore(8)

        async def _fetch_one(market_row: Any) -> tuple[str, dict[int, tuple[float, float]]]:
            market_id = str(getattr(market_row, "id", "") or "")
            async with semaphore:
                try:
                    points = await _history_from_polymarket_api(
                        market_row,
                        start_ms=start_ms,
                        end_ms=now_ms,
                        step_ms=step_ms,
                    )
                except Exception:
                    points = {}
            return market_id, points

        fetched = await asyncio.gather(*[_fetch_one(market) for market in to_fetch])
        for market_id, points in fetched:
            if market_id and len(points) >= 2:
                history_by_market[market_id] = points

    if not history_by_market:
        return ReplayDetectRun()

    timeline = sorted({ts for points in history_by_market.values() for ts in points.keys()})
    if not timeline:
        return ReplayDetectRun(markets_replayed=len(history_by_market))
    if len(timeline) > max_steps:
        timeline = timeline[-max_steps:]

    selected_market_ids = set(history_by_market.keys())
    cloned_markets: list[Any] = []
    market_views: dict[str, Any] = {}
    market_state: dict[str, dict[str, Any]] = {}
    market_tokens: dict[str, tuple[str, str]] = {}

    for market in markets:
        if hasattr(market, "model_copy"):
            market_copy = market.model_copy(deep=True)
        else:
            market_copy = deepcopy(market)
        cloned_markets.append(market_copy)

        market_id = str(getattr(market_copy, "id", "") or "")
        if market_id not in selected_market_ids:
            continue

        market_views[market_id] = market_copy
        token_ids = [str(token or "").strip() for token in (getattr(market_copy, "clob_token_ids", None) or [])]
        yes_token = token_ids[0] if len(token_ids) > 0 else ""
        no_token = token_ids[1] if len(token_ids) > 1 else ""
        market_tokens[market_id] = (yes_token, no_token)

        try:
            default_yes = float(getattr(market_copy, "yes_price", 0.5) or 0.5)
        except Exception:
            default_yes = 0.5
        try:
            default_no = float(getattr(market_copy, "no_price", 1.0 - default_yes) or (1.0 - default_yes))
        except Exception:
            default_no = 1.0 - default_yes

        points = sorted(history_by_market[market_id].items(), key=lambda row: row[0])
        market_state[market_id] = {
            "points": points,
            "idx": 0,
            "yes": default_yes,
            "no": default_no,
        }

    if not market_state:
        return ReplayDetectRun()

    deduped: dict[str, Any] = {}
    step_errors = 0
    steps_run = 0

    for ts_ms in timeline:
        prices_for_step = dict(base_prices or {})

        for market_id, state in market_state.items():
            points = state["points"]
            idx = int(state["idx"])
            while idx < len(points) and points[idx][0] <= ts_ms:
                yes_val, no_val = points[idx][1]
                state["yes"] = yes_val
                state["no"] = no_val
                idx += 1
            state["idx"] = idx

            yes_val = float(state["yes"])
            no_val = float(state["no"])

            market_view = market_views[market_id]
            market_view.outcome_prices = [yes_val, no_val]
            tokens = getattr(market_view, "tokens", None)
            if isinstance(tokens, list):
                if len(tokens) > 0 and hasattr(tokens[0], "price"):
                    tokens[0].price = yes_val
                if len(tokens) > 1 and hasattr(tokens[1], "price"):
                    tokens[1].price = no_val

            yes_token, no_token = market_tokens.get(market_id, ("", ""))
            if yes_token:
                prices_for_step[yes_token] = {"mid": yes_val}
            if no_token:
                prices_for_step[no_token] = {"mid": no_val}

        try:
            step_opps = await _run_detect_once(
                strategy,
                events,
                cloned_markets,
                prices_for_step,
                timeout_seconds=12.0,
            )
        except Exception:
            step_errors += 1
            continue

        steps_run += 1
        for index, opp in enumerate(step_opps or []):
            _annotate_replay_ts(opp, ts_ms)
            key = _opportunity_key(opp, fallback=f"{ts_ms}:{index}")
            existing = deduped.get(key)
            if existing is None or _opportunity_roi(opp) > _opportunity_roi(existing):
                deduped[key] = opp

    return ReplayDetectRun(
        opportunities=list(deduped.values()),
        steps_run=steps_run,
        markets_replayed=len(market_state),
        step_errors=step_errors,
    )


# ---------------------------------------------------------------------------
# Parameter sweep + walk-forward validation
# ---------------------------------------------------------------------------


@dataclass
class GridConfigResult:
    params: dict[str, Any] = field(default_factory=dict)
    num_opportunities: int = 0
    avg_roi: float = 0.0
    total_roi: float = 0.0
    quality_pass_rate: float = 0.0

    def composite_score(self) -> float:
        return self.total_roi * self.quality_pass_rate


@dataclass
class ParameterSweepResult:
    success: bool = False
    grid_results: list[dict[str, Any]] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_train_score: float = 0.0
    best_test_score: float = 0.0
    train_ratio: float = 0.75
    total_configs_tested: int = 0
    sweep_time_ms: float = 0.0
    runtime_error: Optional[str] = None
    runtime_traceback: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_opportunities(opportunities: list[Any]) -> GridConfigResult:
    if not opportunities:
        return GridConfigResult()

    serialized = _serialize_opportunities(opportunities)
    reports = _build_quality_reports(opportunities)

    total_roi = 0.0
    for opp in serialized:
        total_roi += float(opp.get("roi_percent", 0.0) or 0.0)

    num = len(serialized)
    avg_roi = total_roi / num if num > 0 else 0.0
    passed = sum(1 for r in reports if r.get("passed", False))
    quality_pass_rate = passed / num if num > 0 else 0.0

    return GridConfigResult(
        num_opportunities=num,
        avg_roi=round(avg_roi, 4),
        total_roi=round(total_roi, 4),
        quality_pass_rate=round(quality_pass_rate, 4),
    )


async def _fetch_market_data() -> tuple[list[Any], list[Any], dict[str, dict]]:
    events = None
    markets = None
    prices = None

    if (
        hasattr(scanner, "_cached_events")
        and scanner._cached_events
        and hasattr(scanner, "_cached_markets")
        and scanner._cached_markets
    ):
        events = list(scanner._cached_events)
        markets = list(scanner._cached_markets)
        prices = dict(scanner._cached_prices) if hasattr(scanner, "_cached_prices") and scanner._cached_prices else {}

    if not events or not markets:
        from services.polymarket import polymarket_client

        events_raw, markets_raw = await asyncio.gather(
            polymarket_client.get_all_events(closed=False),
            polymarket_client.get_all_markets(active=True),
        )
        events = events_raw
        markets = markets_raw
        prices = await _fetch_prices_for_markets(markets, token_cap=2000, batch_size=250)

    return events, markets, prices or {}


async def _detect_for_config(
    source_code: str,
    slug: str,
    config: dict[str, Any],
    events: list[Any],
    markets: list[Any],
    base_prices: dict[str, dict],
    replay_markets: list[Any],
    history_by_market: dict[str, dict[int, tuple[float, float]]],
    timeline: list[int],
) -> list[Any]:
    loader = StrategyLoader()
    bt_slug = f"_sweep_{slug}_{int(time.time() * 1000)}"
    try:
        loaded = loader.load(bt_slug, source_code, config)
        strategy = loaded.instance

        opportunities = await _run_detect_once(strategy, events, markets, base_prices, timeout_seconds=30.0)

        if not opportunities and timeline and history_by_market:
            replay_run = await _run_ohlc_replay_detection(
                strategy,
                events,
                markets,
                base_prices=base_prices,
                lookback_hours=_DEFAULT_REPLAY_LOOKBACK_HOURS,
                timeframe=_DEFAULT_REPLAY_TIMEFRAME,
                max_markets=_DEFAULT_REPLAY_MAX_MARKETS,
                max_steps=_DEFAULT_REPLAY_MAX_STEPS,
            )
            if replay_run.opportunities:
                opportunities = replay_run.opportunities

        return opportunities or []
    finally:
        try:
            loader.unload(bt_slug)
        except Exception:
            pass


async def run_parameter_sweep(
    source_code: str,
    slug: str = "_sweep_preview",
    param_grid: Optional[dict[str, list[Any]]] = None,
    train_ratio: float = 0.75,
    top_k: int = 10,
) -> ParameterSweepResult:
    result = ParameterSweepResult(train_ratio=train_ratio)
    sweep_start = time.monotonic()

    if not param_grid:
        result.runtime_error = "param_grid is required and must not be empty"
        result.sweep_time_ms = (time.monotonic() - sweep_start) * 1000
        return result

    validation = validate_strategy_source(source_code)
    if not validation["valid"]:
        result.runtime_error = "Strategy validation failed: " + "; ".join(validation.get("errors", []))
        result.sweep_time_ms = (time.monotonic() - sweep_start) * 1000
        return result

    param_names = list(param_grid.keys())
    value_lists = [param_grid[n] for n in param_names]
    all_combos = list(itertools.product(*value_lists))
    result.total_configs_tested = len(all_combos)

    try:
        events, markets, base_prices = await _fetch_market_data()
    except Exception as e:
        result.runtime_error = f"Failed to fetch market data: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.sweep_time_ms = (time.monotonic() - sweep_start) * 1000
        return result

    replay_markets = _select_replay_markets(markets, max_markets=_DEFAULT_REPLAY_MAX_MARKETS)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    step_ms = _timeframe_to_seconds(_DEFAULT_REPLAY_TIMEFRAME) * 1000
    start_ms = now_ms - (_DEFAULT_REPLAY_LOOKBACK_HOURS * 3600 * 1000)

    history_by_market: dict[str, dict[int, tuple[float, float]]] = {}
    to_fetch: list[Any] = []
    for market in replay_markets:
        market_id = str(getattr(market, "id", "") or "")
        if not market_id:
            continue
        cached = _history_from_scanner_cache(market_id, start_ms=start_ms, end_ms=now_ms, step_ms=step_ms)
        if len(cached) >= 2:
            history_by_market[market_id] = cached
        else:
            to_fetch.append(market)

    if to_fetch:
        semaphore = asyncio.Semaphore(8)

        async def _fetch_one(market_row: Any) -> tuple[str, dict[int, tuple[float, float]]]:
            market_id = str(getattr(market_row, "id", "") or "")
            async with semaphore:
                try:
                    points = await _history_from_polymarket_api(
                        market_row, start_ms=start_ms, end_ms=now_ms, step_ms=step_ms
                    )
                except Exception:
                    points = {}
            return market_id, points

        fetched = await asyncio.gather(*[_fetch_one(m) for m in to_fetch])
        for market_id, points in fetched:
            if market_id and len(points) >= 2:
                history_by_market[market_id] = points

    timeline = sorted({ts for pts in history_by_market.values() for ts in pts.keys()})

    # Split timeline into train/test for walk-forward
    split_idx = max(1, int(len(timeline) * train_ratio))
    train_timeline = timeline[:split_idx]
    test_timeline = timeline[split_idx:]

    # Run grid search on train window
    grid_scores: list[tuple[dict[str, Any], GridConfigResult]] = []

    for combo in all_combos:
        config = dict(zip(param_names, combo))

        try:
            opps = await _detect_for_config(
                source_code=source_code,
                slug=slug,
                config=config,
                events=events,
                markets=markets,
                base_prices=base_prices,
                replay_markets=replay_markets,
                history_by_market=history_by_market,
                timeline=train_timeline,
            )
        except Exception:
            opps = []

        scored = _score_opportunities(opps)
        scored.params = config
        grid_scores.append((config, scored))

        result.grid_results.append(
            {
                "params": config,
                "num_opportunities": scored.num_opportunities,
                "avg_roi": scored.avg_roi,
                "total_roi": scored.total_roi,
                "quality_pass_rate": scored.quality_pass_rate,
            }
        )

        await asyncio.sleep(0)

    # Rank by composite metric (ROI * quality_pass_rate)
    grid_scores.sort(key=lambda x: x[1].composite_score(), reverse=True)

    if not grid_scores:
        result.runtime_error = "No configurations produced results"
        result.sweep_time_ms = (time.monotonic() - sweep_start) * 1000
        return result

    # Take top_k and validate on held-out test window
    top_candidates = grid_scores[: min(top_k, len(grid_scores))]

    best_config = top_candidates[0][0]
    best_train = top_candidates[0][1].composite_score()
    best_test = 0.0

    if test_timeline:
        best_test_score_so_far = -float("inf")
        for config, train_scored in top_candidates:
            try:
                test_opps = await _detect_for_config(
                    source_code=source_code,
                    slug=slug,
                    config=config,
                    events=events,
                    markets=markets,
                    base_prices=base_prices,
                    replay_markets=replay_markets,
                    history_by_market=history_by_market,
                    timeline=test_timeline,
                )
            except Exception:
                test_opps = []

            test_scored = _score_opportunities(test_opps)
            test_composite = test_scored.composite_score()

            if test_composite > best_test_score_so_far:
                best_test_score_so_far = test_composite
                best_config = config
                best_train = train_scored.composite_score()
                best_test = test_composite

            await asyncio.sleep(0)
    else:
        best_test = best_train

    result.best_params = best_config
    result.best_train_score = round(best_train, 4)
    result.best_test_score = round(best_test, 4)
    result.success = True
    result.sweep_time_ms = (time.monotonic() - sweep_start) * 1000
    return result


async def run_strategy_backtest(
    source_code: str,
    slug: str = "_backtest_preview",
    config: Optional[dict[str, Any]] = None,
    use_ohlc_replay: bool = True,
    replay_lookback_hours: int = _DEFAULT_REPLAY_LOOKBACK_HOURS,
    replay_timeframe: str = _DEFAULT_REPLAY_TIMEFRAME,
    replay_max_markets: int = _DEFAULT_REPLAY_MAX_MARKETS,
    replay_max_steps: int = _DEFAULT_REPLAY_MAX_STEPS,
    max_opportunities: int = 100,
) -> BacktestResult:
    """Run a strategy's detection code against current and replayed market data."""
    result = BacktestResult(strategy_slug=slug)
    result.replay_window_hours = max(1, int(replay_lookback_hours))
    result.replay_timeframe = str(replay_timeframe or _DEFAULT_REPLAY_TIMEFRAME)
    total_start = time.monotonic()

    # ---- 1. Validate source code ----
    validation = validate_strategy_source(source_code)
    result.validation_errors = validation.get("errors", [])
    result.validation_warnings = validation.get("warnings", [])
    result.class_name = validation.get("class_name") or ""

    if not validation["valid"]:
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result

    # ---- 2. Load strategy via unified loader ----
    loader = StrategyLoader()  # Fresh isolated loader for backtest
    bt_slug = f"_bt_{slug}_{int(time.time())}"
    load_start = time.monotonic()
    try:
        loaded = loader.load(bt_slug, source_code, config)
        strategy = loaded.instance
        result.strategy_name = getattr(strategy, "name", bt_slug)
    except Exception as e:
        result.runtime_error = f"Failed to load strategy: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.load_time_ms = (time.monotonic() - load_start) * 1000

    # ---- 3. Get market data ----
    data_start = time.monotonic()
    try:
        events = None
        markets = None
        prices = None

        # Try scanner cache first (most recent scan data)
        if (
            hasattr(scanner, "_cached_events")
            and scanner._cached_events
            and hasattr(scanner, "_cached_markets")
            and scanner._cached_markets
        ):
            events = list(scanner._cached_events)
            markets = list(scanner._cached_markets)
            prices = (
                dict(scanner._cached_prices) if hasattr(scanner, "_cached_prices") and scanner._cached_prices else {}
            )
            result.data_source = "cache"

        # Fallback: fetch fresh data
        if not events or not markets:
            from services.polymarket import polymarket_client

            events_raw, markets_raw = await asyncio.gather(
                polymarket_client.get_all_events(closed=False),
                polymarket_client.get_all_markets(active=True),
            )
            events = events_raw
            markets = markets_raw
            prices = await _fetch_prices_for_markets(markets, token_cap=2000, batch_size=250)
            result.data_source = "fresh"

        result.num_events = len(events)
        result.num_markets = len(markets)
        result.num_prices = len(prices)

    except Exception as e:
        result.runtime_error = f"Failed to fetch market data: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.data_fetch_time_ms = (time.monotonic() - data_start) * 1000

    # ---- 4. Run detection ----
    detect_start = time.monotonic()
    try:
        opportunities = await _run_detect_once(
            strategy,
            events,
            markets,
            prices,
            timeout_seconds=60.0,
        )

        replay_run = ReplayDetectRun()
        should_run_replay = (
            bool(use_ohlc_replay) and len(opportunities or []) == 0 and not _has_custom_detect_async(strategy)
        )
        if should_run_replay:
            replay_run = await _run_ohlc_replay_detection(
                strategy,
                events,
                markets,
                base_prices=prices or {},
                lookback_hours=max(1, int(replay_lookback_hours)),
                timeframe=str(replay_timeframe or _DEFAULT_REPLAY_TIMEFRAME),
                max_markets=max(1, int(replay_max_markets)),
                max_steps=max(1, int(replay_max_steps)),
            )
            result.replay_steps = replay_run.steps_run
            result.replay_markets = replay_run.markets_replayed
            if replay_run.step_errors > 0:
                result.validation_warnings.append(
                    f"OHLC replay skipped {replay_run.step_errors} snapshots due to strategy/runtime errors."
                )
            if replay_run.opportunities:
                opportunities = replay_run.opportunities
                result.replay_mode = "ohlc_replay"
                result.data_source = f"{result.data_source}+ohlc_replay"
        elif bool(use_ohlc_replay) and _has_custom_detect_async(strategy) and len(opportunities or []) == 0:
            result.validation_warnings.append(
                "OHLC replay is disabled for async detect_async() strategies in code backtest mode."
            )

        capped_opportunities = list(opportunities or [])
        capped_limit = max(1, int(max_opportunities))
        total_found = len(capped_opportunities)
        if total_found > capped_limit:
            capped_opportunities = capped_opportunities[:capped_limit]
            result.validation_warnings.append(
                f"Opportunity output truncated to {capped_limit} rows from {total_found} detected opportunities."
            )

        result.opportunities = _serialize_opportunities(capped_opportunities)
        result.num_opportunities = len(result.opportunities)
        result.quality_reports = _build_quality_reports(capped_opportunities)
        result.success = True

    except asyncio.TimeoutError:
        result.runtime_error = "Strategy detection timed out after 60 seconds"
    except Exception as e:
        result.runtime_error = f"Strategy detection error: {e}"
        result.runtime_traceback = traceback.format_exc()
    finally:
        result.detect_time_ms = (time.monotonic() - detect_start) * 1000

    # ---- 5. Cleanup ----
    try:
        loader.unload(bt_slug)
    except Exception:
        pass

    result.total_time_ms = (time.monotonic() - total_start) * 1000
    return result


# ---------------------------------------------------------------------------
# Evaluate backtest
# ---------------------------------------------------------------------------


@dataclass
class EvaluateBacktestResult:
    """Result of running a strategy's evaluate() against recent trade signals."""

    success: bool = False
    strategy_slug: str = ""
    strategy_name: str = ""
    class_name: str = ""
    num_signals: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    selected: int = 0
    skipped: int = 0
    blocked: int = 0
    load_time_ms: float = 0
    data_fetch_time_ms: float = 0
    evaluate_time_ms: float = 0
    total_time_ms: float = 0
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    runtime_error: Optional[str] = None
    runtime_traceback: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_evaluate_backtest(
    source_code: str,
    slug: str = "_backtest_evaluate",
    config: Optional[dict[str, Any]] = None,
    max_signals: int = 50,
) -> EvaluateBacktestResult:
    """Run a strategy's evaluate() against recent unconsumed trade signals.

    Loads the strategy, fetches recent signals from the DB, and runs evaluate()
    on each to show which would be selected/skipped and why.
    """
    result = EvaluateBacktestResult(strategy_slug=slug)
    total_start = time.monotonic()

    # 1. Validate
    validation = validate_strategy_source(source_code)
    result.validation_errors = validation.get("errors", [])
    result.validation_warnings = validation.get("warnings", [])
    result.class_name = validation.get("class_name") or ""
    if not validation["valid"]:
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result

    # 2. Load
    loader = StrategyLoader()
    bt_slug = f"_bt_eval_{slug}_{int(time.time())}"
    load_start = time.monotonic()
    try:
        loaded = loader.load(bt_slug, source_code, config)
        strategy = loaded.instance
        result.strategy_name = getattr(strategy, "name", bt_slug)
    except Exception as e:
        result.runtime_error = f"Failed to load strategy: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.load_time_ms = (time.monotonic() - load_start) * 1000

    if not hasattr(strategy, "evaluate"):
        result.runtime_error = "Strategy does not implement evaluate()"
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result

    # 3. Fetch recent trade signals
    data_start = time.monotonic()
    try:
        from models.database import BacktestAsyncSessionLocal
        from sqlalchemy import select

        async with BacktestAsyncSessionLocal() as session:
            from models.database import TradeSignalEmission

            query = select(TradeSignalEmission).order_by(TradeSignalEmission.created_at.desc()).limit(max_signals)
            signals = list((await session.execute(query)).scalars().all())
        result.num_signals = len(signals)
    except Exception as e:
        result.runtime_error = f"Failed to fetch signals: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.data_fetch_time_ms = (time.monotonic() - data_start) * 1000

    # 4. Run evaluate() on each signal
    eval_start = time.monotonic()
    try:
        from datetime import datetime, timezone
        from services.trader_orchestrator.decision_gates import (
            apply_platform_decision_gates,
            is_within_trading_schedule_utc,
        )
        from services.trader_orchestrator.risk_manager import evaluate_risk

        merged_config = dict(config or {})
        platform_overrides = merged_config.pop("__platform__", {})
        platform_overrides = platform_overrides if isinstance(platform_overrides, dict) else {}
        strategy_defaults: dict[str, Any] = {}
        loaded_config = getattr(strategy, "config", None)
        if isinstance(loaded_config, dict):
            strategy_defaults = dict(loaded_config)
        else:
            loaded_default_config = getattr(strategy, "default_config", None)
            if isinstance(loaded_default_config, dict):
                strategy_defaults = dict(loaded_default_config)
        params = {**strategy_defaults, **merged_config}
        platform_global_risk = (
            dict(platform_overrides.get("global_risk", {}))
            if isinstance(platform_overrides.get("global_risk", {}), dict)
            else {}
        )
        platform_risk_limits = (
            dict(platform_overrides.get("risk_limits", {}))
            if isinstance(platform_overrides.get("risk_limits", {}), dict)
            else {}
        )
        platform_metadata = (
            {"trading_schedule_utc": platform_overrides.get("trading_schedule_utc")}
            if isinstance(platform_overrides.get("trading_schedule_utc"), dict)
            else {}
        )
        platform_allow_averaging = bool(platform_overrides.get("allow_averaging", False))
        platform_occupied_market_ids = {
            str(value or "").strip()
            for value in (platform_overrides.get("occupied_market_ids") or [])
            if str(value or "").strip()
        }

        for sig in signals:
            try:
                context = {
                    "params": params,
                    "mode": "backtest",
                    "source_config": {},
                }
                decision = strategy.evaluate(sig, context)
                checks_payload: list[dict[str, Any]] = []
                for c in getattr(decision, "checks", None) or []:
                    checks_payload.append(
                        {
                            "check_key": str(getattr(c, "key", "") or getattr(c, "check_key", "")),
                            "check_label": str(getattr(c, "label", "") or getattr(c, "check_label", "")),
                            "passed": bool(getattr(c, "passed", False)),
                            "score": getattr(c, "score", None),
                            "detail": str(getattr(c, "detail", "") or ""),
                        }
                    )

                def _backtest_risk_evaluator(size_for_eval: float):
                    risk_result = evaluate_risk(
                        size_usd=size_for_eval,
                        gross_exposure_usd=0.0,
                        trader_open_positions=0,
                        trader_open_orders=0,
                        market_exposure_usd=0.0,
                        global_limits=platform_global_risk,
                        trader_limits=platform_risk_limits,
                        global_daily_realized_pnl_usd=0.0,
                        trader_daily_realized_pnl_usd=0.0,
                        global_unrealized_pnl_usd=0.0,
                        trader_unrealized_pnl_usd=0.0,
                        trader_consecutive_losses=0,
                        cycle_orders_placed=0,
                        cooldown_active=False,
                        mode="backtest",
                    )
                    return risk_result, {
                        "global_daily_realized_pnl_usd": 0.0,
                        "trader_daily_realized_pnl_usd": 0.0,
                        "global_unrealized_pnl_usd": 0.0,
                        "trader_unrealized_pnl_usd": 0.0,
                        "intra_cycle_committed_usd": 0.0,
                        "adjusted_global_daily_pnl_usd": 0.0,
                        "adjusted_trader_daily_pnl_usd": 0.0,
                        "trader_consecutive_losses": 0,
                        "cooldown_seconds": 0,
                        "cooldown_active": False,
                        "cooldown_remaining_seconds": 0,
                        "trader_open_positions": 0,
                        "trader_open_orders": 0,
                    }

                gate_result = apply_platform_decision_gates(
                    decision_obj=decision,
                    runtime_signal=sig,
                    strategy=None,
                    checks_payload=checks_payload,
                    trading_schedule_ok=is_within_trading_schedule_utc(platform_metadata, datetime.now(timezone.utc)),
                    trading_schedule_config=platform_metadata.get("trading_schedule_utc"),
                    global_limits=platform_global_risk,
                    effective_risk_limits=platform_risk_limits,
                    allow_averaging=platform_allow_averaging,
            occupied_market_ids=platform_occupied_market_ids,
                    portfolio_allocator=None,
                    risk_evaluator=_backtest_risk_evaluator,
                    invoke_hooks=False,
                    strategy_params=params,
                    execution_mode="backtest",
                )

                decision_str = str(gate_result["final_decision"])
                reason_str = str(gate_result["final_reason"])

                result.decisions.append(
                    {
                        "signal_id": getattr(sig, "id", None),
                        "source": getattr(sig, "source", ""),
                        "strategy_type": getattr(sig, "strategy_type", ""),
                        "strategy_decision": gate_result["strategy_decision"],
                        "strategy_reason": gate_result["strategy_reason"],
                        "decision": decision_str,
                        "reason": reason_str,
                        "size_usd": gate_result["size_usd"],
                        "checks": gate_result["checks_payload"],
                        "platform_gates": gate_result["platform_gates"],
                        "risk_snapshot": gate_result["risk_snapshot"],
                    }
                )

                if decision_str == "selected":
                    result.selected += 1
                elif decision_str == "blocked":
                    result.blocked += 1
                else:
                    result.skipped += 1
            except Exception as exc:
                result.decisions.append(
                    {
                        "signal_id": getattr(sig, "id", None),
                        "decision": "error",
                        "reason": str(exc),
                        "checks": [],
                    }
                )

        result.success = True
    except Exception as e:
        result.runtime_error = f"Evaluate backtest error: {e}"
        result.runtime_traceback = traceback.format_exc()
    finally:
        result.evaluate_time_ms = (time.monotonic() - eval_start) * 1000

    try:
        loader.unload(bt_slug)
    except Exception:
        pass

    result.total_time_ms = (time.monotonic() - total_start) * 1000
    return result


# ---------------------------------------------------------------------------
# Exit backtest
# ---------------------------------------------------------------------------


@dataclass
class ExitBacktestResult:
    """Result of running a strategy's should_exit() against open positions."""

    success: bool = False
    strategy_slug: str = ""
    strategy_name: str = ""
    class_name: str = ""
    num_positions: int = 0
    exit_decisions: list[dict[str, Any]] = field(default_factory=list)
    would_close: int = 0
    would_reduce: int = 0
    would_hold: int = 0
    errors: int = 0
    load_time_ms: float = 0
    data_fetch_time_ms: float = 0
    exit_time_ms: float = 0
    total_time_ms: float = 0
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    runtime_error: Optional[str] = None
    runtime_traceback: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_exit_backtest(
    source_code: str,
    slug: str = "_backtest_exit",
    config: Optional[dict[str, Any]] = None,
    max_positions: int = 50,
) -> ExitBacktestResult:
    """Run a strategy's should_exit() against current open positions.

    Loads the strategy, fetches open shadow positions, and runs should_exit()
    on each to show which would be closed and why.
    """
    result = ExitBacktestResult(strategy_slug=slug)
    total_start = time.monotonic()

    # 1. Validate
    validation = validate_strategy_source(source_code)
    result.validation_errors = validation.get("errors", [])
    result.validation_warnings = validation.get("warnings", [])
    result.class_name = validation.get("class_name") or ""
    if not validation["valid"]:
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result

    # 2. Load
    loader = StrategyLoader()
    bt_slug = f"_bt_exit_{slug}_{int(time.time())}"
    load_start = time.monotonic()
    try:
        loaded = loader.load(bt_slug, source_code, config)
        strategy = loaded.instance
        result.strategy_name = getattr(strategy, "name", bt_slug)
    except Exception as e:
        result.runtime_error = f"Failed to load strategy: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.load_time_ms = (time.monotonic() - load_start) * 1000

    if not hasattr(strategy, "should_exit"):
        result.runtime_error = "Strategy does not implement should_exit()"
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result

    # 3. Fetch open shadow positions
    data_start = time.monotonic()
    try:
        from models.database import BacktestAsyncSessionLocal, TraderPosition
        from sqlalchemy import select

        async with BacktestAsyncSessionLocal() as session:
            query = select(TraderPosition).where(TraderPosition.status == "open")
            order_columns = []
            for column_name in ("first_order_at", "opened_at", "created_at"):
                column = getattr(TraderPosition, column_name, None)
                if column is not None:
                    order_columns.append(column.desc())
            if order_columns:
                query = query.order_by(*order_columns)
            query = query.limit(max(1, int(max_positions)))
            positions = list((await session.execute(query)).scalars().all())
        result.num_positions = len(positions)
        if result.num_positions == 0:
            result.validation_warnings.append("No open positions available for exit backtest.")
    except Exception as e:
        result.runtime_error = f"Failed to fetch positions: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.data_fetch_time_ms = (time.monotonic() - data_start) * 1000

    # 4. Run should_exit() on each position
    exit_start = time.monotonic()
    try:
        now_utc = datetime.now(timezone.utc)
        for pos in positions:
            try:
                payload_raw = getattr(pos, "payload_json", None)
                payload = payload_raw if isinstance(payload_raw, dict) else {}
                entry_price = 0.0
                for candidate in (
                    payload.get("entry_price"),
                    getattr(pos, "avg_entry_price", None),
                    payload.get("avg_entry_price"),
                    payload.get("effective_price"),
                    0.0,
                ):
                    try:
                        entry_price = float(candidate or 0.0)
                    except Exception:
                        continue
                    if entry_price > 0:
                        break
                current_price = entry_price
                for candidate in (
                    payload.get("last_price"),
                    payload.get("current_price"),
                    payload.get("mark_price"),
                    payload.get("mid_price"),
                    entry_price,
                ):
                    try:
                        current_price = float(candidate if candidate is not None else entry_price)
                        break
                    except Exception:
                        continue
                highest_price = current_price
                for candidate in (payload.get("highest_price"), current_price):
                    try:
                        highest_price = float(candidate if candidate is not None else current_price)
                        break
                    except Exception:
                        continue
                lowest_price = current_price
                for candidate in (payload.get("lowest_price"), current_price):
                    try:
                        lowest_price = float(candidate if candidate is not None else current_price)
                        break
                    except Exception:
                        continue
                opened_at = getattr(pos, "first_order_at", None) or getattr(pos, "created_at", None)
                opened_at_iso: Optional[str] = None
                age_minutes = 0.0
                if isinstance(opened_at, datetime):
                    opened_at_utc = (
                        opened_at if opened_at.tzinfo is not None else opened_at.replace(tzinfo=timezone.utc)
                    )
                    opened_at_iso = opened_at_utc.isoformat()
                    age_minutes = max(0.0, (now_utc - opened_at_utc).total_seconds() / 60.0)
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                notional_usd = float(getattr(pos, "total_notional_usd", 0.0) or 0.0)
                strategy_context_raw = payload.get("strategy_context")
                strategy_context = strategy_context_raw if isinstance(strategy_context_raw, dict) else {}

                class _PositionView:
                    pass

                pos_view = _PositionView()
                pos_view.entry_price = entry_price
                pos_view.current_price = current_price
                pos_view.highest_price = highest_price
                pos_view.lowest_price = lowest_price
                pos_view.age_minutes = age_minutes
                pos_view.pnl_percent = pnl_pct
                pos_view.strategy_context = strategy_context
                pos_view.config = config or {}
                pos_view.outcome_idx = payload.get("outcome_idx", 0)
                pos_view.market_id = getattr(pos, "market_id", "")
                pos_view.market_question = getattr(pos, "market_question", "")
                pos_view.direction = getattr(pos, "direction", "")
                pos_view.mode = getattr(pos, "mode", "shadow")
                pos_view.total_notional_usd = notional_usd
                pos_view.opened_at = opened_at

                market_state = {
                    "current_price": current_price,
                    "market_tradable": True,
                    "is_resolved": False,
                    "winning_outcome": None,
                    "market_id": getattr(pos, "market_id", None),
                }

                exit_decision = strategy.should_exit(pos_view, market_state)
                action_raw = getattr(exit_decision, "action", "hold") if exit_decision else "hold"
                action = str(action_raw or "hold").strip().lower()
                if action not in {"close", "hold", "reduce"}:
                    action = "hold"
                reason = str(getattr(exit_decision, "reason", "") if exit_decision else "")
                close_price = getattr(exit_decision, "close_price", None) if exit_decision else None
                reduce_fraction = getattr(exit_decision, "reduce_fraction", None) if exit_decision else None
                close_price_value = None
                if close_price is not None:
                    try:
                        close_price_value = float(close_price)
                    except Exception:
                        close_price_value = None
                reduce_fraction_value = None
                if reduce_fraction is not None:
                    try:
                        reduce_fraction_value = max(0.0, min(1.0, float(reduce_fraction)))
                    except Exception:
                        reduce_fraction_value = None

                result.exit_decisions.append(
                    {
                        "position_id": pos.id,
                        "market_id": getattr(pos, "market_id", None),
                        "market_question": getattr(pos, "market_question", None),
                        "direction": getattr(pos, "direction", None),
                        "mode": getattr(pos, "mode", None),
                        "notional_usd": round(notional_usd, 2),
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "highest_price": highest_price,
                        "lowest_price": lowest_price,
                        "pnl_pct": round(pnl_pct, 2),
                        "age_minutes": round(age_minutes, 2),
                        "opened_at": opened_at_iso,
                        "action": action,
                        "reason": reason,
                        "close_price": close_price_value,
                        "reduce_fraction": reduce_fraction_value,
                    }
                )

                if action == "close":
                    result.would_close += 1
                elif action == "reduce":
                    result.would_reduce += 1
                else:
                    result.would_hold += 1
            except Exception as exc:
                result.errors += 1
                result.exit_decisions.append(
                    {
                        "position_id": pos.id,
                        "action": "error",
                        "reason": str(exc),
                    }
                )

        result.success = True
    except Exception as e:
        result.runtime_error = f"Exit backtest error: {e}"
        result.runtime_traceback = traceback.format_exc()
    finally:
        result.exit_time_ms = (time.monotonic() - exit_start) * 1000

    try:
        loader.unload(bt_slug)
    except Exception:
        pass

    result.total_time_ms = (time.monotonic() - total_start) * 1000
    return result


# ---------------------------------------------------------------------------
# Execution-realistic backtest (services.backtest engine)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionBacktestResult:
    """Result of an execution-realistic backtest using the production engine."""

    success: bool = False
    strategy_slug: str = ""
    strategy_name: str = ""
    class_name: str = ""
    initial_capital_usd: float = 0.0
    start_iso: str = ""
    end_iso: str = ""
    n_intents: int = 0
    n_snapshots: int = 0
    final_equity_usd: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe: dict[str, Any] = field(default_factory=dict)
    sortino: dict[str, Any] = field(default_factory=dict)
    calmar: dict[str, Any] = field(default_factory=dict)
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    drawdown_duration_seconds: float = 0.0
    hit_rate: dict[str, Any] = field(default_factory=dict)
    profit_factor: dict[str, Any] = field(default_factory=dict)
    expectancy_usd: dict[str, Any] = field(default_factory=dict)
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    trade_count: int = 0
    fees_paid_usd: float = 0.0
    fees_per_fill_usd: float = 0.0
    fees_resolution_usd: float = 0.0
    total_fills: int = 0
    rejected_orders: int = 0
    cancelled_orders: int = 0
    closed_position_count: int = 0
    open_position_count: int = 0
    expected_shortfall_5pct: dict[str, Any] = field(default_factory=dict)
    expected_shortfall_1pct: dict[str, Any] = field(default_factory=dict)
    tail_ratio: dict[str, Any] = field(default_factory=dict)
    gain_to_pain: dict[str, Any] = field(default_factory=dict)
    correlation_pairs: list[dict[str, Any]] = field(default_factory=list)
    fills_sample: list[dict[str, Any]] = field(default_factory=list)
    equity_curve_sample: list[dict[str, Any]] = field(default_factory=list)
    positions_summary: list[dict[str, Any]] = field(default_factory=list)
    load_time_ms: float = 0.0
    data_fetch_time_ms: float = 0.0
    run_time_ms: float = 0.0
    total_time_ms: float = 0.0
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    runtime_error: Optional[str] = None
    runtime_traceback: Optional[str] = None
    # Pre-flight data coverage stats — populated before the engine runs
    # so the operator can see whether "0 trades" is a strategy outcome
    # or a data-fidelity outcome.  Schema:
    #   {
    #     "opp_tokens": int,                      # tokens with opps in window
    #     "tokens_with_snapshots": int,           # of those, in mms table
    #     "tokens_with_deltas": int,              # of those, in book_delta_events
    #     "snapshots_total": int,                 # total mms rows in window
    #     "deltas_total": int,                    # total bde rows in window
    #     "median_snaps_per_token_per_hour": float,
    #     "p10_snaps_per_token_per_hour": float,
    #     "fidelity_rating": "high"|"medium"|"low"|"none",
    #     "recommended_action": str,              # human-readable advice
    #   }
    data_coverage: dict[str, Any] = field(default_factory=dict)
    # Which book source the engine ran against.  Now always "parquet" — the
    # unified MarketDataView serves all book state from the canonical parquet
    # plane (the SQL snapshot/delta replays were retired in the clean cut).
    replay_source: str = ""
    # Reproducibility fingerprint: a content-hashed manifest of the exact
    # parquet files the run pinned (path/size/mtime/rows/span). Two runs of the
    # same window produce the same content_hash iff the underlying data is
    # byte-identical, so a re-run can detect data drift (e.g. pruning). Wired
    # from MarketDataView.dataset_snapshot(). See services.marketdata.manifest.
    dataset_snapshot: dict[str, Any] = field(default_factory=dict)
    fill_probability_model: dict[str, Any] = field(default_factory=dict)
    # How the strategy discovered the opportunities driving this run.
    # One of:
    #   - "live_opps"            — only OpportunityHistory rows (legacy fast path)
    #   - "historical_synthesis" — only replay-discovery (zero live opps in window)
    #   - "hybrid"               — both live + replay-discovered, deduped
    # The default is hybrid: the strategy's discovery pipeline runs
    # against recorded data AND we cache off live opps when present.
    discovery_mode: str = "live_opps"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _backtest_evaluate_opportunity(
    *,
    strategy: Any,
    opp: Any,
    pdata: dict[str, Any],
    initial_capital_usd: float,
    live_context: dict[str, Any] | None = None,
) -> tuple[Any, Any] | None:
    """Run ``strategy.evaluate()`` on a backtest opportunity row.

    Mirrors the live orchestrator's gate at trader_orchestrator_worker
    line 6474 — the strategy's own ``evaluate()`` decides whether the
    intent should fire RIGHT NOW given current portfolio + market
    context.

    When ``opp`` carries a real TradeSignal at ``_underlying_signal``
    (the canonical replay path), wrap it in the production
    ``RuntimeTradeSignalView`` so evaluate() reads the EXACT same view
    live does — including the ``live_edge_percent`` /
    ``live_selected_price`` overlay derived from market context.
    The ``live_context`` argument is the historically-reconstructed
    market context for this signal at its detected_at time (built by
    ``_build_replay_live_context``); without it, signals from worker-
    driven pipelines (which often persist with null edge_percent) get
    rejected by evaluate's edge-floor check even when their LIVE
    counterparts were selected, because live's runtime overlays the
    edge from current mid.

    Returns ``(decision_obj, signal_view)`` for the caller to hand to
    ``apply_platform_decision_gates``.  Returns None if evaluate()
    raised — fall back to "passthrough" so a strategy bug doesn't tank
    the whole backtest.
    """
    if not hasattr(strategy, "evaluate"):
        return None
    try:
        # Institutional-grade replay path: when the opp carries a
        # real TradeSignal ORM row, use the production
        # RuntimeTradeSignalView with the historically-reconstructed
        # live_context.  This is the same wrapper class the live
        # worker uses — same column access, same overlay semantics,
        # same evaluate input.  No synthesis required.
        underlying = getattr(opp, "_underlying_signal", None)
        if underlying is not None:
            from services.trader_orchestrator.live_market_context import (
                RuntimeTradeSignalView,
            )

            signal = RuntimeTradeSignalView(
                underlying, live_context=live_context or {}
            )
            # Time-sensitive evaluate gates (resolution_window,
            # signal_staleness, days_to_resolution) compute against
            # ``datetime.now(utc)``.  In replay we're evaluating signals
            # from days ago — their original resolution_date is now in
            # the past relative to wall clock, so every dtr-based check
            # rejects.  Synthesize a forward-shifted resolution_date
            # such that ``(synthetic_res - now)`` reproduces the dtr the
            # signal had at its original detect-time.  This is the same
            # trick the legacy ``_SignalView`` path used; we replicate
            # it on the production wrapper so evaluate sees byte-
            # equivalent time semantics.
            try:
                detected_at_orig = getattr(underlying, "created_at", None)
                if isinstance(detected_at_orig, datetime):
                    if detected_at_orig.tzinfo is None:
                        detected_at_orig = detected_at_orig.replace(tzinfo=timezone.utc)
                    base_payload = signal.payload_json or {}
                    if not isinstance(base_payload, dict):
                        base_payload = {}
                    original_res = base_payload.get("resolution_date")
                    if isinstance(original_res, str):
                        try:
                            from datetime import datetime as _dt
                            res_parsed = _dt.fromisoformat(original_res.replace("Z", "+00:00"))
                            if res_parsed.tzinfo is None:
                                res_parsed = res_parsed.replace(tzinfo=timezone.utc)
                            original_dtr_seconds = (res_parsed - detected_at_orig).total_seconds()
                            if original_dtr_seconds > 0:
                                synthetic_res = (
                                    datetime.now(timezone.utc)
                                    + (res_parsed - detected_at_orig)
                                )
                                # Build a patched payload (don't mutate
                                # the ORM row's underlying dict — other
                                # paths may share the reference).
                                signal.payload_json = {
                                    **base_payload,
                                    "resolution_date": synthetic_res.isoformat(),
                                }
                        except Exception:
                            pass
            except Exception:
                pass
            ctx_lm = (
                dict(live_context) if isinstance(live_context, dict) else {}
            )
            ctx: dict[str, Any] = {
                "params": dict(getattr(strategy, "config", {}) or {}),
                "trader": {
                    "id": "backtest",
                    "mode": "shadow",
                    "risk_limits": {
                        "max_trade_notional_usd": float(initial_capital_usd) * 0.10,
                        "max_open_positions": 50,
                    },
                },
                "mode": "shadow",
                "live_market": ctx_lm,
                "source_config": {},
            }
            decision = strategy.evaluate(signal, ctx)
            if hasattr(decision, "decision"):
                return (decision, signal)
            if isinstance(decision, dict):
                class _DecisionView:
                    __slots__ = ("decision", "reason", "score", "size_usd", "checks", "payload")

                    def __init__(self, d: dict[str, Any]):
                        self.decision = str(d.get("decision") or "selected")
                        self.reason = str(d.get("reason") or "")
                        self.score = d.get("score")
                        self.size_usd = float(d.get("size_usd") or 0.0)
                        self.checks = list(d.get("checks") or [])
                        self.payload = dict(d.get("payload") or {})
                return (_DecisionView(decision), signal)
            return None

        # ── Legacy path (OpportunityHistory rows without an underlying
        # TradeSignal ORM): synthesize a TradeSignal-quack object.
        # Used by the discovery synthesis flow which builds opps from
        # detect() output rather than persisted signals.  Here we
        # don't have a real ORM row to wrap, so we reconstruct the
        # TradeSignal contract field by field from the opp's
        # positions_data.
        opp_strategy_type = str(getattr(opp, "strategy_type", "") or "").strip().lower()
        first_pos = (pdata.get("positions_to_take") or [{}])[0]
        if not isinstance(first_pos, dict):
            first_pos = {}
        opp_strategy_type = str(getattr(opp, "strategy_type", "") or "").strip().lower()
        first_pos = (pdata.get("positions_to_take") or [{}])[0]
        if not isinstance(first_pos, dict):
            first_pos = {}

        # Source is per-strategy.  BaseStrategy declares it as a class
        # attribute (``source_key`` ∈ {scanner, crypto, news, weather,
        # traders, manual}); each strategy subclass picks the right
        # data pipeline.  Hard-coding "scanner" for every strategy
        # was wrong — crypto / news / traders strategies would all
        # fail the ``signal.source`` gate that lives in their
        # evaluate() (e.g., btc_eth_directional_edge requires
        # source=crypto).  Read from the loaded strategy class so the
        # backtest mirrors live source-routing exactly.
        strategy_source = str(
            getattr(strategy, "source_key", None)
            or getattr(strategy.__class__, "source_key", None)
            or "scanner"
        ).strip().lower()

        # Build an enriched payload that mirrors the live TradeSignal's
        # payload_json contract.  Strategies fall back to
        # ``payload.get("strategy_type")`` / ``payload.get("strategy")``
        # when ``signal.strategy_type`` is missing — make sure both work.
        enriched_payload = dict(pdata)
        enriched_payload.setdefault("strategy_type", opp_strategy_type)
        enriched_payload.setdefault("strategy", opp_strategy_type)
        enriched_payload.setdefault("source", strategy_source)
        # Surface market_question / market_id at top level too — some
        # strategies inspect those for keyword-block filters.
        if "market_id" not in enriched_payload and first_pos.get("market_id"):
            enriched_payload["market_id"] = first_pos["market_id"]
        if "market_question" not in enriched_payload and first_pos.get("market_question"):
            enriched_payload["market_question"] = first_pos["market_question"]

        # Resolution date — utils/signal_helpers.days_to_resolution()
        # reads ``payload.resolution_date`` and computes against
        # ``datetime.now()``.  In backtest the opp was detected days
        # ago; using the real ``resolution_date`` would give a
        # negative DTR (past resolution) and reject every opp on the
        # resolution-window gate.  Reconstruct a synthetic
        # resolution_date such that ``(resolution_date - now)`` equals
        # the ORIGINAL detect-time DTR.  This makes evaluate()'s DTR
        # computation produce the same number it would have at
        # detect-time, which is the right semantic for backtest replay.
        from datetime import timedelta as _td_eval
        tail_block = first_pos.get("_tail_end") if isinstance(first_pos.get("_tail_end"), dict) else {}
        original_dtr = tail_block.get("days_to_resolution") if isinstance(tail_block, dict) else None
        if isinstance(original_dtr, (int, float)) and original_dtr > 0:
            synthetic_resolution = (
                datetime.now(timezone.utc) + _td_eval(seconds=float(original_dtr) * 86400.0)
            )
            enriched_payload["resolution_date"] = synthetic_resolution.isoformat()
        elif "resolution_date" not in enriched_payload:
            # Fall back to the OpportunityHistory column if the
            # _tail_end block didn't carry it.
            opp_res = getattr(opp, "resolution_date", None)
            if opp_res is not None:
                enriched_payload["resolution_date"] = (
                    opp_res.isoformat() if hasattr(opp_res, "isoformat") else str(opp_res)
                )

        # Liquidity: OpportunityHistory doesn't store the dollar
        # liquidity, but the strategy ALREADY verified it passed its
        # min_liquidity floor at detect-time — that's why the opp is
        # in the history at all.  ``_tail_end.liquidity_ok`` (or any
        # equivalent flag in the position) records that gate's verdict.
        # Use a high default that satisfies any reasonable floor when
        # liquidity_ok was True; let it stay at zero (and re-fail) only
        # if the original detect explicitly rejected it.  This mirrors
        # the live re-evaluate semantics: between detect and execute
        # is microseconds in backtest, so any check that detect passed
        # should still pass at execute-time absent a market move.
        tail_block = first_pos.get("_tail_end") if isinstance(first_pos.get("_tail_end"), dict) else {}
        liq_ok_flag = bool(tail_block.get("liquidity_ok", True))
        # Provide a generous synthetic liquidity number — strategy
        # configs cap their min_liquidity around $1k-$10k; 1M ensures
        # we don't double-fail a check detect already passed.
        synthetic_liquidity = 1_000_000.0 if liq_ok_flag else 0.0

        class _SignalView:
            def __init__(self, opp_obj: Any, pdata_obj: dict[str, Any], enriched: dict[str, Any]):
                self.id = str(getattr(opp_obj, "id", "") or "")
                # The TradeSignal contract uses ``strategy_type``, not
                # ``strategy_key`` — strategies read the former.
                self.strategy_type = opp_strategy_type
                self.strategy_key = opp_strategy_type  # alias for back-compat
                self.source = strategy_source
                self.signal_type = "trade"
                # Edge: prefer expected_roi (already a percent).
                self.edge_percent = float(getattr(opp_obj, "expected_roi", 0) or 0)
                conf_raw = pdata_obj.get("confidence")
                self.confidence = (
                    float(conf_raw) if isinstance(conf_raw, (int, float)) else 0.5
                )
                self.direction = str(
                    first_pos.get("action") or first_pos.get("side") or "BUY"
                ).upper()
                self.entry_price = float(first_pos.get("price") or 0.5)
                self.effective_price = self.entry_price
                self.market_id = str(first_pos.get("market_id") or "")
                self.market_question = str(first_pos.get("market_question") or "")
                self.token_id = str(first_pos.get("token_id") or "")
                self.liquidity = synthetic_liquidity
                self.risk_score = float(getattr(opp_obj, "risk_score", 0) or 0)
                # Both names exist on TradeSignal — set both.
                self.payload_json = enriched
                self.strategy_context_json = pdata_obj.get("strategy_context") or {}
                self.strategy_context = self.strategy_context_json
                self.status = "pending"

        signal = _SignalView(opp, pdata, enriched_payload)

        # Minimal EvaluateContext.  ``params`` are the strategy's
        # default config (we already loaded it via StrategyLoader so
        # strategy.config carries the merged config).  ``trader``
        # gets a minimal stand-in with risk_limits derived from the
        # backtest's portfolio cap; ``mode`` is "shadow" to mirror
        # what the simulator does.
        ctx: dict[str, Any] = {
            "params": dict(getattr(strategy, "config", {}) or {}),
            "trader": {
                "id": "backtest",
                "mode": "shadow",
                "risk_limits": {
                    "max_trade_notional_usd": float(initial_capital_usd) * 0.10,
                    "max_open_positions": 50,
                },
            },
            "mode": "shadow",
            "live_market": {
                "best_bid": signal.entry_price,
                "best_ask": signal.entry_price,
                "mid": signal.entry_price,
            },
            "source_config": {},
        }
        decision = strategy.evaluate(signal, ctx)
        # Normalize to a StrategyDecision-like object so the caller can
        # feed it into apply_platform_decision_gates (which reads
        # ``.decision`` / ``.reason`` / ``.score`` / ``.size_usd``
        # attributes, not dict keys).  When the strategy returned a
        # plain dict (back-compat shape), wrap it in a tiny shim with
        # the same attribute interface; we don't want to materialize
        # a real StrategyDecision dataclass here because some legacy
        # strategies return a dict that *omits* the ``checks`` field.
        if hasattr(decision, "decision"):
            return (decision, signal)
        if isinstance(decision, dict):
            class _DecisionView:
                __slots__ = ("decision", "reason", "score", "size_usd", "checks", "payload")

                def __init__(self, d: dict[str, Any]):
                    self.decision = str(d.get("decision") or "selected")
                    self.reason = str(d.get("reason") or "")
                    self.score = d.get("score")
                    self.size_usd = float(d.get("size_usd") or 0.0)
                    self.checks = list(d.get("checks") or [])
                    self.payload = dict(d.get("payload") or {})
            return (_DecisionView(decision), signal)
    except Exception as exc:
        logger.debug("backtest evaluate() raised — passthrough: %s", exc)
        return None
    return None


def _exec_ci_to_dict(metric: Any) -> dict[str, Any]:
    return {
        "value": float(getattr(metric, "value", 0.0) or 0.0),
        "ci_low": (
            float(getattr(metric, "ci_low", None))
            if getattr(metric, "ci_low", None) is not None
            else None
        ),
        "ci_high": (
            float(getattr(metric, "ci_high", None))
            if getattr(metric, "ci_high", None) is not None
            else None
        ),
    }


# ── Historical discovery replay ──────────────────────────────────────────
#
# Runs strategy.detect_async against historical market state at sampled
# time intervals across the window, returning synthetic
# OpportunityHistory-shaped rows for the existing evaluate / gate /
# matcher pipeline.  This is what "backtest" actually means — re-run
# the strategy's discovery pipeline against recorded data, not just
# replay fill simulation against opps live happened to surface.


class _SyntheticOpp:
    """OpportunityHistory-quack object built from strategy.detect output.

    The existing evaluate path inspects: ``strategy_type``,
    ``detected_at``, ``positions_data`` (with ``positions_to_take``).
    We populate exactly those, plus a ``_synthetic`` marker so
    downstream code can downweight if needed.
    """

    __slots__ = ("id", "stable_id", "strategy", "strategy_type",
                 "detected_at", "positions_data",
                 "title", "event_id", "_synthetic")

    # Class-level sequence counter so multiple opps emitted at the SAME
    # tick on the SAME market (e.g. UP + DOWN sides of a binary, or
    # repeated buys with different size/price within one detect call)
    # get unique ids.  Without this the matcher aborts the entire run
    # at the second submit with ``ValueError: duplicate order_id``,
    # producing 0 fills despite hundreds of valid intents.
    _SEQ = 0

    def __init__(
        self,
        *,
        strategy_type: str,
        detected_at: datetime,
        positions_data: dict[str, Any],
        title: str = "",
        event_id: str | None = None,
    ) -> None:
        self.strategy_type = strategy_type
        # ``strategy`` mirrors ``strategy_type`` — the matcher reads
        # ``opp.strategy`` when building ``TradeIntent.strategy_slug``
        # (see ``strategy_backtester:4730``); without it the slug
        # falls back to the run slug and downstream attribution
        # silently loses the strategy origin.
        self.strategy = strategy_type
        self.detected_at = detected_at
        self.positions_data = positions_data
        self.title = title
        self.event_id = event_id
        self._synthetic = True
        # The matcher builds intents via ``f"opp_{opp.id}_{idx}"`` and
        # ``str(opp.id)`` so synthetic opps need a UNIQUE id per emit.
        # Seed: strategy/tick/title/event ARE NOT ENOUGH (two opps at
        # the same tick on the same market for different sides have
        # the same seed) — fingerprint the first position + add a
        # per-process sequence counter for guaranteed uniqueness.
        import hashlib as _hashlib
        pos_fp = ""
        try:
            ptt = positions_data.get("positions_to_take") if isinstance(positions_data, dict) else None
            if isinstance(ptt, list) and ptt:
                p0 = ptt[0]
                if isinstance(p0, dict):
                    pos_fp = (
                        f"|{p0.get('token_id','')}|{p0.get('side','')}"
                        f"|{p0.get('outcome','')}|{p0.get('price','')}"
                        f"|{p0.get('size_usd','')}"
                    )
        except Exception:
            pos_fp = ""
        _SyntheticOpp._SEQ += 1
        seed = (
            f"{strategy_type}|"
            f"{detected_at.isoformat() if detected_at else ''}|"
            f"{title}|{event_id or ''}{pos_fp}|{_SyntheticOpp._SEQ}"
        )
        digest = _hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        self.id = f"syn_{digest}"
        self.stable_id = self.id


# ── Replay event sourcing ─────────────────────────────────────────────────
#
# Backtest discovery used to hard-code ``events=[]`` when calling
# ``strategy.detect_async``.  That made event-driven strategies — copy
# trading, news, insider detection — silently produce zero opportunities
# in replay, even when their live counterparts had been firing all week.
# The fix: for strategies whose ``detect`` reads ``events``, materialise
# the same historical event stream the live system saw and feed it in
# at the right tick.  The router below picks a replay path purely from
# what the strategy DECLARES it consumes — its ``subscriptions`` list
# (and, as a legacy compat, ``accepted_signal_strategy_types``).  No
# strategy-slug hardcoding, so user-created strategies route correctly.
#
# Adding a new event-driven backtest path:
#   1. Implement a per-tick loader/projector (e.g. _replay_bus_events_into_
#      tick_grid handles the recorded_event_bus side for any topic).
#   2. Add the bus topic to ``_BUS_TOPIC_EVENT_KIND`` so the detect loop
#      knows which event-driven branch to take.

# Bus topic -> event_kind label used by ``_replay_discover_opportunities``
# to pick the right detect-loop branch.  Any strategy whose subscriptions
# (or accepted_signal_strategy_types) resolve to one of these topics
# routes through the corresponding event-driven path; everything else
# falls through to the book-driven path.
def _replay_event_kind_for_strategy(slug: str, strategy: Any) -> str | None:
    """Pick a detect-loop branch for the given strategy.

    Checks (in order):
    1. ``subscriptions`` — the strategy's declared EventType list.
    2. ``accepted_signal_strategy_types`` — strategies that copy-trade without
       declaring explicit subscriptions still route to the wallet-event branch.
    3. ``slug`` — final fallback for strategies whose slug is the canonical
       copy-trade identifier.

    Returns ``None`` (book-driven) when none of the above resolve to a known
    event-driven branch.
    """
    from services.data_events import EventType

    subs = {str(s) for s in (getattr(strategy, "subscriptions", None) or [])}
    if EventType.CRYPTO_UPDATE in subs:
        return "crypto_update"
    if EventType.TRADER_ACTIVITY in subs:
        return "wallet_trade"
    if EventType.MARKET_DATA_REFRESH in subs:
        return "scanner_tick"

    accepted = getattr(strategy, "accepted_signal_strategy_types", None) or []
    if "traders_copy_trade" in accepted:
        return "wallet_trade"

    if slug.lower() == "traders_copy_trade":
        return "wallet_trade"

    return None


def _extract_scope_wallets(strategy: Any) -> set[str] | None:
    """Pull the wallet scope out of a copy-trade strategy's config so we
    can filter ``WalletMonitorEvent`` to the wallets this strategy
    cares about.  Returns ``None`` when no explicit individual-wallet
    scope is configured (caller should fall back to all monitored
    wallets — the table itself is naturally scope-limited because the
    ws-monitor only persists events for tracked wallets).
    """
    cfg = getattr(strategy, "config", {}) or {}
    if not isinstance(cfg, dict):
        return None
    scope = cfg.get("traders_scope")
    if not isinstance(scope, dict):
        return None
    wallets = scope.get("individual_wallets")
    if not isinstance(wallets, list) or not wallets:
        return None
    out: set[str] = set()
    for w in wallets:
        s = str(w or "").strip().lower()
        if s:
            out.add(s)
    return out or None


# Hard cap on wallet events loaded into a single replay.  Sized for the
# ws-monitor's typical 7-day volume (~10k events across the tracked
# wallet set) with headroom; if a window legitimately holds more we
# warn rather than silently truncate.
_REPLAY_WALLET_EVENT_CAP = 50000


class _SignalAsOpp:
    """Wrap a ``trade_signals`` row as an OpportunityHistory-quack object
    for the execution-backtest loop.  TradeSignal is the canonical
    record of what the live trader saw and decided on; the backtest
    loop adapter exposes the ORM object plus the loop's expected
    attribute surface (``id`` / ``strategy_type`` / ``detected_at`` /
    ``positions_data``).

    The original ORM row is preserved at ``_underlying_signal`` so the
    evaluate path can build a ``RuntimeTradeSignalView`` from it —
    same class live uses, ensuring evaluate() sees byte-identical
    column access.  Without this, backtest would have to synthesise a
    duplicate signal-shim that drifts from the live contract.
    """

    __slots__ = (
        "id", "strategy_type", "detected_at", "positions_data",
        "expected_roi", "risk_score", "title", "event_id",
        "_synthetic_source", "_underlying_signal",
    )

    def __init__(
        self,
        *,
        sid: str,
        strategy_type: str,
        detected_at: datetime,
        positions_data: dict[str, Any],
        expected_roi: float = 0.0,
        risk_score: float = 0.0,
        title: str = "",
        event_id: str | None = None,
        underlying_signal: Any = None,
    ) -> None:
        self.id = sid
        self.strategy_type = strategy_type
        self.detected_at = detected_at
        self.positions_data = positions_data
        self.expected_roi = expected_roi
        self.risk_score = risk_score
        self.title = title
        self.event_id = event_id
        self._synthetic_source = "trade_signals"
        self._underlying_signal = underlying_signal


async def _load_opps_from_trade_signals(
    *,
    session: Any,
    slug: str,
    start_dt: datetime,
    end_dt: datetime,
    max_rows: int,
) -> list[_SignalAsOpp]:
    """Load trade_signals rows for ``slug`` in window and shape them
    into OpportunityHistory-quack objects.  Used when opp_history has
    no rows for a strategy (worker-driven flows like crypto_entropy_maker).
    Filters to live-status signals only — ``filtered`` / ``failed`` /
    ``skipped`` would have failed evaluate or risk gates in live and
    would distort the backtest funnel by inflating ``opps_pulled``.
    """
    from sqlalchemy import select as _select
    # The TradeSignal ORM model lives in models.database too.  Import
    # locally to avoid circular import at module load.
    from models.database import TradeSignal as _TS
    import json as _json

    stmt = (
        _select(_TS)
        .where(
            _TS.created_at >= start_dt,
            _TS.created_at <= end_dt,
            _TS.strategy_type == slug,
            # Signals the live trader actually considered actionable.
            # ``expired`` is the dominant terminal status for many
            # strategies (queue TTL ran out before execute) and IS a
            # legitimate replay candidate — the backtest's matcher
            # decides whether the resting order would have filled in
            # the historical book.  Drop only the explicit-failure
            # statuses where the live trader rejected the signal
            # itself (filtered by quality-filter, failed during
            # evaluate, skipped by deduplication).
            _TS.status.notin_(["filtered", "failed", "skipped"]),
        )
        .order_by(_TS.created_at.asc())
        .limit(max(1, int(max_rows)))
    )
    try:
        rows = (await session.execute(stmt)).scalars().all()
    except Exception:
        return []

    out: list[_SignalAsOpp] = []
    for sig in rows:
        payload = sig.payload_json
        if isinstance(payload, str):
            try:
                payload = _json.loads(payload)
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        ptt = payload.get("positions_to_take") or []
        if not isinstance(ptt, list) or not ptt:
            # Reconstruct a minimal single-position from top-level
            # signal fields so worker-driven strategies that didn't
            # serialize positions still produce a usable opp.
            tok = (
                str(payload.get("selected_token_id") or "").strip()
                or str(payload.get("token_id") or "").strip()
                or str(getattr(sig, "market_id", "") or "").strip()
            )
            side = "BUY"
            direction = str(getattr(sig, "direction", "") or "").strip().upper()
            if direction in {"SELL", "SHORT"}:
                side = "SELL"
            entry = float(getattr(sig, "entry_price", 0.0) or 0.0)
            if not tok or entry <= 0.0:
                continue
            ptt = [
                {
                    "token_id": tok,
                    "side": side,
                    "action": side,
                    "price": entry,
                    "market_id": str(getattr(sig, "market_id", "") or ""),
                    "market_question": str(getattr(sig, "market_question", "") or ""),
                }
            ]

        positions_data = dict(payload)
        positions_data["positions_to_take"] = ptt
        # Ensure strategy_context lands in positions_data the same way
        # OpportunityHistory rows carry it.
        sc = getattr(sig, "strategy_context_json", None)
        if isinstance(sc, dict) and sc and "strategy_context" not in positions_data:
            positions_data["strategy_context"] = sc

        out.append(
            _SignalAsOpp(
                sid=str(getattr(sig, "id", "") or ""),
                strategy_type=str(getattr(sig, "strategy_type", "") or slug),
                detected_at=sig.created_at,
                positions_data=positions_data,
                expected_roi=float(getattr(sig, "edge_percent", 0.0) or 0.0),
                title=str(payload.get("title") or ""),
                event_id=str(payload.get("event_id") or "") or None,
                underlying_signal=sig,
            )
        )
    return out


async def _build_replay_live_context(
    *,
    signal: Any,
    pdata: dict[str, Any],
    book_replay: Any,
    detected_at: datetime,
) -> dict[str, Any]:
    """Reconstruct the ``live_context`` shape the live worker builds
    when it pulls a signal off the queue.  ``RuntimeTradeSignalView``
    overlays the context's ``live_selected_price`` and
    ``live_edge_percent`` on top of the persisted signal — without
    that overlay, evaluate() reads ``signal.edge_percent`` which is
    often null on signals from worker-driven pipelines (the live
    runtime computes it fresh from current mid).

    Replay needs the same overlay sourced from HISTORICAL book state:
    look up the mid for the signal's selected token at the signal's
    detected_at via the book replay (snapshots OR delta-replay,
    whichever the run is using), compute live_edge from
    model_probability, and shape the dict identically.
    """
    from services.trader_orchestrator.live_market_context import (
        _extract_model_probability,
    )

    # Determine the selected token for this signal.
    first_pos = (pdata.get("positions_to_take") or [{}])[0]
    if not isinstance(first_pos, dict):
        first_pos = {}
    selected_token = (
        str(pdata.get("selected_token_id") or "").strip()
        or str(first_pos.get("token_id") or "").strip()
        or str(getattr(signal, "market_id", "") or "").strip()
    )
    direction = str(getattr(signal, "direction", "") or "").strip().lower()

    selected_live: float | None = None
    if selected_token and book_replay is not None:
        try:
            snap = await book_replay.snapshot_at(token_id=selected_token, ts=detected_at)
        except Exception:
            snap = None
        if snap is not None:
            mid = snap.mid
            if mid is None:
                mid = snap.best_bid or snap.best_ask
            if mid is not None and mid > 0.0:
                selected_live = float(mid)

    # Model probability: derived from the persisted signal (live's
    # ``_extract_model_probability`` reads the same fields).
    model_probability = _extract_model_probability(signal, direction=direction)
    live_edge = None
    if model_probability is not None and selected_live is not None:
        live_edge = (model_probability - selected_live) * 100.0

    payload = getattr(signal, "payload_json", None)
    if isinstance(payload, str):
        try:
            import json as _json
            payload = _json.loads(payload)
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    seconds_left: float | None = None
    synthetic_end_time: str | None = None
    for key in ("end_time", "resolution_date", "event_end_time", "market_end_time"):
        raw_end = payload.get(key) or pdata.get(key)
        if not raw_end:
            continue
        try:
            parsed_end = datetime.fromisoformat(str(raw_end).replace("Z", "+00:00"))
        except Exception:
            continue
        if parsed_end.tzinfo is None:
            parsed_end = parsed_end.replace(tzinfo=timezone.utc)
        det = detected_at if detected_at.tzinfo is not None else detected_at.replace(tzinfo=timezone.utc)
        ttl = (parsed_end - det).total_seconds()
        if ttl > 0.0:
            seconds_left = float(ttl)
            synthetic_end_time = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
            break
    is_live = bool(seconds_left is not None and seconds_left > 0.0)

    return {
        "available": bool(selected_live is not None),
        "market_id": str(getattr(signal, "market_id", "") or ""),
        "direction": direction,
        "selected_token_id": selected_token,
        "live_selected_price": selected_live,
        "live_edge_percent": live_edge,
        "model_probability": model_probability,
        "signal_entry_price": float(getattr(signal, "entry_price", 0.0) or 0.0),
        "seconds_left": seconds_left,
        "is_live": is_live,
        "is_current": is_live,
        "end_time": synthetic_end_time,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "live_market_fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def _load_wallet_events_for_replay(
    *,
    session: Any,
    start_dt: datetime,
    end_dt: datetime,
    scope_wallets: set[str] | None,
) -> tuple[list[Any], bool]:
    """Load ``WalletMonitorEvent`` rows in window, optionally scoped to
    a set of wallet addresses.  Returns ``(rows, truncated)`` — the
    ``truncated`` flag is True iff the cap was hit.
    """
    from sqlalchemy import select as _select
    from services.wallet_ws_monitor import WalletMonitorEvent

    stmt = (
        _select(WalletMonitorEvent)
        .where(
            WalletMonitorEvent.detected_at >= start_dt,
            WalletMonitorEvent.detected_at <= end_dt,
        )
        .order_by(WalletMonitorEvent.detected_at.asc())
        .limit(_REPLAY_WALLET_EVENT_CAP + 1)
    )
    if scope_wallets:
        stmt = stmt.where(WalletMonitorEvent.wallet_address.in_(list(scope_wallets)))
    rows = list((await session.execute(stmt)).scalars().all())
    truncated = len(rows) > _REPLAY_WALLET_EVENT_CAP
    if truncated:
        rows = rows[:_REPLAY_WALLET_EVENT_CAP]
    return rows, truncated


def _build_token_to_market_lookup(catalog_markets: list[Any]) -> dict[str, dict[str, Any]]:
    """Walk the catalog once and produce a ``token_id → market_payload``
    map matching the shape ``TradersCopyTradeSignalService`` builds via
    ``_resolve_market_snapshot``.  Replay can resolve markets from the
    catalog without an API call.
    """
    import json as _json

    out: dict[str, dict[str, Any]] = {}
    for m in catalog_markets or []:
        if not isinstance(m, dict):
            continue
        market_id = str(m.get("condition_id") or m.get("id") or "").strip()
        question = str(m.get("question") or "").strip()
        slug = str(m.get("event_slug") or m.get("slug") or "").strip() or None
        liquidity_raw = m.get("liquidity")
        try:
            liquidity = float(liquidity_raw) if liquidity_raw is not None else None
        except (TypeError, ValueError):
            liquidity = None

        token_ids_raw = m.get("clob_token_ids")
        if isinstance(token_ids_raw, str):
            try:
                token_ids_raw = _json.loads(token_ids_raw)
            except (_json.JSONDecodeError, TypeError):
                token_ids_raw = []
        outcomes_raw = m.get("outcomes")
        if isinstance(outcomes_raw, str):
            try:
                outcomes_raw = _json.loads(outcomes_raw)
            except (_json.JSONDecodeError, TypeError):
                outcomes_raw = []

        token_ids_list = [str(t).strip() for t in (token_ids_raw or []) if t]
        outcomes_list = [str(o).strip() for o in (outcomes_raw or [])]

        for idx, token_id in enumerate(token_ids_list):
            if not token_id:
                continue
            outcome = outcomes_list[idx] if idx < len(outcomes_list) else ""
            out[token_id] = {
                "market_id": market_id or f"token:{token_id}",
                "market_question": question or f"Token {token_id}",
                "market_slug": slug,
                "outcome": outcome,
                "liquidity": liquidity,
                "token_id": token_id,
            }
    return out


async def _replay_bus_events_into_tick_grid(
    *,
    strategy: Any,
    start_dt: datetime,
    ticks: list[datetime],
    actual_interval: float,
    n_ticks: int,
    events_by_tick: list[list[Any]],
    candidate_token_ids: list[str] | None,
    catalog_markets: list[Any],
) -> int:
    """Stream events for ``strategy.subscriptions`` through the
    recorded-event bus and bin them into ``events_by_tick`` for the
    discovery loop.

    For ``crypto.update.dispatch`` envelopes we synthesise the same
    ``DataEvent(event_type=CRYPTO_UPDATE, ...)`` shape the live
    market_runtime emits, so the strategy's detect_async iterates a
    drop-in replacement for the live event stream.

    For other topics (future news.gdelt.* / external.telonex.*) the
    bridge can grow per-topic projectors here as strategies start
    subscribing to them.  The bridge is the one place that knows how
    to translate "bus envelope" → "strategy-shaped event payload."

    Returns the number of bus envelopes binned.  Best-effort: any
    failure is logged and swallowed (the existing wallet_trade /
    book-driven paths keep working).
    """
    try:
        from services.recorded_event_bus.backtest_bridge import (
            replay_events_for_strategy, resolve_subscriptions_to_topics,
        )
        import services.recorded_event_bus.storage  # noqa: F401 -- attach
        from services.data_events import DataEvent, EventType
    except Exception:
        return 0

    subs = getattr(strategy, "subscriptions", None) or []
    topics = resolve_subscriptions_to_topics(subs)
    if not topics:
        return 0

    # Topic-specific projector: bus envelope → the shape strategy
    # detect() expects in ``events`` list.  Each entry reconstructs
    # the exact DataEvent shape the live event_dispatcher would have
    # delivered, so detect() can't tell live from replay.
    def _project(envelope: Any) -> Any | None:
        t = envelope.topic
        ts = datetime.fromtimestamp(envelope.observed_at_us / 1_000_000, tz=timezone.utc)
        if t == "crypto.update.dispatch":
            return DataEvent(
                event_type=EventType.CRYPTO_UPDATE,
                source=envelope.source or "market_runtime",
                timestamp=ts,
                payload=dict(envelope.payload or {}),
            )
        if t == "polymarket.catalog.snapshot":
            # Recorded catalog snapshot — reconstruct a MARKET_DATA_REFRESH
            # DataEvent carrying the markets/events the live catalog
            # refresh published.  The detect loop carries this snapshot
            # forward across ticks until a newer snapshot replaces it, and
            # augments per-token prices from the parquet book grid.
            p = dict(envelope.payload or {})
            return DataEvent(
                event_type=EventType.MARKET_DATA_REFRESH,
                source=envelope.source or "scanner",
                timestamp=ts,
                payload={
                    "updated_at": p.get("updated_at"),
                    "duration_seconds": p.get("duration_seconds"),
                    "error": p.get("error"),
                },
                markets=list(p.get("markets") or []) or None,
                events=list(p.get("events") or []) or None,
            )
        if t == "news.update":
            return DataEvent(
                event_type=EventType.NEWS_UPDATE,
                source=envelope.source or "news_worker",
                timestamp=ts,
                payload=dict(envelope.payload or {}),
            )
        if t == "weather.update":
            return DataEvent(
                event_type=EventType.WEATHER_UPDATE,
                source=envelope.source or "weather_worker",
                timestamp=ts,
                payload=dict(envelope.payload or {}),
            )
        if t == "trader.activity":
            return DataEvent(
                event_type=EventType.TRADER_ACTIVITY,
                source=envelope.source or "tracked_traders_worker",
                timestamp=ts,
                payload=dict(envelope.payload or {}),
            )
        if t == "polymarket.trade.execution":
            ev = DataEvent(
                event_type=EventType.TRADE_EXECUTION,
                source=envelope.source or "polymarket_ws",
                timestamp=ts,
                payload=dict(envelope.payload or {}),
            )
            # TRADE_EXECUTION on live carries token_id as an attribute,
            # not in payload — match that shape so vpin_toxicity's
            # ``event.token_id`` access works in backtest too.
            try:
                ev.token_id = envelope.entity_id
            except Exception:
                pass
            return ev
        # ``wallet.trade`` is also routed through this bridge for new
        # code, but the existing wallet_trade special case in
        # ``_replay_discover_opportunities`` covers it.  Skip here so
        # we don't double-bin.
        if t == "wallet.trade":
            return None
        # Default: return the envelope as-is.  Strategies subscribing
        # to new topics declare what shape they want; until they exist
        # we just pass the envelope through.
        return envelope

    n_binned = 0
    real_crypto_market_keys: set[str] = set()
    async for envelope in replay_events_for_strategy(
        strategy=strategy,
        start_dt=start_dt,
        end_dt=ticks[-1] + (ticks[-1] - ticks[-2] if len(ticks) >= 2 else timedelta(seconds=actual_interval)),
        entity_filter=None,
    ):
        shaped = _project(envelope)
        if shaped is None:
            continue
        # Bin into the tick whose right edge is closest >= envelope time.
        ev_ts = datetime.fromtimestamp(envelope.observed_at_us / 1_000_000, tz=timezone.utc)
        offset = (ev_ts - start_dt).total_seconds()
        if offset < 0:
            continue
        idx = min(n_ticks - 1, int(offset // max(actual_interval, 1)))
        events_by_tick[idx].append(shaped)
        n_binned += 1
        # Track which crypto markets have REAL recorded dispatch coverage in
        # this window so the synthesizer below only GAP-FILLS the rest —
        # recorded data stays authoritative wherever it exists.
        if envelope.topic == "crypto.update.dispatch":
            payload = getattr(envelope, "payload", None) or {}
            for m in (payload.get("markets") or []):
                if isinstance(m, dict):
                    k = str(m.get("condition_id") or m.get("id") or m.get("slug") or "")
                    if k:
                        real_crypto_market_keys.add(k)

    # ── Imported-parquet gap-fill for event-driven crypto strategies ──
    #
    # Operator-imported historical book data (e.g. the polybacktest
    # provider) lands as canonical SNAPSHOT_SCHEMA parquet but has no
    # recorded crypto.update.dispatch envelopes, so an event-driven crypto
    # strategy would never fire on those markets in backtest.  The unified
    # market-data layer reconstructs the dispatch events from that parquet +
    # the self-describing ProviderDataset metadata, via point-in-time book
    # access (services.marketdata.projection) — gap-filling only markets
    # WITHOUT real recorded coverage so recorded data stays authoritative.
    if "crypto.update.dispatch" in topics:
        try:
            from services.marketdata.projection import project_crypto_update_events

            window_end = ticks[-1] + (
                ticks[-1] - ticks[-2] if len(ticks) >= 2 else timedelta(seconds=actual_interval)
            )
            proj_events, proj_stats = await project_crypto_update_events(
                start=ticks[0],
                end=window_end,
                cadence_seconds=max(1.0, float(actual_interval)),
                token_scope=candidate_token_ids,
                exclude_market_keys=real_crypto_market_keys,
            )
            for ev in proj_events:
                offset = (ev.timestamp - start_dt).total_seconds()
                if offset < 0:
                    continue
                idx = min(n_ticks - 1, int(offset // max(actual_interval, 1)))
                events_by_tick[idx].append(ev)
                n_binned += 1
            if proj_stats.get("events"):
                logger.info(
                    "marketdata.projection: gap-filled %d crypto_update events from "
                    "%d imported markets",
                    proj_stats.get("events"),
                    proj_stats.get("markets_active"),
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "marketdata.projection: imported-parquet crypto_update projection failed",
                exc_info=True,
            )

    return n_binned


def _wallet_event_to_strategy_input(
    event: Any, *, market_payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Shape one ``WalletMonitorEvent`` row into the dict the
    ``traders_copy_trade`` strategy iterates in detect().  Mirrors
    ``TradersCopyTradeSignalService._process_wallet_trade_event``
    so the strategy receives the same payload it would in live.
    """
    side = str(getattr(event, "side", "") or "").strip().upper()
    if side not in {"BUY", "SELL"}:
        return None
    token_id = str(getattr(event, "token_id", "") or "").strip()
    if not token_id:
        return None
    try:
        entry_price = float(getattr(event, "price", 0.0) or 0.0)
        size = float(getattr(event, "size", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if entry_price <= 0.0 or size <= 0.0:
        return None
    source_wallet = str(getattr(event, "wallet_address", "") or "").strip().lower()

    detected_at = getattr(event, "detected_at", None)
    if isinstance(detected_at, datetime) and detected_at.tzinfo is None:
        detected_at = detected_at.replace(tzinfo=timezone.utc)
    detected_iso = detected_at.isoformat() if isinstance(detected_at, datetime) else None

    tx_hash = str(getattr(event, "tx_hash", "") or "")
    order_hash = str(getattr(event, "order_hash", "") or "")
    log_index = int(getattr(event, "log_index", 0) or 0)
    block_number = int(getattr(event, "block_number", 0) or 0)
    latency_ms = float(getattr(event, "detection_latency_ms", 0.0) or 0.0)

    copy_event_payload = {
        "wallet_address": source_wallet,
        "token_id": token_id,
        "side": side,
        "size": size,
        "price": entry_price,
        "tx_hash": tx_hash,
        "order_hash": order_hash,
        "log_index": log_index,
        "block_number": block_number,
        "timestamp": detected_iso,
        "detected_at": detected_iso,
        "latency_ms": latency_ms,
        "confidence": 0.70,
    }
    source_trade_payload = {
        "wallet_address": source_wallet,
        "side": side,
        "source_notional_usd": entry_price * size,
        "size": size,
        "price": entry_price,
        "tx_hash": tx_hash,
        "order_hash": order_hash,
        "log_index": log_index,
        "detected_at": detected_iso,
    }
    source_item_id = (
        f"{tx_hash}:{source_wallet}:{token_id}:{side}:{log_index}:{order_hash}"
    )
    return {
        "copy_event": copy_event_payload,
        "source_trade": source_trade_payload,
        "market": dict(market_payload),
        "source_item_id": source_item_id,
        "dedupe_key": "",
    }


# ── Per-tick price grid (book-driven strategies) ─────────────────────────
#
# The discovery loop needs ``prices_at_tick = {token_id: {best_bid,
# best_ask, mid, ...}}`` for each tick.  Two replay sources can build
# that grid:
#
#   * ``BookReplay``     — reads ``market_microstructure_snapshots``
#     (throttled to 0.5s/token by the live ingestor; sparse on calm
#     markets).
#   * ``BookDeltaReplay`` — reads ``book_delta_events`` and replays them
#     atop a snapshot anchor, so it surfaces every level change the
#     live system saw.
#
# When deltas dominate the window the matcher already auto-selects
# ``BookDeltaReplay``; we mirror that selection here so discovery sees
# the same data the matcher will fill against.  The streaming approach
# below visits each replay snapshot once and freezes per-token state
# at every tick boundary, bounding memory at O(tokens × ticks).


async def _build_per_tick_prices_grid(
    *,
    token_ids: list[str],
    ticks: list[datetime],
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, list[dict[str, Any] | None]]:
    """For each token, return per-tick price state lists (length =
    len(ticks)).  Each entry is ``{best_bid, best_ask, mid, price,
    spread_bps, observed_at}`` or ``None`` when no state was available
    at-or-before that tick.

    Builds a unified :class:`MarketDataView` over the token universe, streams
    its ordered book stream ONCE, and bins into ticks at boundary crossings.
    Source selection (which provider's parquet covers which token) is resolved
    inside the view, so operator-imported (telonex / polybacktest) and
    live-recorded data are all visible to discovery through the one
    point-in-time access path.
    """
    if not token_ids or not ticks:
        return {}
    from services.marketdata.view import MarketDataView
    from services.marketdata.book_source import MarketDataViewSource

    view = await MarketDataView.build(token_ids=token_ids, start=start_dt, end=end_dt)
    replay = MarketDataViewSource(view, token_ids=token_ids)

    cur_state: dict[str, dict[str, Any]] = {}
    grid: dict[str, list[dict[str, Any] | None]] = {}
    next_tick_idx = 0

    def _freeze_tick(tick_idx: int) -> None:
        for tid, st in cur_state.items():
            bucket = grid.get(tid)
            if bucket is None:
                bucket = [None] * len(ticks)
                grid[tid] = bucket
            bucket[tick_idx] = dict(st)

    async for snap in replay.iter_snapshots():
        bb = snap.best_bid or 0.0
        ba = snap.best_ask or 0.0
        if bb <= 0 and ba <= 0:
            continue

        observed = snap.observed_at
        if observed is not None and observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        # Freeze any tick whose boundary is in the past relative to this
        # snap — the strategy at tick ``i`` sees state observed_at <=
        # ticks[i], so we capture state BEFORE applying snaps that
        # arrive after the boundary.
        while (
            next_tick_idx < len(ticks)
            and observed is not None
            and ticks[next_tick_idx] < observed
        ):
            _freeze_tick(next_tick_idx)
            next_tick_idx += 1

        mid = (bb + ba) / 2.0 if bb > 0 and ba > 0 else (bb or ba)
        cur_state[snap.token_id] = {
            "best_bid": bb,
            "best_ask": ba,
            "mid": mid,
            "price": mid,
            "spread_bps": snap.spread_bps,
            "observed_at": observed,
        }

    while next_tick_idx < len(ticks):
        _freeze_tick(next_tick_idx)
        next_tick_idx += 1

    return grid


async def _replay_discover_opportunities(
    *,
    strategy: Any,
    slug: str,
    start_dt: datetime,
    end_dt: datetime,
    sample_interval_seconds: int,
    max_ticks: int,
    candidate_token_ids: list[str] | None = None,
    warmup_seconds: int = 0,
) -> list[_SyntheticOpp]:
    """Replay-discovery: walk historical market state at sampled time
    ticks across [start_dt, end_dt] and call strategy.detect_async at
    each tick, accumulating returned opportunities.

    Inputs reconstructed at each tick:
      * markets — current Polymarket market catalog filtered to
        active-during-window markets, optionally narrowed to
        ``candidate_token_ids`` when caller provides a scope (book-
        driven strategies).  Event-driven strategies bypass that
        narrowing — events drive the universe.
      * prices — best_bid / best_ask / mid reconstructed per token from the
        unified ``MarketDataView`` (canonical parquet, point-in-time), which
        resolves provider coverage internally.
      * events — for strategies whose ``detect`` reads events
        (``traders_copy_trade``, etc.), the same historical event
        stream the live system saw, loaded from the appropriate source
        table and binned by tick so each detect() call sees the
        events that arrived since the previous tick.  For book-only
        strategies this remains an empty list.

    Returns a list of ``_SyntheticOpp`` instances that quack like
    ``OpportunityHistory`` rows — the existing evaluate /
    orchestrator-gate / matcher pipeline consumes them unchanged.
    """
    import json as _json
    from sqlalchemy import text as _text
    from models.database import BacktestAsyncSessionLocal as _Sess

    if (
        not _has_custom_detect_async(strategy)
        and not _has_custom_detect_sync(strategy)
        and not _has_custom_detect_plain(strategy)
    ):
        # Strategy uses none of the three detect methods — base class
        # default ``detect()`` returns []; historical replay has nothing
        # to do.  Hold-only / scheduler-driven strategies fall here.
        return []

    event_kind = _replay_event_kind_for_strategy(slug, strategy)

    # Step 1: build the time grid.  Cap at ``max_ticks`` total samples
    # so a 30-day window doesn't blow up into 1500 detect() calls.
    #
    # Pre-roll warmup: when ``warmup_seconds`` > 0 we prepend whole-
    # interval ticks BEFORE the measured window so rolling windows /
    # CycleTrackers / isolated PersistentState build up from the recorded
    # stream — matching the state live had at window-start instead of a
    # cold start.  Detect runs on warm-up ticks (to build state) but their
    # opportunities are NOT emitted.  ``start_dt`` is rebound to the
    # pre-roll start so all downstream index math (events binning, grid
    # build) stays consistent with tick 0; ``_emit_start`` marks the real
    # window boundary for the emit gate.
    from datetime import timedelta as _td_replay
    _emit_start = start_dt
    measured_seconds = max(60.0, (end_dt - _emit_start).total_seconds())
    n_measured = min(max_ticks, max(1, int(measured_seconds / max(60, sample_interval_seconds))))
    actual_interval = measured_seconds / n_measured
    n_warm = (
        int(max(0, int(warmup_seconds)) // actual_interval)
        if warmup_seconds and actual_interval > 0
        else 0
    )
    start_dt = _emit_start - _td_replay(seconds=actual_interval * n_warm)
    n_ticks = n_measured + n_warm
    ticks: list[datetime] = [
        start_dt + _td_replay(seconds=actual_interval * i) for i in range(n_ticks)
    ]

    # Step 2: load the current market catalog from the live scanner
    # (in-memory cache, no API call).  For book-driven strategies we
    # narrow to markets with at least one token in scope.  For event-
    # driven strategies the candidate filter is bypassed — events drive
    # the universe, and the catalog is used only for token→market
    # lookups.
    try:
        from services.shared_state import _read_market_catalog_file
        catalog = _read_market_catalog_file()
    except Exception:
        catalog = None

    candidate_set: set[str] | None = (
        set(candidate_token_ids) if (candidate_token_ids and event_kind is None) else None
    )

    catalog_markets: list[Any] = []
    if catalog is not None:
        _events_in_catalog, _markets_in_catalog, _meta = catalog
        for m in _markets_in_catalog or []:
            if not isinstance(m, dict):
                continue
            if m.get("closed") or m.get("archived") or m.get("resolved"):
                continue
            if m.get("active") is False:
                continue
            tok_ids = m.get("clob_token_ids") or []
            if isinstance(tok_ids, str):
                try:
                    tok_ids = _json.loads(tok_ids)
                except (_json.JSONDecodeError, TypeError):
                    tok_ids = []
            tok_ids = [str(t).strip() for t in (tok_ids or []) if t]
            if not tok_ids:
                continue
            if candidate_set is not None and not any(t in candidate_set for t in tok_ids):
                continue
            catalog_markets.append(m)

    if not catalog_markets and event_kind not in ("crypto_update", "scanner_tick"):
        # Without a catalog we can't shape event payloads / look up
        # markets-by-token (the ``market`` field is required for
        # wallet_trade etc.).  Bail out quietly for those.
        #
        # crypto_update / scanner_tick / other bus-driven kinds are
        # exempt: each replayed envelope carries its own markets array
        # (crypto.update.dispatch -> oracle/top-of-book per market;
        # scanner.tick -> full catalog snapshot the live scanner pushed),
        # so the strategy's detect() reads from the event payload — the
        # current-only catalog file is not the right source for a
        # historical replay anyway.
        return []

    # Step 3: derive the token universe from the catalog.
    all_token_ids: list[str] = []
    seen_t: set[str] = set()
    for m in catalog_markets:
        for t in m.get("clob_token_ids") or []:
            ts = str(t).strip()
            if ts and ts not in seen_t:
                seen_t.add(ts)
                all_token_ids.append(ts)

    # Step 4: build the per-tick prices grid.  Book-driven strategies
    # require this; event-driven ones don't read prices in detect()
    # (the strategy reads the live mid from the embedded ``market``
    # payload), so we skip the work for them.
    grid: dict[str, list[dict[str, Any] | None]] = {}
    if event_kind is None and all_token_ids:
        # Book-driven discovery reads per-tick prices from the unified
        # MarketDataView (canonical parquet, point-in-time), built inside the
        # grid helper. The view resolves coverage across every provider, so
        # imported (telonex/polybacktest) and live-recorded tokens are all
        # visible to detect().
        try:
            grid = await _build_per_tick_prices_grid(
                token_ids=all_token_ids,
                ticks=ticks,
                start_dt=start_dt,
                end_dt=end_dt,
            )
        except Exception as exc:
            logger.warning("replay_discover: price grid build failed: %s", exc)
            grid = {}

    # Step 5: build the per-tick events grid.  Each tick's slice covers
    # events whose timestamp lands in (prev_tick, this_tick] — same
    # delivery cadence the live scanner sees.
    events_by_tick: list[list[Any]] = [[] for _ in range(n_ticks)]
    wallet_event_truncated = False

    # ── Recorded-event bus replay path ─────────────────────────────
    #
    # For strategies whose subscriptions resolve to bus topics
    # (crypto.update.dispatch, future news.gdelt.* / external.telonex.*,
    # etc.) the bus is the unified replay surface.  Closes the
    # un-backtestable-strategy gap that previously left the entire
    # btc_eth_* / crypto_* family with events=[].
    #
    # Falls back to a no-op on errors so the existing wallet_trade
    # special case + book-driven path keep working unchanged.
    try:
        bus_event_count = await _replay_bus_events_into_tick_grid(
            strategy=strategy,
            start_dt=start_dt,
            ticks=ticks,
            actual_interval=actual_interval,
            n_ticks=n_ticks,
            events_by_tick=events_by_tick,
            candidate_token_ids=candidate_token_ids,
            catalog_markets=catalog_markets,
        )
        if bus_event_count:
            logger.info(
                "replay_discover: bus replay binned %d events across %d ticks",
                bus_event_count, n_ticks,
            )
    except Exception:  # noqa: BLE001
        logger.warning("replay_discover: bus event replay failed", exc_info=True)

    if event_kind == "wallet_trade":
        market_lookup = _build_token_to_market_lookup(catalog_markets)
        scope_wallets = _extract_scope_wallets(strategy)
        async with _Sess() as ev_session:
            await ev_session.execute(_text("SET statement_timeout = 60000"))
            try:
                wme_rows, wallet_event_truncated = await _load_wallet_events_for_replay(
                    session=ev_session,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    scope_wallets=scope_wallets,
                )
            except Exception as exc:
                logger.warning("replay_discover: wallet event load failed: %s", exc)
                wme_rows = []
        for ev in wme_rows:
            ev_ts = getattr(ev, "detected_at", None)
            if not isinstance(ev_ts, datetime):
                continue
            if ev_ts.tzinfo is None:
                ev_ts = ev_ts.replace(tzinfo=timezone.utc)
            offset = (ev_ts - start_dt).total_seconds()
            if offset < 0:
                continue
            idx = min(n_ticks - 1, int(offset // actual_interval))
            token_id = str(getattr(ev, "token_id", "") or "").strip()
            market_payload = market_lookup.get(token_id)
            if not market_payload:
                # Token not in current catalog — same outcome live would
                # produce: ``_resolve_market_snapshot`` would fail and
                # ``_process_wallet_trade_event`` would skip the event.
                continue
            shaped = _wallet_event_to_strategy_input(ev, market_payload=market_payload)
            if shaped is not None:
                events_by_tick[idx].append(shaped)

    # Step 6: walk the time grid + run detect at each tick.
    detected_total: list[_SyntheticOpp] = []
    detect_failures = 0

    # Carry-forward catalog state for scanner_tick (MARKET_DATA_REFRESH)
    # replay.  Catalog snapshots arrive at refresh cadence (~ minutes);
    # backtest ticks fire every sample_interval_seconds.  Maintain the
    # latest-seen markets across ticks so every tick sees the most recent
    # snapshot, exactly the way live strategies do (the live catalog is
    # a shared cache — its state on tick T+1 is whatever the last refresh
    # left it in, even when no refresh happened between T and T+1).
    carry_catalog_markets: dict[str, dict] = {}

    for tick_i in range(n_ticks):
        tick_t = ticks[tick_i]

        # Prices at this tick — built from the streaming grid for book-
        # driven strategies, empty for event-driven ones.
        prices_at_tick: dict[str, dict[str, Any]] = {}
        if grid:
            for token_id, states in grid.items():
                if tick_i < len(states):
                    st = states[tick_i]
                    if st is not None:
                        prices_at_tick[token_id] = st

        events_at_tick = events_by_tick[tick_i]

        # Decide which markets to surface to detect().  Event-driven
        # strategies receive the catalog narrowed to the markets their
        # events actually touch this tick (the strategy then reads the
        # ``market`` payload from the event itself, but a non-empty
        # markets list keeps any defensive ``if not markets`` guards
        # in strategy code happy).  Book-driven strategies see markets
        # whose tokens have reconstructable prices.
        if event_kind == "crypto_update":
            # CRYPTO_UPDATE DataEvents carry the full markets array in
            # their payload — time-correct top-of-book + oracle + price-
            # to-beat, recorded from the live dispatcher.  Surface those
            # markets directly to detect(); ignore catalog_markets (the
            # current-only catalog file is not the right source for a
            # historical replay).  Dedupe by market id across events in
            # this tick — last-seen wins, matching live "latest dispatch".
            if not events_at_tick:
                continue
            markets_by_id: dict[str, dict] = {}
            for ev in events_at_tick:
                payload = getattr(ev, "payload", None) or (ev if isinstance(ev, dict) else {})
                if not isinstance(payload, dict):
                    continue
                for m in (payload.get("markets") or []):
                    if isinstance(m, dict):
                        key = str(m.get("id") or m.get("condition_id") or m.get("slug") or "")
                        if key:
                            markets_by_id[key] = m
            markets_at_tick: list[dict] = list(markets_by_id.values())
        elif event_kind == "scanner_tick":
            # Recorded catalog-snapshot replay — each
            # polymarket.catalog.snapshot envelope projects to a
            # MARKET_DATA_REFRESH DataEvent carrying the full catalog at
            # refresh time.  Carry that catalog forward across ticks until
            # a fresher snapshot arrives (refresh cadence is minutes;
            # tick cadence is seconds).  Per-tick prices come from the
            # parquet book grid (live_ingestor) — we augment each market
            # dict with the current best_bid/best_ask so the strategy
            # sees fresh top-of-book on every tick.
            for ev in events_at_tick:
                for m in (getattr(ev, "markets", None) or []):
                    if isinstance(m, dict):
                        key = str(m.get("id") or m.get("condition_id") or m.get("slug") or "")
                        if key:
                            carry_catalog_markets[key] = m
            if not carry_catalog_markets:
                # Pre-recording window — no catalog snapshot seen yet.
                continue
            # Build markets_at_tick from the carried catalog, augmenting
            # each market's top-of-book from the per-tick price grid when
            # the parquet has coverage.  Markets without any token in the
            # grid still pass through (their last-known book in the
            # catalog payload is used) — same as live behaviour when the
            # WS feed is silent for a token.
            markets_at_tick = []
            for m in carry_catalog_markets.values():
                tok_ids = [str(t).strip() for t in (m.get("clob_token_ids") or []) if t]
                augmented = m
                for tid in tok_ids:
                    px = prices_at_tick.get(tid)
                    if not isinstance(px, dict):
                        continue
                    # Polymarket up/down: best_bid/ask on UP-side book is
                    # the market-level price surface; copy the freshest
                    # snapshot we have for this token into the market.
                    augmented = dict(augmented)
                    if px.get("best_bid") is not None:
                        augmented["best_bid"] = px["best_bid"]
                    if px.get("best_ask") is not None:
                        augmented["best_ask"] = px["best_ask"]
                    if px.get("spread") is not None:
                        augmented["spread"] = px["spread"]
                    break  # one token's price is enough at the market level
                markets_at_tick.append(augmented)
        elif event_kind is not None:
            if not events_at_tick:
                continue
            event_token_ids: set[str] = set()
            for item in events_at_tick:
                if isinstance(item, dict):
                    market_meta = item.get("market") or {}
                    if isinstance(market_meta, dict):
                        tid = str(market_meta.get("token_id") or "").strip()
                        if tid:
                            event_token_ids.add(tid)
            markets_at_tick = []
            for m in catalog_markets:
                tok_ids = [str(t).strip() for t in (m.get("clob_token_ids") or []) if t]
                if any(t in event_token_ids for t in tok_ids):
                    markets_at_tick.append(m)
        else:
            if not prices_at_tick:
                continue
            markets_at_tick = []
            for m in catalog_markets:
                tok_ids = [str(t).strip() for t in (m.get("clob_token_ids") or []) if t]
                if any(t in prices_at_tick for t in tok_ids):
                    markets_at_tick.append(m)
            if not markets_at_tick:
                continue

        # Call strategy.detect_async with the reconstructed inputs.
        # Wrap dict-shaped catalog markets into Market pydantic models
        # because that's what strategies expect (verified — every
        # detect_async signature in the repo annotates ``markets:
        # list[Market]``).
        try:
            from models.market import Market as _Market
            market_models: list[Any] = []
            for m in markets_at_tick:
                try:
                    market_models.append(_Market.from_gamma_response(m))
                except Exception:
                    continue
        except Exception:
            market_models = list(markets_at_tick)

        try:
            opps_at_tick = await _run_detect_once(
                strategy,
                events=events_at_tick,
                markets=market_models,
                prices=prices_at_tick,
                timeout_seconds=8.0,
                now_us=int(tick_t.timestamp() * 1_000_000),
            )
        except Exception:
            detect_failures += 1
            continue

        # Warm-up tick: detect ran (state built up) but we don't emit its
        # opportunities — only the measured window counts.
        if tick_t < _emit_start:
            continue

        for opp in opps_at_tick or []:
            # ``opp`` is whatever the strategy's detect returns — usually
            # an Opportunity-like object with ``positions_to_take`` /
            # ``total_cost`` / ``expected_roi`` etc.  Wrap in a synthetic
            # OpportunityHistory-shaped record.
            pdata = _opp_to_positions_data(opp)
            if not pdata.get("positions_to_take"):
                continue
            detected_total.append(
                _SyntheticOpp(
                    strategy_type=slug,
                    detected_at=tick_t,
                    positions_data=pdata,
                    title=str(getattr(opp, "title", "") or ""),
                    event_id=str(getattr(opp, "event_id", "") or "") or None,
                )
            )

    if detect_failures > 0:
        logger.info(
            "replay_discover: %d detect() failures across %d ticks",
            detect_failures, n_ticks,
        )
    if wallet_event_truncated:
        logger.warning(
            "replay_discover: wallet event load truncated at cap=%d — widen the "
            "time window granularity or raise the cap if this becomes load-bearing",
            _REPLAY_WALLET_EVENT_CAP,
        )

    return detected_total


def _opp_to_positions_data(opp: Any) -> dict[str, Any]:
    """Convert a strategy.detect() return value into the OpportunityHistory
    ``positions_data`` shape: ``{"positions_to_take": [{...}, ...]}``.

    Tolerates several common return shapes:
      * Pydantic Opportunity with ``positions_to_take`` field
      * Dict with ``positions_to_take`` key
      * Bare list of position dicts
      * Single position dict
    """
    if isinstance(opp, dict):
        pdata = dict(opp)
        if "positions_to_take" not in pdata:
            # Treat the dict itself as a single position.
            if any(k in pdata for k in ("token_id", "side", "action")):
                return {"positions_to_take": [pdata]}
            return {}
        return pdata
    pos_list = getattr(opp, "positions_to_take", None)
    if isinstance(pos_list, list):
        # Coerce each entry to a dict.
        out: list[dict[str, Any]] = []
        for p in pos_list:
            if isinstance(p, dict):
                out.append(p)
            elif hasattr(p, "model_dump"):
                out.append(p.model_dump())
            elif hasattr(p, "__dict__"):
                out.append({k: v for k, v in p.__dict__.items() if not k.startswith("_")})
        return {
            "positions_to_take": out,
            "total_cost": float(getattr(opp, "total_cost", 0.0) or 0.0),
            "expected_roi": float(getattr(opp, "expected_roi", 0.0) or 0.0),
            "risk_score": float(getattr(opp, "risk_score", 0.0) or 0.0),
        }
    return {}


# ── Pre-flight data-coverage measurement ─────────────────────────────────
#
# Backtests are only as good as the historical data they replay.  Live
# trading writes every L2 delta to ``book_delta_events`` (3-4M rows/wk
# in steady state); the standalone microstructure recorder writes full
# snapshots to ``market_microstructure_snapshots``.  Operators who never
# ran the recorder discover this only after seeing inexplicable "0
# trades" outcomes when live had real fills in the same window.
#
# This helper measures coverage in BOTH tables and produces a fidelity
# rating + actionable recommendation.  The matching engine only reads
# from market_microstructure_snapshots today (Phase 1) — surfacing
# delta coverage tells the operator that even though the snapshot
# table is sparse, the data DOES exist and a backfill / Phase-2
# delta-replay would unlock it.

_FIDELITY_HIGH_SNAPS_PER_HOUR = 30.0   # ~1 every 2 min — strategies that
                                       # rest GTC limits will see the book
                                       # cross them often enough.
_FIDELITY_MEDIUM_SNAPS_PER_HOUR = 6.0  # ~1 every 10 min — taker-mode
                                       # strategies still work; passive
                                       # rests get sparse fill data.


async def _measure_data_coverage(
    *,
    session: Any,
    opp_tokens: list[str],
    start_dt: datetime,
    end_dt: datetime,
    event_kind: str | None = None,
    scope_wallets: set[str] | None = None,
) -> dict[str, Any]:
    """Compute per-window data-coverage stats for the opp_tokens universe.

    Returns a dict suitable for ``ExecutionBacktestResult.data_coverage``.
    Cheap — two chunked aggregate queries (snapshots, deltas), plus an
    optional event-source query when ``event_kind`` is set.  Failure is
    non-fatal; we return a stub dict with ``error`` set so the caller
    surfaces a single warning rather than aborting the run.

    For event-driven strategies (``event_kind="wallet_trade"``), the
    fidelity rating is anchored on event count rather than snapshot
    density — the live system fires opps on events, so "0 events in
    window" is the right zero, not "0 snapshots/hr".
    """
    from sqlalchemy import select, func as sa_func

    coverage: dict[str, Any] = {
        "opp_tokens": len(opp_tokens),
        "tokens_with_snapshots": 0,
        "tokens_with_deltas": 0,
        "snapshots_total": 0,
        "deltas_total": 0,
        "median_snaps_per_token_per_hour": 0.0,
        "p10_snaps_per_token_per_hour": 0.0,
        "median_deltas_per_token_per_hour": 0.0,
        "event_kind": event_kind,
        "events_total": 0,
        "events_per_hour": 0.0,
        "fidelity_rating": "none",
        "recommended_action": "",
    }
    if (not opp_tokens and not event_kind) or end_dt <= start_dt:
        coverage["recommended_action"] = "No opp tokens in window — nothing to measure."
        return coverage

    window_hours = max((end_dt - start_dt).total_seconds() / 3600.0, 1e-6)

    # Event-source coverage (event-driven strategies only).  Cheap one-
    # query count of the relevant event table; the recommendation logic
    # below uses this when ``event_kind`` is set.
    if event_kind == "wallet_trade":
        try:
            from services.wallet_ws_monitor import WalletMonitorEvent

            ev_stmt = (
                select(sa_func.count(WalletMonitorEvent.id))
                .where(
                    WalletMonitorEvent.detected_at >= start_dt,
                    WalletMonitorEvent.detected_at <= end_dt,
                )
            )
            if scope_wallets:
                ev_stmt = ev_stmt.where(
                    WalletMonitorEvent.wallet_address.in_(list(scope_wallets))
                )
            events_total = int(
                (await session.execute(ev_stmt)).scalar_one() or 0
            )
            coverage["events_total"] = events_total
            coverage["events_per_hour"] = events_total / window_hours
        except Exception as exc:
            coverage["error"] = (
                (coverage.get("error") + " | " if coverage.get("error") else "")
                + f"event count query failed: {exc}"
            )

    # ── Book coverage from canonical parquet (via the unified layer) ──
    # Recording lives in parquet; "deltas" as a separate SQL stream is
    # retired (the canonical SNAPSHOT_SCHEMA carries full L2 depth). Count
    # snapshot rows per covered token from the parquet footers (cheap — no
    # data scan) so the operator still sees per-token density.
    snaps_per_token: dict[str, int] = {}
    if opp_tokens:
        try:
            import pyarrow.parquet as _pq
            from services.marketdata.coverage import resolve_coverage
            cov_map = await resolve_coverage(token_ids=opp_tokens, start=start_dt, end=end_dt)
            for tok in cov_map.covered_tokens:
                rows = 0
                for f in cov_map.files_for(tok):
                    try:
                        rows += int(_pq.read_metadata(f).num_rows)
                    except Exception:
                        pass
                if rows:
                    snaps_per_token[str(tok)] = rows
        except Exception as exc:
            coverage["error"] = f"parquet coverage query failed: {exc}"

    coverage["tokens_with_snapshots"] = len(snaps_per_token)
    coverage["tokens_with_deltas"] = 0
    coverage["snapshots_total"] = sum(snaps_per_token.values())
    coverage["deltas_total"] = 0
    coverage["snaps_per_token"] = dict(snaps_per_token)
    coverage["deltas_per_token"] = {}
    coverage["window_hours"] = float(window_hours)

    # Per-token fidelity: covered tokens carry canonical L2 snapshots
    # ("parquet"); uncovered tokens have no book ("none").
    per_token_fidelity: dict[str, str] = {
        tok: ("parquet" if snaps_per_token.get(tok, 0) > 0 else "none")
        for tok in opp_tokens
    }
    coverage["per_token_fidelity"] = per_token_fidelity
    coverage["snapshot_approx_tokens"] = []
    coverage["delta_exact_token_count"] = 0
    coverage["snapshot_approx_token_count"] = 0

    rates_snaps = sorted(snaps_per_token.get(t, 0) / window_hours for t in opp_tokens)

    def _percentile(sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = max(0, min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1))))
        return float(sorted_vals[idx])

    coverage["median_snaps_per_token_per_hour"] = _percentile(rates_snaps, 0.5)
    coverage["p10_snaps_per_token_per_hour"] = _percentile(rates_snaps, 0.1)
    coverage["median_deltas_per_token_per_hour"] = 0.0

    # Fidelity rating.  Event-driven strategies anchor on the event
    # source — snapshot density is irrelevant when the strategy doesn't
    # discover via books.  Book-driven strategies use snapshot density
    # as the historical default.
    if event_kind == "wallet_trade":
        events_total = int(coverage["events_total"])
        # Thresholds tuned against typical copy-trading workloads:
        # >= 50 events / 7 days is "high" (multiple intents per day).
        if events_total >= 50:
            coverage["fidelity_rating"] = "high"
        elif events_total >= 10:
            coverage["fidelity_rating"] = "medium"
        elif events_total > 0:
            coverage["fidelity_rating"] = "low"
        else:
            coverage["fidelity_rating"] = "none"
    else:
        median_rate = coverage["median_snaps_per_token_per_hour"]
        if median_rate >= _FIDELITY_HIGH_SNAPS_PER_HOUR:
            coverage["fidelity_rating"] = "high"
        elif median_rate >= _FIDELITY_MEDIUM_SNAPS_PER_HOUR:
            coverage["fidelity_rating"] = "medium"
        elif median_rate > 0:
            coverage["fidelity_rating"] = "low"
        else:
            coverage["fidelity_rating"] = "none"

    # Recommendations.
    if event_kind == "wallet_trade":
        events_total = int(coverage["events_total"])
        events_per_hour = float(coverage["events_per_hour"])
        scope_label = (
            f"{len(scope_wallets)} scoped wallets" if scope_wallets
            else "all monitored wallets"
        )
        if events_total == 0:
            coverage["recommended_action"] = (
                "No wallet trade events in window for this strategy's scope "
                f"({scope_label}).  Either the tracked wallets weren't trading "
                "in the chosen period, or wallet monitoring wasn't running.  "
                "Check Data Lab → Traders → Wallet monitor for capture status."
            )
        elif events_total < 10:
            coverage["recommended_action"] = (
                f"Sparse events: {events_total} wallet trade(s) in window "
                f"({events_per_hour:.2f}/hr, {scope_label}).  Backtest will "
                "fire only a handful of intents — not statistically meaningful.  "
                "Widen the time window or broaden the wallet scope for more samples."
            )
        else:
            coverage["recommended_action"] = (
                f"Event coverage OK: {events_total} wallet trade(s) in window "
                f"({events_per_hour:.2f}/hr, {scope_label}). ✓"
            )
        return coverage

    # Book-driven recommendations — most useful when fidelity is bad
    # but data is available in deltas (i.e. the live system has been
    # ingesting but the snapshot table is sparse for the chosen window).
    median_rate = coverage["median_snaps_per_token_per_hour"]
    has_delta_coverage = (
        coverage["tokens_with_deltas"] > 0.5 * len(opp_tokens)
        and coverage["median_deltas_per_token_per_hour"] >= 5.0
    )
    if coverage["fidelity_rating"] in ("low", "none"):
        if has_delta_coverage:
            # The auto-source-selection logic in run_execution_backtest
            # will already have picked BookDeltaReplay for this run, so
            # this message is informational rather than actionable.
            coverage["recommended_action"] = (
                "Snapshot table sparse, BUT book_delta_events has dense coverage "
                f"({coverage['tokens_with_deltas']}/{len(opp_tokens)} tokens, "
                f"median {coverage['median_deltas_per_token_per_hour']:.0f} deltas/hr). "
                "The engine auto-switched to live-parity delta replay — see the "
                "replay-source pill on the result.  No action required."
            )
        else:
            coverage["recommended_action"] = (
                f"Sparse data: median {median_rate:.1f} snapshots/token/hr "
                f"(target ≥{_FIDELITY_HIGH_SNAPS_PER_HOUR:.0f}/hr for high fidelity), "
                f"and book_delta_events is ALSO sparse for this window.  This means "
                f"the live ingestor wasn't capturing these markets during the "
                f"backtest window — go to Data Lab → Record → Proactive coverage "
                f"and confirm the WS subscription cap covers your strategy's opp "
                f"universe.  Options: (1) widen WS coverage so future runs have "
                f"data, (2) backfill historical mids via Data Lab → Providers "
                f"(synthetic single-level book from Polymarket REST), (3) import "
                f"full L2 from polybacktest.com if you have a key (BTC/ETH/SOL only)."
            )
    elif coverage["fidelity_rating"] == "medium":
        coverage["recommended_action"] = (
            f"Medium fidelity: median {median_rate:.1f} snapshots/token/hr. "
            "Taker-mode strategies will replay accurately; passive resting "
            "GTC limits may underfill.  If book_delta_events has denser "
            "coverage, the engine will auto-select the delta-replay path."
        )
    else:  # high
        coverage["recommended_action"] = (
            f"High fidelity: median {median_rate:.1f} snapshots/token/hr. ✓"
        )

    return coverage


async def run_execution_backtest(
    source_code: str,
    slug: str = "_backtest_exec",
    config: Optional[dict[str, Any]] = None,
    *,
    token_ids: Optional[list[str]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    initial_capital_usd: float = 1000.0,
    # Lifted from 1000 → 25k.  Event-market strategies fan out across
    # hundreds of low-volume markets; a 7-day window legitimately
    # produces 5-10k opportunities for slugs like ``stat_arb`` and
    # ``holding_reward_yield``.  Capping at 1000 silently dropped
    # half their opps before the matching engine ever saw them.
    # 25k is the upper bound the matcher can chew through in a
    # reasonable wall-clock budget; operators can pass higher.
    max_intents: int = 25000,
    submit_latency_p50_ms: float = 350.0,
    submit_latency_p95_ms: float = 900.0,
    cancel_latency_p50_ms: float = 200.0,
    cancel_latency_p95_ms: float = 600.0,
    seed: int = 42,
    fills_sample_size: int = 200,
    equity_sample_size: int = 500,
    bootstrap_resamples: int = 2000,
    impact_strength_bps: float = 0.0,
    impact_capacity_threshold: float = 0.5,
    impact_capacity_exponent: float = 1.5,
    maker_rebate_bps: float = 0.0,
    maker_rebate_max_spread_bps: float = 50.0,
    latency_correlation_window_ms: float = 5.0,
    # Optional progress hook for the worker-process job runner.  Fired
    # by ``BacktestEngine.run`` every ~1k snapshots; the runner uses
    # it to update the BacktestRun row's progress + message so the UI
    # can render a live progress bar.  Sync callers leave it at None
    # and the engine treats that as "no callback".
    progress_callback: Any = None,
    # ── Historical discovery synthesis (additive to TradeSignal replay) ──
    # Two complementary signal sources flow into evaluate / gates /
    # matcher, deduped at the (token, 30-min bucket) level:
    #
    #   1. TradeSignal replay (always on) — every signal the live
    #      trader emitted in the window.  Faithful "would my CURRENT
    #      strategy code accept what live actually saw".  Param
    #      tweaks to evaluate-time gates show up here immediately.
    #
    #   2. Discovery synthesis (this flag, default True) — re-runs the
    #      strategy's detect() against historical book state at
    #      sampled ticks, surfacing opps the live trader DIDN'T see.
    #      Required for testing detect-time param changes (the
    #      historical TradeSignals were filtered by the OLD config;
    #      tightening / loosening detect's filters has no effect on
    #      the replay path because those signals don't exist in the
    #      DB).  Set False to scope the backtest strictly to what
    #      live did.
    discover_from_history: bool = True,
    # Time grid resolution for historical discovery.  The default
    # 30-min cadence is a good balance: tighter than 30 min gives
    # diminishing returns (most strategies' discovery filters change
    # state on similar time scales), wider misses fast-moving
    # opportunities.  Capped at 96 ticks per window to bound runtime
    # (each tick = one strategy.detect_async call).
    discovery_sample_interval_seconds: int = 1800,
    discovery_max_ticks: int = 96,
) -> ExecutionBacktestResult:
    """Execution-realistic backtest using full L2 replay + bootstrap CIs.

    Loads the strategy, fetches book snapshots from
    ``MarketMicrostructureSnapshot``, runs strategy.detect_async at
    sampled time intervals across the window (so discovery itself is
    backtested, not just fill simulation), generates trade intents,
    runs the production matching engine, and reports headline + risk-
    adjusted metrics with bootstrap CIs.
    """
    from datetime import timedelta as _td
    from services.backtest import (
        BacktestConfig,
        BacktestEngine,
        LatencyModel,
        LatencyProfile,
        PortfolioConfig,
        TradeIntent,
    )
    from services.backtest.matching_engine import FeeModel, ImpactModel
    from services.trader_orchestrator.decision_gates import (
        apply_platform_decision_gates,
        is_within_trading_schedule_utc,
    )
    from services.trader_orchestrator.risk_manager import evaluate_risk
    from sqlalchemy import select, func as sa_func
    from models.database import BacktestAsyncSessionLocal

    result = ExecutionBacktestResult(
        strategy_slug=slug,
        initial_capital_usd=float(initial_capital_usd),
    )
    total_start = time.monotonic()

    validation = validate_strategy_source(source_code)
    result.validation_errors = validation.get("errors", [])
    result.validation_warnings = validation.get("warnings", [])
    result.class_name = validation.get("class_name") or ""
    if not validation["valid"]:
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result

    loader = StrategyLoader()
    bt_slug = f"_bt_exec_{slug}_{int(time.time())}"
    load_start = time.monotonic()
    # Layer config exactly the way live does:
    #   1. ``strategy.default_config`` (file-level baseline) —
    #      automatically applied by ``StrategyLoader.load``.
    #   2. ``strategies`` table ``config`` column (the live runtime
    #      config the worker loads via ``refresh_from_db``).  This is
    #      the layer the backtest USED to skip — and the reason the
    #      crypto_entropy_maker funnel rejected 99% of signals on
    #      ``edge >= 1.0%`` even though live's edge floor is set to
    #      0.0 in the DB.  Without this layer the backtest evaluates
    #      a strictly-tighter strategy variant than live runs.
    #   3. caller-supplied ``config`` (typically the trader's
    #      ``strategy_params``) — overrides DB on conflict, just like
    #      the live ``_merged_eval_params`` layering at
    #      ``trader_orchestrator_worker:6809``.
    db_config: dict[str, Any] = {}
    try:
        from sqlalchemy import select as _sel
        from models.database import Strategy as _Strategy
        async with AsyncSessionLocal() as _cfg_session:
            _row = (await _cfg_session.execute(
                _sel(_Strategy).where(_Strategy.slug == slug)
            )).scalar_one_or_none()
            if _row is not None and isinstance(_row.config, dict):
                db_config = dict(_row.config)
    except Exception as _cfg_exc:
        logger.warning("backtest could not load DB strategy config for %s: %s", slug, _cfg_exc)
    merged_runtime_config = {**db_config, **(config or {})}
    try:
        loaded = loader.load(bt_slug, source_code, merged_runtime_config)
        strategy = loaded.instance
        result.strategy_name = getattr(strategy, "name", bt_slug)
    except Exception as e:
        result.runtime_error = f"Failed to load strategy: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.load_time_ms = (time.monotonic() - load_start) * 1000

    # Pull risk caps from the loaded strategy's config so the backtest
    # applies the same gates live does at trade-decision time.  The
    # Portfolio's own ``can_submit`` already enforces these — they
    # were just defaulting to None (unlimited) because run_execution_
    # backtest never populated PortfolioConfig with them.  Defined
    # HERE (right after strategy load) so the intent-loop platform
    # gates can read them too.
    strategy_cfg = dict(getattr(strategy, "config", {}) or {})

    def _safe_float_cfg(key: str, default: float | None) -> float | None:
        v = strategy_cfg.get(key)
        if v is None:
            v = merged_runtime_config.get(key)
        if v is None:
            return default
        try:
            f = float(v)
            return f if f > 0 else default
        except (TypeError, ValueError):
            return default

    # max_size_usd is the strategy's per-trade notional cap (see
    # tail_end_carry config).  Map it to per-market AND per-strategy
    # notional caps — Portfolio enforces both, neither stricter than
    # the other on a single-market submission, but per-strategy is
    # the right place for "this strategy may not exceed N at once".
    per_trade_cap = _safe_float_cfg("max_size_usd", None)
    # If a strategy declares max_open_positions, honor it; otherwise
    # leave unlimited.  Live's default is 50 per trader (not per
    # strategy); we use the strategy-level value when present.
    open_pos_cap = strategy_cfg.get("max_open_positions")
    if open_pos_cap is None:
        open_pos_cap = merged_runtime_config.get("max_open_positions")
    if isinstance(open_pos_cap, (int, float)) and open_pos_cap > 0:
        open_pos_cap = int(open_pos_cap)
    else:
        open_pos_cap = None
    # Gross exposure: cap at 50% of capital by default — sane risk
    # ceiling that the live RiskManager would also apply.  Strategies
    # that explicitly want higher exposure can override.
    gross_cap = _safe_float_cfg("max_gross_exposure_usd",
                                 float(initial_capital_usd) * 0.5)

    end_dt = end or datetime.now(timezone.utc)
    # 7 days is the right default for "what would my strategy have
    # done lately": long enough to amass a usable sample for most
    # strategies, short enough that a single backtest run finishes in
    # under a minute on the production microstructure_snapshot volume.
    # The previous 24h default was too narrow — most strategies fire
    # only a handful of opportunities per day.
    start_dt = start or (end_dt - _td(days=7))
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    result.start_iso = start_dt.isoformat()
    result.end_iso = end_dt.isoformat()

    data_start = time.monotonic()
    intents: list[TradeIntent] = []
    tokens: list[str] = []
    try:
        async with BacktestAsyncSessionLocal() as session:
            # Always pull the strategy's opportunities first.  Earlier
            # we picked the token universe by raw microstructure
            # snapshot count (top 25) and THEN filtered opps to those
            # tokens — which silently dropped 99%+ of an event-market
            # strategy's opps because the noisiest microstructure
            # tokens are crypto perpetuals, not the prediction-market
            # tokens the strategy actually fires on.  For tail_end_carry
            # this collapsed 1,523 opps × 653 tokens to 3 intents.
            # Now: opps drive the universe, microstructure follows.
            # Funnel diagnostics — surface the count at each stage so
            # the operator can see where opps are lost (recorder
            # coverage gap vs matching-engine throughput vs cap).
            # ── Canonical opp source: TradeSignal ─────────────────────
            # TradeSignal is the system's authoritative record of "what
            # the live trader saw and decided on" — every strategy that
            # executed in production produced TradeSignal rows.  The
            # earlier OpportunityHistory primary was a downstream
            # cache that some strategies populate (scanner-driven) and
            # others don't (crypto-worker, news, copy-trade), which
            # forced a chain of fallbacks (loose-window broaden,
            # discovery synthesis, events-feeders) just to handle the
            # gap.  Replacing it with TradeSignal removes the gap by
            # construction: a strategy that didn't fire a signal
            # didn't trade live, and there's nothing to replay.
            #
            # Status filter: drop only the live-rejected statuses
            # (``filtered`` quality-gate, ``failed`` evaluate-error,
            # ``skipped`` deduplication).  ``expired`` IS replay-
            # eligible — the live trader's queue just ran out of TTL,
            # which the backtest matcher will independently re-decide.
            from models.database import TradeSignal as _TS
            opps_total_in_window = 0
            try:
                opps_total_in_window = int(
                    (await session.execute(
                        select(sa_func.count(_TS.id)).where(
                            _TS.created_at >= start_dt,
                            _TS.created_at <= end_dt,
                            _TS.strategy_type == slug,
                            _TS.status.notin_(["filtered", "failed", "skipped"]),
                        )
                    )).scalar_one() or 0
                )
            except Exception:
                opps_total_in_window = 0

            try:
                opps = await _load_opps_from_trade_signals(
                    session=session,
                    slug=slug,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    max_rows=int(max_intents),
                )
            except Exception:
                opps = []

            # ── Historical discovery replay ──────────────────────────
            # When enabled (default), run strategy.detect_async against
            # historical market state at sampled time intervals.  This
            # is what "backtest" actually means — the strategy's
            # discovery pipeline runs against recorded data, NOT just
            # against opps live happened to surface.
            #
            # The live ``opps`` set above stays — it's a fast cache of
            # already-discovered opportunities for the strategy.  We
            # APPEND replay-discovered opps to it.  Dedup by (token_id,
            # detected_at-bucket) so we don't double-count when the
            # live system saw the same opp the synthesis would have
            # picked up.
            replay_opps: list = []
            discovery_mode = "live_opps"
            if discover_from_history:
                try:
                    replay_opps = await _replay_discover_opportunities(
                        strategy=strategy,
                        slug=slug,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        sample_interval_seconds=int(discovery_sample_interval_seconds),
                        max_ticks=int(discovery_max_ticks),
                        candidate_token_ids=token_ids,
                    )
                except Exception as exc:
                    logger.warning("Historical discovery replay failed: %s", exc)
                    result.validation_warnings.append(
                        f"Discovery replay FAILED: {type(exc).__name__}: {str(exc)[:200]}"
                    )
                    replay_opps = []

                # Always surface what the discovery path did so we can
                # tell "code ran but found 0" vs "code never ran" from
                # the result alone.
                if replay_opps:
                    discovery_mode = (
                        "hybrid" if opps else "historical_synthesis"
                    )
                    # Dedup by (first-token, 30-min bucket) so
                    # synthesis-time-ticks don't pile on top of live
                    # opps for the same token+moment.
                    def _dedup_key(o: Any) -> tuple[str, int]:
                        pdata = getattr(o, "positions_data", None) or {}
                        if isinstance(pdata, dict):
                            ptt = pdata.get("positions_to_take") or []
                            first = ptt[0] if ptt and isinstance(ptt[0], dict) else {}
                            tok = str(first.get("token_id") or "")
                        else:
                            tok = ""
                        det = getattr(o, "detected_at", None)
                        bucket = int(det.timestamp() // 1800) if det else 0
                        return (tok, bucket)

                    seen = {_dedup_key(o) for o in opps}
                    new_opps = [o for o in replay_opps if _dedup_key(o) not in seen]
                    opps = list(opps) + new_opps
                    result.validation_warnings.append(
                        f"Discovery replay: {len(new_opps)} synthetic opps added "
                        f"(live_opps={len(opps) - len(new_opps)}, replay={len(replay_opps)}, "
                        f"deduped={len(replay_opps) - len(new_opps)})"
                    )
                else:
                    # Code ran but produced 0 opps.  Tell the operator
                    # WHY — most likely: strategy doesn't override
                    # detect_async, or the catalog didn't yield any
                    # markets with reconstructable prices in window,
                    # or every detect_async call timed out.
                    result.validation_warnings.append(
                        "Discovery replay produced 0 synthetic opps "
                        "(strategy may not override detect_async, or no "
                        "catalog markets had reconstructable prices in "
                        "window).  Falling back to live opp_history only."
                    )
            # Surface the discovery mode on the result so the UI can
            # show "this run used historical discovery" / "live opps
            # only" / "hybrid".
            if not hasattr(result, "discovery_mode"):
                pass  # field declared on the dataclass below
            try:
                result.discovery_mode = discovery_mode
            except Exception:
                pass

            tokens = list(token_ids or [])
            if not tokens:
                # Derive the token universe from the strategy's own
                # opportunities.  We then narrow to tokens that
                # actually have book snapshots in the window so the
                # matching engine has something to replay against.
                opp_tokens: dict[str, int] = {}
                for opp in opps or []:
                    pdata = getattr(opp, "positions_data", None) or {}
                    if isinstance(pdata, dict):
                        ptt = pdata.get("positions_to_take") or []
                    else:
                        ptt = []
                    if not ptt:
                        legacy = getattr(opp, "positions_to_take", None) or []
                        if isinstance(legacy, list):
                            ptt = legacy
                    for pos in ptt:
                        if isinstance(pos, dict):
                            tok = str(pos.get("token_id") or "").strip()
                            if tok:
                                opp_tokens[tok] = opp_tokens.get(tok, 0) + 1
                if opp_tokens:
                    # Filter to tokens with actual book coverage in window so
                    # the engine can replay against them.  Recording lives in
                    # canonical parquet (live_ingestor / polybacktest /
                    # telonex), so this uses the unified coverage resolver
                    # against the provider_datasets catalog — one lookup over
                    # the whole token list.
                    candidate_tokens = list(opp_tokens.keys())
                    tokens_with_snaps: set[str] = set()
                    snap_filter_failed = False
                    try:
                        from services.marketdata.coverage import resolve_coverage
                        cov_map = await resolve_coverage(
                            token_ids=candidate_tokens,
                            start=start_dt,
                            end=end_dt,
                        )
                        tokens_with_snaps = set(cov_map.covered_tokens)
                    except Exception as exc:
                        logger.warning(
                            "parquet-availability check failed; trusting opp_tokens universe: %s",
                            exc,
                        )
                        snap_filter_failed = True
                        tokens_with_snaps = set(candidate_tokens)
                    # Sort opp_tokens by intent-frequency desc, take
                    # the ones with snapshots, cap at 500 so a
                    # pathological strategy with 10k tokens doesn't
                    # blow up the matcher.  No-snap tokens still get
                    # logged as a warning so the operator knows what
                    # was skipped.
                    ranked = sorted(opp_tokens.items(), key=lambda kv: kv[1], reverse=True)
                    tokens = [t for t, _ in ranked if t in tokens_with_snaps][:500]
                    no_snap_token_count = (
                        0 if snap_filter_failed
                        else len(opp_tokens) - len(tokens_with_snaps)
                    )
                    capped_universe = len(tokens_with_snaps) > len(tokens)
                    # Funnel summary the operator can read at a glance.
                    snap_label = (
                        "with_book_snapshots=unknown(filter timed out)"
                        if snap_filter_failed
                        else f"with_book_snapshots={len(tokens_with_snaps)}"
                    )
                    funnel_msg = (
                        f"intent funnel — opps_in_window={opps_total_in_window} · "
                        f"opps_pulled={len(opps)} (cap={int(max_intents)}) · "
                        f"opp_tokens={len(opp_tokens)} · "
                        f"{snap_label} · "
                        f"universe={len(tokens)}"
                        + (" (cap=500)" if capped_universe else "")
                    )
                    result.validation_warnings.append(funnel_msg)
                    if snap_filter_failed:
                        result.validation_warnings.append(
                            "Snap-availability check timed out — using all opp tokens "
                            "as the universe.  Tokens with no book data will produce "
                            "zero fills (engine handles them gracefully).  Tighten the "
                            "time window or scope to fewer tokens to restore the "
                            "diagnostic."
                        )
                    elif no_snap_token_count > 0:
                        pct = (
                            no_snap_token_count / len(opp_tokens) * 100.0
                            if opp_tokens else 0.0
                        )
                        result.validation_warnings.append(
                            f"{no_snap_token_count} of {len(opp_tokens)} opp tokens "
                            f"had no book snapshots in window ({pct:.0f}% — recorder "
                            f"didn't capture these markets); their opps were skipped"
                        )
                    if opps_total_in_window > len(opps):
                        result.validation_warnings.append(
                            f"{opps_total_in_window - len(opps)} opportunities "
                            f"exceeded max_intents cap ({int(max_intents)}) — "
                            f"raise max_intents to capture them"
                        )

                    # ── Pre-flight data-coverage measurement ─────────────
                    # Cheap (chunked aggregate queries).  Surfaces how
                    # dense the historical data is for THIS run's
                    # opp universe.  Operators see this BEFORE eating a
                    # 30s replay that ends in "0 trades because the book
                    # was sampled once every 3 hours".
                    #
                    # For event-driven strategies (copy-trade, news,
                    # insider) the fidelity is anchored on event count
                    # rather than book density — the strategy doesn't
                    # discover via books and "low snapshots" is the
                    # wrong signal.
                    bt_event_kind = _replay_event_kind_for_strategy(slug, strategy)
                    bt_scope_wallets = (
                        _extract_scope_wallets(strategy)
                        if bt_event_kind == "wallet_trade"
                        else None
                    )
                    coverage = await _measure_data_coverage(
                        session=session,
                        opp_tokens=candidate_tokens,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        event_kind=bt_event_kind,
                        scope_wallets=bt_scope_wallets,
                    )
                    result.data_coverage = coverage
                    fidelity = coverage.get("fidelity_rating", "unknown")
                    rec = coverage.get("recommended_action") or ""
                    if bt_event_kind:
                        events_total = int(coverage.get("events_total", 0))
                        events_per_hour = float(coverage.get("events_per_hour", 0.0))
                        # Event-driven: report event coverage rather
                        # than snapshot density.  Same severity gating
                        # so the UI badge renders consistently.
                        if fidelity in ("low", "none"):
                            result.validation_warnings.append(
                                f"⚠ EVENT COVERAGE: {fidelity.upper()} — "
                                f"{events_total} {bt_event_kind} event(s) in window "
                                f"({events_per_hour:.2f}/hr).  {rec}"
                            )
                        elif fidelity == "medium":
                            result.validation_warnings.append(
                                f"EVENT COVERAGE: medium — {events_total} "
                                f"{bt_event_kind} event(s) in window "
                                f"({events_per_hour:.2f}/hr).  {rec}"
                            )
                        else:
                            result.validation_warnings.append(
                                f"EVENT COVERAGE: high — {events_total} "
                                f"{bt_event_kind} event(s) in window "
                                f"({events_per_hour:.2f}/hr) ✓"
                            )
                    else:
                        median_rate = coverage.get("median_snaps_per_token_per_hour", 0.0)
                        p10_rate = coverage.get("p10_snaps_per_token_per_hour", 0.0)
                        n_with_deltas = coverage.get("tokens_with_deltas", 0)
                        deltas_total = coverage.get("deltas_total", 0)
                        # Loud, prominent warning when coverage is degraded.
                        # The funnel line above tells the operator how many
                        # tokens HAVE any data; this tells them whether that
                        # data is dense enough to trust the fill simulation.
                        if fidelity in ("low", "none"):
                            # The matcher's per-token router (in the
                            # source-selection block further down) WILL
                            # route any token whose deltas materially
                            # outweigh its snapshots to BookDeltaReplay.
                            # So the headline rate above understates
                            # actual fidelity when deltas dominate for
                            # some tokens — but tokens with sparse data
                            # in BOTH sources are still genuinely low
                            # fidelity.  Word the warning accordingly.
                            result.validation_warnings.append(
                                f"⚠ DATA FIDELITY: {fidelity.upper()} (snapshot-based) — median "
                                f"{median_rate:.1f} snaps/token/hr (p10={p10_rate:.1f}). "
                                f"book_delta_events has {deltas_total:,} rows across "
                                f"{n_with_deltas} of {len(candidate_tokens)} tokens — the matcher's "
                                f"per-token router will use deltas where they dominate; tokens "
                                f"with sparse data in BOTH sources will still backtest off "
                                f"thin observations.  {rec}"
                            )
                        elif fidelity == "medium":
                            result.validation_warnings.append(
                                f"DATA FIDELITY: medium — median {median_rate:.1f} "
                                f"snaps/token/hr.  {rec}"
                            )
                        else:
                            result.validation_warnings.append(
                                f"DATA FIDELITY: high — median {median_rate:.1f} "
                                f"snaps/token/hr ✓"
                            )
                # If the strategy has zero opportunities (e.g. a fresh
                # strategy), fall back to "top parquet-covered tokens in
                # window" so the engine still has something to drive
                # against — a last resort, not the primary path.
                if not tokens:
                    from models.database import ProviderDataset as _PD
                    _s = start_dt.replace(tzinfo=None) if start_dt.tzinfo else start_dt
                    _e = end_dt.replace(tzinfo=None) if end_dt.tzinfo else end_dt
                    pd_stmt = (
                        select(_PD)
                        .where(
                            _PD.storage_type == "parquet",
                            _PD.start_ts <= _e,
                            _PD.end_ts >= _s,
                        )
                        .order_by(_PD.snapshot_count.desc())
                        .limit(25)
                    )
                    pd_rows = (await session.execute(pd_stmt)).scalars().all()
                    seen_fb: list[str] = []
                    for _pd in pd_rows:
                        for t in (_pd.token_ids_json or []):
                            ts = str(t)
                            if ts and ts not in seen_fb:
                                seen_fb.append(ts)
                        if len(seen_fb) >= 25:
                            break
                    tokens = seen_fb[:25]
                    if tokens:
                        result.validation_warnings.append(
                            "No strategy opportunities in window — falling back to top parquet-covered tokens"
                        )

            if not tokens:
                result.runtime_error = (
                    "No tokens to replay against. "
                    "Strategy has no opportunities AND no microstructure snapshots "
                    "in the requested window — pass token_ids explicitly or "
                    "expand the time range."
                )
                result.total_time_ms = (time.monotonic() - total_start) * 1000
                return result

            # Strategy-level evaluate() is the canonical execution
            # gate live uses (trader_orchestrator_worker:6474).  Skipping
            # it in backtest is the single biggest reason backtest
            # results diverge from live: the strategy's own custom
            # checks (capital sizing, market-state filters, edge/
            # confidence thresholds) never run.
            #
            # We construct a minimal EvaluateContext built from the
            # backtest's portfolio + the historical opportunity payload,
            # so each intent is evaluated against the same gate live
            # would apply at submission time.  When evaluate() returns
            # a decision != "selected" we count it as a skip and surface
            # the reasons in the funnel warnings.
            #
            # NOTE: this calls evaluate() against the strategy instance
            # already loaded by the StrategyLoader above (line 1701).
            # Same code path live uses; same custom_checks execute.
            evaluate_skips: dict[str, int] = {}
            # Track WHY each non-selected decision rejected the opp.
            # Strategies emit a free-form ``reason`` string on
            # StrategyDecision; aggregating these is the cheapest way
            # to see which custom_check is the dominant filter.  The
            # top reasons surface in the funnel warning so the operator
            # doesn't need to dig.
            evaluate_skip_reasons: dict[str, int] = {}
            # Per-check failure counts, harvested from
            # ``StrategyDecision.checks``.  Most strategies set a
            # generic ``reason="X filters not met"`` and put the real
            # detail in the per-check list — without aggregating these,
            # the operator just sees one opaque reason for thousands of
            # rejections.  Strategies that don't populate checks
            # contribute nothing here; the reasons map above still
            # surfaces their reason strings.
            evaluate_failed_checks: dict[str, int] = {}
            evaluate_total = 0
            evaluate_selected = 0
            # Opps where evaluate() raised / returned an unknown shape
            # (eval_pair is None).  Counted separately because they
            # bypass the strategy-side gate but still hit the platform
            # gates with decision_obj=None — a passthrough path that
            # used to be silently lumped into "neither selected nor
            # skipped" math, leaving the funnel arithmetic confusing.
            evaluate_passthrough = 0
            # Platform gates funnel — accumulated from the orchestrator's
            # ``apply_platform_decision_gates`` (the canonical decision
            # pipeline live runs at trader_orchestrator_worker:6474).
            # Counts each blocking gate so the operator can read the
            # exact same rejection breakdown the live trader would
            # produce on this signal stream.
            platform_skips: dict[str, int] = {}

            # Backtest-mode portfolio state for the orchestrator's
            # risk evaluator + stacking guard.  Updated as intents
            # accumulate so each subsequent opp sees the realistic
            # gross exposure / occupied markets the previous intents
            # consumed.
            #
            # CRITICAL: positions DECAY over time as they resolve.  Live
            # gross_exposure / open_position counters drop when a
            # position closes; the backtest must mirror that or the
            # caps fire prematurely and we underfill the strategy by
            # 10x+.  ``_bt_open_intents`` is a min-heap-style list of
            # ``(close_at_dt, size_usd, market_id)`` records; before
            # evaluating each opp's risk, we evict every record whose
            # ``close_at_dt`` is <= the opp's ``detected_at`` and
            # decrement the running aggregates by the evicted size.
            # The close timestamp is detected_at + days_to_resolution
            # (from the opp's _tail_end / strategy_context payload),
            # falling back to a 24h default for strategies that don't
            # publish a horizon.
            bt_gross_exposure_usd = 0.0
            bt_open_positions = 0
            bt_cycle_orders = 0
            bt_occupied_market_ids: set[str] = set()
            bt_per_market_exposure: dict[str, float] = {}
            # Each entry: {"close_at": datetime, "size_usd": float, "market_id": str}
            bt_open_intents: list[dict[str, Any]] = []
            from datetime import timedelta as _td_decay
            _DEFAULT_HOLDING_HOURS = 24.0  # safe default for strategies w/o resolution horizon

            def _bt_release_resolved(opp_at: datetime) -> None:
                """Decrement bt_* portfolio counters for any open intent
                whose close_at <= ``opp_at``.  Mirrors how the live
                trader's risk_manager sees gross_exposure decay as
                positions resolve over the day.  O(open_count) per opp.
                """
                nonlocal bt_gross_exposure_usd, bt_open_positions
                if not bt_open_intents:
                    return
                kept: list[dict[str, Any]] = []
                for entry in bt_open_intents:
                    close_at = entry.get("close_at")
                    if isinstance(close_at, datetime) and close_at <= opp_at:
                        # Position resolved.  Drop the size from the
                        # gross + per-market accumulators; decrement
                        # the open-position count.  Markets that fall
                        # to zero exposure are removed from the
                        # occupied-set so the stacking guard sees a
                        # clean slate.
                        bt_gross_exposure_usd = max(
                            0.0, bt_gross_exposure_usd - float(entry.get("size_usd") or 0.0)
                        )
                        bt_open_positions = max(0, bt_open_positions - 1)
                        market_id = str(entry.get("market_id") or "")
                        if market_id:
                            bt_per_market_exposure[market_id] = max(
                                0.0,
                                bt_per_market_exposure.get(market_id, 0.0)
                                - float(entry.get("size_usd") or 0.0),
                            )
                            if bt_per_market_exposure[market_id] <= 1e-9:
                                bt_per_market_exposure.pop(market_id, None)
                                bt_occupied_market_ids.discard(market_id)
                    else:
                        kept.append(entry)
                bt_open_intents[:] = kept

            global_limits = {
                "max_gross_exposure_usd": (
                    float(gross_cap) if gross_cap is not None else float(initial_capital_usd) * 0.5
                ),
                "max_daily_loss_usd": (
                    float(strategy_cfg["max_daily_loss_usd"])
                    if isinstance(strategy_cfg.get("max_daily_loss_usd"), (int, float))
                    else 500.0
                ),
            }
            effective_risk_limits = {
                "max_trade_notional_usd": (
                    float(per_trade_cap) if per_trade_cap is not None else float(initial_capital_usd) * 0.10
                ),
                "max_per_market_exposure_usd": _safe_float_cfg(
                    "max_per_market_exposure_usd",
                    None,
                ),
                "max_position_notional_usd": _safe_float_cfg(
                    "max_position_notional_usd",
                    None,
                ),
                "max_open_positions": (
                    int(open_pos_cap) if open_pos_cap is not None else 50
                ),
                # Backtests fire the entire opp stream as one ``cycle`` —
                # raise the per-cycle order cap so we don't trip it on
                # benign multi-thousand-opp runs.
                "max_orders_per_cycle": int(strategy_cfg.get("max_orders_per_cycle", 100000)),
            }
            allow_averaging = bool(strategy_cfg.get("allow_averaging", False))
            trading_schedule_cfg = (
                dict(strategy_cfg.get("trading_schedule_utc"))
                if isinstance(strategy_cfg.get("trading_schedule_utc"), dict)
                else {}
            )

            # Bulk-pre-fetch the per-opp historical book index used
            # for evaluate-time live_context lookups.  Live's
            # RuntimeTradeSignalView overlays ``live_selected_price``
            # and ``live_edge_percent`` from current market state —
            # without that overlay, evaluate rejects worker-driven
            # signals (whose persisted ``edge_percent`` is null) on
            # the edge floor.  We reconstruct the same overlay from
            # the historical book at each signal's detected_at via
            # an in-memory bisect (one bulk pull, O(log N) per opp),
            # which is ~50× faster than constructing a streaming
            # replay and calling snapshot_at per opp.
            _eval_token_universe = list({
                str(p.get("token_id") or "").strip()
                for o in (opps or [])
                for p in (
                    (getattr(o, "positions_data", None) or {}).get("positions_to_take") or []
                )
                if isinstance(p, dict) and str(p.get("token_id") or "").strip()
            })
            _eval_book_replay: Any = None
            if _eval_token_universe:
                # Point-in-time book lookups for the eval/live-context overlay
                # come from the unified MarketDataView (canonical parquet),
                # adapted to the _BookSource protocol the overlay expects.
                from services.marketdata.view import MarketDataView
                from services.marketdata.book_source import MarketDataViewSource
                _eval_view = await MarketDataView.build(
                    token_ids=list(_eval_token_universe),
                    start=start_dt,
                    end=end_dt,
                )
                _eval_book_replay = MarketDataViewSource(_eval_view)

            for opp in opps or []:
                # OpportunityHistory.positions_data is a JSON blob;
                # positions_to_take lives under the "positions_to_take"
                # key when present, with a top-level fallback for the
                # rare row shape where a strategy wrote the legacy
                # flat schema.
                pdata = getattr(opp, "positions_data", None) or {}
                if isinstance(pdata, dict):
                    positions_to_take = pdata.get("positions_to_take") or []
                else:
                    positions_to_take = []
                if not positions_to_take:
                    legacy = getattr(opp, "positions_to_take", None) or []
                    if isinstance(legacy, list):
                        positions_to_take = legacy
                if not isinstance(positions_to_take, list):
                    continue

                # Run strategy.evaluate() once per opportunity (mirrors
                # live: one TradeSignal → one evaluate() call → one
                # decision that applies to the whole positions_to_take
                # list).  Build a synthetic signal-view from the opp.
                detected = getattr(opp, "detected_at", None)
                if detected is None:
                    continue
                if detected.tzinfo is None:
                    detected = detected.replace(tzinfo=timezone.utc)

                # Decay the running portfolio state by any positions
                # that resolved BEFORE this opp's detected_at — keeps
                # the gate's view of gross_exposure / open_positions
                # honest as backtest time advances.
                _bt_release_resolved(detected)

                evaluate_total += 1
                # Build the historically-reconstructed live_context for
                # this signal — same shape live's worker passes to
                # evaluate, but sourced from the historical book at the
                # signal's detected_at instead of current mid.  This is
                # the single change that takes evaluate() from
                # synthetically-broken to byte-identical with live for
                # worker-driven signals (crypto_*, news_edge, etc.)
                # whose persisted ``edge_percent`` is null and gets
                # overlaid at runtime.
                _eval_live_context: dict[str, Any] = {}
                if (
                    _eval_book_replay is not None
                    and getattr(opp, "_underlying_signal", None) is not None
                ):
                    try:
                        _eval_live_context = await _build_replay_live_context(
                            signal=opp._underlying_signal,
                            pdata=pdata if isinstance(pdata, dict) else {},
                            book_replay=_eval_book_replay,
                            detected_at=detected,
                        )
                    except Exception as _ctx_exc:
                        logger.debug(
                            "live_context build failed for opp %s: %s",
                            getattr(opp, "id", "?"),
                            _ctx_exc,
                        )
                eval_pair = _backtest_evaluate_opportunity(
                    strategy=strategy,
                    opp=opp,
                    pdata=pdata if isinstance(pdata, dict) else {},
                    initial_capital_usd=initial_capital_usd,
                    live_context=_eval_live_context,
                )
                # When evaluate() raised or produced an unknown shape,
                # ``eval_pair`` is None — fall back to passthrough so
                # a buggy strategy doesn't tank the whole backtest.
                if eval_pair is None:
                    decision_obj = None
                    signal_view = None
                    evaluate_passthrough += 1
                else:
                    decision_obj, signal_view = eval_pair
                    eval_status = str(getattr(decision_obj, "decision", "selected") or "selected").lower()
                    if eval_status != "selected":
                        evaluate_skips[eval_status] = evaluate_skips.get(eval_status, 0) + 1
                        # Aggregate the free-form reason so we can see
                        # which custom_check is dominating rejections.
                        # Trim long reason strings to keep the funnel
                        # warning compact.
                        reason = str(
                            getattr(decision_obj, "reason", "") or ""
                        ).strip() or "(no reason)"
                        if len(reason) > 80:
                            reason = reason[:77] + "..."
                        evaluate_skip_reasons[reason] = (
                            evaluate_skip_reasons.get(reason, 0) + 1
                        )
                        # Drill into the per-check list so a generic
                        # reason like "Tail carry filters not met"
                        # decomposes into ``edge=300 · confidence=200
                        # · liquidity=173`` telling the operator which
                        # individual filter is the dominant blocker.
                        try:
                            checks = getattr(decision_obj, "checks", None) or []
                        except Exception:
                            checks = []
                        for chk in checks:
                            try:
                                passed = bool(getattr(chk, "passed", True))
                                if passed:
                                    continue
                                # DecisionCheck.key is the canonical
                                # identifier; fall back to label / name
                                # for any non-standard check shape.
                                name = (
                                    str(
                                        getattr(chk, "key", None)
                                        or getattr(chk, "label", None)
                                        or getattr(chk, "name", None)
                                        or ""
                                    ).strip()
                                    or "(unnamed)"
                                )
                                detail = str(getattr(chk, "detail", "") or "").strip()
                                if detail:
                                    name = f"{name} [{detail}]"
                                if len(name) > 80:
                                    name = name[:77] + "..."
                            except Exception:
                                continue
                            evaluate_failed_checks[name] = (
                                evaluate_failed_checks.get(name, 0) + 1
                            )
                        continue

                # ----------------------------------------------------------
                # Orchestrator decision-gate pipeline (mirrors live).
                # ----------------------------------------------------------
                # ``apply_platform_decision_gates`` is the SAME function
                # the live trader calls at trader_orchestrator_worker:6474
                # right after strategy.evaluate() returns ``selected``.
                # Driving it here means the backtest applies the exact
                # same downstream gates: signal staleness, trading
                # schedule, size cap from effective_risk_limits, the
                # min-exit-notional guard, the stop-loss-vs-upside guard,
                # the risk evaluator (daily loss, gross exposure, open-
                # position counts), occupied-market stacking guard, etc.
                #
                # Backtest-specific knobs:
                #   * ``execution_mode="backtest"`` short-circuits the
                #     live-only gates (strict_ws_pricing, live_market_
                #     revalidation, market_data_freshness, single-market
                #     guard) — those depend on a live WS subscription
                #     that doesn't exist in replay.
                #   * ``invoke_hooks=True`` so the strategy still gets
                #     ``on_blocked`` / ``on_size_capped`` callbacks just
                #     like live, letting strategy-side bookkeeping
                #     (e.g., demote-on-block heuristics) exercise its
                #     real code path.
                #   * Risk evaluator + stacking guard read accumulating
                #     bt_* state, so each gate respects the realistic
                #     portfolio shape that prior intents in this run
                #     have already consumed.
                gate_blocked = False
                size_after_gates: float | None = None
                if decision_obj is not None and signal_view is not None:
                    # Total opp-level economic notional — the orchestrator
                    # gates' size_cap / risk_evaluator / portfolio
                    # checks operate on this single number.  Most opps
                    # carry a single position; multi-position opps fold
                    # into one signal in live too.
                    opp_total_size_usd = 0.0
                    for pos in positions_to_take:
                        if isinstance(pos, dict):
                            opp_total_size_usd += float(pos.get("notional_usd") or 0.0)
                    if opp_total_size_usd <= 0.0:
                        opp_total_size_usd = 50.0
                    # Seed decision_obj.size_usd if the strategy didn't
                    # already do so — gates clamp this as their primary
                    # input.
                    if (
                        getattr(decision_obj, "size_usd", None) is None
                        or float(getattr(decision_obj, "size_usd", 0.0) or 0.0) <= 0.0
                    ):
                        try:
                            decision_obj.size_usd = float(opp_total_size_usd)
                        except Exception:
                            pass

                    def _bt_risk_evaluator(size_for_eval: float, _opp=opp):
                        risk_result = evaluate_risk(
                            size_usd=float(size_for_eval),
                            gross_exposure_usd=float(bt_gross_exposure_usd),
                            trader_open_positions=int(bt_open_positions),
                            trader_open_orders=int(bt_open_positions),
                            market_exposure_usd=float(
                                bt_per_market_exposure.get(
                                    str(getattr(signal_view, "market_id", "") or ""),
                                    0.0,
                                )
                            ),
                            global_limits=global_limits,
                            trader_limits=effective_risk_limits,
                            global_daily_realized_pnl_usd=0.0,
                            trader_daily_realized_pnl_usd=0.0,
                            global_unrealized_pnl_usd=0.0,
                            trader_unrealized_pnl_usd=0.0,
                            trader_consecutive_losses=0,
                            cycle_orders_placed=int(bt_cycle_orders),
                            cooldown_active=False,
                            mode="backtest",
                        )
                        return risk_result, {
                            "global_daily_realized_pnl_usd": 0.0,
                            "trader_daily_realized_pnl_usd": 0.0,
                            "global_unrealized_pnl_usd": 0.0,
                            "trader_unrealized_pnl_usd": 0.0,
                            "intra_cycle_committed_usd": float(bt_gross_exposure_usd),
                            "trader_open_positions": int(bt_open_positions),
                            "trader_open_orders": int(bt_open_positions),
                            "cooldown_active": False,
                        }

                    gate_checks: list[dict[str, Any]] = []
                    try:
                        gate_result = apply_platform_decision_gates(
                            decision_obj=decision_obj,
                            runtime_signal=signal_view,
                            strategy=strategy,
                            checks_payload=gate_checks,
                            trading_schedule_ok=is_within_trading_schedule_utc(
                                {"trading_schedule_utc": trading_schedule_cfg},
                                detected,
                            ),
                            trading_schedule_config=trading_schedule_cfg,
                            global_limits=global_limits,
                            effective_risk_limits=effective_risk_limits,
                            allow_averaging=allow_averaging,
                            occupied_market_ids=set(bt_occupied_market_ids),
                            portfolio_allocator=None,
                            risk_evaluator=_bt_risk_evaluator,
                            invoke_hooks=True,
                            strategy_params=strategy_cfg,
                            execution_mode="backtest",
                        )
                    except Exception as exc:
                        # Don't tank the run on a gate-pipeline bug; log
                        # and proceed as if the gates passed.  This
                        # mirrors how the live worker also catches any
                        # unhandled gate exceptions and falls forward.
                        logger.warning("backtest decision_gates raised: %s", exc)
                        gate_result = None

                    if gate_result is not None:
                        gate_decision = str(gate_result.get("final_decision") or "selected").lower()
                        if gate_decision != "selected":
                            # Find the first blocking gate so the funnel
                            # tag matches what the operator would see in
                            # live's audit trail.
                            blocking_gate = "platform_gate"
                            blocking_detail = ""
                            for g in gate_result.get("platform_gates") or []:
                                if str(g.get("status") or "").lower() == "blocked":
                                    blocking_gate = str(g.get("gate") or "platform_gate")
                                    blocking_detail = str(g.get("detail") or "")
                                    break
                            platform_skips[blocking_gate] = platform_skips.get(blocking_gate, 0) + 1
                            # When the risk gate is the blocker, drill
                            # into its sub-check key (cycle, notional,
                            # gross_exposure, open_positions, market_
                            # exposure...) so the operator sees WHICH
                            # risk dimension is the bottleneck rather
                            # than the catch-all "risk".  Detail looks
                            # like "Risk blocked: trader_open_positions
                            # (next=11 max=10)" — extract the sub-key.
                            if blocking_gate == "risk" and blocking_detail:
                                sub = blocking_detail
                                if sub.startswith("Risk blocked: "):
                                    sub = sub[len("Risk blocked: "):]
                                # Trim trailing parenthetical detail to
                                # keep the key small; full detail is in
                                # the per-skip-reasons map below.
                                sub_key = sub.split(" (", 1)[0].strip() or "(unknown)"
                                platform_skips[f"risk:{sub_key}"] = (
                                    platform_skips.get(f"risk:{sub_key}", 0) + 1
                                )
                            gate_blocked = True
                        else:
                            size_after_gates = float(gate_result.get("size_usd") or opp_total_size_usd)
                            # Track size-cap events in the funnel even
                            # when the gate ultimately allowed the trade.
                            if size_after_gates + 1e-9 < opp_total_size_usd:
                                platform_skips["size_capped"] = (
                                    platform_skips.get("size_capped", 0) + 1
                                )

                if gate_blocked:
                    continue
                evaluate_selected += 1

                # Determine the per-intent USD size.  Two cases:
                #
                # 1. Positions carry their own ``notional_usd``: the
                #    strategy emitted multi-position opps with explicit
                #    weights.  Apply a proportional shrink so the gate-
                #    capped total is distributed across positions in
                #    the strategy's intended ratio.
                #
                # 2. Positions are sizeless (the common case for
                #    single-position scanner strategies like
                #    tail_end_carry — they emit ``action/outcome/price/
                #    token_id`` and rely on the orchestrator to size).
                #    Use the gate-determined ``size_after_gates`` (or
                #    the strategy's ``decision.size_usd`` if the gates
                #    didn't run).  Splitting across N positions equally
                #    matches what live's orchestrator does at submit
                #    time.  Previously the fallback was a hardcoded
                #    ``$50`` which silently quadrupled live's actual
                #    sizing (e.g. tail_end_carry's max_size_usd=$5),
                #    inflating bt_gross_exposure_usd by 10x and
                #    causing the global_gross_exposure cap to fire
                #    after 10 intents instead of the realistic ~100.
                positions_with_size = [
                    p for p in positions_to_take
                    if isinstance(p, dict) and float(p.get("notional_usd") or 0.0) > 0.0
                ]
                fallback_per_position_usd: float | None = None
                shrink = 1.0
                if positions_with_size:
                    if size_after_gates is not None:
                        opp_total_size_usd_check = sum(
                            float(p.get("notional_usd") or 0.0)
                            for p in positions_with_size
                        )
                        if opp_total_size_usd_check > 0.0:
                            shrink = max(0.0, min(1.0, size_after_gates / opp_total_size_usd_check))
                else:
                    # Strategy did not emit per-position notionals.
                    # Source the size from the gate (preferred — already
                    # respects per-trade caps + risk-evaluator clamps)
                    # or from the strategy's decision.
                    candidate_total = (
                        float(size_after_gates)
                        if size_after_gates is not None and size_after_gates > 0.0
                        else float(getattr(decision_obj, "size_usd", 0.0) or 0.0)
                        if decision_obj is not None
                        else 0.0
                    )
                    sized_positions = [
                        p for p in positions_to_take if isinstance(p, dict)
                    ]
                    if candidate_total > 0.0 and sized_positions:
                        fallback_per_position_usd = candidate_total / len(sized_positions)

                for idx, pos in enumerate(positions_to_take):
                    if not isinstance(pos, dict):
                        continue
                    tok = str(pos.get("token_id") or "")
                    if not tok or tok not in tokens:
                        continue
                    side = str(pos.get("action") or pos.get("side") or "BUY").upper()
                    if side not in {"BUY", "SELL"}:
                        side = "BUY"
                    price = float(pos.get("price") or 0.5)
                    pos_notional = float(pos.get("notional_usd") or 0.0)
                    if pos_notional > 0.0:
                        size_usd = pos_notional * shrink
                    elif fallback_per_position_usd is not None:
                        size_usd = fallback_per_position_usd
                    else:
                        # Last-resort guard — same conservative
                        # default the orchestrator uses when it can't
                        # determine a size.  Should be reached only
                        # when strategy.evaluate didn't return a size
                        # and the gates were skipped, which is rare.
                        size_usd = 50.0
                    if size_usd <= 0.0:
                        continue

                    size = size_usd / max(0.01, price)

                    # Pull TIF / post_only from the strategy's emitted
                    # position (was hardcoded IOC).  Tail-end-carry's
                    # ``aggressive_limit_buy_submit_as_gtc`` flag flips
                    # IOC buys above the signal price into GTC — match
                    # that here so the matcher actually sees the
                    # strategy's intended order behavior.
                    tif_raw = str(pos.get("time_in_force") or pos.get("tif") or "GTC").upper()
                    if tif_raw not in {"IOC", "GTC", "FOK", "FAK"}:
                        tif_raw = "GTC"
                    if (
                        tif_raw == "IOC"
                        and side == "BUY"
                        and bool(pos.get("aggressive_limit_buy_submit_as_gtc"))
                    ):
                        tif_raw = "GTC"
                    price_policy = str(pos.get("price_policy") or "").lower()
                    post_only = bool(pos.get("post_only"))
                    if not post_only and price_policy in {"maker", "post_only", "passive"}:
                        post_only = True

                    intents.append(
                        TradeIntent(
                            intent_id=f"opp_{opp.id}_{idx}",
                            emitted_at=detected,
                            token_id=tok,
                            side=side,
                            size=size,
                            limit_price=price,
                            tif=tif_raw,
                            post_only=post_only,
                            strategy_slug=str(getattr(opp, "strategy", "") or slug),
                            meta={
                                "source": "opportunity",
                                "opportunity_id": str(opp.id),
                                "max_execution_price": pos.get("max_execution_price"),
                                "price_policy": price_policy or None,
                                "evaluate_decision": (
                                    str(getattr(decision_obj, "decision", "selected"))
                                    if decision_obj is not None else "passthrough"
                                ),
                                "gate_size_capped": (
                                    bool(size_after_gates is not None and shrink < 1.0 - 1e-9)
                                ),
                                "size_after_gates_usd": (
                                    float(size_after_gates)
                                    if size_after_gates is not None else None
                                ),
                            },
                        )
                    )

                    # Update accumulating backtest portfolio state so
                    # the NEXT opp's gate pass sees realistic gross
                    # exposure / open-position / occupied-market
                    # numbers.  Mirrors how the live cycle accumulator
                    # increments before the next signal in the queue.
                    market_id = str(pos.get("market_id") or "")
                    bt_gross_exposure_usd += size_usd
                    bt_open_positions += 1
                    bt_cycle_orders += 1
                    if market_id:
                        bt_occupied_market_ids.add(market_id)
                        bt_per_market_exposure[market_id] = (
                            bt_per_market_exposure.get(market_id, 0.0) + size_usd
                        )
                    # Compute the position's expected close timestamp
                    # so the next opp sees this slot recycle when the
                    # position resolves.  Sources, in priority order:
                    #
                    #  1. Underlying TradeSignal.expires_at — every
                    #     signal carries the live trader's TTL; for
                    #     short-horizon strategies (crypto 5-min binary
                    #     options, news_edge minute decays) this is
                    #     the right close time.  Without it, the 24h
                    #     fallback pins crypto slots open forever and
                    #     ``risk:trader_open_orders`` rejects every
                    #     opp after the first ~50.
                    #  2. tail_end_carry's ``_tail_end.days_to_resolution``
                    #  3. crypto's ``_entropy.seconds_left`` (signal
                    #     payload's hint for time-to-binary-resolution)
                    #  4. strategy_context.exit_within_seconds variants
                    #  5. 24h default — only as last resort
                    horizon_seconds: float | None = None
                    underlying_sig = getattr(opp, "_underlying_signal", None)
                    if underlying_sig is not None:
                        sig_expires = getattr(underlying_sig, "expires_at", None)
                        if isinstance(sig_expires, datetime):
                            sig_exp = sig_expires
                            if sig_exp.tzinfo is None:
                                sig_exp = sig_exp.replace(tzinfo=timezone.utc)
                            ttl = (sig_exp - detected).total_seconds()
                            if ttl > 0:
                                horizon_seconds = float(ttl)
                    tail_block = pos.get("_tail_end") if isinstance(pos.get("_tail_end"), dict) else None
                    if horizon_seconds is None and isinstance(tail_block, dict):
                        d = tail_block.get("days_to_resolution")
                        if isinstance(d, (int, float)) and d > 0:
                            horizon_seconds = float(d) * 86400.0
                    entropy_block = pos.get("_entropy") if isinstance(pos.get("_entropy"), dict) else None
                    if horizon_seconds is None and isinstance(entropy_block, dict):
                        sl = entropy_block.get("seconds_left")
                        if isinstance(sl, (int, float)) and sl > 0:
                            horizon_seconds = float(sl)
                    if horizon_seconds is None:
                        sc = pdata.get("strategy_context") if isinstance(pdata, dict) else None
                        if isinstance(sc, dict):
                            for key in (
                                "exit_within_seconds",
                                "expected_holding_seconds",
                                "expected_holding_minutes",
                                "max_holding_seconds",
                            ):
                                v = sc.get(key)
                                if isinstance(v, (int, float)) and v > 0:
                                    horizon_seconds = (
                                        float(v) * 60.0 if "minutes" in key else float(v)
                                    )
                                    break
                    if horizon_seconds is None:
                        horizon_seconds = _DEFAULT_HOLDING_HOURS * 3600.0
                    close_at = detected + _td_decay(seconds=horizon_seconds)
                    bt_open_intents.append(
                        {"close_at": close_at, "size_usd": size_usd, "market_id": market_id}
                    )

            if evaluate_total > 0:
                # Surface the evaluate funnel so the operator can see
                # the gate's effect (matches the live orchestrator's
                # rejection breakdown).
                eval_msg_parts = [
                    f"strategy.evaluate() — selected={evaluate_selected}",
                    f"total={evaluate_total}",
                ]
                for st, n in sorted(evaluate_skips.items(), key=lambda kv: -kv[1]):
                    eval_msg_parts.append(f"{st}={n}")
                if evaluate_passthrough > 0:
                    # eval_pair=None opps fall through to the gates with
                    # decision_obj=None — surfaced separately so the
                    # arithmetic ``selected + skipped + passthrough ==
                    # total`` is always intact.
                    eval_msg_parts.append(f"eval_raised_passthrough={evaluate_passthrough}")
                result.validation_warnings.append(" · ".join(eval_msg_parts))

                # Top reasons behind non-selected decisions.  Capped at
                # 5 entries so the warning stays scannable; if a single
                # reason dominates (the common case), it's the first
                # one and tells the operator which custom_check or
                # gate to relax.  Reasons are pulled directly from
                # ``StrategyDecision.reason`` — same string the live
                # audit trail records.
                if evaluate_skip_reasons:
                    top_reasons = sorted(
                        evaluate_skip_reasons.items(), key=lambda kv: -kv[1]
                    )[:5]
                    parts = ["evaluate() top rejection reasons"]
                    for reason, n in top_reasons:
                        parts.append(f"  • {n}× {reason}")
                    result.validation_warnings.append("\n".join(parts))

                # Per-check failure breakdown — decomposes the generic
                # reasons above into specific custom_check names so the
                # operator can see whether ``edge``, ``confidence``,
                # ``liquidity``, ``dtr_window``, etc. is the dominant
                # blocker.  Top 10 keeps the warning useful without
                # exploding for strategies with many checks.
                if evaluate_failed_checks:
                    top_checks = sorted(
                        evaluate_failed_checks.items(), key=lambda kv: -kv[1]
                    )[:10]
                    parts = ["evaluate() failed-check breakdown"]
                    for name, n in top_checks:
                        parts.append(f"  • {n}× {name}")
                    result.validation_warnings.append("\n".join(parts))

            if platform_skips:
                # Platform-gate funnel — emitted directly from the
                # orchestrator's ``apply_platform_decision_gates``
                # output.  Each entry is a real gate name from
                # decision_gates.py (signal_staleness, trading_schedule,
                # size_cap, min_exit_notional, stop_loss_settlement_
                # upside, risk, stacking_guard, etc.) so the operator
                # can read the rejection breakdown using the same
                # vocabulary the live audit trail uses.  Capital +
                # concurrent-position cap rejections that slip through
                # this layer still surface as ``rejected_orders`` on
                # the matching engine result.
                gate_parts = ["orchestrator gates"]
                for reason, n in sorted(platform_skips.items(), key=lambda kv: -kv[1]):
                    gate_parts.append(f"{reason}={n}")
                result.validation_warnings.append(" · ".join(gate_parts))

            # Surface the configured caps so the operator sees what's
            # active vs unlimited.
            cap_parts = []
            if gross_cap is not None:
                cap_parts.append(f"gross_exposure_max=${gross_cap:.0f}")
            if per_trade_cap is not None:
                cap_parts.append(f"per_trade_max=${per_trade_cap:.0f}")
            per_market_cap = _safe_float_cfg("max_per_market_exposure_usd", None)
            if per_market_cap is not None:
                cap_parts.append(f"per_market_max=${per_market_cap:.0f}")
            position_cap = _safe_float_cfg("max_position_notional_usd", None)
            if position_cap is not None:
                cap_parts.append(f"position_max=${position_cap:.0f}")
            if open_pos_cap is not None:
                cap_parts.append(f"open_positions_max={open_pos_cap}")
            if cap_parts:
                result.validation_warnings.append(
                    "risk caps from strategy config — " + " · ".join(cap_parts)
                )

            if not intents:
                result.validation_warnings.append(
                    "No historical opportunities survived evaluate/platform gates; "
                    "execution engine ran zero synthetic orders."
                )
            result.n_intents = len(intents)
    except Exception as e:
        result.runtime_error = f"Failed to fetch data: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.data_fetch_time_ms = (time.monotonic() - data_start) * 1000

    fill_model_snapshot = None
    try:
        from services.fill_simulator import load_active_fill_model

        fill_model_snapshot = await load_active_fill_model(strata_key="pooled")
    except Exception as exc:
        logger.warning("backtest could not load active fill probability model: %s", exc)
        result.validation_warnings.append(
            "Active fill probability model could not be loaded; maker fills used queue-only matching."
        )
    if fill_model_snapshot is None:
        result.validation_warnings.append(
            "No active fill probability model; maker fills used queue-only matching."
        )
    else:
        result.validation_warnings.append(
            "Active fill probability model applied to maker fills: "
            f"{getattr(fill_model_snapshot, 'family', None)}:"
            f"{getattr(fill_model_snapshot, 'strata_key', None)} "
            f"n={getattr(fill_model_snapshot, 'n_events', 0)}."
        )

    engine_config = BacktestConfig(
        portfolio=PortfolioConfig(
            initial_capital_usd=float(initial_capital_usd),
            # ── Risk-cap mapping: backtest = live invariant ─────────────────
            # CRITICAL: ``per_trade_cap`` is the size of an INDIVIDUAL trade
            # ($5 per-position), NOT total per-market or per-strategy
            # exposure.  Mapping it to ``max_per_market_notional_usd`` and
            # ``max_per_strategy_notional_usd`` (the previous wiring)
            # made the FIRST filled trade saturate the cap and silently
            # reject every subsequent intent at the portfolio gate.  Live
            # trading does not have this artificial single-trade ceiling
            # — the strategy fires repeatedly across a market and across
            # the day, with cross-trade exposure governed solely by
            # ``gross_exposure_max`` and the strategy's own dedup logic
            # (``stacking_guard`` etc.).  Backtest must match.
            #
            # The total-exposure ceiling lives on ``max_gross_exposure_usd``
            # (default = 50% of capital, or whatever the strategy config
            # sets via ``max_gross_exposure_usd``).  Per-market and per-
            # strategy caps stay UNCAPPED unless the strategy explicitly
            # exposes its own values for them.
            max_gross_exposure_usd=gross_cap,
            max_per_market_notional_usd=_safe_float_cfg("max_per_market_notional_usd", None),
            max_per_strategy_notional_usd=_safe_float_cfg("max_per_strategy_notional_usd", None),
            max_open_positions=open_pos_cap,
        ),
        latency=LatencyModel(
            submit=LatencyProfile.from_quantiles(
                p50_ms=submit_latency_p50_ms, p95_ms=submit_latency_p95_ms
            ),
            cancel=LatencyProfile.from_quantiles(
                p50_ms=cancel_latency_p50_ms, p95_ms=cancel_latency_p95_ms
            ),
            seed=seed,
            correlation_window_ms=float(latency_correlation_window_ms or 0.0),
        ),
        fees=FeeModel(
            maker_rebate_bps=float(maker_rebate_bps or 0.0),
            maker_rebate_max_spread_bps=float(maker_rebate_max_spread_bps or 50.0),
        ),
        impact=ImpactModel(
            strength_bps=float(impact_strength_bps or 0.0),
            capacity_threshold=float(impact_capacity_threshold),
            capacity_exponent=float(impact_capacity_exponent),
        ),
        seed=seed,
        fill_model_snapshot=fill_model_snapshot,
    )
    engine = BacktestEngine(config=engine_config, strategy=strategy)

    # ── Book source: the unified MarketDataView ─────────────────
    # All book state comes from canonical parquet via the one point-in-time
    # access layer (services.marketdata). The view resolves coverage across
    # every parquet provider (live_ingestor / polybacktest / telonex); tokens
    # with no parquet coverage simply have no book (the SQL replays are gone).
    from services.marketdata.view import MarketDataView
    from services.marketdata.book_source import MarketDataViewSource

    run_start = time.monotonic()
    replay_for_run: Any = None
    _pin_hash: str | None = None
    try:
        _mdv = await MarketDataView.build(token_ids=tokens, start=start_dt, end=end_dt)
        _cov_map = _mdv.coverage()
        _covered = list(_cov_map.covered_tokens)
        replay_for_run = MarketDataViewSource(_mdv, token_ids=_covered)
        # Reproducibility: content-hash the exact parquet this run reads, and
        # pin its window dirs so a pruner can't delete data mid-run.
        try:
            from pathlib import Path as _PPath
            from services.marketdata.pins import pin_paths
            _snap = _mdv.dataset_snapshot()
            result.dataset_snapshot = {
                "content_hash": _snap.content_hash,
                "file_count": len(_snap.entries),
                "total_rows": _snap.total_rows,
                "schema_version": _snap.schema_version,
            }
            if _snap.entries:
                _pin_hash = _snap.content_hash
                pin_paths(
                    _pin_hash,
                    {str(_PPath(e.path).parent) for e in _snap.entries},
                    ttl_seconds=3600,
                )
        except Exception:
            logger.debug("dataset snapshot/pin skipped", exc_info=True)
        result.replay_source = "parquet"
        result.validation_warnings.append(
            f"Replay source: MarketDataView parquet · {len(_covered)}/{len(tokens)} tokens covered"
        )
        if _cov_map.uncovered_tokens:
            result.validation_warnings.append(
                f"{len(_cov_map.uncovered_tokens)} token(s) had no parquet coverage in window"
            )
        bt_result = await engine.run(
            book_source=replay_for_run,
            trade_intents=intents,
            progress_callback=progress_callback,
        )
    except Exception as e:
        result.runtime_error = f"Backtest engine error: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.run_time_ms = (time.monotonic() - run_start) * 1000
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.run_time_ms = (time.monotonic() - run_start) * 1000
        if _pin_hash:
            try:
                from services.marketdata.pins import release_pin
                release_pin(_pin_hash)
            except Exception:
                pass

    # If the book replay had to truncate (e.g. a chunk hit
    # statement_timeout under DB load), surface that loud-and-clear.
    # The matching engine processed whatever snapshots arrived before
    # the truncation, so the run may still be useful — but the
    # operator needs to know the trade count is a lower bound.
    if replay_for_run is not None and getattr(replay_for_run, "truncated", False):
        result.validation_warnings.append(
            f"Book replay TRUNCATED after {replay_for_run.snapshots_yielded} "
            f"snapshots — a chunk query failed: "
            f"{getattr(replay_for_run, 'truncation_reason', None) or 'unknown'}.  "
            f"Trade count is a LOWER BOUND.  Likely cause: live DB load competing "
            f"with the backtest run-session.  Retry when the live system is "
            f"quieter, or shrink the time window / token universe."
        )

    m = bt_result.metrics
    result.success = True
    result.n_snapshots = int(bt_result.notes.get("snapshots_processed", 0) or 0)
    result.fill_probability_model = dict(
        bt_result.notes.get("fill_probability_model") or {}
    )
    result.final_equity_usd = float(bt_result.final_equity_usd)
    result.total_return_pct = float(m.total_return_pct)
    result.annualized_return_pct = float(m.annualized_return_pct)
    result.sharpe = _exec_ci_to_dict(m.sharpe)
    result.sortino = _exec_ci_to_dict(m.sortino)
    result.calmar = _exec_ci_to_dict(m.calmar)
    result.max_drawdown_pct = float(m.max_drawdown_pct)
    result.max_drawdown_usd = float(m.max_drawdown_usd)
    result.drawdown_duration_seconds = float(m.drawdown_duration_seconds)
    result.hit_rate = _exec_ci_to_dict(m.hit_rate)
    result.profit_factor = _exec_ci_to_dict(m.profit_factor)
    result.expectancy_usd = _exec_ci_to_dict(m.expectancy_usd)
    result.avg_win_usd = float(m.avg_win_usd)
    result.avg_loss_usd = float(m.avg_loss_usd)
    result.trade_count = int(m.trade_count)
    result.fees_paid_usd = float(m.fees_paid_usd)
    result.fees_per_fill_usd = float(getattr(bt_result, "fees_per_fill_usd", 0.0) or 0.0)
    result.fees_resolution_usd = float(getattr(bt_result, "fees_resolution_usd", 0.0) or 0.0)
    result.total_fills = int(bt_result.total_fills)
    result.rejected_orders = int(bt_result.rejected_orders)
    result.cancelled_orders = int(bt_result.cancelled_orders)
    result.closed_position_count = int(bt_result.closed_position_count)
    result.open_position_count = int(bt_result.open_position_count)
    result.expected_shortfall_5pct = _exec_ci_to_dict(getattr(m, "expected_shortfall_5pct", None))
    result.expected_shortfall_1pct = _exec_ci_to_dict(getattr(m, "expected_shortfall_1pct", None))
    result.tail_ratio = _exec_ci_to_dict(getattr(m, "tail_ratio", None))
    result.gain_to_pain = _exec_ci_to_dict(getattr(m, "gain_to_pain", None))
    result.correlation_pairs = [
        {"token_a": a, "token_b": b, "correlation": rho}
        for (a, b), rho in (bt_result.correlation_matrix or {}).items()
    ]

    fills = list(bt_result.fills or [])
    if fills_sample_size and len(fills) > fills_sample_size:
        head = fills[:50]
        tail = fills[-max(0, fills_sample_size - 50) :]
        fills = head + tail
    result.fills_sample = [
        {
            "order_id": f.order_id,
            "token_id": f.token_id,
            "side": f.side,
            "price": float(f.price),
            "size": float(f.size),
            "fee_usd": float(f.fee_usd),
            "occurred_at": f.occurred_at.isoformat(),
            "fill_index": int(f.fill_index),
            "is_maker": bool((f.notes or {}).get("maker", False)),
            "fill_probability": (f.notes or {}).get("fill_probability"),
            "fill_probability_horizon_seconds": (
                (f.notes or {}).get("fill_probability_horizon_seconds")
            ),
        }
        for f in fills
    ]

    eq_full = list(bt_result.equity_history or [])
    if equity_sample_size and len(eq_full) > equity_sample_size:
        step = max(1, len(eq_full) // equity_sample_size)
        eq = eq_full[::step]
        # ALWAYS include the true final equity point — otherwise the
        # chart's "ending" value disagrees with total_return_pct (which
        # is computed from final_equity_usd = portfolio.equity_usd()
        # post-mark-to-market).  Stride sampling can drop up to step-1
        # tail entries, including the post-close-out anchor appended in
        # engine._final_mark_to_market().
        if eq_full[-1] is not eq[-1]:
            eq.append(eq_full[-1])
    else:
        eq = eq_full
    result.equity_curve_sample = [
        {"at": ts.isoformat(), "equity_usd": float(value)} for ts, value in eq
    ]
    result.positions_summary = list(getattr(bt_result, "positions_summary", []) or [])

    try:
        loader.unload(bt_slug)
    except Exception:
        pass

    result.total_time_ms = (time.monotonic() - total_start) * 1000
    return result
