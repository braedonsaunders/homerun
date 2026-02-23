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
from datetime import datetime, timezone
from typing import Any, Optional

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
) -> list[Any]:
    loop = asyncio.get_running_loop()
    if _has_custom_detect_async(strategy):
        return await asyncio.wait_for(
            strategy.detect_async(events, markets, prices),
            timeout=timeout_seconds,
        )
    if _has_custom_detect_sync(strategy):
        return await asyncio.wait_for(
            loop.run_in_executor(None, strategy.detect_sync, events, markets, prices),
            timeout=timeout_seconds,
        )
    return await asyncio.wait_for(
        loop.run_in_executor(None, strategy.detect, events, markets, prices),
        timeout=timeout_seconds,
    )


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
        prices = (
            dict(scanner._cached_prices) if hasattr(scanner, "_cached_prices") and scanner._cached_prices else {}
        )

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

        opportunities = await _run_detect_once(
            strategy, events, markets, base_prices, timeout_seconds=30.0
        )

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

        result.grid_results.append({
            "params": config,
            "num_opportunities": scored.num_opportunities,
            "avg_roi": scored.avg_roi,
            "total_roi": scored.total_roi,
            "quality_pass_rate": scored.quality_pass_rate,
        })

        await asyncio.sleep(0)

    # Rank by composite metric (ROI * quality_pass_rate)
    grid_scores.sort(key=lambda x: x[1].composite_score(), reverse=True)

    if not grid_scores:
        result.runtime_error = "No configurations produced results"
        result.sweep_time_ms = (time.monotonic() - sweep_start) * 1000
        return result

    # Take top_k and validate on held-out test window
    top_candidates = grid_scores[:min(top_k, len(grid_scores))]

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

        result.opportunities = _serialize_opportunities(opportunities or [])
        result.num_opportunities = len(result.opportunities)
        result.quality_reports = _build_quality_reports(opportunities or [])
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
        from models.database import AsyncSessionLocal
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
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
        params = dict(merged_config)
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
        platform_open_market_ids = {
            str(value or "").strip()
            for value in (platform_overrides.get("open_market_ids") or [])
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
                    open_market_ids=platform_open_market_ids,
                    portfolio_allocator=None,
                    risk_evaluator=_backtest_risk_evaluator,
                    invoke_hooks=False,
                    strategy_params={},
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

    Loads the strategy, fetches open paper positions, and runs should_exit()
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

    # 3. Fetch open paper positions
    data_start = time.monotonic()
    try:
        from models.database import AsyncSessionLocal, TraderPosition
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
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
                pos_view.mode = getattr(pos, "mode", "paper")
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
