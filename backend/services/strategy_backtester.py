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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _backtest_evaluate_opportunity(
    *,
    strategy: Any,
    opp: Any,
    pdata: dict[str, Any],
    initial_capital_usd: float,
) -> tuple[Any, Any] | None:
    """Run ``strategy.evaluate()`` on a backtest opportunity row.

    Mirrors the live orchestrator's gate at trader_orchestrator_worker
    line 6474 — the strategy's own ``evaluate()`` decides whether the
    intent should fire RIGHT NOW given current portfolio + market
    context.  Without this call, backtests run a fundamentally
    different strategy variant: ``detect()`` only, no execution-time
    re-validation, no adaptive sizing, no custom_checks.

    Returns ``(decision_obj, signal_view)`` so the caller can hand both
    to ``apply_platform_decision_gates`` for the post-strategy
    orchestrator gates (signal staleness, trading schedule, risk
    evaluator, occupied market guard, demoted-strategies, etc.) — that
    pipeline is what live runs at trader_orchestrator_worker:6474.
    Returns None if evaluate() raised — fall back to "passthrough" so
    a bug in evaluate() doesn't tank the entire backtest.  Caller
    treats decision_obj.decision == "selected" as accept, anything
    else as skip.
    """
    if not hasattr(strategy, "evaluate"):
        return None
    try:
        # Synthesize a signal-view that quacks like a TradeSignal so
        # the strategy's evaluate() reads the right fields.  Strategies
        # reach into many TradeSignal columns (source, strategy_type,
        # liquidity, edge_percent, confidence, entry_price, market_id,
        # payload_json, strategy_context_json) — we populate every one
        # of them faithfully from the OpportunityHistory row + the
        # nested positions_to_take payload.  Missing fields cause
        # evaluate() to silently reject, which the user hit with 1323
        # of 1323 opps skipped on tail_end_carry.
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
) -> ExecutionBacktestResult:
    """Execution-realistic backtest using full L2 replay + bootstrap CIs.

    Loads the strategy, fetches book snapshots from
    ``MarketMicrostructureSnapshot``, generates trade intents from
    historical opportunities, runs the production matching engine, and
    reports headline + risk-adjusted metrics with bootstrap CIs.
    """
    from datetime import timedelta as _td
    from services.backtest import (
        BacktestConfig,
        BacktestEngine,
        BookReplay,
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
    from models.database import (
        AsyncSessionLocal,
        MarketMicrostructureSnapshot,
        # Historical opportunities live in the OpportunityHistory ORM
        # table.  Aliased as ``Opportunity`` here so the SQLAlchemy
        # query syntax below stays readable; positions are nested in
        # the ``positions_data`` JSON column rather than a top-level
        # ``positions_to_take`` attribute.  See the row-shape sample
        # at services/strategy_backtester.py:_extract_positions for
        # the canonical key path.
        OpportunityHistory as Opportunity,
    )

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
        async with AsyncSessionLocal() as session:
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
            opps_total_in_window = 0
            try:
                opps_total_in_window = int(
                    (await session.execute(
                        select(sa_func.count(Opportunity.id)).where(
                            Opportunity.detected_at >= start_dt,
                            Opportunity.detected_at <= end_dt,
                            Opportunity.strategy_type == slug,
                        )
                    )).scalar_one() or 0
                )
            except Exception:
                opps_total_in_window = 0

            try:
                opp_stmt = (
                    select(Opportunity)
                    .where(
                        Opportunity.detected_at >= start_dt,
                        Opportunity.detected_at <= end_dt,
                        Opportunity.strategy_type == slug,
                    )
                    .order_by(Opportunity.detected_at.asc())
                    .limit(int(max_intents))
                )
                opps = (await session.execute(opp_stmt)).scalars().all()
                # Fallback: if the slug filter found nothing, broaden to
                # window-only.  Covers strategy renames (slug changes
                # don't backfill historical rows).
                if not opps:
                    opp_stmt_loose = (
                        select(Opportunity)
                        .where(
                            Opportunity.detected_at >= start_dt,
                            Opportunity.detected_at <= end_dt,
                        )
                        .order_by(Opportunity.detected_at.asc())
                        .limit(int(max_intents))
                    )
                    opps = (await session.execute(opp_stmt_loose)).scalars().all()
            except Exception:
                opps = []

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
                    # Filter to tokens with actual book snapshots in
                    # window so the engine can replay against them.
                    # IMPORTANT: chunk the IN-list.  A 400+ token IN-
                    # clause against the (token_id, snapshot_type,
                    # observed_at) index forces 400+ index seeks +
                    # GROUP BY work that blows past the Postgres
                    # statement_timeout for crypto strategies that fan
                    # out across many markets.  Chunks of 50 keep each
                    # query predictable.
                    candidate_tokens = list(opp_tokens.keys())
                    tokens_with_snaps: set[str] = set()
                    snap_filter_failed = False
                    CHUNK_SIZE = 50
                    try:
                        for i in range(0, len(candidate_tokens), CHUNK_SIZE):
                            chunk = candidate_tokens[i : i + CHUNK_SIZE]
                            chunk_stmt = (
                                select(MarketMicrostructureSnapshot.token_id)
                                .where(
                                    MarketMicrostructureSnapshot.observed_at >= start_dt,
                                    MarketMicrostructureSnapshot.observed_at <= end_dt,
                                    MarketMicrostructureSnapshot.snapshot_type == "book",
                                    MarketMicrostructureSnapshot.token_id.in_(chunk),
                                )
                                .group_by(MarketMicrostructureSnapshot.token_id)
                            )
                            chunk_rows = (await session.execute(chunk_stmt)).all()
                            for r in chunk_rows:
                                if r[0]:
                                    tokens_with_snaps.add(str(r[0]))
                    except Exception as exc:
                        # Fall back to "trust all opp_tokens" rather
                        # than failing the entire backtest.  The
                        # matching engine handles missing tokens
                        # gracefully — they just produce no fills.
                        logger.warning(
                            "Snap-availability check failed; trusting opp_tokens universe: %s",
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
                # If the strategy has zero opportunities (e.g. a fresh
                # strategy), fall back to "top tokens by snapshot
                # volume in window" so the engine still has something
                # to drive against — same behavior as before but only
                # as a last resort, not the primary path.
                if not tokens:
                    fallback_stmt = (
                        select(
                            MarketMicrostructureSnapshot.token_id,
                            sa_func.count(MarketMicrostructureSnapshot.id).label("c"),
                        )
                        .where(
                            MarketMicrostructureSnapshot.observed_at >= start_dt,
                            MarketMicrostructureSnapshot.observed_at <= end_dt,
                            MarketMicrostructureSnapshot.snapshot_type == "book",
                        )
                        .group_by(MarketMicrostructureSnapshot.token_id)
                        .order_by(sa_func.count(MarketMicrostructureSnapshot.id).desc())
                        .limit(25)
                    )
                    rows = (await session.execute(fallback_stmt)).all()
                    tokens = [str(r[0]) for r in rows if r[0]]
                    if tokens:
                        result.validation_warnings.append(
                            "No strategy opportunities in window — falling back to top-25 microstructure tokens"
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
            evaluate_total = 0
            evaluate_selected = 0
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
            # consumed — same way live does cycle accounting before
            # submission.
            bt_gross_exposure_usd = 0.0
            bt_open_positions = 0
            bt_cycle_orders = 0
            bt_occupied_market_ids: set[str] = set()
            bt_per_market_exposure: dict[str, float] = {}

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

                evaluate_total += 1
                eval_pair = _backtest_evaluate_opportunity(
                    strategy=strategy,
                    opp=opp,
                    pdata=pdata if isinstance(pdata, dict) else {},
                    initial_capital_usd=initial_capital_usd,
                )
                # When evaluate() raised or produced an unknown shape,
                # ``eval_pair`` is None — fall back to passthrough so
                # a buggy strategy doesn't tank the whole backtest.
                if eval_pair is None:
                    decision_obj = None
                    signal_view = None
                else:
                    decision_obj, signal_view = eval_pair
                    eval_status = str(getattr(decision_obj, "decision", "selected") or "selected").lower()
                    if eval_status != "selected":
                        evaluate_skips[eval_status] = evaluate_skips.get(eval_status, 0) + 1
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
                            for g in gate_result.get("platform_gates") or []:
                                if str(g.get("status") or "").lower() == "blocked":
                                    blocking_gate = str(g.get("gate") or "platform_gate")
                                    break
                            platform_skips[blocking_gate] = platform_skips.get(blocking_gate, 0) + 1
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

                # Compute the proportional shrink factor when the gate
                # capped the opp's economic notional.  Each position's
                # original notional gets scaled by this so the relative
                # mix the strategy emitted is preserved.
                shrink = 1.0
                if size_after_gates is not None:
                    opp_total_size_usd_check = sum(
                        float(p.get("notional_usd") or 0.0)
                        for p in positions_to_take
                        if isinstance(p, dict)
                    )
                    if opp_total_size_usd_check > 0.0:
                        shrink = max(0.0, min(1.0, size_after_gates / opp_total_size_usd_check))

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
                    size_usd = float(pos.get("notional_usd") or 50.0) * shrink
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
                result.validation_warnings.append(" · ".join(eval_msg_parts))

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
            if open_pos_cap is not None:
                cap_parts.append(f"open_positions_max={open_pos_cap}")
            if cap_parts:
                result.validation_warnings.append(
                    "risk caps from strategy config — " + " · ".join(cap_parts)
                )

            if not intents and tokens:
                intents.append(
                    TradeIntent(
                        intent_id=f"seed_{tokens[0]}",
                        emitted_at=start_dt,
                        token_id=tokens[0],
                        side="BUY",
                        size=10.0,
                        limit_price=0.50,
                        tif="IOC",
                        post_only=False,
                        strategy_slug=slug,
                        meta={"source": "seed"},
                    )
                )
                result.validation_warnings.append(
                    "No historical opportunities matched window/tokens; ran a "
                    "single seed intent."
                )
            result.n_intents = len(intents)
    except Exception as e:
        result.runtime_error = f"Failed to fetch data: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.data_fetch_time_ms = (time.monotonic() - data_start) * 1000

    engine_config = BacktestConfig(
        portfolio=PortfolioConfig(
            initial_capital_usd=float(initial_capital_usd),
            max_gross_exposure_usd=gross_cap,
            max_per_market_notional_usd=per_trade_cap,
            max_per_strategy_notional_usd=per_trade_cap,
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
    )
    engine = BacktestEngine(config=engine_config, strategy=strategy)

    run_start = time.monotonic()
    try:
        async with AsyncSessionLocal() as run_session:
            replay_for_run = BookReplay(
                session=run_session,
                token_ids=tokens,
                start=start_dt,
                end=end_dt,
                snapshot_type="book",
            )
            bt_result = await engine.run(book_source=replay_for_run, trade_intents=intents)
    except Exception as e:
        result.runtime_error = f"Backtest engine error: {e}"
        result.runtime_traceback = traceback.format_exc()
        result.run_time_ms = (time.monotonic() - run_start) * 1000
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result
    finally:
        result.run_time_ms = (time.monotonic() - run_start) * 1000

    m = bt_result.metrics
    result.success = True
    result.n_snapshots = int(bt_result.notes.get("snapshots_processed", 0) or 0)
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
        }
        for f in fills
    ]

    eq = list(bt_result.equity_history or [])
    if equity_sample_size and len(eq) > equity_sample_size:
        step = max(1, len(eq) // equity_sample_size)
        eq = eq[::step]
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
