"""
Strategy Backtester

Provides code-level backtesting for all three strategy phases:
  - DETECT: What opportunities would this code find on the current market snapshot?
  - EVALUATE: Given recent trade signals, which would this strategy accept/reject?
  - EXIT: Given current open positions, which would this strategy close?

Uses live/cached data — not historical replay.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from services.strategy_loader import StrategyLoader, validate_strategy_source
from services.scanner import scanner
from utils.logger import get_logger

logger = get_logger(__name__)


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
    # Results
    opportunities: list[dict[str, Any]] = field(default_factory=list)
    num_opportunities: int = 0
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


def _has_custom_detect_async(strategy) -> bool:
    """Check if strategy implements its own detect_async (not just inherited)."""
    method = getattr(type(strategy), "detect_async", None)
    if method is None:
        return False
    from services.strategies.base import BaseStrategy

    base_method = getattr(BaseStrategy, "detect_async", None)
    return method is not base_method


async def run_strategy_backtest(
    source_code: str,
    slug: str = "_backtest_preview",
    config: Optional[dict[str, Any]] = None,
) -> BacktestResult:
    """Run a strategy's detection code against current market data.

    This compiles the strategy source code, loads it into a sandboxed
    plugin instance, fetches the current market snapshot, and runs
    detect()/detect_async() to see what opportunities it would find.

    Args:
        source_code: Python source code of the strategy
        slug: Identifier for the backtest run (used internally by plugin loader)
        config: Optional config overrides for the strategy

    Returns:
        BacktestResult with opportunities found, timing, and any errors
    """
    result = BacktestResult(strategy_slug=slug)
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

            # Get prices for first batch of tokens
            all_token_ids = []
            for m in markets:
                for tid in m.clob_token_ids:
                    if len(tid) > 20:
                        all_token_ids.append(tid)
            prices = {}
            if all_token_ids:
                prices = await polymarket_client.get_prices_batch(all_token_ids[:500])
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
        loop = asyncio.get_running_loop()
        if _has_custom_detect_async(strategy):
            opps = await asyncio.wait_for(
                strategy.detect_async(events, markets, prices),
                timeout=60.0,
            )
        else:
            opps = await asyncio.wait_for(
                loop.run_in_executor(None, strategy.detect, events, markets, prices),
                timeout=60.0,
            )

        # Serialize opportunities
        opp_dicts = []
        for opp in opps or []:
            try:
                if hasattr(opp, "model_dump"):
                    opp_dicts.append(opp.model_dump())
                elif hasattr(opp, "dict"):
                    opp_dicts.append(opp.dict())
                elif hasattr(opp, "__dict__"):
                    opp_dicts.append({k: v for k, v in opp.__dict__.items() if not k.startswith("_")})
                else:
                    opp_dicts.append(str(opp))
            except Exception:
                opp_dicts.append({"error": "Failed to serialize opportunity"})

        result.opportunities = opp_dicts
        result.num_opportunities = len(opp_dicts)
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
            query = (
                select(TradeSignalEmission)
                .order_by(TradeSignalEmission.created_at.desc())
                .limit(max_signals)
            )
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
        params = dict(config or {})
        for sig in signals:
            try:
                context = {
                    "params": params,
                    "mode": "backtest",
                    "source_config": {},
                }
                decision = strategy.evaluate(sig, context)
                decision_str = getattr(decision, "decision", str(decision))
                reason_str = getattr(decision, "reason", "")
                checks = []
                if hasattr(decision, "checks"):
                    for c in (decision.checks or []):
                        checks.append({
                            "check_key": getattr(c, "check_key", ""),
                            "check_label": getattr(c, "check_label", ""),
                            "passed": getattr(c, "passed", False),
                            "score": getattr(c, "score", None),
                            "detail": getattr(c, "detail", ""),
                        })

                result.decisions.append({
                    "signal_id": getattr(sig, "id", None),
                    "source": getattr(sig, "source", ""),
                    "strategy_type": getattr(sig, "strategy_type", ""),
                    "decision": decision_str,
                    "reason": reason_str,
                    "size_usd": getattr(decision, "size_usd", None),
                    "checks": checks,
                })

                if decision_str == "selected":
                    result.selected += 1
                elif decision_str == "blocked":
                    result.blocked += 1
                else:
                    result.skipped += 1
            except Exception as exc:
                result.decisions.append({
                    "signal_id": getattr(sig, "id", None),
                    "decision": "error",
                    "reason": str(exc),
                    "checks": [],
                })

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
    would_hold: int = 0
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
            query = (
                select(TraderPosition)
                .where(TraderPosition.status == "open")
                .order_by(TraderPosition.opened_at.desc())
                .limit(max_positions)
            )
            positions = list((await session.execute(query)).scalars().all())
        result.num_positions = len(positions)
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
        for pos in positions:
            try:
                payload = pos.payload_json if isinstance(pos.payload_json, dict) else {}
                entry_price = float(payload.get("entry_price", 0) or 0)
                current_price = float(payload.get("last_price", entry_price) or entry_price)
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                class _PositionView:
                    pass
                pos_view = _PositionView()
                pos_view.entry_price = entry_price
                pos_view.current_price = current_price
                pos_view.highest_price = float(payload.get("highest_price", current_price) or current_price)
                pos_view.lowest_price = float(payload.get("lowest_price", current_price) or current_price)
                pos_view.age_minutes = 0
                pos_view.pnl_percent = pnl_pct
                pos_view.strategy_context = payload.get("strategy_context", {})
                pos_view.config = config or {}
                pos_view.outcome_idx = payload.get("outcome_idx", 0)

                market_state = {
                    "current_price": current_price,
                    "market_tradable": True,
                    "is_resolved": False,
                    "winning_outcome": None,
                }

                exit_decision = strategy.should_exit(pos_view, market_state)
                action = getattr(exit_decision, "action", "hold") if exit_decision else "hold"
                reason = getattr(exit_decision, "reason", "") if exit_decision else ""

                result.exit_decisions.append({
                    "position_id": pos.id,
                    "market_id": getattr(pos, "market_id", None),
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "pnl_pct": round(pnl_pct, 2),
                    "action": action,
                    "reason": reason,
                })

                if action == "close":
                    result.would_close += 1
                else:
                    result.would_hold += 1
            except Exception as exc:
                result.exit_decisions.append({
                    "position_id": pos.id,
                    "action": "error",
                    "reason": str(exc),
                })

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
