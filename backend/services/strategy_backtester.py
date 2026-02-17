"""
Strategy Backtester

Provides true code-level backtesting for opportunity detection strategies.
Takes a strategy's Python source code, compiles and loads it via the existing
PluginLoader mechanism, then runs it against LIVE current market data (from
the scanner's cache or fetched fresh from the API) to see what opportunities
it would detect RIGHT NOW.

This is NOT about replaying historical data.  It answers the question:
"If I deploy this code, what would it find on the current market snapshot?"
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from services.plugin_loader import PluginLoader, validate_plugin_source
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
    validation = validate_plugin_source(source_code)
    result.validation_errors = validation.get("errors", [])
    result.validation_warnings = validation.get("warnings", [])
    result.class_name = validation.get("class_name") or ""

    if not validation["valid"]:
        result.total_time_ms = (time.monotonic() - total_start) * 1000
        return result

    # ---- 2. Load strategy via plugin loader ----
    loader = PluginLoader()  # Fresh isolated loader for backtest
    bt_slug = f"_bt_{slug}_{int(time.time())}"
    load_start = time.monotonic()
    try:
        loaded = loader.load_plugin(bt_slug, source_code, config)
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
                dict(scanner._cached_prices)
                if hasattr(scanner, "_cached_prices") and scanner._cached_prices
                else {}
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
                prices = await polymarket_client.get_prices_batch(
                    all_token_ids[:500]
                )
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
                loop.run_in_executor(
                    None, strategy.detect, events, markets, prices
                ),
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
                    opp_dicts.append(
                        {
                            k: v
                            for k, v in opp.__dict__.items()
                            if not k.startswith("_")
                        }
                    )
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
        loader.unload_plugin(bt_slug)
    except Exception:
        pass

    result.total_time_ms = (time.monotonic() - total_start) * 1000
    return result
