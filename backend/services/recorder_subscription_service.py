"""Proactive recorder subscription service.

Closes the coverage gap that drove the BacktestStudio fidelity bug:
the recorder was passive — it only captured markets the orchestrator
or a strategy had explicitly subscribed to.  Tail-end-carry-style
strategies that fire on low-volume EOL markets ended up with 38% of
their opportunity tokens having ZERO book data, and another 45%
where book capture started AFTER the strategy's first opportunity
on that token.

This service makes the recorder PROACTIVE.  It runs a periodic loop
on the trading plane that:

  1. Reads MarketCatalog (the canonical "all known active markets"
     view, refreshed every 5 min by market_runtime).
  2. Filters to markets that are: active, accepting orders, not
     closed/archived/resolved, with at least one clob_token_id.
  3. Ranks by liquidity (so when the cap bites, we keep the
     markets that actually trade).
  4. Subscribes the top ``MAX_RECORDED_TOKENS`` tokens via the
     existing single PolymarketWSFeed — same connection live
     trading uses; no separate WS pool, no auth duplication.

Existing 10-minute idle eviction in ``ws_feeds`` automatically prunes
truly-dead tokens from the subscription set so the cap stays
saturated with markets that actually move.

The cap exists because Polymarket CLOB tolerates a few thousand
subscriptions per connection but not infinite.  Default 8000 covers
the bulk of liquid markets (~80% of $100+ liquidity tier).  Operators
can raise it via the ``HOMERUN_RECORDER_MAX_TOKENS`` env var or by
patching the constant if they're willing to take the WS-bandwidth
hit for full fidelity.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("recorder_subscription_service")


_LOOP_INTERVAL_SECONDS = 60.0  # tighter than catalog (5min) so new markets get subscribed fast
_DEFAULT_MAX_TOKENS = int(os.environ.get("HOMERUN_RECORDER_MAX_TOKENS", "8000"))
_MIN_LIQUIDITY_USD = float(os.environ.get("HOMERUN_RECORDER_MIN_LIQUIDITY", "10.0"))


@dataclass
class _ServiceState:
    last_run_at: float = 0.0
    last_run_duration_ms: float = 0.0
    last_run_subscribed_count: int = 0
    last_run_target_count: int = 0
    last_run_catalog_market_count: int = 0
    last_run_catalog_token_count: int = 0
    last_run_dropped_low_liquidity: int = 0
    last_run_dropped_over_cap: int = 0
    last_run_already_subscribed: int = 0
    last_error: str | None = None
    total_runs: int = 0
    recording_enabled: bool = True


_state = _ServiceState()


def get_status() -> dict[str, Any]:
    """Status snapshot for the Data Lab UI / agent diagnostics."""
    return {
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "min_liquidity_usd": _MIN_LIQUIDITY_USD,
        "loop_interval_seconds": _LOOP_INTERVAL_SECONDS,
        "last_run_at_epoch": _state.last_run_at,
        "last_run_age_seconds": (
            (time.time() - _state.last_run_at) if _state.last_run_at > 0 else None
        ),
        "last_run_duration_ms": _state.last_run_duration_ms,
        "last_run_subscribed_count": _state.last_run_subscribed_count,
        "last_run_target_count": _state.last_run_target_count,
        "last_run_catalog_market_count": _state.last_run_catalog_market_count,
        "last_run_catalog_token_count": _state.last_run_catalog_token_count,
        "last_run_dropped_low_liquidity": _state.last_run_dropped_low_liquidity,
        "last_run_dropped_over_cap": _state.last_run_dropped_over_cap,
        "last_run_already_subscribed": _state.last_run_already_subscribed,
        "last_error": _state.last_error,
        "total_runs": _state.total_runs,
        "recording_enabled": _state.recording_enabled,
    }


def _classify_market(m: dict[str, Any]) -> tuple[bool, list[str], float]:
    """Return (is_active, token_ids, liquidity_usd) for a market dict."""
    if not isinstance(m, dict):
        return False, [], 0.0
    if m.get("closed") or m.get("archived") or m.get("resolved"):
        return False, [], 0.0
    if m.get("active") is False:
        return False, [], 0.0
    if m.get("accepting_orders") is False:
        return False, [], 0.0
    raw = m.get("clob_token_ids") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raw = []
    tokens = []
    for t in raw or []:
        ts = str(t or "").strip()
        if ts:
            tokens.append(ts)
    if not tokens:
        return False, [], 0.0
    try:
        liq = float(m.get("liquidity") or 0.0)
    except (TypeError, ValueError):
        liq = 0.0
    return True, tokens, liq


async def _gather_target_tokens() -> tuple[list[str], dict[str, Any]]:
    """Read MarketCatalog and return the token list to subscribe.

    Returns ``(tokens, stats)`` where ``stats`` carries the funnel
    counts for the status endpoint.
    """
    from services.shared_state import _read_market_catalog_file

    stats = {
        "catalog_market_count": 0,
        "catalog_token_count": 0,
        "candidates": 0,
        "after_liquidity_filter": 0,
        "after_cap": 0,
        "dropped_low_liquidity": 0,
        "dropped_over_cap": 0,
    }

    catalog = _read_market_catalog_file()
    if catalog is None:
        return [], stats
    _events, markets, _meta = catalog

    # Build per-token records: (liquidity, token_id).  Dedupe by token.
    # When a token appears in multiple markets (rare), keep the higher
    # liquidity value so it ranks correctly.
    by_token: dict[str, float] = {}
    catalog_tokens: set[str] = set()
    for m in markets:
        active, tokens, liq = _classify_market(m)
        if not active:
            continue
        stats["catalog_market_count"] += 1
        for t in tokens:
            catalog_tokens.add(t)
            if t not in by_token or liq > by_token[t]:
                by_token[t] = liq
    stats["catalog_token_count"] = len(catalog_tokens)
    stats["candidates"] = len(by_token)

    # Liquidity floor — drop markets below the floor before ranking
    # so cheap dust markets don't crowd out the cap.
    above_floor = [(tok, liq) for tok, liq in by_token.items() if liq >= _MIN_LIQUIDITY_USD]
    stats["after_liquidity_filter"] = len(above_floor)
    stats["dropped_low_liquidity"] = len(by_token) - len(above_floor)

    # Rank by liquidity desc, take top max_tokens.
    above_floor.sort(key=lambda kv: kv[1], reverse=True)
    selected = above_floor[: _DEFAULT_MAX_TOKENS]
    stats["after_cap"] = len(selected)
    stats["dropped_over_cap"] = max(0, len(above_floor) - _DEFAULT_MAX_TOKENS)

    return [t for t, _ in selected], stats


async def _ensure_subscribed(target_tokens: list[str]) -> tuple[int, int]:
    """Subscribe ``target_tokens`` via the existing PolymarketWSFeed,
    skipping the ones already in the subscription set.

    Returns ``(newly_subscribed, already_subscribed)``.
    """
    if not target_tokens:
        return 0, 0
    try:
        from services.ws_feeds import get_feed_manager

        feed_mgr = get_feed_manager()
        feed = getattr(feed_mgr, "polymarket_feed", None)
        if feed is None:
            logger.debug("No polymarket_feed on feed manager; skipping")
            return 0, 0
        # Polite dedupe — read the feed's existing subscription set
        # so we only send subscribe messages for new tokens.  Falls
        # back to subscribing the full set if the introspection isn't
        # available (the underlying ws_feeds.subscribe() is itself
        # idempotent — it appends to a set).
        already: set[str] = set()
        for attr in ("get_subscribed_assets", "_subscribed_assets"):
            v = getattr(feed, attr, None)
            if callable(v):
                try:
                    res = v()
                    if asyncio.iscoroutine(res):
                        res = await res
                    if isinstance(res, (set, list, tuple)):
                        already = {str(x) for x in res}
                        break
                except Exception:
                    continue
            elif isinstance(v, (set, frozenset)):
                already = {str(x) for x in v}
                break
        new_tokens = [t for t in target_tokens if t not in already]
        if new_tokens:
            try:
                # ws_feeds.subscribe accepts list[str]; chunks
                # internally at 100/batch with a 50ms throttle.
                await feed.subscribe(new_tokens)
            except Exception as exc:
                logger.warning("subscribe call raised %s", exc, exc_info=False)
        return len(new_tokens), len(target_tokens) - len(new_tokens)
    except Exception as exc:
        logger.exception("Subscribe failed")
        _state.last_error = str(exc)
        return 0, 0


async def loop_tick() -> None:
    """One iteration of the proactive subscription loop."""
    global _state
    started = time.monotonic()
    _state.total_runs += 1
    try:
        # Global recording master switch: when OFF, subscribe nothing new
        # (the ingestor's flush gate also drops any in-flight persistence, so
        # no data is written to disk regardless of existing subscriptions).
        from services.recording_control import is_recording_enabled

        _state.recording_enabled = await is_recording_enabled()
        if not _state.recording_enabled:
            _state.last_run_at = time.time()
            _state.last_run_duration_ms = (time.monotonic() - started) * 1000
            _state.last_run_target_count = 0
            _state.last_error = None
            return

        target_tokens, stats = await _gather_target_tokens()
        newly, already = await _ensure_subscribed(target_tokens)

        _state.last_run_at = time.time()
        _state.last_run_duration_ms = (time.monotonic() - started) * 1000
        _state.last_run_subscribed_count = newly + already
        _state.last_run_target_count = len(target_tokens)
        _state.last_run_catalog_market_count = stats["catalog_market_count"]
        _state.last_run_catalog_token_count = stats["catalog_token_count"]
        _state.last_run_dropped_low_liquidity = stats["dropped_low_liquidity"]
        _state.last_run_dropped_over_cap = stats["dropped_over_cap"]
        _state.last_run_already_subscribed = already
        _state.last_error = None

        if newly > 0:
            logger.info(
                "Proactive recorder subscription tick — "
                "catalog=%d markets / %d tokens · target=%d · "
                "newly_subscribed=%d · already=%d · "
                "dropped_low_liq=%d · dropped_over_cap=%d",
                stats["catalog_market_count"],
                stats["catalog_token_count"],
                len(target_tokens),
                newly,
                already,
                stats["dropped_low_liquidity"],
                stats["dropped_over_cap"],
            )
    except Exception as exc:
        logger.exception("Recorder subscription tick failed")
        _state.last_error = str(exc)


async def run_loop() -> None:
    """Long-running loop entry-point — call from host worker."""
    # Stagger startup so the first tick happens after the catalog is
    # warm but before the first scanner pass produces stale opps.
    await asyncio.sleep(20.0)
    while True:
        try:
            await loop_tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Recorder subscription loop error")
        await asyncio.sleep(_LOOP_INTERVAL_SECONDS)
