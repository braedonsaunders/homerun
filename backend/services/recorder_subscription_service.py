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
  4. Subscribes the top ``ws_max_tokens`` liquidity-ranked tokens
     (plus everything the orchestrator is actively trading) onto the
     ISOLATED RecordingFeedManager pool (services.recording_feed) — a
     sharded WS connection pool + cache fully decoupled from the
     orchestrator's low-latency trading socket.  The long tail is covered
     by the pool's periodic REST baseline pass (carry-forward), so every
     active market is recorded without live-subscribing the whole universe.

Existing 10-minute idle eviction in the recording pool automatically prunes
truly-dead tokens from the subscription set so the cap stays
saturated with markets that actually move.

The WS cap exists because live-subscribing the whole ~37k-token universe
floods the parquet sink — the delta volume saturates the encoder and starves
the event loop / DB pool, which can harm the orchestrator.  So the policy
SPLITS breadth from fidelity: ``ws_max_tokens`` (default 5000) bounds the LIVE
WS subscription to the liquidity-ranked head that actually moves, while the
REST baseline (recording_feed) still snapshots EVERY active market on a
periodic, paced pass for carry-forward.  A strategy authored LATER can still
be backtested on markets no strategy subscribed to at capture time — at
baseline (carry-forward) fidelity for the tail, full WS fidelity for the head
plus anything traded.  Operators tune ``ws_max_tokens`` / ``min_liquidity_usd``
via the recording config (Data Lab) or the ``HOMERUN_RECORDER_WS_MAX_TOKENS`` /
``HOMERUN_RECORDER_MIN_LIQUIDITY`` env vars.
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
# These are only the FAILURE fallbacks used when the operator recording config
# can't be read on a given tick.  The live source of truth for coverage is
# ``recording_control._CONFIG_DEFAULTS`` (read each tick) — kept in sync here so
# even a transient config-read error keeps capture broad rather than narrowing
# the universe.  See recording_control for the broad-by-default rationale.
_DEFAULT_MAX_TOKENS = int(os.environ.get("HOMERUN_RECORDER_MAX_TOKENS", "40000"))
_MIN_LIQUIDITY_USD = float(os.environ.get("HOMERUN_RECORDER_MIN_LIQUIDITY", "1.0"))
# WS tick-fidelity cap (NOT the broad breadth).  Only the top
# ``ws_max_tokens`` liquidity-ranked markets get a LIVE WS subscription; the
# long tail is covered by the periodic REST baseline (carry-forward) instead.
# Bounding the live WS set is what keeps broad recording from flooding the
# parquet sink and starving the orchestrator.  Live source of truth is the
# recording config (Data Lab); this env value is only the config-read fallback.
_DEFAULT_WS_MAX_TOKENS = int(os.environ.get("HOMERUN_RECORDER_WS_MAX_TOKENS", "5000"))
# Safety cap on the DB-derived "what we trade" market lookup used when the
# recorder runs in its own process (no in-process trading feed to read).
_MAX_TRADED_MARKETS = 2000


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
    # Effective coverage knobs applied on the last tick — these reflect the
    # operator recording config (or the env defaults when unset), not the
    # static module constants.
    effective_max_tokens: int = _DEFAULT_MAX_TOKENS
    effective_min_liquidity_usd: float = _MIN_LIQUIDITY_USD


_state = _ServiceState()


def get_status() -> dict[str, Any]:
    """Status snapshot for the Data Lab UI / agent diagnostics."""
    return {
        # Effective (operator-config-aware) values applied last tick; the
        # ``*_default`` keys expose the static env fallback for reference.
        "max_tokens": _state.effective_max_tokens,
        "min_liquidity_usd": _state.effective_min_liquidity_usd,
        "max_tokens_default": _DEFAULT_MAX_TOKENS,
        "min_liquidity_usd_default": _MIN_LIQUIDITY_USD,
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


async def _gather_target_tokens(
    *,
    max_tokens: int | None = None,
    min_liquidity_usd: float | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Read MarketCatalog and return the token list to subscribe.

    ``max_tokens`` / ``min_liquidity_usd`` come from the operator recording
    config (``services.recording_control.get_recorder_config``); when not
    supplied they fall back to the env-var defaults.  Read each tick so
    operator changes take effect without a restart.

    Returns ``(tokens, stats)`` where ``stats`` carries the funnel
    counts for the status endpoint.
    """
    from services.shared_state import _read_market_catalog_file

    cap = _DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens)
    liq_floor = _MIN_LIQUIDITY_USD if min_liquidity_usd is None else float(min_liquidity_usd)

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
    above_floor = [(tok, liq) for tok, liq in by_token.items() if liq >= liq_floor]
    stats["after_liquidity_filter"] = len(above_floor)
    stats["dropped_low_liquidity"] = len(by_token) - len(above_floor)

    # Rank by liquidity desc, take top max_tokens.
    above_floor.sort(key=lambda kv: kv[1], reverse=True)
    selected = above_floor[:cap]
    stats["after_cap"] = len(selected)
    stats["dropped_over_cap"] = max(0, len(above_floor) - cap)

    return [t for t, _ in selected], stats


def _trading_feed_subscribed_tokens() -> list[str]:
    """The TRADING feed's current subscription set, read IN-PROCESS — the markets
    the orchestrator is actively watching/trading (open positions, hot
    opportunities, crypto).  Only populated when recording is COLOCATED with
    trading (the ``all`` plane); on the dedicated ``recording`` plane the trading
    feed lives in another process and this returns ``[]`` (the DB-derive below
    covers that topology).  Best-effort — never raises into the loop."""
    try:
        from services.ws_feeds import get_feed_manager

        feed = getattr(get_feed_manager(), "polymarket_feed", None)
        if feed is None:
            return []
        assets = getattr(feed, "_subscribed_assets", None)
        if isinstance(assets, (set, frozenset)):
            return [str(t) for t in assets]
        getter = getattr(feed, "get_subscribed_assets", None)
        if callable(getter):
            res = getter()
            if isinstance(res, (set, list, tuple)):
                return [str(t) for t in res]
    except Exception:  # noqa: BLE001
        return []
    return []


async def _db_traded_tokens() -> list[str]:
    """clob_token_ids of markets with a live ``trader_orders`` row — the durable,
    cross-process "what we trade" set.  Used on the dedicated ``recording`` plane
    where the in-process trading feed is unavailable: derive the traded markets
    from the DB (reusing the canonical ``OPEN_ORDER_STATUSES``) and map them to
    tokens through the shared market catalog.  Preserves the backtest-fidelity
    guarantee (FULL WS capture of illiquid markets we trade) across the process
    split.  Best-effort; never raises into the loop."""
    from sqlalchemy import select

    from models.database import AsyncSessionLocal, TraderOrder
    from services.shared_state import _read_market_catalog_file
    from services.trader_orchestrator_state import OPEN_ORDER_STATUSES

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(TraderOrder.market_id)
                .where(TraderOrder.status.in_(tuple(OPEN_ORDER_STATUSES)))
                .limit(_MAX_TRADED_MARKETS)
            )
        ).all()
    traded_market_ids = {str(m) for (m,) in rows if m}
    if not traded_market_ids:
        return []

    catalog = _read_market_catalog_file()
    if catalog is None:
        return []
    _events, markets, _meta = catalog
    out: list[str] = []
    seen: set[str] = set()
    for m in markets:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or m.get("market_id") or m.get("condition_id") or "").strip()
        if not mid or mid not in traded_market_ids:
            continue
        _active, toks, _liq = _classify_market(m)
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


async def _trading_subscribed_tokens() -> list[str]:
    """Tokens the orchestrator is actively trading, unioned into the recording
    set so backtests have FULL WS book data for markets we trade — including
    illiquid ones below the liquidity floor (the ``tail_end_carry`` "0 backtest
    fills" gap).  Topology-aware: the in-process trading feed when colocated, a
    DB-derive when the recorder runs in its own process.  Best-effort."""
    tokens = _trading_feed_subscribed_tokens()
    if tokens:
        return tokens
    try:
        return await _db_traded_tokens()
    except Exception:  # noqa: BLE001
        return []


async def _ensure_subscribed(target_tokens: list[str]) -> tuple[int, int]:
    """Subscribe ``target_tokens`` onto the ISOLATED RecordingFeedManager pool
    (services.recording_feed) — NOT the trading feed.  Recording rides its own
    sharded connection pool + cache, so the broad recording set can never load
    the orchestrator's low-latency trading socket.  Skips already-subscribed.

    Returns ``(newly_subscribed, already_subscribed)``.
    """
    if not target_tokens:
        return 0, 0
    try:
        from services.recording_feed import get_recording_feed_manager

        pool = get_recording_feed_manager()
        already = pool.get_subscribed_assets()
        new_tokens = [t for t in target_tokens if t not in already]
        if new_tokens:
            # pool.subscribe shards across connections + dedupes per feed.
            await pool.subscribe(new_tokens)
        return len(new_tokens), len(target_tokens) - len(new_tokens)
    except Exception as exc:
        logger.exception("Recording-pool subscribe failed")
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
        from services.recording_control import get_recorder_config, is_recording_enabled

        _state.recording_enabled = await is_recording_enabled()
        if not _state.recording_enabled:
            _state.last_run_at = time.time()
            _state.last_run_duration_ms = (time.monotonic() - started) * 1000
            _state.last_run_target_count = 0
            _state.last_error = None
            return

        # Read operator coverage knobs live each tick (fallback to env
        # defaults on any error) so changes take effect without a restart.
        # The LIVE WS subscription is bounded by ``ws_max_tokens`` (NOT
        # ``max_tokens``, which is the broad REST-baseline breadth).  Bounding the
        # WS set is what keeps the delta volume off the orchestrator's back; the
        # baseline still records the long tail via carry-forward.
        cap = _DEFAULT_WS_MAX_TOKENS
        liq_floor = _MIN_LIQUIDITY_USD
        try:
            cfg = await get_recorder_config()
            cap = int(cfg.get("ws_max_tokens", _DEFAULT_WS_MAX_TOKENS))
            liq_floor = float(cfg.get("min_liquidity_usd", _MIN_LIQUIDITY_USD))
        except Exception:  # pragma: no cover — keep env defaults on config error
            pass
        _state.effective_max_tokens = cap
        _state.effective_min_liquidity_usd = liq_floor

        target_tokens, stats = await _gather_target_tokens(
            max_tokens=cap,
            min_liquidity_usd=liq_floor,
        )
        # Backtest-fidelity guarantee: ALWAYS record what the orchestrator is
        # actually trading, even when those markets are too illiquid to make the
        # liquidity-ranked broad set (e.g. tail_end_carry's near-resolution
        # favorites — the exact gap that produced 0 backtest fills).  Union the
        # actively-traded tokens (the in-process trading feed when colocated, or
        # a DB-derive when the recorder runs in its own plane) into the target.
        traded = await _trading_subscribed_tokens()
        if traded:
            target_tokens = list(dict.fromkeys([*target_tokens, *traded]))
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
