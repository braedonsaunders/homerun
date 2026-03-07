"""Crypto worker: owns BTC/ETH/SOL/XRP 5m/15m high-frequency data + signal loop.

Runs as a dedicated process and persists worker snapshot + normalized crypto signals.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import deque
from datetime import datetime, timezone

from sqlalchemy import select

from utils.utcnow import utcnow
from config import settings
from models.database import AsyncSessionLocal, Trader
from services.chainlink_feed import get_chainlink_feed
from services.crypto_ws_stream import (
    ack_crypto_ws_update_batches,
    auto_claim_crypto_ws_update_batches,
    ensure_crypto_ws_update_group,
    read_crypto_ws_update_batches,
)
from services.crypto_service import get_crypto_service
from services.data_events import DataEvent
from services.event_dispatcher import event_dispatcher
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.redis_price_cache import redis_price_cache
from services.strategy_signal_bridge import bridge_opportunities_to_signals
from services.strategy_runtime import refresh_strategy_runtime_if_needed
from services.event_bus import event_bus
from services.market_regime import market_regime_classifier
from services.ml_recorder import ml_recorder
from services.worker_state import (
    clear_worker_run_request,
    ensure_worker_control,
    read_worker_control,
    read_worker_snapshot,
    write_worker_snapshot,
)
from utils.logger import setup_logging

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("crypto_worker")

# Keep short in-memory oracle history per asset for chart sparkline payloads.
_MAX_ORACLE_HISTORY_POINTS = 180
_ORACLE_HISTORY_PAYLOAD_POINTS = 80
_oracle_history_by_asset: dict[str, deque[tuple[int, float]]] = {}
_IDLE_INTERVAL_SECONDS = 15
_MIN_LOOP_SLEEP_SECONDS = 0.1
_VIEWER_HEARTBEAT_SECONDS = 20
_WS_FEED_RETRY_INITIAL_SECONDS = 1.0
_WS_FEED_RETRY_MAX_SECONDS = 60.0
_WS_REACTIVE_DEBOUNCE_SECONDS = max(0.01, float(settings.CRYPTO_WS_REACTIVE_DEBOUNCE_SECONDS))
_DEFAULT_WS_TRADE_VOLUME_LOOKBACK_SECONDS = 900.0
_WS_TRADE_VOLUME_LOOKBACK_BY_TIMEFRAME_SECONDS = {
    "5m": 300.0,
    "15m": 900.0,
    "1h": 3600.0,
    "4h": 14400.0,
}
_CHAINLINK_STALE_RESTART_AGE_SECONDS = 45.0
_CHAINLINK_STALE_RESTART_COOLDOWN_SECONDS = 20.0
_CHAINLINK_WARMUP_SECONDS = 20.0
_STREAM_FORCE_FULL_SCAN_INTERVAL_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Market boundary prefetch
# ---------------------------------------------------------------------------
# Polymarket crypto markets open on predictable time boundaries.  When we are
# within ``_BOUNDARY_PREFETCH_WINDOW_SECONDS`` of a boundary, we force-refresh
# the Gamma API so new market IDs are picked up within a single poll cycle.
_BOUNDARY_INTERVALS_SECONDS = [300, 900, 3600, 14400]  # 5m, 15m, 1h, 4h
_BOUNDARY_PREFETCH_WINDOW_SECONDS = 15  # start prefetching this many seconds before boundary
_BOUNDARY_LINGER_WINDOW_SECONDS = 10   # keep force-refreshing this many seconds after boundary
_last_boundary_force_mono: float = 0.0


def _near_market_boundary() -> bool:
    """Return True if current UTC time is within the prefetch window of any
    crypto market time boundary (5m / 15m / 1h / 4h)."""
    now_ts = time.time()
    for interval in _BOUNDARY_INTERVALS_SECONDS:
        seconds_into = now_ts % interval
        seconds_until_next = interval - seconds_into
        # Near UPCOMING boundary
        if seconds_until_next <= _BOUNDARY_PREFETCH_WINDOW_SECONDS:
            return True
        # Just PAST a boundary (new market should exist)
        if seconds_into <= _BOUNDARY_LINGER_WINDOW_SECONDS:
            return True
    return False


def _to_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _chainlink_feed_staleness(feed, *, max_age_seconds: float) -> tuple[bool, float | None, int]:
    if feed is None:
        return True, None, 0
    now_ms = int(utcnow().timestamp() * 1000)
    latest_ms = 0
    samples = 0
    try:
        all_prices = feed.get_all_prices()
    except Exception:
        all_prices = {}
    for oracle in (all_prices or {}).values():
        updated_at = int(getattr(oracle, "updated_at_ms", 0) or 0)
        if updated_at <= 0:
            continue
        latest_ms = max(latest_ms, updated_at)
        samples += 1
    if latest_ms <= 0:
        return True, None, samples
    age_seconds = max(0.0, float(now_ms - latest_ms) / 1000.0)
    return age_seconds > max(0.0, float(max_age_seconds)), age_seconds, samples


def _record_oracle_point(asset: str, timestamp_ms: int, price: float) -> None:
    if not asset:
        return
    history = _oracle_history_by_asset.get(asset)
    if history is None:
        history = deque(maxlen=_MAX_ORACLE_HISTORY_POINTS)
        _oracle_history_by_asset[asset] = history
    if history and history[-1][0] == timestamp_ms:
        history[-1] = (timestamp_ms, price)
    else:
        history.append((timestamp_ms, price))


def _oracle_history_payload(asset: str) -> list[dict]:
    history = _oracle_history_by_asset.get(asset)
    if not history:
        return []
    points = list(history)[-_ORACLE_HISTORY_PAYLOAD_POINTS:]
    return [{"t": t, "p": round(p, 2)} for t, p in points]


def _collect_ws_prices_for_markets(feed_manager, markets: list) -> dict[str, float]:
    """Collect fresh midpoint prices from WS cache for market token IDs."""
    if feed_manager is None or not getattr(feed_manager, "_started", False):
        return {}

    ws_prices: dict[str, float] = {}
    price_cache = getattr(feed_manager, "cache", None)
    if price_cache is None:
        price_cache = getattr(feed_manager, "price_cache", None)
    if price_cache is None:
        return ws_prices

    execution_max_age_seconds = max(0.05, float(settings.WS_EXECUTION_PRICE_STALE_SECONDS))

    for market in markets:
        for token_id in getattr(market, "clob_token_ids", None) or []:
            token = str(token_id or "").strip()
            if not token or len(token) <= 20:
                continue
            try:
                try:
                    is_fresh = price_cache.is_fresh(token, max_age_seconds=execution_max_age_seconds)
                except TypeError:
                    is_fresh = price_cache.is_fresh(token)
                if not is_fresh:
                    continue
                if hasattr(price_cache, "get_mid_price"):
                    mid = price_cache.get_mid_price(token)
                else:
                    mid = price_cache.get_mid(token)
            except Exception as exc:
                logger.debug("WS price cache read failed for token %s: %s", token, exc)
                continue
            if mid is None:
                continue
            try:
                parsed = float(mid)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= parsed <= 1.01):
                continue
            ws_prices[token] = min(1.0, max(0.0, parsed))
    return ws_prices


async def _collect_execution_prices_for_markets(feed_manager, markets: list) -> dict[str, float]:
    ws_prices = _collect_ws_prices_for_markets(feed_manager, markets)
    if ws_prices:
        return ws_prices
    token_ids: list[str] = []
    seen_tokens: set[str] = set()
    for market in markets:
        for token_id in getattr(market, "clob_token_ids", None) or []:
            token = str(token_id or "").strip()
            if not token or len(token) <= 20 or token in seen_tokens:
                continue
            seen_tokens.add(token)
            token_ids.append(token)
    if not token_ids:
        return {}
    strict_stale_seconds = max(0.05, float(settings.WS_EXECUTION_PRICE_STALE_SECONDS))
    redis_rows = await redis_price_cache.read_prices(token_ids, stale_seconds=strict_stale_seconds)
    out: dict[str, float] = {}
    for token in token_ids:
        payload = redis_rows.get(token)
        if not isinstance(payload, dict):
            continue
        mid = _to_float(payload.get("mid"))
        if mid is None or not (0.0 <= mid <= 1.01):
            continue
        out[token] = min(1.0, max(0.0, mid))
    return out


def _market_leg_tokens(market) -> tuple[str | None, str | None]:
    tokens = list(getattr(market, "clob_token_ids", None) or [])
    if not tokens:
        return None, None

    try:
        up_idx = int(getattr(market, "up_token_index", 0) or 0)
    except Exception:
        up_idx = 0
    try:
        down_idx = int(getattr(market, "down_token_index", 1) or 1)
    except Exception:
        down_idx = 1

    up_token = str(tokens[up_idx]).strip() if 0 <= up_idx < len(tokens) and str(tokens[up_idx]).strip() else None
    down_token = (
        str(tokens[down_idx]).strip() if 0 <= down_idx < len(tokens) and str(tokens[down_idx]).strip() else None
    )
    if not up_token and len(tokens) > 0:
        candidate = str(tokens[0]).strip()
        up_token = candidate or None
    if not down_token and len(tokens) > 1:
        candidate = str(tokens[1]).strip()
        down_token = candidate or None
    return up_token, down_token


def _overlay_ws_prices_on_market_row(row: dict, market, ws_prices: dict[str, float]) -> None:
    """Prefer fresh WS prices for up/down legs when available."""
    if not ws_prices:
        return

    up_token, down_token = _market_leg_tokens(market)
    if not up_token and not down_token:
        return

    ws_up = ws_prices.get(up_token) if up_token else None
    ws_down = ws_prices.get(down_token) if down_token else None

    if ws_up is not None and ws_down is not None:
        row["up_price"] = ws_up
        row["down_price"] = ws_down
        row["combined"] = ws_up + ws_down
        return

    if ws_up is not None:
        row["up_price"] = ws_up
        row["down_price"] = min(1.0, max(0.0, 1.0 - ws_up))
        row["combined"] = 1.0
        return

    if ws_down is not None:
        row["down_price"] = ws_down
        row["up_price"] = min(1.0, max(0.0, 1.0 - ws_down))
        row["combined"] = 1.0


def _ws_trade_volume_lookback_seconds(timeframe_value: object) -> float:
    normalized = str(timeframe_value or "").strip().lower()
    return _WS_TRADE_VOLUME_LOOKBACK_BY_TIMEFRAME_SECONDS.get(normalized, _DEFAULT_WS_TRADE_VOLUME_LOOKBACK_SECONDS)


def _overlay_ws_trade_volume_on_market_row(row: dict, market, feed_manager) -> None:
    if feed_manager is None or not getattr(feed_manager, "_started", False):
        return

    price_cache = getattr(feed_manager, "cache", None)
    if price_cache is None:
        price_cache = getattr(feed_manager, "price_cache", None)
    if price_cache is None or not hasattr(price_cache, "get_trade_volume"):
        return

    lookback_seconds = _ws_trade_volume_lookback_seconds(row.get("timeframe") or getattr(market, "timeframe", None))
    up_token, down_token = _market_leg_tokens(market)
    token_ids = {token for token in (up_token, down_token) if token}
    if not token_ids:
        return

    volume_total = 0.0
    observed = False
    for token_id in token_ids:
        try:
            volume_window = price_cache.get_trade_volume(token_id, lookback_seconds=lookback_seconds)
        except TypeError:
            volume_window = price_cache.get_trade_volume(token_id)
        except Exception:
            continue
        if not isinstance(volume_window, dict):
            continue
        token_total = _to_float(volume_window.get("total"))
        if token_total is None or token_total <= 0.0:
            continue
        volume_total += token_total
        observed = True

    if not observed:
        return

    existing_volume_usd = _to_float(row.get("volume_usd"))
    if existing_volume_usd is None or existing_volume_usd <= 0.0 or volume_total > existing_volume_usd:
        row["volume_usd"] = round(volume_total, 6)
        row["volume_usd_source"] = "ws_trade_cache"
        row["volume_usd_lookback_seconds"] = lookback_seconds


async def _sync_ws_subscriptions(feed_manager, markets: list, subscribed_tokens: set[str]) -> set[str]:
    """Keep WS subscriptions aligned with active crypto token IDs."""
    if feed_manager is None or not getattr(feed_manager, "_started", False):
        return subscribed_tokens

    active_tokens: set[str] = set()
    for market in markets:
        for token_id in getattr(market, "clob_token_ids", None) or []:
            token = str(token_id or "").strip()
            if token and len(token) > 20:
                active_tokens.add(token)

    added = sorted(active_tokens - subscribed_tokens)
    removed = sorted(subscribed_tokens - active_tokens)

    try:
        if added:
            await feed_manager.polymarket_feed.subscribe(token_ids=added)
        if removed:
            await feed_manager.polymarket_feed.unsubscribe(token_ids=removed)
    except Exception as exc:
        logger.debug("Crypto WS subscription sync skipped: %s", exc)
        return subscribed_tokens

    return active_tokens


def _restore_price_to_beat_from_snapshot_markets(markets: list[dict]) -> int:
    """Warm ``CryptoService._price_to_beat`` from previous worker snapshot rows."""
    if not markets:
        return 0

    svc = get_crypto_service()
    restored = 0
    now_ts = datetime.now(timezone.utc).timestamp()

    for row in markets:
        if not isinstance(row, dict):
            continue

        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue

        ptb = _to_float(row.get("price_to_beat"))
        if ptb is None:
            continue

        # Skip clearly-expired windows to avoid carrying stale values across days.
        end_time = row.get("end_time")
        if isinstance(end_time, str) and end_time.strip():
            try:
                end_ts = datetime.fromisoformat(end_time.replace("Z", "+00:00")).timestamp()
                if now_ts - end_ts > 1800:
                    continue
            except ValueError:
                pass

        if slug not in svc._price_to_beat:
            svc._price_to_beat[slug] = ptb
            restored += 1

    return restored


def _market_token_ids(market) -> set[str]:
    token_ids = getattr(market, "clob_token_ids", None) or []
    return {
        str(token_id).strip() for token_id in token_ids if str(token_id).strip() and len(str(token_id).strip()) > 20
    }


def _index_market_token_positions(markets: list) -> dict[str, set[int]]:
    index: dict[str, set[int]] = {}
    for market_index, market in enumerate(markets):
        for token_id in _market_token_ids(market):
            positions = index.get(token_id)
            if positions is None:
                positions = set()
                index[token_id] = positions
            positions.add(market_index)
    return index


def _markets_for_updated_tokens(
    markets: list,
    token_index: dict[str, set[int]],
    updated_tokens: set[str],
) -> list:
    if not markets or not updated_tokens:
        return []

    market_positions: set[int] = set()
    for token_id in updated_tokens:
        market_positions.update(token_index.get(token_id, set()))

    if not market_positions:
        return []

    return [markets[i] for i in sorted(market_positions)]


async def _dispatch_crypto_opportunities(
    markets_payload: list[dict],
    *,
    trigger: str,
    run_at: datetime,
    emit_lock: asyncio.Lock,
) -> tuple[int, float]:
    started = time.monotonic()
    if not markets_payload:
        return 0, 0.0

    try:
        async with emit_lock:
            full_source_sweep = str(trigger or "").strip().lower() == "periodic_scan"
            crypto_event = DataEvent(
                event_type="crypto_update",
                source="crypto_worker",
                timestamp=run_at,
                payload={"markets": markets_payload, "trigger": trigger},
            )
            opportunities = await event_dispatcher.dispatch(crypto_event)
            async with AsyncSessionLocal() as session:
                emitted = await bridge_opportunities_to_signals(
                    session,
                    opportunities,
                    source="crypto",
                    sweep_missing=full_source_sweep,
                    refresh_prices=False,
                )
            try:
                await event_bus.publish("crypto_markets_update", {"markets": markets_payload, "trigger": trigger})
            except Exception:
                pass
            return emitted, round(time.monotonic() - started, 3)
    except Exception as exc:
        logger.warning("Crypto signal emission failed (%s): %s", trigger, exc)
        return 0, round(time.monotonic() - started, 3)


async def _is_autotrader_active(session) -> bool:
    """Whether live/paper trader demand is active for crypto fast mode."""
    try:
        from services.trader_orchestrator_state import read_orchestrator_control

        control = await read_orchestrator_control(session)
        if bool(control.get("is_enabled", False)) and not bool(control.get("is_paused", True)):
            return True
    except Exception:
        pass

    try:
        rows = list(
            (
                await session.execute(
                    select(
                        Trader.source_configs_json,
                    ).where(
                        Trader.is_enabled.is_(True),
                        Trader.is_paused.is_(False),
                    )
                )
            ).all()
        )
    except Exception:
        return False

    for (source_configs,) in rows:
        if isinstance(source_configs, list):
            for item in source_configs:
                if not isinstance(item, dict):
                    continue
                if str(item.get("source_key") or "").strip().lower() == "crypto":
                    return True
    return False


def _is_recent_request(requested_run_at, window_seconds: int = _VIEWER_HEARTBEAT_SECONDS) -> bool:
    if not isinstance(requested_run_at, datetime):
        return False
    ts = requested_run_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
    return age <= float(window_seconds)


async def _sleep_with_demand_wake(worker_name: str, sleep_seconds: float) -> None:
    """Sleep up to ``sleep_seconds`` but wake early if fast-mode demand appears."""
    deadline = time.monotonic() + max(0.0, sleep_seconds)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return

        await asyncio.sleep(min(1.0, remaining))

        # Cheap demand probe so active viewers/trader mode don't wait for full idle sleep.
        try:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(session, worker_name, default_interval=2)
                if _is_recent_request(control.get("requested_run_at")):
                    return
                if await _is_autotrader_active(session):
                    return
        except Exception:
            # Best-effort wake checks only; ignore transient DB issues.
            continue


def _build_crypto_market_payload(
    markets: list | None = None,
    *,
    ws_prices: dict[str, float] | None = None,
    feed_manager=None,
) -> list[dict]:
    svc = get_crypto_service()
    feed = get_chainlink_feed()
    markets = markets if markets is not None else svc.get_live_markets()
    svc._update_price_to_beat(markets)

    payload: list[dict] = []
    fetched_at = utcnow()
    fetched_at_iso = fetched_at.isoformat().replace("+00:00", "Z")
    now_ms = fetched_at.timestamp() * 1000
    for market in markets:
        row = market.to_dict()
        row["fetched_at"] = fetched_at_iso
        oracle = feed.get_price(market.asset)
        if oracle:
            row["oracle_price"] = oracle.price
            row["oracle_source"] = getattr(oracle, "source", None)
            row["oracle_updated_at_ms"] = oracle.updated_at_ms
            row["oracle_age_seconds"] = (
                round((now_ms - oracle.updated_at_ms) / 1000, 1) if oracle.updated_at_ms else None
            )
            point_ts = int(oracle.updated_at_ms or now_ms)
            _record_oracle_point(market.asset, point_ts, float(oracle.price))
        else:
            row["oracle_price"] = None
            row["oracle_updated_at_ms"] = None
            row["oracle_age_seconds"] = None

        row["oracle_prices_by_source"] = {}
        for source_name, source_oracle in feed.get_prices_by_source(market.asset).items():
            if not source_oracle:
                continue
            updated_at = getattr(source_oracle, "updated_at_ms", None)
            row["oracle_prices_by_source"][str(source_name)] = {
                "source": source_oracle.source,
                "price": float(source_oracle.price),
                "updated_at_ms": int(updated_at) if updated_at else None,
                "age_seconds": (round((now_ms - int(updated_at)) / 1000, 1) if updated_at else None),
            }

        row["price_to_beat"] = svc._price_to_beat.get(market.slug)
        row["oracle_history"] = _oracle_history_payload(market.asset)
        _overlay_ws_prices_on_market_row(row, market, ws_prices or {})
        _overlay_ws_trade_volume_on_market_row(row, market, feed_manager)

        if oracle:
            market_regime_classifier.update(market.slug, float(oracle.price), fetched_at)
        regime = market_regime_classifier.get_regime(market.slug)
        row["regime"] = regime
        row["regime_params"] = market_regime_classifier.get_regime_params(market.slug)

        payload.append(row)

    market_regime_classifier.prune({m.slug for m in markets})
    return payload


async def _run_loop() -> None:
    logger.info("Crypto worker started")
    worker_name = "crypto"

    try:
        async with AsyncSessionLocal() as session:
            await ensure_all_strategies_seeded(session)
            await refresh_strategy_runtime_if_needed(
                session,
                source_keys=["crypto"],
                force=True,
            )
    except Exception as exc:
        logger.warning("Crypto worker strategy startup sync failed: %s", exc)

    startup_stats = {
        "market_count": 0,
        "signals_emitted_last_run": 0,
        "markets": [],
        "ws_feeds": {
            "healthy": False,
            "started": False,
            "enabled": bool(settings.WS_FEED_ENABLED),
        },
    }
    async with AsyncSessionLocal() as session:
        await ensure_worker_control(session, worker_name, default_interval=2)
        previous_snapshot = await read_worker_snapshot(session, worker_name)
        previous_stats = previous_snapshot.get("stats") if isinstance(previous_snapshot.get("stats"), dict) else {}
        previous_markets = previous_stats.get("markets") if isinstance(previous_stats.get("markets"), list) else []
        if previous_markets:
            startup_stats = {
                "market_count": len(previous_markets),
                "signals_emitted_last_run": int(previous_stats.get("signals_emitted_last_run") or 0),
                "markets": previous_markets,
                "ws_feeds": startup_stats.get("ws_feeds", {"healthy": False, "started": False}),
            }

        restored = _restore_price_to_beat_from_snapshot_markets(previous_markets)
        if restored:
            logger.info("Restored %s price-to-beat entries from last snapshot", restored)

        await write_worker_snapshot(
            session,
            worker_name,
            running=True,
            enabled=True,
            current_activity="Crypto worker started; first cycle pending.",
            interval_seconds=2,
            last_run_at=None,
            last_error=None,
            stats=startup_stats,
        )

    # Chainlink oracle feed is owned by this worker (not API).
    chainlink_feed = get_chainlink_feed()
    try:
        await chainlink_feed.start()
    except Exception as exc:
        logger.warning("Chainlink feed start failed (continuing): %s", exc)
    chainlink_feed_started_mono = time.monotonic()
    chainlink_restart_allowed_after = 0.0

    # Direct Binance WebSocket feed (lower latency than RTDS relay).
    _binance_feed = None
    _binance_feed_started_mono = 0.0
    _BINANCE_STALE_RESTART_AGE_SECONDS = 60.0
    _BINANCE_WARMUP_SECONDS = 15.0
    _binance_restart_allowed_after = 0.0

    if settings.BINANCE_WS_ENABLED:
        from services.binance_feed import get_binance_feed as _get_binance_feed

        _binance_feed = _get_binance_feed()

        def _on_binance_direct_update(asset: str, mid: float, bid: float, ask: float, ts_ms: int) -> None:
            try:
                chainlink_feed.update_from_binance_direct(asset, mid, bid, ask, ts_ms)
            except Exception:
                pass

        _binance_feed.on_update(_on_binance_direct_update)
        try:
            await _binance_feed.start()
            _binance_feed_started_mono = time.monotonic()
        except Exception as exc:
            logger.warning("Binance direct feed start failed (continuing): %s", exc)

    # ML recorder prune cadence: check once per hour.
    _ml_prune_interval_seconds = 3600.0
    _ml_last_prune_mono = 0.0

    feed_manager = None
    ws_feeds_running = False
    subscribed_tokens: set[str] = set()
    ws_feed_retry_delay_seconds = _WS_FEED_RETRY_INITIAL_SECONDS
    ws_feed_next_retry_monotonic = 0.0
    token_to_market_indices: dict[str, set[int]] = {}
    current_markets: list = []
    ws_changed_tokens: set[str] = set()
    ws_reactive_task: asyncio.Task[None] | None = None
    ws_reactive_enabled = True
    ws_reactive_callback_registered = False
    emit_lock = asyncio.Lock()
    stream_consumer_name = f"crypto-{os.getpid()}-{utcnow().strftime('%Y%m%d%H%M%S%f')}"
    stream_claim_cursor = "0-0"
    stream_last_claim_run_at: datetime | None = None
    stream_batches_consumed = 0
    stream_tokens_consumed = 0
    stream_dispatches = 0
    last_full_scan_monotonic = 0.0
    startup_markets = startup_stats.get("markets")
    last_full_markets_payload: list[dict] = list(startup_markets) if isinstance(startup_markets, list) else []

    async def _ensure_chainlink_feed_healthy() -> dict[str, object]:
        nonlocal chainlink_feed_started_mono
        nonlocal chainlink_restart_allowed_after
        feed_started = bool(getattr(chainlink_feed, "started", False))
        stale, age_seconds, samples = _chainlink_feed_staleness(
            chainlink_feed,
            max_age_seconds=_CHAINLINK_STALE_RESTART_AGE_SECONDS,
        )
        now_mono = time.monotonic()
        warm = (now_mono - chainlink_feed_started_mono) < _CHAINLINK_WARMUP_SECONDS
        should_restart = ((not feed_started) or (stale and not warm)) and now_mono >= chainlink_restart_allowed_after
        restarted = False
        if should_restart:
            chainlink_restart_allowed_after = now_mono + _CHAINLINK_STALE_RESTART_COOLDOWN_SECONDS
            try:
                if feed_started:
                    await chainlink_feed.stop()
                await chainlink_feed.start()
                chainlink_feed_started_mono = time.monotonic()
                restarted = True
            except Exception as exc:
                logger.warning("Chainlink feed health restart failed (continuing): %s", exc)
            feed_started = bool(getattr(chainlink_feed, "started", False))
        return {
            "started": feed_started,
            "stale": bool(stale and not warm),
            "age_seconds": age_seconds,
            "samples": int(samples),
            "restarted": restarted,
        }

    async def _ensure_binance_feed_healthy() -> dict[str, object]:
        nonlocal _binance_feed_started_mono
        nonlocal _binance_restart_allowed_after
        if _binance_feed is None or not settings.BINANCE_WS_ENABLED:
            return {"started": False, "enabled": False}
        feed_started = bool(getattr(_binance_feed, "started", False))
        now_mono = time.monotonic()
        last_ms = getattr(_binance_feed, "last_update_ms", 0)
        if last_ms > 0:
            age_seconds = max(0.0, (int(time.time() * 1000) - last_ms) / 1000.0)
        else:
            age_seconds = now_mono - _binance_feed_started_mono
        stale = age_seconds > _BINANCE_STALE_RESTART_AGE_SECONDS
        warm = (now_mono - _binance_feed_started_mono) < _BINANCE_WARMUP_SECONDS
        should_restart = ((not feed_started) or (stale and not warm)) and now_mono >= _binance_restart_allowed_after
        restarted = False
        if should_restart:
            _binance_restart_allowed_after = now_mono + 30.0
            try:
                if feed_started:
                    await _binance_feed.stop()
                await _binance_feed.start()
                _binance_feed_started_mono = time.monotonic()
                restarted = True
            except Exception as exc:
                logger.warning("Binance direct feed health restart failed: %s", exc)
            feed_started = bool(getattr(_binance_feed, "started", False))
        return {
            "started": feed_started,
            "stale": bool(stale and not warm),
            "age_seconds": round(age_seconds, 1),
            "restarted": restarted,
        }

    async def _ensure_ws_feeds_running() -> None:
        nonlocal feed_manager
        nonlocal ws_feeds_running
        nonlocal ws_feed_retry_delay_seconds
        nonlocal ws_feed_next_retry_monotonic
        nonlocal ws_reactive_callback_registered
        if bool(getattr(settings, "MARKET_DATA_WORKER_OWNS_WS", True)):
            return
        if ws_feeds_running or not settings.WS_FEED_ENABLED:
            return
        now_monotonic = time.monotonic()
        if now_monotonic < ws_feed_next_retry_monotonic:
            return
        try:
            from services.ws_feeds import get_feed_manager
            from services.polymarket import polymarket_client

            if feed_manager is None:
                feed_manager = get_feed_manager()

                async def _http_book_fallback(token_id: str):
                    try:
                        return await polymarket_client.get_order_book(token_id)
                    except Exception:
                        return None

                feed_manager.set_http_fallback(_http_book_fallback)

            if not feed_manager._started:
                await feed_manager.start()
            if not ws_reactive_callback_registered:
                feed_manager.cache.add_on_update_callback(_on_ws_price_update)
                ws_reactive_callback_registered = True
            ws_feeds_running = True
            ws_feed_retry_delay_seconds = _WS_FEED_RETRY_INITIAL_SECONDS
            ws_feed_next_retry_monotonic = 0.0
            logger.info("Crypto worker WebSocket feeds started")
        except Exception as exc:
            retry_in = min(_WS_FEED_RETRY_MAX_SECONDS, max(1.0, float(ws_feed_retry_delay_seconds)))
            ws_feed_next_retry_monotonic = time.monotonic() + retry_in
            ws_feed_retry_delay_seconds = min(_WS_FEED_RETRY_MAX_SECONDS, retry_in * 2.0)
            logger.warning(
                "Crypto worker WS feeds failed to start (non-critical): %s; retrying in %.1fs",
                exc,
                retry_in,
            )

    def _snapshot_ws_feed_status() -> dict[str, object]:
        if not settings.WS_FEED_ENABLED:
            return {"healthy": False, "started": False, "enabled": False}
        if bool(getattr(settings, "MARKET_DATA_WORKER_OWNS_WS", True)):
            return {
                "healthy": True,
                "started": False,
                "enabled": True,
                "owner": "market_data_worker",
            }
        if not ws_feeds_running or feed_manager is None:
            return {"healthy": False, "started": False, "enabled": True}
        try:
            return feed_manager.health_check()
        except Exception:
            return {"healthy": False, "started": bool(ws_feeds_running), "enabled": True}

    startup_stats["ws_feeds"] = _snapshot_ws_feed_status()
    startup_stats["chainlink_feed"] = await _ensure_chainlink_feed_healthy()
    startup_stats["binance_feed"] = await _ensure_binance_feed_healthy()

    def _on_ws_price_update(token_id: str, mid: float, bid: float, ask: float) -> None:
        nonlocal ws_reactive_task
        if not ws_reactive_enabled or mid <= 0:
            return
        token = str(token_id or "").strip()
        if not token:
            return
        ws_changed_tokens.add(token)
        if ws_reactive_task is not None and not ws_reactive_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            ws_reactive_task = loop.create_task(_drain_ws_price_updates())
        except RuntimeError:
            pass

    async def _drain_ws_price_updates() -> None:
        while True:
            await asyncio.sleep(_WS_REACTIVE_DEBOUNCE_SECONDS)
            if not ws_reactive_enabled:
                ws_changed_tokens.clear()
                return
            if not current_markets:
                ws_changed_tokens.clear()
                return
            if not ws_changed_tokens:
                return
            tokens = set(ws_changed_tokens)
            ws_changed_tokens.clear()
            selected_markets = _markets_for_updated_tokens(current_markets, token_to_market_indices, tokens)
            if not selected_markets:
                continue

            ws_prices = await _collect_execution_prices_for_markets(feed_manager, selected_markets)
            payload = _build_crypto_market_payload(
                selected_markets,
                ws_prices=ws_prices,
                feed_manager=feed_manager,
            )
            await _dispatch_crypto_opportunities(
                payload,
                trigger="crypto_ws",
                run_at=utcnow(),
                emit_lock=emit_lock,
            )

    async def _read_ws_stream_tokens(timeout_ms: int) -> tuple[set[str], list[str]]:
        nonlocal stream_claim_cursor
        nonlocal stream_last_claim_run_at
        if not bool(getattr(settings, "MARKET_DATA_WORKER_OWNS_WS", True)):
            return set(), []
        normalized_timeout = max(1, int(timeout_ms))
        entry_ids: list[str] = []
        token_ids: set[str] = set()
        claim_interval_seconds = max(
            0.1,
            float(getattr(settings, "CRYPTO_WS_UPDATE_STREAM_CLAIM_INTERVAL_SECONDS", 0.5) or 0.5),
        )
        now_claim = utcnow()
        claim_due = (
            stream_last_claim_run_at is None
            or (now_claim - stream_last_claim_run_at).total_seconds() >= claim_interval_seconds
        )
        if claim_due:
            stream_claim_cursor, claimed_rows = await auto_claim_crypto_ws_update_batches(
                consumer=stream_consumer_name,
                min_idle_ms=int(getattr(settings, "CRYPTO_WS_UPDATE_STREAM_CLAIM_IDLE_MS", 1000)),
                start_id=stream_claim_cursor,
                count=int(getattr(settings, "CRYPTO_WS_UPDATE_STREAM_CLAIM_READ_COUNT", 1000)),
            )
            stream_last_claim_run_at = now_claim
            for entry_id, payload in claimed_rows:
                entry_ids.append(str(entry_id))
                for token in payload.get("token_ids") or []:
                    token_id = str(token or "").strip()
                    if token_id:
                        token_ids.add(token_id)
            if token_ids:
                return token_ids, entry_ids
        rows = await read_crypto_ws_update_batches(
            consumer=stream_consumer_name,
            block_ms=normalized_timeout,
            count=int(getattr(settings, "CRYPTO_WS_UPDATE_STREAM_READ_COUNT", 1000)),
            include_pending=False,
        )
        for entry_id, payload in rows:
            entry_ids.append(str(entry_id))
            for token in payload.get("token_ids") or []:
                token_id = str(token or "").strip()
                if token_id:
                    token_ids.add(token_id)
        return token_ids, entry_ids

    await _ensure_ws_feeds_running()
    if bool(getattr(settings, "MARKET_DATA_WORKER_OWNS_WS", True)):
        if not await ensure_crypto_ws_update_group():
            logger.warning(
                "Crypto WS update stream group ensure failed; worker will continue with periodic polling fallback"
            )

    while True:
        async with AsyncSessionLocal() as session:
            control = await read_worker_control(session, worker_name, default_interval=2)
            autotrader_active = await _is_autotrader_active(session)
            try:
                await refresh_strategy_runtime_if_needed(
                    session,
                    source_keys=["crypto"],
                )
            except Exception as exc:
                logger.warning("Crypto worker strategy refresh check failed: %s", exc)

        configured_interval = max(1, min(60, int(control.get("interval_seconds") or 2)))
        paused = bool(control.get("is_paused", False))
        enabled = bool(control.get("is_enabled", True))
        requested_run_at = control.get("requested_run_at")
        viewer_active = _is_recent_request(requested_run_at)
        fast_mode = bool(viewer_active or autotrader_active)
        ws_reactive_enabled = bool(enabled and not paused)
        interval = configured_interval if fast_mode else max(configured_interval, _IDLE_INTERVAL_SECONDS)
        chainlink_feed_status = await _ensure_chainlink_feed_healthy()
        binance_feed_status = await _ensure_binance_feed_healthy()

        if (not enabled or paused) and not viewer_active:
            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    worker_name,
                    running=True,
                    enabled=enabled and not paused,
                    current_activity="Paused" if paused else "Disabled",
                    interval_seconds=interval,
                    last_run_at=None,
                    stats={
                        "market_count": 0,
                        "signals_emitted_last_run": 0,
                        "ws_stream_batches_consumed": int(stream_batches_consumed),
                        "ws_stream_tokens_consumed": int(stream_tokens_consumed),
                        "ws_stream_dispatches": int(stream_dispatches),
                        "markets": [],
                        "ws_feeds": _snapshot_ws_feed_status(),
                        "chainlink_feed": chainlink_feed_status,
                        "binance_feed": binance_feed_status,
                    },
                )
            ws_reactive_enabled = False
            ws_changed_tokens.clear()
            await asyncio.sleep(min(5, interval))
            continue

        # Garbage-collect stale run requests so one-off triggers do not pin
        # fast mode forever when the crypto page is no longer active.
        if requested_run_at is not None and not viewer_active:
            try:
                async with AsyncSessionLocal() as session:
                    await clear_worker_run_request(session, worker_name)
            except Exception:
                pass
            ws_changed_tokens.clear()

        await _ensure_ws_feeds_running()

        run_at = utcnow()
        markets_payload: list[dict] = []
        emitted = 0
        err_text = None
        started_at = time.monotonic()
        ws_prices: dict[str, float] = {}
        dispatch_trigger = "periodic_scan"
        stream_entry_ids: list[str] = []
        reactive_stream_triggered = False
        boundary_force = False

        try:
            if fast_mode and bool(getattr(settings, "MARKET_DATA_WORKER_OWNS_WS", True)):
                stream_tokens, stream_entry_ids = await _read_ws_stream_tokens(
                    int(getattr(settings, "CRYPTO_WS_UPDATE_STREAM_BLOCK_MS", 10))
                )
                if stream_entry_ids:
                    stream_batches_consumed += len(stream_entry_ids)
                    stream_tokens_consumed += len(stream_tokens)
            else:
                stream_tokens = set()

            svc = get_crypto_service()
            full_scan_required = True
            force_full_scan_interval_seconds = max(
                _STREAM_FORCE_FULL_SCAN_INTERVAL_SECONDS,
                float(configured_interval),
            )
            force_full_scan_due = (time.monotonic() - last_full_scan_monotonic) >= force_full_scan_interval_seconds
            if stream_tokens and current_markets:
                selected_markets = _markets_for_updated_tokens(current_markets, token_to_market_indices, stream_tokens)
                if selected_markets:
                    ws_prices = await _collect_execution_prices_for_markets(feed_manager, selected_markets)
                    markets_payload = _build_crypto_market_payload(
                        selected_markets,
                        ws_prices=ws_prices,
                        feed_manager=feed_manager,
                    )
                    dispatch_trigger = "crypto_ws_stream"
                    emitted, dispatch_elapsed = await _dispatch_crypto_opportunities(
                        markets_payload,
                        trigger=dispatch_trigger,
                        run_at=run_at,
                        emit_lock=emit_lock,
                    )
                    elapsed = max(round(time.monotonic() - started_at, 3), dispatch_elapsed)
                    full_scan_required = False
                    reactive_stream_triggered = True
                    stream_dispatches += 1
            if not full_scan_required and force_full_scan_due:
                full_scan_required = True

            if full_scan_required:
                # Force-refresh near market time boundaries so new market IDs are
                # discovered within a single poll cycle instead of waiting for TTL.
                if fast_mode and _near_market_boundary():
                    global _last_boundary_force_mono
                    now_mono = time.monotonic()
                    # Rate-limit force-refreshes to at most once per 2 seconds
                    if (now_mono - _last_boundary_force_mono) >= 2.0:
                        boundary_force = True
                        _last_boundary_force_mono = now_mono
                markets = await asyncio.to_thread(svc.get_live_markets, boundary_force)
                if markets is None:
                    markets = []
                if ws_feeds_running and feed_manager is not None:
                    subscribed_tokens = await _sync_ws_subscriptions(feed_manager, markets, subscribed_tokens)
                current_markets = markets
                token_to_market_indices = _index_market_token_positions(markets)
                ws_prices = await _collect_execution_prices_for_markets(feed_manager, markets)
                markets_payload = _build_crypto_market_payload(
                    markets,
                    ws_prices=ws_prices,
                    feed_manager=feed_manager,
                )

                # Log when boundary prefetch discovers new markets.
                if boundary_force and markets_payload:
                    prev_slugs = {
                        str(m.get("slug") or "") for m in (startup_stats.get("markets") or []) if isinstance(m, dict)
                    }
                    new_slugs = {str(m.get("slug") or "") for m in markets_payload if isinstance(m, dict)} - prev_slugs
                    if new_slugs:
                        logger.info(
                            "Boundary prefetch discovered %d new market(s): %s",
                            len(new_slugs),
                            ", ".join(sorted(new_slugs)[:6]),
                        )

                dispatch_trigger = "boundary_prefetch" if boundary_force else "periodic_scan"
                emitted, dispatch_elapsed = await _dispatch_crypto_opportunities(
                    markets_payload,
                    trigger=dispatch_trigger,
                    run_at=run_at,
                    emit_lock=emit_lock,
                )
                elapsed = max(round(time.monotonic() - started_at, 3), dispatch_elapsed)
                last_full_scan_monotonic = time.monotonic()
                last_full_markets_payload = list(markets_payload)

            if stream_entry_ids:
                acknowledged = await ack_crypto_ws_update_batches(stream_entry_ids)
                if acknowledged < len(stream_entry_ids):
                    logger.warning(
                        "Crypto WS update stream ack partial requested=%d acknowledged=%d",
                        len(stream_entry_ids),
                        int(acknowledged),
                    )

            # Record ML training snapshots if enabled.
            try:
                await ml_recorder.maybe_record(markets_payload)
            except Exception:
                pass

            # Periodic ML snapshot pruning (once per hour).
            if (time.monotonic() - _ml_last_prune_mono) > _ml_prune_interval_seconds:
                try:
                    await ml_recorder.prune_old_snapshots()
                except Exception:
                    pass
                _ml_last_prune_mono = time.monotonic()

            snapshot_markets = (
                last_full_markets_payload
                if dispatch_trigger == "crypto_ws_stream" and last_full_markets_payload
                else markets_payload
            )

            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    worker_name,
                    running=True,
                    enabled=True,
                    current_activity="Idle - waiting for next crypto cycle.",
                    interval_seconds=interval,
                    last_run_at=run_at,
                    last_error=None,
                    stats={
                        "market_count": len(snapshot_markets),
                        "active_dispatch_market_count": len(markets_payload),
                        "signals_emitted_last_run": int(emitted),
                        "run_duration_seconds": elapsed,
                        "trigger": dispatch_trigger,
                        "fast_mode": fast_mode,
                        "boundary_force_refresh": boundary_force,
                        "reactive_stream_triggered": reactive_stream_triggered,
                        "ws_stream_batches_consumed": int(stream_batches_consumed),
                        "ws_stream_tokens_consumed": int(stream_tokens_consumed),
                        "ws_stream_dispatches": int(stream_dispatches),
                        "ws_prices_used": int(bool(ws_prices)),
                        "ws_token_prices": len(ws_prices),
                        "markets": snapshot_markets,
                        "ws_feeds": _snapshot_ws_feed_status(),
                        "chainlink_feed": chainlink_feed_status,
                        "binance_feed": binance_feed_status,
                    },
                )

            logger.info(
                "Crypto cycle complete: trigger=%s markets=%s signals=%s duration=%.3fs fast_mode=%s sleep=%ss",
                dispatch_trigger,
                len(markets_payload),
                emitted,
                elapsed,
                fast_mode,
                interval,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            err_text = str(exc)
            elapsed = round(time.monotonic() - started_at, 3)
            logger.exception("Crypto worker cycle failed: %s", exc)
            async with AsyncSessionLocal() as session:
                await write_worker_snapshot(
                    session,
                    worker_name,
                    running=True,
                    enabled=True,
                    current_activity=f"Last crypto cycle error: {exc}",
                    interval_seconds=interval,
                    last_run_at=run_at,
                    last_error=str(exc),
                    stats={
                        "market_count": len(markets_payload),
                        "signals_emitted_last_run": int(emitted),
                        "run_duration_seconds": elapsed,
                        "trigger": dispatch_trigger,
                        "fast_mode": fast_mode,
                        "reactive_stream_triggered": reactive_stream_triggered,
                        "ws_stream_batches_consumed": int(stream_batches_consumed),
                        "ws_stream_tokens_consumed": int(stream_tokens_consumed),
                        "ws_stream_dispatches": int(stream_dispatches),
                        "ws_prices_used": int(bool(ws_prices)),
                        "ws_token_prices": len(ws_prices),
                        "markets": markets_payload,
                        "ws_feeds": _snapshot_ws_feed_status(),
                        "chainlink_feed": chainlink_feed_status,
                        "binance_feed": binance_feed_status,
                    },
                )

        if err_text:
            sleep_for = 0.5
        elif fast_mode and reactive_stream_triggered:
            sleep_for = _MIN_LOOP_SLEEP_SECONDS
        elif fast_mode and boundary_force:
            # Near a market boundary — poll as fast as possible to pick up new IDs.
            sleep_for = 0.2
        elif fast_mode:
            # Keep fast mode responsive while avoiding back-to-back cache-only spins.
            sleep_for = 0.5
        else:
            # Keep the cadence close to configured interval between cycle starts,
            # not interval + run_duration.
            sleep_for = max(_MIN_LOOP_SLEEP_SECONDS, interval - elapsed)

        if err_text or fast_mode:
            await asyncio.sleep(sleep_for)
        else:
            await _sleep_with_demand_wake(worker_name, sleep_for)


async def start_loop() -> None:
    """Run the crypto worker loop (called from API process lifespan)."""
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Crypto worker shutting down")
    finally:
        try:
            await get_chainlink_feed().stop()
        except Exception:
            pass
        try:
            if settings.WS_FEED_ENABLED:
                from services.ws_feeds import get_feed_manager

                feed_manager = get_feed_manager()
                if feed_manager._started:
                    await feed_manager.stop()
        except Exception:
            pass
