"""Crypto worker: owns BTC/ETH/SOL/XRP 5m/15m high-frequency data + signal loop.

Runs as a dedicated process and persists worker snapshot + normalized crypto signals.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

from utils.utcnow import utcnow
from config import settings
from models.database import AsyncSessionLocal, init_database
from services.chainlink_feed import get_chainlink_feed
from services.crypto_service import get_crypto_service
from services.signal_bus import emit_crypto_market_signals
from services.worker_state import (
    clear_worker_run_request,
    ensure_worker_control,
    read_worker_control,
    read_worker_snapshot,
    write_worker_snapshot,
)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crypto_worker")

# Keep short in-memory oracle history per asset for chart sparkline payloads.
_MAX_ORACLE_HISTORY_POINTS = 180
_oracle_history_by_asset: dict[str, deque[tuple[int, float]]] = {}
_IDLE_INTERVAL_SECONDS = 15
_MIN_LOOP_SLEEP_SECONDS = 0.1
_VIEWER_HEARTBEAT_SECONDS = 20


def _to_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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
    points = list(history)
    if len(points) > 80:
        step = max(1, len(points) // 80)
        points = points[::step]
    if points and points[-1] != history[-1]:
        points.append(history[-1])
    return [{"t": t, "p": round(p, 2)} for t, p in points]


def _collect_ws_prices_for_markets(feed_manager, markets: list) -> dict[str, float]:
    """Collect fresh midpoint prices from WS cache for market token IDs."""
    if feed_manager is None or not getattr(feed_manager, "_started", False):
        return {}

    ws_prices: dict[str, float] = {}
    price_cache = getattr(feed_manager, "price_cache", None)
    if price_cache is None:
        return ws_prices

    for market in markets:
        for token_id in getattr(market, "clob_token_ids", None) or []:
            token = str(token_id or "").strip()
            if not token or len(token) <= 20:
                continue
            try:
                if not price_cache.is_fresh(token):
                    continue
                mid = price_cache.get_mid(token)
            except Exception:
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


def _overlay_ws_prices_on_market_row(row: dict, market, ws_prices: dict[str, float]) -> None:
    """Prefer fresh WS prices for up/down legs when available."""
    if not ws_prices:
        return

    tokens = list(getattr(market, "clob_token_ids", None) or [])
    if not tokens:
        return

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


async def _is_autotrader_active(session) -> bool:
    """Whether trader orchestrator is actively running strategies."""
    try:
        from services.trader_orchestrator_state import read_orchestrator_control

        control = await read_orchestrator_control(session)
        return bool(control.get("is_enabled", False)) and not bool(control.get("is_paused", True))
    except Exception:
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
) -> list[dict]:
    svc = get_crypto_service()
    feed = get_chainlink_feed()
    markets = markets if markets is not None else svc.get_live_markets()
    svc._update_price_to_beat(markets)

    payload: list[dict] = []
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    for market in markets:
        row = market.to_dict()
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
        payload.append(row)

    return payload


async def _run_loop() -> None:
    logger.info("Crypto worker started")
    worker_name = "crypto"

    startup_stats = {"market_count": 0, "signals_emitted_last_run": 0, "markets": []}
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

    feed_manager = None
    ws_feeds_running = False
    subscribed_tokens: set[str] = set()

    async def _ensure_ws_feeds_running() -> None:
        nonlocal feed_manager, ws_feeds_running
        if ws_feeds_running or not settings.WS_FEED_ENABLED:
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
            ws_feeds_running = True
            logger.info("Crypto worker WebSocket feeds started")
        except Exception as exc:
            logger.warning("Crypto worker WS feeds failed to start (non-critical): %s", exc)

    await _ensure_ws_feeds_running()

    while True:
        async with AsyncSessionLocal() as session:
            control = await read_worker_control(session, worker_name, default_interval=2)
            autotrader_active = await _is_autotrader_active(session)

        configured_interval = max(1, min(60, int(control.get("interval_seconds") or 2)))
        paused = bool(control.get("is_paused", False))
        enabled = bool(control.get("is_enabled", True))
        requested_run_at = control.get("requested_run_at")
        viewer_active = _is_recent_request(requested_run_at)
        fast_mode = bool(viewer_active or autotrader_active)
        interval = configured_interval if fast_mode else max(configured_interval, _IDLE_INTERVAL_SECONDS)

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
                    stats={"market_count": 0, "signals_emitted_last_run": 0, "markets": []},
                )
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

        run_at = utcnow()
        markets_payload: list[dict] = []
        emitted = 0
        err_text = None
        started_at = time.monotonic()
        ws_prices: dict[str, float] = {}

        try:
            svc = get_crypto_service()
            markets = svc.get_live_markets()
            if ws_feeds_running and feed_manager is not None:
                subscribed_tokens = await _sync_ws_subscriptions(feed_manager, markets, subscribed_tokens)
            ws_prices = _collect_ws_prices_for_markets(feed_manager, markets)
            markets_payload = _build_crypto_market_payload(
                markets,
                ws_prices=ws_prices,
            )
            elapsed = round(time.monotonic() - started_at, 3)
            async with AsyncSessionLocal() as session:
                try:
                    emitted = await emit_crypto_market_signals(session, markets_payload)
                except Exception as exc:
                    # Signal persistence should not stop market-data publishing.
                    emitted = 0
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    logger.warning("Crypto signal emission failed; continuing cycle: %s", exc)

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
                        "market_count": len(markets_payload),
                        "signals_emitted_last_run": int(emitted),
                        "run_duration_seconds": elapsed,
                        "fast_mode": fast_mode,
                        "ws_prices_used": int(bool(ws_prices)),
                        "ws_token_prices": len(ws_prices),
                        "markets": markets_payload,
                    },
                )

            logger.info(
                "Crypto cycle complete: markets=%s signals=%s duration=%.3fs fast_mode=%s sleep=%ss",
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
                        "fast_mode": fast_mode,
                        "ws_prices_used": int(bool(ws_prices)),
                        "ws_token_prices": len(ws_prices),
                        "markets": markets_payload,
                    },
                )

        if err_text:
            sleep_for = 0.5
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


async def main() -> None:
    await init_database()
    logger.info("Database initialized")
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


if __name__ == "__main__":
    asyncio.run(main())
