"""Market data worker: dedicated WS ingestion + stale-gap price poller.

Owns Polymarket/Kalshi feed-manager lifecycle, keeps subscriptions synced to
the canonical market catalog, and continuously writes canonical token prices
into Redis via ``redis_price_cache``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import apply_runtime_settings_overrides, settings
from models.database import (
    AsyncSessionLocal,
    ExecutionSession,
    ExecutionSessionLeg,
    Trader,
    TraderOrder,
    TraderPosition,
)
from services.kalshi_client import kalshi_client
from services.optimization.vwap import OrderBook, OrderBookLevel
from services.polymarket import polymarket_client
from services.redis_price_cache import redis_price_cache
from services.shared_state import read_market_catalog
from services.trader_orchestrator_state import (
    ACTIVE_EXECUTION_SESSION_STATUSES,
    OPEN_ORDER_STATUSES,
    list_unconsumed_trade_signals,
)
from services.worker_state import clear_worker_run_request, read_worker_control, write_worker_snapshot
from services.ws_feeds import get_feed_manager
from utils.logger import setup_logging
from utils.utcnow import utcnow

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), json_format=False)
logger = logging.getLogger("market_data_worker")


_TOKEN_SCALAR_KEYS = {
    "token_id",
    "tokenid",
    "selected_token",
    "selected_token_id",
    "selectedtokenid",
    "yes_token",
    "yes_token_id",
    "yestokenid",
    "no_token",
    "no_token_id",
    "notokenid",
    "asset_id",
    "assetid",
    "clob_token_id",
    "clobtokenid",
}
_TOKEN_LIST_KEYS = {
    "token_ids",
    "tokenids",
    "asset_ids",
    "assetids",
    "assets_ids",
    "assetsids",
    "clob_token_ids",
    "clobtokenids",
}
_MAX_EVALUATION_SIGNALS_PER_TRADER = 600


def _is_polymarket_token(token_id: str) -> bool:
    token = str(token_id or "").strip()
    if not token:
        return False
    lower = token.lower()
    return lower.startswith("0x") or len(token) > 20


def _is_polymarket_market_id(value: str) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("0x")


def _normalize_market_id(raw_id: object) -> str:
    return str(raw_id or "").strip().lower()


def _normalize_token_id(raw_token: object) -> str:
    return str(raw_token or "").strip()


def _parse_level(raw: Any) -> OrderBookLevel | None:
    if isinstance(raw, dict):
        price = raw.get("price")
        size = raw.get("size", raw.get("quantity", raw.get("amount", 0)))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        price = raw[0]
        size = raw[1]
    else:
        return None
    try:
        px = float(price)
        sz = float(size)
    except (TypeError, ValueError):
        return None
    if px < 0:
        return None
    if px > 1 and px <= 100:
        px = px / 100.0
    return OrderBookLevel(price=max(0.0, min(1.0, px)), size=max(0.0, sz))


def _book_from_raw(raw: Any) -> OrderBook | None:
    if not isinstance(raw, dict):
        return None
    bids_raw = raw.get("bids") or raw.get("buy") or []
    asks_raw = raw.get("asks") or raw.get("sell") or []
    bids = [_parse_level(item) for item in bids_raw]
    asks = [_parse_level(item) for item in asks_raw]
    bids = [level for level in bids if level is not None]
    asks = [level for level in asks if level is not None]
    bids.sort(key=lambda level: level.price, reverse=True)
    asks.sort(key=lambda level: level.price)
    if not bids and not asks:
        return None
    return OrderBook(bids=bids, asks=asks)


def _market_mid_order_book(mid: float) -> OrderBook:
    mid_px = max(0.0, min(1.0, float(mid)))
    level = OrderBookLevel(price=mid_px, size=1.0)
    return OrderBook(bids=[level], asks=[level])


async def _http_fallback_order_book(token_id: str) -> OrderBook | None:
    token = _normalize_token_id(token_id)
    if not token:
        return None

    if _is_polymarket_token(token):
        raw = await polymarket_client.get_order_book(token)
        return _book_from_raw(raw)

    if token.endswith("_yes") or token.endswith("_no"):
        ticker = token.rsplit("_", 1)[0]
        market = await kalshi_client.get_market(ticker)
        if market is None:
            return None
        mid = market.yes_price if token.endswith("_yes") else market.no_price
        return _market_mid_order_book(mid)

    return None


def _split_token_universe(tokens: set[str]) -> tuple[list[str], list[str]]:
    polymarket_tokens: list[str] = []
    kalshi_tokens: list[str] = []
    for token in sorted(tokens):
        if _is_polymarket_token(token):
            polymarket_tokens.append(token)
        else:
            kalshi_tokens.append(token)
    return polymarket_tokens, kalshi_tokens


def _kalshi_tickers_from_tokens(tokens: set[str]) -> set[str]:
    out: set[str] = set()
    for token in tokens:
        if _is_polymarket_token(token):
            continue
        if token.endswith("_yes") or token.endswith("_no"):
            out.add(token.rsplit("_", 1)[0])
        elif token:
            out.add(token)
    return out


def _collect_catalog_index(markets: list) -> tuple[set[str], dict[str, tuple[str, str]]]:
    token_ids: set[str] = set()
    market_tokens: dict[str, tuple[str, str]] = {}
    for market in markets:
        condition_id = _normalize_market_id(getattr(market, "condition_id", None))
        market_row_id = _normalize_market_id(getattr(market, "id", None))
        market_keys = {key for key in (condition_id, market_row_id) if key}
        if not market_keys:
            continue
        raw_tokens = list(getattr(market, "clob_token_ids", None) or [])
        clean_tokens = [_normalize_token_id(token) for token in raw_tokens if _normalize_token_id(token)]
        if len(clean_tokens) < 2:
            continue
        yes_token = clean_tokens[0]
        no_token = clean_tokens[1]
        token_ids.add(yes_token)
        token_ids.add(no_token)
        for market_key in market_keys:
            if market_key not in market_tokens:
                market_tokens[market_key] = (yes_token, no_token)
    return token_ids, market_tokens


def _normalize_source_key(raw_source: Any) -> str:
    return str(raw_source or "").strip().lower()


def _collect_tokens_from_payload(payload: Any) -> set[str]:
    out: set[str] = set()
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for raw_key, value in item.items():
                key = str(raw_key or "").strip().lower()
                if key in _TOKEN_SCALAR_KEYS:
                    token = _normalize_token_id(value)
                    if token:
                        out.add(token)
                elif key in _TOKEN_LIST_KEYS and isinstance(value, (list, tuple, set)):
                    for candidate in value:
                        token = _normalize_token_id(candidate)
                        if token:
                            out.add(token)
                if isinstance(value, (dict, list, tuple, set)):
                    stack.append(value)
            continue
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                if isinstance(nested, (dict, list, tuple, set)):
                    stack.append(nested)
    return out


def _add_market_tokens(
    token_ids: set[str],
    *,
    market_id: Any,
    market_tokens: dict[str, tuple[str, str]],
) -> None:
    key = _normalize_market_id(market_id)
    if not key:
        return
    pair = market_tokens.get(key)
    if pair is None:
        return
    yes_token, no_token = pair
    if yes_token:
        token_ids.add(yes_token)
    if no_token:
        token_ids.add(no_token)


async def _collect_trader_critical_universe(
    session: AsyncSession,
    *,
    market_tokens: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    held_market_ids: set[str] = set()
    evaluation_market_ids: set[str] = set()
    critical_tokens: set[str] = set()

    open_position_rows = (
        await session.execute(
            select(TraderPosition.market_id, TraderPosition.payload_json).where(
                func.lower(func.coalesce(TraderPosition.status, "")) == "open"
            )
        )
    ).all()
    for row in open_position_rows:
        market_id = _normalize_market_id(row.market_id)
        if market_id:
            held_market_ids.add(market_id)
            _add_market_tokens(critical_tokens, market_id=market_id, market_tokens=market_tokens)
        critical_tokens.update(_collect_tokens_from_payload(row.payload_json))

    open_order_rows = (
        await session.execute(
            select(
                TraderOrder.market_id,
                TraderOrder.payload_json,
            ).where(func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(OPEN_ORDER_STATUSES)))
        )
    ).all()
    for row in open_order_rows:
        market_id = _normalize_market_id(row.market_id)
        if market_id:
            held_market_ids.add(market_id)
            _add_market_tokens(critical_tokens, market_id=market_id, market_tokens=market_tokens)
        critical_tokens.update(_collect_tokens_from_payload(row.payload_json))

    active_execution_leg_rows = (
        await session.execute(
            select(
                ExecutionSessionLeg.market_id,
                ExecutionSessionLeg.token_id,
            )
            .join(ExecutionSession, ExecutionSession.id == ExecutionSessionLeg.session_id)
            .where(func.lower(func.coalesce(ExecutionSession.status, "")).in_(tuple(ACTIVE_EXECUTION_SESSION_STATUSES)))
        )
    ).all()
    for row in active_execution_leg_rows:
        market_id = _normalize_market_id(row.market_id)
        if market_id:
            held_market_ids.add(market_id)
            _add_market_tokens(critical_tokens, market_id=market_id, market_tokens=market_tokens)
        token_id = _normalize_token_id(row.token_id)
        if token_id:
            critical_tokens.add(token_id)

    enabled_traders = (
        await session.execute(
            select(
                Trader.id,
                Trader.source_configs_json,
            ).where(
                and_(
                    Trader.is_enabled == True,  # noqa: E712
                    Trader.is_paused == False,  # noqa: E712
                )
            )
        )
    ).all()
    evaluation_signals_count = 0
    for trader_row in enabled_traders:
        trader_id = str(trader_row.id or "").strip()
        if not trader_id:
            continue
        source_configs = list(trader_row.source_configs_json or [])
        sources: list[str] = []
        for source_config in source_configs:
            if not isinstance(source_config, dict):
                continue
            source_key = _normalize_source_key(source_config.get("source_key"))
            if source_key:
                sources.append(source_key)
        if not sources:
            continue
        signals = await list_unconsumed_trade_signals(
            session,
            trader_id=trader_id,
            sources=sorted(set(sources)),
            statuses=["pending", "selected"],
            limit=_MAX_EVALUATION_SIGNALS_PER_TRADER,
        )
        evaluation_signals_count += len(signals)
        for signal in signals:
            market_id = _normalize_market_id(getattr(signal, "market_id", ""))
            if market_id:
                evaluation_market_ids.add(market_id)
                _add_market_tokens(critical_tokens, market_id=market_id, market_tokens=market_tokens)
            payload_json = getattr(signal, "payload_json", None)
            critical_tokens.update(_collect_tokens_from_payload(payload_json))

    critical_market_ids = set(held_market_ids)
    critical_market_ids.update(evaluation_market_ids)
    for market_id in critical_market_ids:
        _add_market_tokens(critical_tokens, market_id=market_id, market_tokens=market_tokens)

    critical_condition_ids = {
        market_id for market_id in critical_market_ids if _is_polymarket_market_id(market_id)
    }
    return {
        "critical_tokens": critical_tokens,
        "critical_condition_ids": critical_condition_ids,
        "critical_market_ids": critical_market_ids,
        "held_market_ids": held_market_ids,
        "evaluation_market_ids": evaluation_market_ids,
        "evaluation_signals_count": evaluation_signals_count,
    }


def _select_with_priority(
    *,
    critical_items: set[str],
    background_items: set[str],
    cap: int,
) -> tuple[set[str], int]:
    selected: set[str] = set()
    for item in sorted(critical_items):
        selected.add(item)
    if len(selected) >= cap:
        return selected, len(background_items)

    for item in sorted(background_items):
        if item in selected:
            continue
        if len(selected) >= cap:
            break
        selected.add(item)
    dropped = max(0, len(background_items - selected))
    return selected, dropped


def _fallback_updates_from_prices(
    prices: dict[str, dict],
    sequence_start: int,
    now_ts: float,
) -> tuple[dict[str, tuple[float | None, float | None, float | None, float, float, int]], int]:
    updates: dict[str, tuple[float | None, float | None, float | None, float, float, int]] = {}
    seq = int(sequence_start)
    for token_id, payload in prices.items():
        if not isinstance(payload, dict):
            continue
        mid = payload.get("mid")
        if mid is None:
            continue
        try:
            mid_val = float(mid)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= mid_val <= 1.0):
            continue
        seq += 1
        updates[str(token_id)] = (
            mid_val,
            float(payload.get("bid", mid_val)),
            float(payload.get("ask", mid_val)),
            now_ts,
            now_ts,
            seq,
        )
    return updates, seq


async def _run_loop() -> None:
    worker_name = "market_data"
    heartbeat_interval = max(
        1.0,
        float(getattr(settings, "MARKET_DATA_HEARTBEAT_INTERVAL_SECONDS", 3.0) or 3.0),
    )
    sync_interval = max(
        1,
        int(getattr(settings, "MARKET_DATA_REFRESH_INTERVAL_SECONDS", 5) or 5),
    )
    stale_poll_interval = max(
        1,
        int(getattr(settings, "MARKET_DATA_STALE_POLL_INTERVAL_SECONDS", 3) or 3),
    )
    stale_poll_batch_size = max(
        10,
        int(getattr(settings, "MARKET_DATA_STALE_POLL_BATCH_SIZE", 400) or 400),
    )
    ws_subscription_cap = max(
        100,
        int(getattr(settings, "MARKET_DATA_WS_SUBSCRIPTION_CAP", 8000) or 8000),
    )

    state: dict[str, Any] = {
        "enabled": True,
        "interval_seconds": sync_interval,
        "activity": "Market data worker started; awaiting catalog snapshot.",
        "last_error": None,
        "last_run_at": None,
        "run_id": None,
        "phase": "idle",
        "progress": 0.0,
        "catalog_markets": 0,
        "catalog_tokens": 0,
        "fresh_tokens": 0,
        "stale_tokens": 0,
        "stale_markets": 0,
        "fallback_polled_tokens": 0,
        "fallback_updates": 0,
        "ws_started": False,
        "polymarket_subscriptions": 0,
        "kalshi_subscriptions": 0,
        "critical_tokens": 0,
        "critical_markets": 0,
        "held_markets": 0,
        "evaluation_markets": 0,
        "evaluation_signals": 0,
        "dropped_background_poly_tokens": 0,
        "dropped_background_kalshi_tokens": 0,
    }
    heartbeat_stop_event = asyncio.Event()

    feed_manager = get_feed_manager()
    fallback_sequence = 0
    catalog_tokens: set[str] = set()
    critical_tokens: set[str] = set()
    critical_market_ids: set[str] = set()
    desired_tokens: set[str] = set()
    market_tokens: dict[str, tuple[str, str]] = {}
    tracked_market_tokens: dict[str, tuple[str, str]] = {}
    subscribed_poly_tokens: set[str] = set()
    subscribed_kalshi_tickers: set[str] = set()
    next_stale_poll_at = datetime.now(timezone.utc)

    async def _write_heartbeat() -> None:
        async with AsyncSessionLocal() as session:
            await write_worker_snapshot(
                session,
                worker_name,
                running=True,
                enabled=bool(state.get("enabled", True)),
                current_activity=str(state.get("activity") or "Idle"),
                interval_seconds=int(state.get("interval_seconds") or sync_interval),
                last_run_at=state.get("last_run_at"),
                last_error=(str(state["last_error"]) if state.get("last_error") is not None else None),
                stats={
                    "run_id": state.get("run_id"),
                    "phase": state.get("phase"),
                    "progress": float(state.get("progress", 0.0) or 0.0),
                    "catalog_markets": int(state.get("catalog_markets", 0) or 0),
                    "catalog_tokens": int(state.get("catalog_tokens", 0) or 0),
                    "fresh_tokens": int(state.get("fresh_tokens", 0) or 0),
                    "stale_tokens": int(state.get("stale_tokens", 0) or 0),
                    "stale_markets": int(state.get("stale_markets", 0) or 0),
                    "fallback_polled_tokens": int(state.get("fallback_polled_tokens", 0) or 0),
                    "fallback_updates": int(state.get("fallback_updates", 0) or 0),
                    "ws_started": bool(state.get("ws_started", False)),
                    "polymarket_subscriptions": int(state.get("polymarket_subscriptions", 0) or 0),
                    "kalshi_subscriptions": int(state.get("kalshi_subscriptions", 0) or 0),
                    "critical_tokens": int(state.get("critical_tokens", 0) or 0),
                    "critical_markets": int(state.get("critical_markets", 0) or 0),
                    "critical_poly_tokens": int(state.get("critical_poly_tokens", 0) or 0),
                    "critical_kalshi_tokens": int(state.get("critical_kalshi_tokens", 0) or 0),
                    "critical_poly_subscribed_tokens": int(
                        state.get("critical_poly_subscribed_tokens", 0) or 0
                    ),
                    "critical_kalshi_subscribed_tokens": int(
                        state.get("critical_kalshi_subscribed_tokens", 0) or 0
                    ),
                    "critical_poly_missing_tokens": int(state.get("critical_poly_missing_tokens", 0) or 0),
                    "critical_kalshi_missing_tokens": int(
                        state.get("critical_kalshi_missing_tokens", 0) or 0
                    ),
                    "held_markets": int(state.get("held_markets", 0) or 0),
                    "evaluation_markets": int(state.get("evaluation_markets", 0) or 0),
                    "evaluation_signals": int(state.get("evaluation_signals", 0) or 0),
                    "dropped_background_poly_tokens": int(state.get("dropped_background_poly_tokens", 0) or 0),
                    "dropped_background_kalshi_tokens": int(
                        state.get("dropped_background_kalshi_tokens", 0) or 0
                    ),
                    "critical_stale_tokens": int(state.get("critical_stale_tokens", 0) or 0),
                    "critical_stale_markets": int(state.get("critical_stale_markets", 0) or 0),
                },
            )

    async def _heartbeat_loop() -> None:
        while not heartbeat_stop_event.is_set():
            try:
                await _write_heartbeat()
            except Exception as exc:
                state["last_error"] = str(exc)
                logger.warning("Market data heartbeat snapshot write failed: %s", exc)
            try:
                await asyncio.wait_for(heartbeat_stop_event.wait(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                continue

    async def _sync_ws_subscriptions() -> None:
        nonlocal subscribed_poly_tokens
        nonlocal subscribed_kalshi_tickers
        nonlocal desired_tokens

        catalog_poly_tokens, catalog_kalshi_tokens = _split_token_universe(catalog_tokens)
        critical_poly_tokens, critical_kalshi_tokens = _split_token_universe(critical_tokens)

        desired_poly_tokens, dropped_background_poly_tokens = _select_with_priority(
            critical_items=set(critical_poly_tokens),
            background_items=set(catalog_poly_tokens),
            cap=ws_subscription_cap,
        )
        desired_kalshi_tokens, dropped_background_kalshi_tokens = _select_with_priority(
            critical_items=set(critical_kalshi_tokens),
            background_items=set(catalog_kalshi_tokens),
            cap=ws_subscription_cap,
        )
        desired_kalshi = _kalshi_tickers_from_tokens(desired_kalshi_tokens)

        if settings.WS_FEED_ENABLED:
            add_poly_tokens = sorted(desired_poly_tokens - subscribed_poly_tokens)
            remove_poly_tokens = sorted(subscribed_poly_tokens - desired_poly_tokens)
            if add_poly_tokens:
                await feed_manager.polymarket_feed.subscribe(add_poly_tokens)
            if remove_poly_tokens:
                await feed_manager.polymarket_feed.unsubscribe(remove_poly_tokens)
            subscribed_poly_tokens = desired_poly_tokens

            add_kalshi = sorted(desired_kalshi - subscribed_kalshi_tickers)
            remove_kalshi = sorted(subscribed_kalshi_tickers - desired_kalshi)
            if add_kalshi:
                await feed_manager.kalshi_feed.subscribe(add_kalshi)
            if remove_kalshi:
                await feed_manager.kalshi_feed.unsubscribe(remove_kalshi)
            subscribed_kalshi_tickers = desired_kalshi
        else:
            subscribed_poly_tokens = desired_poly_tokens
            subscribed_kalshi_tickers = desired_kalshi

        desired_tokens = set(desired_poly_tokens)
        desired_tokens.update(desired_kalshi_tokens)
        state["dropped_background_poly_tokens"] = dropped_background_poly_tokens
        state["dropped_background_kalshi_tokens"] = dropped_background_kalshi_tokens
        state["polymarket_subscriptions"] = len(subscribed_poly_tokens)
        state["kalshi_subscriptions"] = len(subscribed_kalshi_tickers)
        state["critical_poly_tokens"] = len(critical_poly_tokens)
        state["critical_kalshi_tokens"] = len(critical_kalshi_tokens)
        state["critical_poly_subscribed_tokens"] = len(subscribed_poly_tokens.intersection(critical_poly_tokens))
        state["critical_kalshi_subscribed_tokens"] = len(desired_kalshi_tokens.intersection(critical_kalshi_tokens))
        state["critical_poly_missing_tokens"] = len(set(critical_poly_tokens) - set(subscribed_poly_tokens))
        state["critical_kalshi_missing_tokens"] = len(set(critical_kalshi_tokens) - set(desired_kalshi_tokens))

    async def _run_stale_gap_filler(now_dt: datetime) -> None:
        nonlocal fallback_sequence
        all_tokens = sorted(desired_tokens)
        if not all_tokens:
            state["fresh_tokens"] = 0
            state["stale_tokens"] = 0
            state["stale_markets"] = 0
            state["fallback_polled_tokens"] = 0
            state["fallback_updates"] = 0
            return

        fresh_prices = await redis_price_cache.read_prices(all_tokens)
        fresh_tokens = set(fresh_prices.keys())
        stale_tokens = [token for token in all_tokens if token not in fresh_tokens]
        state["fresh_tokens"] = len(fresh_tokens)
        state["stale_tokens"] = len(stale_tokens)
        critical_token_set = set(critical_tokens)
        critical_stale_tokens = [token for token in critical_token_set if token not in fresh_tokens]
        state["critical_stale_tokens"] = len(critical_stale_tokens)

        stale_market_count = 0
        critical_stale_market_count = 0
        for yes_token, no_token in tracked_market_tokens.values():
            if yes_token not in fresh_tokens or no_token not in fresh_tokens:
                stale_market_count += 1
                critical_stale_market_count += 1
        state["stale_markets"] = stale_market_count
        state["critical_stale_markets"] = critical_stale_market_count

        if not bool(getattr(settings, "MARKET_DATA_WORKER_OWNS_WS", True)):
            state["fallback_polled_tokens"] = 0
            state["fallback_updates"] = 0
            return
        if not stale_tokens:
            state["fallback_polled_tokens"] = 0
            state["fallback_updates"] = 0
            return

        poll_tokens = stale_tokens[:stale_poll_batch_size]
        poly_poll = [token for token in poll_tokens if _is_polymarket_token(token)]
        kalshi_poll = [token for token in poll_tokens if not _is_polymarket_token(token)]
        fallback_prices: dict[str, dict] = {}

        if poly_poll:
            try:
                fallback_prices.update(await polymarket_client.get_prices_batch(poly_poll))
            except Exception as exc:
                logger.warning("Polymarket stale-gap poll failed: %s", exc)
        if kalshi_poll:
            try:
                fallback_prices.update(await kalshi_client.get_prices_batch(kalshi_poll))
            except Exception as exc:
                logger.warning("Kalshi stale-gap poll failed: %s", exc)

        now_ts = max(0.0, now_dt.timestamp())
        updates, fallback_sequence = _fallback_updates_from_prices(fallback_prices, fallback_sequence, now_ts)
        if updates:
            await redis_price_cache.write_prices(updates)
        state["fallback_polled_tokens"] = len(poll_tokens)
        state["fallback_updates"] = len(updates)

    async def _sleep_until_next_cycle(interval_seconds: float) -> None:
        sleep_remaining = max(0.0, float(interval_seconds))
        while sleep_remaining > 1e-6:
            chunk = min(1.0, sleep_remaining)
            await asyncio.sleep(chunk)
            sleep_remaining -= chunk
            if sleep_remaining <= 1e-6:
                return
            try:
                async with AsyncSessionLocal() as session:
                    control_state = await read_worker_control(
                        session,
                        worker_name,
                        default_interval=sync_interval,
                    )
                if control_state.get("requested_run_at") is not None:
                    return
            except Exception:
                continue

    heartbeat_task = asyncio.create_task(_heartbeat_loop(), name="market-data-heartbeat")
    logger.info("Market data worker started")

    try:
        if settings.WS_FEED_ENABLED:
            feed_manager.set_http_fallback(_http_fallback_order_book)
            if not feed_manager._started:
                await feed_manager.start()
            state["ws_started"] = True
            state["activity"] = "Market data feeds connected; syncing catalog subscriptions."
        else:
            state["ws_started"] = False
            state["activity"] = "WS feeds disabled; stale-gap poller active only."

        while True:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(
                    session,
                    worker_name,
                    default_interval=sync_interval,
                )
                try:
                    await apply_runtime_settings_overrides()
                except Exception as exc:
                    logger.warning("Market data runtime settings refresh failed: %s", exc)

            interval_seconds = max(1, int(control.get("interval_seconds") or sync_interval))
            paused = bool(control.get("is_paused", False))
            requested = control.get("requested_run_at") is not None
            enabled = bool(control.get("is_enabled", True)) and not paused
            state["enabled"] = enabled
            state["interval_seconds"] = interval_seconds
            state["progress"] = 0.0
            state["phase"] = "idle"
            if state.get("run_id") is None:
                state["run_id"] = uuid.uuid4().hex[:16]

            if not enabled and not requested:
                state["activity"] = "Paused"
                await asyncio.sleep(heartbeat_interval)
                continue

            cycle_run_id = uuid.uuid4().hex[:16]
            state["run_id"] = cycle_run_id
            state["phase"] = "catalog_sync"
            state["progress"] = 0.1
            state["activity"] = "Syncing market catalog and WS subscriptions..."
            now_dt = datetime.now(timezone.utc)

            try:
                async with AsyncSessionLocal() as session:
                    _, markets, metadata = await read_market_catalog(session)
                    catalog_tokens, market_tokens = _collect_catalog_index(markets)
                    critical_universe = await _collect_trader_critical_universe(
                        session,
                        market_tokens=market_tokens,
                    )
                    await clear_worker_run_request(session, worker_name)

                critical_tokens = set(critical_universe.get("critical_tokens") or [])
                critical_market_ids = set(critical_universe.get("critical_market_ids") or [])
                tracked_market_tokens = {
                    market_id: market_tokens[market_id]
                    for market_id in critical_market_ids
                    if market_id in market_tokens
                }

                state["critical_tokens"] = len(critical_tokens)
                state["critical_markets"] = len(critical_market_ids)
                state["held_markets"] = len(critical_universe.get("held_market_ids") or [])
                state["evaluation_markets"] = len(critical_universe.get("evaluation_market_ids") or [])
                state["evaluation_signals"] = int(critical_universe.get("evaluation_signals_count") or 0)
                state["catalog_markets"] = int(metadata.get("market_count") or len(markets) or 0)
                state["catalog_tokens"] = len(catalog_tokens)

                state["phase"] = "subscription_sync"
                state["progress"] = 0.45
                await _sync_ws_subscriptions()

                if now_dt >= next_stale_poll_at:
                    state["phase"] = "stale_gap_fill"
                    state["progress"] = 0.75
                    await _run_stale_gap_filler(now_dt)
                    next_stale_poll_at = now_dt + timedelta(seconds=stale_poll_interval)

                state["last_error"] = None
                state["last_run_at"] = utcnow()
                state["phase"] = "idle"
                state["progress"] = 1.0
                state["activity"] = (
                    f"Market data synced: catalog={state['catalog_markets']} markets/{state['catalog_tokens']} tokens, "
                    f"critical={state['critical_markets']} markets/{state['critical_tokens']} tokens, "
                    f"stale={state['stale_tokens']}."
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state["last_error"] = str(exc)
                state["phase"] = "error"
                state["activity"] = f"Market data cycle error: {exc}"
                logger.exception("Market data cycle failed: %s", exc)

            await _sleep_until_next_cycle(float(interval_seconds))
    finally:
        heartbeat_stop_event.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

        if feed_manager._started:
            try:
                await feed_manager.stop()
            except Exception:
                pass


async def start_loop() -> None:
    try:
        await _run_loop()
    except asyncio.CancelledError:
        logger.info("Market data worker shutting down")
