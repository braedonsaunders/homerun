from __future__ import annotations

import asyncio
import copy
import math
import time
from datetime import datetime, timezone
from typing import Any

from config import settings
from models.database import AsyncSessionLocal
from services import shared_state
from services.crypto_service import get_crypto_service
from services.data_events import DataEvent, EventType
from services.event_bus import event_bus
from services.event_dispatcher import event_dispatcher
from services.intent_runtime import get_intent_runtime
from services.reference_runtime import get_reference_runtime
from services.runtime_status import runtime_status
from services.worker_state import read_worker_control, summarize_worker_stats, write_worker_snapshot
from services.ws_feeds import get_feed_manager
from utils.converters import normalize_identifier as _normalize_market_id
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)

_WS_REACTIVE_DEBOUNCE_SECONDS = max(0.01, float(getattr(settings, "CRYPTO_WS_REACTIVE_DEBOUNCE_SECONDS", 0.05) or 0.05))
_CATALOG_REFRESH_SECONDS = 5.0
_CRYPTO_SNAPSHOT_PERSIST_INTERVAL_SECONDS = 5.0
_CRYPTO_SNAPSHOT_MARKETS_LIMIT = 64
_FULL_REFRESH_FLOOR_SECONDS = 0.5
_BOUNDARY_INTERVALS_SECONDS = (300, 900, 3600, 14400)
_BOUNDARY_PREFETCH_WINDOW_SECONDS = 15
_BOUNDARY_LINGER_WINDOW_SECONDS = 10


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None

def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _near_market_boundary() -> bool:
    now_ts = time.time()
    for interval in _BOUNDARY_INTERVALS_SECONDS:
        seconds_into = now_ts % interval
        seconds_until_next = interval - seconds_into
        if seconds_until_next <= _BOUNDARY_PREFETCH_WINDOW_SECONDS:
            return True
        if seconds_into <= _BOUNDARY_LINGER_WINDOW_SECONDS:
            return True
    return False


class MarketRuntime:
    def __init__(self) -> None:
        self._started = False
        self._start_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._main_task: asyncio.Task[None] | None = None
        self._reactive_task: asyncio.Task[None] | None = None
        self._opportunity_dispatch_task: asyncio.Task[None] | None = None
        self._feed_manager = None
        self._reference_runtime = get_reference_runtime()
        self._crypto_markets: list[dict[str, Any]] = []
        self._crypto_markets_by_lookup: dict[str, dict[str, Any]] = {}
        self._crypto_token_to_market_ids: dict[str, set[str]] = {}
        self._crypto_asset_to_market_ids: dict[str, set[str]] = {}
        self._event_catalog_markets: dict[str, dict[str, Any]] = {}
        self._event_catalog_updated_at: str | None = None
        self._last_crypto_refresh_at: str | None = None
        self._last_crypto_trigger: str | None = None
        self._current_activity = "Starting"
        self._last_error: str | None = None
        self._last_catalog_refresh_mono = 0.0
        self._last_snapshot_persist_mono = 0.0
        self._pending_tokens: set[str] = set()
        self._pending_assets: set[str] = set()
        self._pending_reactive_lock = asyncio.Lock()
        self._pending_opportunity_payload: list[dict[str, Any]] | None = None
        self._pending_opportunity_trigger: str | None = None
        self._pending_opportunity_full_source_sweep = False
        self._pending_opportunity_lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            self._stop_event.clear()
            await self._reference_runtime.start()
            self._reference_runtime.on_update(self._on_reference_update)
            self._feed_manager = get_feed_manager()
            if not getattr(self._feed_manager, "_started", False):
                await self._feed_manager.start()
            self._feed_manager.cache.add_on_update_callback(self._on_ws_price_update)
            await self._refresh_event_catalog(force=True)
            await self._refresh_crypto_markets(trigger="startup", full_source_sweep=True)
            self._started = True
            self._main_task = asyncio.create_task(self._run_loop(), name="market-runtime")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._reactive_task is not None and not self._reactive_task.done():
            self._reactive_task.cancel()
            try:
                await self._reactive_task
            except asyncio.CancelledError:
                pass
        if self._opportunity_dispatch_task is not None and not self._opportunity_dispatch_task.done():
            self._opportunity_dispatch_task.cancel()
            try:
                await self._opportunity_dispatch_task
            except asyncio.CancelledError:
                pass
        if self._main_task is not None and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        self._reference_runtime.remove_on_update(self._on_reference_update)
        await self._reference_runtime.stop()
        self._started = False

    def get_crypto_markets(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._crypto_markets)

    def get_crypto_status(self) -> dict[str, Any]:
        ws_status = {}
        if self._feed_manager is not None:
            try:
                ws_status = self._feed_manager.health_check()
            except Exception:
                ws_status = {}
        return {
            "running": bool(self._started),
            "enabled": True,
            "current_activity": self._current_activity,
            "last_run_at": self._last_crypto_refresh_at,
            "last_error": self._last_error,
            "stats": {
                "market_count": len(self._crypto_markets),
                "markets": self.get_crypto_markets(),
                "oracle_prices": self.get_oracle_prices(),
                "trigger": self._last_crypto_trigger,
                "ws_feeds": ws_status,
                **self._reference_runtime.get_status(),
            },
        }

    def _last_crypto_refresh_datetime(self) -> datetime | None:
        raw_value = str(self._last_crypto_refresh_at or "").strip()
        if not raw_value:
            return None
        try:
            parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    async def _persist_crypto_worker_snapshot(
        self,
        *,
        control: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        resolved_control = dict(control or await self._read_crypto_control())
        status = self.get_crypto_status()
        stats = status.get("stats") if isinstance(status.get("stats"), dict) else {}
        persisted_stats = summarize_worker_stats(stats)
        markets = stats.get("markets")
        if isinstance(markets, list) and markets:
            persisted_stats["markets"] = copy.deepcopy(markets[:_CRYPTO_SNAPSHOT_MARKETS_LIMIT])
            persisted_stats["markets_count"] = len(markets)
        oracle_prices = stats.get("oracle_prices")
        if isinstance(oracle_prices, dict) and oracle_prices:
            persisted_stats["oracle_prices"] = copy.deepcopy(oracle_prices)
        enabled = bool(resolved_control.get("is_enabled", True))
        paused = bool(resolved_control.get("is_paused", False))
        runtime_status.update_crypto(
            running=True,
            enabled=enabled and not paused,
            current_activity=str(status.get("current_activity") or self._current_activity or "Idle"),
            interval_seconds=int(resolved_control.get("interval_seconds") or 1),
            last_run_at=self._last_crypto_refresh_at,
            last_error=self._last_error,
            stats=stats,
            control=resolved_control,
        )
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_snapshot_persist_mono) < _CRYPTO_SNAPSHOT_PERSIST_INTERVAL_SECONDS:
            return
        async with AsyncSessionLocal() as session:
            await write_worker_snapshot(
                session,
                "crypto",
                running=True,
                enabled=enabled and not paused,
                current_activity=str(status.get("current_activity") or self._current_activity or "Idle"),
                interval_seconds=int(resolved_control.get("interval_seconds") or 1),
                last_run_at=self._last_crypto_refresh_datetime(),
                last_error=self._last_error,
                stats=persisted_stats,
            )
        self._last_snapshot_persist_mono = now_mono

    def get_market_snapshot(self, market_id: str, *, hint: dict[str, Any] | None = None) -> dict[str, Any] | None:
        normalized = _normalize_market_id(market_id)
        if normalized:
            crypto = self._crypto_markets_by_lookup.get(normalized)
            if crypto is not None:
                return copy.deepcopy(crypto)
            event_market = self._event_catalog_markets.get(normalized)
            if event_market is not None:
                return self._build_event_market_snapshot(event_market)
        hinted = hint if isinstance(hint, dict) else {}
        for key in (
            "condition_id",
            "conditionId",
            "id",
            "market_id",
        ):
            normalized_hint = _normalize_market_id(hinted.get(key))
            if not normalized_hint:
                continue
            crypto = self._crypto_markets_by_lookup.get(normalized_hint)
            if crypto is not None:
                return copy.deepcopy(crypto)
            event_market = self._event_catalog_markets.get(normalized_hint)
            if event_market is not None:
                return self._build_event_market_snapshot(event_market)
        token_ids = hinted.get("clob_token_ids") or hinted.get("token_ids") or []
        if isinstance(token_ids, list):
            for raw_token_id in token_ids:
                token_id = _normalize_market_id(raw_token_id)
                if not token_id:
                    continue
                crypto = self._crypto_markets_by_lookup.get(token_id)
                if crypto is not None:
                    return copy.deepcopy(crypto)
                event_market = self._event_catalog_markets.get(token_id)
                if event_market is not None:
                    return self._build_event_market_snapshot(event_market)
        return copy.deepcopy(hinted) if hinted else None

    def get_token_mid_price(self, token_id: str) -> float | None:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return None
        snapshot = self.get_market_snapshot(normalized)
        if isinstance(snapshot, dict):
            token_ids = [
                _normalize_market_id(raw_token_id)
                for raw_token_id in (snapshot.get("clob_token_ids") or [])
                if _normalize_market_id(raw_token_id)
            ]
            if token_ids:
                if token_ids[0] == normalized and snapshot.get("up_price") is not None:
                    return _to_float(snapshot.get("up_price"))
                if len(token_ids) > 1 and token_ids[1] == normalized and snapshot.get("down_price") is not None:
                    return _to_float(snapshot.get("down_price"))
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return None
        if not feed_manager.cache.is_fresh(normalized):
            return None
        return _to_float(feed_manager.cache.get_mid_price(normalized))

    def get_token_spread_bps(self, token_id: str) -> float | None:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return None
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return None
        return _to_float(feed_manager.cache.get_spread_bps(normalized))

    def get_price_history(self, token_id: str, *, max_snapshots: int = 60) -> list[dict[str, Any]]:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return []
        snapshot = self.get_market_snapshot(normalized)
        if isinstance(snapshot, dict):
            history_tail = snapshot.get("history_tail")
            if isinstance(history_tail, list) and history_tail:
                return copy.deepcopy(history_tail[-max(1, int(max_snapshots)) :])
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return []
        if not hasattr(feed_manager.cache, "get_price_history"):
            return []
        history = feed_manager.cache.get_price_history(normalized, max_snapshots=max_snapshots)
        return copy.deepcopy(history or [])

    def get_price_change(self, token_id: str, *, lookback_seconds: int = 300) -> dict[str, Any] | None:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return None
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return None
        if not hasattr(feed_manager.cache, "get_price_change"):
            return None
        change = feed_manager.cache.get_price_change(normalized, lookback_seconds=lookback_seconds)
        return copy.deepcopy(change) if isinstance(change, dict) else None

    def get_recent_trades(self, token_id: str, *, max_trades: int = 100) -> list[Any]:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return []
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return []
        if not hasattr(feed_manager.cache, "get_recent_trades"):
            return []
        return list(feed_manager.cache.get_recent_trades(normalized, max_trades) or [])

    def get_trade_volume(self, token_id: str, *, lookback_seconds: float = 300.0) -> dict[str, Any]:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}
        if not hasattr(feed_manager.cache, "get_trade_volume"):
            return {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}
        volume = feed_manager.cache.get_trade_volume(normalized, lookback_seconds)
        return copy.deepcopy(volume) if isinstance(volume, dict) else {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}

    def get_buy_sell_imbalance(self, token_id: str, *, lookback_seconds: float = 300.0) -> float:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return 0.0
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return 0.0
        if not hasattr(feed_manager.cache, "get_buy_sell_imbalance"):
            return 0.0
        imbalance = _to_float(feed_manager.cache.get_buy_sell_imbalance(normalized, lookback_seconds))
        return float(imbalance or 0.0)

    def get_oracle_prices(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for market in self._crypto_markets:
            asset = str(market.get("asset") or "").strip().upper()
            if not asset:
                continue
            out[asset] = {
                "price": market.get("oracle_price"),
                "updated_at_ms": market.get("oracle_updated_at_ms"),
                "age_seconds": market.get("oracle_age_seconds"),
            }
        return out

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if (time.monotonic() - self._last_catalog_refresh_mono) >= _CATALOG_REFRESH_SECONDS:
                    await self._refresh_event_catalog()
                control = await self._read_crypto_control()
                enabled = bool(control.get("is_enabled", True))
                paused = bool(control.get("is_paused", False))
                interval_seconds = max(_FULL_REFRESH_FLOOR_SECONDS, float(control.get("interval_seconds") or 1.0))
                if enabled and not paused:
                    await self._refresh_crypto_markets(
                        trigger="periodic_scan",
                        full_source_sweep=True,
                        force_refresh=_near_market_boundary(),
                    )
                    self._current_activity = "Live"
                else:
                    self._current_activity = "Paused" if paused else "Disabled"
                    try:
                        await self._persist_crypto_worker_snapshot(control=control, force=True)
                    except Exception as snapshot_exc:
                        logger.warning("Failed to persist crypto worker snapshot", exc_info=snapshot_exc)
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                self._current_activity = f"Error: {exc}"
                logger.warning("Market runtime loop failed", exc_info=exc)
                try:
                    await self._persist_crypto_worker_snapshot(
                        control={"is_enabled": True, "is_paused": False, "interval_seconds": 1},
                        force=True,
                    )
                except Exception as snapshot_exc:
                    logger.warning("Failed to persist crypto worker snapshot after runtime error", exc_info=snapshot_exc)
                await asyncio.sleep(1.0)

    async def _read_crypto_control(self) -> dict[str, Any]:
        try:
            async with AsyncSessionLocal() as session:
                return await read_worker_control(session, "crypto", default_interval=1)
        except Exception:
            return {
                "is_enabled": True,
                "is_paused": False,
                "interval_seconds": 1,
                "requested_run_at": None,
            }

    async def _refresh_event_catalog(self, *, force: bool = False) -> None:
        if not force and (time.monotonic() - self._last_catalog_refresh_mono) < _CATALOG_REFRESH_SECONDS:
            return
        async with AsyncSessionLocal() as session:
            _events, markets, metadata = await shared_state.read_market_catalog(
                session,
                include_events=False,
                include_markets=True,
                validate=False,
            )
        lookup: dict[str, dict[str, Any]] = {}
        for market in markets:
            row = _copy_dict(market)
            market_id = _normalize_market_id(row.get("id"))
            condition_id = _normalize_market_id(row.get("condition_id") or row.get("conditionId"))
            token_ids = [
                _normalize_market_id(token_id)
                for token_id in (row.get("clob_token_ids") or row.get("token_ids") or [])
                if _normalize_market_id(token_id)
            ]
            for key in (market_id, condition_id, *token_ids):
                if key:
                    lookup[key] = row
        updated_at = metadata.get("updated_at")
        self._event_catalog_markets = lookup
        self._event_catalog_updated_at = updated_at.isoformat() if hasattr(updated_at, "isoformat") else None
        self._last_catalog_refresh_mono = time.monotonic()

    def _build_event_market_snapshot(self, market: dict[str, Any]) -> dict[str, Any]:
        row = dict(market)
        token_ids = [
            str(token_id or "").strip()
            for token_id in (row.get("clob_token_ids") or row.get("token_ids") or [])
            if str(token_id or "").strip()
        ]
        yes_token = token_ids[0] if token_ids else None
        no_token = token_ids[1] if len(token_ids) > 1 else None
        feed_manager = self._feed_manager
        if feed_manager is not None and getattr(feed_manager, "_started", False):
            if yes_token and feed_manager.cache.is_fresh(yes_token):
                row["yes_price"] = feed_manager.cache.get_mid_price(yes_token)
            if no_token and feed_manager.cache.is_fresh(no_token):
                row["no_price"] = feed_manager.cache.get_mid_price(no_token)
            selected_token = yes_token or no_token
            if selected_token and hasattr(feed_manager.cache, "get_price_history"):
                row["history_tail"] = feed_manager.cache.get_price_history(selected_token, max_snapshots=20)
        return row

    def _index_crypto_market_row(self, row: dict[str, Any]) -> None:
        market_id = _normalize_market_id(row.get("id") or row.get("slug"))
        for key in (
            row.get("id"),
            row.get("slug"),
            row.get("condition_id"),
            row.get("conditionId"),
        ):
            normalized = _normalize_market_id(key)
            if normalized:
                self._crypto_markets_by_lookup[normalized] = row
        for token_id in row.get("clob_token_ids") or []:
            normalized = _normalize_market_id(token_id)
            if not normalized:
                continue
            self._crypto_markets_by_lookup[normalized] = row
            if market_id:
                self._crypto_token_to_market_ids.setdefault(normalized, set()).add(market_id)
        asset = str(row.get("asset") or "").strip().upper()
        if asset and market_id:
            self._crypto_asset_to_market_ids.setdefault(asset, set()).add(market_id)

    def _build_crypto_market_payload(self, markets: list[Any]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        feed_manager = self._feed_manager
        reference_runtime = self._reference_runtime
        now_iso = utcnow().isoformat().replace("+00:00", "Z")
        for market in markets:
            row = market.to_dict()
            row["fetched_at"] = now_iso
            asset = str(row.get("asset") or "").strip().upper()
            oracle = reference_runtime.get_oracle_price(asset) if asset else None
            row["oracle_price"] = oracle.get("price") if oracle else None
            row["oracle_source"] = oracle.get("source") if oracle else None
            row["oracle_updated_at_ms"] = oracle.get("updated_at_ms") if oracle else None
            row["oracle_age_seconds"] = oracle.get("age_seconds") if oracle else None
            row["oracle_prices_by_source"] = reference_runtime.get_oracle_prices_by_source(asset) if asset else {}
            row["oracle_history"] = reference_runtime.get_oracle_history(asset, points=80) if asset else []
            row["price_updated_at"] = now_iso

            token_ids = [str(token_id or "").strip() for token_id in (row.get("clob_token_ids") or []) if str(token_id or "").strip()]
            if feed_manager is not None and getattr(feed_manager, "_started", False):
                if len(token_ids) > 0 and feed_manager.cache.is_fresh(token_ids[0], max_age_seconds=float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 1.0) or 1.0)):
                    row["up_price"] = feed_manager.cache.get_mid_price(token_ids[0])
                if len(token_ids) > 1 and feed_manager.cache.is_fresh(token_ids[1], max_age_seconds=float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 1.0) or 1.0)):
                    row["down_price"] = feed_manager.cache.get_mid_price(token_ids[1])
                if token_ids:
                    row["history_tail"] = feed_manager.cache.get_price_history(token_ids[0], max_snapshots=20)
            payload.append(row)
        return payload

    async def _refresh_crypto_markets(
        self,
        *,
        trigger: str,
        full_source_sweep: bool,
        force_refresh: bool = False,
    ) -> None:
        svc = get_crypto_service()
        markets = await asyncio.to_thread(svc.get_live_markets, bool(force_refresh))
        payload = self._build_crypto_market_payload(markets or [])
        self._crypto_markets = payload
        self._crypto_markets_by_lookup = {}
        self._crypto_token_to_market_ids = {}
        self._crypto_asset_to_market_ids = {}
        for row in self._crypto_markets:
            self._index_crypto_market_row(row)
        self._last_crypto_refresh_at = utcnow().isoformat().replace("+00:00", "Z")
        self._last_crypto_trigger = str(trigger)
        await self._sync_crypto_subscriptions()
        await self._publish_crypto_snapshot(payload, trigger=trigger)
        await self._queue_opportunity_dispatch(
            payload,
            trigger=trigger,
            full_source_sweep=full_source_sweep,
        )

    async def _sync_crypto_subscriptions(self) -> None:
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return
        active_tokens = sorted(
            {
                str(token_id or "").strip()
                for market in self._crypto_markets
                for token_id in (market.get("clob_token_ids") or [])
                if str(token_id or "").strip()
            }
        )
        if active_tokens:
            await feed_manager.polymarket_feed.subscribe(active_tokens)

    async def _publish_crypto_snapshot(
        self,
        payload: list[dict[str, Any]],
        *,
        trigger: str,
    ) -> None:
        try:
            await event_bus.publish("crypto_markets_update", {"markets": copy.deepcopy(payload), "trigger": str(trigger)})
        except Exception:
            pass
        self._last_error = None
        try:
            await self._persist_crypto_worker_snapshot()
        except Exception as snapshot_exc:
            logger.warning("Failed to persist crypto worker snapshot after refresh", exc_info=snapshot_exc)

    async def _queue_opportunity_dispatch(
        self,
        payload: list[dict[str, Any]],
        *,
        trigger: str,
        full_source_sweep: bool,
    ) -> None:
        async with self._pending_opportunity_lock:
            self._pending_opportunity_payload = copy.deepcopy(payload)
            self._pending_opportunity_trigger = str(trigger)
            self._pending_opportunity_full_source_sweep = (
                self._pending_opportunity_full_source_sweep or bool(full_source_sweep)
            )
        if self._opportunity_dispatch_task is None or self._opportunity_dispatch_task.done():
            self._opportunity_dispatch_task = asyncio.create_task(
                self._run_opportunity_dispatch_loop(),
                name="market-runtime-opportunity-dispatch",
            )

    async def _run_opportunity_dispatch_loop(self) -> None:
        while True:
            async with self._pending_opportunity_lock:
                payload = self._pending_opportunity_payload
                trigger = self._pending_opportunity_trigger
                full_source_sweep = self._pending_opportunity_full_source_sweep
                self._pending_opportunity_payload = None
                self._pending_opportunity_trigger = None
                self._pending_opportunity_full_source_sweep = False
            if payload is None or trigger is None:
                return
            try:
                event = DataEvent(
                    event_type=EventType.CRYPTO_UPDATE,
                    source="market_runtime",
                    timestamp=utcnow(),
                    payload={"markets": copy.deepcopy(payload), "trigger": str(trigger)},
                )
                opportunities = await event_dispatcher.dispatch(event)
                await get_intent_runtime().publish_opportunities(
                    opportunities,
                    source="crypto",
                    sweep_missing=bool(full_source_sweep),
                    refresh_prices=False,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Crypto opportunity dispatch failed", trigger=str(trigger), exc_info=exc)

    def _on_ws_price_update(
        self,
        token_id: str,
        mid: float,
        bid: float,
        ask: float,
        exchange_ts: float,
        ingest_ts: float,
        sequence: int,
    ) -> None:
        if not self._started:
            return
        normalized = _normalize_market_id(token_id)
        if not normalized or normalized not in self._crypto_token_to_market_ids:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._queue_reactive_token(normalized))

    def _on_reference_update(self, asset: str) -> None:
        if not self._started:
            return
        normalized = str(asset or "").strip().upper()
        if not normalized or normalized not in self._crypto_asset_to_market_ids:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._queue_reactive_asset(normalized))

    async def _queue_reactive_token(self, token_id: str) -> None:
        async with self._pending_reactive_lock:
            self._pending_tokens.add(token_id)
        if self._reactive_task is None or self._reactive_task.done():
            self._reactive_task = asyncio.create_task(self._drain_reactive_updates(), name="market-runtime-reactive")

    async def _queue_reactive_asset(self, asset: str) -> None:
        async with self._pending_reactive_lock:
            self._pending_assets.add(asset)
        if self._reactive_task is None or self._reactive_task.done():
            self._reactive_task = asyncio.create_task(self._drain_reactive_updates(), name="market-runtime-reactive")

    async def _drain_reactive_updates(self) -> None:
        await asyncio.sleep(_WS_REACTIVE_DEBOUNCE_SECONDS)
        async with self._pending_reactive_lock:
            tokens = set(self._pending_tokens)
            self._pending_tokens.clear()
            assets = set(self._pending_assets)
            self._pending_assets.clear()
        if not tokens and not assets:
            return
        market_ids = {
            market_id
            for token_id in tokens
            for market_id in self._crypto_token_to_market_ids.get(token_id, set())
            if market_id
        }
        market_ids.update(
            market_id
            for asset in assets
            for market_id in self._crypto_asset_to_market_ids.get(asset, set())
            if market_id
        )
        if not market_ids:
            return
        selected_rows = [
            row
            for row in self._crypto_markets
            if _normalize_market_id(row.get("id") or row.get("slug")) in market_ids
        ]
        if not selected_rows:
            return
        refreshed_rows = self._rebuild_crypto_rows_from_cache(selected_rows)
        merged_by_id = {
            _normalize_market_id(row.get("id") or row.get("slug")): row
            for row in self._crypto_markets
        }
        for row in refreshed_rows:
            merged_by_id[_normalize_market_id(row.get("id") or row.get("slug"))] = row
        ordered_ids = [_normalize_market_id(row.get("id") or row.get("slug")) for row in self._crypto_markets]
        self._crypto_markets = [merged_by_id[row_id] for row_id in ordered_ids if row_id in merged_by_id]
        self._crypto_markets_by_lookup = {}
        self._crypto_token_to_market_ids = {}
        self._crypto_asset_to_market_ids = {}
        for row in self._crypto_markets:
            self._index_crypto_market_row(row)
        self._last_crypto_refresh_at = utcnow().isoformat().replace("+00:00", "Z")
        trigger = "reference_ws" if assets and not tokens else "crypto_ws" if tokens and not assets else "crypto_reference_ws"
        self._last_crypto_trigger = trigger
        await self._publish_crypto_snapshot(refreshed_rows, trigger=trigger)
        await self._queue_opportunity_dispatch(
            refreshed_rows,
            trigger=trigger,
            full_source_sweep=False,
        )

    def _rebuild_crypto_rows_from_cache(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        feed_manager = self._feed_manager
        reference_runtime = self._reference_runtime
        rebuilt: list[dict[str, Any]] = []
        now_iso = utcnow().isoformat().replace("+00:00", "Z")
        strict_age = float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 1.0) or 1.0)
        for existing in rows:
            row = dict(existing)
            token_ids = [str(token_id or "").strip() for token_id in (row.get("clob_token_ids") or []) if str(token_id or "").strip()]
            if feed_manager is not None and getattr(feed_manager, "_started", False):
                if len(token_ids) > 0 and feed_manager.cache.is_fresh(token_ids[0], max_age_seconds=strict_age):
                    row["up_price"] = feed_manager.cache.get_mid_price(token_ids[0])
                if len(token_ids) > 1 and feed_manager.cache.is_fresh(token_ids[1], max_age_seconds=strict_age):
                    row["down_price"] = feed_manager.cache.get_mid_price(token_ids[1])
                if token_ids:
                    row["history_tail"] = feed_manager.cache.get_price_history(token_ids[0], max_snapshots=20)
            asset = str(row.get("asset") or "").strip().upper()
            oracle = reference_runtime.get_oracle_price(asset) if asset else None
            row["oracle_price"] = oracle.get("price") if oracle else row.get("oracle_price")
            row["oracle_source"] = oracle.get("source") if oracle else row.get("oracle_source")
            row["oracle_updated_at_ms"] = oracle.get("updated_at_ms") if oracle else row.get("oracle_updated_at_ms")
            row["oracle_age_seconds"] = oracle.get("age_seconds") if oracle else row.get("oracle_age_seconds")
            row["oracle_prices_by_source"] = reference_runtime.get_oracle_prices_by_source(asset) if asset else row.get("oracle_prices_by_source")
            row["oracle_history"] = reference_runtime.get_oracle_history(asset, points=80) if asset else row.get("oracle_history")
            row["price_updated_at"] = now_iso
            rebuilt.append(row)
        return rebuilt


_market_runtime: MarketRuntime | None = None


def get_market_runtime() -> MarketRuntime:
    global _market_runtime
    if _market_runtime is None:
        _market_runtime = MarketRuntime()
    return _market_runtime
