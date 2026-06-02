"""Isolated WebSocket feed pool for BROAD market recording.

The trading ``FeedManager`` (services.ws_feeds) owns a single low-latency
``PolymarketWSFeed`` whose ``PriceCache`` drives the orchestrator's price reads,
stop-loss / exit evaluation, reactive scan, position marks, and the frontend
push.  Putting the broad recording subscription set (up to ``max_tokens``=40k,
liquidity-ranked) on that same feed loads the trading-critical socket.

This module gives recording its OWN world, fully decoupled from trading:

  * its own :class:`PriceCache` — the ``market_data_ingestor`` recording
    callbacks (``record_book`` / ``record_trade``) bind HERE, not on the trading
    cache, so book/trade persistence rides this pool;
  * a POOL of :class:`PolymarketWSFeed` connections (Polymarket CLOB tolerates
    only ~a few thousand subscriptions per socket), with the token set sharded
    across them by a stable hash so one connection never carries the whole
    universe;
  * its own stale-eviction loop (mirrors the trading feed's 10-min prune) so the
    pool's subscription sets stay lean.

It deliberately registers NONE of the trading callbacks (no exit eval, no
reactive scan, no strategy dispatch, no frontend push) — recording must never
drive trading.  The orchestrator continues to read prices exclusively from the
trading ``FeedManager``.
"""
from __future__ import annotations

import asyncio
import os
from typing import Iterable, Optional

from services.ws_feeds import PolymarketWSFeed, PriceCache
from utils.logger import get_logger

logger = get_logger(__name__)

# Connections in the recording pool.  The broad recorder targets up to ~40k
# tokens; ~4k/connection keeps each socket within Polymarket CLOB's comfortable
# range, so 10 connections covers the active universe with headroom.  Operator-
# tunable via env for capacity-constrained hosts.
_DEFAULT_POOL_SIZE = max(1, int(os.environ.get("HOMERUN_RECORDER_POOL_SIZE", "10")))
_EVICTION_INTERVAL_SECONDS = 60.0
_EVICTION_MAX_AGE_SECONDS = 600.0


class RecordingFeedManager:
    """Process-wide singleton owning the broad-recording feed pool + cache."""

    _instance: Optional["RecordingFeedManager"] = None

    def __init__(self, pool_size: int = _DEFAULT_POOL_SIZE) -> None:
        self._cache = PriceCache()
        self._pool_size = max(1, int(pool_size))
        self._feeds = [PolymarketWSFeed(cache=self._cache) for _ in range(self._pool_size)]
        self._started = False
        self._start_lock = asyncio.Lock()
        self._eviction_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Recording-only callbacks: the ingestor records book + trade off THIS
        # cache.  No on_change (exit eval) / dispatch / frontend — those belong
        # to the trading feed and must never fire from recording.
        self._cache.add_on_update_callback(self._on_book)
        self._cache.add_on_trade_callback(self._on_trade)

    # -- singleton ----------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "RecordingFeedManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    @property
    def cache(self) -> PriceCache:
        return self._cache

    # -- sharding -----------------------------------------------------------
    def _shard_index(self, token_id: str) -> int:
        return (hash(token_id) & 0x7FFFFFFF) % self._pool_size

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            self._loop = asyncio.get_running_loop()
            for feed in self._feeds:
                await feed.start()
            self._eviction_task = self._loop.create_task(self._eviction_loop())
            self._started = True
            logger.info(
                "RecordingFeedManager started: %d-connection recording pool (isolated from trading feed)",
                self._pool_size,
            )

    async def stop(self) -> None:
        if self._eviction_task is not None:
            self._eviction_task.cancel()
            self._eviction_task = None
        for feed in self._feeds:
            try:
                await feed.stop()
            except Exception:  # noqa: BLE001
                logger.debug("recording feed stop failed", exc_info=True)
        self._started = False

    # -- subscriptions ------------------------------------------------------
    async def subscribe(self, token_ids: Iterable[str]) -> int:
        """Shard ``token_ids`` across the pool and subscribe each shard on its
        own connection.  Idempotent (PolymarketWSFeed.subscribe dedupes)."""
        ids = [str(t).strip() for t in (token_ids or []) if str(t or "").strip()]
        if not ids:
            return 0
        buckets: list[list[str]] = [[] for _ in range(self._pool_size)]
        for tid in ids:
            buckets[self._shard_index(tid)].append(tid)
        subscribed = 0
        for feed, shard in zip(self._feeds, buckets):
            if not shard:
                continue
            try:
                await feed.subscribe(shard)
                subscribed += len(shard)
            except Exception as exc:  # noqa: BLE001
                logger.warning("recording pool subscribe failed (%d tokens): %s", len(shard), exc)
        return subscribed

    async def unsubscribe(self, token_ids: Iterable[str]) -> None:
        ids = [str(t).strip() for t in (token_ids or []) if str(t or "").strip()]
        if not ids:
            return
        buckets: list[list[str]] = [[] for _ in range(self._pool_size)]
        for tid in ids:
            buckets[self._shard_index(tid)].append(tid)
        for feed, shard in zip(self._feeds, buckets):
            if shard:
                try:
                    await feed.unsubscribe(shard)
                except Exception:  # noqa: BLE001
                    logger.debug("recording pool unsubscribe failed", exc_info=True)

    def get_subscribed_assets(self) -> set[str]:
        out: set[str] = set()
        for feed in self._feeds:
            out |= {str(x) for x in getattr(feed, "_subscribed_assets", set())}
        return out

    # -- recording callbacks (bound to THIS cache only) ---------------------
    def _on_book(
        self,
        token_id: str,
        mid: float,
        bid: float,
        ask: float,
        exchange_ts: float | None = None,
        ingest_ts: float | None = None,
        sequence: int | None = None,
    ) -> None:
        try:
            from services.market_data_ingestor import get_market_data_ingestor

            get_market_data_ingestor().record_book(
                token_id=token_id,
                order_book=self._cache.get_order_book(token_id),
                best_bid=bid,
                best_ask=ask,
                exchange_ts=exchange_ts,
                ingest_ts=ingest_ts,
                sequence=sequence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("recording ingestor book record failed", exc_info=exc)

    def _on_trade(self, token_id: str, trade) -> None:
        try:
            from services.market_data_ingestor import get_market_data_ingestor

            get_market_data_ingestor().record_trade(token_id=token_id, trade=trade)
        except Exception as exc:  # noqa: BLE001
            logger.debug("recording ingestor trade record failed", exc_info=exc)

    # -- eviction (mirrors the trading feed's 10-min stale prune) -----------
    async def _eviction_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_EVICTION_INTERVAL_SECONDS)
                evicted = self._cache.evict_stale_ids(_EVICTION_MAX_AGE_SECONDS)
                if evicted:
                    ev = {str(x) for x in evicted}
                    for feed in self._feeds:
                        lock = getattr(feed, "_sub_lock", None)
                        if lock is not None:
                            async with lock:
                                feed._subscribed_assets.difference_update(ev)
                        else:
                            feed._subscribed_assets.difference_update(ev)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.debug("recording pool eviction failed", exc_info=True)

    def status(self) -> dict:
        return {
            "started": self._started,
            "pool_size": self._pool_size,
            "subscribed_tokens": len(self.get_subscribed_assets()),
            "per_connection": [len(getattr(f, "_subscribed_assets", set())) for f in self._feeds],
        }


def get_recording_feed_manager() -> RecordingFeedManager:
    """Shorthand for the process-wide recording pool singleton."""
    return RecordingFeedManager.get_instance()
