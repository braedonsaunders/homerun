"""Sub-second crypto reference-price recorder → Parquet.

Taps the **same** WS sources the live trading stack uses — Chainlink via
Polymarket RTDS (the resolution source of truth) and Binance direct
bookTicker (the ~1s lead signal) — and persists every tick to Parquet
under the canonical provider layout so the data is:

  * **catalogued** automatically by ``parquet_scanner`` as
    ``provider_datasets`` rows (``storage_type='parquet'``,
    ``asset_class='reference'``), and
  * **browsable** in Data Lab, and
  * **available to backtests** as a reference price series (Phase 2 seam).

Why a dedicated recorder rather than the in-memory ``ChainlinkFeed``
history buffer:

  * the live feeds keep only a rolling 3h in-memory buffer and never
    persist — restart loses everything;
  * we want a durable, queryable, file-backed record for offline
    analysis (e.g. reverse-engineering oracle-lag strategies) and for
    replaying in the backtester.

This recorder runs its OWN feed instances (not the trading singletons)
so it is fully independent of the trading lifecycle — it records even
when trading is off, and never interferes with the hot path.

Storage layout (see ``parquet_schema.parquet_path_for``)::

    {root}/crypto_ohlc/{coin}/{window_start__window_end}/reference__{series_id}.parquet

where ``series_id`` is ``ref_{coin}_{source}`` (e.g. ``ref_btc_chainlink``,
``ref_btc_binance_direct``).  Files are rewritten atomically on each flush
for the active window; closed windows are finalised once.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from services.binance_feed import BinanceFeed
from services.chainlink_feed import ChainlinkFeed, OraclePrice
from services.external_data.parquet_schema import (
    REFERENCE_SCHEMA,
    SCHEMA_VERSION,
    parquet_path_for,
)
from utils.logger import get_logger

logger = get_logger(__name__)

PROVIDER = "crypto_ohlc"
DEFAULT_ASSETS = ("BTC", "ETH")
DEFAULT_WINDOW_SECONDS = 900  # 15-min buckets, aligned with the markets
DEFAULT_FLUSH_INTERVAL_SECONDS = 10.0
DEFAULT_CATALOG_INTERVAL_SECONDS = 30.0


def _series_id(asset: str, source: str) -> str:
    return f"ref_{asset.lower()}_{source}"


class _Tick:
    __slots__ = ("observed_at_us", "price", "bid", "ask", "source_ts_ms")

    def __init__(self, observed_at_us: int, price: float, bid: Optional[float], ask: Optional[float], source_ts_ms: int):
        self.observed_at_us = observed_at_us
        self.price = price
        self.bid = bid
        self.ask = ask
        self.source_ts_ms = source_ts_ms


class CryptoOHLCRecorder:
    """Records Chainlink + Binance-direct ticks to Parquet."""

    def __init__(
        self,
        *,
        assets: tuple[str, ...] = DEFAULT_ASSETS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        catalog_interval_seconds: float = DEFAULT_CATALOG_INTERVAL_SECONDS,
        root: Path | None = None,
        reference_runtime: object | None = None,
    ):
        self._assets = {a.upper() for a in assets}
        self._window_seconds = max(60, int(window_seconds))
        self._flush_interval = max(1.0, float(flush_interval_seconds))
        self._catalog_interval = max(5.0, float(catalog_interval_seconds))
        self._root = root

        # Integrated mode: tap a shared ReferenceRuntime (no duplicate WS
        # connections — this is how the recorder runs inside the app).
        # Standalone mode: own dedicated feeds (CLI / offline captures).
        self._reference_runtime = reference_runtime
        self._integrated = reference_runtime is not None
        self._chainlink = None if self._integrated else ChainlinkFeed()
        self._binance = None if self._integrated else BinanceFeed()
        # Dedup latest source tick by (asset, source) -> source_ts_ms so an
        # on_update notify (which fires per underlying feed update) doesn't
        # re-record an unchanged source's last price.
        self._last_seen_ms: dict[tuple[str, str], int] = {}

        # buffer: (asset, source, bucket_start_s) -> list[_Tick]
        self._buffers: dict[tuple[str, str, int], list[_Tick]] = {}
        # buckets whose ticks changed since last flush
        self._dirty: set[tuple[str, str, int]] = set()
        # buckets already finalised (closed + flushed) — memory freed
        self._finalised: set[tuple[str, str, int]] = set()

        self._flush_task: Optional[asyncio.Task] = None
        self._stopped = False
        self._last_catalog_mono = 0.0
        self._tick_count = 0
        self._flush_count = 0

    # Sources worth persisting (skip the slow RTDS-relayed "binance"; the
    # direct feed supersedes it).  Chainlink + chainlink_direct = resolution
    # truth; binance_direct = the ~1s lead signal.
    _RECORDED_SOURCES = ("chainlink", "chainlink_direct", "binance_direct")

    # ── stats ────────────────────────────────────────────────────────
    @property
    def started(self) -> bool:
        return self._flush_task is not None and not self._flush_task.done()

    def stats(self) -> dict:
        return {
            "running": self.started,
            "mode": "integrated" if self._integrated else "standalone",
            "assets": sorted(self._assets),
            "window_seconds": self._window_seconds,
            "ticks_recorded": self._tick_count,
            "flushes": self._flush_count,
            "active_buffers": len(self._buffers),
            "chainlink_connected": bool(self._chainlink.started) if self._chainlink else None,
            "binance_connected": bool(self._binance.started) if self._binance else None,
        }

    # ── tick ingestion (runs on the asyncio loop via feed callbacks) ──
    def _bucket_start(self, ts_s: float) -> int:
        return int(ts_s // self._window_seconds) * self._window_seconds

    def _record(self, asset: str, source: str, price: float, bid, ask, source_ts_ms: int) -> None:
        if asset not in self._assets:
            return
        now = time.time()
        bucket = self._bucket_start(now)
        key = (asset, source, bucket)
        buf = self._buffers.get(key)
        if buf is None:
            buf = []
            self._buffers[key] = buf
        buf.append(_Tick(int(now * 1_000_000), float(price), bid, ask, int(source_ts_ms)))
        self._dirty.add(key)
        self._tick_count += 1

    def _on_chainlink(self, oracle: OraclePrice) -> None:
        # Only persist the authoritative Chainlink source from this feed.
        if getattr(oracle, "source", None) != "chainlink":
            return
        self._record(oracle.asset, "chainlink", oracle.price, None, None, oracle.updated_at_ms)

    def _on_binance(self, asset: str, mid: float, bid: float, ask: float, ts_ms: int) -> None:
        self._record(asset, "binance_direct", mid, bid, ask, ts_ms)

    def _on_ref_asset(self, asset: str) -> None:
        """Integrated-mode tap: ReferenceRuntime fires this per asset update.

        Reads the latest price per source from the shared feed and records
        any source whose timestamp advanced since we last saw it.  Bid/ask
        aren't exposed per-source by ReferenceRuntime, so reference rows
        carry price only (bid/ask null) in integrated mode.
        """
        asset_u = str(asset or "").strip().upper()
        if asset_u not in self._assets or self._reference_runtime is None:
            return
        try:
            by_source = self._reference_runtime.get_oracle_prices_by_source(asset_u)
        except Exception:
            return
        for source, row in (by_source or {}).items():
            if source not in self._RECORDED_SOURCES:
                continue
            ts_ms = int(row.get("updated_at_ms") or 0)
            price = row.get("price")
            if not ts_ms or price is None or float(price) <= 0:
                continue
            key = (asset_u, source)
            if self._last_seen_ms.get(key) == ts_ms:
                continue  # unchanged since last notify — skip
            self._last_seen_ms[key] = ts_ms
            self._record(asset_u, source, float(price), None, None, ts_ms)

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start(self) -> None:
        if self.started:
            return
        self._stopped = False
        if self._integrated:
            # Tap the shared runtime; do NOT start/own the feeds.
            self._reference_runtime.on_update(self._on_ref_asset)
        else:
            self._chainlink.on_update(self._on_chainlink)
            self._binance.on_update(self._on_binance)
            await self._chainlink.start()
            await self._binance.start()
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "CryptoOHLCRecorder: started (mode=%s, assets=%s, window=%ss, flush=%ss)",
            "integrated" if self._integrated else "standalone",
            sorted(self._assets), self._window_seconds, self._flush_interval,
        )

    async def stop(self) -> None:
        self._stopped = True
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        if self._integrated:
            try:
                self._reference_runtime.remove_on_update(self._on_ref_asset)
            except Exception:
                pass
        else:
            await self._chainlink.stop()
            await self._binance.stop()
        # Final flush of everything still buffered.
        await self._flush(force_all=True)
        await self._catalog()
        logger.info("CryptoOHLCRecorder: stopped (%d ticks, %d flushes)", self._tick_count, self._flush_count)

    async def _flush_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush()
                if (time.monotonic() - self._last_catalog_mono) >= self._catalog_interval:
                    await self._catalog()
                    self._last_catalog_mono = time.monotonic()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("CryptoOHLCRecorder: flush loop error")

    async def _flush(self, *, force_all: bool = False) -> None:
        current_bucket = self._bucket_start(time.time())
        # Snapshot the dirty set on the loop thread, then write off-thread.
        keys = list(self._dirty) if not force_all else list(self._buffers.keys())
        if not keys:
            return
        for key in keys:
            asset, source, bucket = key
            buf = self._buffers.get(key)
            if not buf:
                self._dirty.discard(key)
                continue
            # Copy tick rows on the loop thread (callbacks append on the
            # same loop, so this snapshot is race-free), then hand off.
            rows = list(buf)
            await asyncio.to_thread(self._write_parquet, asset, source, bucket, rows)
            self._dirty.discard(key)
            self._flush_count += 1
            # Finalise + free memory for closed buckets.
            if bucket < current_bucket and not force_all:
                self._finalised.add(key)
                self._buffers.pop(key, None)

    def _write_parquet(self, asset: str, source: str, bucket: int, rows: list[_Tick]) -> None:
        start = datetime.fromtimestamp(bucket, tz=timezone.utc)
        end = datetime.fromtimestamp(bucket + self._window_seconds, tz=timezone.utc)
        sid = _series_id(asset, source)
        pp = parquet_path_for(
            provider=PROVIDER, coin=asset.lower(), token_id=sid,
            start=start, end=end, kind="reference", root=self._root,
        )
        pp.window_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "token_id": pa.array([sid] * len(rows), pa.string()),
                "observed_at_us": pa.array([r.observed_at_us for r in rows], pa.int64()),
                "asset": pa.array([asset] * len(rows), pa.string()),
                "source": pa.array([source] * len(rows), pa.string()),
                "price": pa.array([r.price for r in rows], pa.float64()),
                "bid": pa.array([r.bid for r in rows], pa.float64()),
                "ask": pa.array([r.ask for r in rows], pa.float64()),
                "source_ts_ms": pa.array([r.source_ts_ms for r in rows], pa.int64()),
            },
            schema=REFERENCE_SCHEMA,
        )
        table = table.replace_schema_metadata({"schema_version": SCHEMA_VERSION})
        tmp = pp.file_path.with_suffix(".parquet.tmp")
        pq.write_table(table, str(tmp), compression="zstd")
        os.replace(tmp, pp.file_path)

    async def _catalog(self) -> None:
        try:
            from services.external_data.parquet_scanner import rescan_parquet_root
            await rescan_parquet_root(root=self._root) if self._root else await rescan_parquet_root()
        except Exception:
            logger.exception("CryptoOHLCRecorder: catalog rescan failed")


# ── module-level manager — runs on a DEDICATED thread + event loop ────
#
# Critical: when started from inside a worker (e.g. the trading plane),
# the recorder must NOT share that worker's asyncio loop.  The trading
# loop is saturated by the hot path, so the recorder's WS-recv coroutines
# get starved — they wake rarely and drain a backlog of buffered frames in
# one slice (observed: dozens of ticks at an identical microsecond, then a
# long stall).  Running the feeds + flush on a dedicated thread with its
# own loop gives the recorder its own scheduling budget and yields steady,
# dense capture regardless of how busy the host process is.
_recorder: Optional[CryptoOHLCRecorder] = None
_rec_thread: Optional[threading.Thread] = None
_rec_loop: Optional[asyncio.AbstractEventLoop] = None


def get_recorder() -> Optional[CryptoOHLCRecorder]:
    return _recorder


def _thread_main(kwargs: dict, ready: threading.Event) -> None:
    global _rec_loop, _recorder
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _rec_loop = loop
    rec = CryptoOHLCRecorder(**kwargs)
    _recorder = rec
    try:
        loop.run_until_complete(rec.start())
    except Exception:
        logger.exception("CryptoOHLCRecorder: thread start failed")
    finally:
        ready.set()
    try:
        loop.run_forever()
    finally:
        try:
            loop.close()
        except Exception:
            pass
        _rec_loop = None
        _recorder = None


async def start_recorder(**kwargs) -> Optional[CryptoOHLCRecorder]:
    """Start the recorder on its own thread + loop (idempotent)."""
    global _rec_thread
    if _rec_thread is not None and _rec_thread.is_alive():
        return _recorder
    ready = threading.Event()
    _rec_thread = threading.Thread(
        target=_thread_main, args=(dict(kwargs), ready), daemon=True, name="crypto-ohlc-recorder"
    )
    _rec_thread.start()
    await asyncio.to_thread(ready.wait, 15.0)
    return _recorder


async def stop_recorder() -> None:
    """Stop the recorder thread: final flush + catalog on its loop, then halt."""
    global _rec_thread
    loop = _rec_loop
    rec = _recorder
    if loop is not None and rec is not None:
        def _shutdown() -> None:
            try:
                fut = asyncio.run_coroutine_threadsafe(rec.stop(), loop)
                fut.result(timeout=25.0)
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        await asyncio.to_thread(_shutdown)
    if _rec_thread is not None:
        await asyncio.to_thread(_rec_thread.join, 10.0)
    _rec_thread = None


# ── standalone runner (for validation / one-off captures) ─────────────
async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Standalone crypto reference recorder")
    parser.add_argument("--seconds", type=int, default=90, help="record duration")
    parser.add_argument("--assets", default="BTC,ETH")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--flush", type=float, default=5.0)
    args = parser.parse_args()

    rec = CryptoOHLCRecorder(
        assets=tuple(a.strip().upper() for a in args.assets.split(",") if a.strip()),
        window_seconds=args.window,
        flush_interval_seconds=args.flush,
        catalog_interval_seconds=max(10.0, args.flush * 2),
    )
    await rec.start()
    print(f"recording for {args.seconds}s ...", flush=True)
    try:
        for _ in range(args.seconds):
            await asyncio.sleep(1)
            s = rec.stats()
            print(f"  ticks={s['ticks_recorded']} flushes={s['flushes']} "
                  f"cl={s['chainlink_connected']} bn={s['binance_connected']} bufs={s['active_buffers']}",
                  end="\r", flush=True)
    finally:
        print()
        await rec.stop()
        print("final stats:", rec.stats())


if __name__ == "__main__":
    asyncio.run(_main())
