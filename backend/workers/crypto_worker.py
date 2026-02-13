"""Crypto worker: owns BTC/ETH/SOL/XRP 5m/15m high-frequency data + signal loop.

Runs as a dedicated process and persists worker snapshot + normalized crypto signals.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from collections import deque
from datetime import datetime, timezone

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if os.getcwd() != _BACKEND:
    os.chdir(_BACKEND)

from utils.utcnow import utcnow
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
                end_ts = datetime.fromisoformat(
                    end_time.replace("Z", "+00:00")
                ).timestamp()
                if now_ts - end_ts > 1800:
                    continue
            except ValueError:
                pass

        if slug not in svc._price_to_beat:
            svc._price_to_beat[slug] = ptb
            restored += 1

    return restored


def _build_crypto_market_payload() -> list[dict]:
    svc = get_crypto_service()
    feed = get_chainlink_feed()
    markets = svc.get_live_markets()
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
                round((now_ms - oracle.updated_at_ms) / 1000, 1)
                if oracle.updated_at_ms
                else None
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
                "age_seconds": (
                    round((now_ms - int(updated_at)) / 1000, 1) if updated_at else None
                ),
            }

        row["price_to_beat"] = svc._price_to_beat.get(market.slug)
        row["oracle_history"] = _oracle_history_payload(market.asset)
        payload.append(row)

    return payload


async def _run_loop() -> None:
    logger.info("Crypto worker started")
    worker_name = "crypto"

    startup_stats = {"market_count": 0, "signals_emitted_last_run": 0, "markets": []}
    async with AsyncSessionLocal() as session:
        await ensure_worker_control(session, worker_name, default_interval=2)
        previous_snapshot = await read_worker_snapshot(session, worker_name)
        previous_stats = (
            previous_snapshot.get("stats")
            if isinstance(previous_snapshot.get("stats"), dict)
            else {}
        )
        previous_markets = (
            previous_stats.get("markets")
            if isinstance(previous_stats.get("markets"), list)
            else []
        )
        if previous_markets:
            startup_stats = {
                "market_count": len(previous_markets),
                "signals_emitted_last_run": int(
                    previous_stats.get("signals_emitted_last_run") or 0
                ),
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

    while True:
        async with AsyncSessionLocal() as session:
            control = await read_worker_control(session, worker_name, default_interval=2)

        interval = max(1, min(60, int(control.get("interval_seconds") or 2)))
        paused = bool(control.get("is_paused", False))
        enabled = bool(control.get("is_enabled", True))
        requested = control.get("requested_run_at") is not None

        if (not enabled or paused) and not requested:
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

        run_at = utcnow()
        markets_payload: list[dict] = []
        emitted = 0
        err_text = None

        try:
            markets_payload = _build_crypto_market_payload()
            async with AsyncSessionLocal() as session:
                emitted = await emit_crypto_market_signals(session, markets_payload)
                if requested:
                    await clear_worker_run_request(session, worker_name)

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
                        "markets": markets_payload,
                    },
                )

            logger.info(
                "Crypto cycle complete: markets=%s signals=%s",
                len(markets_payload),
                emitted,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            err_text = str(exc)
            logger.exception("Crypto worker cycle failed: %s", exc)
            async with AsyncSessionLocal() as session:
                if requested:
                    await clear_worker_run_request(session, worker_name)
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
                        "markets": markets_payload,
                    },
                )

        sleep_for = 0.5 if err_text else interval
        await asyncio.sleep(max(0.5, sleep_for))


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


if __name__ == "__main__":
    asyncio.run(main())
