from __future__ import annotations

import math
import time
from typing import Any

from config import settings
from services.redis_streams import redis_streams


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


class RedisPriceCache:
    def __init__(self) -> None:
        self._key_prefix = "homerun:live_price"

    def _key(self, token_id: str) -> str:
        return f"{self._key_prefix}:{token_id}"

    async def write_prices(
        self,
        updates: dict[
            str,
            tuple[float | None, float | None, float | None]
            | tuple[float | None, float | None, float | None, float | None, float | None, int | None],
        ],
    ) -> None:
        if not updates:
            return
        stale_seconds = float(settings.WS_PRICE_STALE_SECONDS)
        ttl = max(1, int(math.ceil(stale_seconds)))
        payload: dict[str, dict[str, str]] = {}
        for token_id, raw in updates.items():
            if not token_id:
                continue
            mid: float | None = None
            bid: float | None = None
            ask: float | None = None
            exchange_ts: float | None = None
            ingest_ts: float | None = None
            sequence: int | None = None

            if isinstance(raw, tuple):
                if len(raw) >= 3:
                    mid = raw[0]
                    bid = raw[1]
                    ask = raw[2]
                if len(raw) >= 6:
                    exchange_ts = raw[3]
                    ingest_ts = raw[4]
                    sequence = raw[5]

            ingest = _to_float(ingest_ts)
            if ingest is None or ingest <= 0:
                ingest = time.time()
            exchange = _to_float(exchange_ts)
            if exchange is None or exchange <= 0:
                exchange = ingest

            fields: dict[str, str] = {
                "ts": str(ingest),  # legacy compatibility
                "ingest_ts": str(ingest),
                "exchange_ts": str(exchange),
                "fresh": "1",
            }
            if mid is not None and mid >= 0:
                fields["mid"] = str(float(mid))
            if bid is not None and bid >= 0:
                fields["bid"] = str(float(bid))
            if ask is not None and ask >= 0:
                fields["ask"] = str(float(ask))
            if sequence is not None and int(sequence) > 0:
                fields["seq"] = str(int(sequence))
            if "mid" not in fields and "bid" not in fields and "ask" not in fields:
                continue
            payload[self._key(token_id)] = fields

        if payload:
            await redis_streams.hset_many(payload, expire_seconds=ttl)

    async def read_prices(
        self, token_ids: list[str], *, stale_seconds: float | None = None
    ) -> dict[str, dict[str, float]]:
        if not token_ids:
            return {}
        ttl = float(stale_seconds if stale_seconds is not None else settings.WS_PRICE_STALE_SECONDS)
        if ttl < 0:
            return {}
        keys = [self._key(str(token_id).strip()) for token_id in token_ids if str(token_id).strip()]
        if not keys:
            return {}

        now = time.time()
        rows = await redis_streams.hgetall_many(keys)

        out: dict[str, dict[str, float]] = {}
        for token_id, row in zip(token_ids, rows):
            if not isinstance(row, dict):
                continue
            ts = _to_float(row.get("ingest_ts"))
            if ts is None:
                ts = _to_float(row.get("ts"))
            if ts is None or (now - ts) > ttl:
                continue

            mid = _to_float(row.get("mid"))
            bid = _to_float(row.get("bid"))
            ask = _to_float(row.get("ask"))
            exchange_ts = _to_float(row.get("exchange_ts"))
            if exchange_ts is None:
                exchange_ts = ts
            sequence = int(_to_float(row.get("seq")) or 0)

            if mid is None and bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
            if mid is None and bid is not None:
                mid = bid
            if mid is None and ask is not None:
                mid = ask

            if mid is None:
                continue

            token_key = str(token_id).strip()
            if not token_key:
                continue

            out[token_key] = {
                "mid": float(mid),
                "bid": float(bid) if bid is not None else float(mid),
                "ask": float(ask) if ask is not None else float(mid),
                "ts": float(ts),
                "ingest_ts": float(ts),
                "exchange_ts": float(exchange_ts),
                "sequence": int(sequence),
                "is_fresh": True,
            }

        return out


redis_price_cache = RedisPriceCache()
