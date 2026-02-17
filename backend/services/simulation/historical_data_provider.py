from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.kalshi_client import kalshi_client
from services.polymarket import polymarket_client


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _timeframe_to_seconds(value: str | int | None) -> int:
    if isinstance(value, int):
        return max(60, value)
    text = str(value or "15m").strip().lower()
    if text.endswith("m"):
        return max(60, _as_int(text[:-1], 15) * 60)
    if text.endswith("h"):
        return max(60, _as_int(text[:-1], 1) * 3600)
    if text.endswith("d"):
        return max(60, _as_int(text[:-1], 1) * 86400)
    return max(60, _as_int(text, 900))


def _normalize_ts_seconds(ts: int | None) -> int | None:
    if ts is None:
        return None
    parsed = int(ts)
    if parsed > 10_000_000_000:
        return parsed // 1000
    return parsed


def _normalize_ts_ms(ts: int | None) -> int | None:
    if ts is None:
        return None
    parsed = int(ts)
    if parsed < 10_000_000_000:
        return parsed * 1000
    return parsed


class HistoricalDataProvider:
    """Historical market data fetcher with local cache for reruns."""

    def __init__(self, cache_dir: str | None = None) -> None:
        default_dir = Path(__file__).resolve().parents[2] / "data" / "execution_sim_cache"
        self._cache_dir = Path(cache_dir) if cache_dir else default_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in key)
        return self._cache_dir / f"{safe}.json"

    def _read_cache(self, key: str) -> list[dict[str, Any]] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
        except Exception:
            return None
        return None

    def _write_cache(self, key: str, rows: list[dict[str, Any]]) -> None:
        path = self._cache_path(key)
        try:
            path.write_text(json.dumps(rows, ensure_ascii=True), encoding="utf-8")
        except Exception:
            return

    @staticmethod
    def _points_to_ohlc(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candles: list[dict[str, Any]] = []
        for point in points:
            price = float(point.get("p", 0.0))
            t_ms = _normalize_ts_ms(_as_int(point.get("t"), 0))
            if t_ms is None:
                continue
            price = max(0.0001, min(0.9999, price))
            candles.append(
                {
                    "t": int(t_ms),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": float(point.get("v") or 0.0),
                }
            )
        candles.sort(key=lambda row: row["t"])
        return candles

    async def get_polymarket_candles(
        self,
        *,
        token_id: str,
        start_ts: int | None,
        end_ts: int | None,
        timeframe: str | int | None,
        outcome: str = "yes",
    ) -> list[dict[str, Any]]:
        tf_seconds = _timeframe_to_seconds(timeframe)
        key = f"poly:{token_id}:{start_ts}:{end_ts}:{tf_seconds}:{outcome}"
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        history = await polymarket_client.get_prices_history(
            token_id=str(token_id),
            fidelity=tf_seconds,
            start_ts=_normalize_ts_seconds(start_ts),
            end_ts=_normalize_ts_seconds(end_ts),
        )

        points: list[dict[str, Any]] = []
        use_yes = str(outcome or "yes").strip().lower() not in {"no", "buy_no"}
        for item in history:
            if not isinstance(item, dict):
                continue
            p = float(item.get("p") or 0.0)
            price = p if use_yes else (1.0 - p)
            points.append({"t": item.get("t"), "p": price})

        candles = self._points_to_ohlc(points)
        self._write_cache(key, candles)
        return candles

    async def get_kalshi_candles(
        self,
        *,
        market_ticker: str,
        start_ts: int | None,
        end_ts: int | None,
        timeframe: str | int | None,
        outcome: str = "yes",
    ) -> list[dict[str, Any]]:
        tf_seconds = _timeframe_to_seconds(timeframe)
        key = f"kalshi:{market_ticker}:{start_ts}:{end_ts}:{tf_seconds}:{outcome}"
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        points_by_market = await kalshi_client.get_market_candlesticks_batch(
            [str(market_ticker)],
            start_ts=_normalize_ts_seconds(start_ts),
            end_ts=_normalize_ts_seconds(end_ts),
            period_interval=max(1, tf_seconds // 60),
            include_latest_before_start=True,
        )

        raw_points = points_by_market.get(str(market_ticker), [])
        use_yes = str(outcome or "yes").strip().lower() not in {"no", "buy_no"}
        points: list[dict[str, Any]] = []
        for row in raw_points:
            if not isinstance(row, dict):
                continue
            yes = float(row.get("yes") or 0.0)
            no = float(row.get("no") or max(0.0, 1.0 - yes))
            points.append({"t": row.get("t"), "p": yes if use_yes else no})

        candles = self._points_to_ohlc(points)
        self._write_cache(key, candles)
        return candles
