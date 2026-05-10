from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from models.database import BacktestAsyncSessionLocal, MarketMicrostructureSnapshot
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


def _normalize_price(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _ts_to_utc(ts: int | None) -> datetime | None:
    normalized = _normalize_ts_ms(ts)
    if normalized is None:
        return None
    return datetime.fromtimestamp(normalized / 1000.0, tz=timezone.utc)


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
    def _points_to_candles(points: list[dict[str, Any]], *, timeframe_seconds: int) -> list[dict[str, Any]]:
        if timeframe_seconds <= 0:
            return []
        bucket_ms = int(timeframe_seconds * 1000)
        grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
        for point in points:
            t_ms = _normalize_ts_ms(_as_int(point.get("t"), 0))
            price = _normalize_price(point.get("p"))
            if t_ms is None or price is None:
                continue
            grouped[(t_ms // bucket_ms) * bucket_ms].append(
                {
                    "t": float(t_ms),
                    "p": float(max(0.0001, min(0.9999, price))),
                    "v": float(point.get("v") or 0.0),
                }
            )
        candles: list[dict[str, Any]] = []
        for bucket_start in sorted(grouped.keys()):
            rows = sorted(grouped[bucket_start], key=lambda row: row["t"])
            prices = [float(row["p"]) for row in rows]
            candles.append(
                {
                    "t": int(bucket_start),
                    "open": prices[0],
                    "high": max(prices),
                    "low": min(prices),
                    "close": prices[-1],
                    "volume": sum(float(row["v"]) for row in rows),
                }
            )
        return candles

    async def get_polymarket_points(
        self,
        *,
        token_id: str,
        start_ts: int | None,
        end_ts: int | None,
        timeframe: str | int | None,
        outcome: str = "yes",
    ) -> list[dict[str, Any]]:
        tf_seconds = _timeframe_to_seconds(timeframe)
        key = f"poly_points:{token_id}:{start_ts}:{end_ts}:{tf_seconds}:{outcome}"
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        start_ms = _normalize_ts_ms(start_ts)
        end_ms = _normalize_ts_ms(end_ts)
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
            t_ms = _normalize_ts_ms(_as_int(item.get("t"), 0))
            if t_ms is None:
                continue
            if start_ms is not None and t_ms < start_ms:
                continue
            if end_ms is not None and t_ms > end_ms:
                continue
            raw_price = _normalize_price(item.get("p"))
            if raw_price is None:
                continue
            selected_price = raw_price if use_yes else max(0.0, min(1.0, 1.0 - raw_price))
            points.append({"t": int(t_ms), "p": float(selected_price), "v": float(item.get("v") or 0.0)})

        points.sort(key=lambda row: int(row["t"]))
        self._write_cache(key, points)
        return points

    async def get_kalshi_points(
        self,
        *,
        market_ticker: str,
        start_ts: int | None,
        end_ts: int | None,
        timeframe: str | int | None,
        outcome: str = "yes",
    ) -> list[dict[str, Any]]:
        tf_seconds = _timeframe_to_seconds(timeframe)
        key = f"kalshi_points:{market_ticker}:{start_ts}:{end_ts}:{tf_seconds}:{outcome}"
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        start_ms = _normalize_ts_ms(start_ts)
        end_ms = _normalize_ts_ms(end_ts)
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
            t_ms = _normalize_ts_ms(_as_int(row.get("t"), 0))
            if t_ms is None:
                continue
            if start_ms is not None and t_ms < start_ms:
                continue
            if end_ms is not None and t_ms > end_ms:
                continue
            yes = _normalize_price(row.get("yes"))
            if yes is None:
                continue
            no = max(0.0, min(1.0, 1.0 - yes))
            points.append(
                {
                    "t": int(t_ms),
                    "p": float(yes if use_yes else no),
                    "v": float(row.get("volume") or row.get("v") or 0.0),
                }
            )

        points.sort(key=lambda point: int(point["t"]))
        self._write_cache(key, points)
        return points

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
        key = f"poly_candles:{token_id}:{start_ts}:{end_ts}:{tf_seconds}:{outcome}"
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        points = await self.get_polymarket_points(
            token_id=token_id,
            start_ts=start_ts,
            end_ts=end_ts,
            timeframe=tf_seconds,
            outcome=outcome,
        )
        candles = self._points_to_candles(points, timeframe_seconds=tf_seconds)
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
        key = f"kalshi_candles:{market_ticker}:{start_ts}:{end_ts}:{tf_seconds}:{outcome}"
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        points = await self.get_kalshi_points(
            market_ticker=market_ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            timeframe=tf_seconds,
            outcome=outcome,
        )
        candles = self._points_to_candles(points, timeframe_seconds=tf_seconds)
        self._write_cache(key, candles)
        return candles

    async def get_polymarket_microstructure_context(
        self,
        *,
        token_id: str,
        event_ts: int | None,
        lookback_seconds: float = 30.0,
        forward_seconds: float = 6.0,
    ) -> dict[str, Any]:
        normalized_token = str(token_id or "").strip().lower()
        event_dt = _ts_to_utc(event_ts)
        if not normalized_token or event_dt is None:
            return {}
        start_dt = event_dt - timedelta(seconds=max(1.0, float(lookback_seconds)))
        end_dt = event_dt + timedelta(seconds=max(0.0, float(forward_seconds)))
        async with BacktestAsyncSessionLocal() as session:
            before_book_query = (
                select(MarketMicrostructureSnapshot)
                .where(
                    MarketMicrostructureSnapshot.token_id == normalized_token,
                    MarketMicrostructureSnapshot.snapshot_type == "book",
                    MarketMicrostructureSnapshot.observed_at <= event_dt,
                )
                .order_by(MarketMicrostructureSnapshot.observed_at.desc())
                .limit(1)
            )
            book_row = (await session.execute(before_book_query)).scalar_one_or_none()
            if book_row is None:
                after_book_query = (
                    select(MarketMicrostructureSnapshot)
                    .where(
                        MarketMicrostructureSnapshot.token_id == normalized_token,
                        MarketMicrostructureSnapshot.snapshot_type == "book",
                        MarketMicrostructureSnapshot.observed_at > event_dt,
                        MarketMicrostructureSnapshot.observed_at <= event_dt + timedelta(seconds=2),
                    )
                    .order_by(MarketMicrostructureSnapshot.observed_at.asc())
                    .limit(1)
                )
                book_row = (await session.execute(after_book_query)).scalar_one_or_none()

            trades_query = (
                select(MarketMicrostructureSnapshot)
                .where(
                    MarketMicrostructureSnapshot.token_id == normalized_token,
                    MarketMicrostructureSnapshot.snapshot_type == "trade",
                    MarketMicrostructureSnapshot.observed_at >= start_dt,
                    MarketMicrostructureSnapshot.observed_at <= end_dt,
                )
                .order_by(MarketMicrostructureSnapshot.observed_at.asc())
            )
            trade_rows = list((await session.execute(trades_query)).scalars().all())

        if book_row is None:
            return {}
        observed = book_row.observed_at
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        else:
            observed = observed.astimezone(timezone.utc)
        age_ms = (event_dt - observed).total_seconds() * 1000.0
        trades = []
        for row in trade_rows:
            observed_trade = row.observed_at
            if observed_trade.tzinfo is None:
                observed_trade = observed_trade.replace(tzinfo=timezone.utc)
            else:
                observed_trade = observed_trade.astimezone(timezone.utc)
            if row.trade_price is None or row.trade_size is None:
                continue
            trades.append(
                {
                    "price": float(row.trade_price),
                    "size": float(row.trade_size),
                    "side": str(row.trade_side or "BUY").upper(),
                    "timestamp": observed_trade.timestamp(),
                }
            )
        return {
            "execution_order_book": {
                "bids": list(book_row.bids_json or []),
                "asks": list(book_row.asks_json or []),
            },
            "execution_recent_trades": trades,
            "execution_order_book_age_ms": max(0.0, age_ms),
            "execution_microstructure": {
                "source": "market_microstructure_snapshots",
                "book_observed_at": observed.isoformat(),
                "book_sequence": book_row.sequence,
                "trade_count": len(trades),
                "lookback_seconds": float(lookback_seconds),
                "forward_seconds": float(forward_seconds),
            },
        }
