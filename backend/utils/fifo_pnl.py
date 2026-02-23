from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _ts_from_epoch(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _parse_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        if value > 1e12:
            value = value / 1000.0
        return _ts_from_epoch(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            num = float(s)
            if num > 1e12:
                num = num / 1000.0
            return _ts_from_epoch(num)
        except ValueError:
            pass
        # ISO format variants
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
    return None


@dataclass
class TradeLot:
    open_ts: datetime
    close_ts: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    size: float
    side: str
    market_id: str
    realized_pnl: Optional[float]


@dataclass
class FIFOResult:
    closed_lots: List[TradeLot]
    open_lots: List[TradeLot]
    total_realized_pnl: float
    total_trades: int
    winning_lots: int
    losing_lots: int
    avg_hold_seconds: float
    accuracy: float


def _empty_result() -> FIFOResult:
    return FIFOResult(
        closed_lots=[],
        open_lots=[],
        total_realized_pnl=0.0,
        total_trades=0,
        winning_lots=0,
        losing_lots=0,
        avg_hold_seconds=0.0,
        accuracy=0.0,
    )


@dataclass
class _OpenLot:
    ts: datetime
    price: float
    remaining: float


def compute_fifo_pnl(trades: List[dict], market_id: str) -> FIFOResult:
    if not trades:
        return _empty_result()

    # Normalise and sort by timestamp
    parsed: List[dict] = []
    for t in trades:
        ts = _parse_ts(t.get("timestamp") or t.get("created_at"))
        if ts is None:
            continue
        price = float(t.get("price", 0) or 0)
        size = float(t.get("size", 0) or t.get("amount", 0) or 0)
        side = (t.get("side", "") or "").upper()
        if side not in ("BUY", "SELL") or size <= 0:
            continue
        parsed.append({"ts": ts, "price": price, "size": size, "side": side})

    parsed.sort(key=lambda x: x["ts"])

    buy_queue: deque[_OpenLot] = deque()
    closed_lots: List[TradeLot] = []
    total_realized = 0.0

    for t in parsed:
        if t["side"] == "BUY":
            buy_queue.append(_OpenLot(ts=t["ts"], price=t["price"], remaining=t["size"]))
        else:
            # SELL: match against oldest buys (FIFO)
            sell_remaining = t["size"]
            while sell_remaining > 1e-12 and buy_queue:
                oldest = buy_queue[0]
                matched = min(sell_remaining, oldest.remaining)
                pnl = (t["price"] - oldest.price) * matched

                closed_lots.append(TradeLot(
                    open_ts=oldest.ts,
                    close_ts=t["ts"],
                    entry_price=oldest.price,
                    exit_price=t["price"],
                    size=matched,
                    side="buy",
                    market_id=market_id,
                    realized_pnl=pnl,
                ))
                total_realized += pnl

                oldest.remaining -= matched
                sell_remaining -= matched
                if oldest.remaining < 1e-12:
                    buy_queue.popleft()

    # Remaining open lots
    open_lots: List[TradeLot] = []
    for lot in buy_queue:
        if lot.remaining > 1e-12:
            open_lots.append(TradeLot(
                open_ts=lot.ts,
                close_ts=None,
                entry_price=lot.price,
                exit_price=None,
                size=lot.remaining,
                side="buy",
                market_id=market_id,
                realized_pnl=None,
            ))

    # Summary statistics
    winning = sum(1 for c in closed_lots if c.realized_pnl is not None and c.realized_pnl > 0)
    losing = sum(1 for c in closed_lots if c.realized_pnl is not None and c.realized_pnl < 0)
    total_closed = len(closed_lots)

    hold_seconds: List[float] = []
    for c in closed_lots:
        if c.open_ts and c.close_ts:
            hold_seconds.append((c.close_ts - c.open_ts).total_seconds())

    avg_hold = sum(hold_seconds) / len(hold_seconds) if hold_seconds else 0.0
    accuracy = winning / total_closed if total_closed > 0 else 0.0

    return FIFOResult(
        closed_lots=closed_lots,
        open_lots=open_lots,
        total_realized_pnl=total_realized,
        total_trades=len(parsed),
        winning_lots=winning,
        losing_lots=losing,
        avg_hold_seconds=avg_hold,
        accuracy=accuracy,
    )


def compute_fifo_pnl_multi_market(trades_by_market: Dict[str, List[dict]]) -> Dict[str, FIFOResult]:
    results: Dict[str, FIFOResult] = {}
    for market_id, market_trades in trades_by_market.items():
        results[market_id] = compute_fifo_pnl(market_trades, market_id)
    return results
