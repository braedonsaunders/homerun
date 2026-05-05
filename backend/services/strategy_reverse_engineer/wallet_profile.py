"""Wallet profiling — convert raw trades into a structured trader profile.

The profile is the primary fact-base passed to the LLM in every
iteration of the reverse-engineer loop.  Keep it dense and concrete:
the LLM does not see the raw trade list; it sees these computed
features plus a curated sample of representative trades.

Fields are deliberately conservative — we avoid invented metrics that
might mislead the LLM into making up rules that don't generalize.
"""
from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_and_profile_wallet(
    wallet_address: str,
    *,
    max_trades: int = 2000,
) -> dict[str, Any]:
    """Pull the wallet's trade history from Polymarket and build a profile.

    Returns a dict shaped like:
        {
            "address": str,
            "fetched_count": int,
            "window_start": iso str | None,
            "window_end": iso str | None,
            "summary": {...},     # dense numeric features
            "markets": [...],     # top markets by frequency / volume
            "sample_trades": [...]  # up to N representative trades
        }
    """
    from services.polymarket import polymarket_client

    address = (wallet_address or "").strip().lower()
    if not address:
        raise ValueError("Empty wallet_address")

    try:
        raw_trades = await polymarket_client.get_wallet_trades_paginated(
            address,
            max_trades=int(max_trades),
            page_size=500,
        )
    except Exception as exc:
        logger.exception(
            "fetch_and_profile_wallet: get_wallet_trades_paginated failed for %s",
            address,
        )
        raise RuntimeError(f"failed to fetch trades for wallet {address}: {exc}") from exc

    return profile_trades(address=address, raw_trades=raw_trades or [])


def profile_trades(*, address: str, raw_trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure function: profile a list of trades.  No I/O — used in tests."""
    normalized = [_normalize_trade(t) for t in raw_trades or []]
    normalized = [t for t in normalized if t is not None]
    normalized.sort(key=lambda t: t["timestamp"])

    if not normalized:
        return {
            "address": address,
            "fetched_count": 0,
            "window_start": None,
            "window_end": None,
            "summary": _empty_summary(),
            "markets": [],
            "sample_trades": [],
        }

    summary = _summarize(normalized)
    markets = _market_breakdown(normalized)
    samples = _sample_representative(normalized)

    return {
        "address": address,
        "fetched_count": len(normalized),
        "window_start": normalized[0]["timestamp"].isoformat(),
        "window_end": normalized[-1]["timestamp"].isoformat(),
        "summary": summary,
        "markets": markets,
        "sample_trades": samples,
    }


# ---------------------------------------------------------------------------
# Trade normalization
# ---------------------------------------------------------------------------


def _normalize_trade(raw: Any) -> Optional[dict[str, Any]]:
    """Coerce one Polymarket trade dict into a typed shape we can reason about.

    Polymarket's API surface varies slightly across endpoints; this helper
    is forgiving — it prefers the first non-empty value across known
    aliases for each field.  Trades missing essential fields (timestamp
    or price) are dropped.
    """
    if not isinstance(raw, dict):
        return None
    ts = _coerce_timestamp(
        raw.get("match_time")
        or raw.get("timestamp")
        or raw.get("time")
        or raw.get("created_at")
        or raw.get("createdAt")
    )
    if ts is None:
        return None
    price = _coerce_float(raw.get("price") or raw.get("trade_price") or raw.get("avg_price"))
    if price is None:
        return None
    size = _coerce_float(raw.get("size") or raw.get("amount") or raw.get("shares") or raw.get("qty"))
    side = (str(raw.get("side") or raw.get("trade_side") or "").strip().upper()) or None
    outcome = (str(raw.get("outcome") or raw.get("position") or "").strip().upper()) or None
    # Polymarket /trades returns camelCase keys (conditionId, eventSlug,
    # asset, outcomeIndex).  We accept both camel and snake so this
    # normalizer also handles internal-DB rows that use snake_case.
    market_id = str(
        raw.get("conditionId")
        or raw.get("condition_id")
        or raw.get("market_id")
        or raw.get("marketId")
        or raw.get("market")
        or ""
    ).strip() or None
    title = str(
        raw.get("title")
        or raw.get("market_title")
        or raw.get("question")
        or ""
    ).strip() or None
    # Slug is polybacktest-compatible for crypto Up/Down markets
    # (e.g. ``btc-updown-5m-1777943400``).  Surface it so the
    # agent + dataset resolver can map the trade to a polybacktest
    # market_id without an extra round-trip.
    event_slug = str(
        raw.get("eventSlug")
        or raw.get("event_slug")
        or raw.get("slug")
        or ""
    ).strip() or None
    asset_token = str(raw.get("asset") or "").strip() or None
    notional = _coerce_float(raw.get("notional") or raw.get("usdc_size"))
    if notional is None and size is not None:
        notional = float(price) * float(size)
    return {
        "timestamp": ts,
        "price": float(price),
        "size": float(size) if size is not None else None,
        "notional_usd": float(notional) if notional is not None else None,
        "side": side,
        "outcome": outcome,
        "market_id": market_id,
        "market_title": title,
        "event_slug": event_slug,
        "asset_token": asset_token,
        "raw": raw,
    }


def _coerce_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        if isinstance(value, (int, float)):
            v = float(value)
            if v > 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if "T" in text or "-" in text:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            v = float(text)
            if v > 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def _summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    sides = Counter(t["side"] or "?" for t in trades)
    outcomes = Counter(t["outcome"] or "?" for t in trades)
    notional = [t["notional_usd"] for t in trades if t.get("notional_usd") is not None]
    sizes = [t["size"] for t in trades if t.get("size") is not None]
    prices = [t["price"] for t in trades if t.get("price") is not None]

    by_hour = Counter(t["timestamp"].hour for t in trades)
    by_dow = Counter(t["timestamp"].isoweekday() for t in trades)

    # Time gap between consecutive trades — a proxy for trade cadence.
    inter_trade_seconds: list[float] = []
    prev_ts: Optional[datetime] = None
    for t in trades:
        if prev_ts is not None:
            inter_trade_seconds.append((t["timestamp"] - prev_ts).total_seconds())
        prev_ts = t["timestamp"]

    repeat_markets = Counter(t["market_id"] for t in trades if t.get("market_id"))
    revisited_markets = sum(1 for v in repeat_markets.values() if v > 1)

    return {
        "trade_count": n,
        "unique_markets": len(repeat_markets),
        "revisited_markets": revisited_markets,
        "side_distribution": dict(sides),
        "outcome_distribution": dict(outcomes),
        "notional": _series_stats(notional),
        "size": _series_stats(sizes),
        "price": _series_stats(prices),
        "inter_trade_seconds": _series_stats(inter_trade_seconds),
        "hour_of_day_top": [
            {"hour": int(h), "count": int(c)}
            for h, c in by_hour.most_common(6)
        ],
        "day_of_week_top": [
            {"dow": int(d), "count": int(c)}
            for d, c in by_dow.most_common(7)
        ],
    }


def _series_stats(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p95": None}
    sorted_vals = sorted(values)
    p95_index = max(0, min(len(sorted_vals) - 1, int(round(0.95 * (len(sorted_vals) - 1)))))
    return {
        "count": len(sorted_vals),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": float(statistics.fmean(sorted_vals)),
        "median": float(statistics.median(sorted_vals)),
        "p95": float(sorted_vals[p95_index]),
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "unique_markets": 0,
        "revisited_markets": 0,
        "side_distribution": {},
        "outcome_distribution": {},
        "notional": _series_stats([]),
        "size": _series_stats([]),
        "price": _series_stats([]),
        "inter_trade_seconds": _series_stats([]),
        "hour_of_day_top": [],
        "day_of_week_top": [],
    }


def _market_breakdown(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top markets by trade count and total notional.  Capped at 25 to
    keep the LLM context tight.
    """
    by_market: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trade_count": 0, "notional": 0.0, "title": None, "first_ts": None, "last_ts": None}
    )
    for t in trades:
        mid = t.get("market_id") or "unknown"
        bucket = by_market[mid]
        bucket["trade_count"] += 1
        if t.get("notional_usd") is not None:
            bucket["notional"] += float(t["notional_usd"])
        if not bucket["title"] and t.get("market_title"):
            bucket["title"] = t["market_title"]
        ts = t["timestamp"]
        if bucket["first_ts"] is None or ts < bucket["first_ts"]:
            bucket["first_ts"] = ts
        if bucket["last_ts"] is None or ts > bucket["last_ts"]:
            bucket["last_ts"] = ts
    rows = sorted(by_market.items(), key=lambda kv: kv[1]["trade_count"], reverse=True)[:25]
    return [
        {
            "market_id": mid,
            "title": data["title"],
            "trade_count": int(data["trade_count"]),
            "total_notional_usd": round(float(data["notional"] or 0.0), 2),
            "first_trade": data["first_ts"].isoformat() if data["first_ts"] else None,
            "last_trade": data["last_ts"].isoformat() if data["last_ts"] else None,
        }
        for mid, data in rows
    ]


def _sample_representative(trades: list[dict[str, Any]], *, max_samples: int = 50) -> list[dict[str, Any]]:
    """Pick a curated sample the LLM can read — first, last, and an
    even spread in between.  Cap at ~50 to keep prompts under control.
    """
    if not trades:
        return []
    if len(trades) <= max_samples:
        chosen = list(trades)
    else:
        step = len(trades) / float(max_samples)
        chosen = [trades[int(i * step)] for i in range(max_samples)]
    return [
        {
            "timestamp": t["timestamp"].isoformat(),
            "market_id": t.get("market_id"),
            "market_title": t.get("market_title"),
            "side": t.get("side"),
            "outcome": t.get("outcome"),
            "price": t.get("price"),
            "size": t.get("size"),
            "notional_usd": t.get("notional_usd"),
        }
        for t in chosen
    ]
