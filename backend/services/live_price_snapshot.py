from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from typing import Iterable

from services.polymarket import polymarket_client

# Keep HTTP load bounded while still feeling "live" in the UI.
DEFAULT_PRICE_TTL_SECONDS = 8.0
_MAX_CACHE_ENTRIES = 6000

# Keep chart behavior consistent between initial snapshot and live updates.
DEFAULT_HISTORY_WINDOW_SECONDS = 2 * 60 * 60
DEFAULT_HISTORY_MAX_POINTS = 120
DEFAULT_HISTORY_MIN_APPEND_SECONDS = 60
DEFAULT_HISTORY_SIGNIFICANT_MOVE = 0.01  # 1 cent

_price_cache_lock = asyncio.Lock()
_price_cache: dict[str, tuple[float | None, float]] = {}


def _normalize_token_id(value: object) -> str:
    return str(value or "").strip().lower()


def _coerce_mid_price(value: object) -> float | None:
    try:
        price = float(value)
    except Exception:
        return None
    if price != price or price in (float("inf"), float("-inf")):
        return None
    if price < 0.0 or price > 1.0:
        return None
    return price


def _coerce_epoch_ms(value: object) -> int | None:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except Exception:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return int(dt.timestamp() * 1000)
            except Exception:
                return None
        else:
            value = parsed

    try:
        ts = float(value)
    except Exception:
        return None
    if ts != ts or ts in (float("inf"), float("-inf")):
        return None
    if ts < 10_000_000_000:
        ts *= 1000.0
    return int(ts)


def _downsample_points(
    points: list[tuple[int, float, float]],
    max_points: int,
) -> list[tuple[int, float, float]]:
    if len(points) <= max_points:
        return points
    if max_points <= 1:
        return [points[-1]]

    span = len(points) - 1
    indexes = {0, span}
    for i in range(1, max_points - 1):
        idx = round(i * span / (max_points - 1))
        indexes.add(int(idx))

    return [points[idx] for idx in sorted(indexes)]


def normalize_binary_price_history(
    points: object,
    *,
    now_ms: int | None = None,
    window_seconds: int = DEFAULT_HISTORY_WINDOW_SECONDS,
    max_points: int = DEFAULT_HISTORY_MAX_POINTS,
) -> list[dict[str, float]]:
    if not isinstance(points, list):
        return []

    parsed: list[tuple[int, float, float]] = []
    for raw in points:
        if not isinstance(raw, dict):
            continue

        yes_price = _coerce_mid_price(raw.get("yes"))
        no_price = _coerce_mid_price(raw.get("no"))
        if yes_price is None:
            yes_price = _coerce_mid_price(raw.get("y"))
        if yes_price is None:
            yes_price = _coerce_mid_price(raw.get("idx_0"))
        if yes_price is None:
            yes_price = _coerce_mid_price(raw.get("p"))
        if no_price is None:
            no_price = _coerce_mid_price(raw.get("n"))
        if no_price is None:
            no_price = _coerce_mid_price(raw.get("idx_1"))

        if yes_price is None and no_price is not None and 0.0 <= no_price <= 1.0:
            yes_price = float(1.0 - no_price)
        if no_price is None and yes_price is not None and 0.0 <= yes_price <= 1.0:
            no_price = float(1.0 - yes_price)
        if yes_price is None or no_price is None:
            continue

        ts_ms = _coerce_epoch_ms(
            raw.get("t")
            if raw.get("t") is not None
            else (
                raw.get("ts")
                if raw.get("ts") is not None
                else (
                    raw.get("time")
                    if raw.get("time") is not None
                    else raw.get("timestamp")
                )
            )
        )
        if ts_ms is None:
            continue

        parsed.append((ts_ms, round(yes_price, 6), round(no_price, 6)))

    if not parsed:
        return []

    parsed.sort(key=lambda point: point[0])

    now_value = now_ms if now_ms is not None else int(time.time() * 1000)
    cutoff_ms = now_value - max(60, int(window_seconds)) * 1000
    filtered = [point for point in parsed if point[0] >= cutoff_ms]
    if not filtered:
        filtered = parsed[-max_points:]

    if len(filtered) > max_points:
        filtered = _downsample_points(filtered, max_points)

    return [
        {
            "t": float(ts_ms),
            "yes": yes_price,
            "no": no_price,
            "idx_0": yes_price,
            "idx_1": no_price,
        }
        for ts_ms, yes_price, no_price in filtered
    ]


def _should_append_to_normalized_history(
    normalized_history: list[dict[str, float]],
    *,
    yes_price: float,
    no_price: float,
    now_ms: int,
    min_append_seconds: int = DEFAULT_HISTORY_MIN_APPEND_SECONDS,
    significant_move: float = DEFAULT_HISTORY_SIGNIFICANT_MOVE,
) -> bool:
    if not normalized_history:
        return True

    last = normalized_history[-1]
    last_ts = _coerce_epoch_ms(last.get("t"))
    if last_ts is None:
        return True

    if now_ms - last_ts >= max(1, int(min_append_seconds)) * 1000:
        return True

    last_yes = _coerce_mid_price(last.get("yes"))
    last_no = _coerce_mid_price(last.get("no"))
    if last_yes is None or last_no is None:
        return True

    move_threshold = abs(float(significant_move))
    if abs(last_yes - yes_price) >= move_threshold:
        return True
    if abs(last_no - no_price) >= move_threshold:
        return True
    return False


def append_live_binary_price_point(
    history: object,
    *,
    yes_price: float,
    no_price: float,
    now_ms: int | None = None,
    window_seconds: int = DEFAULT_HISTORY_WINDOW_SECONDS,
    max_points: int = DEFAULT_HISTORY_MAX_POINTS,
    min_append_seconds: int = DEFAULT_HISTORY_MIN_APPEND_SECONDS,
    significant_move: float = DEFAULT_HISTORY_SIGNIFICANT_MOVE,
) -> list[dict[str, float]]:
    now_value = now_ms if now_ms is not None else int(time.time() * 1000)
    normalized = normalize_binary_price_history(
        history,
        now_ms=now_value,
        window_seconds=window_seconds,
        max_points=max_points,
    )
    if not _should_append_to_normalized_history(
        normalized,
        yes_price=yes_price,
        no_price=no_price,
        now_ms=now_value,
        min_append_seconds=min_append_seconds,
        significant_move=significant_move,
    ):
        return normalized

    normalized.append(
        {
            "t": float(now_value),
            "yes": round(yes_price, 6),
            "no": round(no_price, 6),
            "idx_0": round(yes_price, 6),
            "idx_1": round(no_price, 6),
        }
    )
    return normalize_binary_price_history(
        normalized,
        now_ms=now_value,
        window_seconds=window_seconds,
        max_points=max_points,
    )


def _prune_cache(now_monotonic: float) -> None:
    if len(_price_cache) <= _MAX_CACHE_ENTRIES:
        return

    expired_keys = [key for key, (_, expires_at) in _price_cache.items() if expires_at <= now_monotonic]
    for key in expired_keys:
        _price_cache.pop(key, None)

    if len(_price_cache) <= _MAX_CACHE_ENTRIES:
        return

    # Fallback pruning: drop oldest expiring entries first.
    overflow = len(_price_cache) - _MAX_CACHE_ENTRIES
    for key, _ in sorted(_price_cache.items(), key=lambda item: item[1][1])[:overflow]:
        _price_cache.pop(key, None)


async def get_live_mid_prices(
    token_ids: Iterable[object],
    *,
    ttl_seconds: float = DEFAULT_PRICE_TTL_SECONDS,
) -> dict[str, float]:
    """Return live-ish CLOB midpoint prices keyed by normalized token id.

    Uses a short-lived in-memory cache to avoid hammering CLOB endpoints when
    the UI refreshes frequently across multiple views.
    """
    normalized_tokens: list[str] = []
    seen: set[str] = set()
    for raw in token_ids:
        token_id = _normalize_token_id(raw)
        if not token_id or token_id in seen:
            continue
        seen.add(token_id)
        normalized_tokens.append(token_id)

    if not normalized_tokens:
        return {}

    now_monotonic = time.monotonic()
    cached_prices: dict[str, float] = {}
    missing_tokens: list[str] = []

    async with _price_cache_lock:
        for token_id in normalized_tokens:
            cached = _price_cache.get(token_id)
            if not cached:
                missing_tokens.append(token_id)
                continue
            cached_price, expires_at = cached
            if expires_at <= now_monotonic:
                missing_tokens.append(token_id)
                continue
            if cached_price is not None:
                cached_prices[token_id] = cached_price

    if not missing_tokens:
        return cached_prices

    fetched_prices: dict[str, float] = {}
    try:
        payload = await polymarket_client.get_prices_batch(missing_tokens)
    except Exception:
        payload = {}

    if isinstance(payload, dict):
        for raw_token_id, raw_row in payload.items():
            token_id = _normalize_token_id(raw_token_id)
            if not token_id:
                continue
            row = raw_row if isinstance(raw_row, dict) else {}
            price = _coerce_mid_price(row.get("mid"))
            if price is not None:
                fetched_prices[token_id] = price

    expiry = now_monotonic + max(1.0, float(ttl_seconds))
    async with _price_cache_lock:
        for token_id in missing_tokens:
            _price_cache[token_id] = (fetched_prices.get(token_id), expiry)
        _prune_cache(now_monotonic)

    cached_prices.update(fetched_prices)
    return cached_prices
