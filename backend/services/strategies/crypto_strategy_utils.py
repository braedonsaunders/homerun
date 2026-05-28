from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from models import Market
from utils.converters import clamp, safe_float
from utils.kelly import polymarket_taker_fee_pct
from utils.utcnow import utcnow  # replay-clock-aware "now" (honors backtest sim time)


def parse_datetime_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts <= 0:
            return None
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except Exception:
        numeric = None
    if numeric is not None and numeric > 0:
        return parse_datetime_utc(numeric)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_binary_market_from_row(row: dict[str, Any]) -> Market | None:
    return build_binary_crypto_market(row)


def build_binary_crypto_market(row: dict[str, Any]) -> Market | None:
    market_id = str(row.get("condition_id") or row.get("id") or row.get("slug") or "").strip()
    if not market_id:
        return None

    up_price = safe_float(row.get("up_price"), None)
    down_price = safe_float(row.get("down_price"), None)
    if up_price is None or down_price is None:
        return None

    end_date = parse_datetime_utc(row.get("end_time"))
    token_ids = [
        str(token).strip()
        for token in list(row.get("clob_token_ids") or [])
        if str(token).strip() and len(str(token).strip()) > 20
    ]

    return Market(
        id=market_id,
        condition_id=market_id,
        question=str(row.get("question") or row.get("slug") or market_id),
        slug=str(row.get("slug") or market_id),
        outcome_prices=[float(up_price), float(down_price)],
        liquidity=max(0.0, float(safe_float(row.get("liquidity"), 0.0) or 0.0)),
        end_date=end_date,
        platform="polymarket",
        clob_token_ids=token_ids,
    )


def seconds_left_from_row(row: dict[str, Any], *, fallback_seconds: float | None = None) -> float | None:
    seconds_left = safe_float(row.get("seconds_left"), None)
    if seconds_left is not None and seconds_left >= 0.0:
        return float(seconds_left)
    end_time = parse_datetime_utc(row.get("end_time"))
    if end_time is None:
        return fallback_seconds
    return max(0.0, (end_time - utcnow()).total_seconds())


def spread_pct_from_row(row: dict[str, Any]) -> float:
    spread = safe_float(row.get("spread"), None)
    if spread is None or spread < 0.0:
        best_bid = safe_float(row.get("best_bid"), None)
        best_ask = safe_float(row.get("best_ask"), None)
        if best_bid is not None and best_ask is not None and best_ask >= best_bid:
            spread = best_ask - best_bid
    if spread is None:
        spread = 0.0
    return clamp(float(spread), 0.0, 1.0)


def taker_fee_pct(entry_price: float) -> float:
    price = clamp(float(entry_price), 0.0001, 0.9999)
    return 0.25 * ((price * (1.0 - price)) ** 2)


def bounded_sigmoid(z: float) -> float:
    bounded = clamp(float(z), -60.0, 60.0)
    return 1.0 / (1.0 + math.exp(-bounded))


def normalize_ratio(value: Any) -> float | None:
    parsed = safe_float(value, None)
    if parsed is None:
        return None
    ratio = float(parsed)
    if ratio > 1.0 and ratio <= 100.0:
        ratio /= 100.0
    return max(0.0, min(1.0, ratio))


def normalize_signed_ratio(value: Any) -> float | None:
    parsed = safe_float(value, None)
    if parsed is None:
        return None
    ratio = float(parsed)
    if abs(ratio) > 1.0 and abs(ratio) <= 100.0:
        ratio /= 100.0
    return max(-1.0, min(1.0, ratio))


def history_cancel_peak(history_tail: Any) -> float | None:
    if not isinstance(history_tail, list):
        return None
    peak: float | None = None
    for row in history_tail:
        if not isinstance(row, dict):
            continue
        candidate = normalize_ratio(
            row.get("cancel_rate_30s")
            if row.get("cancel_rate_30s") is not None
            else row.get("maker_cancel_rate_30s")
        )
        if candidate is None:
            continue
        peak = candidate if peak is None else max(peak, candidate)
    return peak


def market_ml_probability_yes(row: dict[str, Any]) -> float | None:
    if not isinstance(row, dict):
        return None

    machine_learning = row.get("machine_learning")
    if not isinstance(machine_learning, dict):
        return None

    prediction = machine_learning.get("prediction")
    if isinstance(prediction, dict):
        probability_yes = safe_float(prediction.get("probability_yes"), None)
        if probability_yes is not None:
            return clamp(float(probability_yes), 0.03, 0.97)

    probability_yes = safe_float(machine_learning.get("probability_yes"), None)
    if probability_yes is not None:
        return clamp(float(probability_yes), 0.03, 0.97)
    return None


# Source preference order when ages tie or no per-source data is available.
# binance_direct first (sub-second receive-time stamp), then chainlink
# (canonical resolution price), then RTDS-relayed binance, then anything
# else. Mirrors the ranking already implemented in
# the canonical extract_oracle_status flow used by every crypto strategy.
_ORACLE_SOURCE_RANK: dict[str, int] = {
    "binance_direct": 0,
    "chainlink": 1,
    "binance": 2,
}


def _coerce_age_ms(point: dict[str, Any], now_ms: float) -> float | None:
    age_ms = safe_float(point.get("age_ms"), None)
    if age_ms is not None and age_ms >= 0.0:
        return float(age_ms)
    age_seconds = safe_float(point.get("age_seconds"), None)
    if age_seconds is not None and age_seconds >= 0.0:
        return float(age_seconds) * 1000.0
    updated_at_ms = safe_float(point.get("updated_at_ms"), None)
    if updated_at_ms is not None and updated_at_ms > 0.0:
        return max(0.0, now_ms - float(updated_at_ms))
    return None


def pick_oracle_source(
    market: dict[str, Any],
    *,
    prefer: str | None = None,
    max_age_ms: float | None = None,
    now_ms: float | None = None,
) -> dict[str, Any] | None:
    """Pick a fresh oracle price source from a crypto market dict.

    Strategies that want a sub-second price (e.g. ``binance_direct`` from
    the Binance WS feed) should call this instead of reading the
    Chainlink-dominated ``oracle_price`` field, which moves on Chainlink's
    on-chain heartbeat and can lag several seconds.

    Resolution order:

    1. If ``prefer`` is supplied and that source has a fresh price (price > 0
       and ``age_ms`` ≤ ``max_age_ms`` when set), return it.
    2. Otherwise pick the freshest source by ``age_ms``, breaking ties with
       ``_ORACLE_SOURCE_RANK`` (binance_direct → chainlink → binance → other).
    3. If ``max_age_ms`` is set, only return a source whose age is within
       that bound. Returns ``None`` when no source qualifies.

    Returns a dict shaped ``{"source", "price", "updated_at_ms", "age_ms"}``
    or ``None`` when no usable source exists.
    """
    by_source = market.get("oracle_prices_by_source")
    if not isinstance(by_source, dict) or not by_source:
        return None

    resolved_now = float(now_ms) if now_ms is not None else _wall_now_ms()

    candidates: list[tuple[str, dict[str, Any], float]] = []
    for raw_source, raw_point in by_source.items():
        if not isinstance(raw_point, dict):
            continue
        price = safe_float(raw_point.get("price"), None)
        if price is None or price <= 0.0:
            continue
        source_key = str(raw_source).strip().lower() or str(raw_point.get("source") or "").strip().lower()
        if not source_key:
            continue
        age_ms = _coerce_age_ms(raw_point, resolved_now)
        if age_ms is None:
            continue
        if max_age_ms is not None and age_ms > float(max_age_ms):
            continue
        candidates.append((source_key, raw_point, age_ms))

    if not candidates:
        return None

    if prefer:
        preferred_key = str(prefer).strip().lower()
        for source_key, point, age_ms in candidates:
            if source_key == preferred_key:
                return _shape_oracle_pick(source_key, point, age_ms)

    candidates.sort(key=lambda row: (row[2], _ORACLE_SOURCE_RANK.get(row[0], 99)))
    source_key, point, age_ms = candidates[0]
    return _shape_oracle_pick(source_key, point, age_ms)


def _shape_oracle_pick(source: str, point: dict[str, Any], age_ms: float) -> dict[str, Any]:
    return {
        "source": source,
        "price": float(safe_float(point.get("price"), 0.0) or 0.0),
        "updated_at_ms": safe_float(point.get("updated_at_ms"), None),
        "age_ms": float(age_ms),
    }


def _wall_now_ms() -> float:
    import time as _time
    return _time.time() * 1000.0


# ---------------------------------------------------------------------------
# Timeframe + fee helpers (shared between crypto strategies)
# ---------------------------------------------------------------------------

# Seconds-per-window for Polymarket crypto binary timeframes. Falls back to
# 15m / 900s for unknown values so a missing/garbled field doesn't crash
# downstream math; callers should still treat unknown timeframes as a soft
# error and refuse to enter (see ``min_seconds_left_for_entry``).
_TIMEFRAME_SECONDS: dict[str, int] = {
    "5m": 300,
    "5min": 300,
    "15m": 900,
    "15min": 900,
    "1h": 3600,
    "1hr": 3600,
    "60m": 3600,
    "60min": 3600,
    "4h": 14400,
    "4hr": 14400,
    "240m": 14400,
    "240min": 14400,
}


def normalize_timeframe(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"5m", "5min", "5"}:
        return "5m"
    if raw in {"15m", "15min", "15"}:
        return "15m"
    if raw in {"1h", "1hr", "60m", "60min"}:
        return "1h"
    if raw in {"4h", "4hr", "240m", "240min"}:
        return "4h"
    return raw


def timeframe_seconds(value: Any, default: int = 900) -> int:
    """Return the window length in seconds for a Polymarket crypto timeframe.

    Canonical timeframe-seconds resolution shared across all crypto strategies
    agree on what "5m"/"15m"/"1h"/"4h" mean. Returns ``default`` (15 min) for
    unknown values; callers should also gate on a min-seconds-left check so
    an unrecognised timeframe doesn't silently allow trades through with the
    wrong assumption.
    """
    normalized = normalize_timeframe(value)
    return _TIMEFRAME_SECONDS.get(normalized, int(default))


def fee_aware_min_edge_pct(price: float, multiplier: float = 2.0) -> float:
    """Minimum edge (in %, NOT fraction) the strategy should require to clear
    fees by ``multiplier``× at the given entry price. Returns a percentage so
    callers can compare directly against existing edge fields that are also
    expressed in percent."""
    return polymarket_taker_fee_pct(price) * 100.0 * float(multiplier)


# ---------------------------------------------------------------------------
# Resolution-boundary safety
# ---------------------------------------------------------------------------

# Per-timeframe minimum seconds-left required to open a new position. Mirrors
# the values enforced via
# ``min_seconds_left_for_entry_*`` config knobs. Goal: don't enter so close
# to resolution that fills race the Chainlink heartbeat.
_DEFAULT_MIN_SECONDS_LEFT_FOR_ENTRY: dict[str, float] = {
    "5m": 35.0,
    "15m": 60.0,
    "1h": 180.0,
    "4h": 600.0,
}


def default_min_seconds_left_for_entry(timeframe: Any) -> float:
    return _DEFAULT_MIN_SECONDS_LEFT_FOR_ENTRY.get(normalize_timeframe(timeframe), 60.0)


# Per-timeframe maximum age for the cached worker market-data snapshot. If
# the row is older than this, the maker/taker decision is built on phantom
# liquidity; refuse to trade and let the next refresh catch up.
_DEFAULT_MAX_MARKET_DATA_AGE_MS: dict[str, float] = {
    "5m": 2500.0,
    "15m": 4000.0,
    "1h": 8000.0,
    "4h": 15000.0,
}


def default_max_market_data_age_ms(timeframe: Any) -> float:
    return _DEFAULT_MAX_MARKET_DATA_AGE_MS.get(normalize_timeframe(timeframe), 4000.0)


# Per-timeframe maximum oracle age. Chainlink heartbeats can run several
# seconds even on liquid feeds; binance_direct is sub-second. ``pick_oracle_source``
# returns whichever source is freshest, so this cap rejects only when *every*
# source is stale.
_DEFAULT_MAX_ORACLE_AGE_MS: dict[str, float] = {
    "5m": 5000.0,
    "15m": 7500.0,
    "1h": 15000.0,
    "4h": 30000.0,
}


def default_max_oracle_age_ms(timeframe: Any) -> float:
    return _DEFAULT_MAX_ORACLE_AGE_MS.get(normalize_timeframe(timeframe), 7500.0)
