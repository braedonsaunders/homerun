"""Re-exports of services.strategies.crypto_strategy_utils plus oracle-status helpers.

All public symbols from the strategies module are available here so that
importing code never needs to distinguish between the two locations.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

# Re-export everything from the canonical location so importers of
# services.strategy_helpers.crypto_strategy_utils get the same symbols.
from services.strategies.crypto_strategy_utils import (  # noqa: F401
    build_binary_crypto_market,
    build_binary_market_from_row,
    bounded_sigmoid,
    default_max_market_data_age_ms,
    default_max_oracle_age_ms,
    default_min_seconds_left_for_entry,
    fee_aware_min_edge_pct,
    history_cancel_peak,
    market_ml_probability_yes,
    normalize_ratio,
    normalize_signed_ratio,
    normalize_timeframe,
    parse_datetime_utc,
    pick_oracle_source,
    seconds_left_from_row,
    spread_pct_from_row,
    taker_fee_pct,
    timeframe_seconds,
)
from utils.kelly import polymarket_taker_fee_pct  # noqa: F401
from utils.converters import safe_float as _safe_float, to_float


# ---------------------------------------------------------------------------
# Shared crypto signal-edge + JSON helpers (consolidated from the btc_eth_*
# strategies so they live in one place, reachable via StrategySDK.crypto).
# ---------------------------------------------------------------------------


def to_iso_utc(value: Optional[datetime]) -> Optional[str]:
    """UTC ISO-8601 (Z-suffixed) string for a datetime, or None."""
    if value is None:
        return None
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def get_component_edge(payload: dict[str, Any], direction: str, mode: str) -> float:
    """Per-mode (directional / maker_quote / convergence) component edge for a
    crypto signal, read from the strategy payload's edge breakdown."""
    component_edges = payload.get("component_edges")
    if isinstance(component_edges, dict):
        side_edges = component_edges.get(direction)
        if isinstance(side_edges, dict):
            return max(0.0, to_float(side_edges.get(mode), 0.0))
        return max(0.0, to_float(component_edges.get(mode), 0.0))

    mode_edges = payload.get("mode_edges")
    if isinstance(mode_edges, dict):
        side_mode_edges = mode_edges.get(direction)
        if isinstance(side_mode_edges, dict):
            return max(0.0, to_float(side_mode_edges.get(mode), 0.0))
        return max(0.0, to_float(mode_edges.get(mode), 0.0))

    per_mode_key = {
        "directional": "directional_edge",
        "maker_quote": "maker_quote_edge",
        "convergence": "convergence_edge",
    }.get(mode)
    if per_mode_key is None:
        return 0.0
    return max(0.0, to_float(payload.get(per_mode_key), 0.0))


def get_net_edge(payload: dict[str, Any], direction: str, fallback: float) -> float:
    """Net (post-fee) edge for a direction from the payload, else fallback."""
    net_edges = payload.get("net_edges")
    if not isinstance(net_edges, dict):
        return fallback
    return to_float(net_edges.get(direction), fallback)


def json_safe(value: Any) -> Any:
    """Recursively coerce a value into JSON-serialisable form."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return to_iso_utc(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = json_safe(item)
        return out
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


# ---------------------------------------------------------------------------
# first_present — return the first non-None positional argument
# ---------------------------------------------------------------------------

def first_present(*args: Any) -> Any:
    for arg in args:
        if arg is not None:
            return arg
    return None


# ---------------------------------------------------------------------------
# Oracle availability helpers
# ---------------------------------------------------------------------------

def resolve_oracle_availability(
    *,
    price: float | None,
    price_to_beat: float | None,
    age_ms: float | None,
    updated_at_ms: int | None,
) -> dict[str, Any]:
    """Return a dict describing whether an oracle point is usable."""
    has_price = price is not None and float(price) > 0.0
    has_timestamp = updated_at_ms is not None or age_ms is not None
    available = has_price
    fresh_available = has_price and age_ms is not None
    directional_available = available

    return {
        "available": available,
        "has_price": has_price,
        "has_timestamp": has_timestamp,
        "fresh_available": fresh_available,
        "directional_available": directional_available,
        "availability_reasons": [],
        "freshness_state": "fresh" if fresh_available else "unknown",
        "freshness_reasons": [],
        "directional_state": "available" if directional_available else "unavailable",
        "directional_reasons": [],
    }


def extract_oracle_status(
    *,
    live_market: dict[str, Any],
    payload: dict[str, Any],
    now_ms: float,
) -> dict[str, Any]:
    """Extract a normalised oracle-status dict from a live_market + signal payload.

    Merges per-source oracle data from both dicts, picks the best source via
    the same ranking used by ``pick_oracle_source``, and returns a dict
    compatible with the fields consumed by crypto-strategy evaluate() methods.
    """
    # Prefer payload over live_market for primary fields.
    oracle_price = _safe_float(
        first_present(payload.get("oracle_price"), live_market.get("oracle_price")), None
    )
    oracle_source_raw = first_present(
        payload.get("oracle_source"), live_market.get("oracle_source")
    )
    oracle_source = str(oracle_source_raw or "").strip().lower() or None
    oracle_updated_at_ms = _safe_float(
        first_present(payload.get("oracle_updated_at_ms"), live_market.get("oracle_updated_at_ms")), None
    )
    oracle_age_seconds = _safe_float(
        first_present(payload.get("oracle_age_seconds"), live_market.get("oracle_age_seconds")), None
    )
    price_to_beat = _safe_float(
        first_present(payload.get("price_to_beat"), live_market.get("price_to_beat")), None
    )

    # Compute age_ms for the primary source.
    age_ms: float | None = None
    if oracle_age_seconds is not None and oracle_age_seconds >= 0:
        age_ms = float(oracle_age_seconds) * 1000.0
    elif oracle_updated_at_ms is not None and oracle_updated_at_ms > 0:
        age_ms = max(0.0, float(now_ms) - float(oracle_updated_at_ms))

    # Build per-source dict from oracle_prices_by_source in either dict.
    by_source_raw = first_present(
        payload.get("oracle_prices_by_source"), live_market.get("oracle_prices_by_source")
    )
    by_source: dict[str, Any] = {}
    if isinstance(by_source_raw, dict):
        for raw_key, raw_point in by_source_raw.items():
            if not isinstance(raw_point, dict):
                continue
            s = str(raw_key).strip().lower()
            if not s:
                continue
            pt_updated = _safe_float(raw_point.get("updated_at_ms"), None)
            pt_age: float | None = _safe_float(raw_point.get("age_ms"), None)
            if pt_age is None and pt_updated is not None and pt_updated > 0:
                pt_age = max(0.0, float(now_ms) - float(pt_updated))
            by_source[s] = {
                "source": s,
                "price": _safe_float(raw_point.get("price"), 0.0),
                "updated_at_ms": pt_updated,
                "age_ms": pt_age,
            }

    # If we have a primary source but no by_source entry for it, synthesise one.
    if oracle_source and oracle_source not in by_source and oracle_price is not None:
        by_source[oracle_source] = {
            "source": oracle_source,
            "price": oracle_price,
            "updated_at_ms": oracle_updated_at_ms,
            "age_ms": age_ms,
        }

    # When the primary oracle_price is missing but we have by_source data, pick
    # the best available source to fill in the primary fields.
    if (oracle_price is None or oracle_source is None) and by_source:
        synthetic = {
            "oracle_prices_by_source": {
                s: {**p, "age_ms": p.get("age_ms") or 0.0}
                for s, p in by_source.items()
            }
        }
        best = pick_oracle_source(synthetic, now_ms=float(now_ms))
        if best:
            if oracle_price is None:
                oracle_price = _safe_float(best.get("price"), None)
            if oracle_source is None:
                oracle_source = str(best.get("source") or "").strip().lower() or None
            if age_ms is None:
                age_ms = _safe_float(best.get("age_ms"), None)
            if oracle_updated_at_ms is None:
                oracle_updated_at_ms = _safe_float(best.get("updated_at_ms"), None)

    has_price = oracle_price is not None and float(oracle_price) > 0.0
    has_timestamp = oracle_updated_at_ms is not None or age_ms is not None
    available = has_price
    fresh_available = has_price and age_ms is not None
    directional_available = available

    return {
        "price": oracle_price,
        "price_to_beat": price_to_beat,
        "source": oracle_source,
        "by_source": by_source,
        "age_ms": age_ms,
        "updated_at_ms": (int(oracle_updated_at_ms) if oracle_updated_at_ms is not None else None),
        "has_price": has_price,
        "has_timestamp": has_timestamp,
        "available": available,
        "fresh_available": fresh_available,
        "directional_available": directional_available,
        "availability_reasons": [],
        "freshness_state": "fresh" if fresh_available else "unknown",
        "freshness_reasons": [],
        "directional_state": "available" if directional_available else "unavailable",
        "directional_reasons": [],
    }


# ---------------------------------------------------------------------------
# Shared per-market enrichment — runs ONCE in market_runtime, strategies READ
# ---------------------------------------------------------------------------


def compute_regime(seconds_left: float, timeframe_seconds_value: int) -> str:
    """Classify a crypto market window as opening / mid / closing."""
    denom = float(max(1, int(timeframe_seconds_value)))
    ratio = float(seconds_left) / denom
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < 0.0:
        ratio = 0.0
    if ratio > 0.67:
        return "opening"
    if ratio < 0.33:
        return "closing"
    return "mid"


def enrich_crypto_market_row(
    row: dict[str, Any],
    *,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Stamp shared per-market derived fields onto a crypto market row in-place.

    Computes once what the 3 BTC/ETH strategies were each duplicating per
    crypto_update event:

      * ``oracle_status`` — normalised dict from ``extract_oracle_status``
      * ``timeframe_seconds`` — canonical window length in seconds
      * ``regime`` — opening / mid / closing
      * ``market_data_age_ms`` — populated from ``fetched_at`` if absent
      * ``oracle_diff_pct`` / ``oracle_move_pct`` — signed and absolute %
        between oracle price and price_to_beat (0.0 when oracle missing)

    Idempotent. Safe on partial inputs — missing oracle yields a status
    dict with ``available == False`` and zero diffs.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    oracle_status = extract_oracle_status(
        live_market={},
        payload={
            "oracle_price": row.get("oracle_price"),
            "oracle_source": row.get("oracle_source"),
            "oracle_updated_at_ms": row.get("oracle_updated_at_ms"),
            "oracle_age_seconds": row.get("oracle_age_seconds"),
            "price_to_beat": row.get("price_to_beat"),
            "oracle_prices_by_source": row.get("oracle_prices_by_source"),
        },
        now_ms=float(now_ms),
    )
    row["oracle_status"] = oracle_status

    tf_seconds = timeframe_seconds(row.get("timeframe"), default=900)
    row["timeframe_seconds"] = tf_seconds

    seconds_left_raw = row.get("seconds_left")
    if seconds_left_raw is None:
        seconds_left_value = float(tf_seconds)
    else:
        try:
            seconds_left_value = float(seconds_left_raw)
        except (TypeError, ValueError):
            seconds_left_value = float(tf_seconds)
    row["regime"] = compute_regime(seconds_left_value, tf_seconds)

    if row.get("market_data_age_ms") is None:
        fetched_at = row.get("fetched_at") or row.get("snapshot_fetched_at")
        if fetched_at:
            parsed = parse_datetime_utc(str(fetched_at))
            if parsed is not None:
                row["market_data_age_ms"] = max(
                    0.0,
                    (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
                    * 1000.0,
                )

    oracle_price = oracle_status.get("price")
    price_to_beat = oracle_status.get("price_to_beat")
    diff_pct = 0.0
    if (
        oracle_price is not None
        and price_to_beat is not None
        and float(price_to_beat) > 0.0
    ):
        diff_pct = ((float(oracle_price) - float(price_to_beat)) / float(price_to_beat)) * 100.0
    row["oracle_diff_pct"] = diff_pct
    row["oracle_move_pct"] = abs(diff_pct)

    return row
