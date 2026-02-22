"""Shared helpers for strategy evaluate() and should_exit() methods.

These utilities parse signal data, extract nested payloads, and compute
time-based metrics used across all unified strategies.

# Helpers for READING signal data from DB TradeSignal rows.
# For helpers that BUILD/WRITE signal payloads from Opportunity objects,
# see services/signal_bus.py (build_signal_contract_from_opportunity).
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from utils.converters import safe_float


def _to_float(value: Any, default: float = 0.0) -> float:
    result = safe_float(value, default=default)
    return default if result is None else result


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _extract_position_context(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
    crypto_context: dict[str, Any] = {}
    highfreq_context: dict[str, Any] = {}
    for raw_position in positions:
        if not isinstance(raw_position, dict):
            continue
        if isinstance(raw_position.get("_crypto_context"), dict):
            crypto_context.update(raw_position.get("_crypto_context") or {})
        if raw_position.get("_highfreq_metadata"):
            highfreq_context.update({key: value for key, value in raw_position.items() if key != "_highfreq_metadata"})
    return {
        "crypto": crypto_context,
        "highfreq": highfreq_context,
    }


def _infer_crypto_dimensions(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    chunks: list[str] = []
    for key in ("title", "description"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip().lower())
    markets = payload.get("markets") if isinstance(payload.get("markets"), list) else []
    if markets and isinstance(markets[0], dict):
        first_market = markets[0]
        for key in ("slug", "question", "market_slug"):
            value = first_market.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip().lower())
    blob = " | ".join(chunks)
    if not blob:
        return None, None

    asset: str | None = None
    if re.search(r"\b(bitcoin|btc|xbt)\b", blob):
        asset = "BTC"
    elif re.search(r"\b(ethereum|eth)\b", blob):
        asset = "ETH"
    elif re.search(r"\b(sol|solana)\b", blob):
        asset = "SOL"
    elif re.search(r"\b(xrp|ripple)\b", blob):
        asset = "XRP"

    timeframe: str | None = None
    if re.search(r"\b(5m|5min|5 min|5-minute|5 minute)\b", blob):
        timeframe = "5m"
    elif re.search(r"\b(15m|15min|15 min|15-minute|15 minute)\b", blob):
        timeframe = "15m"
    elif re.search(r"\b(1h|1hr|1 hour|60m)\b", blob):
        timeframe = "1h"
    elif re.search(r"\b(4h|4hr|4 hour|240m)\b", blob):
        timeframe = "4h"

    return asset, timeframe


def signal_payload(signal: Any) -> dict[str, Any]:
    """Extract payload_json and merged strategy context from a signal object."""
    payload = _as_dict(getattr(signal, "payload_json", None))
    strategy_context = _as_dict(getattr(signal, "strategy_context_json", None))
    payload_strategy_context = _as_dict(payload.get("strategy_context"))
    position_context = _extract_position_context(payload)

    merged: dict[str, Any] = {}
    merged.update(strategy_context)
    merged.update(payload_strategy_context)
    merged.update(position_context.get("highfreq") or {})
    merged.update(position_context.get("crypto") or {})
    merged.update(payload)

    if payload_strategy_context:
        merged.setdefault("strategy_context", payload_strategy_context)
    elif strategy_context:
        merged.setdefault("strategy_context", strategy_context)
    if strategy_context:
        merged.setdefault("strategy_context_json", strategy_context)

    firehose = merged.get("firehose")
    if isinstance(firehose, dict):
        for key in (
            "strength",
            "conviction_score",
            "signal_type",
            "wallet_count",
            "source_count",
            "model_agreement",
            "traders_channel",
        ):
            if merged.get(key) is None and firehose.get(key) is not None:
                merged[key] = firehose.get(key)
    if merged.get("strength") is None and merged.get("confluence_strength") is not None:
        merged["strength"] = merged.get("confluence_strength")
    if merged.get("conviction_score") is None and merged.get("strength") is not None:
        merged["conviction_score"] = merged.get("strength")

    inferred_asset, inferred_timeframe = _infer_crypto_dimensions(merged)
    if merged.get("asset") in {None, ""} and inferred_asset is not None:
        merged["asset"] = inferred_asset
    if merged.get("timeframe") in {None, ""} and inferred_timeframe is not None:
        merged["timeframe"] = inferred_timeframe

    if merged.get("agreement") is None and merged.get("ensemble_fraction") is not None:
        merged["agreement"] = merged.get("ensemble_fraction")
    if merged.get("source_count") is None and merged.get("ensemble_count") is not None:
        merged["source_count"] = merged.get("ensemble_count")
    if merged.get("model_agreement") is None and merged.get("agreement") is not None:
        merged["model_agreement"] = merged.get("agreement")

    return merged


def signal_strategy_context(signal: Any) -> dict[str, Any]:
    """Return canonical strategy context attached to a signal row."""
    context = getattr(signal, "strategy_context_json", None)
    if isinstance(context, dict):
        return dict(context)
    payload = getattr(signal, "payload_json", None)
    if isinstance(payload, dict):
        payload_context = payload.get("strategy_context")
        if isinstance(payload_context, dict):
            return dict(payload_context)
    return {}


def weather_signal_context(signal: Any) -> dict[str, Any]:
    """Return strict weather strategy context from signal strategy_context."""
    context = signal_strategy_context(signal)
    weather = context.get("weather")
    if isinstance(weather, dict):
        return dict(weather)
    return {}


def parse_iso(value: Any) -> datetime | None:
    """Parse an ISO datetime string to timezone-aware datetime."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def days_to_resolution(payload: dict[str, Any]) -> float | None:
    """Compute days until resolution from payload resolution_date."""
    resolution = parse_iso(payload.get("resolution_date"))
    if resolution is None:
        return None
    now = datetime.now(timezone.utc)
    return (resolution - now).total_seconds() / 86400.0


def selected_probability(
    signal: Any,
    payload: dict[str, Any],
    direction: str,
) -> float | None:
    """Extract the best probability estimate for the selected side."""
    entry = _to_float(getattr(signal, "entry_price", None), -1.0)
    if 0.0 < entry < 1.0:
        return entry

    positions = (
        payload.get("positions_to_take")
        if isinstance(
            payload.get("positions_to_take"),
            list,
        )
        else []
    )
    if positions:
        candidate = _to_float((positions[0] or {}).get("price"), -1.0)
        if 0.0 < candidate < 1.0:
            return candidate

    model_prob = _to_float(payload.get("model_probability"), -1.0)
    if 0.0 <= model_prob <= 1.0:
        return model_prob

    edge = _to_float(getattr(signal, "edge_percent", 0.0), 0.0)
    if 0.0 < entry < 1.0:
        implied = entry + (edge / 100.0)
        if 0.0 <= implied <= 1.0:
            return implied

    return None


def live_move(context: dict[str, Any], key: str) -> float | None:
    """Extract a price move percentage from live market context."""
    summary = context.get("history_summary")
    if not isinstance(summary, dict):
        return None
    move = summary.get(key)
    if not isinstance(move, dict):
        return None
    raw = move.get("percent")
    if raw is None:
        return None
    return _to_float(raw, 0.0)


def hours_to_target(target_time: Any) -> float | None:
    """Compute hours until a target time."""
    text = str(target_time or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)
        return delta.total_seconds() / 3600.0
    except Exception:
        return None
