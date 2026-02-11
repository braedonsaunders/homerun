from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from .signal_engine import WeatherSignal


def _bounded(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_weather_intent(
    signal: WeatherSignal,
    market_id: str,
    market_question: str,
    settings: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic weather intent payload suitable for DB upsert."""
    now = datetime.now(timezone.utc)
    bucket = now.strftime("%Y%m%d%H")
    raw = f"{market_id}|{signal.direction}|{bucket}"
    intent_id = hashlib.sha256(raw.encode()).hexdigest()[:24]

    base_size = float(settings.get("default_size_usd", 10.0))
    max_size = float(settings.get("max_size_usd", 50.0))

    edge_scale = _bounded(signal.edge_percent / 20.0, 0.0, 1.0)
    conf_scale = _bounded(signal.confidence, 0.0, 1.0)
    size = base_size + (max_size - base_size) * ((edge_scale * 0.5) + (conf_scale * 0.5))

    return {
        "id": intent_id,
        "market_id": market_id,
        "market_question": market_question,
        "direction": signal.direction,
        "entry_price": signal.market_price,
        "take_profit_price": float(settings.get("take_profit_price", 0.85)),
        "stop_loss_pct": float(settings.get("stop_loss_pct", 50.0)),
        "model_probability": signal.model_probability,
        "edge_percent": signal.edge_percent,
        "confidence": signal.confidence,
        "model_agreement": signal.model_agreement,
        "suggested_size_usd": round(size, 2),
        "metadata_json": metadata,
        "status": "pending",
        "created_at": now,
    }
