from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


async def submit_order(
    *,
    mode: str,
    signal: Any,
    size_usd: float,
) -> tuple[str, float | None, str | None, dict[str, Any]]:
    """Submit an order.

    v1 behavior:
    - paper: mark as executed immediately.
    - live: mark as submitted for operator review.
    """

    entry_price = _safe_float(getattr(signal, "entry_price", None), 0.0) or None

    if str(mode) == "live":
        return (
            "submitted",
            entry_price,
            None,
            {
                "mode": "live",
                "submission": "queued",
                "note": "Live execution is operator-gated in orchestrator v1.",
            },
        )

    return (
        "executed",
        entry_price,
        None,
        {
            "mode": "paper",
            "submission": "simulated",
            "filled_notional_usd": float(max(0.0, size_usd)),
        },
    )
