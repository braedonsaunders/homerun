from __future__ import annotations

from typing import Any

from utils.converters import clamp


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_policy(value: Any) -> str:
    text = str(value or "linear").strip().lower()
    if text in {"fixed", "linear", "adaptive", "kelly"}:
        return text
    return "linear"


def kelly_fraction(
    *,
    win_probability: float,
    entry_price: float,
    payout_price: float = 1.0,
) -> float:
    """Generalized Kelly fraction with bounded, defensive assumptions.

    - `entry_price`: stake paid per unit outcome share.
    - `payout_price`: payout if correct (defaults to binary $1 settlement).
    - Loss assumes full stake loss on incorrect outcome.
    """

    p = clamp(_to_float(win_probability, 0.0), 0.0, 1.0)
    q = 1.0 - p

    entry = clamp(_to_float(entry_price, 0.0), 0.0001, 0.9999)
    payout = max(entry + 0.0001, _to_float(payout_price, 1.0))

    b = (payout - entry) / entry
    if b <= 0.0:
        return 0.0

    raw = (b * p - q) / b
    return clamp(raw, 0.0, 1.0)


def compute_position_size(
    *,
    base_size_usd: float,
    max_size_usd: float,
    edge_percent: float,
    confidence: float,
    sizing_policy: str = "linear",
    probability: float | None = None,
    entry_price: float | None = None,
    payout_price: float = 1.0,
    kelly_fractional_scale: float = 0.5,
    liquidity_usd: float | None = None,
    liquidity_cap_fraction: float = 0.08,
    min_size_usd: float = 1.0,
) -> dict[str, float | str | None]:
    """Unified position sizing policy for trader strategies.

    Kelly is intentionally implemented as a sizing policy (not a strategy).
    """

    base = max(min_size_usd, _to_float(base_size_usd, 10.0))
    max_size = max(base, _to_float(max_size_usd, base))

    edge = max(0.0, _to_float(edge_percent, 0.0))
    conf = clamp(_to_float(confidence, 0.0), 0.0, 1.0)
    policy = _normalize_policy(sizing_policy)

    edge_mult = 1.0 + (edge / 100.0)
    conf_mult = 0.70 + conf

    raw_size = base
    kelly_raw: float | None = None

    if policy == "fixed":
        raw_size = base
    elif policy == "adaptive":
        adaptive_edge_mult = 0.75 + clamp(edge / 25.0, 0.0, 1.1)
        raw_size = base * adaptive_edge_mult * conf_mult
    elif policy == "kelly":
        prob = None if probability is None else clamp(_to_float(probability, 0.0), 0.0, 1.0)
        entry = None if entry_price is None else clamp(_to_float(entry_price, 0.0), 0.0001, 0.9999)
        if prob is not None and entry is not None:
            kelly_raw = kelly_fraction(
                win_probability=prob,
                entry_price=entry,
                payout_price=payout_price,
            )
        scaled_kelly = clamp(_to_float(kelly_fractional_scale, 0.5), 0.05, 1.0) * (kelly_raw or 0.0)
        # Map Kelly fraction into a practical notional multiplier around base.
        kelly_mult = 0.55 + (scaled_kelly * 1.75)
        raw_size = base * edge_mult * conf_mult * kelly_mult
    else:  # linear
        raw_size = base * edge_mult * conf_mult

    liquidity_cap_usd: float | None = None
    liquidity = None if liquidity_usd is None else max(0.0, _to_float(liquidity_usd, 0.0))
    if liquidity is not None and liquidity > 0.0:
        liquidity_cap = max(min_size_usd, liquidity * clamp(liquidity_cap_fraction, 0.01, 1.0))
        liquidity_cap_usd = liquidity_cap
        raw_size = min(raw_size, liquidity_cap)

    size_usd = clamp(raw_size, min_size_usd, max_size)

    return {
        "size_usd": float(size_usd),
        "raw_size_usd": float(raw_size),
        "policy": policy,
        "kelly_fraction": None if kelly_raw is None else float(kelly_raw),
        "liquidity_cap_usd": None if liquidity_cap_usd is None else float(liquidity_cap_usd),
    }
