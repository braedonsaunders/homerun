"""Kelly criterion and fee-aware edge calculation for prediction markets."""
from __future__ import annotations
import math


def kelly_fraction(p_estimated: float, p_market: float, fraction: float = 0.25) -> float:
    """Quarter-Kelly fraction for a binary prediction market bet.

    Args:
        p_estimated: Your estimated true probability of the event
        p_market: Market-implied probability (contract price)
        fraction: Kelly fraction (0.25 = quarter-Kelly, default)

    Returns:
        Fraction of bankroll to wager (0 if no edge)
    """
    if p_estimated <= p_market or p_market <= 0 or p_market >= 1:
        return 0.0
    f_star = (p_estimated - p_market) / (1.0 - p_market)
    return max(0.0, min(1.0, fraction * f_star))


def kelly_size(p_estimated: float, p_market: float, bankroll: float,
               fraction: float = 0.25, min_size: float = 1.0, max_size: float = 500.0) -> float:
    """Position size in USD using fractional Kelly criterion.

    Args:
        p_estimated: Your estimated true probability
        p_market: Market price (implied probability)
        bankroll: Total available capital
        fraction: Kelly fraction (default 0.25 = quarter-Kelly)
        min_size: Minimum position size
        max_size: Maximum position size

    Returns:
        Position size in USD
    """
    f = kelly_fraction(p_estimated, p_market, fraction)
    size = bankroll * f
    if size < min_size:
        return 0.0  # Below minimum, don't trade
    return min(size, max_size)


def polymarket_taker_fee(p: float, fee_rate: float = 0.0625) -> float:
    """Polymarket taker fee for a contract at price p.

    Fee = p * (1-p) * fee_rate
    Maximum at p=0.50 (~1.56%), minimum at extremes.
    Makers pay zero fees.

    Args:
        p: Contract price (0 to 1)
        fee_rate: Fee rate (default 0.0625 = 6.25 bps quadratic)

    Returns:
        Fee per contract in USD
    """
    return p * (1.0 - p) * fee_rate


def kalshi_taker_fee(p: float, contracts: int = 1, fee_rate: float = 0.07) -> float:
    """Kalshi taker fee.

    Fee = ceil(fee_rate * contracts * price * (1-price))
    Range: ~0.6% at tails to ~1.75% at p=0.50.
    """
    return math.ceil(fee_rate * contracts * p * (1.0 - p) * 100) / 100


def fee_adjusted_edge(p_estimated: float, p_market: float,
                       platform: str = "polymarket", side: str = "buy") -> float:
    """Calculate edge after platform fees.

    Args:
        p_estimated: Your estimated true probability
        p_market: Market price
        platform: "polymarket" or "kalshi"
        side: "buy" (taker) or "sell" (maker, 0 fee on polymarket)

    Returns:
        Net edge after fees (as fraction, not percent)
    """
    gross_edge = p_estimated - p_market

    if platform == "polymarket":
        if side == "buy":
            fee = polymarket_taker_fee(p_market)
        else:
            fee = 0.0  # Makers pay zero
    elif platform == "kalshi":
        fee = kalshi_taker_fee(p_market)
    else:
        fee = 0.0

    return gross_edge - fee


def breakeven_edge(p_market: float, platform: str = "polymarket") -> float:
    """Minimum edge needed to break even after fees.

    Returns edge as fraction (multiply by 100 for percent).
    """
    if platform == "polymarket":
        fee = polymarket_taker_fee(p_market)
    elif platform == "kalshi":
        fee = kalshi_taker_fee(p_market)
    else:
        fee = 0.0
    return fee
