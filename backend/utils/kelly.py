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


def kelly_size(
    p_estimated: float,
    p_market: float,
    bankroll: float,
    fraction: float = 0.25,
    min_size: float = 1.0,
    max_size: float = 500.0,
) -> float:
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


# Polymarket taker-fee rates by market category (Fee Structure V2, live
# since 2026-03-30).  Source: https://docs.polymarket.com/trading/fees
#
#     fee_per_share = feeRate * p * (1 - p)
#
# The rate is a plain multiplier on the ``p*(1-p)`` curve — NOT a squared
# term.  Makers are never charged; only takers pay.
POLYMARKET_TAKER_FEE_RATES: dict[str, float] = {
    "crypto": 0.07,
    "sports": 0.05,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "general": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "mentions": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
}

# Default when the caller has no category on hand.  Deliberately the
# HIGHEST non-zero rate (crypto): over-estimating fees can only cost a
# skipped trade, whereas under-estimating books a loser as a winner.
# Callers that know their category should pass it — see
# ``polymarket_fee_rate_for_category``.
POLYMARKET_DEFAULT_TAKER_FEE_RATE: float = 0.07


def polymarket_fee_rate_for_category(category: str | None) -> float:
    """Taker fee rate for a Polymarket market category.

    Unknown/empty categories fall back to
    ``POLYMARKET_DEFAULT_TAKER_FEE_RATE`` (the conservative crypto rate).
    Matching is case-insensitive and tolerates the ``"Crypto"`` /
    ``"crypto"`` / ``" CRYPTO "`` forms Gamma returns.
    """
    if not category:
        return POLYMARKET_DEFAULT_TAKER_FEE_RATE
    key = str(category).strip().lower()
    return POLYMARKET_TAKER_FEE_RATES.get(key, POLYMARKET_DEFAULT_TAKER_FEE_RATE)


def polymarket_taker_fee(
    p: float,
    fee_rate: float | None = None,
    *,
    category: str | None = None,
) -> float:
    """Polymarket taker fee for one contract at price ``p`` (USD per share).

    Per Polymarket's published schedule:

        fee_per_share = feeRate * p * (1 - p)

    ``feeRate`` is category-dependent (crypto 0.07, sports/economics/
    culture/weather/other 0.05, finance/politics/mentions/tech 0.04,
    geopolitics 0.0).  The fee peaks at ``p=0.50`` — for crypto that is
    ``0.07 * 0.25 = $0.0175`` per share, i.e. **3.5% of notional**, and it
    decays linearly-in-``p(1-p)`` toward the tails.  Makers pay zero.

    Args:
        p: Contract price in [0, 1].
        fee_rate: Explicit rate override.  Takes precedence over
            ``category``.
        category: Polymarket market category used to look up the rate when
            ``fee_rate`` is not given.

    Returns:
        Fee per share in USD.
    """
    if fee_rate is None:
        rate = polymarket_fee_rate_for_category(category)
    else:
        rate = float(fee_rate)
    p_clamped = max(0.0, min(1.0, float(p or 0.0)))
    return rate * p_clamped * (1.0 - p_clamped)


def polymarket_taker_fee_pct(
    p: float,
    fee_rate: float | None = None,
    *,
    category: str | None = None,
) -> float:
    """Polymarket taker fee as a fraction of contract price.

    At the default (crypto) rate this tops out at 0.035 (3.5%) when
    ``p=0.50`` and falls toward ``feeRate`` as ``p`` approaches 1.0.
    """
    p_value = float(p or 0.0)
    if p_value <= 0.0:
        return 0.0
    return polymarket_taker_fee(p_value, fee_rate, category=category) / p_value


def kalshi_taker_fee(p: float, contracts: int = 1, fee_rate: float = 0.07) -> float:
    """Kalshi taker fee.

    Fee = ceil(fee_rate * contracts * price * (1-price))
    Range: ~0.6% at tails to ~1.75% at p=0.50.
    """
    return math.ceil(fee_rate * contracts * p * (1.0 - p) * 100) / 100


def fee_adjusted_edge(p_estimated: float, p_market: float, platform: str = "polymarket", side: str = "buy") -> float:
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
