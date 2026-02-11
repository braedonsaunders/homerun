"""Dedicated crypto market API routes.

Completely independent from the scanner/opportunities pipeline.
Always returns live 15-minute crypto market data from the Polymarket
Gamma series API + Chainlink oracle prices.
"""

import time as _time

from fastapi import APIRouter

from services.crypto_service import get_crypto_service
from services.chainlink_feed import get_chainlink_feed

router = APIRouter()


@router.get("/crypto/markets")
async def get_crypto_markets():
    """Return live crypto 15-minute markets with real-time pricing.

    Always returns one market per configured asset (BTC, ETH, SOL, XRP)
    regardless of whether any arbitrage opportunity exists.
    Includes Chainlink oracle prices when available.
    """
    svc = get_crypto_service()
    feed = get_chainlink_feed()
    markets = svc.get_live_markets()

    # Update price-to-beat tracking
    svc._update_price_to_beat(markets)

    result = []
    for m in markets:
        d = m.to_dict()
        # Attach Chainlink oracle price for this asset
        oracle = feed.get_price(m.asset)
        if oracle:
            d["oracle_price"] = oracle.price
            d["oracle_updated_at_ms"] = oracle.updated_at_ms
            d["oracle_age_seconds"] = round(
                (_time.time() * 1000 - oracle.updated_at_ms) / 1000, 1
            ) if oracle.updated_at_ms else None
        else:
            d["oracle_price"] = None
            d["oracle_updated_at_ms"] = None
            d["oracle_age_seconds"] = None

        # Attach price to beat
        d["price_to_beat"] = svc._price_to_beat.get(m.slug)

        result.append(d)

    return result


@router.get("/crypto/oracle-prices")
async def get_oracle_prices():
    """Return current Chainlink oracle prices for all assets."""
    feed = get_chainlink_feed()
    prices = feed.get_all_prices()
    return {asset: p.to_dict() for asset, p in prices.items()}
