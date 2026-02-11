"""Crypto routes backed by dedicated crypto-worker snapshots."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db_session
from services.worker_state import read_worker_snapshot

router = APIRouter()


@router.get("/crypto/markets")
async def get_crypto_markets(session: AsyncSession = Depends(get_db_session)):
    """Return latest crypto market payload produced by ``crypto-worker``."""
    snapshot = await read_worker_snapshot(session, "crypto")
    stats = snapshot.get("stats") or {}
    markets = stats.get("markets") if isinstance(stats, dict) else []
    return markets if isinstance(markets, list) else []


@router.get("/crypto/oracle-prices")
async def get_oracle_prices(session: AsyncSession = Depends(get_db_session)):
    """Return latest oracle prices derived from the crypto-worker market snapshot."""
    snapshot = await read_worker_snapshot(session, "crypto")
    stats = snapshot.get("stats") or {}
    markets = stats.get("markets") if isinstance(stats, dict) else []

    out: dict[str, dict] = {}
    if isinstance(markets, list):
        for market in markets:
            asset = (market or {}).get("asset")
            if not asset:
                continue
            out[str(asset)] = {
                "price": (market or {}).get("oracle_price"),
                "updated_at_ms": (market or {}).get("oracle_updated_at_ms"),
                "age_seconds": (market or {}).get("oracle_age_seconds"),
            }
    return out
