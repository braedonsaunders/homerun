from __future__ import annotations

from fastapi import APIRouter

from models.database import AsyncSessionLocal
from services.worker_state import read_worker_snapshot

router = APIRouter()


async def _read_crypto_stats() -> dict:
    async with AsyncSessionLocal() as session:
        snapshot = await read_worker_snapshot(session, "crypto")
    stats = snapshot.get("stats") if isinstance(snapshot.get("stats"), dict) else {}
    return dict(stats)


@router.get("/crypto/markets")
async def get_crypto_markets(viewer_active: bool = False) -> list[dict]:
    del viewer_active
    stats = await _read_crypto_stats()
    return list(stats.get("markets") or [])


@router.get("/crypto/oracle-prices")
async def get_oracle_prices() -> dict[str, dict]:
    stats = await _read_crypto_stats()
    return dict(stats.get("oracle_prices") or {})
