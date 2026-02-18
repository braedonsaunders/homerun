from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal
from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
from services.strategy_loader import strategy_loader


async def preload_strategy_runtime(*, session: AsyncSession | None = None) -> dict[str, Any]:
    async def _run(target_session: AsyncSession) -> dict[str, Any]:
        seeded = await ensure_all_strategies_seeded(target_session)
        loaded = await strategy_loader.refresh_all_from_db(session=target_session)
        loaded_keys = list(loaded.get("loaded", []))
        errors = dict(loaded.get("errors", {}))
        return {
            "seeded": int(seeded.get("seeded", 0) or 0),
            "loaded": loaded_keys,
            "errors": errors,
            "loaded_count": len(loaded_keys),
            "error_count": len(errors),
        }

    if session is not None:
        return await _run(session)

    async with AsyncSessionLocal() as local_session:
        return await _run(local_session)
