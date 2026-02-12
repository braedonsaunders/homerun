"""API routes for trader datasource adapters."""

from __future__ import annotations

from fastapi import APIRouter

from services.trader_orchestrator.sources.registry import list_source_adapters

router = APIRouter(prefix="/trader-sources", tags=["Trader Sources"])


@router.get("")
async def get_trader_sources():
    return {
        "sources": [
            {
                "key": adapter.key,
                "label": adapter.label,
                "description": adapter.description,
                "domains": adapter.domains,
                "signal_types": adapter.signal_types,
            }
            for adapter in list_source_adapters()
        ]
    }
