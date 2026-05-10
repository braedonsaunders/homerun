"""External data provider API.

Routes mounted at ``/api/providers``.  Powers:
  * Data Lab → Providers tab (browse markets, kick off imports, watch
    progress, list imported datasets)
  * Backtest Studio dataset picker (select an imported provider
    dataset to back the next backtest run)
  * Settings → Data Sources → Providers (test the polybacktest
    connection)

The actual import work runs asynchronously on the discovery plane via
``workers/provider_import_worker.py`` — these routes are thin wrappers
over ``services/external_data/provider_import_service.py``.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from models.database import AppSettings, AsyncSessionLocal, ProviderImportJob
from services.external_data.polybacktest_client import (
    PolybacktestAuthError,
    PolybacktestError,
    PolybacktestNotConfiguredError,
    build_client_from_settings,
    supported_coins,
)
from utils.secrets import decrypt_secret, encrypt_secret
from services.external_data.provider_import_service import (
    PROVIDER_POLYBACKTEST,
    CreatePolybacktestJobSpec,
    cancel_job,
    delete_provider_dataset,
    enqueue_polybacktest_import,
    get_provider_dataset,
    list_provider_datasets,
    resolve_dataset_scope,
)

logger = logging.getLogger("routes.providers")
router = APIRouter(prefix="/providers", tags=["Providers"])


# ---------------------------------------------------------------------------
# Provider catalog (static for now — one entry per supported provider)
# ---------------------------------------------------------------------------


@router.get("")
async def list_providers() -> dict[str, Any]:
    """List the external data providers this build supports.

    The UI uses this to render the provider switcher in Data Lab.
    Each entry carries enough metadata for the UI to:
      * decide whether the provider is configured (api key present)
      * render the provider's market browser tabs
      * surface a link to the docs / pricing page
    """
    polybacktest_status = await _polybacktest_status()
    return {
        "providers": [
            {
                "key": PROVIDER_POLYBACKTEST,
                "label": "Polybacktest",
                "description": (
                    "Sub-second Polymarket Up/Down book history plus "
                    "Binance reference prices.  Paid SaaS — buy a Pro "
                    "tier at polybacktest.com to lift the free-tier cap."
                ),
                "homepage": "https://polybacktest.com",
                "docs_url": "https://docs.polybacktest.com",
                "asset_classes": ["prediction"],
                "supported_coins": list(supported_coins()),
                "configured": polybacktest_status["configured"],
                "health": polybacktest_status,
            }
        ]
    }


async def _polybacktest_status() -> dict[str, Any]:
    """Lightweight status — no network call when no key configured."""
    try:
        client = await build_client_from_settings()
    except PolybacktestNotConfiguredError as exc:
        return {"configured": False, "ok": False, "error": str(exc)}
    except Exception as exc:
        return {"configured": False, "ok": False, "error": str(exc)}
    try:
        result = await client.health()
        result["configured"] = True
        return result
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Polybacktest — market browser
# ---------------------------------------------------------------------------


@router.get("/polybacktest/markets")
async def list_polybacktest_markets(
    coin: str = Query(default="btc"),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = Query(default=None),
    market_type: Optional[str] = Query(
        default=None,
        description="Filter to a specific Up/Down horizon: 5m | 15m | 1h | 4h | 24h",
    ),
    resolved: Optional[bool] = Query(
        default=None,
        description="True → only resolved markets; False → only open markets; null → both",
    ),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """Search / list Polybacktest markets.  Offset-paginated.

    Each market carries a synthesized human ``title`` so the UI never
    has to render an opaque ID.  Format:
    ``BTC Up/Down · 5m · 2026-05-04 12:30 UTC (open $80,149.13)``.
    """
    try:
        client = await build_client_from_settings()
    except PolybacktestNotConfiguredError as exc:
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    try:
        markets, total = await client.list_markets(
            coin=coin,
            offset=offset,
            limit=limit,
            search=search,
            market_type=market_type,
            resolved=resolved,
        )
    except PolybacktestAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except PolybacktestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        await client.close()
    return {
        "coin": (coin or "").lower(),
        "total": int(total),
        "limit": int(limit),
        "offset": int(offset),
        "markets": [
            {
                "market_id": m.market_id,
                "slug": m.slug,
                "title": m.title,
                "market_type": m.market_type,
                "start_time": m.start_time.isoformat() if m.start_time else None,
                "end_time": m.end_time.isoformat() if m.end_time else None,
                "winner": m.winner,
                "final_volume": m.final_volume,
                "final_liquidity": m.final_liquidity,
                "coin_price_start": m.coin_price_start,
                "coin_price_end": m.coin_price_end,
            }
            for m in markets
        ],
    }


# ---------------------------------------------------------------------------
# Imports — enqueue + track
# ---------------------------------------------------------------------------


class PolybacktestImportRequest(BaseModel):
    coin: str = Field(default="btc")
    market_ids: list[str] = Field(min_length=1)
    start: datetime
    end: datetime


@router.post("/polybacktest/import")
async def import_polybacktest(req: PolybacktestImportRequest) -> dict[str, Any]:
    """Enqueue a polybacktest import job.

    The discovery-plane worker picks it up within a few seconds.  The
    response carries the job_id + initial status so the UI can start
    polling ``GET /providers/import/{job_id}`` immediately.

    Always pulls the full L2 order book (15 levels per side) and both
    UP / DOWN outcomes per snapshot.  Polybacktest does not expose a
    trades endpoint for prediction markets so trades are not imported.
    """
    try:
        spec = CreatePolybacktestJobSpec(
            coin=req.coin,
            market_ids=list(req.market_ids),
            start_ms=int(req.start.timestamp() * 1000),
            end_ms=int(req.end.timestamp() * 1000),
        )
        job = await enqueue_polybacktest_import(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_job(job)


@router.get("/import")
async def list_import_jobs(
    provider: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        stmt = select(ProviderImportJob)
        if provider:
            stmt = stmt.where(ProviderImportJob.provider == provider)
        if status:
            stmt = stmt.where(ProviderImportJob.status == status)
        stmt = stmt.order_by(ProviderImportJob.created_at.desc()).limit(int(limit))
        rows = list((await session.execute(stmt)).scalars().all())
    return {"jobs": [_serialize_job(r) for r in rows]}


@router.get("/import/{job_id}")
async def get_import_job(job_id: str) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderImportJob).where(ProviderImportJob.id == job_id)
            )
        ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Import job '{job_id}' not found")
    return _serialize_job(row)


@router.post("/import/{job_id}/cancel")
async def cancel_import_job(job_id: str) -> dict[str, Any]:
    ok = await cancel_job(job_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or no longer cancellable",
        )
    return {"cancelled": True, "id": job_id}


# ---------------------------------------------------------------------------
# Imported datasets catalog
# ---------------------------------------------------------------------------


@router.get("/datasets")
async def list_datasets(
    provider: Optional[str] = None,
    coin: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    rows = await list_provider_datasets(provider=provider, coin=coin, limit=limit)
    return {"datasets": [_serialize_dataset(r) for r in rows]}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    row = await get_provider_dataset(dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return _serialize_dataset(row, include_payload=True)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str) -> dict[str, Any]:
    ok = await delete_provider_dataset(dataset_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return {"deleted": True, "id": dataset_id}


@router.post("/datasets/scope")
async def resolve_scope(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve a list of dataset IDs into the (token_ids, start, end) tuple
    the unified backtester expects.  Surfaced as a separate endpoint so
    the BacktestStudio UI can preview the resolved scope before kicking
    off the run.
    """
    ids = payload.get("dataset_ids") or []
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="dataset_ids must be a list")
    scope = await resolve_dataset_scope([str(x) for x in ids])
    if scope is None:
        raise HTTPException(status_code=404, detail="No matching datasets")
    return {
        "dataset_ids": scope["dataset_ids"],
        "labels": scope["labels"],
        "token_ids": scope["token_ids"],
        "start": scope["start"].isoformat() if scope["start"] else None,
        "end": scope["end"].isoformat() if scope["end"] else None,
    }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_job(row: ProviderImportJob) -> dict[str, Any]:
    return {
        "id": row.id,
        "provider": row.provider,
        "status": row.status,
        "progress": float(row.progress or 0.0),
        "message": row.message,
        "payload": row.payload_json,
        "result": row.result_json,
        "error": row.error,
        "snapshots_fetched": int(row.snapshots_fetched or 0),
        "snapshots_inserted": int(row.snapshots_inserted or 0),
        "trades_fetched": int(row.trades_fetched or 0),
        "api_calls": int(row.api_calls or 0),
        "bytes_downloaded": int(row.bytes_downloaded or 0),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


# ---------------------------------------------------------------------------
# Settings (polybacktest API key + reverse-engineer defaults).
# Dedicated lightweight endpoint so the UI doesn't need to round-trip
# the full settings bundle to flip a single key.
# ---------------------------------------------------------------------------


_API_KEY_PRESENT_MASK = "********"


class ProviderSettings(BaseModel):
    polybacktest_api_key_set: bool = False
    polybacktest_base_url: Optional[str] = None
    # Reverse-engineer model lives in app_settings.llm_model_assignments
    # under the 'strategy_reverse_engineer' key — managed in AI → Models.
    reverse_engineer_max_iterations: Optional[int] = None
    reverse_engineer_target_score: Optional[float] = None
    reverse_engineer_max_cost_usd: Optional[float] = None
    reverse_engineer_max_wallet_trades: Optional[int] = None


class ProviderSettingsUpdate(BaseModel):
    # When the field is null we ignore it; explicit empty string clears.
    polybacktest_api_key: Optional[str] = None
    polybacktest_base_url: Optional[str] = None
    reverse_engineer_max_iterations: Optional[int] = Field(default=None, ge=1, le=100)
    reverse_engineer_target_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reverse_engineer_max_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    reverse_engineer_max_wallet_trades: Optional[int] = Field(default=None, ge=10, le=250_000)


@router.get("/settings", response_model=ProviderSettings)
async def get_provider_settings() -> ProviderSettings:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    if row is None:
        return ProviderSettings()
    return ProviderSettings(
        polybacktest_api_key_set=bool(decrypt_secret(getattr(row, "polybacktest_api_key", None))),
        polybacktest_base_url=getattr(row, "polybacktest_base_url", None),
        reverse_engineer_max_iterations=getattr(row, "reverse_engineer_max_iterations", None),
        reverse_engineer_target_score=getattr(row, "reverse_engineer_target_score", None),
        reverse_engineer_max_cost_usd=getattr(row, "reverse_engineer_max_cost_usd", None),
        reverse_engineer_max_wallet_trades=getattr(row, "reverse_engineer_max_wallet_trades", None),
    )


@router.put("/settings")
async def update_provider_settings(req: ProviderSettingsUpdate) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
        if row is None:
            row = AppSettings(id="default")
            session.add(row)

        if req.polybacktest_api_key is not None:
            cleaned = req.polybacktest_api_key.strip()
            if cleaned == "":
                row.polybacktest_api_key = None
            elif cleaned == _API_KEY_PRESENT_MASK:
                # UI sent the mask back unchanged — leave the stored secret alone.
                pass
            else:
                row.polybacktest_api_key = encrypt_secret(cleaned)
        if req.polybacktest_base_url is not None:
            row.polybacktest_base_url = req.polybacktest_base_url.strip() or None
        if req.reverse_engineer_max_iterations is not None:
            row.reverse_engineer_max_iterations = int(req.reverse_engineer_max_iterations)
        if req.reverse_engineer_target_score is not None:
            row.reverse_engineer_target_score = float(req.reverse_engineer_target_score)
        if req.reverse_engineer_max_cost_usd is not None:
            row.reverse_engineer_max_cost_usd = float(req.reverse_engineer_max_cost_usd)
        if req.reverse_engineer_max_wallet_trades is not None:
            row.reverse_engineer_max_wallet_trades = int(req.reverse_engineer_max_wallet_trades)

        await session.commit()
    return {"ok": True}


def _serialize_dataset(row: Any, include_payload: bool = False) -> dict[str, Any]:
    out = {
        "id": row.id,
        "provider": row.provider,
        "coin": row.coin,
        "external_id": row.external_id,
        "external_slug": row.external_slug,
        "title": row.title,
        "asset_class": row.asset_class,
        "token_ids": list(row.token_ids_json or []),
        "start_ts": row.start_ts.isoformat() if row.start_ts else None,
        "end_ts": row.end_ts.isoformat() if row.end_ts else None,
        "snapshot_count": int(row.snapshot_count or 0),
        "trade_count": int(row.trade_count or 0),
        "last_imported_at": row.last_imported_at.isoformat() if row.last_imported_at else None,
        "last_import_job_id": row.last_import_job_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    if include_payload:
        out["payload"] = row.payload_json
    return out


# ---------------------------------------------------------------------------
# Parquet datasets (operator drops files into HOMERUN_PARQUET_ROOT;
# auto-discovery scanner upserts provider_datasets rows so the
# backtester's source resolver can find them).
# ---------------------------------------------------------------------------


@router.get("/parquet/root")
async def get_parquet_root() -> dict[str, Any]:
    """Surface the storage root the operator should drop parquet files
    into.  UI shows this so users know where to copy vendor data."""
    from services.external_data.parquet_schema import parquet_root

    root = parquet_root()
    return {
        "root": str(root),
        "exists": root.exists(),
        "env_var": "HOMERUN_PARQUET_ROOT",
    }


@router.get("/parquet/datasets")
async def list_parquet_datasets_route() -> dict[str, Any]:
    """List every parquet-backed dataset currently in the catalog.
    These rows are written by the auto-discovery scanner; if a file
    you just dropped doesn't appear, hit ``POST /parquet/rescan``."""
    from services.external_data import parquet_scanner

    rows = await parquet_scanner.list_parquet_datasets()
    return {"count": len(rows), "datasets": rows}


@router.post("/parquet/rescan")
async def rescan_parquet_route() -> dict[str, Any]:
    """Walk the parquet root and UPSERT a row per discovered group.
    Idempotent.  Returns a per-group report so the UI can show what
    was added / updated / errored.
    """
    from services.external_data import parquet_scanner

    try:
        report = await parquet_scanner.rescan_parquet_root()
    except Exception as exc:
        logger.exception("parquet rescan failed")
        raise HTTPException(status_code=500, detail=f"rescan failed: {exc}") from exc
    return report
