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
from pathlib import Path
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
from services.external_data.telonex_client import (
    TelonexError,
    TelonexNotConfiguredError,
    TelonexNotFoundError,
    TelonexValidationError,
    build_client_from_settings as build_telonex_client,
    channels_for as telonex_channels_for,
    supported_exchanges as telonex_supported_exchanges,
)
from services.external_data import telonex_import_service, telonex_markets_cache
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
    telonex_status = await _telonex_status()
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
            },
            {
                "key": "telonex",
                "label": "Telonex",
                "description": (
                    "Historical Polymarket prediction-market data (trades, "
                    "quotes, L2 book snapshots, on-chain fills) plus Binance "
                    "reference prices, delivered as daily Parquet files.  "
                    "Free trial = 5 total downloads; Plus tier = unlimited."
                ),
                "homepage": "https://telonex.io",
                "docs_url": "https://telonex.io/docs/api/overview",
                "asset_classes": ["prediction", "crypto"],
                "supported_exchanges": list(telonex_supported_exchanges()),
                "supported_coins": [],
                "configured": telonex_status["configured"],
                "health": telonex_status,
            },
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


async def _telonex_status() -> dict[str, Any]:
    """Status probe — hits the public ``/datasets/polymarket/markets``
    endpoint (no auth, no quota cost) so the operator can verify the API
    is reachable even before they enter their key.  ``configured`` still
    reflects whether the key has been saved.
    """
    # First: do we have a key saved?  This drives the "configured" pill.
    try:
        configured_client = await build_telonex_client(require_api_key=True)
        configured = True
        await configured_client.close()
    except TelonexNotConfiguredError:
        configured = False
    except Exception:
        configured = False

    # Reachability probe — always runs, uses no quota.
    try:
        probe = await build_telonex_client(require_api_key=False)
    except Exception as exc:
        return {"configured": configured, "ok": False, "error": str(exc)}
    try:
        result = await probe.health()
        result["configured"] = configured
        return result
    finally:
        await probe.close()


# ---------------------------------------------------------------------------
# Telonex — markets catalog, availability, import, quota
# ---------------------------------------------------------------------------


@router.get("/telonex/catalog")
async def get_telonex_catalog_status(
    exchange: str = Query(default="polymarket"),
) -> dict[str, Any]:
    """Return cache info for the local markets catalog.

    The UI shows ``Last refreshed: ...`` and a "Refresh now" button
    based on this.  The catalog is the public Telonex markets dataset
    cached locally — does NOT count against the download quota.
    """
    status = telonex_markets_cache.catalog_status(exchange)
    return {
        "exchange": status.exchange,
        "exists": status.exists,
        "size_bytes": status.size_bytes,
        "rows": status.rows,
        "downloaded_at_epoch": status.downloaded_at_epoch,
        "path": status.path,
    }


@router.post("/telonex/catalog/refresh")
async def refresh_telonex_catalog(
    exchange: str = Query(default="polymarket"),
) -> dict[str, Any]:
    """Re-download the public markets dataset.  ~660 MB for polymarket,
    no API key required, no quota cost.  Synchronous (the FE shows a
    loading spinner while it streams)."""
    if exchange.lower() == "binance":
        raise HTTPException(
            status_code=400,
            detail="binance has no markets dataset — type a symbol directly (e.g. btcusdt)",
        )
    try:
        return await telonex_markets_cache.refresh_markets_catalog(exchange)
    except Exception as exc:
        logger.exception("telonex catalog refresh failed")
        raise HTTPException(status_code=502, detail=f"refresh failed: {exc}") from exc


@router.get("/telonex/markets")
async def list_telonex_markets(
    exchange: str = Query(default="polymarket"),
    search: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    channel: Optional[str] = Query(default=None),
    event_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Paginated, filtered query against the cached markets catalog.

    Returns ``catalog_missing: true`` when the operator hasn't pulled
    the catalog yet — the UI surfaces this with a "Refresh catalog"
    call-to-action.
    """
    if exchange.lower() == "binance":
        # Binance has no markets dataset.  Return an empty page so the
        # UI shows the "type a symbol directly" fallback.
        return {
            "exchange": exchange,
            "total": 0,
            "limit": int(limit),
            "offset": int(offset),
            "markets": [],
            "catalog_missing": False,
            "no_catalog_support": True,
        }
    return await telonex_markets_cache.list_markets(
        exchange=exchange,
        search=search,
        status=status,
        channel=channel,
        event_id=event_id,
        limit=int(limit),
        offset=int(offset),
    )


@router.get("/telonex/availability/{exchange}")
async def get_telonex_availability(
    exchange: str,
    asset_id: Optional[str] = Query(default=None),
    market_id: Optional[str] = Query(default=None),
    slug: Optional[str] = Query(default=None),
    outcome: Optional[str] = Query(default=None),
    outcome_id: Optional[int] = Query(default=None),
) -> dict[str, Any]:
    """Pass-through to ``GET /v1/availability/{exchange}`` — no auth,
    no quota cost.  Useful for Binance (no markets catalog) and as a
    second opinion for Polymarket assets that aren't in the operator's
    cached catalog yet.
    """
    client = await build_telonex_client(require_api_key=False)
    try:
        return await client.availability(
            exchange=exchange,
            asset_id=asset_id,
            market_id=market_id,
            slug=slug,
            outcome=outcome,
            outcome_id=outcome_id,
        )
    except TelonexValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TelonexNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TelonexError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        await client.close()


@router.get("/telonex/quota")
async def get_telonex_quota() -> dict[str, Any]:
    snap = await telonex_import_service.get_quota_snapshot()
    return snap


@router.get("/telonex/channels")
async def get_telonex_channels(exchange: str = Query(default="polymarket")) -> dict[str, Any]:
    return {"exchange": exchange, "channels": list(telonex_channels_for(exchange))}


class TelonexImportRequest(BaseModel):
    exchange: str = Field(default="polymarket")
    channel: str
    start_date: str  # YYYY-MM-DD inclusive
    end_date: str    # YYYY-MM-DD inclusive
    asset_id: Optional[str] = None
    market_id: Optional[str] = None
    slug: Optional[str] = None
    outcome: Optional[str] = None
    outcome_id: Optional[int] = None


@router.post("/telonex/import")
async def import_telonex(req: TelonexImportRequest) -> dict[str, Any]:
    """Download every day in the range to disk + register as a
    :class:`ProviderDataset` row.  Synchronous (one HTTP call per day,
    sequential — keeps quota usage atomic).  Aborts on the first 403
    but reports the partial slice that landed.
    """
    spec = telonex_import_service.TelonexImportSpec(
        exchange=req.exchange,
        channel=req.channel,
        start_date=req.start_date,
        end_date=req.end_date,
        asset_id=req.asset_id,
        market_id=req.market_id,
        slug=req.slug,
        outcome=req.outcome,
        outcome_id=req.outcome_id,
    )
    try:
        result = await telonex_import_service.import_range(spec)
    except TelonexNotConfiguredError as exc:
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    except TelonexValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TelonexError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "dataset_id": result.dataset_id,
        "storage_uri": result.storage_uri,
        "days_requested": result.days_requested,
        "days_succeeded": result.days_succeeded,
        "days_failed": result.days_failed,
        "bytes_downloaded": result.bytes_downloaded,
        "quota_remaining": result.quota_remaining,
        "day_results": [
            {
                "date": d.date, "ok": d.ok, "bytes": d.bytes,
                "path": d.path, "error": d.error,
            }
            for d in result.day_results
        ],
    }


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
    telonex_api_key_set: bool = False
    telonex_base_url: Optional[str] = None
    # Reverse-engineer model lives in app_settings.llm_model_assignments
    # under the 'strategy_reverse_engineer' key — managed in AI → Models.
    reverse_engineer_max_iterations: Optional[int] = None
    reverse_engineer_target_score: Optional[float] = None
    reverse_engineer_max_cost_usd: Optional[float] = None
    reverse_engineer_max_wallet_trades: Optional[int] = None
    # Parquet ingest roots — UI-editable list.  Empty list / null
    # falls back to the built-in default <repo>/data/parquet.  The
    # scanner walks every root in order; backtests resolve coverage
    # against the union.
    parquet_root_overrides: Optional[list[str]] = None


class ProviderSettingsUpdate(BaseModel):
    # When the field is null we ignore it; explicit empty string clears.
    polybacktest_api_key: Optional[str] = None
    polybacktest_base_url: Optional[str] = None
    telonex_api_key: Optional[str] = None
    telonex_base_url: Optional[str] = None
    reverse_engineer_max_iterations: Optional[int] = Field(default=None, ge=1, le=100)
    reverse_engineer_target_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reverse_engineer_max_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    reverse_engineer_max_wallet_trades: Optional[int] = Field(default=None, ge=10, le=250_000)
    # Pass an explicit empty list [] to clear all roots; null leaves
    # the stored list unchanged.  Each entry must be an absolute
    # directory path that exists on the host.
    parquet_root_overrides: Optional[list[str]] = None


@router.get("/settings", response_model=ProviderSettings)
async def get_provider_settings() -> ProviderSettings:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    if row is None:
        return ProviderSettings()
    return ProviderSettings(
        polybacktest_api_key_set=bool(decrypt_secret(getattr(row, "polybacktest_api_key", None))),
        polybacktest_base_url=getattr(row, "polybacktest_base_url", None),
        telonex_api_key_set=bool(decrypt_secret(getattr(row, "telonex_api_key", None))),
        telonex_base_url=getattr(row, "telonex_base_url", None),
        reverse_engineer_max_iterations=getattr(row, "reverse_engineer_max_iterations", None),
        reverse_engineer_target_score=getattr(row, "reverse_engineer_target_score", None),
        reverse_engineer_max_cost_usd=getattr(row, "reverse_engineer_max_cost_usd", None),
        reverse_engineer_max_wallet_trades=getattr(row, "reverse_engineer_max_wallet_trades", None),
        parquet_root_overrides=list(getattr(row, "parquet_root_overrides", None) or []) or None,
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
        if req.telonex_api_key is not None:
            cleaned = req.telonex_api_key.strip()
            if cleaned == "":
                row.telonex_api_key = None
            elif cleaned == _API_KEY_PRESENT_MASK:
                pass
            else:
                row.telonex_api_key = encrypt_secret(cleaned)
        if req.telonex_base_url is not None:
            row.telonex_base_url = req.telonex_base_url.strip() or None
        if req.reverse_engineer_max_iterations is not None:
            row.reverse_engineer_max_iterations = int(req.reverse_engineer_max_iterations)
        if req.reverse_engineer_target_score is not None:
            row.reverse_engineer_target_score = float(req.reverse_engineer_target_score)
        if req.reverse_engineer_max_cost_usd is not None:
            row.reverse_engineer_max_cost_usd = float(req.reverse_engineer_max_cost_usd)
        if req.reverse_engineer_max_wallet_trades is not None:
            row.reverse_engineer_max_wallet_trades = int(req.reverse_engineer_max_wallet_trades)
        if req.parquet_root_overrides is not None:
            # Normalise: drop empties + de-dupe + persist as JSON list.
            cleaned_list: list[str] = []
            seen: set[str] = set()
            for v in req.parquet_root_overrides:
                if v is None:
                    continue
                s = str(v).strip()
                if not s or s in seen:
                    continue
                cleaned_list.append(s)
                seen.add(s)
            row.parquet_root_overrides = cleaned_list or None
            # Push the new list into the in-process cache so the next
            # ``parquet_roots()`` call (e.g. the Rescan button the
            # operator is about to click) sees it without restart.
            from services.external_data.parquet_schema import set_parquet_root_overrides
            set_parquet_root_overrides(cleaned_list)

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
        # Storage routing — surfaces whether this dataset is backed
        # by Postgres (legacy polybacktest imports → mms table) or
        # by an on-disk parquet file (auto-discovered from
        # HOMERUN_PARQUET_ROOT).  The studio's data-source picker
        # uses this to badge the row so the operator sees at a
        # glance which route the backtest will take.  storage_uri
        # is the file:// URI for parquet rows; null for postgres.
        "storage_type": getattr(row, "storage_type", None) or "postgres",
        "storage_uri": getattr(row, "storage_uri", None),
    }
    if include_payload:
        out["payload"] = row.payload_json
    return out


# ---------------------------------------------------------------------------
# Parquet datasets — operators configure ingest roots in Data Lab →
# Providers → Parquet (multiple roots supported).  The HOMERUN_PARQUET_ROOT
# env var is no longer consulted; the DB-backed list is the single source
# of truth.  Auto-discovery scanner walks every root and upserts
# provider_datasets rows so the backtester's source resolver picks them up.
# ---------------------------------------------------------------------------


def _validate_parquet_root_path(raw: str) -> Path:
    """Reject non-absolute / non-existent / non-directory paths early
    so the operator gets a clear 400 instead of a silent no-op when
    the scanner later finds no files."""
    try:
        p = Path(raw).expanduser().resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid path: {exc}") from exc
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail=f"path must be absolute: {raw}")
    if not p.exists():
        raise HTTPException(
            status_code=400,
            detail=f"path does not exist: {p}.  Create the directory first, then save.",
        )
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"path is not a directory: {p}")
    return p


def _resolved_root_dict(p: Path) -> dict[str, Any]:
    """Per-root status payload returned by GET / PUT.  ``exists`` and
    ``writable`` are quick filesystem checks the UI uses to decorate
    the row (red dot if missing, etc.)."""
    return {
        "path": str(p),
        "exists": p.exists(),
        "writable": p.exists() and p.is_dir(),
    }


@router.get("/parquet/root")
async def get_parquet_root() -> dict[str, Any]:
    """Surface the configured parquet ingest roots.  UI shows these so
    operators see where backtests will read parquet data from + where
    new files should be dropped.

    Also lazy-loads the persisted overrides from ``app_settings`` into
    the in-process cache on every call — survives backend restarts
    without a dedicated startup hook, and lets a value set by another
    process / migration become visible immediately.
    """
    from services.external_data.parquet_schema import (
        parquet_roots,
        parquet_root_source,
        set_parquet_root_overrides,
    )

    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    persisted = list(getattr(row, "parquet_root_overrides", None) or []) if row is not None else []
    set_parquet_root_overrides(persisted)

    roots = parquet_roots()
    return {
        "roots": [_resolved_root_dict(p) for p in roots],
        # 'configured' = UI-set in app_settings; 'default' = <repo>/data/parquet
        "source": parquet_root_source(),
        # The configured override list (empty when falling back to default).
        "overrides": persisted,
    }


class ParquetRootUpdate(BaseModel):
    # Full replacement of the configured roots list.  Pass [] to clear
    # all overrides and fall back to the built-in default.  Each entry
    # must be an absolute existing directory.
    roots: list[str] = Field(default_factory=list, max_length=32)


@router.put("/parquet/root")
async def set_parquet_root(req: ParquetRootUpdate) -> dict[str, Any]:
    """Replace the UI-configured parquet ingest roots.  Persists the
    full list to ``app_settings.parquet_root_overrides`` and primes
    the in-process cache so the next Rescan picks up the new state
    immediately."""
    from services.external_data.parquet_schema import (
        parquet_roots,
        parquet_root_source,
        set_parquet_root_overrides,
    )

    # Normalise: drop empties + de-dupe + validate every entry.
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in req.roots:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        validated = _validate_parquet_root_path(s)
        canonical = str(validated)
        if canonical in seen:
            continue
        cleaned.append(canonical)
        seen.add(canonical)

    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
        if row is None:
            row = AppSettings(id="default")
            session.add(row)
        row.parquet_root_overrides = cleaned or None
        await session.commit()

    set_parquet_root_overrides(cleaned)
    roots = parquet_roots()
    return {
        "ok": True,
        "roots": [_resolved_root_dict(p) for p in roots],
        "source": parquet_root_source(),
        "overrides": cleaned,
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
