"""Telonex download + catalog-registration service.

The Telonex API exposes parquet files via 302 redirects to short-lived
presigned URLs.  Each call counts against the operator's quota (5
downloads on the free trial).  This service:

  1. Resolves the presigned URL via :class:`TelonexClient.download_url`
  2. Streams the parquet bytes onto disk under
     ``{parquet_root}/_telonex/data/{exchange}/{channel}/{asset_key}/{date}.parquet``
  3. Upserts a single :class:`ProviderDataset` row per (asset, channel)
     so the existing Data Lab "Imported datasets" panel and the
     Backtest Studio dataset picker can find it.
  4. Persists the latest ``X-Downloads-Remaining`` count back to
     ``AppSettings`` so the UI quota pill stays accurate.

We do *not* parse the parquet rows — Telonex's schemas differ per
channel (trades vs quotes vs book_snapshot_5) and the existing
parquet scanner expects Homerun's own snapshot/delta layout.  Treating
Telonex files as opaque blobs at known storage_uris keeps this service
shippable today; conversion to the unified microstructure schema is a
follow-up for whoever wires the backtester reader.

Concurrency notes:
  * Synchronous-from-the-caller-perspective today — downloads happen
    inline on the request task.  Multiple days run sequentially so we
    never hit Telonex's per-account parallel cap and so the quota
    counter updates atomically.
  * A future async worker job is trivial to layer on top: bundle the
    range into a ``ProviderImportJob`` and have the worker call
    ``import_range()`` here.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from models.database import AppSettings, AsyncSessionLocal, ProviderDataset
from services.external_data.telonex_client import (
    TelonexAuthError,
    TelonexError,
    TelonexNotFoundError,
    TelonexValidationError,
    build_client_from_settings,
)
from services.external_data.telonex_markets_cache import catalog_dir

logger = logging.getLogger(__name__)


PROVIDER_TELONEX = "telonex"
_STORAGE_TYPE = "telonex_parquet"


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelonexImportSpec:
    """One asset + channel + date range to fetch.

    Identifier resolution (the API requires exactly one):
      * ``asset_id`` (highest precision)
      * ``market_id`` + ``outcome`` / ``outcome_id``
      * ``slug`` + ``outcome`` / ``outcome_id``

    For Binance, slug / market_id / asset_id all hold the same value
    (the lowercase symbol, e.g. ``btcusdt``) and ``outcome`` is
    irrelevant.
    """

    exchange: str
    channel: str
    start_date: str  # YYYY-MM-DD (inclusive)
    end_date: str    # YYYY-MM-DD (inclusive)
    asset_id: Optional[str] = None
    market_id: Optional[str] = None
    slug: Optional[str] = None
    outcome: Optional[str] = None
    outcome_id: Optional[int] = None

    def validate(self) -> None:
        if not self.exchange:
            raise TelonexValidationError("exchange is required")
        if not self.channel:
            raise TelonexValidationError("channel is required")
        if not (_DATE_RE.match(self.start_date) and _DATE_RE.match(self.end_date)):
            raise TelonexValidationError("dates must be ISO YYYY-MM-DD")
        if not (self.asset_id or self.market_id or self.slug):
            raise TelonexValidationError(
                "provide one of asset_id, market_id, or slug"
            )
        if (self.market_id or self.slug) and self.exchange.lower() == "polymarket":
            if not (self.outcome or self.outcome_id is not None):
                raise TelonexValidationError(
                    "polymarket market_id/slug requires outcome or outcome_id"
                )

    def asset_key(self) -> str:
        """Stable, filesystem-safe key identifying this asset.

        Priority asset_id > market_id+outcome > slug+outcome.  Long
        polymarket asset_ids are hashed to keep paths short.
        """
        if self.asset_id:
            return _safe_asset_segment(self.asset_id)
        outcome_part = (
            self.outcome
            if self.outcome
            else (f"o{self.outcome_id}" if self.outcome_id is not None else "")
        )
        base = self.market_id or self.slug or "unknown"
        seg = base if not outcome_part else f"{base}__{outcome_part}"
        return _safe_asset_segment(seg)

    def external_id(self) -> str:
        """Unique key for ProviderDataset.(provider, external_id)."""
        return f"{self.exchange}:{self.channel}:{self.asset_key()}"

    def label(self) -> str:
        outcome_part = (
            f" / {self.outcome}"
            if self.outcome
            else (f" / outcome_{self.outcome_id}" if self.outcome_id is not None else "")
        )
        asset = self.asset_id or self.market_id or self.slug or "?"
        # Asset IDs can be 70+ chars — truncate for human-readable label.
        if len(asset) > 32:
            asset = asset[:14] + "…" + asset[-8:]
        return f"{self.exchange} · {self.channel} · {asset}{outcome_part}"


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelonexDayResult:
    date: str
    ok: bool
    bytes: int
    path: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class TelonexImportResult:
    spec: TelonexImportSpec
    dataset_id: Optional[str]
    storage_uri: Optional[str]
    days_requested: int
    days_succeeded: int
    days_failed: int
    bytes_downloaded: int
    quota_remaining: Optional[int]
    day_results: list[TelonexDayResult]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def import_range(spec: TelonexImportSpec) -> TelonexImportResult:
    """Download every day in the spec's range and register the dataset.

    This is the only function in the module that spends quota.  One
    HTTP call per requested day; aborts early on the first 403 (quota
    exhausted) but still registers the partial slice that landed.
    """
    spec.validate()

    target_dir = _data_dir_for(spec)
    target_dir.mkdir(parents=True, exist_ok=True)
    days = list(_iter_dates(spec.start_date, spec.end_date))

    client = await build_client_from_settings(require_api_key=True)
    day_results: list[TelonexDayResult] = []
    total_bytes = 0
    succeeded: list[str] = []
    last_remaining: Optional[int] = None
    aborted = False
    try:
        for d in days:
            target_path = target_dir / f"{d}.parquet"
            try:
                result = await client.download_to_path(
                    exchange=spec.exchange,
                    channel=spec.channel,
                    date=d,
                    target_path=target_path,
                    asset_id=spec.asset_id,
                    market_id=spec.market_id,
                    slug=spec.slug,
                    outcome=spec.outcome,
                    outcome_id=spec.outcome_id,
                )
            except TelonexNotFoundError as exc:
                # No data for this day — log + continue with the rest
                # of the range.  Doesn't burn the quota counter (Telonex
                # returns 404 before charging).
                day_results.append(TelonexDayResult(
                    date=d, ok=False, bytes=0, path=None,
                    error=f"no data: {exc}",
                ))
                continue
            except TelonexAuthError as exc:
                # 403 = quota exhausted.  Record the latest remaining
                # value and stop — burning further days won't help.
                last_remaining = exc.downloads_remaining
                day_results.append(TelonexDayResult(
                    date=d, ok=False, bytes=0, path=None, error=str(exc),
                ))
                aborted = True
                break
            except (TelonexError, OSError) as exc:
                day_results.append(TelonexDayResult(
                    date=d, ok=False, bytes=0, path=None, error=str(exc),
                ))
                continue

            total_bytes += int(result.get("bytes") or 0)
            last_remaining = result.get("downloads_remaining") if result.get("downloads_remaining") is not None else last_remaining
            succeeded.append(d)
            day_results.append(TelonexDayResult(
                date=d, ok=True, bytes=int(result.get("bytes") or 0),
                path=str(target_path), error=None,
            ))
    finally:
        # Always update the cached quota counter when we observed one,
        # even on partial failure.
        try:
            if last_remaining is not None or client.stats().get("last_downloads_remaining") is not None:
                rem = last_remaining if last_remaining is not None else client.stats().get("last_downloads_remaining")
                await _persist_quota(int(rem))
        finally:
            await client.close()

    dataset_id: Optional[str] = None
    storage_uri: Optional[str] = None
    if succeeded:
        dataset_id, storage_uri = await _upsert_dataset(spec, target_dir, succeeded, total_bytes)

    return TelonexImportResult(
        spec=spec,
        dataset_id=dataset_id,
        storage_uri=storage_uri,
        days_requested=len(days),
        days_succeeded=len(succeeded),
        days_failed=len(days) - len(succeeded) - (0 if not aborted else 0),
        bytes_downloaded=total_bytes,
        quota_remaining=last_remaining,
        day_results=day_results,
    )


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _data_dir_for(spec: TelonexImportSpec) -> Path:
    return (
        catalog_dir()
        / "data"
        / spec.exchange.lower()
        / spec.channel.lower()
        / spec.asset_key()
    )


_FORBIDDEN_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_asset_segment(value: str, *, max_len: int = 48) -> str:
    """Filesystem-safe representation of a slug / asset_id.

    Polymarket asset_ids are 70+ char decimal strings — preserve full
    fidelity but hash long values to keep path lengths sane on Windows
    (260-char limit unless long-path support is on).
    """
    cleaned = _FORBIDDEN_PATH_CHARS.sub("-", value or "")
    if len(cleaned) <= max_len:
        return cleaned
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    prefix = cleaned[: max_len - 9]  # leave room for "-{8-hex-chars}"
    return f"{prefix}-{h}"


def _iter_dates(start: str, end: str):
    """Inclusive day iterator, ISO YYYY-MM-DD strings."""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    if e < s:
        return
    d = s
    while d <= e:
        yield d.isoformat()
        d = d + timedelta(days=1)


def _file_uri(path: Path) -> str:
    """Cross-platform ``file://`` URI for a filesystem path."""
    abs_path = path.resolve()
    return abs_path.as_uri()


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------


async def _upsert_dataset(
    spec: TelonexImportSpec,
    target_dir: Path,
    succeeded_dates: list[str],
    total_bytes: int,
) -> tuple[str, str]:
    """Upsert a ProviderDataset row pointing at the directory of day-files.

    Stable ID: ``telonex:{sha1(external_id)[:16]}``.  Re-running the
    same import expands ``start_ts/end_ts`` and the ``payload_json``
    file index rather than creating a duplicate.
    """
    external_id = spec.external_id()
    dataset_id = "telonex:" + hashlib.sha1(external_id.encode("utf-8")).hexdigest()[:16]
    storage_uri = _file_uri(target_dir)

    # Determine the cumulative date window by walking the directory —
    # picks up days from prior import calls.
    all_dates: list[str] = []
    try:
        for p in sorted(target_dir.glob("*.parquet")):
            stem = p.stem
            if _DATE_RE.match(stem):
                all_dates.append(stem)
    except OSError:
        all_dates = list(succeeded_dates)

    if not all_dates:
        all_dates = list(succeeded_dates)
    start_ts = datetime.strptime(min(all_dates), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_ts = datetime.strptime(max(all_dates), "%Y-%m-%d").replace(tzinfo=timezone.utc)

    payload: dict[str, Any] = {
        "exchange": spec.exchange,
        "channel": spec.channel,
        "asset_id": spec.asset_id,
        "market_id": spec.market_id,
        "slug": spec.slug,
        "outcome": spec.outcome,
        "outcome_id": spec.outcome_id,
        "files": all_dates,
        "last_run_bytes": int(total_bytes),
    }

    token_id = f"{PROVIDER_TELONEX}:{spec.exchange}:{spec.channel}:{spec.asset_key()}"
    asset_class = "prediction" if spec.exchange.lower() == "polymarket" else "spot"

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderDataset).where(
                    ProviderDataset.provider == PROVIDER_TELONEX,
                    ProviderDataset.external_id == external_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = ProviderDataset(
                id=dataset_id,
                provider=PROVIDER_TELONEX,
                external_id=external_id,
                external_slug=spec.slug,
                title=spec.label(),
                asset_class=asset_class,
                token_ids_json=[token_id],
                storage_type=_STORAGE_TYPE,
                storage_uri=storage_uri,
                start_ts=start_ts,
                end_ts=end_ts,
                snapshot_count=len(all_dates),
                trade_count=0,
                last_imported_at=datetime.now(timezone.utc),
                payload_json=payload,
            )
            session.add(row)
        else:
            row.title = spec.label()
            row.asset_class = asset_class
            row.token_ids_json = [token_id]
            row.storage_type = _STORAGE_TYPE
            row.storage_uri = storage_uri
            row.start_ts = start_ts
            row.end_ts = end_ts
            row.snapshot_count = len(all_dates)
            row.last_imported_at = datetime.now(timezone.utc)
            row.payload_json = payload
        await session.commit()

    return dataset_id, storage_uri


async def _persist_quota(remaining: int) -> None:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
        if row is None:
            row = AppSettings(id="default")
            session.add(row)
        row.telonex_downloads_remaining = int(remaining)
        row.telonex_downloads_remaining_at = datetime.now(timezone.utc)
        await session.commit()


async def get_quota_snapshot() -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    if row is None:
        return {"remaining": None, "checked_at": None}
    val = getattr(row, "telonex_downloads_remaining", None)
    at = getattr(row, "telonex_downloads_remaining_at", None)
    return {
        "remaining": int(val) if val is not None else None,
        "checked_at": at.isoformat() if at else None,
    }


__all__ = [
    "PROVIDER_TELONEX",
    "TelonexImportSpec",
    "TelonexImportResult",
    "TelonexDayResult",
    "import_range",
    "get_quota_snapshot",
]
