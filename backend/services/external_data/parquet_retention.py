"""Retention / pruning for the parquet data plane.

Now that ALL recorded/imported market data lives in parquet (not
Postgres), the parquet root grows unbounded.  This module enforces a
user-configurable retention policy — by **max age** and/or **max total
size** — over every configured parquet root, deleting whole window
directories (the atomic unit) oldest-first and pruning the matching
``ProviderDataset`` catalog rows.

Config lives in a small JSON file (control-plane, no DB write pressure,
no migration): ``data/cache/parquet_retention.json``::

    {"enabled": true, "max_age_days": 30, "max_total_gb": 50.0,
     "interval_minutes": 60}

Either threshold may be null (disabled).  ``prune`` is safe to run
repeatedly and reports exactly what it freed.  An optional background
loop auto-prunes on ``interval_minutes`` when ``enabled``.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from services.external_data.parquet_schema import parquet_roots
from utils.logger import get_logger

logger = get_logger("parquet_retention")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "parquet_retention.json"
_WINDOW_DIR_RE = re.compile(r"^(?P<start>\d{8}T\d{6})__(?P<end>\d{8}T\d{6})$")

_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "max_age_days": None,      # e.g. 30 — delete windows ending older than this
    "max_total_gb": None,      # e.g. 50 — keep newest windows under this size cap
    "interval_minutes": 60,
}


# ── config (JSON file) ────────────────────────────────────────────────
def get_config() -> dict[str, Any]:
    try:
        if _CONFIG_PATH.exists():
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {**_DEFAULT_CONFIG, **data}
    except Exception:
        logger.warning("parquet_retention: bad config file; using defaults", exc_info=True)
    return dict(_DEFAULT_CONFIG)


def set_config(patch: dict[str, Any]) -> dict[str, Any]:
    cfg = get_config()
    for k in ("enabled", "max_age_days", "max_total_gb", "interval_minutes"):
        if k in patch:
            cfg[k] = patch[k]
    # validation / clamping
    if cfg.get("max_age_days") is not None:
        cfg["max_age_days"] = max(1, int(cfg["max_age_days"]))
    if cfg.get("max_total_gb") is not None:
        cfg["max_total_gb"] = max(0.1, float(cfg["max_total_gb"]))
    cfg["interval_minutes"] = max(5, int(cfg.get("interval_minutes") or 60))
    cfg["enabled"] = bool(cfg.get("enabled"))
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


# ── window enumeration ────────────────────────────────────────────────
def _parse_end(window_name: str) -> Optional[datetime]:
    m = _WINDOW_DIR_RE.match(window_name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("end"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for f in path.rglob("*.parquet"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def _enumerate_windows() -> list[dict[str, Any]]:
    """Every parquet window dir across all roots: {path, end, size_bytes}."""
    windows: list[dict[str, Any]] = []
    for root in parquet_roots():
        if not root.exists():
            continue
        for provider_dir in root.iterdir():
            if not provider_dir.is_dir():
                continue
            for coin_dir in provider_dir.iterdir():
                if not coin_dir.is_dir():
                    continue
                for window_dir in coin_dir.iterdir():
                    if not window_dir.is_dir():
                        continue
                    end = _parse_end(window_dir.name)
                    if end is None:
                        continue
                    windows.append({
                        "path": window_dir,
                        "end": end,
                        "size_bytes": _dir_size_bytes(window_dir),
                    })
    return windows


# ── prune ─────────────────────────────────────────────────────────────
def _select_for_prune(
    windows: list[dict[str, Any]], *, max_age_days: int | None, max_total_gb: float | None,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    doomed: dict[Path, dict[str, Any]] = {}
    # Age policy
    if max_age_days is not None:
        cutoff_s = max_age_days * 86400
        for w in windows:
            if (now - w["end"]).total_seconds() > cutoff_s:
                doomed[w["path"]] = w
    # Size policy: keep newest under cap, delete oldest overflow.
    if max_total_gb is not None:
        cap = int(max_total_gb * (1024 ** 3))
        survivors = [w for w in windows if w["path"] not in doomed]
        survivors.sort(key=lambda w: w["end"], reverse=True)  # newest first
        running = 0
        for w in survivors:
            running += w["size_bytes"]
            if running > cap:
                doomed[w["path"]] = w
    return list(doomed.values())


async def preview_prune(*, max_age_days: int | None = None, max_total_gb: float | None = None) -> dict[str, Any]:
    cfg = get_config()
    age = max_age_days if max_age_days is not None else cfg.get("max_age_days")
    size = max_total_gb if max_total_gb is not None else cfg.get("max_total_gb")
    windows = await asyncio.to_thread(_enumerate_windows)
    total_bytes = sum(w["size_bytes"] for w in windows)
    doomed = _select_for_prune(windows, max_age_days=age, max_total_gb=size)
    return {
        "windows_total": len(windows),
        "bytes_total": total_bytes,
        "windows_to_delete": len(doomed),
        "bytes_to_free": sum(w["size_bytes"] for w in doomed),
        "max_age_days": age,
        "max_total_gb": size,
        "sample": [str(w["path"]) for w in sorted(doomed, key=lambda w: w["end"])[:10]],
    }


async def prune(*, max_age_days: int | None = None, max_total_gb: float | None = None) -> dict[str, Any]:
    cfg = get_config()
    age = max_age_days if max_age_days is not None else cfg.get("max_age_days")
    size = max_total_gb if max_total_gb is not None else cfg.get("max_total_gb")
    started = time.monotonic()
    windows = await asyncio.to_thread(_enumerate_windows)
    doomed = _select_for_prune(windows, max_age_days=age, max_total_gb=size)
    freed = 0
    deleted = 0
    for w in doomed:
        try:
            await asyncio.to_thread(shutil.rmtree, w["path"])
            freed += w["size_bytes"]
            deleted += 1
        except Exception:
            logger.warning("parquet_retention: failed to delete %s", w["path"], exc_info=True)
    # Drop catalog rows whose storage_uri no longer exists (DB-pressure-aware).
    catalog_pruned = await _prune_orphan_catalog_rows()
    logger.info("parquet_retention: deleted %d windows, freed %d bytes, pruned %d catalog rows",
                deleted, freed, catalog_pruned)
    return {
        "windows_deleted": deleted,
        "bytes_freed": freed,
        "catalog_rows_pruned": catalog_pruned,
        "duration_seconds": round(time.monotonic() - started, 2),
        "max_age_days": age,
        "max_total_gb": size,
    }


async def _prune_orphan_catalog_rows() -> int:
    """Delete ProviderDataset rows whose parquet window dir no longer exists.
    Skipped when the orchestrator DB is under pressure."""
    try:
        from services.live_pressure import is_db_pressure_active
        if is_db_pressure_active():
            return 0
    except Exception:
        pass
    try:
        from sqlalchemy import select
        from models.database import AsyncSessionLocal, ProviderDataset
        from services.external_data.parquet_scanner import _uri_to_path
    except Exception:
        return 0
    pruned = 0
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(ProviderDataset).where(ProviderDataset.storage_type == "parquet")
        )).scalars().all()
        for r in rows:
            if not r.storage_uri or not r.storage_uri.startswith("file://"):
                continue
            try:
                if not Path(_uri_to_path(r.storage_uri)).exists():
                    await session.delete(r)
                    pruned += 1
            except Exception:
                continue
        if pruned:
            await session.commit()
    return pruned


# ── background auto-prune loop ────────────────────────────────────────
async def run_retention_loop() -> None:
    """Optional background loop — auto-prune on the configured interval
    when enabled.  Registered by the host worker."""
    while True:
        try:
            cfg = get_config()
            interval = max(5, int(cfg.get("interval_minutes") or 60))
            if cfg.get("enabled") and (cfg.get("max_age_days") or cfg.get("max_total_gb")):
                await prune()
            await asyncio.sleep(interval * 60)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("parquet_retention: loop error")
            await asyncio.sleep(300)
