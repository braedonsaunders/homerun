"""Filesystem auto-discovery for parquet datasets.

Walks every directory configured in Data Lab → Providers → Parquet
(``app_settings.parquet_root_overrides``) looking for parquet files
matching the canonical layout (see ``parquet_schema.parquet_path_for``):

    {root}/{provider}/{coin}/{window_slug}/{kind}__{token_id}.parquet

For every group of sibling files (same window_dir + provider + coin),
emits one row to ``provider_datasets`` with ``storage_type='parquet'``,
``storage_uri='file://{window_dir}'``, and ``token_ids_json`` listing
every token covered.  Re-running is idempotent — UPSERTs by the
``(provider, external_id)`` unique constraint.

The scanner does NOT validate row contents (that would require reading
every file's data section).  It only validates that the file is a
readable parquet with the expected schema column names.  Bad files are
logged and skipped.

Triggered three ways:

  1. ``rescan_parquet_root()``        — single shot, called by API
     route + CLI.
  2. ``ensure_recent_scan(max_age)``  — invoked at backtest start so
     newly-dropped files are picked up without an explicit rescan.
  3. ``run_loop()``                   — periodic background loop on
     the discovery plane (60s default, configurable).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from services.external_data.parquet_schema import (
    DELTA_SCHEMA,
    SNAPSHOT_SCHEMA,
    parquet_roots,
)

logger = logging.getLogger(__name__)


# ── Filename parsing ─────────────────────────────────────────────────

_WINDOW_DIR_RE = re.compile(
    r"^(?P<start>\d{8}T\d{6})__(?P<end>\d{8}T\d{6})$"
)
_FILE_RE = re.compile(
    r"^(?P<kind>snapshots|deltas)__(?P<token_id>[A-Za-z0-9_\-]+)\.parquet$"
)


def _parse_window_dir(name: str) -> tuple[datetime, datetime] | None:
    m = _WINDOW_DIR_RE.match(name)
    if not m:
        return None
    try:
        start = datetime.strptime(m.group("start"), "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )
        end = datetime.strptime(m.group("end"), "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return (start, end)


def _parse_filename(name: str) -> tuple[str, str] | None:
    m = _FILE_RE.match(name)
    if not m:
        return None
    return (m.group("kind"), m.group("token_id"))


# ── Per-window group ─────────────────────────────────────────────────


@dataclass
class _DatasetGroup:
    provider: str
    coin: str
    window_dir: Path
    start: datetime
    end: datetime
    snapshot_files: dict[str, Path] = field(default_factory=dict)  # token_id -> path
    delta_files: dict[str, Path] = field(default_factory=dict)


# ── Filesystem walk ──────────────────────────────────────────────────


def _walk_parquet_root(root: Path) -> list[_DatasetGroup]:
    """Collect every valid (provider, coin, window) group under root."""
    groups: dict[tuple[str, str, str], _DatasetGroup] = {}
    if not root.exists():
        return []
    for provider_dir in sorted(root.iterdir()):
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name
        for coin_dir in sorted(provider_dir.iterdir()):
            if not coin_dir.is_dir():
                continue
            coin = coin_dir.name
            for window_dir in sorted(coin_dir.iterdir()):
                if not window_dir.is_dir():
                    continue
                window = _parse_window_dir(window_dir.name)
                if window is None:
                    logger.debug(
                        "parquet_scanner: skipping non-window dir %s", window_dir
                    )
                    continue
                start, end = window
                key = (provider, coin, window_dir.name)
                group = groups.setdefault(
                    key,
                    _DatasetGroup(
                        provider=provider,
                        coin=coin,
                        window_dir=window_dir,
                        start=start,
                        end=end,
                    ),
                )
                for f in window_dir.iterdir():
                    if not f.is_file() or not f.name.endswith(".parquet"):
                        continue
                    parsed = _parse_filename(f.name)
                    if parsed is None:
                        logger.debug(
                            "parquet_scanner: skipping unparseable filename %s", f
                        )
                        continue
                    kind, token_id = parsed
                    if kind == "snapshots":
                        group.snapshot_files[token_id] = f
                    elif kind == "deltas":
                        group.delta_files[token_id] = f
    return list(groups.values())


def _group_for_window_dir(window_dir: Path) -> _DatasetGroup | None:
    """Build a single ``_DatasetGroup`` from one window dir without walking the
    whole tree.  Used by the live sink's incremental catalog path so only the
    windows it just wrote get re-UPSERTed (not every historical window).

    Layout: ``{root}/{provider}/{coin}/{window_slug}/{kind}__{token}.parquet``.
    """
    try:
        if not window_dir.is_dir():
            return None
    except OSError:
        return None
    window = _parse_window_dir(window_dir.name)
    if window is None:
        return None
    start, end = window
    group = _DatasetGroup(
        provider=window_dir.parent.parent.name,
        coin=window_dir.parent.name,
        window_dir=window_dir,
        start=start,
        end=end,
    )
    for f in window_dir.iterdir():
        if not f.is_file() or not f.name.endswith(".parquet"):
            continue
        parsed = _parse_filename(f.name)
        if parsed is None:
            continue
        kind, token_id = parsed
        if kind == "snapshots":
            group.snapshot_files[token_id] = f
        elif kind == "deltas":
            group.delta_files[token_id] = f
    if not group.snapshot_files and not group.delta_files:
        return None
    return group


# ── DB upsert ────────────────────────────────────────────────────────


def _validate_and_count_group(
    group: _DatasetGroup,
) -> tuple[list[str], list[str], int, int, list[str]]:
    """Synchronous parquet validation + row counting for a group.  Reads file
    footers, so async callers offload it via ``asyncio.to_thread`` to keep the
    event loop free — an active recording window can carry tens of token files
    and several seconds of footer reads.
    """
    import pyarrow.parquet as pq

    valid_snap_tokens: list[str] = []
    valid_delta_tokens: list[str] = []
    errors: list[str] = []
    snapshot_count = 0
    trade_count = 0
    for kind, files, valid_list in (
        ("snapshots", group.snapshot_files, valid_snap_tokens),
        ("deltas", group.delta_files, valid_delta_tokens),
    ):
        expected_names = set((SNAPSHOT_SCHEMA if kind == "snapshots" else DELTA_SCHEMA).names)
        for token_id, fp in files.items():
            # One footer read per file: validates column names + row count
            # (the old path read the footer three times per file).
            try:
                meta = pq.read_metadata(str(fp))
                file_names = set(meta.schema.to_arrow_schema().names)
            except Exception as exc:
                errors.append(f"{fp.name}: unreadable: {str(exc)[:120]}")
                continue
            missing = expected_names - file_names
            if missing:
                errors.append(f"{fp.name}: missing columns: {sorted(missing)}")
                continue
            if meta.num_rows <= 0:
                errors.append(f"{fp.name}: empty file (0 rows)")
                continue
            valid_list.append(token_id)
            if kind == "snapshots":
                snapshot_count += meta.num_rows
            else:
                trade_count += meta.num_rows
    return valid_snap_tokens, valid_delta_tokens, snapshot_count, trade_count, errors


async def _upsert_group(group: _DatasetGroup) -> dict[str, Any]:
    """Validate every file in the group and UPSERT a single
    ``provider_datasets`` row.  Returns a small dict of stats for
    the caller's report.
    """
    from sqlalchemy.dialects.postgresql import insert as _pg_insert
    from models.database import AsyncSessionLocal, ProviderDataset

    # Validate + count OFF the event loop — reading footers for an active
    # window's token files is pure file IO and can take seconds.
    valid_snap_tokens, valid_delta_tokens, snapshot_count, trade_count, errors = (
        await asyncio.to_thread(_validate_and_count_group, group)
    )

    if not valid_snap_tokens and not valid_delta_tokens:
        return {
            "provider": group.provider,
            "coin": group.coin,
            "window": group.window_dir.name,
            "skipped": True,
            "reason": "no valid files in group",
            "errors": errors,
        }

    # Stable id: hash of (provider, coin, window_slug).  Idempotent —
    # re-scanning the same dir produces the same id, so the UPSERT
    # path actually upserts rather than appending duplicates.
    import hashlib
    ext_id_raw = f"{group.provider}|{group.coin}|{group.window_dir.name}"
    ext_id = "parquet:" + hashlib.sha1(ext_id_raw.encode()).hexdigest()[:16]
    row_id = ext_id  # use external_id as the primary key for parquet rows
    storage_uri = group.window_dir.resolve().as_uri()
    union_tokens = sorted(set(valid_snap_tokens) | set(valid_delta_tokens))

    values = {
        "id": row_id,
        "provider": group.provider,
        "coin": group.coin,
        "external_id": ext_id,
        "external_slug": group.window_dir.name,
        "title": (
            f"{group.provider}/{group.coin} "
            f"{group.start.strftime('%Y-%m-%d')} to {group.end.strftime('%Y-%m-%d')}"
        ),
        "asset_class": "spot" if group.coin in {"btc", "eth", "sol", "xrp"} else "prediction",
        "token_ids_json": union_tokens,
        "start_ts": group.start.replace(tzinfo=None),
        "end_ts": group.end.replace(tzinfo=None),
        "snapshot_count": snapshot_count,
        "trade_count": trade_count,
        "last_imported_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "payload_json": {
            "snapshot_token_count": len(valid_snap_tokens),
            "delta_token_count": len(valid_delta_tokens),
            "errors": errors[:20],  # cap so a corrupt-files dir doesn't bloat the row
        },
        "storage_type": "parquet",
        "storage_uri": storage_uri,
    }
    # Mirror the columns the runner owns (everything except the
    # worker-managed cols on backtest_runs).  For provider_datasets
    # all columns above are runner-owned.
    update_cols = {k: v for k, v in values.items() if k not in {"id", "created_at"}}
    async with AsyncSessionLocal() as session:
        stmt = _pg_insert(ProviderDataset).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_=update_cols,
        )
        try:
            await session.execute(stmt)
            await session.commit()
        except Exception as exc:
            logger.warning(
                "parquet_scanner: UPSERT failed for %s: %s", row_id, exc
            )
            return {
                "provider": group.provider,
                "coin": group.coin,
                "window": group.window_dir.name,
                "error": str(exc)[:300],
            }

    return {
        "provider": group.provider,
        "coin": group.coin,
        "window": group.window_dir.name,
        "id": row_id,
        "tokens": len(union_tokens),
        "snapshot_files": len(valid_snap_tokens),
        "delta_files": len(valid_delta_tokens),
        "snapshot_rows": snapshot_count,
        "delta_rows": trade_count,
        "errors": errors[:20],
    }


# ── Public API ───────────────────────────────────────────────────────


_LAST_SCAN_AT: float = 0.0
_RESCAN_LOCK = asyncio.Lock()

# When > 0, ``ensure_recent_scan`` is a no-op.  A backtest replays a FROZEN,
# pinned dataset — the filesystem can't change underneath it — so re-walking
# every root + re-UPSERTing the catalog every 60s mid-run is pure waste and, on
# a long sub-second run, storms the connection pool (the rescan opens its own
# sessions).  The engine suspends scanning for the duration of a run.  Counted
# so nested/concurrent suspensions compose.
_SCAN_SUSPEND_DEPTH: int = 0


def suspend_scan() -> None:
    """Suspend ``ensure_recent_scan`` (backtest replays frozen data)."""
    global _SCAN_SUSPEND_DEPTH
    _SCAN_SUSPEND_DEPTH += 1


def resume_scan() -> None:
    global _SCAN_SUSPEND_DEPTH
    _SCAN_SUSPEND_DEPTH = max(0, _SCAN_SUSPEND_DEPTH - 1)


async def rescan_parquet_root(*, root: Path | None = None) -> dict[str, Any]:
    """Walk every configured parquet root (or the single ``root`` arg
    when caller wants a one-off), validate, and UPSERT one row per
    group.  Idempotent.  Safe to invoke concurrently — guarded by a
    process-local lock.  Returns a structured report the API + CLI
    surface to the operator.

    When the operator has configured multiple roots in Data Lab →
    Providers → Parquet, all of them are walked in order; results
    are aggregated under per-root sub-reports so the UI can show
    "root A: 3 groups, root B: 12 groups" cleanly.  When ``root``
    is supplied explicitly, ONLY that root is scanned (back-compat
    for tests + one-off CLI invocations).
    """
    global _LAST_SCAN_AT
    async with _RESCAN_LOCK:
        scan_roots: list[Path]
        if root is not None:
            scan_roots = [root.resolve()]
        else:
            scan_roots = parquet_roots()
        started = time.monotonic()
        results: list[dict[str, Any]] = []
        per_root_reports: list[dict[str, Any]] = []
        total_groups = 0
        for scan_root in scan_roots:
            root_started = time.monotonic()
            groups = _walk_parquet_root(scan_root)
            root_results: list[dict[str, Any]] = []
            for g in groups:
                try:
                    res = await _upsert_group(g)
                except Exception as exc:
                    logger.exception("parquet_scanner: group upsert raised")
                    res = {
                        "provider": g.provider,
                        "coin": g.coin,
                        "window": g.window_dir.name,
                        "error": str(exc)[:300],
                    }
                # Tag every result with which root it came from so a
                # multi-root scan's per-row report stays unambiguous.
                res["root"] = str(scan_root)
                root_results.append(res)
                results.append(res)
            total_groups += len(groups)
            per_root_reports.append({
                "root": str(scan_root),
                "groups_seen": len(groups),
                "elapsed_ms": (time.monotonic() - root_started) * 1000.0,
                "exists": scan_root.exists(),
            })
        _LAST_SCAN_AT = time.time()
        elapsed_ms = (time.monotonic() - started) * 1000.0
        # Keep the legacy top-level ``root`` key (= first scan root)
        # for back-compat with any frontend / CLI parsing the report
        # before this multi-root upgrade.  ``roots`` is the new
        # source of truth.
        return {
            "root": str(scan_roots[0]) if scan_roots else "",
            "roots": [str(p) for p in scan_roots],
            "per_root": per_root_reports,
            "groups_seen": total_groups,
            "results": results,
            "elapsed_ms": elapsed_ms,
            "scanned_at_epoch": _LAST_SCAN_AT,
        }


async def register_window_dirs(window_dirs: Iterable[Path]) -> dict[str, Any]:
    """Incrementally UPSERT ``provider_datasets`` rows for ONLY the given window
    dirs — the windows the live sink just wrote.

    The live recording sink calls this each catalog cycle with the handful of
    windows it actually touched, instead of ``rescan_parquet_root`` re-walking,
    re-validating, and re-UPSERTing every historical window on disk every pass
    (which made a parquet-only recorder the busiest Postgres writer + churned the
    connection pool).  Idempotent; shares ``_RESCAN_LOCK`` with the full rescan so
    the two can never double-write the same row.
    """
    started = time.monotonic()
    seen: set[Path] = set()
    groups: list[_DatasetGroup] = []
    for wd in window_dirs:
        try:
            resolved = Path(wd).resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        group = _group_for_window_dir(resolved)
        if group is not None:
            groups.append(group)
    results: list[dict[str, Any]] = []
    if groups:
        async with _RESCAN_LOCK:
            for g in groups:
                try:
                    results.append(await _upsert_group(g))
                except Exception as exc:
                    logger.exception("parquet_scanner: incremental group upsert raised")
                    results.append({
                        "provider": g.provider,
                        "coin": g.coin,
                        "window": g.window_dir.name,
                        "error": str(exc)[:300],
                    })
    return {
        "groups_seen": len(groups),
        "results": results,
        "elapsed_ms": (time.monotonic() - started) * 1000.0,
    }


async def ensure_recent_scan(*, max_age_seconds: float = 60.0) -> bool:
    """Invoke ``rescan_parquet_root`` if the cached scan is older than
    ``max_age_seconds``.  Cheap fast-path: if a recent scan already
    happened we return immediately without walking the tree.

    Called from the backtester just before resolving sources so
    newly-dropped files are picked up automatically.  Returns True
    iff a fresh scan ran.
    """
    if _SCAN_SUSPEND_DEPTH > 0:
        return False
    if (time.time() - _LAST_SCAN_AT) <= max_age_seconds:
        return False
    await rescan_parquet_root()
    return True


async def list_parquet_datasets() -> list[dict[str, Any]]:
    """Read-only catalog query — returns every parquet-backed
    ``provider_datasets`` row in a UI-friendly shape."""
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, ProviderDataset

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(ProviderDataset)
                .where(ProviderDataset.storage_type == "parquet")
                .order_by(ProviderDataset.start_ts.desc().nullslast())
            )
        ).scalars().all()
    return [
        {
            "id": r.id,
            "provider": r.provider,
            "coin": r.coin,
            "title": r.title,
            "start_ts": r.start_ts.isoformat() if r.start_ts else None,
            "end_ts": r.end_ts.isoformat() if r.end_ts else None,
            "token_count": len(r.token_ids_json or []),
            "snapshot_count": r.snapshot_count,
            "trade_count": r.trade_count,
            "storage_uri": r.storage_uri,
            "last_imported_at": (
                r.last_imported_at.isoformat() if r.last_imported_at else None
            ),
        }
        for r in rows
    ]


# NOTE: per-token coverage resolution moved to the unified
# ``services.marketdata.coverage.resolve_coverage`` (returns a structured
# CoverageMap with all covering files per token). This module now owns only
# filesystem scanning + catalog upserts; consumers read coverage via the
# market-data layer.


__all__ = [
    "rescan_parquet_root",
    "register_window_dirs",
    "ensure_recent_scan",
    "list_parquet_datasets",
]
