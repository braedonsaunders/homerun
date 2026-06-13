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
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from services.external_data.parquet_schema import (
    DELTA_SCHEMA,
    SNAPSHOT_SCHEMA,
    manifest_path_for,
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
_BUNDLE_FILE_RE = re.compile(r"^(?P<kind>snapshots|deltas)\.parquet$")


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
    snapshot_bundle: Path | None = None
    delta_bundle: Path | None = None


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
                    bundle = _BUNDLE_FILE_RE.match(f.name)
                    if bundle is not None:
                        if bundle.group("kind") == "snapshots":
                            group.snapshot_bundle = f
                        else:
                            group.delta_bundle = f
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
        bundle = _BUNDLE_FILE_RE.match(f.name)
        if bundle is not None:
            if bundle.group("kind") == "snapshots":
                group.snapshot_bundle = f
            else:
                group.delta_bundle = f
            continue
        parsed = _parse_filename(f.name)
        if parsed is None:
            continue
        kind, token_id = parsed
        if kind == "snapshots":
            group.snapshot_files[token_id] = f
        elif kind == "deltas":
            group.delta_files[token_id] = f
    if (
        not group.snapshot_files
        and not group.delta_files
        and group.snapshot_bundle is None
        and group.delta_bundle is None
    ):
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
    valid_snap_tokens: list[str] = []
    valid_delta_tokens: list[str] = []
    errors: list[str] = []
    snapshot_count = 0
    trade_count = 0
    for kind in ("snapshots", "deltas"):
        valid_tokens, row_count = _validate_kind(group, kind, errors)
        if kind == "snapshots":
            valid_snap_tokens = valid_tokens
            snapshot_count = row_count
        else:
            valid_delta_tokens = valid_tokens
            trade_count = row_count
    return valid_snap_tokens, valid_delta_tokens, snapshot_count, trade_count, errors


def _validate_kind(group: _DatasetGroup, kind: str, errors: list[str]) -> tuple[list[str], int]:
    expected_names = set((SNAPSHOT_SCHEMA if kind == "snapshots" else DELTA_SCHEMA).names)
    bundle = group.snapshot_bundle if kind == "snapshots" else group.delta_bundle
    legacy_files = group.snapshot_files if kind == "snapshots" else group.delta_files

    if bundle is not None:
        meta = _read_metadata(bundle, expected_names, errors)
        if meta is not None:
            tokens = _bundle_tokens(group.window_dir, bundle, kind, int(meta.num_rows), errors)
            if tokens:
                return tokens, int(meta.num_rows)
            errors.append(f"{bundle.name}: no token_id values found")

    valid_tokens: list[str] = []
    row_count = 0
    for token_id, fp in legacy_files.items():
        meta = _read_metadata(fp, expected_names, errors)
        if meta is None:
            continue
        valid_tokens.append(token_id)
        row_count += int(meta.num_rows)
    return valid_tokens, row_count


def _read_metadata(fp: Path, expected_names: set[str], errors: list[str]):
    import pyarrow.parquet as pq

    try:
        meta = pq.read_metadata(str(fp))
        file_names = set(meta.schema.to_arrow_schema().names)
    except Exception as exc:
        errors.append(f"{fp.name}: unreadable: {str(exc)[:120]}")
        return None
    missing = expected_names - file_names
    if missing:
        errors.append(f"{fp.name}: missing columns: {sorted(missing)}")
        return None
    if meta.num_rows <= 0:
        errors.append(f"{fp.name}: empty file (0 rows)")
        return None
    return meta


def _load_manifest(window_dir: Path) -> dict[str, Any]:
    mp = manifest_path_for(window_dir)
    try:
        raw = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _bundle_tokens(
    window_dir: Path,
    bundle: Path,
    kind: str,
    row_count: int,
    errors: list[str],
) -> list[str]:
    import pyarrow.parquet as pq

    manifest = _load_manifest(window_dir)
    entry = manifest.get(kind) if isinstance(manifest.get(kind), dict) else {}
    tokens = entry.get("tokens") if isinstance(entry, dict) else None
    manifest_rows = entry.get("rows") if isinstance(entry, dict) else None
    if isinstance(manifest_rows, int) and manifest_rows != row_count:
        errors.append(f"{bundle.name}: manifest rows {manifest_rows} != footer rows {row_count}")
    if isinstance(tokens, list) and tokens:
        return sorted({str(t) for t in tokens if str(t)})
    try:
        table = pq.read_table(str(bundle), columns=["token_id"])
    except Exception as exc:
        errors.append(f"{bundle.name}: could not read token_id column: {str(exc)[:120]}")
        return []
    return sorted({str(t) for t in table.column("token_id").to_pylist() if t})


def _group_row_id(group: _DatasetGroup) -> str:
    """Stable id: hash of (provider, coin, window_slug).  Idempotent —
    re-scanning the same dir produces the same id, so the UPSERT path
    actually upserts rather than appending duplicates."""
    import hashlib

    ext_id_raw = f"{group.provider}|{group.coin}|{group.window_dir.name}"
    return "parquet:" + hashlib.sha1(ext_id_raw.encode()).hexdigest()[:16]


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

    ext_id = _group_row_id(group)
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
        # Conflict-target the NATURAL key (provider, external_id), not the PK:
        # rows imported by earlier paths can hold the same (provider,
        # external_id) under a DIFFERENT id, so an id-only arbiter never
        # matched them and every rescan of that window died with
        # UniqueViolation on uq_provider_dataset_provider_extid (observed
        # spamming the jobs plane each cycle). The PK stays untouched on
        # update (id is excluded from update_cols).
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "external_id"],
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


def _scan_stamp_path() -> Path | None:
    """Cross-process scan-freshness stamp, kept in the first parquet root's
    ``.pins`` metadata dir.  ``_LAST_SCAN_AT`` is process-local, so without
    this every worker restart re-paid the full multi-minute root walk even
    when another process had just completed one."""
    roots = parquet_roots()
    if not roots:
        return None
    return roots[0] / ".pins" / "last_full_scan.stamp"


def _read_scan_stamp_epoch() -> float:
    p = _scan_stamp_path()
    if p is None:
        return 0.0
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _write_scan_stamp() -> None:
    p = _scan_stamp_path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        logger.debug("parquet_scanner: scan stamp write failed", exc_info=True)

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


async def rescan_parquet_root(
    *, root: Path | None = None, force: bool = False
) -> dict[str, Any]:
    """Walk every configured parquet root (or the single ``root`` arg
    when caller wants a one-off), validate, and UPSERT one row per
    group.  Idempotent.  Safe to invoke concurrently — guarded by a
    process-local lock.  Returns a structured report the API + CLI
    surface to the operator.

    ``force=False`` (the backtest path) skips footer-validation for groups
    that are ALREADY cataloged and whose window ended in the past: recording
    windows are append-only while active and immutable once closed, so
    re-reading every historical file's footer is pure waste — at tens of
    thousands of recorded files it turned "pick up newly-dropped files"
    into a multi-GB, tens-of-minutes stall before every backtest.  The
    operator's explicit Rescan (``force=True``) still validates everything.

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

        existing_ids: set[str] = set()
        if not force:
            try:
                from sqlalchemy import select as _select
                from models.database import AsyncSessionLocal, ProviderDataset

                async with AsyncSessionLocal() as session:
                    existing_ids = {
                        row[0]
                        for row in (
                            await session.execute(
                                _select(ProviderDataset.id).where(
                                    ProviderDataset.storage_type == "parquet"
                                )
                            )
                        ).all()
                    }
            except Exception:
                logger.warning(
                    "parquet_scanner: existing-id preload failed; falling back "
                    "to full validation", exc_info=True,
                )
                existing_ids = set()
        # A window that ended this recently may still receive its final
        # flush — always revalidate those; everything older is immutable.
        immutable_cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

        results: list[dict[str, Any]] = []
        per_root_reports: list[dict[str, Any]] = []
        total_groups = 0
        total_skipped = 0
        for scan_root in scan_roots:
            root_started = time.monotonic()
            # The walk stats every file under the root (hundreds of
            # thousands on a long-running recorder) — pure blocking IO
            # that must never run on the event loop (same py-spy-caught
            # stall class as the live sink's old synchronous walk).
            groups = await asyncio.to_thread(_walk_parquet_root, scan_root)
            root_results: list[dict[str, Any]] = []
            root_skipped = 0
            for g in groups:
                g_end = g.end if g.end.tzinfo else g.end.replace(tzinfo=timezone.utc)
                if (
                    not force
                    and existing_ids
                    and g_end < immutable_cutoff
                    and _group_row_id(g) in existing_ids
                ):
                    root_skipped += 1
                    continue
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
            total_skipped += root_skipped
            per_root_reports.append({
                "root": str(scan_root),
                "groups_seen": len(groups),
                "skipped_unchanged": root_skipped,
                "elapsed_ms": (time.monotonic() - root_started) * 1000.0,
                "exists": scan_root.exists(),
            })
        _LAST_SCAN_AT = time.time()
        if root is None:
            # Only a full multi-root scan counts as global freshness.
            _write_scan_stamp()
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
            "skipped_unchanged": total_skipped,
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
    # Build groups via filesystem resolve/is_dir/iterdir/is_file OFF the event
    # loop. This walk previously ran synchronously inside the async function —
    # for the live sink that is the trading orchestrator's loop, so a
    # recording-only catalog pass stalled trading (caught via py-spy: the loop
    # blocked in pathlib.stat under _group_for_window_dir). Recording must never
    # affect the orchestrator; only the async DB upsert below belongs on the loop.
    materialized = list(window_dirs)

    def _build_groups() -> list[_DatasetGroup]:
        seen: set[Path] = set()
        built: list[_DatasetGroup] = []
        for wd in materialized:
            try:
                resolved = Path(wd).resolve()
            except Exception:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            group = _group_for_window_dir(resolved)
            if group is not None:
                built.append(group)
        return built

    groups: list[_DatasetGroup] = await asyncio.to_thread(_build_groups)
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
    global _LAST_SCAN_AT
    if _SCAN_SUSPEND_DEPTH > 0:
        return False
    if (time.time() - _LAST_SCAN_AT) <= max_age_seconds:
        return False
    # Cross-process freshness: another process (background loop, operator
    # rescan, a prior worker) may have completed a full scan recently —
    # honor its stamp instead of re-walking the entire store.
    stamp_epoch = _read_scan_stamp_epoch()
    if stamp_epoch and (time.time() - stamp_epoch) <= max_age_seconds:
        _LAST_SCAN_AT = stamp_epoch
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
