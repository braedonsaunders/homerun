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
    REFERENCE_SCHEMA,
    SNAPSHOT_SCHEMA,
    parquet_roots,
)

logger = logging.getLogger(__name__)


# ── Filename parsing ─────────────────────────────────────────────────

_WINDOW_DIR_RE = re.compile(
    r"^(?P<start>\d{8}T\d{6})__(?P<end>\d{8}T\d{6})$"
)
_FILE_RE = re.compile(
    r"^(?P<kind>snapshots|deltas|reference)__(?P<token_id>[A-Za-z0-9_\-]+)\.parquet$"
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
    reference_files: dict[str, Path] = field(default_factory=dict)  # synthetic series id -> path


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
                    elif kind == "reference":
                        group.reference_files[token_id] = f
    return list(groups.values())


# ── Schema validation (cheap; reads footer only) ─────────────────────


def _validate_schema(file_path: Path, kind: str) -> tuple[bool, str | None]:
    """Open the file's footer and check column names match the
    expected schema.  ~1ms per file; doesn't read row data.
    """
    import pyarrow.parquet as pq

    try:
        meta = pq.read_metadata(str(file_path))
    except Exception as exc:
        return False, f"unreadable: {str(exc)[:120]}"
    expected = {
        "snapshots": SNAPSHOT_SCHEMA,
        "deltas": DELTA_SCHEMA,
        "reference": REFERENCE_SCHEMA,
    }.get(kind, SNAPSHOT_SCHEMA)
    file_schema_arrow = pq.read_schema(str(file_path))
    expected_names = set(expected.names)
    file_names = set(file_schema_arrow.names)
    missing = expected_names - file_names
    if missing:
        return False, f"missing columns: {sorted(missing)}"
    if meta.num_rows <= 0:
        return False, "empty file (0 rows)"
    return True, None


# ── DB upsert ────────────────────────────────────────────────────────


async def _upsert_group(group: _DatasetGroup) -> dict[str, Any]:
    """Validate every file in the group and UPSERT a single
    ``provider_datasets`` row.  Returns a small dict of stats for
    the caller's report.
    """
    from sqlalchemy.dialects.postgresql import insert as _pg_insert
    from models.database import AsyncSessionLocal, ProviderDataset

    valid_snap_tokens: list[str] = []
    valid_delta_tokens: list[str] = []
    valid_reference_tokens: list[str] = []
    errors: list[str] = []
    snapshot_count = 0
    trade_count = 0
    reference_count = 0
    import pyarrow.parquet as pq

    for token_id, fp in group.snapshot_files.items():
        ok, reason = _validate_schema(fp, "snapshots")
        if not ok:
            errors.append(f"{fp.name}: {reason}")
            continue
        valid_snap_tokens.append(token_id)
        try:
            snapshot_count += pq.read_metadata(str(fp)).num_rows
        except Exception:
            pass
    for token_id, fp in group.delta_files.items():
        ok, reason = _validate_schema(fp, "deltas")
        if not ok:
            errors.append(f"{fp.name}: {reason}")
            continue
        valid_delta_tokens.append(token_id)
        try:
            trade_count += pq.read_metadata(str(fp)).num_rows
        except Exception:
            pass
    for token_id, fp in group.reference_files.items():
        ok, reason = _validate_schema(fp, "reference")
        if not ok:
            errors.append(f"{fp.name}: {reason}")
            continue
        valid_reference_tokens.append(token_id)
        try:
            reference_count += pq.read_metadata(str(fp)).num_rows
        except Exception:
            pass

    if not valid_snap_tokens and not valid_delta_tokens and not valid_reference_tokens:
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
    union_tokens = sorted(
        set(valid_snap_tokens) | set(valid_delta_tokens) | set(valid_reference_tokens)
    )

    # Reference (underlying price-series) groups are tagged distinctly so
    # they're never confused with order-book ('spot'/'prediction') datasets.
    # The backtest coverage lookup only ever resolves ``snapshots__*`` files,
    # so reference datasets are cataloged + browsable but never replayed as
    # a fillable book.
    is_reference_only = bool(valid_reference_tokens) and not valid_snap_tokens and not valid_delta_tokens
    if is_reference_only:
        asset_class = "reference"
    elif group.coin in {"btc", "eth", "sol", "xrp"}:
        asset_class = "spot"
    else:
        asset_class = "prediction"

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
        "asset_class": asset_class,
        "token_ids_json": union_tokens,
        "start_ts": group.start.replace(tzinfo=None),
        "end_ts": group.end.replace(tzinfo=None),
        "snapshot_count": snapshot_count + reference_count,
        "trade_count": trade_count,
        "last_imported_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "payload_json": {
            "snapshot_token_count": len(valid_snap_tokens),
            "delta_token_count": len(valid_delta_tokens),
            "reference_token_count": len(valid_reference_tokens),
            "reference_rows": reference_count,
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
        "reference_files": len(valid_reference_tokens),
        "snapshot_rows": snapshot_count,
        "delta_rows": trade_count,
        "reference_rows": reference_count,
        "errors": errors[:20],
    }


# ── Public API ───────────────────────────────────────────────────────


_LAST_SCAN_AT: float = 0.0
_RESCAN_LOCK = asyncio.Lock()


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


async def ensure_recent_scan(*, max_age_seconds: float = 60.0) -> bool:
    """Invoke ``rescan_parquet_root`` if the cached scan is older than
    ``max_age_seconds``.  Cheap fast-path: if a recent scan already
    happened we return immediately without walking the tree.

    Called from the backtester just before resolving sources so
    newly-dropped files are picked up automatically.  Returns True
    iff a fresh scan ran.
    """
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


async def find_parquet_coverage(
    *,
    token_ids: Iterable[str],
    start: datetime,
    end: datetime,
) -> dict[str, str]:
    """For each requested token, return the parquet file path covering
    its ``[start, end]`` window (or omit the entry if no parquet
    dataset covers it).  Used by the backtester's source resolver to
    decide which tokens get the parquet path vs. the SQL replays.

    A dataset "covers" a token iff the dataset's ``[start_ts, end_ts]``
    overlaps the requested window AND the token is in ``token_ids_json``.
    """
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, ProviderDataset

    requested = {str(t) for t in token_ids if t}
    if not requested:
        return {}
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(ProviderDataset).where(
                    ProviderDataset.storage_type == "parquet",
                    ProviderDataset.start_ts <= end,
                    ProviderDataset.end_ts >= start,
                )
            )
        ).scalars().all()

    out: dict[str, str] = {}
    for r in rows:
        tokens_in_dataset = set(r.token_ids_json or [])
        intersect = requested & tokens_in_dataset
        if not intersect:
            continue
        # Build the file path per token from the storage_uri (window dir)
        if not r.storage_uri or not r.storage_uri.startswith("file://"):
            continue
        try:
            window_dir = Path(_uri_to_path(r.storage_uri))
        except Exception:
            continue
        for tok in intersect:
            # Token id may have been "safe-segmented" on write; we
            # accept either the raw token or the canonical safe form.
            from services.external_data.parquet_schema import _safe_segment
            safe = _safe_segment(tok)
            candidates = [
                window_dir / f"snapshots__{safe}.parquet",
                window_dir / f"snapshots__{tok}.parquet",
            ]
            for c in candidates:
                if c.exists():
                    # Don't overwrite an earlier dataset for the same
                    # token — first dataset (most recent by ORDER BY)
                    # wins.
                    out.setdefault(tok, str(c))
                    break
    return out


async def find_reference_coverage(
    *,
    assets: Iterable[str] | None,
    start: datetime,
    end: datetime,
) -> dict[str, list[str]]:
    """Reference-series sibling of :func:`find_parquet_coverage`.

    Returns ``{series_id -> [file_path, ...]}`` for every reference
    (underlying price) parquet whose ``[start_ts, end_ts]`` overlaps the
    requested window.  ``series_id`` is the synthetic id stored in
    ``token_ids_json`` (e.g. ``ref_btc_chainlink``).  When ``assets`` is
    given, only series whose coin matches are returned.

    Deliberately separate from ``find_parquet_coverage`` (which only ever
    resolves ``snapshots__*`` book files): reference data is a SIGNAL feed,
    never a fillable book, so it must never enter the matching engine.
    The backtest's reference-replay shim consumes THIS, annotating
    ``oracle_prices_by_source`` onto discovery rows exactly like the live
    ``market_runtime`` does from ``ReferenceRuntime``.
    """
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, ProviderDataset

    wanted_coins = {str(a).strip().lower() for a in (assets or []) if str(a).strip()}
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(ProviderDataset).where(
                    ProviderDataset.storage_type == "parquet",
                    ProviderDataset.asset_class == "reference",
                    ProviderDataset.start_ts <= end,
                    ProviderDataset.end_ts >= start,
                )
            )
        ).scalars().all()

    out: dict[str, list[str]] = {}
    for r in rows:
        if wanted_coins and str(r.coin or "").strip().lower() not in wanted_coins:
            continue
        if not r.storage_uri or not r.storage_uri.startswith("file://"):
            continue
        try:
            window_dir = Path(_uri_to_path(r.storage_uri))
        except Exception:
            continue
        for sid in (r.token_ids_json or []):
            from services.external_data.parquet_schema import _safe_segment
            safe = _safe_segment(str(sid))
            for cand in (window_dir / f"reference__{safe}.parquet", window_dir / f"reference__{sid}.parquet"):
                if cand.exists():
                    out.setdefault(str(sid), [])
                    if str(cand) not in out[str(sid)]:
                        out[str(sid)].append(str(cand))
                    break
    return out


def _uri_to_path(uri: str) -> Path:
    """file:///C:/foo → C:/foo on Windows; file:///foo/bar → /foo/bar
    on POSIX.  Cross-platform via ``urllib.parse``.
    """
    from urllib.parse import urlparse, unquote
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    # Windows: ``file:///C:/foo`` parses with path="/C:/foo"; strip the leading "/"
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


__all__ = [
    "rescan_parquet_root",
    "ensure_recent_scan",
    "list_parquet_datasets",
    "find_parquet_coverage",
    "find_reference_coverage",
]
