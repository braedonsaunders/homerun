"""Window compactor for the recorded parquet data plane (v2 bundle layout).

Purpose
-------
The live recorder used to leave ONE FILE PER (kind, token) in every 15-minute
window directory — ~17k tokens per window meant >1M files/day, which degraded
every tree walker (a single storage-summary walk wedged the API event loop),
the pruner, the catalog scanner, and NTFS itself.  The v2 layout stores ONE
BUNDLE PER (kind, window) with ``token_id`` as a column (see
``parquet_schema``); this module is the machinery that produces those bundles.

Inputs per window dir (any combination):
  * ``_parts/{kind}__part-XXXXXX.parquet`` — incremental flushes from the live
    sink for the (then-open) window;
  * legacy ``{kind}__{token}.parquet`` per-token files (pre-v2 recordings or
    operator imports);
  * an existing ``{kind}.parquet`` bundle (re-compaction after late parts).

Output per window dir:
  * ``{kind}.parquet`` — all rows, sorted (token_id, observed_at_us) so parquet
    row-group statistics give per-token predicate pushdown on read;
  * ``manifest.json``   — {kind: {"rows": N, "tokens": [...]}} so discovery
    (catalog scanner, coverage) never needs a column scan.

Integrity contract (financial-grade):
  1. Source files are NEVER deleted until the bundle is written AND verified:
     bundle row count must equal the sum of source row counts (from parquet
     footers) and the bundle's distinct token set must equal the union of the
     sources' token sets.
  2. The bundle write is atomic (canonical writer: tmp + os.replace), so a
     crash mid-compaction leaves the window exactly as it was — the next pass
     redoes the work idempotently.
  3. The manifest is written atomically AFTER the bundle, and sources are
     deleted only after both.  A window with a verified bundle and leftover
     sources (crash between steps) re-verifies and just finishes the cleanup.

Everything here is PURE SYNC — callers dispatch via ``asyncio.to_thread``
(the live sink's loop) or run it directly (the one-shot migration tool).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from services.external_data.parquet_schema import (
    PARTS_DIRNAME,
    bundle_path_for,
    manifest_path_for,
    parts_dir_for,
    schema_for,
)
from services.marketdata.schema import BOOK_SCHEMA_VERSION
from services.marketdata.writer import write_canonical_table
from utils.logger import get_logger

logger = get_logger("parquet_compactor")

_KINDS = ("snapshots", "deltas")
_ROW_GROUP_SIZE = 64_000
# Legacy per-token filename: {kind}__{token}.parquet.  Bundle files are named
# exactly "{kind}.parquet" (no dunder) so they can never collide with this.
_LEGACY_FILE_RE = re.compile(r"^(snapshots|deltas)__([A-Za-z0-9_\-]+)\.parquet$")
_PART_FILE_RE = re.compile(r"^(snapshots|deltas)__part-\d+\.parquet$")
_WINDOW_DIR_RE = re.compile(r"^(\d{8}T\d{6})__(\d{8}T\d{6})$")


@dataclass
class CompactResult:
    window_dir: Path
    kinds_compacted: list[str] = field(default_factory=list)
    rows: int = 0
    source_files_removed: int = 0
    skipped: bool = False
    error: Optional[str] = None


@dataclass
class MigrateResult:
    source_root: Path
    dest_root: Path
    windows_processed: int = 0
    windows_compacted: int = 0
    rows_written: int = 0
    source_files_removed: int = 0
    passthrough_files_copied: int = 0
    passthrough_files_removed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class OptimizeResult:
    root: Path
    windows_checked: int = 0
    files_checked: int = 0
    files_rewritten: int = 0
    rows_rewritten: int = 0
    errors: list[str] = field(default_factory=list)


def _window_end_utc(window_dir: Path) -> Optional[datetime]:
    m = _WINDOW_DIR_RE.match(window_dir.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _source_files(window_dir: Path, kind: str) -> tuple[list[Path], list[Path]]:
    """Return (part_files, legacy_token_files) for *kind* in *window_dir*."""
    parts: list[Path] = []
    pdir = parts_dir_for(window_dir)
    if pdir.is_dir():
        for f in sorted(pdir.iterdir()):
            m = _PART_FILE_RE.match(f.name)
            if m and m.group(1) == kind and f.is_file():
                parts.append(f)
    legacy: list[Path] = []
    try:
        for f in window_dir.iterdir():
            if not f.is_file():
                continue
            m = _LEGACY_FILE_RE.match(f.name)
            if m and m.group(1) == kind:
                legacy.append(f)
    except OSError:
        pass
    return parts, legacy


def _read_table(fp: Path, kind: str) -> tuple[pa.Table, int]:
    """Read one source file; returns (table-cast-to-canonical-schema, rows)."""
    schema = schema_for(kind)
    table = pq.read_table(str(fp))
    # Tolerate column-order / missing-nullable drift from older writers: select
    # canonical columns (filling absent nullable ones), then cast.
    cols = {}
    for fld in schema:
        if fld.name in table.column_names:
            cols[fld.name] = table.column(fld.name)
        else:
            cols[fld.name] = pa.nulls(table.num_rows, type=fld.type)
    out = pa.table(cols, schema=pa.schema([(f.name, f.type) for f in schema]))
    return out.cast(schema), out.num_rows


def _token_values(table: pa.Table) -> set[str]:
    if table.num_rows <= 0 or "token_id" not in table.column_names:
        return set()
    return {str(t) for t in pc.unique(table.column("token_id")).to_pylist() if t is not None}


def _legacy_token_from_name(fp: Path) -> str:
    m = _LEGACY_FILE_RE.match(fp.name)
    return m.group(2) if m else fp.name


def _verified_bundle_from_manifest(dest_bundle: Path, manifest: dict[str, Any], kind: str) -> bool:
    entry = manifest.get(kind)
    if not isinstance(entry, dict) or not dest_bundle.exists():
        return False
    try:
        expected_rows = int(entry.get("rows", -1))
        expected_tokens = {str(t) for t in entry.get("tokens", []) if t is not None}
        md = pq.read_metadata(str(dest_bundle))
        if int(md.num_rows) != expected_rows:
            return False
        persisted_tokens = _token_values(pq.read_table(str(dest_bundle), columns=["token_id"]))
        return persisted_tokens == expected_tokens
    except Exception:
        return False


def _remove_sources(paths: list[Path], *, window_name: str) -> int:
    removed = 0
    for fp in paths:
        try:
            fp.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("compact %s: could not remove source %s: %s", window_name, fp, exc)
    return removed


def _write_legacy_bundle_stream(
    legacy_files: list[Path],
    *,
    dest_bundle: Path,
    kind: str,
) -> tuple[int, set[str]]:
    schema = schema_for(kind)
    metadata = {
        **(schema.metadata or {}),
        b"schema_version": str(BOOK_SCHEMA_VERSION).encode(),
        b"canonical_kind": str(kind).encode(),
        b"provider": b"live_ingestor",
    }
    schema = schema.with_metadata(metadata)
    dest_bundle.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_bundle.with_suffix(dest_bundle.suffix + ".tmp")
    try:
        tmp.unlink()
    except OSError:
        pass
    expected_rows = 0
    expected_tokens: set[str] = set()
    writer: pq.ParquetWriter | None = None
    buffer_tables: list[pa.Table] = []
    buffer_rows = 0

    def flush_buffer() -> None:
        nonlocal buffer_rows
        if not buffer_tables:
            return
        merged = pa.concat_tables(buffer_tables)
        merged = merged.sort_by([("token_id", "ascending"), ("observed_at_us", "ascending")])
        writer.write_table(merged, row_group_size=_ROW_GROUP_SIZE)
        buffer_tables.clear()
        buffer_rows = 0

    try:
        writer = pq.ParquetWriter(str(tmp), schema=schema, compression="zstd")
        for fp in sorted(legacy_files, key=_legacy_token_from_name):
            table, rows = _read_table(fp, kind)
            buffer_tables.append(table)
            buffer_rows += rows
            expected_rows += rows
            expected_tokens.update(_token_values(table))
            if buffer_rows >= _ROW_GROUP_SIZE:
                flush_buffer()
        flush_buffer()
    except Exception:
        if writer is not None:
            writer.close()
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    if writer is not None:
        writer.close()
    os.replace(tmp, dest_bundle)
    return expected_rows, expected_tokens


def compact_window_dir(
    window_dir: Path,
    *,
    dest_window_dir: Path | None = None,
    delete_sources: bool = True,
) -> CompactResult:
    """Compact one window directory into per-kind bundles.

    With ``dest_window_dir`` set, bundles+manifest are written THERE (the
    migration's cross-drive move); otherwise in place.  Sources (parts +
    legacy per-token files) are deleted only after verification, and only
    when ``delete_sources`` is True.
    """
    window_dir = Path(window_dir)
    dest = Path(dest_window_dir) if dest_window_dir is not None else window_dir
    result = CompactResult(window_dir=window_dir)
    manifest: dict[str, Any] = {}
    # Preserve an existing manifest's entries for kinds we don't touch.
    mp = manifest_path_for(dest)
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8")) or {}
        except Exception:
            manifest = {}

    any_work = False
    sources_to_delete: list[Path] = []
    for kind in _KINDS:
        parts, legacy = _source_files(window_dir, kind)
        existing_bundle_src = bundle_path_for(window_dir, kind)
        dest_bundle = bundle_path_for(dest, kind)
        loose_sources: list[Path] = parts + legacy
        cleanup_sources = list(loose_sources)
        if existing_bundle_src.exists() and existing_bundle_src != dest_bundle:
            cleanup_sources.append(existing_bundle_src)
        if cleanup_sources and _verified_bundle_from_manifest(dest_bundle, manifest, kind):
            any_work = True
            if delete_sources:
                result.source_files_removed += _remove_sources(cleanup_sources, window_name=window_dir.name)
            continue

        sources: list[Path] = list(loose_sources)
        # Merge-in an existing bundle when re-compacting in place with new
        # parts, or when migrating (src bundle moves to dest).
        include_existing = existing_bundle_src.exists() and (
            loose_sources or dest != window_dir
        )
        if include_existing:
            sources = [existing_bundle_src] + sources
        if not sources:
            continue
        any_work = True

        expected_rows = 0
        expected_tokens: set[str] = set()
        if legacy and not parts and not include_existing:
            try:
                expected_rows, expected_tokens = _write_legacy_bundle_stream(
                    legacy,
                    dest_bundle=dest_bundle,
                    kind=kind,
                )
            except Exception as exc:
                result.error = f"{kind}: stream bundle write failed: {exc}"
                logger.error("compact %s: %s", window_dir.name, result.error)
                continue
        else:
            tables: list[pa.Table] = []
            for fp in sources:
                try:
                    t, n = _read_table(fp, kind)
                except Exception as exc:
                    # A torn/corrupt source file must not silently vanish from the
                    # record: skip the WHOLE kind this pass and surface the error.
                    result.error = f"{kind}: unreadable source {fp.name}: {exc}"
                    logger.error("compact %s: unreadable source %s: %s", window_dir.name, fp, exc)
                    tables = []
                    break
                tables.append(t)
                expected_rows += n
            if not tables:
                continue

            merged = pa.concat_tables(tables)
            merged = merged.sort_by([("token_id", "ascending"), ("observed_at_us", "ascending")])
            if merged.num_rows != expected_rows:
                result.error = f"{kind}: merged rows {merged.num_rows} != sources {expected_rows}"
                logger.error("compact %s: %s", window_dir.name, result.error)
                continue
            expected_tokens = _token_values(merged)

            dest.mkdir(parents=True, exist_ok=True)
            # Canonical writer: schema-validates + lineage-stamps + atomic
            # (tmp + replace).  Write to a sibling tmp name first so an existing
            # bundle at dest is replaced only by a fully-written file.
            write_canonical_table(
                merged,
                dest_path=dest_bundle,
                kind=kind,
                provider="live_ingestor",
                row_group_size=_ROW_GROUP_SIZE,
            )

        # Verify the persisted artifact independently (footer + token set).
        md = pq.read_metadata(str(dest_bundle))
        if int(md.num_rows) != expected_rows:
            result.error = f"{kind}: bundle verify failed rows {md.num_rows} != {expected_rows}"
            logger.error("compact %s: %s", window_dir.name, result.error)
            continue
        persisted_tokens = _token_values(pq.read_table(str(dest_bundle), columns=["token_id"]))
        if persisted_tokens != expected_tokens:
            result.error = f"{kind}: bundle verify failed token-set mismatch"
            logger.error("compact %s: %s", window_dir.name, result.error)
            continue

        manifest[kind] = {
            "rows": expected_rows,
            "tokens": sorted(expected_tokens),
            "row_groups": int(md.num_row_groups),
            "compacted_at": datetime.now(timezone.utc).isoformat(),
        }
        result.kinds_compacted.append(kind)
        result.rows += expected_rows
        sources_to_delete.extend(fp for fp in sources if fp != dest_bundle)

    if not any_work:
        result.skipped = True
        return result

    if result.kinds_compacted:
        # Atomic manifest write (tmp + replace) AFTER bundles verified.
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=0), encoding="utf-8")
        os.replace(tmp, mp)
        if delete_sources and result.error is None:
            result.source_files_removed += _remove_sources(sources_to_delete, window_name=window_dir.name)
        # Drop a now-empty parts dir.
        pdir = parts_dir_for(window_dir)
        if delete_sources and result.error is None and pdir.is_dir():
            try:
                if not any(pdir.iterdir()):
                    pdir.rmdir()
            except OSError:
                pass
    return result


def compact_closed_windows(
    base_root: Path,
    *,
    provider: str | None = None,
    max_windows: int = 8,
    now: datetime | None = None,
) -> list[Path]:
    """Compact closed windows under ``base_root`` (oldest first).

    A window is eligible when its end time is in the past AND it still has
    parts or legacy per-token files.  Bounded by ``max_windows`` per pass so
    the recording plane's loop never takes an unbounded bite.  Returns the
    window dirs whose bundles changed (for incremental catalog registration).
    """
    now = now or datetime.now(timezone.utc)
    base_root = Path(base_root)
    if not base_root.exists():
        return []
    candidates: list[tuple[datetime, Path]] = []
    provider_dirs = (
        [base_root / provider] if provider else [d for d in base_root.iterdir() if d.is_dir()]
    )
    for prov_dir in provider_dirs:
        if not prov_dir.is_dir():
            continue
        for coin_dir in prov_dir.iterdir():
            if not coin_dir.is_dir():
                continue
            for win_dir in coin_dir.iterdir():
                if not win_dir.is_dir():
                    continue
                end = _window_end_utc(win_dir)
                if end is None or end >= now:
                    continue
                pdir = parts_dir_for(win_dir)
                has_parts = pdir.is_dir() and any(_PART_FILE_RE.match(f.name) for f in pdir.iterdir() if f.is_file())
                has_legacy = any(
                    _LEGACY_FILE_RE.match(f.name) for f in win_dir.iterdir() if f.is_file()
                )
                if has_parts or has_legacy:
                    candidates.append((end, win_dir))
    candidates.sort(key=lambda x: x[0])
    compacted: list[Path] = []
    for _end, win_dir in candidates[: max(1, int(max_windows))]:
        started = time.monotonic()
        res = compact_window_dir(win_dir)
        if res.kinds_compacted:
            compacted.append(win_dir)
            logger.info(
                "compacted window %s kinds=%s rows=%d sources_removed=%d in %.1fs",
                win_dir.name,
                ",".join(res.kinds_compacted),
                res.rows,
                res.source_files_removed,
                time.monotonic() - started,
            )
        elif res.error:
            logger.warning("compaction incomplete for %s: %s", win_dir.name, res.error)
    return compacted


def _iter_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [Path(root)]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            child = Path(entry.path)
            out.append(child)
            stack.append(child)
    return out


def _dir_has_canonical_window_files(window_dir: Path) -> bool:
    try:
        with os.scandir(window_dir) as entries:
            for entry in entries:
                name = entry.name
                try:
                    if entry.is_file(follow_symlinks=False) and (
                        name in {"snapshots.parquet", "deltas.parquet"} or _LEGACY_FILE_RE.match(name)
                    ):
                        return True
                    if entry.is_dir(follow_symlinks=False) and name == PARTS_DIRNAME:
                        with os.scandir(entry.path) as part_entries:
                            for part in part_entries:
                                if part.is_file(follow_symlinks=False) and _PART_FILE_RE.match(part.name):
                                    return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


def _canonical_window_dirs(root: Path) -> list[Path]:
    windows: list[Path] = []
    stack = [Path(root)]
    while stack:
        current = stack.pop()
        if _WINDOW_DIR_RE.match(current.name):
            if _dir_has_canonical_window_files(current):
                windows.append(current)
            continue
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    windows.sort(key=lambda p: str(p))
    return windows


def _is_under(path: Path, roots: set[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _copy_verified(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(src, tmp)
    src_stat = src.stat()
    tmp_stat = tmp.stat()
    if int(src_stat.st_size) != int(tmp_stat.st_size):
        try:
            tmp.unlink()
        except OSError:
            pass
        raise OSError(f"copy size mismatch for {src}: {tmp_stat.st_size} != {src_stat.st_size}")
    os.replace(tmp, dest)


def _remove_empty_dirs(root: Path) -> None:
    for d in sorted(_iter_dirs(root), key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            continue


def _target_row_groups(rows: int) -> int:
    return max(1, (int(rows) + _ROW_GROUP_SIZE - 1) // _ROW_GROUP_SIZE)


def _optimize_window_bundles(window_dir: Path) -> tuple[int, int, int, list[str]]:
    window_dir = Path(window_dir)
    manifest: dict[str, Any] = {}
    mp = manifest_path_for(window_dir)
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8")) or {}
        except Exception:
            manifest = {}

    files_checked = 0
    files_rewritten = 0
    rows_rewritten = 0
    errors: list[str] = []
    manifest_changed = False
    for kind in _KINDS:
        bundle = bundle_path_for(window_dir, kind)
        if not bundle.exists():
            continue
        files_checked += 1
        try:
            md = pq.read_metadata(str(bundle))
        except Exception as exc:
            errors.append(f"{bundle}: metadata read failed: {exc}")
            continue
        if int(md.num_row_groups) <= _target_row_groups(int(md.num_rows)):
            continue

        try:
            table, rows = _read_table(bundle, kind)
            expected_tokens = _token_values(table)
            table = table.sort_by([("token_id", "ascending"), ("observed_at_us", "ascending")])
            write_canonical_table(
                table,
                dest_path=bundle,
                kind=kind,
                provider="live_ingestor",
                row_group_size=_ROW_GROUP_SIZE,
            )
            rewritten_md = pq.read_metadata(str(bundle))
            if int(rewritten_md.num_rows) != rows:
                errors.append(f"{bundle}: rewrite row mismatch {rewritten_md.num_rows} != {rows}")
                continue
            persisted_tokens = _token_values(pq.read_table(str(bundle), columns=["token_id"]))
            if persisted_tokens != expected_tokens:
                errors.append(f"{bundle}: rewrite token-set mismatch")
                continue
            entry = manifest.get(kind)
            if not isinstance(entry, dict):
                entry = {}
            entry.update(
                {
                    "rows": rows,
                    "tokens": sorted(expected_tokens),
                    "row_groups": int(rewritten_md.num_row_groups),
                    "optimized_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            manifest[kind] = entry
            manifest_changed = True
            files_rewritten += 1
            rows_rewritten += rows
        except Exception as exc:
            errors.append(f"{bundle}: rewrite failed: {exc}")

    if manifest_changed:
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=0), encoding="utf-8")
        os.replace(tmp, mp)
    return files_checked, files_rewritten, rows_rewritten, errors


def optimize_bundle_row_groups(root: Path, *, workers: int = 1) -> OptimizeResult:
    root = Path(root).resolve()
    result = OptimizeResult(root=root)
    if not root.exists():
        return result

    windows = _canonical_window_dirs(root)

    def optimize_window(window_dir: Path) -> tuple[Path, int, int, int, list[str], float]:
        started = time.monotonic()
        checked, rewritten, rows, errors = _optimize_window_bundles(window_dir)
        return window_dir, checked, rewritten, rows, errors, time.monotonic() - started

    def record_window(
        window_dir: Path,
        checked: int,
        rewritten: int,
        rows: int,
        errors: list[str],
        elapsed: float,
    ) -> None:
        result.windows_checked += 1
        result.files_checked += checked
        result.files_rewritten += rewritten
        result.rows_rewritten += rows
        result.errors.extend(errors)
        if rewritten:
            logger.info(
                "optimized bundles %s files=%d rows=%d in %.1fs",
                window_dir,
                rewritten,
                rows,
                elapsed,
            )

    worker_count = max(1, int(workers))
    if worker_count == 1 or len(windows) <= 1:
        for window_dir in windows:
            record_window(*optimize_window(window_dir))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(optimize_window, window_dir): window_dir for window_dir in windows}
            for future in as_completed(future_map):
                try:
                    record_window(*future.result())
                except Exception as exc:
                    result.windows_checked += 1
                    result.errors.append(f"{future_map[future]}: {exc}")
    return result


def migrate_parquet_root(
    source_root: Path,
    dest_root: Path,
    *,
    delete_sources: bool = True,
    workers: int = 1,
) -> MigrateResult:
    """Compact and move a full parquet root to ``dest_root``.

    Canonical market-data windows are compacted into v2 bundles at the matching
    destination path. Non-canonical parquet-plane files (for example recorded
    event-bus topics with their own layout) are copied byte-for-byte with size
    verification. Source deletion happens only after each destination artifact
    is verified.
    """
    source = Path(source_root).resolve()
    dest = Path(dest_root).resolve()
    result = MigrateResult(source_root=source, dest_root=dest)
    if not source.exists():
        return result
    dest.mkdir(parents=True, exist_ok=True)

    windows = _canonical_window_dirs(source)
    window_roots = {w.resolve() for w in windows}

    def migrate_window(window_dir: Path) -> tuple[Path, CompactResult, float]:
        rel = window_dir.relative_to(source)
        started = time.monotonic()
        compacted = compact_window_dir(
            window_dir,
            dest_window_dir=dest / rel,
            delete_sources=delete_sources,
        )
        return window_dir, compacted, time.monotonic() - started

    def record_window(window_dir: Path, compacted: CompactResult, elapsed: float) -> None:
        result.windows_processed += 1
        result.rows_written += int(compacted.rows)
        result.source_files_removed += int(compacted.source_files_removed)
        if compacted.kinds_compacted:
            result.windows_compacted += 1
            logger.info(
                "migrated compacted window %s kinds=%s rows=%d removed=%d in %.1fs",
                window_dir,
                ",".join(compacted.kinds_compacted),
                compacted.rows,
                compacted.source_files_removed,
                elapsed,
            )
        if compacted.error:
            result.errors.append(f"{window_dir}: {compacted.error}")

    worker_count = max(1, int(workers))
    if worker_count == 1 or len(windows) <= 1:
        for window_dir in windows:
            record_window(*migrate_window(window_dir))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(migrate_window, window_dir): window_dir for window_dir in windows}
            for future in as_completed(future_map):
                try:
                    record_window(*future.result())
                except Exception as exc:
                    result.windows_processed += 1
                    result.errors.append(f"{future_map[future]}: {exc}")

    for fp in sorted(p for p in source.rglob("*") if p.is_file()):
        if _is_under(fp, window_roots):
            continue
        rel = fp.relative_to(source)
        target = dest / rel
        try:
            _copy_verified(fp, target)
            result.passthrough_files_copied += 1
            if delete_sources:
                fp.unlink()
                result.passthrough_files_removed += 1
        except Exception as exc:
            result.errors.append(f"{fp}: {exc}")

    if delete_sources:
        _remove_empty_dirs(source)
    return result


__all__ = [
    "CompactResult",
    "MigrateResult",
    "OptimizeResult",
    "compact_window_dir",
    "compact_closed_windows",
    "migrate_parquet_root",
    "optimize_bundle_row_groups",
]
