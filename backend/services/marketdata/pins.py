"""Dataset pins — protect the exact parquet a running backtest reads from
pruning, cross-process and without a DB.

The live-ingestor sink and the recorded-event-bus pruner delete old data on
their own schedules (retention + size caps), in different processes from the
backtest worker. A backtest pins the window directories its DatasetSnapshot
references; the pruners skip any path under an active (non-expired) pin, so a
long replay near the retention edge can't have data deleted out from under it
mid-run. Pins are tiny JSON files under ``{parquet_root}/.pins/{hash}.json``
with a TTL, so a crashed run can never leak a permanent pin.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

from services.external_data.parquet_schema import parquet_root

logger = logging.getLogger(__name__)

_PINS_DIRNAME = ".pins"
_DEFAULT_TTL_SECONDS = 3600  # 1h — generous upper bound on a single backtest


def _pins_dir() -> Path:
    d = parquet_root() / _PINS_DIRNAME
    return d


def _norm(p: str | Path) -> str:
    return str(p).replace("\\", "/").rstrip("/")


def pin_paths(content_hash: str, paths: Iterable[str | Path], *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> Path:
    """Write a pin protecting ``paths`` (files or dirs) for ``ttl_seconds``.

    Returns the pin file path (pass its stem / the same content_hash to
    :func:`release_pin`). Best-effort: failures are logged, never raised — a
    pin is a safety nicety, not a correctness requirement of the backtest.
    """
    safe_hash = "".join(c for c in str(content_hash) if c.isalnum() or c in "-_") or "pin"
    pin_file = _pins_dir() / f"{safe_hash}.json"
    try:
        _pins_dir().mkdir(parents=True, exist_ok=True)
        payload = {
            "content_hash": str(content_hash),
            "paths": sorted({_norm(p) for p in paths}),
            "expires_at_us": int((time.time() + max(1, int(ttl_seconds))) * 1_000_000),
        }
        tmp = pin_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(pin_file)
    except Exception as exc:  # noqa: BLE001
        logger.debug("pin_paths: failed to write pin %s: %s", safe_hash, exc)
    return pin_file


def release_pin(content_hash: str) -> None:
    safe_hash = "".join(c for c in str(content_hash) if c.isalnum() or c in "-_") or "pin"
    try:
        (_pins_dir() / f"{safe_hash}.json").unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("release_pin: failed to remove pin %s: %s", safe_hash, exc)


def active_pinned_paths() -> set[str]:
    """Union of all paths under non-expired pins (normalized, no trailing /).

    Garbage-collects expired pin files as a side effect. Returns an empty set
    when there are no pins (the common case — zero overhead for pruners).
    """
    out: set[str] = set()
    d = _pins_dir()
    if not d.exists():
        return out
    now_us = int(time.time() * 1_000_000)
    for pf in d.glob("*.json"):
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(data.get("expires_at_us", 0)) < now_us:
            try:
                pf.unlink(missing_ok=True)  # GC expired
            except Exception:
                pass
            continue
        for p in data.get("paths", []):
            out.add(_norm(p))
    return out


def is_path_pinned(path: str | Path, pinned: set[str] | None = None) -> bool:
    """True if ``path`` (or any ancestor) is under an active pin."""
    pinned = active_pinned_paths() if pinned is None else pinned
    if not pinned:
        return False
    norm = _norm(path)
    if norm in pinned:
        return True
    # a pinned ancestor dir protects everything under it
    return any(norm == p or norm.startswith(p + "/") for p in pinned)


__all__ = ["pin_paths", "release_pin", "active_pinned_paths", "is_path_pinned"]
