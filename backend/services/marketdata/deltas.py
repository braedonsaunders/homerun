"""Canonical book-delta reading + aggregation for the unified layer.

Book deltas (trade / cancel events) live in ``DELTA_SCHEMA`` parquet
(``deltas__{token}.parquet``) written by the live ingestor — the same
canonical plane the snapshots use. This module reads them the way
``book.py`` reads snapshots, and provides the trade-vs-cancel aggregate the
fill-model calibration needs (it previously queried the now-dropped
``book_delta_events`` SQL table).

The aggregate is computed straight from the parquet footers' columns (no data
scan beyond the four needed columns), bounded to the requested window via the
window-dir name and a per-row timestamp filter.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from services.external_data.parquet_schema import bundle_path_for, parquet_roots

logger = logging.getLogger(__name__)

_WINDOW_DIR_RE = re.compile(r"^(?P<start>\d{8}T\d{6})__(?P<end>\d{8}T\d{6})$")


def _dt_to_us(dt: datetime) -> int:
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return int(aware.astimezone(timezone.utc).timestamp() * 1_000_000)


def _window_overlaps(dir_name: str, start_us: int, end_us: int) -> bool:
    """True if a ``YYYYMMDDTHHMMSS__YYYYMMDDTHHMMSS`` dir overlaps the window.

    Unparseable names return True (don't skip data we can't bound cheaply).
    """
    m = _WINDOW_DIR_RE.match(dir_name)
    if not m:
        return True
    try:
        w_start = int(datetime.strptime(m.group("start"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
        w_end = int(datetime.strptime(m.group("end"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
    except ValueError:
        return True
    return w_start <= end_us and w_end >= start_us


def _iter_delta_files(start_us: int, end_us: int, providers: Optional[set[str]]) -> Iterable[Path]:
    """Yield delta parquet files whose window-dir overlaps [start,end]."""
    for root in parquet_roots():
        root = Path(root)
        if not root.exists():
            continue
        for provider_dir in root.iterdir():
            if not provider_dir.is_dir():
                continue
            if providers is not None and provider_dir.name not in providers:
                continue
            # provider/coin/window/deltas__*.parquet
            for coin_dir in provider_dir.iterdir():
                if not coin_dir.is_dir():
                    continue
                for window_dir in coin_dir.iterdir():
                    if not window_dir.is_dir() or not _window_overlaps(window_dir.name, start_us, end_us):
                        continue
                    bundle = bundle_path_for(window_dir, "deltas")
                    if bundle.exists():
                        yield bundle
                        continue
                    for f in window_dir.glob("deltas__*.parquet"):
                        yield f


def aggregate_delta_events(
    *,
    start: datetime,
    end: datetime,
    providers: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Aggregate trade-vs-cancel delta events over ``[start, end]`` from the
    canonical parquet plane.

    Returns ``{n_trade, n_cancel, trade_size_sum, cancel_size_sum, total,
    files_scanned}`` — the inputs the fill-model calibration needs.
    """
    import pyarrow.parquet as pq

    start_us = _dt_to_us(start)
    end_us = _dt_to_us(end)
    provider_set = {str(p) for p in providers} if providers is not None else None

    n_trade = n_cancel = 0
    trade_sz = cancel_sz = 0.0
    files = 0
    for fp in _iter_delta_files(start_us, end_us, provider_set):
        try:
            t = pq.read_table(str(fp), columns=["observed_at_us", "event_type", "trade_size", "cancel_size"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("aggregate_delta_events: unreadable %s: %s", fp, exc)
            continue
        files += 1
        obs = t.column("observed_at_us").to_pylist()
        etype = t.column("event_type").to_pylist()
        tsz = t.column("trade_size").to_pylist()
        csz = t.column("cancel_size").to_pylist()
        for i in range(len(obs)):
            o = obs[i]
            if o is None or o < start_us or o > end_us:
                continue
            et = etype[i]
            if et == "trade":
                n_trade += 1
                trade_sz += float(tsz[i] or 0.0)
            elif et == "cancel":
                n_cancel += 1
                cancel_sz += float(csz[i] or 0.0)
    return {
        "n_trade": n_trade,
        "n_cancel": n_cancel,
        "trade_size_sum": trade_sz,
        "cancel_size_sum": cancel_sz,
        "total": n_trade + n_cancel,
        "files_scanned": files,
    }


__all__ = ["aggregate_delta_events"]
