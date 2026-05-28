"""External-parquet storage adapter.

The bus's native parquet writer (``parquet_backend.py``) controls its
own file layout — ``{storage_uri}/{entity_id}/{YYYY-MM-DD}/events__*.parquet``
— because it owns both the writer and the reader.  But several
existing storage paths write parquet in DIFFERENT layouts the bus
doesn't own:

  * Telonex imports → ``{root}/telonex/{coin}/{window_iso}/{kind}__{token}.parquet``
    where each file is one (asset_id, window) in SNAPSHOT_SCHEMA shape
  * Polybacktest parquet exports → same canonical SNAPSHOT_SCHEMA
    layout (managed by ``parquet_schema.parquet_path_for``)
  * Operator-dropped manual datasets → same canonical layout

This adapter is the bus's reader for that **canonical external layout**.
It does NOT write — external paths own their own writers — and it
projects SNAPSHOT_SCHEMA rows into ``RecordedEvent`` envelopes on the fly
so backtest replay sees a unified stream regardless of which side
created the files.

Layout discovery:

    {storage_uri}/                           # e.g. data/parquet/telonex/btc
      {window_iso}/                          # YYYYMMDDTHHMMSS__YYYYMMDDTHHMMSS
        snapshots__{token_id}.parquet        # canonical SNAPSHOT_SCHEMA file

Each ``snapshots__*.parquet`` is a time-sorted, per-token sequence of
L2 book observations.  The adapter heap-merges across files in time
order so a multi-token replay (e.g. both outcomes of a binary market)
yields events interleaved by ``observed_at_us``.

Entity filter: when the topic's ``entity_filter[topic]`` is set, only
files whose token suffix matches one of the entities is opened — same
selectivity the bus-native parquet adapter gives.

Future direction: when a topic outgrows the canonical external layout
(e.g. operator wants payload-versioned schemas), the operator can
``copy/migrate`` the data into the bus-native layout via the parquet
writer and re-register the topic with ``storage_kind='parquet'``.
That migration is one ``pq.read_table → bus.publish`` loop away.
"""
from __future__ import annotations

import asyncio
import heapq
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pyarrow.parquet as pq

from services.recorded_event_bus.catalog import TopicSpec
from services.recorded_event_bus.envelope import RecordedEvent

logger = logging.getLogger(__name__)


# The canonical filename pattern produced by ``parquet_path_for``:
# ``{kind}__{safe_token_segment}.parquet``.  We pull the token from
# the suffix so entity-filter matching works without opening the file.
_FILENAME_RE = re.compile(r"^([a-zA-Z_]+)__(.+)\.parquet$")


async def external_parquet_replayer(
    spec: TopicSpec,
    window,  # ReplayWindow — forward ref
) -> AsyncIterator[RecordedEvent]:
    """Yield RecordedEvent envelopes from canonical-external-layout
    parquet, time-merged across files within the window.
    """
    from services.recorded_event_bus.bus import ReplayWindow

    if not isinstance(window, ReplayWindow):
        raise TypeError("window must be ReplayWindow")
    if not spec.storage_uri:
        return

    base = Path(spec.storage_uri)
    if not base.exists():
        return

    # The parquet "kind" this topic projects.  ``*.delta`` topics read
    # ``deltas__`` files; every other book topic reads ``snapshots__``.
    expected_kind = "deltas" if spec.slug.endswith(".delta") else "snapshots"

    # Entity (token_id) filter from the window.
    entity_set: Optional[frozenset[str]] = None
    if window.entity_filter is not None:
        entity_set = window.entity_filter.get(spec.slug)

    # Discover candidate files.  The canonical external layout is
    #     ``{root}/{provider}/{coin}/{window_dir}/{kind}__{token}.parquet``
    # but operators point ``storage_uri`` at any level:
    #     ``{root}``                                — federated over everything
    #     ``{root}/telonex``                        — one provider, all coins
    #     ``{root}/telonex/btc``                    — one coin
    #     ``{root}/telonex/btc/<window_dir>``       — one window (rare)
    # ``_find_window_dirs`` walks recursively, identifying window dirs
    # by name shape (``YYYYMMDDTHHMMSS__YYYYMMDDTHHMMSS``) regardless
    # of depth.  Bounded recursion (max 5 levels) prevents accidents.
    files: list[tuple[Path, str]] = []  # (path, entity_id_token)
    for window_dir in _find_window_dirs(base, max_depth=5):
        try:
            window_start_dt, window_end_dt = _parse_window_dir(window_dir.name)
        except ValueError:
            continue
        # Quick prune by window overlap.
        win_start_us = int(window_start_dt.timestamp() * 1_000_000)
        win_end_us = int(window_end_dt.timestamp() * 1_000_000)
        if win_end_us < window.start_us or win_start_us > window.end_us:
            continue
        for fp in sorted(window_dir.glob("*.parquet")):
            m = _FILENAME_RE.match(fp.name)
            if m is None:
                continue
            file_kind, token = m.groups()
            # Kind filter — a window dir can hold BOTH ``snapshots__`` and
            # ``deltas__`` files (the live ingestor writes both side by
            # side).  Only read the kind this topic projects, or we'd try
            # to parse delta files as SNAPSHOT_SCHEMA.  ``*.delta`` topics
            # read ``deltas``; everything else reads ``snapshots``.
            if file_kind != expected_kind:
                continue
            if entity_set is not None and token not in entity_set:
                continue
            files.append((fp, token))

    if not files:
        return

    # Group files by token so per-token files (multiple window dirs
    # for the same token) feed a single time-ordered iterator.
    by_token: dict[str, list[Path]] = {}
    for fp, token in files:
        by_token.setdefault(token, []).append(fp)

    # Build one async iterator per token — each is monotone in
    # observed_at because writer guarantees per-file sort AND each
    # window_dir's name encodes a non-overlapping time slice.
    _reader = _read_delta_rows if expected_kind == "deltas" else _read_snapshot_rows

    async def _token_stream(token: str, paths: list[Path]) -> AsyncIterator[RecordedEvent]:
        for fp in paths:
            try:
                rows = await asyncio.to_thread(_reader, fp, token, window, spec)
            except Exception:
                logger.exception("external_parquet: failed to read %s", fp)
                continue
            for ev in rows:
                yield ev

    iterators: dict[str, AsyncIterator[RecordedEvent]] = {
        tok: _token_stream(tok, sorted(paths)) for tok, paths in by_token.items()
    }
    head: dict[str, Optional[RecordedEvent]] = {}
    heap: list[tuple[Any, str]] = []
    time_attr = window.time_field

    for key, it in iterators.items():
        try:
            first = await it.__anext__()
        except StopAsyncIteration:
            first = None
        head[key] = first
        if first is not None:
            heapq.heappush(heap, (getattr(first, time_attr), key))

    while heap:
        _ts, key = heapq.heappop(heap)
        ev = head[key]
        if ev is None:
            continue
        yield ev
        try:
            nxt = await iterators[key].__anext__()
            head[key] = nxt
            heapq.heappush(heap, (getattr(nxt, time_attr), key))
        except StopAsyncIteration:
            head[key] = None


def _parse_window_dir(name: str) -> tuple[datetime, datetime]:
    """Parse ``YYYYMMDDTHHMMSS__YYYYMMDDTHHMMSS`` into UTC datetimes."""
    parts = name.split("__")
    if len(parts) != 2:
        raise ValueError(name)
    start = datetime.strptime(parts[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    end = datetime.strptime(parts[1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return start, end


_WINDOW_DIR_RE = re.compile(r"^\d{8}T\d{6}__\d{8}T\d{6}$")


def _infer_provider_from_path(fp: Path) -> str:
    """Best-effort provider attribution from the file's path.

    The canonical layout is ``{root}/{provider}/{coin}/{window}/file.parquet``
    so the provider is two levels above the window dir.  When the
    layout is shallower (operator pointed straight at a window) or
    deeper, returns 'external' as a safe fallback rather than guessing
    a name that might mislead a downstream consumer."""
    try:
        parts = fp.parts
        # parts[-1] = file, parts[-2] = window_dir, parts[-3] = coin,
        # parts[-4] = provider
        if len(parts) >= 4 and _WINDOW_DIR_RE.match(parts[-2]):
            return parts[-4]
    except Exception:
        pass
    return "external"


def _find_window_dirs(root: Path, *, max_depth: int = 5) -> list[Path]:
    """Walk recursively up to ``max_depth`` and return every directory
    whose name matches the window-dir shape.

    Lets the operator point ``storage_uri`` at any level of the
    canonical external layout — root, provider, coin, or specific
    window — without re-wiring.  Returns a sorted list for
    deterministic file ordering (writes from the same producer arrive
    in name-order, so this preserves time order across siblings).
    """
    out: list[Path] = []

    def _walk(p: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(p.iterdir())
        except OSError:
            return
        for child in entries:
            if not child.is_dir():
                continue
            if _WINDOW_DIR_RE.match(child.name):
                out.append(child)
            else:
                _walk(child, depth + 1)

    _walk(root, 0)
    out.sort(key=lambda p: p.name)
    return out


def _read_snapshot_rows(
    fp: Path,
    token: str,
    window,
    spec: TopicSpec,
) -> list[RecordedEvent]:
    """Read one canonical SNAPSHOT_SCHEMA file and project rows into
    RecordedEvent envelopes filtered to the window.

    Per-row payload mirrors what the live SQL adapter for
    ``polymarket.book.snapshot`` returns — best_bid / best_ask /
    spread_bps / bids_price / bids_size / asks_price / asks_size /
    trade fields.  Strategies that subscribe to the topic get the
    same payload shape whether the data is sourced from SQL, live
    parquet, or external import.
    """
    try:
        table = pq.read_table(str(fp))
    except Exception:
        return []
    cols = table.to_pydict()
    n = len(cols.get("observed_at_us", []))
    if n == 0:
        return []
    times = cols["observed_at_us"]
    out: list[RecordedEvent] = []
    for i in range(n):
        t = int(times[i])
        if t < window.start_us or t >= window.end_us:
            continue
        # Project the canonical SNAPSHOT_SCHEMA columns into the
        # envelope payload.  Use the same field names live + SQL
        # adapters use so strategies don't branch on source.
        payload: dict[str, Any] = {
            "best_bid":   cols["best_bid"][i],
            "best_ask":   cols["best_ask"][i],
            "spread_bps": cols["spread_bps"][i],
            "bids_price": list(cols["bids_price"][i] or []),
            "bids_size":  list(cols["bids_size"][i] or []),
            "asks_price": list(cols["asks_price"][i] or []),
            "asks_size":  list(cols["asks_size"][i] or []),
            "trade_price": cols["trade_price"][i],
            "trade_size":  cols["trade_size"][i],
            "trade_side":  cols["trade_side"][i],
            # Provider attribution: derived from the file path so the
            # federated topic's payload shape matches what SQL adapters
            # emit (``payload['provider']`` always present).  Walks the
            # path upward — the immediate parent is ``window_dir``, two
            # levels up is the coin (btc/eth/...), three is the
            # provider (telonex / polybacktest / vendor_acme / ...).
            "provider": _infer_provider_from_path(fp),
        }
        seq = cols.get("sequence", [None] * n)[i]
        try:
            ev = RecordedEvent(
                topic=spec.slug,
                entity_id=cols["token_id"][i] or token,  # token_id column is authoritative
                observed_at_us=t,
                ingested_at_us=t,  # canonical layout has no separate ingest_at
                source="external_import",
                sequence=int(seq) if seq is not None else None,
                schema_version=spec.schema_version,
                payload=payload,
            )
        except Exception:
            continue
        out.append(ev)
    out.sort(key=lambda e: (e.observed_at_us, e.sequence or 0))
    return out


def _read_delta_rows(
    fp: Path,
    token: str,
    window,
    spec: TopicSpec,
) -> list[RecordedEvent]:
    """Read one canonical DELTA_SCHEMA file (``deltas__*.parquet``) and
    project rows into RecordedEvent envelopes — the parquet equivalent of
    the SQL ``BookDeltaEvent`` adapter so ``polymarket.book.delta`` replays
    off parquet just like the snapshot topic."""
    try:
        table = pq.read_table(str(fp))
    except Exception:
        return []
    cols = table.to_pydict()
    n = len(cols.get("observed_at_us", []))
    if n == 0:
        return []
    times = cols["observed_at_us"]
    out: list[RecordedEvent] = []
    for i in range(n):
        t = int(times[i])
        if t < window.start_us or t >= window.end_us:
            continue
        payload: dict[str, Any] = {
            "event_type": cols.get("event_type", [None] * n)[i],
            "side": cols.get("side", [None] * n)[i],
            "price": cols.get("price", [None] * n)[i],
            "trade_size": cols.get("trade_size", [None] * n)[i],
            "cancel_size": cols.get("cancel_size", [None] * n)[i],
            "queue_depth_before": cols.get("queue_depth_before", [None] * n)[i],
            "queue_depth_after": cols.get("queue_depth_after", [None] * n)[i],
            "spread_bps_at_event": cols.get("spread_bps_at_event", [None] * n)[i],
            "provider": _infer_provider_from_path(fp),
        }
        seq = cols.get("sequence", [None] * n)[i]
        try:
            ev = RecordedEvent(
                topic=spec.slug,
                entity_id=cols["token_id"][i] or token,
                observed_at_us=t,
                ingested_at_us=t,
                source="external_import",
                sequence=int(seq) if seq is not None else None,
                schema_version=spec.schema_version,
                payload=payload,
            )
        except Exception:
            continue
        out.append(ev)
    out.sort(key=lambda e: (e.observed_at_us, e.sequence or 0))
    return out
