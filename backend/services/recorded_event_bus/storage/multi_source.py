"""Universal multi-source read AND write dispatch.

Every bus topic in the catalog declares 1..N **sources**.  A source
is a tuple of (storage kind, source-specific config) — postgres table
adapter, parquet path, external parquet root, etc.  The topic's
identity is its **payload shape contract** (declared in the catalog
row's title / description / schema_version / payload_schema); the
sources are an implementation detail of where the bytes happen to live.

This module is the single replay primitive used by every topic.  The
old design treated single-source vs. multi-source asymmetrically
(``storage_kind = sql_table | parquet | external_parquet`` for one
source, ``storage_kind = federated`` for many).  That was the wrong
factoring — it leaked the "where does the data live" detail into the
topic identity.

The universal model:

  storage_uri JSON shape (every topic uses this, length 1 or N):

      {
          "sources": [
              {"kind": "sql_table", "adapter": "...", "table": "...", "provider": "..."},
              {"kind": "external_parquet", "uri": "/data/.../telonex/btc"},
              {"kind": "parquet", "uri": "/data/parquet/recorded_event_bus/crypto.update.dispatch"}
          ]
      }

For backward compatibility with the older single-source shape (where
storage_uri was the raw string for parquet topics, or a JSON dict
without a ``sources`` array for sql_table topics) the loader detects
and adapts — see :func:`_resolve_sources_from_spec`.

The replayer:
  * Resolves the source list (1..N).
  * Builds an ephemeral TopicSpec per source so the existing adapter
    dispatch (parquet_replayer / external_parquet_replayer /
    sql_adapters.get_sql_adapter) works unchanged.
  * For length=1, delegates directly to the one source's reader (no
    heap-merge overhead).
  * For length>1, heap-merges by ``window.time_field`` so the
    strategy sees one ordered stream.

Strategies remain entirely storage-agnostic: they ask for a topic,
the bus delivers the union of every source registered for it.
"""
from __future__ import annotations

import heapq
import json
import logging
from dataclasses import replace
from typing import Any, AsyncIterator, Optional

from services.recorded_event_bus.catalog import TopicSpec
from services.recorded_event_bus.envelope import RecordedEvent

logger = logging.getLogger(__name__)


# ── Source resolution ────────────────────────────────────────────────


def _resolve_sources_from_spec(spec: TopicSpec) -> list[dict[str, Any]]:
    """Parse a topic's source list.

    Accepts three shapes, in order of preference:

    1. ``storage_uri`` is JSON with a ``sources`` array — the
       canonical multi-source shape.  Return it as-is.

    2. ``storage_uri`` is JSON without ``sources`` (legacy single-
       source sql_table or parquet config).  Wrap in a single-element
       sources list inferring ``kind`` from ``spec.storage_kind``.

    3. ``storage_uri`` is a plain filesystem path (legacy single-
       source parquet / external_parquet).  Wrap in a single-element
       sources list with ``kind=spec.storage_kind`` and ``uri=path``.

    All three shapes coexist during the transition — operators can
    upgrade rows lazily and existing seed data keeps working.
    """
    if not spec.storage_uri:
        return []

    # Try JSON first.
    raw = spec.storage_uri
    try:
        cfg = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        cfg = None

    if isinstance(cfg, dict) and isinstance(cfg.get("sources"), list):
        # Shape 1: canonical multi-source.
        return [s for s in cfg["sources"] if isinstance(s, dict)]

    if isinstance(cfg, dict):
        # Shape 2: legacy single-source config (sql_table mainly).
        # Promote to a one-element sources list.
        kind = spec.storage_kind
        if kind == "federated":
            # An old federation row without ``sources`` is malformed.
            logger.warning(
                "topic %s storage_kind=federated but no 'sources' key in storage_uri",
                spec.slug,
            )
            return []
        return [{"kind": kind, **cfg}]

    # Shape 3: legacy plain filesystem path (parquet / external_parquet).
    if spec.storage_kind in ("parquet", "external_parquet"):
        return [{"kind": spec.storage_kind, "uri": raw}]

    logger.warning(
        "topic %s has unparseable storage_uri (kind=%s): %r",
        spec.slug, spec.storage_kind, raw,
    )
    return []


def _build_source_spec(parent: TopicSpec, source_cfg: dict[str, Any]) -> Optional[TopicSpec]:
    """Construct an ephemeral TopicSpec for one source.

    Inherits parent metadata (slug, schema_version, payload_schema)
    — projected envelopes carry the PARENT topic name — and replaces
    only ``storage_kind`` + ``storage_uri`` with the source-specific
    values.  This lets us reuse the existing per-kind replayers
    without modification.
    """
    kind = source_cfg.get("kind")
    if not kind:
        logger.warning("topic %s source missing 'kind': %r", parent.slug, source_cfg)
        return None

    if kind == "sql_table":
        # SQL adapter reads ``adapter`` + ``table`` + optional ``provider``
        # from storage_uri JSON.  Reconstruct that shape.
        if not source_cfg.get("adapter"):
            logger.warning(
                "topic %s sql_table source missing 'adapter': %r",
                parent.slug, source_cfg,
            )
            return None
        sub_uri = {
            "adapter": source_cfg["adapter"],
            "table": source_cfg.get("table"),
        }
        if "provider" in source_cfg:
            sub_uri["provider"] = source_cfg["provider"]
        return replace(parent, storage_kind="sql_table", storage_uri=json.dumps(sub_uri))

    if kind in ("parquet", "external_parquet"):
        uri = source_cfg.get("uri")
        if not uri:
            logger.warning(
                "topic %s %s source missing 'uri': %r",
                parent.slug, kind, source_cfg,
            )
            return None
        return replace(parent, storage_kind=kind, storage_uri=uri)

    if kind == "memory":
        return None  # memory has no replay

    logger.warning(
        "topic %s: unknown source kind %r",
        parent.slug, kind,
    )
    return None


# ── Reader dispatch ──────────────────────────────────────────────────


# ── Writer-side resolution ──────────────────────────────────────────


def resolve_writable_parquet_source(spec: TopicSpec) -> Optional[TopicSpec]:
    """Pick the writable parquet source from a topic's sources list.

    Used by ``bus.publish`` to know where to land a new envelope.
    Only ``parquet`` sources are writable today (SQL sources are
    owned by their existing recorders; external_parquet sources are
    operator-managed import outputs).  Returns an ephemeral
    ``TopicSpec`` shaped like a single-source parquet topic so the
    existing parquet_writer (which expects a plain path uri) works
    unchanged.

    Returns None if the topic has no writable parquet source — the
    bus then knows publish is fan-out-only (live subscribers receive
    the envelope; nothing is persisted via the bus, but the
    underlying recorder may still be writing through its own path).
    """
    sources = _resolve_sources_from_spec(spec)
    for src in sources:
        if src.get("kind") == "parquet":
            built = _build_source_spec(spec, src)
            if built is not None:
                return built
    return None


def _open_source_stream(source_spec: TopicSpec, window) -> AsyncIterator[RecordedEvent]:
    """Route one source's spec to its kind-specific reader."""
    if source_spec.storage_kind == "sql_table":
        from services.recorded_event_bus.storage.sql_adapters import get_sql_adapter
        return get_sql_adapter(source_spec)(source_spec, window)
    if source_spec.storage_kind == "parquet":
        from services.recorded_event_bus.storage.parquet_backend import parquet_replayer
        return parquet_replayer(source_spec, window)
    if source_spec.storage_kind == "external_parquet":
        from services.recorded_event_bus.storage.external_parquet import external_parquet_replayer
        return external_parquet_replayer(source_spec, window)
    raise RuntimeError(f"unsupported source kind {source_spec.storage_kind}")


# ── The universal replayer ──────────────────────────────────────────


async def multi_source_replayer(
    spec: TopicSpec,
    window,
) -> AsyncIterator[RecordedEvent]:
    """Replay one topic across every source registered for it.

    For single-source topics, this is a near-zero-cost passthrough
    (one source, no heap-merge — yields straight from the adapter).
    For multi-source topics, heap-merges by ``window.time_field``.
    """
    sources = _resolve_sources_from_spec(spec)
    if not sources:
        return

    # Build the per-source streams.
    source_specs = [_build_source_spec(spec, s) for s in sources]
    source_specs = [s for s in source_specs if s is not None]
    if not source_specs:
        return

    if len(source_specs) == 1:
        # Single-source fast path — skip heap setup entirely.
        try:
            async for ev in _open_source_stream(source_specs[0], window):
                yield ev
        except Exception:
            logger.exception(
                "topic %s: source 0 (kind=%s) replay failed",
                spec.slug, source_specs[0].storage_kind,
            )
        return

    # Multi-source heap-merge.  Same algorithm as bus.replay across
    # topics — prime each iterator's first envelope, yield-min /
    # refill until exhausted.
    iters: list[AsyncIterator[RecordedEvent]] = []
    for i, sub in enumerate(source_specs):
        try:
            iters.append(_open_source_stream(sub, window))
        except Exception:
            logger.exception(
                "topic %s: source %d (kind=%s) failed to open",
                spec.slug, i, sub.storage_kind,
            )

    if not iters:
        return

    head: list[Optional[RecordedEvent]] = []
    heap: list[tuple[Any, int]] = []
    time_attr = window.time_field

    for idx, it in enumerate(iters):
        try:
            first = await it.__anext__()
        except StopAsyncIteration:
            first = None
        except Exception:
            logger.exception("topic %s: source %d first read failed", spec.slug, idx)
            first = None
        head.append(first)
        if first is not None:
            heapq.heappush(heap, (getattr(first, time_attr), idx))

    while heap:
        _ts, idx = heapq.heappop(heap)
        ev = head[idx]
        if ev is None:
            continue
        yield ev
        try:
            nxt = await iters[idx].__anext__()
            head[idx] = nxt
            heapq.heappush(heap, (getattr(nxt, time_attr), idx))
        except StopAsyncIteration:
            head[idx] = None
        except Exception:
            logger.exception("topic %s: source %d mid-read failed", spec.slug, idx)
            head[idx] = None
