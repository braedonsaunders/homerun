from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_ACTIONABLE_EVENT_TYPES = frozenset(
    {
        "upsert_insert",
        "upsert_update",
        "upsert_reactivated",
        "reverse_entry",
    }
)
_LANE_GENERAL = "general"
_LANE_CRYPTO = "crypto"
_LANES = (_LANE_GENERAL, _LANE_CRYPTO)
# Depth at which a producer compacts the lane's backlog in place (merge by
# source, loss-free for signal ids) instead of letting it grow unbounded. A live
# consumer fully drains its lane every cycle (see wait_for_signal_batch), so this
# only trips when the trader cycle has fallen behind; it caps memory and bounds
# emit->wake latency to ~one cycle rather than an ever-growing backlog. Any wake
# trigger collapsed here is still authoritative-readable via the trader cycle's
# unconsumed-signal DB prefetch, so compaction never drops a tradable signal.
_QUEUE_COMPACT_THRESHOLD = 512
_COMPACT_LOG_INTERVAL_S = 30.0
_last_compact_log_at: dict[str, float] = {}
_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {
    lane: asyncio.Queue()
    for lane in _LANES
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_signal_ids(raw_ids: Any, *, limit: int = 5000) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_ids:
        signal_id = str(raw or "").strip()
        if not signal_id or signal_id in seen:
            continue
        seen.add(signal_id)
        out.append(signal_id)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _normalize_source(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_event_type(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_actionable_event(event_type: str, trigger: str) -> bool:
    if event_type in _ACTIONABLE_EVENT_TYPES:
        return True
    return event_type == "" and trigger == "strategy_signal_bridge"


def _default_lane_for_source(source: str) -> str:
    if _normalize_source(source) == _LANE_CRYPTO:
        return _LANE_CRYPTO
    return _LANE_GENERAL


def _coalesce_batches(batches: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not batches:
        return None
    source_signal_ids: dict[str, list[str]] = {}
    source_seen_ids: dict[str, set[str]] = {}
    source_signal_snapshots: dict[str, dict[str, dict[str, Any]]] = {}
    batch_ids: list[str] = []
    batch_event_types: list[str] = []
    for payload in batches:
        batch_id = str(payload.get("batch_id") or "").strip()
        if batch_id:
            batch_ids.append(batch_id)
        batch_event_type = str(payload.get("batch_event_type") or "").strip().lower()
        if batch_event_type:
            batch_event_types.append(batch_event_type)
        source_key = _normalize_source(payload.get("source")) or "__all__"
        source_signal_ids.setdefault(source_key, [])
        source_seen_ids.setdefault(source_key, set())
        source_signal_snapshots.setdefault(source_key, {})
        raw_snapshots = payload.get("signal_snapshots")
        snapshot_map = raw_snapshots if isinstance(raw_snapshots, dict) else {}
        for raw_signal_id in payload.get("signal_ids") or []:
            signal_id = str(raw_signal_id or "").strip()
            if not signal_id or signal_id in source_seen_ids[source_key]:
                continue
            source_seen_ids[source_key].add(signal_id)
            source_signal_ids[source_key].append(signal_id)
            snapshot = snapshot_map.get(signal_id)
            if isinstance(snapshot, dict):
                source_signal_snapshots[source_key][signal_id] = dict(snapshot)
    non_global_sources = sorted(source for source in source_signal_ids.keys() if source != "__all__")
    trigger_source = non_global_sources[0] if len(non_global_sources) == 1 and "__all__" not in source_signal_ids else ""
    return {
        "event_type": "runtime_signal_batch",
        "source": trigger_source,
        "source_signal_ids": source_signal_ids,
        "source_signal_snapshots": source_signal_snapshots,
        "batch_ids": batch_ids,
        "batch_event_types": batch_event_types,
    }


def _merge_same_source_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse input batches to at most one per source.

    Loss-free for signal ids and snapshots; used to bound a backlogged lane in
    place. The merged payload keeps the newest batch identity/timestamp.
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for payload in payloads:
        source_key = _normalize_source(payload.get("source")) or "__all__"
        existing = merged.get(source_key)
        if existing is None:
            clone = dict(payload)
            clone["signal_ids"] = list(payload.get("signal_ids") or [])
            snap = payload.get("signal_snapshots")
            if isinstance(snap, dict):
                clone["signal_snapshots"] = dict(snap)
            merged[source_key] = clone
            order.append(source_key)
            continue
        seen = {str(s) for s in existing.get("signal_ids") or []}
        for raw_sid in payload.get("signal_ids") or []:
            sid = str(raw_sid or "").strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            existing["signal_ids"].append(sid)
        existing["signal_count"] = len(existing["signal_ids"])
        new_snap = payload.get("signal_snapshots")
        if isinstance(new_snap, dict) and new_snap:
            ex_snap = existing.get("signal_snapshots")
            if not isinstance(ex_snap, dict):
                ex_snap = {}
            ex_snap.update(new_snap)
            existing["signal_snapshots"] = ex_snap
        existing["batch_id"] = payload.get("batch_id", existing.get("batch_id"))
        existing["batch_event_type"] = payload.get("batch_event_type", existing.get("batch_event_type"))
        existing["emitted_at"] = payload.get("emitted_at", existing.get("emitted_at"))
    return [merged[key] for key in order]


def _compact_lane_queue(lane: str, queue: "asyncio.Queue[dict[str, Any]]") -> None:
    drained: list[dict[str, Any]] = []
    while True:
        try:
            drained.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    if not drained:
        return
    for compacted in _merge_same_source_payloads(drained):
        queue.put_nowait(compacted)
    now = time.monotonic()
    last = _last_compact_log_at.get(lane, 0.0)
    if now - last >= _COMPACT_LOG_INTERVAL_S:
        _last_compact_log_at[lane] = now
        logger.warning(
            "runtime_signal_queue lane=%s backlog compacted %d->%d batches "
            "(consumer behind; signals remain DB-authoritative)",
            lane,
            len(drained),
            queue.qsize(),
        )


async def publish_signal_batch(
    *,
    event_type: str,
    source: str | None,
    signal_ids: list[str],
    trigger: str,
    reason: str | None = None,
    emitted_at: str | None = None,
    metadata: dict[str, Any] | None = None,
    signal_snapshots: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    normalized_ids = _normalize_signal_ids(signal_ids)
    if not normalized_ids:
        return None

    normalized_event_type = _normalize_event_type(event_type)
    normalized_trigger = str(trigger or "").strip().lower()
    if not _is_actionable_event(normalized_event_type, normalized_trigger):
        return None

    payload: dict[str, Any] = {
        "batch_id": uuid.uuid4().hex,
        "batch_event_type": normalized_event_type,
        "source": _normalize_source(source),
        "signal_ids": normalized_ids,
        "signal_count": int(len(normalized_ids)),
        "trigger": normalized_trigger,
        "reason": reason,
        "emitted_at": str(emitted_at or _utc_now_iso()),
    }
    if isinstance(metadata, dict) and metadata:
        payload["metadata"] = metadata
    if isinstance(signal_snapshots, dict) and signal_snapshots:
        payload["signal_snapshots"] = signal_snapshots

    target_lane = _default_lane_for_source(str(source or ""))
    payload["lane"] = target_lane
    queue = _queues[target_lane]
    # Bound the backlog: if the consumer has fallen behind, collapse the queued
    # batches into one merged payload per source before enqueuing the new one.
    if queue.qsize() >= _QUEUE_COMPACT_THRESHOLD:
        _compact_lane_queue(target_lane, queue)
    await queue.put(copy.deepcopy(payload))
    return str(payload["batch_id"])


async def wait_for_signal_batch(
    *,
    lane: str,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    lane_key = str(lane or _LANE_GENERAL).strip().lower()
    queue = _queues.get(lane_key)
    if queue is None:
        queue = _queues[_LANE_GENERAL]
    timeout = max(0.05, float(timeout_seconds))
    try:
        first = await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None
    batches = [first]
    # Drain the entire lane every wake. A bounded drain left residue that
    # compounded across cycles, so emit->wake latency grew without bound over the
    # 20h soak; draining fully caps it at ~one cycle. Coalescing dedups by signal
    # id, so the merged trigger size stays bounded by distinct pending signals.
    while True:
        try:
            batches.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return _coalesce_batches(batches)

def get_queue_depth(*, lane: str | None = None) -> int | dict[str, int]:
    if lane is None:
        return {name: queue.qsize() for name, queue in _queues.items()}
    lane_key = str(lane or _LANE_GENERAL).strip().lower()
    queue = _queues.get(lane_key)
    return int(queue.qsize()) if queue is not None else 0
