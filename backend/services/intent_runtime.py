from __future__ import annotations

import asyncio
import copy
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from models.database import AsyncSessionLocal, TradeSignal
from models.opportunity import Opportunity
from services.event_bus import event_bus
from services.runtime_signal_queue import publish_signal_batch
from services.signal_bus import (
    build_signal_contract_from_opportunity,
    expire_source_signals_except,
    make_dedupe_key,
    refresh_trade_signal_snapshots,
    set_trade_signal_status as project_trade_signal_status,
    upsert_trade_signal,
)
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)

_SIGNAL_ACTIVE_STATUSES = {"pending", "selected", "submitted"}
_SIGNAL_TERMINAL_STATUSES = {"executed", "skipped", "expired", "failed"}
_STATUS_PROJECTION_BATCH_MAX = 256
_PAYLOAD_VOLATILE_KEYS = {
    "bridge_run_at",
    "bridge_source",
    "ingested_at",
    "market_data_age_ms",
    "signal_emitted_at",
    "source_observed_at",
}


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return _to_utc(dt).isoformat().replace("+00:00", "Z")


def _normalize_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _to_utc(value)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return _to_utc(parsed)


def _normalize_payload_for_compare(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key in _PAYLOAD_VOLATILE_KEYS:
                continue
            normalized[key] = _normalize_payload_for_compare(raw_value)
        return normalized
    if isinstance(value, list):
        return [_normalize_payload_for_compare(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 10)
    return value


def _snapshot_sort_key(snapshot: dict[str, Any]) -> tuple[datetime, str]:
    updated_at = _normalize_datetime(snapshot.get("updated_at"))
    created_at = _normalize_datetime(snapshot.get("created_at"))
    return (updated_at or created_at or datetime(1970, 1, 1, tzinfo=timezone.utc), str(snapshot.get("id") or ""))


def _material_signal_change(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    for key in (
        "source",
        "source_item_id",
        "signal_type",
        "strategy_type",
        "market_id",
        "market_question",
        "direction",
        "entry_price",
        "edge_percent",
        "confidence",
        "liquidity",
        "quality_passed",
        "dedupe_key",
    ):
        if existing.get(key) != incoming.get(key):
            return True
    if _normalize_payload_for_compare(existing.get("payload_json")) != _normalize_payload_for_compare(incoming.get("payload_json")):
        return True
    if _normalize_payload_for_compare(existing.get("strategy_context_json")) != _normalize_payload_for_compare(incoming.get("strategy_context_json")):
        return True
    return False


def _coerce_runtime_signal(snapshot: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=str(snapshot.get("id") or "").strip(),
        source=str(snapshot.get("source") or "").strip(),
        source_item_id=str(snapshot.get("source_item_id") or "").strip(),
        signal_type=str(snapshot.get("signal_type") or "").strip(),
        strategy_type=str(snapshot.get("strategy_type") or "").strip(),
        market_id=str(snapshot.get("market_id") or "").strip(),
        market_question=str(snapshot.get("market_question") or "").strip(),
        direction=str(snapshot.get("direction") or "").strip(),
        entry_price=snapshot.get("entry_price"),
        effective_price=snapshot.get("effective_price"),
        edge_percent=snapshot.get("edge_percent"),
        confidence=snapshot.get("confidence"),
        liquidity=snapshot.get("liquidity"),
        expires_at=_normalize_datetime(snapshot.get("expires_at")),
        status=str(snapshot.get("status") or "").strip(),
        payload_json=copy.deepcopy(snapshot.get("payload_json") or {}),
        strategy_context_json=copy.deepcopy(snapshot.get("strategy_context_json") or {}),
        quality_passed=snapshot.get("quality_passed"),
        dedupe_key=str(snapshot.get("dedupe_key") or "").strip(),
        created_at=_normalize_datetime(snapshot.get("created_at")),
        updated_at=_normalize_datetime(snapshot.get("updated_at")),
    )


class IntentRuntime:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._started = False
        self._signals_by_id: dict[str, dict[str, Any]] = {}
        self._signal_ids_by_dedupe_key: dict[str, str] = {}
        self._source_signal_ids: dict[str, set[str]] = {}
        self._projection_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._projection_task: asyncio.Task[None] | None = None

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await self.hydrate_from_db()
        self._projection_task = asyncio.create_task(self._run_projection_loop(), name="intent-runtime-projection")

    async def stop(self) -> None:
        if self._projection_task is not None and not self._projection_task.done():
            self._projection_task.cancel()
            try:
                await self._projection_task
            except asyncio.CancelledError:
                pass
        self._projection_task = None
        self._started = False

    async def hydrate_from_db(self) -> None:
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        TradeSignal.__table__.select().where(TradeSignal.status.in_(tuple(sorted(_SIGNAL_ACTIVE_STATUSES | _SIGNAL_TERMINAL_STATUSES))))
                    )
                )
                .mappings()
                .all()
            )
        async with self._lock:
            self._signals_by_id.clear()
            self._signal_ids_by_dedupe_key.clear()
            self._source_signal_ids.clear()
            for row in rows:
                snapshot = {
                    "id": str(row.get("id") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                    "source_item_id": str(row.get("source_item_id") or "").strip(),
                    "signal_type": str(row.get("signal_type") or "").strip(),
                    "strategy_type": str(row.get("strategy_type") or "").strip(),
                    "market_id": str(row.get("market_id") or "").strip(),
                    "market_question": str(row.get("market_question") or "").strip(),
                    "direction": str(row.get("direction") or "").strip(),
                    "entry_price": row.get("entry_price"),
                    "effective_price": row.get("effective_price"),
                    "edge_percent": row.get("edge_percent"),
                    "confidence": row.get("confidence"),
                    "liquidity": row.get("liquidity"),
                    "expires_at": _to_iso(_normalize_datetime(row.get("expires_at"))),
                    "status": str(row.get("status") or "pending").strip().lower(),
                    "payload_json": copy.deepcopy(row.get("payload_json") or {}),
                    "strategy_context_json": copy.deepcopy(row.get("strategy_context_json") or {}),
                    "quality_passed": row.get("quality_passed"),
                    "dedupe_key": str(row.get("dedupe_key") or "").strip(),
                    "created_at": _to_iso(_normalize_datetime(row.get("created_at"))),
                    "updated_at": _to_iso(_normalize_datetime(row.get("updated_at"))),
                }
                if not snapshot["id"]:
                    continue
                self._signals_by_id[snapshot["id"]] = snapshot
                if snapshot["dedupe_key"]:
                    self._signal_ids_by_dedupe_key[snapshot["dedupe_key"]] = snapshot["id"]
                self._source_signal_ids.setdefault(snapshot["source"], set()).add(snapshot["id"])
        await self._publish_signal_stats()

    async def publish_opportunities(
        self,
        opportunities: list[Opportunity],
        *,
        source: str,
        signal_type_override: str | None = None,
        default_ttl_minutes: int = 120,
        quality_filter_pipeline: Any | None = None,
        quality_reports: dict[str, Any] | None = None,
        sweep_missing: bool = False,
        refresh_prices: bool = True,
    ) -> int:
        del refresh_prices
        now = utcnow()
        signal_type = str(signal_type_override or f"{source}_opportunity").strip().lower()
        published_snapshots: dict[str, dict[str, Any]] = {}
        active_dedupe_keys: set[str] = set()

        async with self._lock:
            for opportunity in opportunities:
                market_id, direction, entry_price, market_question, payload_json, strategy_context_json = (
                    build_signal_contract_from_opportunity(opportunity)
                )
                if not market_id:
                    continue
                dedupe_key = make_dedupe_key(
                    getattr(opportunity, "stable_id", None),
                    getattr(opportunity, "strategy", None),
                    market_id,
                )
                expires_at = getattr(opportunity, "resolution_date", None) or (now + timedelta(minutes=default_ttl_minutes))
                payload = copy.deepcopy(payload_json or {})
                strategy_context = copy.deepcopy(strategy_context_json or {})
                payload["ingested_at"] = _to_iso(now)
                payload["signal_emitted_at"] = payload.get("signal_emitted_at") or _to_iso(now)
                payload["bridge_source"] = str(source)
                payload["bridge_run_at"] = _to_iso(now)
                strategy_context["ingested_at"] = _to_iso(now)
                strategy_context["bridge_source"] = str(source)
                strategy_context["bridge_run_at"] = _to_iso(now)

                opp_quality_passed: bool | None = None
                if quality_filter_pipeline is not None:
                    report = quality_filter_pipeline.evaluate(opportunity)
                    opp_quality_passed = bool(report.passed)
                elif quality_reports is not None:
                    report = quality_reports.get(getattr(opportunity, "stable_id", None) or getattr(opportunity, "id", None))
                    if report is not None:
                        opp_quality_passed = bool(report.passed)

                incoming_snapshot = {
                    "id": "",
                    "source": str(source),
                    "source_item_id": str(getattr(opportunity, "stable_id", None) or "").strip(),
                    "signal_type": signal_type,
                    "strategy_type": str(getattr(opportunity, "strategy", None) or "").strip(),
                    "market_id": str(market_id),
                    "market_question": str(market_question or "").strip(),
                    "direction": str(direction or "").strip(),
                    "entry_price": float(entry_price) if entry_price is not None else None,
                    "effective_price": None,
                    "edge_percent": float(getattr(opportunity, "roi_percent", 0.0) or 0.0),
                    "confidence": float(getattr(opportunity, "confidence", 0.0) or 0.0),
                    "liquidity": float(getattr(opportunity, "min_liquidity", 0.0) or 0.0),
                    "expires_at": _to_iso(_normalize_datetime(expires_at)),
                    "status": "pending",
                    "payload_json": payload,
                    "strategy_context_json": strategy_context,
                    "quality_passed": opp_quality_passed,
                    "dedupe_key": dedupe_key,
                    "created_at": _to_iso(now),
                    "updated_at": _to_iso(now),
                }

                existing_id = self._signal_ids_by_dedupe_key.get(dedupe_key)
                existing = self._signals_by_id.get(existing_id or "")
                if existing is not None:
                    incoming_snapshot["id"] = existing["id"]
                    incoming_snapshot["created_at"] = existing.get("created_at") or incoming_snapshot["created_at"]
                    if existing.get("status") in {"selected", "submitted"} and not _material_signal_change(existing, incoming_snapshot):
                        incoming_snapshot["status"] = str(existing.get("status") or "pending")
                        incoming_snapshot["effective_price"] = existing.get("effective_price")
                    elif existing.get("status") in _SIGNAL_TERMINAL_STATUSES and not _material_signal_change(existing, incoming_snapshot):
                        incoming_snapshot["status"] = str(existing.get("status") or "pending")
                        incoming_snapshot["effective_price"] = existing.get("effective_price")
                    self._signals_by_id[existing["id"]] = incoming_snapshot
                    published_snapshots[existing["id"]] = copy.deepcopy(incoming_snapshot)
                else:
                    signal_id = uuid.uuid4().hex
                    incoming_snapshot["id"] = signal_id
                    self._signals_by_id[signal_id] = incoming_snapshot
                    self._signal_ids_by_dedupe_key[dedupe_key] = signal_id
                    self._source_signal_ids.setdefault(str(source), set()).add(signal_id)
                    published_snapshots[signal_id] = copy.deepcopy(incoming_snapshot)

                active_dedupe_keys.add(dedupe_key)

            if sweep_missing:
                source_signal_ids = list(self._source_signal_ids.get(str(source), set()))
                for signal_id in source_signal_ids:
                    existing = self._signals_by_id.get(signal_id)
                    if existing is None:
                        continue
                    if str(existing.get("dedupe_key") or "") in active_dedupe_keys:
                        continue
                    if str(existing.get("status") or "").strip().lower() not in _SIGNAL_ACTIVE_STATUSES:
                        continue
                    existing["status"] = "expired"
                    existing["updated_at"] = _to_iso(now)

        if published_snapshots:
            await publish_signal_batch(
                event_type="upsert_insert",
                source=source,
                signal_ids=sorted(published_snapshots.keys()),
                trigger="intent_runtime",
                emitted_at=_to_iso(now),
                signal_snapshots=published_snapshots,
            )
        await self._enqueue_projection(
            {
                "kind": "upsert",
                "source": str(source),
                "signal_type": signal_type,
                "snapshots": copy.deepcopy(published_snapshots),
                "sweep_missing": bool(sweep_missing),
                "keep_dedupe_keys": sorted(active_dedupe_keys),
            }
        )
        await self._publish_signal_stats()
        return len(published_snapshots)

    async def update_signal_status(
        self,
        *,
        signal_id: str,
        status: str,
        effective_price: float | None = None,
    ) -> None:
        normalized_signal_id = str(signal_id or "").strip()
        if not normalized_signal_id:
            return
        normalized_status = str(status or "").strip().lower()
        source = ""
        async with self._lock:
            snapshot = self._signals_by_id.get(normalized_signal_id)
            if snapshot is None:
                return
            snapshot["status"] = normalized_status
            snapshot["updated_at"] = _to_iso(utcnow())
            if effective_price is not None:
                snapshot["effective_price"] = float(effective_price)
            source = str(snapshot.get("source") or "")
        await self._enqueue_projection(
            {
                "kind": "status",
                "signal_id": normalized_signal_id,
                "status": normalized_status,
                "effective_price": effective_price,
            }
        )
        await self._publish_signal_stats()
        snapshot = self._signals_by_id.get(normalized_signal_id)
        if snapshot is not None:
            await publish_signal_batch(
                event_type="upsert_update",
                source=source,
                signal_ids=[normalized_signal_id],
                trigger="intent_runtime_status",
                emitted_at=_to_iso(utcnow()),
                signal_snapshots={normalized_signal_id: copy.deepcopy(snapshot)},
            )

    async def list_unconsumed_signals(
        self,
        *,
        trader_id: str,
        sources: list[str] | None = None,
        statuses: list[str] | None = None,
        strategy_types_by_source: dict[str, Any] | None = None,
        cursor_created_at: datetime | None = None,
        cursor_signal_id: str | None = None,
        limit: int = 200,
    ) -> list[Any]:
        del trader_id
        normalized_sources = {str(source or "").strip().lower() for source in (sources or []) if str(source or "").strip()}
        normalized_statuses = {str(status or "").strip().lower() for status in (statuses or []) if str(status or "").strip()}
        normalized_strategy_types: dict[str, set[str]] = {}
        for source_key, strategy_types in (strategy_types_by_source or {}).items():
            normalized_source = str(source_key or "").strip().lower()
            if not normalized_source:
                continue
            normalized_strategy_types[normalized_source] = {
                str(strategy_type or "").strip().lower()
                for strategy_type in (strategy_types or [])
                if str(strategy_type or "").strip()
            }
        cursor_dt = _to_utc(cursor_created_at)
        cursor_id = str(cursor_signal_id or "").strip()
        rows: list[dict[str, Any]] = []
        async with self._lock:
            for snapshot in self._signals_by_id.values():
                source = str(snapshot.get("source") or "").strip().lower()
                status = str(snapshot.get("status") or "").strip().lower()
                expires_at = _normalize_datetime(snapshot.get("expires_at"))
                if normalized_sources and source not in normalized_sources:
                    continue
                if normalized_statuses and status not in normalized_statuses:
                    continue
                if expires_at is not None and expires_at < utcnow():
                    continue
                allowed_strategy_types = normalized_strategy_types.get(source)
                strategy_type = str(snapshot.get("strategy_type") or "").strip().lower()
                if allowed_strategy_types and strategy_type not in allowed_strategy_types:
                    continue
                if cursor_dt is not None:
                    row_dt = _normalize_datetime(snapshot.get("updated_at")) or _normalize_datetime(snapshot.get("created_at"))
                    if row_dt is None:
                        continue
                    if row_dt < cursor_dt:
                        continue
                    if row_dt == cursor_dt and cursor_id and str(snapshot.get("id") or "") <= cursor_id:
                        continue
                rows.append(copy.deepcopy(snapshot))
        rows.sort(key=_snapshot_sort_key)
        return [_coerce_runtime_signal(row) for row in rows[: max(1, min(int(limit), 5000))]]

    def get_signal_snapshot_rows(self) -> list[dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for snapshot in self._signals_by_id.values():
            source = str(snapshot.get("source") or "").strip().lower()
            if not source:
                continue
            row = stats.setdefault(
                source,
                {
                    "source": source,
                    "pending_count": 0,
                    "selected_count": 0,
                    "submitted_count": 0,
                    "executed_count": 0,
                    "skipped_count": 0,
                    "expired_count": 0,
                    "failed_count": 0,
                    "latest_signal_at": None,
                    "updated_at": None,
                },
            )
            status = str(snapshot.get("status") or "").strip().lower()
            key = f"{status}_count"
            if key in row:
                row[key] += 1
            updated_at = str(snapshot.get("updated_at") or snapshot.get("created_at") or "")
            if updated_at and (row["latest_signal_at"] is None or updated_at > row["latest_signal_at"]):
                row["latest_signal_at"] = updated_at
            if updated_at and (row["updated_at"] is None or updated_at > row["updated_at"]):
                row["updated_at"] = updated_at
        return [stats[key] for key in sorted(stats.keys())]

    async def _publish_signal_stats(self) -> None:
        rows = self.get_signal_snapshot_rows()
        try:
            await event_bus.publish("signals_update", {"sources": copy.deepcopy(rows)})
        except Exception:
            logger.debug("Failed to publish runtime signals_update event")

    async def _enqueue_projection(self, payload: dict[str, Any]) -> None:
        if not self._started:
            return
        await self._projection_queue.put(payload)

    async def _run_projection_loop(self) -> None:
        while True:
            payload = await self._projection_queue.get()
            try:
                kind = str(payload.get("kind") or "").strip().lower()
                if kind == "status":
                    status_payloads = [payload]
                    carry_payload: dict[str, Any] | None = None
                    while len(status_payloads) < _STATUS_PROJECTION_BATCH_MAX:
                        try:
                            queued = self._projection_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        queued_kind = str(queued.get("kind") or "").strip().lower()
                        if queued_kind == "status":
                            status_payloads.append(queued)
                            continue
                        carry_payload = queued
                        break
                    await self._project_status_batch(status_payloads)
                    if carry_payload is not None:
                        await self._dispatch_projection_payload(carry_payload)
                else:
                    await self._dispatch_projection_payload(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Intent runtime DB projection failed", exc_info=exc)

    async def _dispatch_projection_payload(self, payload: dict[str, Any]) -> None:
        kind = str(payload.get("kind") or "").strip().lower()
        if kind == "upsert":
            await self._project_upsert_batch(payload)
        elif kind == "status":
            await self._project_status(payload)

    async def _project_upsert_batch(self, payload: dict[str, Any]) -> None:
        source = str(payload.get("source") or "").strip()
        snapshots = payload.get("snapshots")
        if not isinstance(snapshots, dict) or not snapshots:
            return
        sweep_missing = bool(payload.get("sweep_missing"))
        keep_dedupe_keys = {str(key) for key in (payload.get("keep_dedupe_keys") or []) if str(key).strip()}
        async with AsyncSessionLocal() as session:
            signal_types_by_source: set[str] = set()
            for snapshot in snapshots.values():
                signal_type = str(snapshot.get("signal_type") or "").strip().lower()
                if signal_type:
                    signal_types_by_source.add(signal_type)
                row = await upsert_trade_signal(
                    session,
                    source=str(snapshot.get("source") or source),
                    source_item_id=snapshot.get("source_item_id"),
                    signal_type=signal_type,
                    strategy_type=snapshot.get("strategy_type"),
                    market_id=str(snapshot.get("market_id") or ""),
                    market_question=snapshot.get("market_question"),
                    direction=snapshot.get("direction"),
                    entry_price=snapshot.get("entry_price"),
                    edge_percent=snapshot.get("edge_percent"),
                    confidence=snapshot.get("confidence"),
                    liquidity=snapshot.get("liquidity"),
                    expires_at=_normalize_datetime(snapshot.get("expires_at")),
                    payload_json=copy.deepcopy(snapshot.get("payload_json") or {}),
                    strategy_context_json=copy.deepcopy(snapshot.get("strategy_context_json") or {}),
                    quality_passed=snapshot.get("quality_passed"),
                    quality_rejection_reasons=None,
                    dedupe_key=str(snapshot.get("dedupe_key") or ""),
                    commit=False,
                )
                desired_status = str(snapshot.get("status") or "").strip().lower()
                if desired_status and desired_status != str(getattr(row, "status", "") or "").strip().lower():
                    row.status = desired_status
                    row.updated_at = _normalize_datetime(snapshot.get("updated_at")) or utcnow()
                effective_price = snapshot.get("effective_price")
                if effective_price is not None:
                    row.effective_price = effective_price
            if sweep_missing:
                await expire_source_signals_except(
                    session,
                    source=source,
                    keep_dedupe_keys=keep_dedupe_keys,
                    signal_types=sorted(signal_types_by_source),
                    commit=False,
                )
            await session.commit()
            await refresh_trade_signal_snapshots(session)

    async def _project_status(self, payload: dict[str, Any]) -> None:
        await self._project_status_batch([payload])

    async def _project_status_batch(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            return
        latest_by_signal_id: dict[str, dict[str, Any]] = {}
        for payload in payloads:
            signal_id = str(payload.get("signal_id") or "").strip()
            if not signal_id:
                continue
            latest_by_signal_id[signal_id] = {
                "signal_id": signal_id,
                "status": str(payload.get("status") or ""),
                "effective_price": payload.get("effective_price"),
            }
        if not latest_by_signal_id:
            return
        async with AsyncSessionLocal() as session:
            changed_any = False
            for item in latest_by_signal_id.values():
                changed = await project_trade_signal_status(
                    session,
                    str(item.get("signal_id") or ""),
                    str(item.get("status") or ""),
                    effective_price=item.get("effective_price"),
                    commit=False,
                )
                changed_any = changed_any or bool(changed)
            if changed_any:
                await session.commit()
                await refresh_trade_signal_snapshots(session)


_intent_runtime: IntentRuntime | None = None


def get_intent_runtime() -> IntentRuntime:
    global _intent_runtime
    if _intent_runtime is None:
        _intent_runtime = IntentRuntime()
    return _intent_runtime
