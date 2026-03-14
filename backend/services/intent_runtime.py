from __future__ import annotations

import asyncio
import copy
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from config import settings
from models.database import AsyncSessionLocal, TradeSignal
from models.opportunity import Opportunity
from services.event_bus import event_bus
from services.runtime_signal_queue import publish_signal_batch
from services.signal_bus import (
    build_signal_contract_from_opportunity,
    expire_source_signals_except,
    make_dedupe_key,
    set_trade_signal_status as project_trade_signal_status,
    upsert_trade_signal,
)
from services.ws_feeds import get_feed_manager
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)

_SIGNAL_ACTIVE_STATUSES = {"pending", "selected", "submitted"}
_SIGNAL_TERMINAL_STATUSES = {"executed", "skipped", "expired", "failed"}
_STATUS_PROJECTION_BATCH_MAX = 256
_RUNTIME_LANE_BY_SOURCE = {"crypto": "crypto"}
_PREWARM_SOURCES = {"scanner"}
_PREWARM_WAIT_TIMEOUT_SECONDS = 0.5
_PREWARM_WAIT_POLL_SECONDS = 0.01
_PAYLOAD_VOLATILE_KEYS = {
    "bridge_run_at",
    "bridge_source",
    "execution_armed_at",
    "ingested_at",
    "market_data_age_ms",
    "signal_emitted_at",
    "source_observed_at",
}


def _strict_ws_ttl_seconds_for_source(source: Any) -> float:
    default_ttl = max(0.01, float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 0.1) or 0.1))
    normalized_source = str(source or "").strip().lower()
    if normalized_source != "scanner":
        return default_ttl
    try:
        scanner_max_age_ms = float(getattr(settings, "SCANNER_STRICT_WS_MAX_AGE_MS", 30000) or 30000)
    except Exception:
        scanner_max_age_ms = 30000.0
    return max(default_ttl, max(100.0, scanner_max_age_ms) / 1000.0)


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


def _normalize_text_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        text = str(value or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _signal_runtime_metadata(payload: Any, strategy_context: Any) -> dict[str, Any]:
    payload_json = payload if isinstance(payload, dict) else {}
    strategy_context_json = strategy_context if isinstance(strategy_context, dict) else {}
    runtime = payload_json.get("strategy_runtime")
    runtime_json = dict(runtime) if isinstance(runtime, dict) else {}
    source_key = str(
        runtime_json.get("source_key")
        or strategy_context_json.get("source_key")
        or payload_json.get("source_key")
        or ""
    ).strip().lower()
    subscriptions = _normalize_text_list(
        runtime_json.get("subscriptions")
        or strategy_context_json.get("subscriptions")
        or payload_json.get("subscriptions")
        or []
    )
    execution_activation = str(
        runtime_json.get("execution_activation")
        or strategy_context_json.get("execution_activation")
        or ""
    ).strip().lower()
    if not execution_activation:
        execution_activation = "immediate" if source_key == "crypto" else "ws_post_arm_tick"
    return {
        "source_key": source_key,
        "subscriptions": subscriptions,
        "execution_activation": execution_activation or "immediate",
    }


def _requires_post_arm_ws_tick(payload: Any, strategy_context: Any, *, source: Any) -> bool:
    metadata = _signal_runtime_metadata(payload, strategy_context)
    if str(metadata.get("execution_activation") or "").strip().lower() == "ws_post_arm_tick":
        return True
    normalized_source = str(source or "").strip().lower()
    return normalized_source != "crypto"


def _set_execution_armed_at(
    payload: Any,
    strategy_context: Any,
    *,
    armed_at_iso: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload_json = dict(payload) if isinstance(payload, dict) else {}
    strategy_context_json = dict(strategy_context) if isinstance(strategy_context, dict) else {}
    payload_json["execution_armed_at"] = armed_at_iso
    strategy_context_json["execution_armed_at"] = armed_at_iso
    return payload_json, strategy_context_json


def _required_observed_after_epoch(payload: Any, strategy_context: Any, *, source: Any) -> float | None:
    if not _requires_post_arm_ws_tick(payload, strategy_context, source=source):
        return None
    armed_at = _normalize_datetime(
        (
            (payload or {}).get("execution_armed_at")
            if isinstance(payload, dict)
            else None
        )
        or (
            (strategy_context or {}).get("execution_armed_at")
            if isinstance(strategy_context, dict)
            else None
        )
    )
    if armed_at is None:
        return None
    return float(armed_at.timestamp())


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

def _normalize_runtime_sequence(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _runtime_sort_key(snapshot: dict[str, Any]) -> tuple[int, str]:
    return (_normalize_runtime_sequence(snapshot.get("runtime_sequence")) or 0, str(snapshot.get("id") or ""))


def _runtime_lane_for_source(source: str) -> str:
    return _RUNTIME_LANE_BY_SOURCE.get(str(source or "").strip().lower(), "general")


def _restamp_signal_emitted_at(snapshot: dict[str, Any], emitted_at_iso: str) -> None:
    payload = snapshot.get("payload_json")
    payload_json = dict(payload) if isinstance(payload, dict) else {}
    payload_json["signal_emitted_at"] = emitted_at_iso
    snapshot["payload_json"] = payload_json


def _normalize_token_ids(raw: Any) -> list[str]:
    values = raw if isinstance(raw, list) else []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token_id = str(value or "").strip().lower()
        if not token_id or token_id in seen:
            continue
        seen.add(token_id)
        out.append(token_id)
    return out


def _extract_required_token_ids(payload: Any, *, direction: str) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _append(token_id: Any) -> None:
        normalized = str(token_id or "").strip().lower()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        out.append(normalized)

    execution_plan = payload.get("execution_plan")
    if isinstance(execution_plan, dict):
        raw_legs = [leg for leg in (execution_plan.get("legs") or []) if isinstance(leg, dict)]
        buy_leg = next(
            (leg for leg in raw_legs if str(leg.get("side") or "").strip().lower() == "buy"),
            None,
        )
        primary_leg = buy_leg or (raw_legs[0] if raw_legs else None)
        if isinstance(primary_leg, dict):
            _append(primary_leg.get("token_id"))
        for leg in raw_legs:
            _append(leg.get("token_id"))

    for position in payload.get("positions_to_take") or []:
        if isinstance(position, dict):
            _append(position.get("token_id") or position.get("clob_token_id"))

    _append(payload.get("selected_token_id"))
    _append(payload.get("token_id"))
    _append(payload.get("clob_token_id"))
    _append(payload.get("yes_token_id"))
    _append(payload.get("no_token_id"))

    for market in payload.get("markets") or []:
        if not isinstance(market, dict):
            continue
        token_ids = _normalize_token_ids(
            market.get("clob_token_ids") or market.get("token_ids") or market.get("tokenIds") or []
        )
        if not token_ids:
            continue
        if direction == "buy_yes":
            _append(token_ids[0])
        elif direction == "buy_no" and len(token_ids) > 1:
            _append(token_ids[1])
        else:
            for token_id in token_ids:
                _append(token_id)

    return out


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
        "expires_at",
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
        runtime_sequence=_normalize_runtime_sequence(snapshot.get("runtime_sequence")),
        required_token_ids=list(snapshot.get("required_token_ids") or []),
        deferred_until_ws=bool(snapshot.get("deferred_until_ws")),
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
        self._deferred_signal_ids_by_token: dict[str, set[str]] = {}
        self._deferred_tokens_by_signal_id: dict[str, set[str]] = {}
        self._hot_subscription_tokens: set[str] = set()
        self._next_runtime_sequence = 1
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_callback_registered = False
        self._projection_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._projection_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _track_task(self, task: asyncio.Task[Any], *, name: str) -> asyncio.Task[Any]:
        self._background_tasks.add(task)

        def _finalize(done_task: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Intent runtime background task failed", task_name=name, exc_info=exc)

        task.add_done_callback(_finalize)
        return task

    def _start_task(self, coro: Any, *, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        return self._track_task(task, name=name)

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._projection_task = self._start_task(
            self._run_projection_loop(),
            name="intent-runtime-projection",
        )
        self._register_ws_callback()
        await self.hydrate_from_db()

    async def stop(self) -> None:
        background_tasks = [
            task
            for task in self._background_tasks
            if task is not self._projection_task and not task.done()
        ]
        for task in background_tasks:
            task.cancel()
        if self._projection_task is not None and not self._projection_task.done():
            self._projection_task.cancel()
            try:
                await self._projection_task
            except asyncio.CancelledError:
                pass
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        self._projection_task = None
        self._loop = None
        self._started = False

    def _register_ws_callback(self) -> None:
        if self._ws_callback_registered:
            return
        try:
            feed_manager = get_feed_manager()
        except Exception:
            return
        try:
            feed_manager.cache.add_on_update_callback(self._on_ws_price_update)
            self._ws_callback_registered = True
        except Exception:
            logger.debug("Intent runtime failed to register WS callback")

    def _allocate_runtime_sequence_locked(self) -> int:
        sequence = int(self._next_runtime_sequence)
        self._next_runtime_sequence += 1
        return sequence

    def _clear_deferred_state_locked(self, signal_id: str) -> None:
        existing_tokens = self._deferred_tokens_by_signal_id.pop(signal_id, set())
        for token_id in existing_tokens:
            signal_ids = self._deferred_signal_ids_by_token.get(token_id)
            if not signal_ids:
                continue
            signal_ids.discard(signal_id)
            if not signal_ids:
                self._deferred_signal_ids_by_token.pop(token_id, None)

    def _set_deferred_state_locked(
        self,
        signal_id: str,
        *,
        required_token_ids: list[str],
        reason: str,
    ) -> None:
        self._clear_deferred_state_locked(signal_id)
        token_set = {token_id for token_id in _normalize_token_ids(required_token_ids) if token_id}
        if token_set:
            self._deferred_tokens_by_signal_id[signal_id] = set(token_set)
            for token_id in token_set:
                self._deferred_signal_ids_by_token.setdefault(token_id, set()).add(signal_id)
        snapshot = self._signals_by_id.get(signal_id)
        if snapshot is not None:
            snapshot["required_token_ids"] = sorted(token_set)
            snapshot["deferred_until_ws"] = True
            snapshot["deferred_reason"] = str(reason or "strict_ws_context_missing")

    def _tokens_have_fresh_ws_quotes(
        self,
        token_ids: list[str],
        *,
        source: str | None = None,
        min_observed_at_epoch: float | None = None,
    ) -> bool:
        normalized = _normalize_token_ids(token_ids)
        if not normalized:
            return False
        try:
            feed_manager = get_feed_manager()
        except Exception:
            return False
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return False
        strict_ttl = _strict_ws_ttl_seconds_for_source(source)
        for token_id in normalized:
            try:
                if not feed_manager.cache.is_fresh(token_id, max_age_seconds=strict_ttl):
                    return False
                if feed_manager.cache.get_mid_price(token_id) is None:
                    return False
                if min_observed_at_epoch is not None:
                    observed_at_epoch = feed_manager.cache.get_observed_at_epoch(token_id)
                    if observed_at_epoch is None or float(observed_at_epoch) + 1e-9 < float(min_observed_at_epoch):
                        return False
            except Exception:
                return False
        return True

    async def _wait_for_fresh_ws_quotes(
        self,
        token_ids: list[str],
        *,
        timeout_seconds: float,
        source: str | None = None,
        min_observed_at_epoch: float | None = None,
    ) -> bool:
        normalized = _normalize_token_ids(token_ids)
        if not normalized:
            return False
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            if self._tokens_have_fresh_ws_quotes(
                normalized,
                source=source,
                min_observed_at_epoch=min_observed_at_epoch,
            ):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_PREWARM_WAIT_POLL_SECONDS)

    def _snapshot_ready_for_runtime(self, snapshot: dict[str, Any]) -> bool:
        return self._tokens_have_fresh_ws_quotes(
            list(snapshot.get("required_token_ids") or []),
            source=str(snapshot.get("source") or ""),
            min_observed_at_epoch=_required_observed_after_epoch(
                snapshot.get("payload_json"),
                snapshot.get("strategy_context_json"),
                source=snapshot.get("source"),
            ),
        )

    async def _ensure_hot_subscriptions(self, token_ids: list[str]) -> None:
        normalized = _normalize_token_ids(token_ids)
        if not normalized:
            return
        try:
            feed_manager = get_feed_manager()
        except Exception as exc:
            logger.debug("Intent runtime failed to get feed manager for prewarm", exc_info=exc)
            return
        try:
            if not getattr(feed_manager, "_started", False):
                await feed_manager.start()
            missing = [token_id for token_id in normalized if token_id not in self._hot_subscription_tokens]
            if missing:
                await feed_manager.polymarket_feed.subscribe(missing)
                self._hot_subscription_tokens.update(missing)
        except Exception as exc:
            logger.warning("Intent runtime token prewarm subscribe failed", exc_info=exc)

    async def prewarm_source_tokens(
        self,
        opportunities: list[Opportunity],
        *,
        source: str,
    ) -> None:
        normalized_source = str(source or "").strip().lower()

        token_ids: list[str] = []
        wait_token_ids: list[str] = []
        seen: set[str] = set()
        wait_seen: set[str] = set()
        for opportunity in opportunities:
            try:
                _market_id, direction, _entry_price, _market_question, payload_json, strategy_context_json = (
                    build_signal_contract_from_opportunity(opportunity)
                )
            except Exception:
                continue
            required = _extract_required_token_ids(
                copy.deepcopy(payload_json or {}),
                direction=str(direction or "").strip().lower(),
            )
            requires_post_arm_release = _requires_post_arm_ws_tick(
                payload_json,
                strategy_context_json,
                source=normalized_source,
            )
            for token_id in required:
                normalized_token = str(token_id or "").strip().lower()
                if not normalized_token or normalized_token in seen:
                    continue
                seen.add(normalized_token)
                token_ids.append(normalized_token)
                if requires_post_arm_release or normalized_token in wait_seen:
                    continue
                wait_seen.add(normalized_token)
                wait_token_ids.append(normalized_token)

        if not token_ids:
            return

        await self._ensure_hot_subscriptions(token_ids)
        if wait_token_ids:
            await self._wait_for_fresh_ws_quotes(
                wait_token_ids,
                timeout_seconds=max(
                    _PREWARM_WAIT_TIMEOUT_SECONDS,
                    min(_strict_ws_ttl_seconds_for_source(normalized_source), 1.0),
                ),
                source=normalized_source,
            )

    def _on_ws_price_update(
        self,
        token_id: str,
        _mid: float,
        _bid: float,
        _ask: float,
        _exchange_ts: float,
        _ingest_ts: float,
        _sequence: int,
    ) -> None:
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            self._start_task(
                self._reactivate_deferred_signals_for_token(normalized_token),
                name=f"intent-runtime-deferred-{normalized_token}",
            )
            return
        asyncio.run_coroutine_threadsafe(
            self._reactivate_deferred_signals_for_token(normalized_token),
            loop,
        )

    async def _reactivate_deferred_signals_for_token(self, token_id: str) -> None:
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return
        now = utcnow()
        published_by_source: dict[str, dict[str, dict[str, Any]]] = {}
        projection_snapshots: dict[str, dict[str, Any]] = {}
        async with self._lock:
            signal_ids = sorted(self._deferred_signal_ids_by_token.get(normalized_token, set()))
            for signal_id in signal_ids:
                snapshot = self._signals_by_id.get(signal_id)
                if snapshot is None:
                    self._clear_deferred_state_locked(signal_id)
                    continue
                if not self._snapshot_ready_for_runtime(snapshot):
                    continue
                self._clear_deferred_state_locked(signal_id)
                snapshot["status"] = "pending"
                snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                snapshot["deferred_until_ws"] = False
                snapshot["deferred_reason"] = None
                snapshot["updated_at"] = _to_iso(now)
                published_by_source.setdefault(str(snapshot.get("source") or ""), {})[signal_id] = copy.deepcopy(snapshot)
                projection_snapshots[signal_id] = copy.deepcopy(snapshot)
        for source_key, snapshots in published_by_source.items():
            emitted_at_iso = _to_iso(now)
            for snapshot in snapshots.values():
                _restamp_signal_emitted_at(snapshot, emitted_at_iso)
            for signal_id, projection_snapshot in projection_snapshots.items():
                if signal_id in snapshots:
                    _restamp_signal_emitted_at(projection_snapshot, emitted_at_iso)
            await publish_signal_batch(
                event_type="upsert_reactivated",
                source=source_key,
                signal_ids=sorted(snapshots.keys()),
                trigger="intent_runtime_ws_tick",
                reason="strict_ws_context_ready",
                emitted_at=emitted_at_iso,
                signal_snapshots=snapshots,
            )
        if projection_snapshots:
            await self._enqueue_projection(
                {
                    "kind": "upsert",
                    "source": "__deferred__",
                    "snapshots": projection_snapshots,
                    "sweep_missing": False,
                    "keep_dedupe_keys": [],
                }
            )
            await self._publish_signal_stats()

    def get_runtime_sequence(self, signal_id: str) -> int | None:
        snapshot = self._signals_by_id.get(str(signal_id or "").strip())
        if snapshot is None:
            return None
        return _normalize_runtime_sequence(snapshot.get("runtime_sequence"))

    async def defer_signal(
        self,
        *,
        signal_id: str,
        required_token_ids: list[str] | None = None,
        reason: str = "strict_ws_context_missing",
    ) -> bool:
        normalized_signal_id = str(signal_id or "").strip()
        if not normalized_signal_id:
            return False
        snapshot_copy: dict[str, Any] | None = None
        token_ids: list[str] = []
        async with self._lock:
            snapshot = self._signals_by_id.get(normalized_signal_id)
            if snapshot is None:
                return False
            snapshot["status"] = "pending"
            snapshot["updated_at"] = _to_iso(utcnow())
            snapshot["runtime_sequence"] = None
            token_ids = _normalize_token_ids(
                required_token_ids
                or list(snapshot.get("required_token_ids") or [])
                or _extract_required_token_ids(
                    snapshot.get("payload_json") or {},
                    direction=str(snapshot.get("direction") or "").strip().lower(),
                )
            )
            self._set_deferred_state_locked(
                normalized_signal_id,
                required_token_ids=token_ids,
                reason=reason,
            )
            snapshot_copy = copy.deepcopy(snapshot)
        if snapshot_copy is not None:
            await self._enqueue_projection(
                {
                    "kind": "upsert",
                    "source": str(snapshot_copy.get("source") or ""),
                    "snapshots": {normalized_signal_id: snapshot_copy},
                    "sweep_missing": False,
                    "keep_dedupe_keys": [],
                }
            )
            await self._publish_signal_stats()
        return True

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
        tokens_to_subscribe: set[str] = set()
        async with self._lock:
            self._signals_by_id.clear()
            self._signal_ids_by_dedupe_key.clear()
            self._source_signal_ids.clear()
            self._deferred_signal_ids_by_token.clear()
            self._deferred_tokens_by_signal_id.clear()
            self._next_runtime_sequence = 1
            bootstrap_snapshots: dict[str, dict[str, Any]] = {}
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
                    "runtime_sequence": _normalize_runtime_sequence(row.get("runtime_sequence")),
                    "required_token_ids": _extract_required_token_ids(
                        copy.deepcopy(row.get("payload_json") or {}),
                        direction=str(row.get("direction") or "").strip().lower(),
                    ),
                    "runtime_lane": _runtime_lane_for_source(str(row.get("source") or "")),
                    "deferred_until_ws": False,
                    "deferred_reason": None,
                    "created_at": _to_iso(_normalize_datetime(row.get("created_at"))),
                    "updated_at": _to_iso(_normalize_datetime(row.get("updated_at"))),
                }
                if not snapshot["id"]:
                    continue
                if snapshot["runtime_sequence"] is None and str(snapshot.get("status") or "").strip().lower() in _SIGNAL_ACTIVE_STATUSES:
                    if (
                        snapshot["required_token_ids"]
                        and _required_observed_after_epoch(
                            snapshot.get("payload_json"),
                            snapshot.get("strategy_context_json"),
                            source=snapshot.get("source"),
                        ) is not None
                    ):
                        if self._snapshot_ready_for_runtime(snapshot):
                            snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                            bootstrap_snapshots[snapshot["id"]] = copy.deepcopy(snapshot)
                        else:
                            self._signals_by_id[snapshot["id"]] = snapshot
                            self._set_deferred_state_locked(
                                snapshot["id"],
                                required_token_ids=list(snapshot.get("required_token_ids") or []),
                                reason="awaiting_post_arm_ws_tick",
                            )
                    else:
                        snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                        bootstrap_snapshots[snapshot["id"]] = copy.deepcopy(snapshot)
                sequence = _normalize_runtime_sequence(snapshot.get("runtime_sequence"))
                if sequence is not None:
                    self._next_runtime_sequence = max(self._next_runtime_sequence, sequence + 1)
                self._signals_by_id[snapshot["id"]] = snapshot
                if snapshot["dedupe_key"]:
                    self._signal_ids_by_dedupe_key[snapshot["dedupe_key"]] = snapshot["id"]
                self._source_signal_ids.setdefault(snapshot["source"], set()).add(snapshot["id"])
                if str(snapshot.get("status") or "").strip().lower() in _SIGNAL_ACTIVE_STATUSES:
                    tokens_to_subscribe.update(snapshot.get("required_token_ids") or [])
        if bootstrap_snapshots:
            await self._enqueue_projection(
                {
                    "kind": "upsert",
                    "source": "__bootstrap__",
                    "snapshots": bootstrap_snapshots,
                    "sweep_missing": False,
                    "keep_dedupe_keys": [],
                }
            )
        if tokens_to_subscribe:
            self._start_task(
                self._ensure_hot_subscriptions(sorted(tokens_to_subscribe)),
                name="intent-runtime-hydrate-prewarm",
            )
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
        actionable_snapshots: dict[str, dict[str, Any]] = {}
        actionable_event_types: dict[str, str] = {}
        projection_snapshots: dict[str, dict[str, Any]] = {}
        active_dedupe_keys: set[str] = set()
        prewarm_token_ids: set[str] = set()
        normalized_source = str(source or "").strip().lower()

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
                requires_post_arm_release = _requires_post_arm_ws_tick(
                    payload,
                    strategy_context,
                    source=normalized_source,
                )

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
                    "runtime_sequence": None,
                    "required_token_ids": _extract_required_token_ids(
                        payload,
                        direction=str(direction or "").strip().lower(),
                    ),
                    "runtime_lane": _runtime_lane_for_source(source),
                    "deferred_until_ws": False,
                    "deferred_reason": None,
                    "created_at": _to_iso(now),
                    "updated_at": _to_iso(now),
                }
                if incoming_snapshot["required_token_ids"]:
                    prewarm_token_ids.update(incoming_snapshot["required_token_ids"])

                existing_id = self._signal_ids_by_dedupe_key.get(dedupe_key)
                existing = self._signals_by_id.get(existing_id or "")
                if existing is not None:
                    incoming_snapshot["id"] = existing["id"]
                    incoming_snapshot["created_at"] = existing.get("created_at") or incoming_snapshot["created_at"]
                    incoming_snapshot["effective_price"] = existing.get("effective_price")
                    existing_status = str(existing.get("status") or "pending").strip().lower()
                    material_change = _material_signal_change(existing, incoming_snapshot)
                    if not material_change:
                        active_dedupe_keys.add(dedupe_key)
                        continue
                    incoming_snapshot["status"] = "pending"
                    incoming_snapshot["effective_price"] = None
                    self._clear_deferred_state_locked(existing["id"])
                    if requires_post_arm_release and incoming_snapshot["required_token_ids"]:
                        (
                            incoming_snapshot["payload_json"],
                            incoming_snapshot["strategy_context_json"],
                        ) = _set_execution_armed_at(
                            incoming_snapshot.get("payload_json"),
                            incoming_snapshot.get("strategy_context_json"),
                            armed_at_iso=_to_iso(now) or "",
                        )
                        self._set_deferred_state_locked(
                            existing["id"],
                            required_token_ids=incoming_snapshot["required_token_ids"],
                            reason="awaiting_post_arm_ws_tick",
                        )
                        incoming_snapshot["deferred_until_ws"] = True
                        incoming_snapshot["deferred_reason"] = "awaiting_post_arm_ws_tick"
                        incoming_snapshot["runtime_sequence"] = None
                    elif (
                        normalized_source in _PREWARM_SOURCES
                        and incoming_snapshot["required_token_ids"]
                        and not self._tokens_have_fresh_ws_quotes(
                            incoming_snapshot["required_token_ids"],
                            source=normalized_source,
                        )
                    ):
                        self._set_deferred_state_locked(
                            existing["id"],
                            required_token_ids=incoming_snapshot["required_token_ids"],
                            reason="prewarm_waiting_for_strict_ws_quote",
                        )
                        incoming_snapshot["deferred_until_ws"] = True
                        incoming_snapshot["deferred_reason"] = "prewarm_waiting_for_strict_ws_quote"
                        incoming_snapshot["runtime_sequence"] = None
                    else:
                        incoming_snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                    self._signals_by_id[existing["id"]] = incoming_snapshot
                    projection_snapshots[existing["id"]] = copy.deepcopy(incoming_snapshot)
                    if incoming_snapshot["runtime_sequence"] is not None:
                        actionable_snapshots[existing["id"]] = copy.deepcopy(incoming_snapshot)
                        actionable_event_types[existing["id"]] = (
                            "upsert_reactivated"
                            if existing_status in _SIGNAL_TERMINAL_STATUSES
                            else "upsert_update"
                        )
                else:
                    signal_id = uuid.uuid4().hex
                    incoming_snapshot["id"] = signal_id
                    if requires_post_arm_release and incoming_snapshot["required_token_ids"]:
                        (
                            incoming_snapshot["payload_json"],
                            incoming_snapshot["strategy_context_json"],
                        ) = _set_execution_armed_at(
                            incoming_snapshot.get("payload_json"),
                            incoming_snapshot.get("strategy_context_json"),
                            armed_at_iso=_to_iso(now) or "",
                        )
                        self._signals_by_id[signal_id] = incoming_snapshot
                        self._set_deferred_state_locked(
                            signal_id,
                            required_token_ids=incoming_snapshot["required_token_ids"],
                            reason="awaiting_post_arm_ws_tick",
                        )
                        incoming_snapshot["deferred_until_ws"] = True
                        incoming_snapshot["deferred_reason"] = "awaiting_post_arm_ws_tick"
                    elif (
                        normalized_source in _PREWARM_SOURCES
                        and incoming_snapshot["required_token_ids"]
                        and not self._tokens_have_fresh_ws_quotes(
                            incoming_snapshot["required_token_ids"],
                            source=normalized_source,
                        )
                    ):
                        self._set_deferred_state_locked(
                            signal_id,
                            required_token_ids=incoming_snapshot["required_token_ids"],
                            reason="prewarm_waiting_for_strict_ws_quote",
                        )
                        incoming_snapshot["deferred_until_ws"] = True
                        incoming_snapshot["deferred_reason"] = "prewarm_waiting_for_strict_ws_quote"
                    else:
                        incoming_snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                    self._signals_by_id[signal_id] = incoming_snapshot
                    self._signal_ids_by_dedupe_key[dedupe_key] = signal_id
                    self._source_signal_ids.setdefault(str(source), set()).add(signal_id)
                    projection_snapshots[signal_id] = copy.deepcopy(incoming_snapshot)
                    if incoming_snapshot["runtime_sequence"] is not None:
                        actionable_snapshots[signal_id] = copy.deepcopy(incoming_snapshot)
                        actionable_event_types[signal_id] = "upsert_insert"

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
                    self._clear_deferred_state_locked(signal_id)
                    existing["status"] = "expired"
                    existing["updated_at"] = _to_iso(now)

        if prewarm_token_ids:
            self._start_task(
                self._ensure_hot_subscriptions(sorted(prewarm_token_ids)),
                name=f"intent-runtime-prewarm-{normalized_source or 'signals'}",
            )

        if actionable_snapshots:
            snapshots_by_event_type: dict[str, dict[str, dict[str, Any]]] = {}
            for signal_id, snapshot in actionable_snapshots.items():
                event_type = str(actionable_event_types.get(signal_id) or "upsert_update")
                snapshots_by_event_type.setdefault(event_type, {})[signal_id] = snapshot
            for event_type, snapshots in snapshots_by_event_type.items():
                emitted_at_iso = _to_iso(utcnow())
                for signal_id, snapshot in snapshots.items():
                    _restamp_signal_emitted_at(snapshot, emitted_at_iso)
                    projection_snapshot = projection_snapshots.get(signal_id)
                    if isinstance(projection_snapshot, dict):
                        _restamp_signal_emitted_at(projection_snapshot, emitted_at_iso)
                await publish_signal_batch(
                    event_type=event_type,
                    source=source,
                    signal_ids=sorted(snapshots.keys()),
                    trigger="intent_runtime",
                    emitted_at=emitted_at_iso,
                    signal_snapshots=snapshots,
                )
        if projection_snapshots or sweep_missing:
            await self._enqueue_projection(
                {
                    "kind": "upsert",
                    "source": str(source),
                    "signal_type": signal_type,
                    "snapshots": copy.deepcopy(projection_snapshots),
                    "sweep_missing": bool(sweep_missing),
                    "keep_dedupe_keys": sorted(active_dedupe_keys),
                }
            )
        await self._publish_signal_stats()
        return len(actionable_snapshots)

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
        projection_snapshot: dict[str, Any] | None = None
        async with self._lock:
            snapshot = self._signals_by_id.get(normalized_signal_id)
            if snapshot is None:
                return
            previous_status = str(snapshot.get("status") or "").strip().lower()
            snapshot["status"] = normalized_status
            snapshot["updated_at"] = _to_iso(utcnow())
            if effective_price is not None:
                snapshot["effective_price"] = float(effective_price)
            if normalized_status in _SIGNAL_TERMINAL_STATUSES:
                self._clear_deferred_state_locked(normalized_signal_id)
            elif normalized_status == "pending" and previous_status != "pending" and not bool(snapshot.get("deferred_until_ws")):
                snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
            projection_snapshot = copy.deepcopy(snapshot)
        if (
            projection_snapshot is not None
            and normalized_status == "pending"
            and _normalize_runtime_sequence(projection_snapshot.get("runtime_sequence")) is not None
        ):
            emitted_at_iso = _to_iso(utcnow())
            _restamp_signal_emitted_at(projection_snapshot, emitted_at_iso)
            await self._enqueue_projection(
                {
                    "kind": "upsert",
                    "source": str(projection_snapshot.get("source") or ""),
                    "snapshots": {normalized_signal_id: projection_snapshot},
                    "sweep_missing": False,
                    "keep_dedupe_keys": [],
                }
            )
        else:
            await self._enqueue_projection(
                {
                    "kind": "status",
                    "signal_id": normalized_signal_id,
                    "status": normalized_status,
                    "effective_price": effective_price,
                }
            )
        await self._publish_signal_stats()
        if (
            projection_snapshot is not None
            and normalized_status == "pending"
            and _normalize_runtime_sequence(projection_snapshot.get("runtime_sequence")) is not None
        ):
            await publish_signal_batch(
                event_type="upsert_update",
                source=str(projection_snapshot.get("source") or ""),
                signal_ids=[normalized_signal_id],
                trigger="intent_runtime_status",
                emitted_at=emitted_at_iso,
                signal_snapshots={normalized_signal_id: projection_snapshot},
            )

    async def list_unconsumed_signals(
        self,
        *,
        trader_id: str,
        sources: list[str] | None = None,
        statuses: list[str] | None = None,
        strategy_types_by_source: dict[str, Any] | None = None,
        cursor_runtime_sequence: int | None = None,
        cursor_created_at: datetime | None = None,
        cursor_signal_id: str | None = None,
        limit: int = 200,
    ) -> list[Any]:
        del trader_id
        del cursor_created_at
        del cursor_signal_id
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
                if bool(snapshot.get("deferred_until_ws")):
                    continue
                if expires_at is not None and expires_at < utcnow():
                    continue
                allowed_strategy_types = normalized_strategy_types.get(source)
                strategy_type = str(snapshot.get("strategy_type") or "").strip().lower()
                if allowed_strategy_types and strategy_type not in allowed_strategy_types:
                    continue
                row_sequence = _normalize_runtime_sequence(snapshot.get("runtime_sequence"))
                if row_sequence is None:
                    continue
                if cursor_runtime_sequence is not None and row_sequence <= int(cursor_runtime_sequence):
                    continue
                rows.append(copy.deepcopy(snapshot))
        rows.sort(key=_runtime_sort_key)
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
        if not isinstance(snapshots, dict):
            snapshots = {}
        sweep_missing = bool(payload.get("sweep_missing"))
        keep_dedupe_keys = {str(key) for key in (payload.get("keep_dedupe_keys") or []) if str(key).strip()}
        if not snapshots and not sweep_missing:
            return
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
                    signal_id=str(snapshot.get("id") or "") or None,
                    runtime_sequence=snapshot.get("runtime_sequence"),
                    commit=False,
                )
                desired_status = str(snapshot.get("status") or "").strip().lower()
                if desired_status and desired_status != str(getattr(row, "status", "") or "").strip().lower():
                    row.status = desired_status
                    row.updated_at = _normalize_datetime(snapshot.get("updated_at")) or utcnow()
                row.runtime_sequence = _normalize_runtime_sequence(snapshot.get("runtime_sequence"))
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


_intent_runtime: IntentRuntime | None = None


def get_intent_runtime() -> IntentRuntime:
    global _intent_runtime
    if _intent_runtime is None:
        _intent_runtime = IntentRuntime()
    return _intent_runtime
