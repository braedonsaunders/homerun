from __future__ import annotations

import asyncio
import copy
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from sqlalchemy import Text, and_, cast, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import settings
from models.database import (
    AsyncSessionLocal,
    AuditAsyncSessionLocal,
    TradeSignal,
    TradeSignalEmission,
)
from models.opportunity import Opportunity
from services.event_bus import event_bus
from services.live_pressure import db_pressure_remaining_seconds, is_db_pressure_active, maybe_mark_db_pressure, publish_backpressure
from services.runtime_signal_queue import publish_signal_batch
from services.signal_bus import (
    SIGNAL_ACTIVE_STATUSES as BUS_SIGNAL_ACTIVE_STATUSES,
    SIGNAL_REACTIVATABLE_STATUSES as BUS_SIGNAL_REACTIVATABLE_STATUSES,
    build_signal_contract_from_opportunity,
    expire_source_signals_except,
    make_dedupe_key,
    upsert_trade_signal,
)
from services.strategy_loader import strategy_loader
from services.strategy_sdk import StrategySDK
from services.ws_feeds import get_feed_manager
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)

_SIGNAL_ACTIVE_STATUSES = {"pending", "selected", "submitted"}
_SIGNAL_TERMINAL_STATUSES = {"executed", "skipped", "expired", "failed"}
_STATUS_PROJECTION_BATCH_MAX = 64
_UPSERT_PROJECTION_BATCH_MAX = 8
_STATUS_PROJECTION_CHUNK_SIZE = 4
_STATUS_PROJECTION_PRESSURE_CHUNK_SIZE = 1
# Fix Q (chunk_size 10→1) eliminated the multi-row lock-hold pattern
# that produced ``LOCK CONTENTION wait=Lock/transactionid
# query='UPDATE trade_signals SET source_item_id=$1, signal_type=$2,
# ...'`` waits of 4-8s — verified across cycles 12-14: lock_contentions
# went 6+/17min → 0 sustained.
#
# Fix V: re-tune the trade-off.  The 5/2026/05 22:42-22:49 live
# capture (post-Fix-Q + post-S/T/U) showed the projection queue
# backing up to ``projection_queue_size=684`` under sustained burst
# load.  Cause: chunk_size=1 means every snapshot pays the full
# per-transaction overhead — session checkout + 2× SET LOCAL + 1
# dedupe-key SELECT + 1 upsert + 1 commit ≈ 5 round trips per row.
# When the burst arrival rate exceeds the consumer's per-row cost
# under shared-pool contention with concurrent trader cycles, the
# queue grows monotonically until the next quiet window.
#
# Compromise: chunk_size=3 in the steady-state path.  Three row
# locks held briefly (single-digit ms each commit) is far below
# the contention regime that motivated Fix Q (cycle 11 / pre-Fix-Q
# saw 10-row chunks holding locks 1-3s under the same load).
# Three-row chunks amortize the per-transaction overhead 3× while
# keeping lock-hold time bounded to a small multiple of the
# fastest commit time.  Pressure-mode chunk_size stays at 1 — when
# ``is_db_pressure_active()`` is True (recent DBAPIError or
# pool-watchdog flag) we prioritize lock-hold-time minimization
# over throughput.
_UPSERT_PROJECTION_CHUNK_SIZE = 1  # ITER-?: 3->1 isolates per-row lock failures so one contended dedupe_key can't poison a 3-row batch
_UPSERT_PROJECTION_PRESSURE_CHUNK_SIZE = 1
_PROJECTION_RETRY_MAX_ATTEMPTS = 3
_PROJECTION_RETRY_BASE_DELAY_SECONDS = 0.25
_PROJECTION_DB_PRESSURE_TTL_SECONDS = 15.0
_PROJECTION_STATEMENT_TIMEOUT_MS = 5000
# Decoupled, batched trade_signal_emissions history writes (see
# _run_emission_flush_loop). Append-only + loss-tolerant, so coalesce into one
# bulk insert per interval off the projection hot path instead of an inline
# insert+commit per status chunk (which held a main-pool connection 2-3.5s).
_EMISSION_FLUSH_INTERVAL_SECONDS = 0.5
_EMISSION_BUFFER_MAX = 50_000
_PROJECTION_LOCK_TIMEOUT_MS = 3000  # ITER-?: 1000->3000ms gives more headroom for transient row-lock contention before the projection batch aborts and the signal ages out
_RUNTIME_LANE_BY_SOURCE = {"crypto": "crypto"}
_PREWARM_SOURCES = {"scanner"}
_PREWARM_WAIT_TIMEOUT_SECONDS = 0.5
_PREWARM_WAIT_POLL_SECONDS = 0.01
_SIGNAL_PUBLICATION_BATCH_SIZE = 200
# Lookback for terminal-signal reactivation.  Was 24h, but production
# (5 h soak, 5/2026/05) hit 5 GB RSS — terminal signals lingered for a
# day each, accumulating ~50k entries × ~30KB = ~1.5 GB just for the
# in-memory ``_signals_by_id`` map, with most of those being dead
# terminals.  4h is enough to recover from a worker restart while
# bounding memory: signals older than 4h are functionally never
# reactivated.
_BOOTSTRAP_REACTIVATABLE_LOOKBACK_HOURS = 4.0
_UNCHANGED_SCANNER_TERMINAL_REACTIVATION_COOLDOWN_SECONDS = 180.0
_HOT_SUBSCRIPTION_SEED_CONCURRENCY = 8
_HOT_SUBSCRIPTION_SEED_RETRY_SECONDS = 30.0
_DEFERRED_QUOTE_MIN_OBSERVED_AT_KEY = "deferred_quote_min_observed_at"
_DEFERRED_REQUIRED_MAX_AGE_MS_KEY = "strict_ws_required_max_age_ms"
_LIVE_MARKET_PAYLOAD_KEY = "live_market"
_PAYLOAD_VOLATILE_KEYS = {
    "bridge_run_at",
    "bridge_source",
    "execution_armed_at",
    "ingested_at",
    "market_data_age_ms",
    "signal_emitted_at",
    "source_observed_at",
}
_NON_REACTIVATABLE_DEFERRED_REASONS_BY_SOURCE = {
    "crypto": {
        "strict_ws_pricing_live_context_unavailable",
        "strict_ws_pricing_signal_release_stale",
    },
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


def _deferred_reason_is_nonreactivatable(snapshot: dict[str, Any]) -> bool:
    source = str(snapshot.get("source") or "").strip().lower()
    deferred_reason = str(snapshot.get("deferred_reason") or "").strip().lower()
    if not source or not deferred_reason:
        return False
    return deferred_reason in _NON_REACTIVATABLE_DEFERRED_REASONS_BY_SOURCE.get(source, set())


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


def _snapshot_signal_view(snapshot: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**copy.deepcopy(snapshot))


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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _opportunity_dedupe_key(opportunity: Opportunity, contract_market_id: Any) -> str:
    """Compute the ``trade_signals.dedupe_key`` for an opportunity.

    Plan 0041 added per-trader scope to the dedupe key so two clones
    of the same strategy emitting on the same market produce distinct
    rows. The scope is appended ONLY when ``intended_trader_id`` is
    truthy — passing ``None`` as the 4th arg to ``make_dedupe_key``
    would silently shift every legacy (singleton-emitted) hash because
    ``"|".join(... or "")`` adds a trailing pipe. Keeping the 3-arg
    form for unscoped opportunities preserves backward compatibility
    with pre-Plan-0041 ``trade_signals`` rows across deploys.
    """
    intended = (
        str(getattr(opportunity, "intended_trader_id", None) or "").strip()
        or None
    )
    parts: list[Any] = [
        getattr(opportunity, "stable_id", None),
        getattr(opportunity, "strategy", None),
        contract_market_id,
    ]
    if intended:
        parts.append(intended)
    return make_dedupe_key(*parts)


def _opportunity_signal_expires_at(now: datetime, opportunity: Opportunity, default_ttl_minutes: int) -> datetime:
    strategy_key = str(getattr(opportunity, "strategy", "") or "").strip().lower()
    try:
        instance = strategy_loader.get_instance(strategy_key) if strategy_key else None
    except Exception:
        instance = None
    candidates: list[Any] = []
    config = getattr(instance, "config", None)
    if isinstance(config, dict):
        candidates.extend(
            [
                config.get("retention_max_age_minutes"),
                config.get("retention_window"),
                config.get("retention_period"),
                config.get("retention_duration"),
                config.get("opportunity_ttl_minutes"),
                config.get("opportunity_ttl"),
            ]
        )
    candidates.extend(
        [
            getattr(instance, "retention_max_age_minutes", None),
            getattr(instance, "retention_window", None),
            getattr(instance, "opportunity_ttl_minutes", None),
            getattr(opportunity, "retention_max_age_minutes", None),
            getattr(opportunity, "retention_window", None),
            getattr(opportunity, "opportunity_ttl_minutes", None),
        ]
    )
    for candidate in candidates:
        parsed = StrategySDK.parse_duration_minutes(candidate)
        if parsed is None:
            continue
        ttl_minutes = max(1, min(int(parsed), 60 * 24 * 90))
        return now + timedelta(minutes=ttl_minutes)
    return now + timedelta(minutes=max(1, int(default_ttl_minutes or 120)))


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
        if source_key == "crypto":
            execution_activation = "immediate"
        elif source_key == "scanner":
            execution_activation = "ws_current"
        else:
            execution_activation = "ws_post_arm_tick"
    return {
        "source_key": source_key,
        "subscriptions": subscriptions,
        "execution_activation": execution_activation or "immediate",
    }





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


def _projection_pressure_ttl_seconds(retry_count: int) -> float | None:
    if int(retry_count or 0) <= 0:
        return None
    return min(
        _PROJECTION_DB_PRESSURE_TTL_SECONDS * float(max(1, int(retry_count or 0))),
        45.0,
    )


def _runtime_sort_key(snapshot: dict[str, Any]) -> tuple[int, str]:
    return (_normalize_runtime_sequence(snapshot.get("runtime_sequence")) or 0, str(snapshot.get("id") or ""))


def _runtime_lane_for_source(source: str) -> str:
    return _RUNTIME_LANE_BY_SOURCE.get(str(source or "").strip().lower(), "general")


def _restamp_signal_emitted_at(snapshot: dict[str, Any], emitted_at_iso: str) -> None:
    payload = snapshot.get("payload_json")
    payload_json = dict(payload) if isinstance(payload, dict) else {}
    payload_json["signal_emitted_at"] = emitted_at_iso
    snapshot["payload_json"] = payload_json


def _ensure_execution_armed_at(snapshot: dict[str, Any], armed_at_iso: str) -> None:
    payload = snapshot.get("payload_json")
    payload_json = dict(payload) if isinstance(payload, dict) else {}
    payload_json["execution_armed_at"] = str(payload_json.get("execution_armed_at") or armed_at_iso)
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


def _snapshot_live_market(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload = snapshot.get("payload_json")
    payload = payload if isinstance(payload, dict) else {}
    live_market = payload.get(_LIVE_MARKET_PAYLOAD_KEY)
    return dict(live_market) if isinstance(live_market, dict) else {}


async def build_cached_live_signal_contexts(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
    from services.trader_orchestrator.live_market_context import (
        build_cached_live_signal_contexts as _build_cached_live_signal_contexts,
    )

    return await _build_cached_live_signal_contexts(*args, **kwargs)


def _clear_snapshot_live_market(snapshot: dict[str, Any]) -> None:
    payload = snapshot.get("payload_json")
    if not isinstance(payload, dict) or _LIVE_MARKET_PAYLOAD_KEY not in payload:
        return
    payload = dict(payload)
    payload.pop(_LIVE_MARKET_PAYLOAD_KEY, None)
    snapshot["payload_json"] = payload


def _snapshot_requires_scanner_strict_live_market(snapshot: dict[str, Any]) -> bool:
    if str(snapshot.get("source") or "").strip().lower() != "scanner":
        return False
    return bool(_normalize_token_ids(snapshot.get("required_token_ids")))


def _snapshot_has_strict_scanner_live_market(snapshot: dict[str, Any]) -> bool:
    if not _snapshot_requires_scanner_strict_live_market(snapshot):
        return True
    live_market = _snapshot_live_market(snapshot)
    live_source = str(
        live_market.get("market_data_source")
        or live_market.get("live_selected_price_source")
        or ""
    ).strip().lower()
    live_price = _safe_float(live_market.get("live_selected_price"))
    live_age_ms = _safe_float(
        live_market.get("market_data_age_ms")
        if live_market.get("market_data_age_ms") is not None
        else live_market.get("age_ms")
    )
    ws_subscription_current = bool(live_market.get("ws_subscription_current"))
    strict_max_age_ms = max(100, int(_strict_ws_ttl_seconds_for_source("scanner") * 1000.0))
    return (
        bool(live_market.get("available"))
        and live_source in {"ws_strict", "redis_strict"}
        and live_price is not None
        and live_price > 0.0
        and (
            ws_subscription_current
            or (live_age_ms is not None and live_age_ms <= strict_max_age_ms)
        )
    )


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
        for leg in raw_legs:
            _append(leg.get("token_id"))

    for position in payload.get("positions_to_take") or []:
        if isinstance(position, dict):
            _append(position.get("token_id") or position.get("clob_token_id"))

    _append(payload.get("selected_token_id"))
    _append(payload.get("token_id"))
    _append(payload.get("clob_token_id"))

    if out:
        return out

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
        "quality_rejection_reasons",
        "dedupe_key",
    ):
        if existing.get(key) != incoming.get(key):
            return True
    if _normalize_payload_for_compare(existing.get("payload_json")) != _normalize_payload_for_compare(incoming.get("payload_json")):
        return True
    if _normalize_payload_for_compare(existing.get("strategy_context_json")) != _normalize_payload_for_compare(incoming.get("strategy_context_json")):
        return True
    return False


def _should_reactivate_unchanged_terminal_signal(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    now: datetime,
) -> bool:
    existing_source = str(existing.get("source") or "").strip().lower()
    incoming_source = str(incoming.get("source") or "").strip().lower()
    if existing_source != "scanner" or incoming_source != "scanner":
        return False
    if str(incoming.get("status") or "").strip().lower() != "pending":
        return False
    existing_status = str(existing.get("status") or "").strip().lower()
    if existing_status not in _SIGNAL_TERMINAL_STATUSES:
        return False
    updated_at = _normalize_datetime(existing.get("updated_at")) or _normalize_datetime(existing.get("created_at"))
    if updated_at is None:
        return True
    return (now - updated_at).total_seconds() >= _UNCHANGED_SCANNER_TERMINAL_REACTIVATION_COOLDOWN_SECONDS


def _coerce_runtime_signal(snapshot: dict[str, Any]) -> Any:
    intended_trader_id_raw = snapshot.get("intended_trader_id")
    intended_trader_id = (
        str(intended_trader_id_raw).strip()
        if intended_trader_id_raw is not None
        else ""
    ) or None
    return SimpleNamespace(
        id=str(snapshot.get("id") or "").strip(),
        source=str(snapshot.get("source") or "").strip(),
        source_item_id=str(snapshot.get("source_item_id") or "").strip(),
        signal_type=str(snapshot.get("signal_type") or "").strip(),
        strategy_type=str(snapshot.get("strategy_type") or "").strip(),
        intended_trader_id=intended_trader_id,
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
        quality_rejection_reasons=list(snapshot.get("quality_rejection_reasons") or []),
        dedupe_key=str(snapshot.get("dedupe_key") or "").strip(),
        runtime_sequence=_normalize_runtime_sequence(snapshot.get("runtime_sequence")),
        required_token_ids=list(snapshot.get("required_token_ids") or []),
        deferred_until_ws=bool(snapshot.get("deferred_until_ws")),
        created_at=_normalize_datetime(snapshot.get("created_at")),
        updated_at=_normalize_datetime(snapshot.get("updated_at")),
    )


class IntentRuntime:
    # Hard ceiling on the in-memory signal table.  Hydrate-from-db only
    # Cap on in-memory signal map.  Was 50k (giving ~1.5 GB worst case
    # at ~30KB/snapshot); production saw 5 GB RSS leaks.  15k bounds
    # signals at ~450 MB worst case while still leaving headroom for
    # active + recent-terminal across 4h.  When exceeded, the pruner
    # drops oldest terminals first, then oldest non-terminals as a
    # last-resort guard against stuck signals (stuck = orchestrator
    # never finished processing them, e.g. during event-loop stalls).
    _MAX_SIGNALS_IN_MEMORY: int = 15_000
    # Cadence for the periodic pruner.  Was 300s (5 min); halved to
    # 60s so memory pressure is relieved sooner under high signal
    # throughput.
    _PRUNE_INTERVAL_SECONDS: float = 60.0
    # Cap on the WS hot-seed retry-cooldown table.  Each entry is a
    # token_id → monotonic-time pair that's never cleaned up after the
    # token leaves the active universe.  ~14K active markets × 2
    # tokens = ~28K, with churn over time → set 64K to leave generous
    # headroom before the LRU-style trim kicks in.
    _MAX_HOT_SEED_RETRIES: int = 64_000

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
        self._projection_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=5000)
        self._projection_task: asyncio.Task[None] | None = None
        self._deferred_timeout_task: asyncio.Task[None] | None = None
        self._signal_pruner_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._hot_seed_retry_not_before: dict[str, float] = {}
        # Coalesced reactivation drain — replaces the per-WS-update
        # ``asyncio.create_task(_reactivate_deferred_signals_for_token)``
        # that produced 48 concurrent tasks during a 41.6 s event-loop
        # stall in production (5 h soak, 5/2026/05).  WS price updates
        # arrive at hundreds per second across all subscribed tokens;
        # spawning a task per update is unscalable, and they all
        # contend on ``self._lock``.  The drain coalesces pending
        # token IDs into a set and processes them in one pass under
        # the same lock.
        self._pending_reactivation_tokens: set[str] = set()
        self._reactivation_drain_task: asyncio.Task[None] | None = None
        self._reactivation_drain_event: asyncio.Event | None = None
        # Drop counters for queue saturation. Without this, a queue at
        # the 5000 cap silently blocks producers (every callsite of
        # _enqueue_projection awaits put()), turning a backpressure
        # signal into a stall on the trading loop. The drop path
        # (only triggered when backpressure has been published and
        # producers still didn't yield) is preferable to an unbounded
        # producer wait. Operator can read these via the SLO snapshot.
        self._projection_dropped_at_cap: int = 0
        self._projection_drop_log_at: float = 0.0
        self._projection_coalesced_on_enqueue: int = 0
        # Producer-side coalescing buffer: one live "upsert" payload per
        # (source, signal_type) is in-flight on the queue at a time. A
        # second enqueue for the same key mutates the queued payload in
        # place (merging snapshots / sweep_missing / keep_dedupe_keys)
        # rather than adding a new entry. Previously a burst of 40 upserts
        # from the same source took 40 queue slots; at 5000 cap the queue
        # could saturate in 3 minutes and drop projections. With coalescing,
        # the queue holds at most N_distinct_upsert_keys entries (~10).
        # The pending map is accessed only from the event loop, so no lock
        # is needed.
        self._pending_upsert_buffers: dict[tuple[str, str], dict[str, Any]] = {}
        # Decoupled emission-history writes. trade_signal_emissions is
        # append-only + loss-tolerant (offline backtester only, no live reader).
        # Writing it inline in the status projection held a main-pool connection
        # 2-3.5s per chunk under contention (the #1 recurring slow_tx), starving
        # orchestrator hot-path checkouts. Buffer here; a single background
        # flusher batch-inserts it, so each chunk releases its connection right
        # after the trade_signals UPDATE commits.
        self._emission_buffer: list[dict[str, Any]] = []
        self._emission_buffer_dropped: int = 0
        self._emission_drop_log_at: float = 0.0
        self._emission_flush_task: asyncio.Task[None] | None = None

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
        self._deferred_timeout_task = self._start_task(
            self._run_deferred_timeout_loop(),
            name="intent-runtime-deferred-timeout",
        )
        self._signal_pruner_task = self._start_task(
            self._run_signal_pruner_loop(),
            name="intent-runtime-signal-pruner",
        )
        self._emission_flush_task = self._start_task(
            self._run_emission_flush_loop(),
            name="intent-runtime-emission-flush",
        )
        self._register_ws_callback()
        await self.hydrate_from_db()

    async def stop(self) -> None:
        background_tasks = [
            task
            for task in self._background_tasks
            if task is not self._projection_task
            and task is not self._deferred_timeout_task
            and task is not self._signal_pruner_task
            and task is not self._emission_flush_task
            and not task.done()
        ]
        for task in background_tasks:
            task.cancel()
        for managed_task in (
            self._projection_task,
            self._deferred_timeout_task,
            self._signal_pruner_task,
            self._emission_flush_task,
        ):
            if managed_task is not None and not managed_task.done():
                managed_task.cancel()
                try:
                    await managed_task
                except asyncio.CancelledError:
                    pass
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        self._projection_task = None
        self._deferred_timeout_task = None
        self._signal_pruner_task = None
        self._emission_flush_task = None
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

    def _clear_deferred_state_locked(self, signal_id: str, *, clear_snapshot_state: bool = True) -> None:
        existing_tokens = self._deferred_tokens_by_signal_id.pop(signal_id, set())
        for token_id in existing_tokens:
            signal_ids = self._deferred_signal_ids_by_token.get(token_id)
            if not signal_ids:
                continue
            signal_ids.discard(signal_id)
            if not signal_ids:
                self._deferred_signal_ids_by_token.pop(token_id, None)
        if clear_snapshot_state:
            snapshot = self._signals_by_id.get(signal_id)
            if snapshot is not None:
                snapshot["deferred_until_ws"] = False
                snapshot["deferred_reason"] = None
                snapshot["deferred_started_at"] = None
                payload = snapshot.get("payload_json")
                if isinstance(payload, dict):
                    payload.pop(_DEFERRED_QUOTE_MIN_OBSERVED_AT_KEY, None)
                    payload.pop(_DEFERRED_REQUIRED_MAX_AGE_MS_KEY, None)

    def _set_deferred_state_locked(
        self,
        signal_id: str,
        *,
        required_token_ids: list[str],
        reason: str,
        min_observed_at_iso: str | None = None,
        required_max_age_ms: int | None = None,
    ) -> None:
        snapshot = self._signals_by_id.get(signal_id)
        deferred_started_at = None
        preserved_min_observed_at_iso: str | None = None
        preserved_required_max_age_ms: int | None = None
        if snapshot is not None and bool(snapshot.get("deferred_until_ws")):
            deferred_started_at = _to_iso(
                _normalize_datetime(snapshot.get("deferred_started_at"))
                or _normalize_datetime(snapshot.get("updated_at"))
                or utcnow()
            )
            payload = snapshot.get("payload_json")
            if isinstance(payload, dict):
                preserved_min_observed_at_iso = str(payload.get(_DEFERRED_QUOTE_MIN_OBSERVED_AT_KEY) or "").strip() or None
                try:
                    preserved_required_max_age_ms = int(payload.get(_DEFERRED_REQUIRED_MAX_AGE_MS_KEY))
                    if preserved_required_max_age_ms <= 0:
                        preserved_required_max_age_ms = None
                except Exception:
                    preserved_required_max_age_ms = None
        self._clear_deferred_state_locked(signal_id, clear_snapshot_state=False)
        token_set = {token_id for token_id in _normalize_token_ids(required_token_ids) if token_id}
        if token_set:
            self._deferred_tokens_by_signal_id[signal_id] = set(token_set)
            for token_id in token_set:
                self._deferred_signal_ids_by_token.setdefault(token_id, set()).add(signal_id)
        snapshot = self._signals_by_id.get(signal_id)
        if snapshot is not None:
            is_scanner_source = str(snapshot.get("source") or "").strip().lower() == "scanner"
            if is_scanner_source:
                preserved_min_observed_at_iso = None
            snapshot["required_token_ids"] = sorted(token_set)
            snapshot["deferred_until_ws"] = True
            snapshot["deferred_reason"] = str(reason or "strict_ws_context_missing")
            snapshot["deferred_started_at"] = deferred_started_at or _to_iso(utcnow())
            payload = snapshot.get("payload_json")
            if not isinstance(payload, dict):
                payload = {}
                snapshot["payload_json"] = payload
            if is_scanner_source:
                payload.pop("execution_armed_at", None)
                payload.pop(_DEFERRED_QUOTE_MIN_OBSERVED_AT_KEY, None)
            else:
                resolved_min_observed_at_iso = preserved_min_observed_at_iso or min_observed_at_iso
                normalized_reason = str(reason or "").strip().lower()
                if resolved_min_observed_at_iso is None and normalized_reason == "awaiting_post_arm_ws_tick":
                    resolved_min_observed_at_iso = str(payload.get("execution_armed_at") or "").strip() or None
                if resolved_min_observed_at_iso is None and normalized_reason in {
                    "awaiting_post_arm_ws_tick",
                    "prewarm_waiting_for_strict_ws_quote",
                    "strict_ws_pricing_live_context_unavailable",
                    "strict_ws_pricing_signal_release_stale",
                }:
                    resolved_min_observed_at_iso = snapshot["deferred_started_at"]
                if resolved_min_observed_at_iso:
                    payload[_DEFERRED_QUOTE_MIN_OBSERVED_AT_KEY] = str(resolved_min_observed_at_iso)
                else:
                    payload.pop(_DEFERRED_QUOTE_MIN_OBSERVED_AT_KEY, None)
            resolved_required_max_age_ms = preserved_required_max_age_ms
            if resolved_required_max_age_ms is None:
                try:
                    if required_max_age_ms is not None and int(required_max_age_ms) > 0:
                        resolved_required_max_age_ms = int(required_max_age_ms)
                except Exception:
                    resolved_required_max_age_ms = None
            if resolved_required_max_age_ms is not None:
                payload[_DEFERRED_REQUIRED_MAX_AGE_MS_KEY] = int(resolved_required_max_age_ms)
            else:
                payload.pop(_DEFERRED_REQUIRED_MAX_AGE_MS_KEY, None)

    def _tokens_have_fresh_ws_quotes(
        self,
        token_ids: list[str],
        *,
        source: str | None = None,
        min_observed_at_epoch: float | None = None,
        max_age_seconds: float | None = None,
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
        strict_ttl = (
            max(0.01, float(max_age_seconds))
            if max_age_seconds is not None
            else _strict_ws_ttl_seconds_for_source(source)
        )
        normalized_source = str(source or "").strip().lower()
        for token_id in normalized:
            try:
                quote_current = feed_manager.cache.is_fresh(token_id, max_age_seconds=strict_ttl)
                if not quote_current and normalized_source == "scanner" and min_observed_at_epoch is None:
                    quote_current = bool(
                        feed_manager.has_current_subscription_price(
                            token_id,
                            max_age_seconds=strict_ttl,
                            allow_stale_subscribed=True,
                        )
                    )
                if not quote_current:
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

    async def _attach_live_market_contexts(
        self,
        snapshots: dict[str, dict[str, Any]],
    ) -> None:
        if not snapshots:
            return
        for snapshot in snapshots.values():
            payload = snapshot.get("payload_json")
            if not isinstance(payload, dict):
                payload = {}
            else:
                payload = dict(payload)
            payload.pop(_LIVE_MARKET_PAYLOAD_KEY, None)
            snapshot["payload_json"] = payload
        try:
            signal_views = [_snapshot_signal_view(snapshot) for snapshot in snapshots.values()]
            live_contexts = await build_cached_live_signal_contexts(
                signal_views,
                max_history_points=20,
                strict_ws_only=True,
            )
        except Exception as exc:
            logger.debug("Intent runtime activation live-context attach failed", exc_info=exc)
            live_contexts = {}

        for signal_id, live_context in live_contexts.items():
            snapshot = snapshots.get(str(signal_id or "").strip())
            if snapshot is None or not isinstance(live_context, dict) or not live_context:
                continue
            payload = snapshot.get("payload_json")
            if not isinstance(payload, dict):
                payload = {}
            else:
                payload = dict(payload)
            payload[_LIVE_MARKET_PAYLOAD_KEY] = copy.deepcopy(live_context)
            snapshot["payload_json"] = payload

        for signal_id, snapshot in snapshots.items():
            if str(signal_id or "").strip() in live_contexts:
                continue
            live_context = self._build_minimal_scanner_live_market_context(snapshot)
            if not isinstance(live_context, dict) or not live_context:
                continue
            payload = snapshot.get("payload_json")
            if not isinstance(payload, dict):
                payload = {}
            else:
                payload = dict(payload)
            payload[_LIVE_MARKET_PAYLOAD_KEY] = live_context
            snapshot["payload_json"] = payload

    def _build_minimal_scanner_live_market_context(
        self,
        snapshot: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not _snapshot_requires_scanner_strict_live_market(snapshot):
            return None
        required_token_ids = _normalize_token_ids(snapshot.get("required_token_ids") or [])
        if not required_token_ids:
            return None
        source = str(snapshot.get("source") or "").strip().lower()
        if not self._tokens_have_fresh_ws_quotes(required_token_ids, source=source):
            return None
        try:
            feed_manager = get_feed_manager()
        except Exception:
            return None
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return None

        payload = snapshot.get("payload_json")
        payload = payload if isinstance(payload, dict) else {}
        selected_token_id = str(
            payload.get("selected_token_id")
            or payload.get("token_id")
            or (required_token_ids[0] if required_token_ids else "")
        ).strip().lower()
        if not selected_token_id:
            return None
        try:
            live_selected_price = _safe_float(feed_manager.cache.get_mid_price(selected_token_id))
        except Exception:
            return None
        if live_selected_price is None or live_selected_price <= 0.0:
            return None

        observed_at_epoch = None
        try:
            observed_at_epoch = feed_manager.cache.get_observed_at_epoch(selected_token_id)
        except Exception:
            observed_at_epoch = None
        if observed_at_epoch is None:
            try:
                staleness_seconds = _safe_float(feed_manager.cache.staleness(selected_token_id))
            except Exception:
                staleness_seconds = None
            if staleness_seconds is None:
                return None
            observed_at_epoch = time.time() - max(0.0, float(staleness_seconds))

        observed_at_epoch = float(observed_at_epoch)
        now_epoch = time.time()
        market_data_age_ms = max(0.0, (now_epoch - observed_at_epoch) * 1000.0)
        strict_max_age_ms = max(100, int(_strict_ws_ttl_seconds_for_source(source) * 1000.0))
        ws_subscription_current = False
        if market_data_age_ms > strict_max_age_ms:
            try:
                ws_subscription_current = bool(
                    feed_manager.has_current_subscription_price(
                        selected_token_id,
                        max_age_seconds=(strict_max_age_ms / 1000.0),
                        allow_stale_subscribed=True,
                    )
                )
            except Exception:
                ws_subscription_current = False
            if not ws_subscription_current:
                return None

        signal_entry_price = _safe_float(snapshot.get("entry_price"))
        entry_price_delta = (
            float(live_selected_price) - float(signal_entry_price)
            if signal_entry_price is not None
            else None
        )
        entry_price_delta_pct = (
            ((entry_price_delta / float(signal_entry_price)) * 100.0)
            if entry_price_delta is not None and signal_entry_price not in {None, 0.0}
            else None
        )
        return {
            "available": True,
            "market_id": str(snapshot.get("market_id") or "").strip() or None,
            "market_question": str(snapshot.get("market_question") or "").strip() or None,
            "direction": str(snapshot.get("direction") or "").strip().lower() or None,
            "token_ids": list(required_token_ids),
            "selected_token_id": selected_token_id,
            "live_selected_price": float(live_selected_price),
            "live_selected_price_source": "ws_strict",
            "market_data_source": "ws_strict",
            "source_observed_at": _to_iso(datetime.fromtimestamp(observed_at_epoch, tz=timezone.utc)),
            "market_data_age_ms": market_data_age_ms,
            "market_data_age_seconds": market_data_age_ms / 1000.0,
            "ws_subscription_current": ws_subscription_current,
            "signal_entry_price": signal_entry_price,
            "entry_price_delta": entry_price_delta,
            "entry_price_delta_pct": entry_price_delta_pct,
            "live_edge_percent": None,
        }

    async def _defer_scanner_snapshots_without_strict_live_market(
        self,
        snapshots: dict[str, dict[str, Any]],
        *,
        projection_snapshots: dict[str, dict[str, Any]],
        event_types: dict[str, str] | None = None,
        reason: str,
    ) -> None:
        if not snapshots:
            return
        required_max_age_ms = max(100, int(_strict_ws_ttl_seconds_for_source("scanner") * 1000.0))
        deferred_signal_ids: list[str] = []
        for signal_id, snapshot in list(snapshots.items()):
            if _snapshot_has_strict_scanner_live_market(snapshot):
                continue
            if not _snapshot_requires_scanner_strict_live_market(snapshot):
                continue
            _clear_snapshot_live_market(snapshot)
            projection_snapshot = projection_snapshots.get(signal_id)
            if isinstance(projection_snapshot, dict):
                _clear_snapshot_live_market(projection_snapshot)
            deferred = await self.defer_signal(
                signal_id=signal_id,
                required_token_ids=list(snapshot.get("required_token_ids") or []),
                reason=reason,
                required_max_age_ms=required_max_age_ms,
                clear_live_market=True,
            )
            if not deferred:
                continue
            deferred_signal_ids.append(signal_id)
            snapshots.pop(signal_id, None)
            projection_snapshots.pop(signal_id, None)
            if event_types is not None:
                event_types.pop(signal_id, None)
        if deferred_signal_ids:
            logger.debug(
                "Deferred scanner snapshots without strict live market context",
                signal_ids=deferred_signal_ids,
                reason=reason,
            )

    def _snapshot_ready_for_runtime(self, snapshot: dict[str, Any]) -> bool:
        deferred_reason = str(snapshot.get("deferred_reason") or "")
        source = str(snapshot.get("source") or "").strip().lower()
        payload = snapshot.get("payload_json")
        payload = payload if isinstance(payload, dict) else {}
        min_observed_at_epoch = None
        deferred_min_observed_at = _normalize_datetime(payload.get(_DEFERRED_QUOTE_MIN_OBSERVED_AT_KEY))
        if deferred_min_observed_at is not None:
            min_observed_at_epoch = deferred_min_observed_at.timestamp()
        max_age_seconds = None
        try:
            required_max_age_ms = payload.get(_DEFERRED_REQUIRED_MAX_AGE_MS_KEY)
            if required_max_age_ms is not None and int(required_max_age_ms) > 0:
                max_age_seconds = max(0.01, int(required_max_age_ms) / 1000.0)
        except Exception:
            max_age_seconds = None
        if source == "scanner":
            return self._tokens_have_fresh_ws_quotes(
                list(snapshot.get("required_token_ids") or []),
                source=source,
                max_age_seconds=max_age_seconds,
            )
        if deferred_reason == "awaiting_post_arm_ws_tick":
            if min_observed_at_epoch is None:
                armed_at = _normalize_datetime(payload.get("execution_armed_at"))
                if armed_at is not None:
                    min_observed_at_epoch = armed_at.timestamp()
            return self._tokens_have_fresh_ws_quotes(
                list(snapshot.get("required_token_ids") or []),
                source=str(snapshot.get("source") or ""),
                min_observed_at_epoch=min_observed_at_epoch,
                max_age_seconds=max_age_seconds,
            )
        if deferred_reason in {
            "prewarm_waiting_for_strict_ws_quote",
            "strict_ws_pricing_live_context_unavailable",
            "strict_ws_pricing_signal_release_stale",
        }:
            return self._tokens_have_fresh_ws_quotes(
                list(snapshot.get("required_token_ids") or []),
                source=str(snapshot.get("source") or ""),
                min_observed_at_epoch=min_observed_at_epoch,
                max_age_seconds=max_age_seconds,
            )
        return self._tokens_have_fresh_ws_quotes(
            list(snapshot.get("required_token_ids") or []),
            source=str(snapshot.get("source") or ""),
            max_age_seconds=max_age_seconds,
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
            polymarket_feed = getattr(feed_manager, "polymarket_feed", None)
            live_subscribed = {
                str(token_id or "").strip().lower()
                for token_id in list(getattr(polymarket_feed, "_subscribed_assets", set()) or set())
                if str(token_id or "").strip()
            }
            missing = [
                token_id
                for token_id in normalized
                if token_id not in self._hot_subscription_tokens or token_id not in live_subscribed
            ]
            if missing:
                await polymarket_feed.subscribe(missing)
                self._hot_subscription_tokens.update(missing)
            seed_order_book = getattr(feed_manager, "get_order_book", None)
            if callable(seed_order_book):
                now_mono = time.monotonic()
                seed_candidates: list[str] = []
                for token_id in normalized:
                    try:
                        if feed_manager.cache.get_mid_price(token_id) is not None:
                            continue
                    except Exception:
                        continue
                    retry_not_before = float(self._hot_seed_retry_not_before.get(token_id, 0.0) or 0.0)
                    if retry_not_before > now_mono:
                        continue
                    self._hot_seed_retry_not_before[token_id] = now_mono + _HOT_SUBSCRIPTION_SEED_RETRY_SECONDS
                    seed_candidates.append(token_id)
                if seed_candidates:
                    # Bounded worker pool — was ``[task for t in
                    # candidates] + asyncio.gather(*tasks)`` with a
                    # Semaphore.  Production watchdog caught 141 tasks
                    # parked at this line during a token prewarm
                    # burst, drowning the trading loop.  Live count
                    # is now exactly _HOT_SUBSCRIPTION_SEED_CONCURRENCY
                    # regardless of input.
                    seed_queue: asyncio.Queue = asyncio.Queue()
                    for token_id in seed_candidates:
                        seed_queue.put_nowait(token_id)

                    async def _seed_worker() -> None:
                        while True:
                            try:
                                token_id = seed_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                return
                            try:
                                await seed_order_book(token_id)
                            except Exception:
                                pass
                            finally:
                                seed_queue.task_done()
                                await asyncio.sleep(0)

                    seed_worker_count = max(
                        1,
                        min(
                            _HOT_SUBSCRIPTION_SEED_CONCURRENCY,
                            len(seed_candidates),
                        ),
                    )
                    seed_workers = [
                        asyncio.create_task(
                            _seed_worker(), name=f"intent-runtime-seed-{i}"
                        )
                        for i in range(seed_worker_count)
                    ]
                    try:
                        await asyncio.gather(*seed_workers, return_exceptions=True)
                    finally:
                        for w in seed_workers:
                            if not w.done():
                                w.cancel()
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
            for token_id in required:
                normalized_token = str(token_id or "").strip().lower()
                if not normalized_token or normalized_token in seen:
                    continue
                seen.add(normalized_token)
                token_ids.append(normalized_token)
                if normalized_token in wait_seen:
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
        # Coalesced reactivation: add the token to a pending-set and
        # wake the single drain task.  Pre-coalesce production saw
        # 48 concurrent ``_reactivate_deferred_signals_for_token``
        # tasks during a 41.6 s event-loop stall — one task per WS
        # price update, each contending on ``self._lock``.  The drain
        # processes the entire pending set in one lock acquisition.
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
            self._enqueue_reactivation_locked_or_loop(normalized_token)
            return
        loop.call_soon_threadsafe(
            self._enqueue_reactivation_locked_or_loop, normalized_token
        )

    def _enqueue_reactivation_locked_or_loop(self, normalized_token: str) -> None:
        """Add a token to the pending-reactivation set and wake the
        drain task.  Called either directly from the event loop
        (synchronous WS callback path) or from
        ``call_soon_threadsafe``.  Idempotent — repeated calls for
        the same token before the drain runs collapse into one."""
        self._pending_reactivation_tokens.add(normalized_token)
        # Lazy-create the drain event + task on first use so we don't
        # need an explicit start hook.
        if self._reactivation_drain_event is None:
            self._reactivation_drain_event = asyncio.Event()
        self._reactivation_drain_event.set()
        if self._reactivation_drain_task is None or self._reactivation_drain_task.done():
            self._reactivation_drain_task = self._start_task(
                self._reactivation_drain_loop(),
                name="intent-runtime-reactivation-drain",
            )

    async def _reactivation_drain_loop(self) -> None:
        """Single long-lived task that drains the pending-reactivation
        set.  Wakes on the drain event, snapshots the pending set
        under no-lock (set ops are atomic in CPython), processes each
        token, then sleeps until the next set."""
        while True:
            event = self._reactivation_drain_event
            if event is None:
                # Should not happen — the event is created by the
                # enqueue path before the task is spawned.  Defensive.
                return
            try:
                await event.wait()
            except asyncio.CancelledError:
                return
            # Snapshot + clear in one step so concurrent enqueues during
            # the drain re-arm the event for the next iteration.
            pending = self._pending_reactivation_tokens
            self._pending_reactivation_tokens = set()
            event.clear()
            if not pending:
                continue
            # Process each token sequentially.  Was 48 concurrent
            # tasks contending on ``self._lock``; now one task pulls
            # the lock once per token in series.  Token count per
            # drain cycle is bounded by the WS update rate within
            # one event-loop tick — typically 1-5 tokens, occasionally
            # spiking to dozens.  Sequential is fine; the lock would
            # serialize them anyway.
            for token in pending:
                try:
                    await self._reactivate_deferred_signals_for_token(token)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Deferred-signal reactivation failed token=%s: %s",
                        token,
                        exc,
                    )

    async def _reactivate_deferred_signals_for_token(self, token_id: str) -> None:
        normalized_token = str(token_id or "").strip().lower()
        if not normalized_token:
            return
        now = utcnow()
        published_by_source: dict[str, dict[str, dict[str, Any]]] = {}
        projection_snapshots: dict[str, dict[str, Any]] = {}
        expired_snapshots: dict[str, dict[str, Any]] = {}
        async with self._lock:
            signal_ids = sorted(self._deferred_signal_ids_by_token.get(normalized_token, set()))
            for signal_id in signal_ids:
                snapshot = self._signals_by_id.get(signal_id)
                if snapshot is None:
                    self._clear_deferred_state_locked(signal_id)
                    continue
                if _deferred_reason_is_nonreactivatable(snapshot):
                    self._clear_deferred_state_locked(signal_id)
                    snapshot["status"] = "expired"
                    snapshot["runtime_sequence"] = None
                    snapshot["deferred_until_ws"] = False
                    snapshot["deferred_reason"] = None
                    snapshot["updated_at"] = _to_iso(now)
                    expired_snapshots[signal_id] = copy.deepcopy(snapshot)
                    continue
                if not self._snapshot_ready_for_runtime(snapshot):
                    continue
                self._clear_deferred_state_locked(signal_id)
                snapshot["status"] = "pending"
                snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                _ensure_execution_armed_at(snapshot, _to_iso(now))
                snapshot["deferred_until_ws"] = False
                snapshot["deferred_reason"] = None
                snapshot["updated_at"] = _to_iso(now)
                published_by_source.setdefault(str(snapshot.get("source") or ""), {})[signal_id] = copy.deepcopy(snapshot)
                projection_snapshots[signal_id] = copy.deepcopy(snapshot)
        for source_key, snapshots in published_by_source.items():
            await self._attach_live_market_contexts(snapshots)
            await self._defer_scanner_snapshots_without_strict_live_market(
                snapshots,
                projection_snapshots=projection_snapshots,
                reason="strict_ws_pricing_live_context_unavailable",
            )
            for signal_id, projection_snapshot in projection_snapshots.items():
                live_market = (snapshots.get(signal_id, {}).get("payload_json") or {}).get(_LIVE_MARKET_PAYLOAD_KEY)
                if not isinstance(live_market, dict):
                    continue
                payload = projection_snapshot.get("payload_json")
                if not isinstance(payload, dict):
                    payload = {}
                else:
                    payload = dict(payload)
                payload[_LIVE_MARKET_PAYLOAD_KEY] = copy.deepcopy(live_market)
                projection_snapshot["payload_json"] = payload
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
        if expired_snapshots:
            await self._enqueue_projection(
                {
                    "kind": "upsert",
                    "source": "__deferred__",
                    "snapshots": expired_snapshots,
                    "sweep_missing": False,
                    "keep_dedupe_keys": [],
                }
            )
            await self._publish_signal_stats()

    async def _run_deferred_timeout_loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            try:
                await self._release_stale_deferred_signals()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Deferred timeout sweep error", exc_info=exc)

    async def _run_signal_pruner_loop(self) -> None:
        """Periodic terminal-signal pruner.

        Removes terminal signals (executed/skipped/expired/failed) past
        the 24h reactivation lookback from the in-memory map.  Without
        this loop, every signal the runtime ever sees stays resident
        forever — a steady leak proportional to signal throughput.
        Also enforces ``_MAX_SIGNALS_IN_MEMORY`` and
        ``_MAX_HOT_SEED_RETRIES`` ceilings as a safety net.
        """
        while True:
            await asyncio.sleep(self._PRUNE_INTERVAL_SECONDS)
            try:
                await self._prune_in_memory_signals()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Intent runtime signal prune error", exc_info=exc)

    async def _prune_in_memory_signals(self) -> None:
        now = utcnow()
        cutoff = now - timedelta(hours=_BOOTSTRAP_REACTIVATABLE_LOOKBACK_HOURS)
        removed_terminal = 0
        removed_overflow = 0
        async with self._lock:
            terminal_to_drop: list[tuple[datetime, str]] = []
            for signal_id, snapshot in self._signals_by_id.items():
                status = str(snapshot.get("status") or "").strip().lower()
                if status not in _SIGNAL_TERMINAL_STATUSES:
                    continue
                updated_at = _normalize_datetime(snapshot.get("updated_at")) or _normalize_datetime(
                    snapshot.get("created_at")
                )
                if updated_at is None or updated_at < cutoff:
                    # Treat undated terminals as ancient — they're
                    # effectively unreactivatable and should be evicted.
                    terminal_to_drop.append((updated_at or datetime.min.replace(tzinfo=timezone.utc), signal_id))
            for _, signal_id in terminal_to_drop:
                self._drop_signal_locked(signal_id)
                removed_terminal += 1

            # Hard cap fallback: if the table is still oversized after
            # pruning expired terminals (e.g. high-frequency runtime
            # adds outpacing the prune cadence), drop oldest-by-
            # updated_at terminal signals until under the ceiling.
            overflow = len(self._signals_by_id) - self._MAX_SIGNALS_IN_MEMORY
            if overflow > 0:
                terminal_candidates: list[tuple[datetime, str]] = []
                for signal_id, snapshot in self._signals_by_id.items():
                    status = str(snapshot.get("status") or "").strip().lower()
                    if status not in _SIGNAL_TERMINAL_STATUSES:
                        continue
                    updated_at = _normalize_datetime(snapshot.get("updated_at")) or _normalize_datetime(
                        snapshot.get("created_at")
                    ) or datetime.min.replace(tzinfo=timezone.utc)
                    terminal_candidates.append((updated_at, signal_id))
                terminal_candidates.sort()
                for _, signal_id in terminal_candidates[:overflow]:
                    self._drop_signal_locked(signal_id)
                    removed_overflow += 1

            # Last-resort: if STILL oversized after dropping every
            # eligible terminal, drop the oldest non-terminals too.
            # This handles the pathological case where signals are
            # stuck in pending/deferred because the orchestrator
            # couldn't process them in time (e.g. during event-loop
            # stalls in production 5/2026/05 — 41 s loop freezes
            # backed up the deferred queue).  Dropping them here is
            # a memory-safety guard: the canonical signal still
            # lives in trade_signals on DB, and a worker restart
            # will reload the recent ones via bootstrap.  Logged as
            # a warning so this branch firing surfaces in audit.
            removed_non_terminal_overflow = 0
            still_overflow = len(self._signals_by_id) - self._MAX_SIGNALS_IN_MEMORY
            if still_overflow > 0:
                non_terminal_candidates: list[tuple[datetime, str]] = []
                for signal_id, snapshot in self._signals_by_id.items():
                    status = str(snapshot.get("status") or "").strip().lower()
                    if status in _SIGNAL_TERMINAL_STATUSES:
                        continue
                    updated_at = _normalize_datetime(snapshot.get("updated_at")) or _normalize_datetime(
                        snapshot.get("created_at")
                    ) or datetime.min.replace(tzinfo=timezone.utc)
                    non_terminal_candidates.append((updated_at, signal_id))
                non_terminal_candidates.sort()
                for _, signal_id in non_terminal_candidates[:still_overflow]:
                    self._drop_signal_locked(signal_id)
                    removed_non_terminal_overflow += 1
                if removed_non_terminal_overflow:
                    logger.warning(
                        "Intent runtime evicted non-terminal signals to "
                        "stay under memory cap (orchestrator may be "
                        "falling behind on signal processing)",
                        non_terminal_evicted=removed_non_terminal_overflow,
                        signals_remaining=len(self._signals_by_id),
                    )

            # Bound the WS hot-seed retry-cooldown table.  Entries are
            # never removed when a token leaves the active universe;
            # the simplest cap is to clear the whole table when over
            # the ceiling (next subscription cycle re-seeds the tokens
            # that are still active).  Cheaper than tracking liveness.
            if len(self._hot_seed_retry_not_before) > self._MAX_HOT_SEED_RETRIES:
                self._hot_seed_retry_not_before.clear()
        if removed_terminal or removed_overflow:
            logger.info(
                "Intent runtime pruned in-memory signals",
                terminal_aged_out=removed_terminal,
                overflow_dropped=removed_overflow,
                signals_remaining=len(self._signals_by_id),
            )

    def _drop_signal_locked(self, signal_id: str) -> None:
        """Remove a signal from all secondary indices.  Caller holds ``self._lock``."""
        snapshot = self._signals_by_id.pop(signal_id, None)
        if snapshot is None:
            return
        dedupe_key = str(snapshot.get("dedupe_key") or "")
        if dedupe_key and self._signal_ids_by_dedupe_key.get(dedupe_key) == signal_id:
            self._signal_ids_by_dedupe_key.pop(dedupe_key, None)
        source = str(snapshot.get("source") or "")
        if source:
            source_ids = self._source_signal_ids.get(source)
            if source_ids is not None:
                source_ids.discard(signal_id)
                if not source_ids:
                    self._source_signal_ids.pop(source, None)
        self._clear_deferred_state_locked(signal_id)

    async def _release_stale_deferred_signals(self) -> None:
        max_age = max(5.0, float(getattr(settings, "INTENT_RUNTIME_DEFERRED_MAX_AGE_SECONDS", 15.0) or 15.0))
        now = utcnow()
        published_by_source: dict[str, dict[str, dict[str, Any]]] = {}
        projection_snapshots: dict[str, dict[str, Any]] = {}
        expired_snapshots: dict[str, dict[str, Any]] = {}
        quote_gated_reasons = {
            "awaiting_post_arm_ws_tick",
            "prewarm_waiting_for_strict_ws_quote",
            "strict_ws_pricing_live_context_unavailable",
        }
        async with self._lock:
            all_deferred_signal_ids: set[str] = set()
            for token_signal_ids in self._deferred_signal_ids_by_token.values():
                all_deferred_signal_ids.update(token_signal_ids)
            for signal_id in sorted(all_deferred_signal_ids):
                snapshot = self._signals_by_id.get(signal_id)
                if snapshot is None:
                    self._clear_deferred_state_locked(signal_id)
                    continue
                source = str(snapshot.get("source") or "").strip().lower()
                deferred_started_at = _normalize_datetime(snapshot.get("deferred_started_at"))
                if deferred_started_at is None:
                    deferred_started_at = _normalize_datetime(snapshot.get("updated_at"))
                if deferred_started_at is None:
                    deferred_started_at = _normalize_datetime(snapshot.get("created_at"))
                if deferred_started_at is None:
                    continue
                age_seconds = (now - deferred_started_at).total_seconds()
                required_age_seconds = 0.0 if source == "scanner" else max_age
                if age_seconds < required_age_seconds:
                    continue
                deferred_reason = str(snapshot.get("deferred_reason") or "").strip().lower()
                if _deferred_reason_is_nonreactivatable(snapshot):
                    self._clear_deferred_state_locked(signal_id)
                    snapshot["status"] = "expired"
                    snapshot["runtime_sequence"] = None
                    snapshot["deferred_until_ws"] = False
                    snapshot["deferred_reason"] = None
                    snapshot["updated_at"] = _to_iso(now)
                    expired_snapshots[signal_id] = copy.deepcopy(snapshot)
                    continue
                if deferred_reason in quote_gated_reasons and not self._snapshot_ready_for_runtime(snapshot):
                    continue
                self._clear_deferred_state_locked(signal_id)
                snapshot["status"] = "pending"
                snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                _ensure_execution_armed_at(snapshot, _to_iso(now))
                snapshot["deferred_until_ws"] = False
                snapshot["deferred_reason"] = None
                snapshot["updated_at"] = _to_iso(now)
                published_by_source.setdefault(str(snapshot.get("source") or ""), {})[signal_id] = copy.deepcopy(snapshot)
                projection_snapshots[signal_id] = copy.deepcopy(snapshot)
        if not published_by_source:
            return
        released_count = sum(len(v) for v in published_by_source.values())
        logger.info("Released %d deferred signals past max age %.1fs", released_count, max_age)
        for source_key, snapshots in published_by_source.items():
            await self._attach_live_market_contexts(snapshots)
            await self._defer_scanner_snapshots_without_strict_live_market(
                snapshots,
                projection_snapshots=projection_snapshots,
                reason="strict_ws_pricing_live_context_unavailable",
            )
            for signal_id, projection_snapshot in projection_snapshots.items():
                live_market = (snapshots.get(signal_id, {}).get("payload_json") or {}).get(_LIVE_MARKET_PAYLOAD_KEY)
                if not isinstance(live_market, dict):
                    continue
                payload = projection_snapshot.get("payload_json")
                if not isinstance(payload, dict):
                    payload = {}
                else:
                    payload = dict(payload)
                payload[_LIVE_MARKET_PAYLOAD_KEY] = copy.deepcopy(live_market)
                projection_snapshot["payload_json"] = payload
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
                trigger="intent_runtime_deferred_timeout",
                reason="deferred_max_age_exceeded",
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
        if expired_snapshots:
            await self._enqueue_projection(
                {
                    "kind": "upsert",
                    "source": "__deferred__",
                    "snapshots": expired_snapshots,
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
        min_observed_at: Any = None,
        required_max_age_ms: int | None = None,
        clear_live_market: bool = False,
    ) -> bool:
        normalized_signal_id = str(signal_id or "").strip()
        if not normalized_signal_id:
            return False
        snapshot_copy: dict[str, Any] | None = None
        token_ids: list[str] = []
        min_observed_at_iso = _to_iso(_normalize_datetime(min_observed_at))
        async with self._lock:
            snapshot = self._signals_by_id.get(normalized_signal_id)
            if snapshot is None:
                return False
            if clear_live_market:
                _clear_snapshot_live_market(snapshot)
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
                min_observed_at_iso=min_observed_at_iso,
                required_max_age_ms=required_max_age_ms,
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
        if token_ids:
            self._start_task(
                self._ensure_hot_subscriptions(token_ids),
                name=f"intent-runtime-defer-prewarm-{normalized_signal_id}",
            )
        return True

    async def hydrate_from_db(self) -> None:
        now = utcnow()
        active_statuses = tuple(sorted(_SIGNAL_ACTIVE_STATUSES))
        terminal_statuses = tuple(sorted(_SIGNAL_TERMINAL_STATUSES))
        terminal_cutoff = now - timedelta(hours=_BOOTSTRAP_REACTIVATABLE_LOOKBACK_HOURS)
        # Cast the three JSON columns to text in the SELECT.  The
        # ``trade_signals`` JSON columns total ~6.7 GB across the table;
        # asyncpg's default behaviour decodes each row's JSON synchronously
        # on the event loop, which can block for many seconds when this
        # bootstrap fetch returns thousands of rows.  Decoding the text
        # in ``asyncio.to_thread`` keeps the loop responsive.
        payload_text_col = cast(TradeSignal.payload_json, Text).label("payload_json_text")
        strategy_context_text_col = cast(
            TradeSignal.strategy_context_json, Text
        ).label("strategy_context_json_text")
        quality_rejection_reasons_text_col = cast(
            TradeSignal.quality_rejection_reasons, Text
        ).label("quality_rejection_reasons_text")
        async with AsyncSessionLocal() as session:
            # The bootstrap fetch from ``trade_signals`` reads up to
            # several thousand active + recent-terminal rows from a
            # 6.7 GB JSON-heavy table.  Under live load this routinely
            # exceeds the regular pool's 30s statement_timeout —
            # observed as "Connection held for 31.2s task=Task-62"
            # right after every restart, with the bootstrap raising
            # DBAPIError (statement_timeout) and intent-runtime
            # initialization failing.  This is a one-time startup
            # hydration, not a hot-path query, so a longer cap is
            # appropriate.  ``SET LOCAL`` only applies to this
            # transaction's connection.
            from sqlalchemy import text as _sa_text

            await session.execute(_sa_text("SET LOCAL statement_timeout = '300000'"))
            # ITER-? (Fix II): also probe ``trader_signal_cursor`` for
            # the highest consumed-runtime-sequence so the rebuilt
            # ``_next_runtime_sequence`` never lands below a trader's
            # cursor (which IS persisted across restarts).  Used inside
            # the lock below alongside the per-row hydrated sequences.
            cursor_max_seq = 0
            try:
                cursor_max_row = (
                    await session.execute(
                        _sa_text("SELECT COALESCE(MAX(last_runtime_sequence), 0) AS max_seq FROM trader_signal_cursor")
                    )
                ).first()
                cursor_max_seq = int(cursor_max_row.max_seq) if cursor_max_row else 0
            except Exception as _seq_seed_exc:
                logger.debug(
                    "Cursor-based runtime_sequence seed query failed (non-fatal); "
                    "falling back to trade_signals scan",
                    exc_info=_seq_seed_exc,
                )
            raw_rows = (
                (
                    await session.execute(
                        select(
                            TradeSignal.id,
                            TradeSignal.source,
                            TradeSignal.source_item_id,
                            TradeSignal.signal_type,
                            TradeSignal.strategy_type,
                            TradeSignal.market_id,
                            TradeSignal.market_question,
                            TradeSignal.direction,
                            TradeSignal.entry_price,
                            TradeSignal.effective_price,
                            TradeSignal.edge_percent,
                            TradeSignal.confidence,
                            TradeSignal.liquidity,
                            TradeSignal.expires_at,
                            TradeSignal.status,
                            payload_text_col,
                            strategy_context_text_col,
                            TradeSignal.quality_passed,
                            quality_rejection_reasons_text_col,
                            TradeSignal.dedupe_key,
                            TradeSignal.runtime_sequence,
                            TradeSignal.created_at,
                            TradeSignal.updated_at,
                        ).where(
                            or_(
                                TradeSignal.status.in_(active_statuses),
                                and_(
                                    TradeSignal.status.in_(terminal_statuses),
                                    TradeSignal.updated_at >= terminal_cutoff,
                                ),
                            )
                        )
                    )
                )
                .mappings()
                .all()
            )

        import json as _json

        def _decode_rows() -> list[dict[str, Any]]:
            decoded: list[dict[str, Any]] = []
            for raw in raw_rows:
                row = dict(raw)
                for src_key, dst_key in (
                    ("payload_json_text", "payload_json"),
                    ("strategy_context_json_text", "strategy_context_json"),
                    ("quality_rejection_reasons_text", "quality_rejection_reasons"),
                ):
                    text_value = row.pop(src_key, None)
                    if text_value:
                        try:
                            row[dst_key] = _json.loads(text_value)
                        except Exception:
                            row[dst_key] = None
                    else:
                        row[dst_key] = None
                decoded.append(row)
            return decoded

        rows = await asyncio.to_thread(_decode_rows)
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
                # Plan 0041: recover intended_trader_id from the persisted
                # payload_json so the trader-scope filter survives worker
                # restart. Empty / missing -> None (legacy multi-trader
                # visibility).
                payload_dict_for_intent = row.get("payload_json") or {}
                intent_trader_raw = (
                    payload_dict_for_intent.get("intended_trader_id")
                    if isinstance(payload_dict_for_intent, dict)
                    else None
                )
                hydrated_intended_trader_id = (
                    str(intent_trader_raw).strip()
                    if intent_trader_raw is not None
                    else ""
                ) or None
                snapshot = {
                    "id": str(row.get("id") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                    "source_item_id": str(row.get("source_item_id") or "").strip(),
                    "signal_type": str(row.get("signal_type") or "").strip(),
                    "strategy_type": str(row.get("strategy_type") or "").strip(),
                    "intended_trader_id": hydrated_intended_trader_id,
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
                    "deferred_started_at": None,
                    "created_at": _to_iso(_normalize_datetime(row.get("created_at"))),
                    "updated_at": _to_iso(_normalize_datetime(row.get("updated_at"))),
                }
                if not snapshot["id"]:
                    continue
                source = str(snapshot.get("source") or "").strip().lower()
                if source == "scanner":
                    payload = snapshot.get("payload_json")
                    if isinstance(payload, dict):
                        payload = dict(payload)
                        payload.pop("execution_armed_at", None)
                        payload.pop(_DEFERRED_QUOTE_MIN_OBSERVED_AT_KEY, None)
                        payload.pop(_DEFERRED_REQUIRED_MAX_AGE_MS_KEY, None)
                        snapshot["payload_json"] = payload
                status = str(snapshot.get("status") or "").strip().lower()
                updated_at = _normalize_datetime(row.get("updated_at")) or _normalize_datetime(row.get("created_at"))
                if status in _SIGNAL_TERMINAL_STATUSES and (updated_at is None or updated_at < terminal_cutoff):
                    continue
                runtime = _signal_runtime_metadata(
                    snapshot.get("payload_json"),
                    snapshot.get("strategy_context_json"),
                )
                activation = str(runtime.get("execution_activation") or "").strip().lower()
                defer_reason: str | None = None
                if snapshot["runtime_sequence"] is None and status in _SIGNAL_ACTIVE_STATUSES:
                    if source == "scanner" and snapshot["required_token_ids"]:
                        if self._tokens_have_fresh_ws_quotes(
                            snapshot["required_token_ids"],
                            source=source,
                        ):
                            snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                            _ensure_execution_armed_at(snapshot, _to_iso(now))
                            bootstrap_snapshots[snapshot["id"]] = copy.deepcopy(snapshot)
                        else:
                            defer_reason = "prewarm_waiting_for_strict_ws_quote"
                    elif activation == "ws_post_arm_tick" and snapshot["required_token_ids"]:
                        payload = snapshot.get("payload_json")
                        if not isinstance(payload, dict):
                            payload = {}
                        else:
                            payload = dict(payload)
                        payload["execution_armed_at"] = str(
                            payload.get("execution_armed_at")
                            or snapshot.get("updated_at")
                            or snapshot.get("created_at")
                            or _to_iso(utcnow())
                        )
                        snapshot["payload_json"] = payload
                        defer_reason = "awaiting_post_arm_ws_tick"
                    else:
                        snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                        _ensure_execution_armed_at(snapshot, _to_iso(now))
                        bootstrap_snapshots[snapshot["id"]] = copy.deepcopy(snapshot)
                sequence = _normalize_runtime_sequence(snapshot.get("runtime_sequence"))
                if sequence is not None:
                    self._next_runtime_sequence = max(self._next_runtime_sequence, sequence + 1)
                self._signals_by_id[snapshot["id"]] = snapshot
                if snapshot["dedupe_key"]:
                    self._signal_ids_by_dedupe_key[snapshot["dedupe_key"]] = snapshot["id"]
                self._source_signal_ids.setdefault(snapshot["source"], set()).add(snapshot["id"])
                if status in _SIGNAL_ACTIVE_STATUSES:
                    tokens_to_subscribe.update(snapshot.get("required_token_ids") or [])
                if defer_reason is not None:
                    self._set_deferred_state_locked(
                        snapshot["id"],
                        required_token_ids=snapshot["required_token_ids"],
                        reason=defer_reason,
                    )
            # ITER-? (Fix II): after hydrating from trade_signals,
            # ensure ``_next_runtime_sequence`` is also above every
            # trader's consumed cursor.  The trade_signals hydration
            # only sees rows kept after the 24h prune cutoff — older
            # signals' sequences are gone, but the cursors that
            # consumed them are persisted.  Without this max(), new
            # signals get sequences BELOW the cursor and the trader
            # silently treats them as already-consumed, manifesting
            # as "strategies fire but trader sees nothing" after every
            # restart.  Found in ITER-4 production debugging.
            if cursor_max_seq > 0:
                self._next_runtime_sequence = max(
                    self._next_runtime_sequence, cursor_max_seq + 1
                )
                logger.info(
                    "intent_runtime sequence floor raised from cursor max",
                    cursor_max=cursor_max_seq,
                    next_runtime_sequence=self._next_runtime_sequence,
                )
        if bootstrap_snapshots:
            await self._attach_live_market_contexts(bootstrap_snapshots)
            await self._defer_scanner_snapshots_without_strict_live_market(
                bootstrap_snapshots,
                projection_snapshots=bootstrap_snapshots,
                reason="strict_ws_pricing_live_context_unavailable",
            )
            bootstrap_snapshots_by_source: dict[str, dict[str, dict[str, Any]]] = {}
            for signal_id, snapshot in bootstrap_snapshots.items():
                source_key = str(snapshot.get("source") or "").strip()
                if not source_key:
                    continue
                bootstrap_snapshots_by_source.setdefault(source_key, {})[signal_id] = copy.deepcopy(snapshot)
            for source_key, snapshots in bootstrap_snapshots_by_source.items():
                await publish_signal_batch(
                    event_type="upsert_reactivated",
                    source=source_key,
                    signal_ids=sorted(snapshots.keys()),
                    trigger="intent_runtime_hydrate",
                    reason="bootstrap_pending_signals",
                    signal_snapshots=snapshots,
                )
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

        # Plan 0010: ensure every published `(source, dedupe_key)` has a
        # committed `trade_signals` row by the time `publish_opportunities`
        # returns.  This closes both modes of the FK race that surface
        # downstream as `trader_decisions_signal_id_fkey` violations:
        #
        # 1. **Post-restart staleness.**  The DB carried a row for
        #    `(source, dedupe_key)` from a previous worker-trading
        #    process; the in-memory cache is empty after restart.  A
        #    naïve `uuid.uuid4().hex` mints a fresh id; the projection's
        #    `signal_bus.upsert_trade_signal` finds the row by
        #    `(source, dedupe_key)` and UPDATEs it in place — keeping
        #    the OLD id.  The runtime cache holds the NEW id, the
        #    orchestrator's `_ensure_runtime_signal_persisted` ON
        #    CONFLICT silences, and `trader_decisions.signal_id`
        #    references the in-memory id that never lands.
        # 2. **In-process publish→consume gap.**  The dedupe_key is
        #    genuinely new (no DB row, no cache entry).  `publish_*`
        #    mints a uuid, populates the cache, and the orchestrator
        #    (in-process callback for traders source) picks the signal
        #    up microseconds later — long before the asynchronous
        #    projection loop has committed the corresponding row.  The
        #    orchestrator's `_ensure_runtime_signal_persisted` issues
        #    its own INSERT inside an open trader-cycle transaction,
        #    but production has shown that this in-tx INSERT is not
        #    sufficient to satisfy the FK at flush time on the
        #    `worker-trading` plane (the row stays uncommitted across
        #    multiple inner queries; the trader cycle's 10s budget
        #    can lapse mid-tx; concurrent projection-side INSERTs for
        #    the same `(source, dedupe_key)` can interleave under load).
        #
        # The fix:
        #   (a) Outside `self._lock`, prefetch existing
        #       `(source, dedupe_key) → id` rows from `trade_signals`.
        #   (b) For dedupe_keys with no row yet, mint candidate uuids
        #       and synchronously INSERT skeleton rows
        #       (`ON CONFLICT (source, dedupe_key) DO NOTHING`) in a
        #       dedicated, committed session BEFORE the lock is
        #       acquired.  The projection loop's later UPSERT updates
        #       the same row in place (so all the rich fields — payload,
        #       runtime_sequence, expires_at, etc. — still flow through
        #       the projection's batched path).
        #   (c) Re-query the table for the canonical id of any dedupe_key
        #       whose INSERT lost a race to a peer publisher.
        #   (d) Inside the lock, the new-id allocation prefers the
        #       canonical id (`prefetched_ids ∪ committed_ids`) over a
        #       fresh uuid.
        #
        # Cost: one SELECT + at most one batched INSERT per
        # `publish_opportunities` call, scoped to cache-missing
        # dedupe_keys.  Steady-state scanner publishes hit the cache
        # for >99% of dedupe_keys, so both queries are no-ops; traders
        # publishes are bounded to a handful of dedupe_keys per call.
        prefetch_meta_by_dedupe: dict[str, dict[str, str]] = {}
        for opportunity in opportunities:
            try:
                contract_market_id, *_unused = build_signal_contract_from_opportunity(opportunity)
            except Exception:
                continue
            if not contract_market_id:
                continue
            # Plan 0041: see `_opportunity_dedupe_key` for the scope
            # logic — per-trader clones get a distinct row, but
            # unscoped (singleton) emissions keep their pre-Plan-0041
            # hash so existing in-flight ``trade_signals`` survive
            # deploys.
            candidate_dedupe_key = _opportunity_dedupe_key(opportunity, contract_market_id)
            if (
                candidate_dedupe_key
                and candidate_dedupe_key not in self._signal_ids_by_dedupe_key
                and candidate_dedupe_key not in prefetch_meta_by_dedupe
            ):
                prefetch_meta_by_dedupe[candidate_dedupe_key] = {
                    "market_id": str(contract_market_id),
                }

        prefetch_dedupe_keys = list(prefetch_meta_by_dedupe.keys())
        prefetched_ids: dict[str, str] = {}
        if prefetch_dedupe_keys:
            try:
                async with AsyncSessionLocal() as prefetch_session:
                    prefetch_result = await prefetch_session.execute(
                        select(TradeSignal.id, TradeSignal.dedupe_key).where(
                            TradeSignal.source == str(source),
                            TradeSignal.dedupe_key.in_(prefetch_dedupe_keys),
                        )
                    )
                    for row_id, row_dedupe in prefetch_result.all():
                        normalized_dedupe = str(row_dedupe or "").strip()
                        if normalized_dedupe:
                            prefetched_ids[normalized_dedupe] = str(row_id)
            except Exception as prefetch_exc:
                # Prefetch is a hint; on failure, fall back to fresh
                # uuids and rely on the projection loop's existing
                # behaviour.  Logged at debug because the FK race
                # surfaces loudly enough downstream if we ever hit
                # this branch under steady-state load.
                logger.debug(
                    "intent_runtime: trade_signals dedupe_key prefetch failed; falling back to uuid",
                    exc_info=prefetch_exc,
                    source=str(source),
                    prefetch_dedupe_count=len(prefetch_dedupe_keys),
                )

        # Skeleton-INSERT pass for dedupe_keys with no existing DB row.
        # We commit before acquiring `self._lock` so the row is visible
        # to every consumer (orchestrator, fast_trader, projection loop,
        # API surface) the moment publish_opportunities returns.  The
        # projection loop's later UPSERT then fills in the rich fields
        # via UPDATE on the same row.
        committed_ids: dict[str, str] = {}
        skeleton_dedupe_keys = [
            dk for dk in prefetch_dedupe_keys if dk not in prefetched_ids
        ]
        if skeleton_dedupe_keys:
            # Defensive TTL (plan 0011): if the rest of publish_opportunities
            # dies between this commit and the projection loop's UPSERT
            # (process kill, connection drop, unhandled exception), the
            # skeleton row would otherwise live in ``trade_signals``
            # forever — invisible to the terminal-row pruner which keys
            # on ``expires_at < now()``.  The projection loop overwrites
            # ``expires_at`` with the strategy's intended value as soon
            # as it commits, so this TTL only takes effect for orphaned
            # skeletons.  Bounded by the retention sweep on the
            # discovery plane (see ``services.skeleton_signal_retention``).
            skeleton_ttl_seconds = max(
                60,
                int(getattr(settings, "INTENT_RUNTIME_SKELETON_TTL_SECONDS", 300) or 300),
            )
            skeleton_expires_at = now + timedelta(seconds=skeleton_ttl_seconds)
            skeleton_rows = [
                {
                    "id": uuid.uuid4().hex,
                    "source": str(source),
                    "signal_type": signal_type,
                    "market_id": prefetch_meta_by_dedupe[dk]["market_id"],
                    "dedupe_key": dk,
                    "status": "pending",
                    "expires_at": skeleton_expires_at,
                    "created_at": now,
                    "updated_at": now,
                }
                for dk in skeleton_dedupe_keys
            ]
            try:
                async with AsyncSessionLocal() as skeleton_session:
                    await skeleton_session.execute(
                        pg_insert(TradeSignal)
                        .values(skeleton_rows)
                        .on_conflict_do_nothing(
                            index_elements=["source", "dedupe_key"]
                        )
                    )
                    await skeleton_session.commit()
                    # Re-query to get the canonical id for every
                    # skeleton dedupe_key — covers both the rows we
                    # just inserted and the rare conflict-loser rows
                    # where a peer publisher beat us between prefetch
                    # and skeleton-INSERT.
                    after_result = await skeleton_session.execute(
                        select(TradeSignal.id, TradeSignal.dedupe_key).where(
                            TradeSignal.source == str(source),
                            TradeSignal.dedupe_key.in_(skeleton_dedupe_keys),
                        )
                    )
                    for row_id, row_dedupe in after_result.all():
                        normalized_dedupe = str(row_dedupe or "").strip()
                        if normalized_dedupe:
                            committed_ids[normalized_dedupe] = str(row_id)
            except Exception as skeleton_exc:
                # Skeleton-INSERT failure must not break publish.  The
                # downstream FK race re-surfaces loudly enough that we
                # do NOT need to log at WARNING for every transient DB
                # blip; the orchestrator-side log already captures the
                # failure mode if this fallback ever fires.
                logger.debug(
                    "intent_runtime: trade_signals skeleton INSERT failed; falling back to uuid",
                    exc_info=skeleton_exc,
                    source=str(source),
                    skeleton_dedupe_count=len(skeleton_dedupe_keys),
                )

        async with self._lock:
            for opportunity in opportunities:
                market_id, direction, entry_price, market_question, payload_json, strategy_context_json = (
                    build_signal_contract_from_opportunity(opportunity)
                )
                if not market_id:
                    continue
                # Plan 0041: see `_opportunity_dedupe_key` for the
                # scope logic; both call sites use the same helper so
                # the prefetch-vs-emit paths can never disagree on a
                # key.
                dedupe_key = _opportunity_dedupe_key(opportunity, market_id)
                expires_at = _opportunity_signal_expires_at(now, opportunity, default_ttl_minutes)
                payload = copy.deepcopy(payload_json or {})
                strategy_context = copy.deepcopy(strategy_context_json or {})
                payload["ingested_at"] = _to_iso(now)
                payload["signal_emitted_at"] = payload.get("signal_emitted_at") or _to_iso(now)
                payload["bridge_source"] = str(source)
                payload["bridge_run_at"] = _to_iso(now)
                strategy_context["ingested_at"] = _to_iso(now)
                strategy_context["bridge_source"] = str(source)
                strategy_context["bridge_run_at"] = _to_iso(now)
                # Plan 0041: persist the per-trader scope into the JSON blob
                # so it survives worker restarts (the in-memory snapshot is
                # rehydrated from ``trade_signals.payload_json`` on boot).
                opp_intended_trader_id_value = getattr(opportunity, "intended_trader_id", None)
                opp_intended_trader_id = (
                    str(opp_intended_trader_id_value).strip()
                    if opp_intended_trader_id_value is not None
                    else ""
                ) or None
                if opp_intended_trader_id:
                    payload["intended_trader_id"] = opp_intended_trader_id
                else:
                    payload.pop("intended_trader_id", None)
                opp_quality_passed: bool | None = None
                opp_quality_rejection_reasons: list[str] = []
                if quality_filter_pipeline is not None:
                    report = quality_filter_pipeline.evaluate(opportunity)
                    opp_quality_passed = bool(report.passed)
                    opp_quality_rejection_reasons = list(getattr(report, "rejection_reasons", []) or [])
                elif quality_reports is not None:
                    report = quality_reports.get(getattr(opportunity, "stable_id", None) or getattr(opportunity, "id", None))
                    if report is not None:
                        opp_quality_passed = bool(report.passed)
                        opp_quality_rejection_reasons = list(getattr(report, "rejection_reasons", []) or [])
                desired_status = "filtered" if opp_quality_passed is False else "pending"

                # Plan 0041: when the strategy was invoked from a per-trader
                # clone, ``Opportunity.intended_trader_id`` carries the id of
                # the trader the signal is scoped to. Other traders bound to
                # the same source filter the row out in
                # ``list_unconsumed_signals`` so they never act on signals
                # generated from a peer's per-trader gates.
                intended_trader_id_value = getattr(opportunity, "intended_trader_id", None)
                normalized_intended_trader_id = (
                    str(intended_trader_id_value).strip()
                    if intended_trader_id_value is not None
                    else ""
                ) or None
                incoming_snapshot = {
                    "id": "",
                    "source": str(source),
                    "source_item_id": str(getattr(opportunity, "stable_id", None) or "").strip(),
                    "signal_type": signal_type,
                    "strategy_type": str(getattr(opportunity, "strategy", None) or "").strip(),
                    "intended_trader_id": normalized_intended_trader_id,
                    "market_id": str(market_id),
                    "market_question": str(market_question or "").strip(),
                    "direction": str(direction or "").strip(),
                    "entry_price": float(entry_price) if entry_price is not None else None,
                    "effective_price": None,
                    # Prefer the strategy-emitted edge_percent (probabilistic
                    # conviction in price-cents x 100). Source it from either
                    # the Opportunity field (post-migration) or from
                    # strategy_context["edge_percent"] (transitional path used
                    # by directional strategies that need to decouple
                    # conviction from capital-efficiency ROI). Fall back to
                    # roi_percent only when neither is present — the
                    # back-compat path for arb strategies where edge == ROI.
                    "edge_percent": float(
                        (getattr(opportunity, "edge_percent", None)
                         if getattr(opportunity, "edge_percent", None) is not None
                         else (strategy_context.get("edge_percent")
                               if isinstance(strategy_context, dict)
                                  and strategy_context.get("edge_percent") is not None
                               else getattr(opportunity, "roi_percent", 0.0) or 0.0))
                    ),
                    "confidence": float(getattr(opportunity, "confidence", 0.0) or 0.0),
                    "liquidity": float(getattr(opportunity, "min_liquidity", 0.0) or 0.0),
                    "expires_at": _to_iso(_normalize_datetime(expires_at)),
                    "status": desired_status,
                    "payload_json": payload,
                    "strategy_context_json": strategy_context,
                    "quality_passed": opp_quality_passed,
                    "quality_rejection_reasons": opp_quality_rejection_reasons,
                    "dedupe_key": dedupe_key,
                    "runtime_sequence": None,
                    "required_token_ids": _extract_required_token_ids(
                        payload,
                        direction=str(direction or "").strip().lower(),
                    ),
                    "runtime_lane": _runtime_lane_for_source(source),
                    "deferred_until_ws": False,
                    "deferred_reason": None,
                    "deferred_started_at": None,
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
                    reactivated_unchanged_terminal = False
                    if not material_change:
                        reactivated_unchanged_terminal = _should_reactivate_unchanged_terminal_signal(
                            existing,
                            incoming_snapshot,
                            now=now,
                        )
                    if not material_change and not reactivated_unchanged_terminal:
                        active_dedupe_keys.add(dedupe_key)
                        continue
                    incoming_snapshot["effective_price"] = None
                    existing_deferred_started_at = existing.get("deferred_started_at")
                    self._clear_deferred_state_locked(existing["id"], clear_snapshot_state=False)
                    _ea = str(
                        (incoming_snapshot.get("payload_json") or {})
                        .get("strategy_runtime", {})
                        .get("execution_activation", "")
                        or ""
                    )
                    if normalized_source == "scanner":
                        _ea = "ws_current"
                    if incoming_snapshot["status"] == "filtered":
                        incoming_snapshot["runtime_sequence"] = None
                        incoming_snapshot["deferred_until_ws"] = False
                        incoming_snapshot["deferred_reason"] = None
                        incoming_snapshot["deferred_started_at"] = None
                    elif _ea == "ws_post_arm_tick" and incoming_snapshot["required_token_ids"]:
                        incoming_snapshot["payload_json"]["execution_armed_at"] = _to_iso(now)
                        self._set_deferred_state_locked(
                            existing["id"],
                            required_token_ids=incoming_snapshot["required_token_ids"],
                            reason="awaiting_post_arm_ws_tick",
                        )
                        incoming_snapshot["deferred_until_ws"] = True
                        incoming_snapshot["deferred_reason"] = "awaiting_post_arm_ws_tick"
                        incoming_snapshot["deferred_started_at"] = (
                            existing.get("deferred_started_at") or existing_deferred_started_at or _to_iso(now)
                        )
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
                        incoming_snapshot["deferred_started_at"] = (
                            existing.get("deferred_started_at") or existing_deferred_started_at or _to_iso(now)
                        )
                        incoming_snapshot["runtime_sequence"] = None
                    else:
                        incoming_snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                        _ensure_execution_armed_at(incoming_snapshot, _to_iso(now))
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
                    # Plan 0010: prefer the canonical `trade_signals.id`
                    # for this dedupe_key (prefetched OR skeleton-inserted
                    # outside the lock) over a fresh uuid, so the
                    # in-memory cache never diverges from the row the
                    # projection loop will eventually update AND every
                    # downstream consumer can FK against a row that is
                    # already committed in `trade_signals`.
                    signal_id = (
                        prefetched_ids.get(dedupe_key)
                        or committed_ids.get(dedupe_key)
                        or uuid.uuid4().hex
                    )
                    incoming_snapshot["id"] = signal_id
                    _ea = str(
                        (incoming_snapshot.get("payload_json") or {})
                        .get("strategy_runtime", {})
                        .get("execution_activation", "")
                        or ""
                    )
                    if normalized_source == "scanner":
                        _ea = "ws_current"
                    if incoming_snapshot["status"] == "filtered":
                        incoming_snapshot["runtime_sequence"] = None
                    elif _ea == "ws_post_arm_tick" and incoming_snapshot["required_token_ids"]:
                        incoming_snapshot["payload_json"]["execution_armed_at"] = _to_iso(now)
                        self._set_deferred_state_locked(
                            signal_id,
                            required_token_ids=incoming_snapshot["required_token_ids"],
                            reason="awaiting_post_arm_ws_tick",
                        )
                        incoming_snapshot["deferred_until_ws"] = True
                        incoming_snapshot["deferred_reason"] = "awaiting_post_arm_ws_tick"
                        incoming_snapshot["deferred_started_at"] = _to_iso(now)
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
                        incoming_snapshot["deferred_started_at"] = _to_iso(now)
                    else:
                        incoming_snapshot["runtime_sequence"] = self._allocate_runtime_sequence_locked()
                        _ensure_execution_armed_at(incoming_snapshot, _to_iso(now))
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
            actionable_items = list(actionable_snapshots.items())
            for chunk_start in range(0, len(actionable_items), _SIGNAL_PUBLICATION_BATCH_SIZE):
                actionable_chunk = dict(
                    actionable_items[chunk_start : chunk_start + _SIGNAL_PUBLICATION_BATCH_SIZE]
                )
                projection_chunk = {
                    signal_id: projection_snapshots[signal_id]
                    for signal_id in actionable_chunk.keys()
                    if signal_id in projection_snapshots
                }
                event_type_chunk = {
                    signal_id: str(actionable_event_types.get(signal_id) or "upsert_update")
                    for signal_id in actionable_chunk.keys()
                }
                await self._attach_live_market_contexts(actionable_chunk)
                await self._defer_scanner_snapshots_without_strict_live_market(
                    actionable_chunk,
                    projection_snapshots=projection_chunk,
                    event_types=event_type_chunk,
                    reason="strict_ws_pricing_live_context_unavailable",
                )
                for signal_id, snapshot in actionable_chunk.items():
                    projection_snapshot = projection_chunk.get(signal_id)
                    live_market = (snapshot.get("payload_json") or {}).get(_LIVE_MARKET_PAYLOAD_KEY)
                    if projection_snapshot is None or not isinstance(live_market, dict):
                        continue
                    payload = projection_snapshot.get("payload_json")
                    if not isinstance(payload, dict):
                        payload = {}
                    else:
                        payload = dict(payload)
                    payload[_LIVE_MARKET_PAYLOAD_KEY] = copy.deepcopy(live_market)
                    projection_snapshot["payload_json"] = payload
                snapshots_by_event_type: dict[str, dict[str, dict[str, Any]]] = {}
                for signal_id, snapshot in actionable_chunk.items():
                    event_type = str(event_type_chunk.get(signal_id) or "upsert_update")
                    snapshots_by_event_type.setdefault(event_type, {})[signal_id] = snapshot
                for event_type, snapshots in snapshots_by_event_type.items():
                    emitted_at_iso = _to_iso(utcnow())
                    for signal_id, snapshot in snapshots.items():
                        _restamp_signal_emitted_at(snapshot, emitted_at_iso)
                        projection_snapshot = projection_chunk.get(signal_id)
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
        if projection_snapshots:
            projection_items = list(projection_snapshots.items())
            for chunk_start in range(0, len(projection_items), _SIGNAL_PUBLICATION_BATCH_SIZE):
                projection_chunk = dict(
                    projection_items[chunk_start : chunk_start + _SIGNAL_PUBLICATION_BATCH_SIZE]
                )
                await self._enqueue_projection(
                    {
                        "kind": "upsert",
                        "source": str(source),
                        "signal_type": signal_type,
                        "snapshots": projection_chunk,
                        "sweep_missing": False,
                        "keep_dedupe_keys": [],
                    }
                )
        if sweep_missing:
            await self._enqueue_projection(
                {
                    "kind": "upsert",
                    "source": str(source),
                    "signal_type": signal_type,
                    "snapshots": {},
                    "sweep_missing": True,
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
                _ensure_execution_armed_at(snapshot, _to_iso(utcnow()))
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
        del cursor_created_at
        del cursor_signal_id
        # Plan 0041: ``intended_trader_id`` on the runtime snapshot scopes
        # the signal to a single trader. ``None`` keeps the legacy
        # multi-trader-visible routing for un-tagged emissions (sources
        # without per-trader strategy_params overrides). When set, only
        # the consuming trader whose id matches sees the row.
        normalized_consumer_trader_id = str(trader_id or "").strip() or None
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
                # Per-trader signal scope (plan 0041).
                snapshot_intended_trader_id = snapshot.get("intended_trader_id")
                if snapshot_intended_trader_id:
                    if (
                        normalized_consumer_trader_id is None
                        or str(snapshot_intended_trader_id).strip()
                        != normalized_consumer_trader_id
                    ):
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

    def _merge_upsert_payload_inplace(
        self, target: dict[str, Any], incoming: dict[str, Any]
    ) -> None:
        """Merge ``incoming`` upsert payload into ``target`` in place.

        Mirrors ``_coalesce_upsert_payloads`` but mutates the target so
        the queued reference stays live for the drainer. Later snapshots
        supersede earlier ones per dedupe key (dict ``update``);
        ``keep_dedupe_keys`` are unioned; ``sweep_missing`` is ORed.
        """
        tgt_snapshots = target.get("snapshots")
        if not isinstance(tgt_snapshots, dict):
            tgt_snapshots = {}
            target["snapshots"] = tgt_snapshots
        inc_snapshots = incoming.get("snapshots")
        if isinstance(inc_snapshots, dict):
            tgt_snapshots.update(inc_snapshots)

        merged_keep: set[str] = {
            str(key)
            for key in (target.get("keep_dedupe_keys") or [])
            if str(key).strip()
        }
        merged_keep.update(
            str(key)
            for key in (incoming.get("keep_dedupe_keys") or [])
            if str(key).strip()
        )
        target["keep_dedupe_keys"] = sorted(merged_keep)

        target["sweep_missing"] = bool(target.get("sweep_missing")) or bool(
            incoming.get("sweep_missing")
        )

        tgt_retry = int(target.get("_projection_retry_count") or 0)
        inc_retry = int(incoming.get("_projection_retry_count") or 0)
        merged_retry = max(tgt_retry, inc_retry)
        if merged_retry > 0:
            target["_projection_retry_count"] = merged_retry
        else:
            target.pop("_projection_retry_count", None)

    @staticmethod
    def _upsert_coalesce_key(payload: dict[str, Any]) -> tuple[str, str]:
        source = str(payload.get("source") or "").strip().lower()
        signal_type = str(payload.get("signal_type") or "").strip().lower()
        return (source, signal_type)

    async def _enqueue_projection(self, payload: dict[str, Any]) -> None:
        if not self._started:
            return
        # Producer-side coalescing for upsert payloads. If a payload
        # for the same (source, signal_type) is already in the queue,
        # merge the new snapshots into the queued buffer in place
        # rather than enqueueing a second entry. This bounds queue
        # growth to O(distinct-upsert-keys) instead of O(producer-rate),
        # which previously saturated the 5000 cap during bursts.
        coalesce_key: tuple[str, str] | None = None
        kind = str(payload.get("kind") or "").strip().lower()
        if kind == "upsert":
            coalesce_key = self._upsert_coalesce_key(payload)
            existing = self._pending_upsert_buffers.get(coalesce_key)
            if existing is not None:
                self._merge_upsert_payload_inplace(existing, payload)
                self._projection_coalesced_on_enqueue += 1
                return
            # First upsert for this key in-flight: register the buffer
            # so subsequent enqueues can coalesce into it.
            self._pending_upsert_buffers[coalesce_key] = payload

        # Try non-blocking first to avoid stalling producers on the
        # trading loop. If the queue is full, the backpressure signal
        # has already been published (at 50% via _run_projection_loop)
        # and producers should have voluntarily slowed. If they didn't,
        # drop with a counted warning rather than block — a stalled
        # producer is worse than a dropped projection (the next state
        # snapshot from the same source supersedes anyway via the
        # upsert coalesce path).
        try:
            self._projection_queue.put_nowait(payload)
            return
        except asyncio.QueueFull:
            # Roll back the pending registration so a future enqueue
            # is not permanently suppressed by a reference that never
            # made it to the queue.
            if coalesce_key is not None:
                self._pending_upsert_buffers.pop(coalesce_key, None)
        self._projection_dropped_at_cap += 1
        now = time.monotonic()
        if (now - self._projection_drop_log_at) >= 5.0:
            self._projection_drop_log_at = now
            logger.warning(
                "Intent runtime projection dropped at queue cap",
                projection_kind=str(payload.get("kind") or "").strip().lower() or None,
                source=str(payload.get("source") or "").strip().lower() or None,
                queue_size=self._projection_queue.qsize(),
                queue_max=self._projection_queue.maxsize,
                cumulative_dropped=self._projection_dropped_at_cap,
                coalesced_on_enqueue=self._projection_coalesced_on_enqueue,
            )

    async def _retry_projection_payload(self, payload: dict[str, Any]) -> None:
        retry_count = int(payload.get("_projection_retry_count") or 0)
        delay_seconds = _PROJECTION_RETRY_BASE_DELAY_SECONDS * (2 ** max(0, retry_count - 1))
        if is_db_pressure_active():
            delay_seconds = max(delay_seconds, min(db_pressure_remaining_seconds() + 0.25, 30.0))
        await asyncio.sleep(delay_seconds)
        # Re-queue through _enqueue_projection so the coalesce buffer
        # is refreshed — merging into any in-flight payload for the
        # same key rather than adding a second queue slot.
        kind = str(payload.get("kind") or "").strip().lower()
        if kind == "upsert":
            await self._enqueue_projection(payload)
            return
        await self._projection_queue.put(payload)

    async def _run_projection_loop(self) -> None:
        while True:
            payload = await self._projection_queue.get()
            # Publish queue saturation as backpressure so producers can
            # voluntarily slow down before the queue fills (5000 cap).
            # We compute the level here rather than on every put because
            # the projection loop is the single drainer — saturation is
            # bounded by what we can pull through, so this is the right
            # spot to observe it. Curve: 0% at <50% full, ramps to 100%
            # at full so producers feel pressure as the queue grows.
            qsize = self._projection_queue.qsize()
            qmax = max(1, self._projection_queue.maxsize)
            qpct = qsize / qmax
            if qpct < 0.5:
                publish_backpressure("intent_runtime_queue", level=0.0)
            else:
                bp_level = (qpct - 0.5) / 0.5  # 0.0 at 50%, 1.0 at 100%
                publish_backpressure(
                    "intent_runtime_queue",
                    level=bp_level,
                    reason=f"queue@{qpct:.0%}({qsize}/{qmax})",
                )
            try:
                kind = str(payload.get("kind") or "").strip().lower()
                # Remove from the enqueue-side coalescing buffer so the
                # NEXT upsert for the same (source, signal_type) starts
                # a fresh queued payload instead of mutating this one
                # after we've started processing it.
                if kind == "upsert":
                    self._pending_upsert_buffers.pop(
                        self._upsert_coalesce_key(payload), None
                    )
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
                        if queued_kind == "upsert":
                            self._pending_upsert_buffers.pop(
                                self._upsert_coalesce_key(queued), None
                            )
                        carry_payload = queued
                        break
                    await self._project_status_batch(status_payloads)
                    if carry_payload is not None:
                        await self._dispatch_projection_payload(carry_payload)
                elif kind == "upsert":
                    upsert_payloads = [payload]
                    carry_payload: dict[str, Any] | None = None
                    source = str(payload.get("source") or "").strip().lower()
                    while len(upsert_payloads) < _UPSERT_PROJECTION_BATCH_MAX:
                        try:
                            queued = self._projection_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        queued_kind = str(queued.get("kind") or "").strip().lower()
                        queued_source = str(queued.get("source") or "").strip().lower()
                        if queued_kind == "upsert":
                            # Always retire the coalesce slot when we
                            # pop an upsert, regardless of whether it
                            # matches the current batch source.
                            self._pending_upsert_buffers.pop(
                                self._upsert_coalesce_key(queued), None
                            )
                        if queued_kind == "upsert" and queued_source == source:
                            upsert_payloads.append(queued)
                            continue
                        carry_payload = queued
                        break
                    await self._project_upsert_batch(self._coalesce_upsert_payloads(upsert_payloads))
                    if carry_payload is not None:
                        await self._dispatch_projection_payload(carry_payload)
                else:
                    await self._dispatch_projection_payload(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retry_count = int(payload.get("_projection_retry_count") or 0)
                pressure_ttl_seconds = _projection_pressure_ttl_seconds(retry_count)
                db_pressure_marked = False
                if pressure_ttl_seconds is not None:
                    db_pressure_marked = maybe_mark_db_pressure(
                        exc,
                        component="intent_projection",
                        ttl_seconds=pressure_ttl_seconds,
                    )
                if retry_count < _PROJECTION_RETRY_MAX_ATTEMPTS:
                    payload["_projection_retry_count"] = retry_count + 1
                    self._start_task(
                        self._retry_projection_payload(payload),
                        name=f"intent-runtime-projection-retry-{retry_count + 1}",
                    )
                logger.warning(
                    "Intent runtime DB projection failed",
                    projection_kind=str(payload.get("kind") or "").strip().lower() or None,
                    retry_count=retry_count,
                    error_type=type(exc).__name__,
                    db_pressure_marked=db_pressure_marked,
                    projection_queue_size=self._projection_queue.qsize(),
                    exc_info=exc,
                )

    def _coalesce_upsert_payloads(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        if not payloads:
            return {"kind": "upsert", "source": "", "snapshots": {}, "sweep_missing": False, "keep_dedupe_keys": []}
        merged = dict(payloads[-1])
        snapshots: dict[str, Any] = {}
        keep_dedupe_keys: set[str] = set()
        sweep_missing = False
        retry_count = 0
        for payload in payloads:
            raw_snapshots = payload.get("snapshots")
            if isinstance(raw_snapshots, dict):
                snapshots.update(raw_snapshots)
            keep_dedupe_keys.update(
                str(key)
                for key in (payload.get("keep_dedupe_keys") or [])
                if str(key).strip()
            )
            sweep_missing = sweep_missing or bool(payload.get("sweep_missing"))
            retry_count = max(retry_count, int(payload.get("_projection_retry_count") or 0))
        merged["snapshots"] = snapshots
        merged["keep_dedupe_keys"] = sorted(keep_dedupe_keys)
        merged["sweep_missing"] = sweep_missing
        if retry_count > 0:
            merged["_projection_retry_count"] = retry_count
        else:
            merged.pop("_projection_retry_count", None)
        return merged

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

        _UPSERT_CHUNK_SIZE = (
            _UPSERT_PROJECTION_PRESSURE_CHUNK_SIZE
            if is_db_pressure_active()
            else _UPSERT_PROJECTION_CHUNK_SIZE
        )
        snapshot_items = list(snapshots.values())
        signal_types_in_batch: set[str] = set()
        strategy_types_in_batch: set[str] = set()

        # Process in chunks to avoid holding a connection for minutes
        for chunk_start in range(0, max(len(snapshot_items), 1), _UPSERT_CHUNK_SIZE):
            chunk = snapshot_items[chunk_start:chunk_start + _UPSERT_CHUNK_SIZE]
            if not chunk:
                break
            condition_ids = sorted(
                {
                    str(snapshot.get("market_id") or "").strip()
                    for snapshot in chunk
                    if str(snapshot.get("market_id") or "").strip().startswith("0x")
                }
            )
            if condition_ids:
                try:
                    from services.ws_feeds import get_feed_manager

                    await get_feed_manager().ensure_user_subscribed(condition_ids)
                except Exception as exc:
                    logger.debug(
                        "intent_runtime user-channel subscribe before projection failed",
                        exc_info=exc,
                    )
            async with AsyncSessionLocal() as session:
                # Mirrors Fix J's unmute in shared_state: if SET LOCAL
                # silently fails (e.g. session not yet in transaction or
                # connection in error state) the server-side timeouts
                # never apply and slow queries can hang for minutes
                # before the asyncio outer wait_for / pool watchdog
                # recovers.  Log the failure so we can diagnose if this
                # ever happens in production.
                # ``set_config(name, value, is_local)`` is the function-form of
                # ``SET LOCAL`` (third argument ``true`` makes it transaction-
                # scoped).  Bundling both timeouts into a single ``SELECT`` halves
                # the wire round-trips per chunk.  At ``chunk_size=1`` (the
                # current pressure-aware default) the projection loop runs
                # 30+ chunks/sec; saving one round-trip per chunk is ~30 fewer
                # wire ops/sec on the DB pool that's already showing up in
                # ``Long transaction held`` warnings.  Failures are still
                # logged but cover both timeouts at once.
                try:
                    await session.execute(
                        text(
                            "SELECT "
                            "set_config('statement_timeout', :stmt_ms, true), "
                            "set_config('lock_timeout', :lock_ms, true)"
                        ),
                        {
                            "stmt_ms": str(_PROJECTION_STATEMENT_TIMEOUT_MS),
                            "lock_ms": str(_PROJECTION_LOCK_TIMEOUT_MS),
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "intent_runtime: SET LOCAL statement_timeout=%dms / lock_timeout=%dms failed; "
                        "backend will rely on outer timeouts",
                        _PROJECTION_STATEMENT_TIMEOUT_MS,
                        _PROJECTION_LOCK_TIMEOUT_MS,
                        exc_info=exc,
                    )
                chunk_dedupe_keys = [
                    str(snapshot.get("dedupe_key") or "").strip()
                    for snapshot in chunk
                    if str(snapshot.get("dedupe_key") or "").strip()
                ]
                nonreactivable_dedupe_keys: set[str] = set()
                if chunk_dedupe_keys:
                    existing_rows = (
                        await session.execute(
                            select(
                                TradeSignal.dedupe_key,
                                TradeSignal.status,
                                TradeSignal.runtime_sequence,
                                TradeSignal.effective_price,
                            ).where(
                                TradeSignal.source == source,
                                TradeSignal.dedupe_key.in_(chunk_dedupe_keys),
                            )
                        )
                    ).all()
                    updatable_statuses = BUS_SIGNAL_ACTIVE_STATUSES | BUS_SIGNAL_REACTIVATABLE_STATUSES
                    existing_by_dedupe_key = {
                        str(existing_dedupe_key or "").strip(): {
                            "status": str(existing_status or "").strip().lower(),
                            "runtime_sequence": _normalize_runtime_sequence(existing_runtime_sequence),
                            "effective_price": _safe_float(existing_effective_price),
                        }
                        for existing_dedupe_key, existing_status, existing_runtime_sequence, existing_effective_price in existing_rows
                        if str(existing_dedupe_key or "").strip()
                    }
                    nonreactivable_dedupe_keys = {
                        str(existing_dedupe_key or "").strip()
                        for existing_dedupe_key, existing_status, _existing_runtime_sequence, _existing_effective_price in existing_rows
                        if str(existing_dedupe_key or "").strip()
                        and str(existing_status or "").strip().lower() not in updatable_statuses
                    }
                else:
                    existing_by_dedupe_key = {}
                for snapshot in chunk:
                    signal_type = str(snapshot.get("signal_type") or "").strip().lower()
                    if signal_type:
                        signal_types_in_batch.add(signal_type)
                    strategy_type = str(snapshot.get("strategy_type") or "").strip().lower()
                    if strategy_type:
                        strategy_types_in_batch.add(strategy_type)
                    dedupe_key = str(snapshot.get("dedupe_key") or "").strip()
                    if dedupe_key and dedupe_key in nonreactivable_dedupe_keys:
                        continue
                    existing = existing_by_dedupe_key.get(dedupe_key) if dedupe_key else None
                    incoming_runtime_sequence = _normalize_runtime_sequence(snapshot.get("runtime_sequence"))
                    incoming_effective_price = _safe_float(snapshot.get("effective_price"))
                    desired_status = str(snapshot.get("status") or "").strip().lower()
                    if (
                        existing
                        and incoming_runtime_sequence is not None
                        and existing["runtime_sequence"] is not None
                        and incoming_runtime_sequence <= existing["runtime_sequence"]
                        and (not desired_status or desired_status == existing["status"])
                        and (
                            incoming_effective_price is None
                            or incoming_effective_price == existing["effective_price"]
                        )
                    ):
                        continue
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
                        quality_rejection_reasons=list(snapshot.get("quality_rejection_reasons") or []),
                        dedupe_key=dedupe_key,
                        signal_id=str(snapshot.get("id") or "") or None,
                        runtime_sequence=snapshot.get("runtime_sequence"),
                        commit=False,
                        ensure_subscription=False,
                    )
                    if desired_status and desired_status != str(getattr(row, "status", "") or "").strip().lower():
                        row.status = desired_status
                        row.updated_at = _normalize_datetime(snapshot.get("updated_at")) or utcnow()
                    # Lock-contention fix (mirrors Fix O for opportunity_state):
                    # only assign these attributes when the new value differs
                    # from the existing one.  SQLAlchemy marks the row dirty
                    # on EVERY assignment even when the value is equal, which
                    # was emitting redundant UPDATE statements per snapshot
                    # and stretching the per-batch transaction's lock-hold
                    # time.  Observed in the overnight log as recurring
                    # ``LOCK CONTENTION ... UPDATE trade_signals SET
                    # edge_percent=..., expires_at=..., payload_json=...,
                    # strategy_context_json=..., runtime_sequence=...``
                    # cascading into projection_queue_size=5000.
                    new_runtime_sequence = _normalize_runtime_sequence(snapshot.get("runtime_sequence"))
                    if new_runtime_sequence != getattr(row, "runtime_sequence", None):
                        row.runtime_sequence = new_runtime_sequence
                    effective_price = snapshot.get("effective_price")
                    if effective_price is not None and effective_price != getattr(row, "effective_price", None):
                        row.effective_price = effective_price
                await session.commit()

        if sweep_missing:
            async with AsyncSessionLocal() as session:
                # Mirrors Fix J's unmute in shared_state: if SET LOCAL
                # silently fails (e.g. session not yet in transaction or
                # connection in error state) the server-side timeouts
                # never apply and slow queries can hang for minutes
                # before the asyncio outer wait_for / pool watchdog
                # recovers.  Log the failure so we can diagnose if this
                # ever happens in production.
                # ``set_config(name, value, is_local)`` is the function-form of
                # ``SET LOCAL`` (third argument ``true`` makes it transaction-
                # scoped).  Bundling both timeouts into a single ``SELECT`` halves
                # the wire round-trips per chunk.  At ``chunk_size=1`` (the
                # current pressure-aware default) the projection loop runs
                # 30+ chunks/sec; saving one round-trip per chunk is ~30 fewer
                # wire ops/sec on the DB pool that's already showing up in
                # ``Long transaction held`` warnings.  Failures are still
                # logged but cover both timeouts at once.
                try:
                    await session.execute(
                        text(
                            "SELECT "
                            "set_config('statement_timeout', :stmt_ms, true), "
                            "set_config('lock_timeout', :lock_ms, true)"
                        ),
                        {
                            "stmt_ms": str(_PROJECTION_STATEMENT_TIMEOUT_MS),
                            "lock_ms": str(_PROJECTION_LOCK_TIMEOUT_MS),
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "intent_runtime: SET LOCAL statement_timeout=%dms / lock_timeout=%dms failed; "
                        "backend will rely on outer timeouts",
                        _PROJECTION_STATEMENT_TIMEOUT_MS,
                        _PROJECTION_LOCK_TIMEOUT_MS,
                        exc_info=exc,
                    )
                await expire_source_signals_except(
                    session,
                    source=source,
                    keep_dedupe_keys=keep_dedupe_keys,
                    signal_types=sorted(signal_types_in_batch),
                    strategy_types=sorted(strategy_types_in_batch),
                    commit=False,
                )
                await session.commit()

    async def _project_status(self, payload: dict[str, Any]) -> None:
        await self._project_status_batch([payload])

    def _buffer_emissions(self, emissions: list[dict[str, Any]]) -> None:
        # Loss-tolerant drop guard so the buffer can't grow unbounded if the
        # flusher ever falls behind (it shouldn't — one bulk insert per
        # _EMISSION_FLUSH_INTERVAL_SECONDS). Runs only on the event loop, so the
        # list ops are atomic with the flusher's swap below (no lock needed).
        if len(self._emission_buffer) + len(emissions) > _EMISSION_BUFFER_MAX:
            self._emission_buffer_dropped += len(emissions)
            now = time.monotonic()
            if (now - self._emission_drop_log_at) >= 5.0:
                self._emission_drop_log_at = now
                logger.warning(
                    "Intent runtime emission buffer dropped at cap "
                    "(loss-tolerant history; flusher fell behind)",
                    buffer_size=len(self._emission_buffer),
                    buffer_max=_EMISSION_BUFFER_MAX,
                    dropped_now=len(emissions),
                    cumulative_dropped=self._emission_buffer_dropped,
                )
            return
        self._emission_buffer.extend(emissions)

    async def _flush_emissions_once(self) -> None:
        """Drain the emission buffer in a single bulk insert via the audit pool.
        No-op when the buffer is empty. Loss-tolerant: a failed flush is logged
        and dropped, never retried (retrying would re-introduce hot-path
        pressure). Split out from the loop below so it is directly drivable in
        tests without spinning the timer."""
        if not self._emission_buffer:
            return
        # Atomic swap on the event loop: no await between read and reset.
        batch = self._emission_buffer
        self._emission_buffer = []
        try:
            # Audit pool: emissions are loss-tolerant + recoverable, so they
            # ride the dedicated audit tier and never contend with the main
            # (trading hot-path) pool — same isolation trader_hot_state uses.
            async with AuditAsyncSessionLocal() as session:
                await session.execute(TradeSignalEmission.__table__.insert(), batch)
                await session.commit()
        except Exception as exc:
            logger.warning(
                "intent_runtime: buffered emission flush failed (loss-tolerant)",
                emission_count=len(batch),
                error_type=type(exc).__name__,
                exc_info=exc,
            )

    async def _run_emission_flush_loop(self) -> None:
        """Batch-flush buffered trade_signal_emissions off the projection hot
        path. Inline insert+commit per status chunk held a main-pool connection
        2-3.5s under contention; coalescing into one periodic bulk insert frees
        that connection right after the trade_signals UPDATE commits."""
        while True:
            try:
                await asyncio.sleep(_EMISSION_FLUSH_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                raise
            await self._flush_emissions_once()

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

        _STATUS_CHUNK_SIZE = (
            _STATUS_PROJECTION_PRESSURE_CHUNK_SIZE
            if is_db_pressure_active()
            else _STATUS_PROJECTION_CHUNK_SIZE
        )
        items = list(latest_by_signal_id.values())
        for chunk_start in range(0, len(items), _STATUS_CHUNK_SIZE):
            chunk = items[chunk_start:chunk_start + _STATUS_CHUNK_SIZE]
            signal_ids = [
                str(item.get("signal_id") or "").strip()
                for item in chunk
                if str(item.get("signal_id") or "").strip()
            ]
            if not signal_ids:
                continue
            async with AsyncSessionLocal() as session:
                # Mirrors Fix J's unmute in shared_state: if SET LOCAL
                # silently fails (e.g. session not yet in transaction or
                # connection in error state) the server-side timeouts
                # never apply and slow queries can hang for minutes
                # before the asyncio outer wait_for / pool watchdog
                # recovers.  Log the failure so we can diagnose if this
                # ever happens in production.
                # ``set_config(name, value, is_local)`` is the function-form of
                # ``SET LOCAL`` (third argument ``true`` makes it transaction-
                # scoped).  Bundling both timeouts into a single ``SELECT`` halves
                # the wire round-trips per chunk.  At ``chunk_size=1`` (the
                # current pressure-aware default) the projection loop runs
                # 30+ chunks/sec; saving one round-trip per chunk is ~30 fewer
                # wire ops/sec on the DB pool that's already showing up in
                # ``Long transaction held`` warnings.  Failures are still
                # logged but cover both timeouts at once.
                try:
                    await session.execute(
                        text(
                            "SELECT "
                            "set_config('statement_timeout', :stmt_ms, true), "
                            "set_config('lock_timeout', :lock_ms, true)"
                        ),
                        {
                            "stmt_ms": str(_PROJECTION_STATEMENT_TIMEOUT_MS),
                            "lock_ms": str(_PROJECTION_LOCK_TIMEOUT_MS),
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "intent_runtime: SET LOCAL statement_timeout=%dms / lock_timeout=%dms failed; "
                        "backend will rely on outer timeouts",
                        _PROJECTION_STATEMENT_TIMEOUT_MS,
                        _PROJECTION_LOCK_TIMEOUT_MS,
                        exc_info=exc,
                    )
                changed_at = _to_utc(utcnow()).replace(tzinfo=None)
                values_sql: list[str] = []
                params: dict[str, Any] = {"changed_at": changed_at}
                for idx, item in enumerate(chunk):
                    signal_id = str(item.get("signal_id") or "").strip()
                    if not signal_id:
                        continue
                    normalized_status = str(item.get("status") or "").strip().lower()
                    effective_price = item.get("effective_price")
                    params[f"signal_id_{idx}"] = signal_id
                    params[f"status_{idx}"] = normalized_status
                    params[f"effective_price_{idx}"] = effective_price
                    params[f"has_effective_price_{idx}"] = effective_price is not None
                    values_sql.append(
                        f"("
                        f"CAST(:signal_id_{idx} AS VARCHAR), "
                        f"CAST(:status_{idx} AS VARCHAR), "
                        f"CAST(:effective_price_{idx} AS NUMERIC(24, 12)), "
                        f"CAST(:has_effective_price_{idx} AS BOOLEAN)"
                        f")"
                    )
                if not values_sql:
                    await session.rollback()
                    continue
                update_sql = text(
                    f"""
                    WITH incoming(signal_id, status, effective_price, has_effective_price) AS (
                        VALUES {", ".join(values_sql)}
                    ),
                    updated AS (
                        UPDATE trade_signals AS ts
                        SET
                            status = incoming.status,
                            updated_at = :changed_at,
                            effective_price = CASE
                                WHEN incoming.has_effective_price THEN incoming.effective_price
                                ELSE ts.effective_price
                            END
                        FROM incoming
                        WHERE ts.id = incoming.signal_id
                          AND (
                            lower(coalesce(ts.status, '')) IS DISTINCT FROM incoming.status
                            OR (
                                incoming.has_effective_price
                                AND ts.effective_price IS DISTINCT FROM incoming.effective_price
                            )
                          )
                        RETURNING
                            ts.id,
                            ts.source,
                            ts.source_item_id,
                            ts.signal_type,
                            ts.strategy_type,
                            ts.market_id,
                            ts.direction,
                            ts.entry_price,
                            ts.effective_price,
                            ts.edge_percent,
                            ts.confidence,
                            ts.liquidity,
                            ts.status,
                            ts.dedupe_key
                    )
                    SELECT * FROM updated
                    """
                )
                updated_rows = list((await session.execute(update_sql, params)).mappings().all())
                if not updated_rows:
                    await session.rollback()
                    continue
                # Commit the trade_signals UPDATE FIRST so its row locks
                # release immediately.  The emission history INSERT used to
                # ride this same transaction, holding the trade_signals
                # locks across the history write — the 2026-05-22 soak
                # showed that as the #1 ``Long transaction held
                # origin=intent-runtime-projection`` source (156 holds,
                # up to 2s each) and ``LOCK CONTENTION ... UPDATE
                # trade_signals``.  Emissions are immutable append-only
                # history with no FK to trade_signals and no live-trading
                # reader (only the offline backtester / simulator), so
                # writing them out-of-band (buffered here, batch-flushed by
                # _run_emission_flush_loop on the dedicated audit pool) is
                # correctness-neutral and keeps the emission write off both the
                # trade_signals lock window AND the main trading pool.
                # ``updated_rows`` is already materialized, so it survives
                # the commit.
                await session.commit()
                emissions = [
                    {
                        "id": uuid.uuid4().hex,
                        "signal_id": row["id"],
                        "source": str(row["source"] or ""),
                        "source_item_id": row["source_item_id"],
                        "signal_type": str(row["signal_type"] or ""),
                        "strategy_type": row["strategy_type"],
                        "market_id": str(row["market_id"] or ""),
                        "direction": row["direction"],
                        "entry_price": row["entry_price"],
                        "effective_price": row["effective_price"],
                        "edge_percent": row["edge_percent"],
                        "confidence": row["confidence"],
                        "liquidity": row["liquidity"],
                        "status": str(row["status"] or ""),
                        "dedupe_key": str(row["dedupe_key"] or ""),
                        "event_type": "status_update",
                        "reason": f"status:{str(row['status'] or '').strip().lower()}",
                        "payload_json": None,
                        "snapshot_json": None,
                        "created_at": changed_at,
                    }
                    for row in updated_rows
                ]
                # Emission history is append-only + loss-tolerant (offline
                # backtester only, no live reader). Hand it to the background
                # flusher (_run_emission_flush_loop) instead of an inline
                # insert+commit, which held this main-pool connection 2-3.5s per
                # chunk under contention and starved the orchestrator hot path.
                # The trade_signals UPDATE is already committed above, so the
                # session's connection is released as soon as this block exits.
                self._buffer_emissions(emissions)


_intent_runtime: IntentRuntime | None = None


def get_intent_runtime() -> IntentRuntime:
    global _intent_runtime
    if _intent_runtime is None:
        _intent_runtime = IntentRuntime()
    return _intent_runtime
