from __future__ import annotations

import asyncio
import copy
import math
import time
from datetime import datetime, timezone
from typing import Any

from config import settings
from models.database import AsyncSessionLocal
from services import shared_state, trader_binding_cache
from services.crypto_service import get_crypto_service
from services.data_events import DataEvent, EventType
from services.strategy_inputs import build_crypto_update_inputs
from services.event_bus import event_bus
from services.event_dispatcher import event_dispatcher
from services.intent_runtime import get_intent_runtime
from services.machine_learning_sdk import get_machine_learning_sdk
from services.reference_runtime import get_reference_runtime
from services.runtime_status import runtime_status
from services.strategy_helpers.crypto_strategy_utils import enrich_crypto_market_row
from services.strategy_loader import strategy_loader
from services.worker_state import read_worker_control, summarize_worker_stats, write_worker_snapshot
from services.ws_feeds import get_feed_manager
from utils.converters import normalize_identifier as _normalize_market_id
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger(__name__)

# Crypto-latency harness sub-second target: this debounce is a pure
# floor on the WS-tick → strategy-dispatch path.  Default was 0.05s
# (50ms artificial wait per dispatch), which was originally added to
# coalesce burst ticks into one strategy evaluation per token.  In
# practice the ``_pending_tokens`` set already coalesces ticks that
# arrive in the same event-loop iteration without any sleep — the
# drain task acquires the lock, snapshot+clears the set in one
# critical section, so any tick that arrives between drain-task
# creation and lock acquisition gets coalesced for free.  The sleep
# was therefore pure overhead in steady-state single-tick flow.
#
# Default is now 0.0 — the configurable env var
# (``CRYPTO_WS_REACTIVE_DEBOUNCE_SECONDS``) remains an escape hatch
# if a future load profile shows the natural coalescing isn't
# enough.  ``max(0.0, ...)`` clamps negatives but allows true-zero.
_WS_REACTIVE_DEBOUNCE_SECONDS = max(0.0, float(getattr(settings, "CRYPTO_WS_REACTIVE_DEBOUNCE_SECONDS", 0.0) or 0.0))
_CATALOG_REFRESH_SECONDS = 300.0
_CATALOG_MISS_REFRESH_SECONDS = 15.0
_CRYPTO_SNAPSHOT_PERSIST_INTERVAL_SECONDS = 5.0
_FULL_REFRESH_FLOOR_SECONDS = 0.5
_LOOP_ITERATION_TIMEOUT_SECONDS = 30.0
_ASYNC_TIMEOUT_CANCEL_GRACE_SECONDS = 5.0
_ML_PRUNE_INTERVAL_SECONDS = 3600.0
_ML_RUNTIME_GATE_TTL_SECONDS = 10.0
_ML_RUNTIME_FAILURE_BACKOFF_SECONDS = 30.0
_ML_RUNTIME_FAILURE_LOG_INTERVAL_SECONDS = 300.0
_ML_RUNTIME_STATE_TIMEOUT_SECONDS = 3.0
_ML_ANNOTATE_TIMEOUT_SECONDS = 8.0
_ML_RECORD_TIMEOUT_SECONDS = 8.0
_ML_PRUNE_TIMEOUT_SECONDS = 8.0
_CRYPTO_MARKET_FETCH_TIMEOUT_SECONDS = 10.0
_CRYPTO_SUBSCRIPTION_SYNC_TIMEOUT_SECONDS = 3.0
_CRYPTO_SNAPSHOT_PUBLISH_TIMEOUT_SECONDS = 5.0
_BOUNDARY_INTERVALS_SECONDS = (300, 900, 3600, 14400)
_BOUNDARY_PREFETCH_WINDOW_SECONDS = 15
_BOUNDARY_LINGER_WINDOW_SECONDS = 10
_CRYPTO_ML_TASK_KEY = "crypto_directional"
_CRYPTO_ORACLE_HISTORY_POINTS = 120


def _oracle_history_max_age_seconds(timeframe: Any) -> float:
    """Map a market timeframe to the chart's visible window in seconds.

    Mirrors the frontend ``livelineWindow`` switch in CryptoMarketsPanel
    so the backend always sends just enough history to fill the chart
    plus a small buffer.  Without this the dense WS tick stream (~10/s)
    pushes backfilled minute-klines out of the response within seconds
    and the chart squishes to ~12s of data on a 900s axis.
    """
    raw = str(timeframe or "").lower()
    if "4h" in raw or "240" in raw:
        return 4 * 3600.0
    if "1h" in raw or "60" in raw:
        return 3600.0
    if "15m" in raw or "15min" in raw or "15-min" in raw:
        return 900.0
    if "5m" in raw or "5min" in raw or "5-min" in raw or (raw.startswith("5") and "15" not in raw):
        return 300.0
    return 900.0


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


# Hot-tier projection: fields that the event-catalog lookup path actually
# consumes (via ``_build_event_market_snapshot`` / ``get_market_snapshot``
# / ``get_token_mid_price``).  Anything outside this set is dropped before
# storage in ``_event_catalog_markets`` to keep the runtime heap small.
# Heavy excluded fields: ``description`` (~hundreds of bytes per market),
# ``tags`` (list of dicts), ``events`` (back-reference to parent event),
# ``image`` / ``icon`` (URLs), ``outcomes`` (rarely consulted at lookup).
# If a downstream consumer needs one of those, fetch from the canonical
# Market source (DB / catalog) rather than relying on the runtime cache.
_HOT_MARKET_FIELDS: frozenset[str] = frozenset({
    "id",
    "market_id",
    "condition_id",
    "conditionId",
    "slug",
    "question",
    "group_item_title",
    "category",
    "platform",
    "active",
    "closed",
    "archived",
    "resolved",
    "accepting_orders",
    "enable_order_book",
    "status",
    "neg_risk",
    "sports_market_type",
    "asset",
    "timeframe",
    "line",
    "volume",
    "liquidity",
    "yes_price",
    "no_price",
    "up_price",
    "down_price",
    "outcome_prices",
    "clob_token_ids",
    "token_ids",
    "end_date",
    "end_time",
    "start_date",
    "start_time",
    "game_start_time",
    "expires_at",
    "is_current",
    "is_live",
    "fetched_at",
    "event_id",
    "event_slug",
    "event_title",
    "event_category",
    "seconds_left",
    "price_to_beat",
    "combined",
})


def _project_hot_market(value: Any) -> dict[str, Any]:
    """Project a market dict down to fields the runtime hot path consumes.

    Slims the stored row from ~30+ fields (~2KB) to ~15-20 fields
    (~600 bytes), saving ~60% per ``_event_catalog_markets`` entry.
    Used only for the in-memory lookup cache; full Market rows still
    flow through the boundary code (DB writes, API responses) untouched.
    """
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if k in _HOT_MARKET_FIELDS}


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _enrich_crypto_motion_fields(row: dict[str, Any], motion_summary: dict[str, Any] | None) -> None:
    summary = dict(motion_summary or {})
    if not summary:
        return

    history_summary = dict(row.get("oracle_history_summary") or {})
    for key in ("move_5m", "move_30m", "move_2h"):
        move = summary.get(key)
        if not isinstance(move, dict):
            continue
        percent = _to_float(move.get("percent"))
        if percent is None:
            continue
        history_summary[key] = dict(move)
        percent_key = f"{key}_percent"
        pct_key = f"{key}_pct"
        if row.get(percent_key) is None:
            row[percent_key] = percent
        if row.get(pct_key) is None:
            row[pct_key] = percent

    if summary.get("history_coverage_seconds") is not None:
        history_summary["history_coverage_seconds"] = summary.get("history_coverage_seconds")
    if summary.get("latest_ts_ms") is not None:
        history_summary["latest_ts_ms"] = summary.get("latest_ts_ms")
    if summary.get("latest_price") is not None:
        history_summary["latest_price"] = summary.get("latest_price")
    if history_summary:
        row["oracle_history_summary"] = history_summary

    if row.get("recent_move_zscore") is not None:
        return
    move_5m = _to_float(row.get("move_5m_percent"))
    if move_5m is None:
        return
    move_30m = _to_float(row.get("move_30m_percent"))
    move_2h = _to_float(row.get("move_2h_percent"))
    move_scale = max(
        abs(move_30m or 0.0) / 2.0,
        abs(move_2h or 0.0) / 3.0,
        0.25,
    )
    row["recent_move_zscore"] = abs(move_5m) / move_scale if move_scale > 0 else None


def _metadata_updated_at_iso(value: Any) -> str | None:
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            return None
    text = str(value or "").strip()
    return text or None


def _near_market_boundary() -> bool:
    now_ts = time.time()
    for interval in _BOUNDARY_INTERVALS_SECONDS:
        seconds_into = now_ts % interval
        seconds_until_next = interval - seconds_into
        if seconds_until_next <= _BOUNDARY_PREFETCH_WINDOW_SECONDS:
            return True
        if seconds_into <= _BOUNDARY_LINGER_WINDOW_SECONDS:
            return True
    return False


def _crypto_lane_is_active(control: dict[str, Any]) -> bool:
    """Return True when the crypto fast-binary lane should run.

    Both ``is_enabled = false`` (operator wants the lane permanently
    off) and ``is_paused = true`` (operator wants it temporarily
    paused — this is what ``POST /api/workers/crypto/pause`` sets) map
    to "lane off" for the purposes of cache management and the
    reactive payload rebuild.
    """
    enabled = bool(control.get("is_enabled", True))
    paused = bool(control.get("is_paused", False))
    return enabled and not paused


def _loaded_crypto_strategy_instances() -> list[tuple[str, Any]]:
    seen: set[str] = set()
    out: list[tuple[str, Any]] = []
    for slug, _handler in list(event_dispatcher._handlers.get(EventType.CRYPTO_UPDATE, [])):
        normalized_slug = str(slug or "").strip().lower()
        if not normalized_slug or normalized_slug in seen:
            continue
        seen.add(normalized_slug)
        instance = strategy_loader.get_instance(normalized_slug)
        if instance is None:
            continue
        if str(getattr(instance, "source_key", "") or "").strip().lower() != "crypto":
            continue
        out.append((normalized_slug, instance))
    return out


_PER_TRADER_ON_EVENT_TIMEOUT_SECONDS = 15.0


async def _dispatch_with_per_trader_fanout(event: DataEvent) -> list[Any]:
    """Dispatch a crypto ``DataEvent`` honoring per-trader strategy params.

    Plan 0041. For each strategy slug subscribed to the event's type:

    - If at least one enabled trader has a per-trader binding for that slug
      (via ``traders.source_configs_json``), invoke the per-trader instance
      (built lazily by ``strategy_loader.get_or_clone_for_trader``) once per
      bound trader. Tag every emitted opportunity with ``intended_trader_id``
      so ``intent_runtime`` can route the resulting ``trade_signals`` row
      only to that trader.
    - Otherwise (no enabled trader bound to the slug), dispatch through the
      shared singleton via ``event_dispatcher.dispatch`` — preserves the
      existing 60 s handler timeout + force-kill machinery and emits
      un-tagged opportunities that any trader bound to the source can pick
      up (legacy behaviour).

    Returns the flattened opportunity list. Per-trader invocations are
    bounded by ``_PER_TRADER_ON_EVENT_TIMEOUT_SECONDS`` so a single hung
    trader instance cannot stall the dispatch loop.
    """
    subscribed = event_dispatcher.subscribed_slugs(event.event_type)
    if not subscribed:
        return []

    bindings = await trader_binding_cache.get_bindings_for_source("crypto")
    per_trader_slugs = {slug for slug in subscribed if bindings.get(slug)}
    singleton_slugs = subscribed - per_trader_slugs

    opportunities: list[Any] = []

    if singleton_slugs:
        opportunities.extend(
            await event_dispatcher.dispatch(event, include_strategies=singleton_slugs)
        )

    for slug in sorted(per_trader_slugs):
        for trader_id, trader_params in bindings[slug]:
            loaded = strategy_loader.get_or_clone_for_trader(slug, trader_id, trader_params)
            if loaded is None:
                continue
            try:
                result = await asyncio.wait_for(
                    loaded.instance.on_event(event),
                    timeout=_PER_TRADER_ON_EVENT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "per-trader on_event timed out",
                    slug=slug,
                    trader_id=trader_id,
                    timeout_seconds=_PER_TRADER_ON_EVENT_TIMEOUT_SECONDS,
                )
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "per-trader on_event failed",
                    slug=slug,
                    trader_id=trader_id,
                    exc_info=exc,
                )
                continue
            if not result:
                continue
            for opp in result:
                try:
                    opp.intended_trader_id = trader_id
                except Exception as tag_exc:
                    # Plan 0041: if Opportunity ever turns Pydantic-strict
                    # (`extra="forbid"` / `frozen=True`), the attribute
                    # write fails and the per-trader fan-out would drop
                    # the opp silently. Log it loudly so the regression
                    # surfaces — Bug 2 from the Plan 0041 audit.
                    logger.warning(
                        "per-trader opportunity tag failed; dropping opp",
                        slug=slug,
                        trader_id=trader_id,
                        opp_type=type(opp).__name__,
                        exc_info=tag_exc,
                    )
                    continue
                opportunities.append(opp)

    return opportunities


def _diagnostic_rejection_counts(diag: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    rejections = diag.get("rejections")
    if isinstance(rejections, dict):
        for key, value in rejections.items():
            try:
                counts[str(key or "").strip() or "other"] = int(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(rejections, list):
        for item in rejections:
            if not isinstance(item, dict):
                continue
            key = str(item.get("gate") or item.get("reason") or "other").strip() or "other"
            counts[key] = counts.get(key, 0) + 1

    summary = diag.get("summary")
    if isinstance(summary, dict):
        for key, value in summary.items():
            normalized_key = str(key or "").strip()
            if not normalized_key.startswith("rejected_"):
                continue
            try:
                counts[normalized_key.removeprefix("rejected_") or "other"] = int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counts.items()))


def _build_crypto_filter_diagnostics(
    strategy_instances: list[tuple[str, Any]],
    opportunities: list[Any],
) -> dict[str, Any]:
    if not strategy_instances:
        return {}

    opportunity_counts: dict[str, int] = {}
    for opportunity in opportunities:
        strategy_key = str(getattr(opportunity, "strategy", "") or "").strip().lower()
        if not strategy_key:
            continue
        opportunity_counts[strategy_key] = opportunity_counts.get(strategy_key, 0) + 1

    per_strategy: dict[str, dict[str, Any]] = {}
    rejection_counts_by_strategy: dict[str, dict[str, int]] = {}
    strategies_missing_diagnostics: list[str] = []
    primary_strategy_key = ""
    primary_rank = (-1, -1)
    markets_scanned = 0

    for slug, instance in strategy_instances:
        diag_fn = getattr(instance, "get_filter_diagnostics", None)
        diag = diag_fn() if callable(diag_fn) else None
        if isinstance(diag, dict) and diag:
            strategy_diag = copy.deepcopy(diag)
            has_diag = 1
        else:
            strategy_diag = {
                "strategy_key": slug,
                "markets_scanned": 0,
                "signals_emitted": 0,
                "message": "No diagnostics reported",
                "summary": {},
            }
            strategies_missing_diagnostics.append(slug)
            has_diag = 0

        strategy_diag["strategy_key"] = slug
        strategy_diag["signals_emitted"] = int(opportunity_counts.get(slug, strategy_diag.get("signals_emitted") or 0))
        strategy_diag["opportunities_emitted"] = int(opportunity_counts.get(slug, 0))
        per_strategy[slug] = strategy_diag
        markets_scanned = max(markets_scanned, int(strategy_diag.get("markets_scanned") or 0))

        rejection_counts = _diagnostic_rejection_counts(strategy_diag)
        if rejection_counts:
            rejection_counts_by_strategy[slug] = rejection_counts

        rank = (int(strategy_diag.get("signals_emitted") or 0), has_diag)
        if rank > primary_rank:
            primary_rank = rank
            primary_strategy_key = slug

    primary = copy.deepcopy(per_strategy.get(primary_strategy_key) or {})
    summary = dict(primary.get("summary") or {})
    summary.update({
        "strategies_loaded": len(strategy_instances),
        "strategies_reporting_diagnostics": len(strategy_instances) - len(strategies_missing_diagnostics),
        "strategies_with_signals": sum(1 for value in opportunity_counts.values() if value > 0),
        "total_signals_emitted": len(opportunities),
    })
    ordered_strategy_keys = sorted(
        per_strategy.keys(),
        key=lambda slug: (-int(per_strategy[slug].get("signals_emitted") or 0), slug),
    )
    detail_parts: list[str] = []
    for slug in ordered_strategy_keys[:4]:
        detail = str(per_strategy[slug].get("message") or "").strip()
        if not detail:
            detail = f"{int(per_strategy[slug].get('signals_emitted') or 0)} signals"
        detail_parts.append(f"{slug}: {detail}")

    primary.update({
        "strategy_key": primary_strategy_key or None,
        "scanned_at": str(primary.get("scanned_at") or utcnow().isoformat().replace("+00:00", "Z")),
        "markets_scanned": markets_scanned,
        "signals_emitted": len(opportunities),
        "summary": summary,
        "primary_strategy_key": primary_strategy_key or None,
        "strategies": per_strategy,
        "dispatch_summary": {
            "strategies_loaded": len(strategy_instances),
            "strategies_reporting_diagnostics": len(strategy_instances) - len(strategies_missing_diagnostics),
            "strategies_missing_diagnostics": strategies_missing_diagnostics,
            "opportunities_by_strategy": dict(sorted(opportunity_counts.items())),
            "rejection_counts_by_strategy": rejection_counts_by_strategy,
        },
    })
    primary["message"] = (
        f"Scanned {markets_scanned} markets across {len(strategy_instances)} crypto strategies, "
        f"{len(opportunities)} signals total"
    )
    if detail_parts:
        primary["message"] = f"{primary['message']} — {' | '.join(detail_parts)}"
    return primary


class MarketRuntime:
    def __init__(self) -> None:
        self._started = False
        self._start_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._main_task: asyncio.Task[None] | None = None
        self._loop_iteration_task: asyncio.Task[None] | None = None
        self._reactive_task: asyncio.Task[None] | None = None
        self._opportunity_dispatch_task: asyncio.Task[None] | None = None
        self._ml_pipeline_task: asyncio.Task[None] | None = None
        self._snapshot_persist_task: asyncio.Task[None] | None = None
        self._feed_manager = None
        self._reference_runtime = get_reference_runtime()
        self._crypto_markets: list[dict[str, Any]] = []
        self._crypto_markets_by_lookup: dict[str, dict[str, Any]] = {}
        self._crypto_token_to_market_ids: dict[str, set[str]] = {}
        self._crypto_asset_to_market_ids: dict[str, set[str]] = {}
        self._event_catalog_markets: dict[str, dict[str, Any]] = {}
        self._event_catalog_updated_at: str | None = None
        self._last_crypto_refresh_at: str | None = None
        self._last_crypto_trigger: str | None = None
        self._current_activity = "Starting"
        self._last_error: str | None = None
        self._last_catalog_refresh_mono = 0.0
        self._last_snapshot_persist_mono = 0.0
        self._last_ml_prune_mono = 0.0
        self._last_ml_gate_check_mono = 0.0
        self._ml_runtime_retry_not_before_mono = 0.0
        self._ml_runtime_recording_enabled = False
        self._ml_runtime_deployment_active = False
        self._last_ml_runtime_failure_log_mono = 0.0
        self._ml_runtime_state_lock = asyncio.Lock()
        self._ml_runtime_refresh_task: asyncio.Task[None] | None = None
        self._event_catalog_refresh_task: asyncio.Task[None] | None = None
        self._ml_pipeline_lock = asyncio.Lock()
        self._abandoned_tasks: set[asyncio.Task[Any]] = set()
        self._pending_tokens: set[str] = set()
        self._pending_assets: set[str] = set()
        self._pending_reactive_lock = asyncio.Lock()
        self._pending_opportunity_payload: list[dict[str, Any]] | None = None
        self._pending_opportunity_trigger: str | None = None
        self._pending_opportunity_full_source_sweep = False
        self._pending_opportunity_lock = asyncio.Lock()
        self._pending_ml_payload: list[dict[str, Any]] | None = None
        self._pending_ml_allow_record = False
        self._pending_ml_lock = asyncio.Lock()
        self._pending_snapshot_persist = False
        self._pending_snapshot_control: dict[str, Any] | None = None
        self._pending_snapshot_force = False
        self._pending_snapshot_lock = asyncio.Lock()
        # Dispatch telemetry — visible via worker snapshot stats
        self._dispatch_count: int = 0
        self._dispatch_last_at: str | None = None
        self._dispatch_last_trigger: str | None = None
        self._dispatch_last_handlers: int = 0
        self._dispatch_last_opportunities: int = 0
        self._dispatch_last_signals_published: int = 0
        self._dispatch_last_error: str | None = None
        self._dispatch_filter_diagnostics: dict[str, Any] = {}
        # Crypto-lane toggle plumbing — operator-driven on/off via the
        # `worker_control(name='crypto')` row. The reactive-tick path
        # reads the control on every Binance update, so a TTL cache is
        # used to keep the DB out of the hot loop. The transition flag
        # lets `_run_loop_iteration` clear/refresh the cache exactly
        # once on edge changes.
        self._crypto_control_cache: dict[str, Any] | None = None
        self._crypto_control_cache_at: float = 0.0
        self._crypto_lane_was_enabled: bool | None = None
        self._crypto_lane_pending_refresh: bool = False

    def _retain_abandoned_task(self, task: asyncio.Task[Any]) -> None:
        self._abandoned_tasks.add(task)
        task.add_done_callback(self._abandoned_tasks.discard)

    def _clear_loop_iteration_task(self, task: asyncio.Task[Any]) -> None:
        if self._loop_iteration_task is task:
            self._loop_iteration_task = None

    def _clear_ml_runtime_refresh_task(self, task: asyncio.Task[Any]) -> None:
        if self._ml_runtime_refresh_task is task:
            self._ml_runtime_refresh_task = None

    def _clear_event_catalog_refresh_task(self, task: asyncio.Task[Any]) -> None:
        if self._event_catalog_refresh_task is task:
            self._event_catalog_refresh_task = None

    def _clear_snapshot_persist_task(self, task: asyncio.Task[Any]) -> None:
        if self._snapshot_persist_task is task:
            self._snapshot_persist_task = None

    def _cached_ml_runtime_state(self, *, allow_record: bool) -> dict[str, bool]:
        return {
            "recording_enabled": (self._ml_runtime_recording_enabled if allow_record else False),
            "deployment_active": self._ml_runtime_deployment_active,
        }

    def _schedule_ml_runtime_state_refresh(self) -> None:
        refresh_task = self._ml_runtime_refresh_task
        if refresh_task is not None and not refresh_task.done():
            return

        async def _refresh() -> None:
            await self._resolve_ml_runtime_state(allow_record=True)

        refresh_task = asyncio.create_task(
            _refresh(),
            name="market-runtime-ml-state-refresh",
        )
        self._ml_runtime_refresh_task = refresh_task
        refresh_task.add_done_callback(self._clear_ml_runtime_refresh_task)

    def _schedule_event_catalog_refresh(self, *, force: bool = False) -> None:
        refresh_task = self._event_catalog_refresh_task
        if refresh_task is not None and not refresh_task.done():
            return

        async def _refresh() -> None:
            await self._refresh_event_catalog(force=force)

        refresh_task = asyncio.create_task(
            _refresh(),
            name="market-runtime-event-catalog-refresh",
        )
        self._event_catalog_refresh_task = refresh_task
        refresh_task.add_done_callback(self._clear_event_catalog_refresh_task)

    def _schedule_event_catalog_refresh_on_miss(self) -> None:
        if not self._started:
            return
        if (time.monotonic() - self._last_catalog_refresh_mono) < _CATALOG_MISS_REFRESH_SECONDS:
            return
        self._schedule_event_catalog_refresh(force=True)

    async def _await_with_cancel_grace(
        self,
        awaitable: Any,
        *,
        timeout: float,
        task_name: str,
    ) -> Any:
        task = awaitable if isinstance(awaitable, asyncio.Task) else asyncio.create_task(awaitable, name=task_name)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if done:
                return task.result()

            task.cancel()
            done_after, _ = await asyncio.wait({task}, timeout=_ASYNC_TIMEOUT_CANCEL_GRACE_SECONDS)
            if not done_after:
                self._retain_abandoned_task(task)
            else:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            raise asyncio.TimeoutError()
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.shield(asyncio.wait({task}, timeout=_ASYNC_TIMEOUT_CANCEL_GRACE_SECONDS))
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                if not task.done():
                    self._retain_abandoned_task(task)
            raise

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            self._stop_event.clear()
            # Register the bus topic for the live CRYPTO_UPDATE tap.
            # Done before the dispatch loop ever fires so the first
            # publish doesn't fail-closed on an unregistered topic.
            # Idempotent — operator-controlled retention/enabled fields
            # survive recorder restarts.
            try:
                await _ensure_crypto_update_topic_registered()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to register crypto.update bus topic; tap will be no-op",
                    exc_info=True,
                )
            await self._reference_runtime.start()
            self._reference_runtime.on_update(self._on_reference_update)
            self._feed_manager = get_feed_manager()
            if not getattr(self._feed_manager, "_started", False):
                await self._feed_manager.start()
            self._feed_manager.cache.add_on_update_callback(self._on_ws_price_update)
            self._schedule_event_catalog_refresh(force=True)
            await self._backfill_oracle_history_from_binance()
            startup_control = await self._read_crypto_control()
            startup_active = _crypto_lane_is_active(startup_control)
            if startup_active:
                await self._refresh_crypto_markets(trigger="startup", full_source_sweep=True)
            else:
                logger.info(
                    "Crypto fast-binary lane disabled by worker_control; skipping startup refresh"
                )
                self._crypto_markets = []
                self._crypto_markets_by_lookup = {}
                self._crypto_token_to_market_ids = {}
                self._crypto_asset_to_market_ids = {}
            self._crypto_lane_was_enabled = startup_active
            self._started = True
            self._main_task = asyncio.create_task(self._run_loop(), name="market-runtime")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._event_catalog_refresh_task is not None and not self._event_catalog_refresh_task.done():
            self._event_catalog_refresh_task.cancel()
            try:
                await self._event_catalog_refresh_task
            except asyncio.CancelledError:
                pass
        if self._ml_runtime_refresh_task is not None and not self._ml_runtime_refresh_task.done():
            self._ml_runtime_refresh_task.cancel()
            try:
                await self._ml_runtime_refresh_task
            except asyncio.CancelledError:
                pass
        if self._reactive_task is not None and not self._reactive_task.done():
            self._reactive_task.cancel()
            try:
                await self._reactive_task
            except asyncio.CancelledError:
                pass
        if self._opportunity_dispatch_task is not None and not self._opportunity_dispatch_task.done():
            self._opportunity_dispatch_task.cancel()
            try:
                await self._opportunity_dispatch_task
            except asyncio.CancelledError:
                pass
        if self._ml_pipeline_task is not None and not self._ml_pipeline_task.done():
            self._ml_pipeline_task.cancel()
            try:
                await self._ml_pipeline_task
            except asyncio.CancelledError:
                pass
        if self._snapshot_persist_task is not None and not self._snapshot_persist_task.done():
            self._snapshot_persist_task.cancel()
            try:
                await self._snapshot_persist_task
            except asyncio.CancelledError:
                pass
        if self._main_task is not None and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        self._reference_runtime.remove_on_update(self._on_reference_update)
        await self._reference_runtime.stop()
        self._started = False

    def get_crypto_markets(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._crypto_markets)

    def _get_ws_status(self) -> dict[str, Any]:
        ws_status = {}
        if self._feed_manager is not None:
            try:
                ws_status = self._feed_manager.health_check()
            except Exception:
                ws_status = {}
        return ws_status

    def _build_crypto_stats(
        self,
        *,
        include_markets: bool = False,
    ) -> dict[str, Any]:
        stats = {
            "market_count": len(self._crypto_markets),
            "oracle_prices": self.get_oracle_prices(),
            "trigger": self._last_crypto_trigger,
            "ws_feeds": self._get_ws_status(),
            **self._reference_runtime.get_status(),
            "dispatch": {
                "total_dispatches": self._dispatch_count,
                "last_at": self._dispatch_last_at,
                "last_trigger": self._dispatch_last_trigger,
                "last_handlers": self._dispatch_last_handlers,
                "last_opportunities": self._dispatch_last_opportunities,
                "last_signals_published": self._dispatch_last_signals_published,
                "last_error": self._dispatch_last_error,
            },
            "filter_diagnostics": self._dispatch_filter_diagnostics,
        }
        if include_markets:
            stats["markets"] = self.get_crypto_markets()
        return stats

    def get_crypto_status(self) -> dict[str, Any]:
        return {
            "running": bool(self._started),
            "enabled": True,
            "current_activity": self._current_activity,
            "last_run_at": self._last_crypto_refresh_at,
            "last_error": self._last_error,
            "stats": self._build_crypto_stats(include_markets=True),
        }

    def _last_crypto_refresh_datetime(self) -> datetime | None:
        raw_value = str(self._last_crypto_refresh_at or "").strip()
        if not raw_value:
            return None
        try:
            parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    async def _persist_crypto_worker_snapshot(
        self,
        *,
        control: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        resolved_control = dict(control or await self._read_crypto_control())
        stats = self._build_crypto_stats(include_markets=False)
        persisted_stats = summarize_worker_stats(stats)
        persisted_stats["markets_count"] = len(self._crypto_markets)
        persisted_stats["markets"] = self.get_crypto_markets()
        oracle_prices = stats.get("oracle_prices")
        if isinstance(oracle_prices, dict) and oracle_prices:
            persisted_stats["oracle_prices"] = copy.deepcopy(oracle_prices)
        enabled = bool(resolved_control.get("is_enabled", True))
        paused = bool(resolved_control.get("is_paused", False))
        runtime_status.update_crypto(
            running=True,
            enabled=enabled and not paused,
            current_activity=str(self._current_activity or "Idle"),
            interval_seconds=int(resolved_control.get("interval_seconds") or 1),
            last_run_at=self._last_crypto_refresh_at,
            last_error=self._last_error,
            stats=stats,
            control=resolved_control,
        )
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_snapshot_persist_mono) < _CRYPTO_SNAPSHOT_PERSIST_INTERVAL_SECONDS:
            return
        async with AsyncSessionLocal() as session:
            await write_worker_snapshot(
                session,
                "crypto",
                running=True,
                enabled=enabled and not paused,
                current_activity=str(self._current_activity or "Idle"),
                interval_seconds=int(resolved_control.get("interval_seconds") or 1),
                last_run_at=self._last_crypto_refresh_datetime(),
                last_error=self._last_error,
                stats=persisted_stats,
            )
        self._last_snapshot_persist_mono = now_mono

    async def _queue_crypto_worker_snapshot_persist(
        self,
        *,
        control: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        async with self._pending_snapshot_lock:
            self._pending_snapshot_persist = True
            self._pending_snapshot_control = dict(control) if isinstance(control, dict) else None
            self._pending_snapshot_force = self._pending_snapshot_force or bool(force)
        if self._snapshot_persist_task is None or self._snapshot_persist_task.done():
            task = asyncio.create_task(
                self._run_snapshot_persist_loop(),
                name="market-runtime-snapshot-persist",
            )
            self._snapshot_persist_task = task
            task.add_done_callback(self._clear_snapshot_persist_task)

    async def _run_snapshot_persist_loop(self) -> None:
        while True:
            async with self._pending_snapshot_lock:
                pending = bool(self._pending_snapshot_persist)
                control = self._pending_snapshot_control
                force = bool(self._pending_snapshot_force)
                self._pending_snapshot_persist = False
                self._pending_snapshot_control = None
                self._pending_snapshot_force = False
            if not pending:
                return
            try:
                await self._persist_crypto_worker_snapshot(control=control, force=force)
            except asyncio.CancelledError:
                raise
            except Exception as snapshot_exc:
                logger.warning("Failed to persist crypto worker snapshot after refresh", exc_info=snapshot_exc)

    def get_market_snapshot(self, market_id: str, *, hint: dict[str, Any] | None = None) -> dict[str, Any] | None:
        normalized = _normalize_market_id(market_id)
        hinted = hint if isinstance(hint, dict) else {}
        if not self._started:
            return copy.deepcopy(hinted) if hinted else None
        if normalized:
            crypto = self._crypto_markets_by_lookup.get(normalized)
            if crypto is not None:
                return copy.deepcopy(crypto)
            event_market = self._event_catalog_markets.get(normalized)
            if event_market is not None:
                return self._build_event_market_snapshot(event_market)
        for key in (
            "condition_id",
            "conditionId",
            "id",
            "market_id",
        ):
            normalized_hint = _normalize_market_id(hinted.get(key))
            if not normalized_hint:
                continue
            crypto = self._crypto_markets_by_lookup.get(normalized_hint)
            if crypto is not None:
                return copy.deepcopy(crypto)
            event_market = self._event_catalog_markets.get(normalized_hint)
            if event_market is not None:
                return self._build_event_market_snapshot(event_market)
        token_ids = hinted.get("clob_token_ids") or hinted.get("token_ids") or []
        if isinstance(token_ids, list):
            for raw_token_id in token_ids:
                token_id = _normalize_market_id(raw_token_id)
                if not token_id:
                    continue
                crypto = self._crypto_markets_by_lookup.get(token_id)
                if crypto is not None:
                    return copy.deepcopy(crypto)
                event_market = self._event_catalog_markets.get(token_id)
                if event_market is not None:
                    return self._build_event_market_snapshot(event_market)
        if normalized or hinted:
            self._schedule_event_catalog_refresh_on_miss()
        return copy.deepcopy(hinted) if hinted else None

    def get_token_mid_price(self, token_id: str) -> float | None:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return None
        snapshot = self.get_market_snapshot(normalized)
        if isinstance(snapshot, dict):
            token_ids = [
                _normalize_market_id(raw_token_id)
                for raw_token_id in (snapshot.get("clob_token_ids") or [])
                if _normalize_market_id(raw_token_id)
            ]
            if token_ids:
                if token_ids[0] == normalized and snapshot.get("up_price") is not None:
                    return _to_float(snapshot.get("up_price"))
                if len(token_ids) > 1 and token_ids[1] == normalized and snapshot.get("down_price") is not None:
                    return _to_float(snapshot.get("down_price"))
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return None
        if not feed_manager.cache.is_fresh(normalized):
            return None
        return _to_float(feed_manager.cache.get_mid_price(normalized))

    def get_token_spread_bps(self, token_id: str) -> float | None:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return None
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return None
        return _to_float(feed_manager.cache.get_spread_bps(normalized))

    def get_price_history(self, token_id: str, *, max_snapshots: int = 60) -> list[dict[str, Any]]:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return []
        snapshot = self.get_market_snapshot(normalized)
        if isinstance(snapshot, dict):
            history_tail = snapshot.get("history_tail")
            if isinstance(history_tail, list) and history_tail:
                return copy.deepcopy(history_tail[-max(1, int(max_snapshots)) :])
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return []
        if not hasattr(feed_manager.cache, "get_price_history"):
            return []
        history = feed_manager.cache.get_price_history(normalized, max_snapshots=max_snapshots)
        return copy.deepcopy(history or [])

    def get_price_change(self, token_id: str, *, lookback_seconds: int = 300) -> dict[str, Any] | None:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return None
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return None
        if not hasattr(feed_manager.cache, "get_price_change"):
            return None
        change = feed_manager.cache.get_price_change(normalized, lookback_seconds=lookback_seconds)
        return copy.deepcopy(change) if isinstance(change, dict) else None

    def get_recent_trades(self, token_id: str, *, max_trades: int = 100) -> list[Any]:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return []
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return []
        if not hasattr(feed_manager.cache, "get_recent_trades"):
            return []
        return list(feed_manager.cache.get_recent_trades(normalized, max_trades) or [])

    def get_trade_volume(self, token_id: str, *, lookback_seconds: float = 300.0) -> dict[str, Any]:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}
        if not hasattr(feed_manager.cache, "get_trade_volume"):
            return {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}
        volume = feed_manager.cache.get_trade_volume(normalized, lookback_seconds)
        return copy.deepcopy(volume) if isinstance(volume, dict) else {"buy_volume": 0.0, "sell_volume": 0.0, "total": 0.0, "trade_count": 0}

    def get_buy_sell_imbalance(self, token_id: str, *, lookback_seconds: float = 300.0) -> float:
        normalized = _normalize_market_id(token_id)
        if not normalized:
            return 0.0
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return 0.0
        if not hasattr(feed_manager.cache, "get_buy_sell_imbalance"):
            return 0.0
        imbalance = _to_float(feed_manager.cache.get_buy_sell_imbalance(normalized, lookback_seconds))
        return float(imbalance or 0.0)

    def get_oracle_prices(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for market in self._crypto_markets:
            asset = str(market.get("asset") or "").strip().upper()
            if not asset:
                continue
            out[asset] = {
                "price": market.get("oracle_price"),
                "updated_at_ms": market.get("oracle_updated_at_ms"),
                "age_seconds": market.get("oracle_age_seconds"),
            }
        return out

    async def _run_loop(self) -> None:
        # Zombie-iteration bookkeeping: an iteration that survives the cancel
        # grace in _await_with_cancel_grace is retained as abandoned, and this
        # loop then waits here until it actually finishes. The 24h soak showed
        # one such wedge lasting 431s (22 iteration timeouts overall): crypto
        # market data went stale and no snapshot heartbeat landed, so the SLO
        # alert fired blind ("cycle took 431s of 30s") with no clue WHERE the
        # iteration was stuck. Two iterations must NOT run concurrently (the
        # iteration mutates shared crypto state without a generation guard),
        # so we keep waiting — but now with a truthful degraded heartbeat
        # every ~15s and a one-shot await-stack dump of the stuck task so the
        # next occurrence pinpoints the wedge line.
        zombie_wait_started_mono: float | None = None
        zombie_last_heartbeat_mono: float = 0.0
        zombie_stack_dumped = False
        while not self._stop_event.is_set():
            stuck_task = self._loop_iteration_task
            if stuck_task is not None and not stuck_task.done():
                now_mono = time.monotonic()
                if zombie_wait_started_mono is None:
                    zombie_wait_started_mono = now_mono
                    zombie_stack_dumped = False
                waited = now_mono - zombie_wait_started_mono
                self._current_activity = f"Degraded: waiting for stuck iteration ({waited:.0f}s)"
                if waited >= 90.0 and not zombie_stack_dumped:
                    zombie_stack_dumped = True
                    try:
                        import io as _io

                        _buf = _io.StringIO()
                        stuck_task.print_stack(limit=14, file=_buf)
                        logger.warning(
                            "Market runtime iteration STUCK %.0fs past its %.0fs timeout — await stack:\n%s",
                            waited,
                            _LOOP_ITERATION_TIMEOUT_SECONDS,
                            _buf.getvalue(),
                        )
                    except Exception:
                        logger.warning(
                            "Market runtime iteration STUCK %.0fs past timeout (stack unavailable)",
                            waited,
                        )
                if (now_mono - zombie_last_heartbeat_mono) >= 15.0:
                    zombie_last_heartbeat_mono = now_mono
                    try:
                        await self._persist_crypto_worker_snapshot(force=True)
                    except Exception as snapshot_exc:
                        logger.debug(
                            "Degraded-heartbeat snapshot persist failed",
                            exc_info=snapshot_exc,
                        )
                await asyncio.sleep(1.0)
                continue
            zombie_wait_started_mono = None
            iteration_task = asyncio.create_task(self._run_loop_iteration(), name="market-runtime-loop-iteration")
            self._loop_iteration_task = iteration_task
            iteration_task.add_done_callback(self._clear_loop_iteration_task)
            try:
                sleep_seconds = await self._await_with_cancel_grace(
                    iteration_task,
                    task_name="market-runtime-loop-iteration",
                    timeout=_LOOP_ITERATION_TIMEOUT_SECONDS,
                )
                await asyncio.sleep(max(_FULL_REFRESH_FLOOR_SECONDS, float(sleep_seconds or 0.0)))
            except asyncio.TimeoutError:
                self._last_error = "Loop iteration timed out"
                self._current_activity = "Error: loop iteration timeout"
                logger.warning(
                    "Market runtime loop iteration timed out after %.0fs",
                    _LOOP_ITERATION_TIMEOUT_SECONDS,
                )
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                self._current_activity = f"Error: {exc}"
                logger.warning("Market runtime loop failed", exc_info=exc)
                try:
                    await self._persist_crypto_worker_snapshot(
                        control={"is_enabled": True, "is_paused": False, "interval_seconds": 1},
                        force=True,
                    )
                except Exception as snapshot_exc:
                    logger.warning("Failed to persist crypto worker snapshot after runtime error", exc_info=snapshot_exc)
                await asyncio.sleep(1.0)

    async def _run_loop_iteration(self) -> float:
        if (time.monotonic() - self._last_catalog_refresh_mono) >= _CATALOG_REFRESH_SECONDS:
            self._schedule_event_catalog_refresh()
        control = await self._read_crypto_control()
        enabled = bool(control.get("is_enabled", True))
        paused = bool(control.get("is_paused", False))
        active = enabled and not paused
        interval_seconds = max(_FULL_REFRESH_FLOOR_SECONDS, float(control.get("interval_seconds") or 1.0))
        # Lane on/off transition handling — clear cache on disable so
        # `get_crypto_markets()` returns [] immediately, and request a
        # one-shot refresh on re-enable so the cache repopulates on
        # this same iteration. The active state collapses both
        # ``is_enabled = false`` and ``is_paused = true`` into a single
        # condition so the toggle works regardless of which the
        # operator flipped.
        was_active = self._crypto_lane_was_enabled
        if was_active is True and not active:
            logger.info("Crypto fast-binary lane disabled; clearing cache")
            self._crypto_markets = []
            self._crypto_markets_by_lookup = {}
            self._crypto_token_to_market_ids = {}
            self._crypto_asset_to_market_ids = {}
            self._crypto_lane_pending_refresh = False
        elif was_active is False and active:
            logger.info("Crypto fast-binary lane re-enabled; scheduling refresh")
            self._crypto_lane_pending_refresh = True
        self._crypto_lane_was_enabled = active
        if active:
            trigger = "periodic_scan"
            if self._crypto_lane_pending_refresh:
                trigger = "lane_re_enabled"
                self._crypto_lane_pending_refresh = False
            await self._refresh_crypto_markets(
                trigger=trigger,
                full_source_sweep=True,
                force_refresh=_near_market_boundary() or trigger == "lane_re_enabled",
            )
            self._current_activity = "Live"
        else:
            self._current_activity = "Paused" if paused else "Disabled"
            try:
                await self._persist_crypto_worker_snapshot(control=control, force=True)
            except Exception as snapshot_exc:
                logger.warning("Failed to persist crypto worker snapshot", exc_info=snapshot_exc)
        return interval_seconds

    async def _read_crypto_control(self) -> dict[str, Any]:
        try:
            async with AsyncSessionLocal() as session:
                control = await read_worker_control(session, "crypto", default_interval=1)
        except Exception:
            control = {
                "is_enabled": True,
                "is_paused": False,
                "interval_seconds": 1,
                "requested_run_at": None,
            }
        self._crypto_control_cache = control
        self._crypto_control_cache_at = time.monotonic()
        return control

    async def _read_crypto_control_cached(self, *, ttl_seconds: float = 5.0) -> dict[str, Any]:
        # Reactive-tick path reads this on every Binance update, so a
        # short TTL keeps the DB out of the hot loop. The 5 s default
        # matches the operator's expectation for an interactive
        # toggle (lag of at most one TTL between save and effect).
        cache = self._crypto_control_cache
        if cache is not None and (time.monotonic() - self._crypto_control_cache_at) < ttl_seconds:
            return cache
        control = await self._read_crypto_control()
        # Always update the cache, even when ``_read_crypto_control`` is
        # patched in tests and bypasses the cache write inside it.
        self._crypto_control_cache = control
        self._crypto_control_cache_at = time.monotonic()
        return control

    async def _refresh_event_catalog(self, *, force: bool = False) -> None:
        if not force and (time.monotonic() - self._last_catalog_refresh_mono) < _CATALOG_REFRESH_SECONDS:
            return
        async with AsyncSessionLocal() as session:
            _events, _markets, metadata = await shared_state.read_market_catalog(
                session,
                include_events=False,
                include_markets=False,
                validate=False,
            )
        updated_at_iso = _metadata_updated_at_iso(metadata.get("updated_at"))
        if (
            not force
            and self._event_catalog_markets
            and updated_at_iso
            and updated_at_iso == self._event_catalog_updated_at
        ):
            self._last_catalog_refresh_mono = time.monotonic()
            return
        async with AsyncSessionLocal() as session:
            _events, markets, metadata = await shared_state.read_market_catalog(
                session,
                include_events=False,
                include_markets=True,
                validate=False,
            )
        lookup: dict[str, dict[str, Any]] = {}
        for market in markets:
            # Project to the hot-tier field set before storing — drops
            # ~60% of per-row bytes (description, tags, events, etc.)
            # that the runtime lookup path never reads.
            row = _project_hot_market(market)
            market_id = _normalize_market_id(row.get("id"))
            condition_id = _normalize_market_id(row.get("condition_id") or row.get("conditionId"))
            token_ids = [
                _normalize_market_id(token_id)
                for token_id in (row.get("clob_token_ids") or row.get("token_ids") or [])
                if _normalize_market_id(token_id)
            ]
            for key in (market_id, condition_id, *token_ids):
                if key:
                    lookup[key] = row
        self._event_catalog_markets = lookup
        self._event_catalog_updated_at = _metadata_updated_at_iso(metadata.get("updated_at"))
        self._last_catalog_refresh_mono = time.monotonic()

    def _build_event_market_snapshot(self, market: dict[str, Any]) -> dict[str, Any]:
        row = dict(market)
        token_ids = [
            str(token_id or "").strip()
            for token_id in (row.get("clob_token_ids") or row.get("token_ids") or [])
            if str(token_id or "").strip()
        ]
        yes_token = token_ids[0] if token_ids else None
        no_token = token_ids[1] if len(token_ids) > 1 else None
        feed_manager = self._feed_manager
        if feed_manager is not None and getattr(feed_manager, "_started", False):
            if yes_token and feed_manager.cache.is_fresh(yes_token):
                row["yes_price"] = feed_manager.cache.get_mid_price(yes_token)
                row["yes_price_source"] = "redis_strict"
                try:
                    observed_at_epoch = feed_manager.cache.get_observed_at_epoch(yes_token)
                except Exception:
                    observed_at_epoch = None
                if observed_at_epoch is not None:
                    row["yes_price_updated_at"] = datetime.fromtimestamp(
                        float(observed_at_epoch),
                        tz=timezone.utc,
                    ).isoformat().replace("+00:00", "Z")
            if no_token and feed_manager.cache.is_fresh(no_token):
                row["no_price"] = feed_manager.cache.get_mid_price(no_token)
                row["no_price_source"] = "redis_strict"
                try:
                    observed_at_epoch = feed_manager.cache.get_observed_at_epoch(no_token)
                except Exception:
                    observed_at_epoch = None
                if observed_at_epoch is not None:
                    row["no_price_updated_at"] = datetime.fromtimestamp(
                        float(observed_at_epoch),
                        tz=timezone.utc,
                    ).isoformat().replace("+00:00", "Z")
            selected_token = yes_token or no_token
            if selected_token and hasattr(feed_manager.cache, "get_price_history"):
                row["history_tail"] = feed_manager.cache.get_price_history(selected_token, max_snapshots=20)
        return row

    def _index_crypto_market_row(self, row: dict[str, Any]) -> None:
        market_id = _normalize_market_id(row.get("id") or row.get("slug"))
        for key in (
            row.get("id"),
            row.get("slug"),
            row.get("condition_id"),
            row.get("conditionId"),
        ):
            normalized = _normalize_market_id(key)
            if normalized:
                self._crypto_markets_by_lookup[normalized] = row
        for token_id in row.get("clob_token_ids") or []:
            normalized = _normalize_market_id(token_id)
            if not normalized:
                continue
            self._crypto_markets_by_lookup[normalized] = row
            if market_id:
                self._crypto_token_to_market_ids.setdefault(normalized, set()).add(market_id)
        asset = str(row.get("asset") or "").strip().upper()
        if asset and market_id:
            self._crypto_asset_to_market_ids.setdefault(asset, set()).add(market_id)

    def _build_crypto_market_payload(self, markets: list[Any]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        feed_manager = self._feed_manager
        reference_runtime = self._reference_runtime
        now_iso = utcnow().isoformat().replace("+00:00", "Z")
        motion_summary_by_asset: dict[str, dict[str, Any]] = {}
        for market in markets:
            row = market.to_dict()
            row["fetched_at"] = now_iso
            asset = str(row.get("asset") or "").strip().upper()
            oracle = reference_runtime.get_oracle_price(asset) if asset else None
            row["oracle_price"] = oracle.get("price") if oracle else None
            row["oracle_source"] = oracle.get("source") if oracle else None
            row["oracle_updated_at_ms"] = oracle.get("updated_at_ms") if oracle else None
            row["oracle_age_seconds"] = oracle.get("age_seconds") if oracle else None
            row["oracle_prices_by_source"] = reference_runtime.get_oracle_prices_by_source(asset) if asset else {}
            oracle_history = (
                reference_runtime.get_oracle_history(
                    asset,
                    points=_CRYPTO_ORACLE_HISTORY_POINTS,
                    max_age_seconds=_oracle_history_max_age_seconds(row.get("timeframe")),
                )
                if asset
                else []
            )
            row["oracle_history"] = oracle_history
            if asset:
                motion_summary = motion_summary_by_asset.get(asset)
                if motion_summary is None:
                    motion_summary = reference_runtime.get_oracle_motion_summary(asset)
                    motion_summary_by_asset[asset] = motion_summary
                _enrich_crypto_motion_fields(row, motion_summary)
            row["price_updated_at"] = now_iso

            # Fallback: if CryptoService couldn't resolve price_to_beat,
            # derive it from oracle history at the market's start_time.
            if row.get("price_to_beat") is None and oracle_history and row.get("start_time"):
                try:
                    start_dt = datetime.fromisoformat(
                        str(row["start_time"]).replace("Z", "+00:00")
                    )
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                    target_ms = int(start_dt.timestamp() * 1000)
                    best_price = None
                    best_dist = float("inf")
                    for point in oracle_history:
                        ts_ms = int(point.get("t") or 0)
                        price = point.get("p")
                        if not ts_ms or price is None:
                            continue
                        dist = abs(ts_ms - target_ms)
                        if dist < best_dist:
                            best_dist = dist
                            best_price = float(price)
                    # Accept if within 120 seconds of market start
                    if best_price is not None and best_price > 0 and best_dist <= 120_000:
                        row["price_to_beat"] = best_price
                except Exception:
                    pass

            token_ids = [str(token_id or "").strip() for token_id in (row.get("clob_token_ids") or []) if str(token_id or "").strip()]
            if feed_manager is not None and getattr(feed_manager, "_started", False):
                if len(token_ids) > 0 and feed_manager.cache.is_fresh(token_ids[0], max_age_seconds=float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 1.0) or 1.0)):
                    row["up_price"] = feed_manager.cache.get_mid_price(token_ids[0])
                if len(token_ids) > 1 and feed_manager.cache.is_fresh(token_ids[1], max_age_seconds=float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 1.0) or 1.0)):
                    row["down_price"] = feed_manager.cache.get_mid_price(token_ids[1])
                if token_ids:
                    row["history_tail"] = feed_manager.cache.get_price_history(token_ids[0], max_snapshots=20)
            payload.append(row)
        return payload

    async def _refresh_ml_pipeline(self, payload: list[dict[str, Any]], *, allow_record: bool) -> None:
        if not payload:
            return
        if not allow_record and self._ml_pipeline_lock.locked():
            return
        async with self._ml_pipeline_lock:
            now_mono = time.monotonic()
            cache_fresh = (
                now_mono < self._ml_runtime_retry_not_before_mono
                or (now_mono - self._last_ml_gate_check_mono) < _ML_RUNTIME_GATE_TTL_SECONDS
            )
            if not cache_fresh:
                self._schedule_ml_runtime_state_refresh()
            runtime_state = self._cached_ml_runtime_state(allow_record=allow_record)
            # ML training data is sourced from the recorded_event_bus
            # ``crypto.update.dispatch`` topic (parquet) — the SAME crypto
            # market state this dispatch carries is already archived there.
            # The standalone SQL ``ml_training_snapshots`` recorder has been
            # retired (it duplicated the bus and wrote to Postgres on the
            # trading process).  Here we only run live INFERENCE (annotate)
            # when a deployment is active; recording is the bus's job.
            if not runtime_state.get("deployment_active"):
                return
            sdk = get_machine_learning_sdk()
            try:
                await self._await_with_cancel_grace(
                    sdk.annotate_market_batch(task_key=_CRYPTO_ML_TASK_KEY, markets=payload),
                    timeout=_ML_ANNOTATE_TIMEOUT_SECONDS,
                    task_name="market-runtime-ml-annotate",
                )
            except Exception as exc:
                logger.warning("Failed to annotate crypto markets with ML predictions", exc_info=exc)

    async def _queue_ml_pipeline_refresh(self, payload: list[dict[str, Any]], *, allow_record: bool) -> None:
        async with self._pending_ml_lock:
            self._pending_ml_payload = payload
            self._pending_ml_allow_record = self._pending_ml_allow_record or bool(allow_record)
        if self._ml_pipeline_task is None or self._ml_pipeline_task.done():
            self._ml_pipeline_task = asyncio.create_task(
                self._run_ml_pipeline_loop(),
                name="market-runtime-ml-pipeline",
            )

    async def _run_ml_pipeline_loop(self) -> None:
        while True:
            async with self._pending_ml_lock:
                payload = self._pending_ml_payload
                allow_record = self._pending_ml_allow_record
                self._pending_ml_payload = None
                self._pending_ml_allow_record = False
            if not payload:
                return
            try:
                payload_copy = await asyncio.to_thread(copy.deepcopy, payload)
                await self._refresh_ml_pipeline(payload_copy, allow_record=allow_record)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Market runtime ML pipeline refresh failed", exc_info=exc)

    async def _resolve_ml_runtime_state(self, *, allow_record: bool) -> dict[str, bool] | None:
        now_mono = time.monotonic()
        if now_mono < self._ml_runtime_retry_not_before_mono:
            return {
                "recording_enabled": False,
                "deployment_active": False,
            }
        if (now_mono - self._last_ml_gate_check_mono) < _ML_RUNTIME_GATE_TTL_SECONDS:
            return {
                "recording_enabled": (self._ml_runtime_recording_enabled if allow_record else False),
                "deployment_active": self._ml_runtime_deployment_active,
            }

        async with self._ml_runtime_state_lock:
            now_mono = time.monotonic()
            if now_mono < self._ml_runtime_retry_not_before_mono:
                return {
                    "recording_enabled": False,
                    "deployment_active": False,
                }
            if (now_mono - self._last_ml_gate_check_mono) < _ML_RUNTIME_GATE_TTL_SECONDS:
                return {
                    "recording_enabled": (self._ml_runtime_recording_enabled if allow_record else False),
                    "deployment_active": self._ml_runtime_deployment_active,
                }

            sdk = get_machine_learning_sdk()
            try:
                runtime_state = await self._await_with_cancel_grace(
                    sdk.get_runtime_state(_CRYPTO_ML_TASK_KEY),
                    timeout=_ML_RUNTIME_STATE_TIMEOUT_SECONDS,
                    task_name="market-runtime-ml-runtime-state",
                )
            except Exception as exc:
                should_log_warning = self._ml_runtime_recording_enabled or self._ml_runtime_deployment_active
                if should_log_warning:
                    if isinstance(exc, asyncio.TimeoutError):
                        logger.info("ML runtime state unavailable for crypto markets; temporarily disabling ML enrichment")
                    else:
                        logger.warning("Failed to resolve ML runtime state for crypto markets", exc_info=exc)
                    self._last_ml_runtime_failure_log_mono = now_mono
                self._last_ml_gate_check_mono = now_mono
                self._ml_runtime_retry_not_before_mono = now_mono + _ML_RUNTIME_FAILURE_BACKOFF_SECONDS
                self._ml_runtime_recording_enabled = False
                self._ml_runtime_deployment_active = False
                return None

        self._last_ml_gate_check_mono = now_mono
        self._ml_runtime_retry_not_before_mono = 0.0
        self._last_ml_runtime_failure_log_mono = 0.0
        self._ml_runtime_recording_enabled = bool(runtime_state.get("recording_enabled"))
        self._ml_runtime_deployment_active = bool(runtime_state.get("deployment_active"))
        return {
            "recording_enabled": (self._ml_runtime_recording_enabled if allow_record else False),
            "deployment_active": self._ml_runtime_deployment_active,
        }

    async def _refresh_crypto_markets(
        self,
        *,
        trigger: str,
        full_source_sweep: bool,
        force_refresh: bool = False,
    ) -> None:
        step_started = time.monotonic()
        svc = get_crypto_service()
        markets = await self._await_with_cancel_grace(
            asyncio.to_thread(svc.get_live_markets, bool(force_refresh)),
            timeout=_CRYPTO_MARKET_FETCH_TIMEOUT_SECONDS,
            task_name="market-runtime-crypto-market-fetch",
        )
        fetch_elapsed = time.monotonic() - step_started
        step_started = time.monotonic()
        payload = self._build_crypto_market_payload(markets or [])
        build_elapsed = time.monotonic() - step_started
        # 2026-05-08: the combined timing below used to be labeled
        # "ml_queue_seconds" even though it also included the synchronous
        # ``_attach_polymarket_price_history`` call + build payload. During
        # the 2026-05-07 soak this showed ml_queue_seconds=6.5-9.0s,
        # misleadingly pointing at the ML queue when the ML queue itself
        # is fire-and-forget; the true cost was the price-history attach
        # (which in turn delegates into scanner.attach_price_history_to_markets
        # with timeout_seconds=0.0, so the cost is presumably from the
        # in-memory hot-path hydration over many rows rather than a
        # network call). Split the timings so operators can see which
        # part is slow.
        step_started = time.monotonic()
        await self._attach_polymarket_price_history(payload)
        history_elapsed = time.monotonic() - step_started
        step_started = time.monotonic()
        await self._queue_ml_pipeline_refresh(payload, allow_record=True)
        ml_queue_elapsed = time.monotonic() - step_started
        step_started = time.monotonic()
        self._crypto_markets = payload
        self._crypto_markets_by_lookup = {}
        self._crypto_token_to_market_ids = {}
        self._crypto_asset_to_market_ids = {}
        for row in self._crypto_markets:
            self._index_crypto_market_row(row)
        self._last_crypto_refresh_at = utcnow().isoformat().replace("+00:00", "Z")
        self._last_crypto_trigger = str(trigger)

        # Crypto-latency Fix Y: fire-and-forget prewarm of the trading
        # SDK's per-token metadata cache (tick_size / neg_risk /
        # fee_rate_bps) for every crypto market we just discovered.
        # ``py_clob_client_v2.create_order`` otherwise issues up to
        # THREE synchronous HTTP round-trips on the order-creation
        # path for any token whose metadata isn't in the SDK's
        # per-instance dicts — directly observed at 1,000 ms in the
        # 5/2026/05 latency harness ``place_order`` breakdown.  These
        # are static market properties so an aggressive prewarm here,
        # refreshed on every market-runtime crypto refresh (every
        # ~3 min), keeps the cache hot indefinitely.  Background task
        # so it never blocks the dispatch loop; failures inside the
        # prewarm method are non-raising and degrade gracefully back
        # to the SDK's lazy-fetch path.
        try:
            condition_ids = [
                str(row.get("condition_id") or row.get("conditionId") or "").strip()
                for row in payload
                if (row.get("condition_id") or row.get("conditionId"))
            ]
            condition_ids = [cid for cid in condition_ids if cid]
            if condition_ids:
                from services.live_execution_service import live_execution_service

                # Fire-and-forget per-refresh prewarm.  Fix HH attempted
                # to AWAIT this with a 2 s timeout but logs showed 36%
                # of refreshes hitting the timeout — that's 2 s of
                # added dispatch latency on a third of all market
                # ticks, much worse than the cold-start it was trying
                # to prevent.  The initialize-time synchronous prewarm
                # (Fix HH on live_execution_service) already covers
                # the FIRST batch of markets at boot; the per-refresh
                # task here only matters when a NEW market enters the
                # batch, and those rare cases pay one cold-start (an
                # acceptable trade vs blocking every refresh).
                asyncio.create_task(
                    live_execution_service.prewarm_clob_market_info_cache(
                        condition_ids
                    ),
                    name="market-runtime-clob-cache-prewarm",
                )
        except Exception as exc:
            # Never let a prewarm scheduling error trip the runtime
            # refresh.  The SDK lazy-fetch path is the fallback.
            logger.debug("CLOB cache prewarm scheduling failed: %s", exc)
        try:
            await self._await_with_cancel_grace(
                self._sync_crypto_subscriptions(),
                timeout=_CRYPTO_SUBSCRIPTION_SYNC_TIMEOUT_SECONDS,
                task_name="market-runtime-crypto-subscription-sync",
            )
        except asyncio.TimeoutError:
            logger.info("Crypto subscription sync exceeded runtime budget; keeping existing subscriptions")
        subscription_elapsed = time.monotonic() - step_started
        step_started = time.monotonic()
        try:
            await self._await_with_cancel_grace(
                self._publish_crypto_snapshot(payload, trigger=trigger),
                timeout=_CRYPTO_SNAPSHOT_PUBLISH_TIMEOUT_SECONDS,
                task_name="market-runtime-crypto-snapshot-publish",
            )
        except asyncio.TimeoutError:
            logger.info("Crypto snapshot publish exceeded runtime budget; continuing with in-memory state")
        publish_elapsed = time.monotonic() - step_started
        step_started = time.monotonic()
        await self._queue_opportunity_dispatch(
            payload,
            trigger=trigger,
            full_source_sweep=full_source_sweep,
        )
        dispatch_elapsed = time.monotonic() - step_started
        total_elapsed = (
            fetch_elapsed
            + build_elapsed
            + history_elapsed
            + ml_queue_elapsed
            + subscription_elapsed
            + publish_elapsed
            + dispatch_elapsed
        )
        if total_elapsed >= float(settings.MARKET_RUNTIME_REFRESH_WARN_SECONDS):
            logger.warning(
                "Market runtime refresh timing",
                trigger=trigger,
                fetch_seconds=round(fetch_elapsed, 3),
                build_seconds=round(build_elapsed, 3),
                history_seconds=round(history_elapsed, 3),
                ml_queue_seconds=round(ml_queue_elapsed, 3),
                subscription_seconds=round(subscription_elapsed, 3),
                publish_seconds=round(publish_elapsed, 3),
                dispatch_seconds=round(dispatch_elapsed, 3),
                total_seconds=round(total_elapsed, 3),
            )

    async def _sync_crypto_subscriptions(self) -> None:
        feed_manager = self._feed_manager
        if feed_manager is None or not getattr(feed_manager, "_started", False):
            return
        active_tokens = sorted(
            {
                str(token_id or "").strip()
                for market in self._crypto_markets
                for token_id in (market.get("clob_token_ids") or [])
                if str(token_id or "").strip()
            }
        )
        if active_tokens:
            await feed_manager.polymarket_feed.subscribe(active_tokens)

    async def _backfill_oracle_history_from_binance(self) -> None:
        """Seed ``chainlink_feed._history`` from Binance klines on startup.

        Fixes the "chart starts empty after backend restart" gap on the
        crypto opportunity cards: live WS ticks would otherwise need
        minutes to fill the rolling buffer that drives the sparkline.
        Fetches 1m klines for the last 4 hours per supported asset
        (covering 5m / 15m / 1h / 4h windows in one shot) in parallel,
        merges into the rolling history via ``seed_history_from_klines``,
        and tolerates per-asset failures silently.

        Idempotent — re-running just merges any new closes; existing
        same-timestamp points are skipped.  Safe to call from start()
        before the first crypto refresh so the first published snapshot
        already carries a populated ``oracle_history`` array.
        """
        try:
            from services.chainlink_feed import get_chainlink_feed
            from services.crypto_service import fetch_binance_klines
        except Exception as exc:
            logger.debug("Oracle-history backfill skipped (imports unavailable)", exc_info=exc)
            return
        feed = get_chainlink_feed()
        assets = ("BTC", "ETH", "SOL", "XRP")
        lookback_seconds = 4 * 60 * 60  # 4h covers every active timeframe
        results = await asyncio.gather(
            *(
                fetch_binance_klines(asset, lookback_seconds=lookback_seconds, interval="1m")
                for asset in assets
            ),
            return_exceptions=True,
        )
        seeded_total = 0
        for asset, result in zip(assets, results):
            if isinstance(result, BaseException) or not isinstance(result, list):
                continue
            try:
                seeded_total += feed.seed_history_from_klines(asset, result)
            except Exception:
                continue
        if seeded_total > 0:
            logger.info("Oracle-history backfill seeded %d kline points across %d assets", seeded_total, len(assets))

    async def _attach_polymarket_price_history(self, payload: list[dict[str, Any]]) -> None:
        """Attach scanner-managed Polymarket up/down price history to crypto rows.

        Reuses the shared sparkline backfill infrastructure
        (services/scanner.py::attach_price_history_to_markets) so each
        crypto card gets the same multi-hour up/down evolution that
        Opportunities, News, Weather, and Trader panels already render.

        Non-blocking: hydrates from the persisted ``scanner_market_history``
        table on the hot path and queues async Polymarket backfill for any
        market whose history isn't cached yet.
        """
        if not payload:
            return
        try:
            from services.scanner import scanner as market_scanner
        except Exception as exc:
            logger.debug("Crypto price-history attach skipped (scanner unavailable)", exc_info=exc)
            return
        try:
            # 2026-05-08: even with timeout_seconds=0.0 and block_for_backfill
            # =False (so no network round-trip on the hot path), the
            # in-memory hydration over every crypto row still showed
            # multi-second elapsed times during the 2026-05-07 soak. Wrap
            # in a hard wall-clock bound so a pathological row can't
            # stall the refresh cycle — if we overrun, the rows just
            # miss this tick's sparkline and get backfilled on the next.
            await asyncio.wait_for(
                market_scanner.attach_price_history_to_markets(
                    payload,
                    timeout_seconds=0.0,
                    block_for_backfill=False,
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            logger.info(
                "Crypto price-history attach exceeded 3s budget; skipping this tick",
                rows=len(payload),
            )
        except Exception as exc:
            logger.debug("Crypto price-history attach failed", exc_info=exc)

    async def _publish_crypto_snapshot(
        self,
        payload: list[dict[str, Any]],
        *,
        trigger: str,
    ) -> None:
        try:
            await event_bus.publish("crypto_markets_update", {"markets": [dict(row) for row in payload], "trigger": str(trigger)})
        except Exception:
            pass
        self._last_error = None
        await self._queue_crypto_worker_snapshot_persist()

    async def _queue_opportunity_dispatch(
        self,
        payload: list[dict[str, Any]],
        *,
        trigger: str,
        full_source_sweep: bool,
    ) -> None:
        # Deepcopy the (potentially large) crypto markets payload OFF the
        # event loop. The 2026-05-07 soak observed dispatch_seconds=7.67s
        # in the refresh-timing log — almost entirely this deepcopy. The
        # lock-protected slot is single-writer (only _refresh_crypto_markets
        # produces) so doing the copy in a thread doesn't introduce a race.
        copied_payload = await asyncio.to_thread(copy.deepcopy, payload)
        async with self._pending_opportunity_lock:
            self._pending_opportunity_payload = copied_payload
            self._pending_opportunity_trigger = str(trigger)
            self._pending_opportunity_full_source_sweep = (
                self._pending_opportunity_full_source_sweep or bool(full_source_sweep)
            )
        if self._opportunity_dispatch_task is None or self._opportunity_dispatch_task.done():
            self._opportunity_dispatch_task = asyncio.create_task(
                self._run_opportunity_dispatch_loop(),
                name="market-runtime-opportunity-dispatch",
            )

    async def _run_opportunity_dispatch_loop(self) -> None:
        while True:
            async with self._pending_opportunity_lock:
                payload = self._pending_opportunity_payload
                trigger = self._pending_opportunity_trigger
                full_source_sweep = self._pending_opportunity_full_source_sweep
                self._pending_opportunity_payload = None
                self._pending_opportunity_trigger = None
                self._pending_opportunity_full_source_sweep = False
            if payload is None or trigger is None:
                return
            try:
                # Same off-loop deepcopy rationale — the event payload is
                # passed to every event_dispatcher handler and we don't
                # want one slow handler to mutate the source list.
                copied_for_event = await asyncio.to_thread(copy.deepcopy, payload)
                event = build_crypto_update_inputs(
                    markets=copied_for_event,
                    trigger=trigger,
                    source="market_runtime",
                    timestamp=utcnow(),
                )
                # Tee to the recorded-event bus so the same CRYPTO_UPDATE
                # event is captured for backtest replay.  Before this
                # tap, all 5 crypto strategies (btc_eth_*, crypto_*)
                # were un-backtestable — the events that drive their
                # detect() existed only in-memory, fire-and-forget.
                # Bus subscribers get the live fan-out; the parquet
                # writer captures one envelope per dispatch so a
                # historical window can be replayed later.
                #
                # Errors here are deliberately swallowed: the recorded-
                # event bus is the secondary delivery path and must not
                # break the primary _dispatch_with_per_trader_fanout() call.
                try:
                    await _publish_crypto_update_to_bus(event, copied_for_event, trigger)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "recorded_event_bus tee failed for crypto_update",
                        exc_info=True,
                    )
                handler_count = len(event_dispatcher._handlers.get(EventType.CRYPTO_UPDATE, []))
                opportunities = await _dispatch_with_per_trader_fanout(event)
                signals_published = await get_intent_runtime().publish_opportunities(
                    opportunities,
                    source="crypto",
                    sweep_missing=bool(full_source_sweep),
                    refresh_prices=False,
                )
                self._dispatch_count += 1
                self._dispatch_last_at = utcnow().isoformat().replace("+00:00", "Z")
                self._dispatch_last_trigger = str(trigger)
                self._dispatch_last_handlers = handler_count
                self._dispatch_last_opportunities = len(opportunities)
                self._dispatch_last_signals_published = int(signals_published or 0)
                self._dispatch_last_error = None
                try:
                    self._dispatch_filter_diagnostics = _build_crypto_filter_diagnostics(
                        _loaded_crypto_strategy_instances(),
                        opportunities,
                    )
                except Exception:
                    self._dispatch_filter_diagnostics = {}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._dispatch_last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Crypto opportunity dispatch failed", trigger=str(trigger), exc_info=exc)

    def _on_ws_price_update(
        self,
        token_id: str,
        mid: float,
        bid: float,
        ask: float,
        exchange_ts: float,
        ingest_ts: float,
        sequence: int,
    ) -> None:
        if not self._started:
            return
        normalized = _normalize_market_id(token_id)
        if not normalized or normalized not in self._crypto_token_to_market_ids:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        # Sync inline append + drain-task ensure.  Production saw
        # 30+ ``_queue_reactive_*`` tasks parked on the lock as
        # chainlink/binance feeds delivered updates in bursts —
        # creating one task per WS tick is pure overhead.  Set.add
        # is atomic in CPython single-threaded asyncio, no lock
        # needed for the writer side; the drain side still locks
        # to snapshot+clear in a single critical section.
        self._pending_tokens.add(normalized)
        if self._reactive_task is None or self._reactive_task.done():
            self._reactive_task = loop.create_task(
                self._drain_reactive_updates(), name="market-runtime-reactive"
            )

    def _on_reference_update(self, asset: str) -> None:
        if not self._started:
            return
        normalized = str(asset or "").strip().upper()
        if not normalized or normalized not in self._crypto_asset_to_market_ids:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        # Same coalescing pattern as ``_on_token_quote_update``: skip
        # the per-tick task creation, just add to the set and ensure
        # the single debounced drain task is scheduled.  Drops peak
        # task count by ~30 per crypto-feed burst.
        self._pending_assets.add(normalized)
        if self._reactive_task is None or self._reactive_task.done():
            self._reactive_task = loop.create_task(
                self._drain_reactive_updates(), name="market-runtime-reactive"
            )

    async def _drain_reactive_updates(self) -> None:
        # Skip the explicit sleep when the debounce is zero — even a
        # ``await asyncio.sleep(0)`` round-trip through the event loop
        # is non-zero cost and unnecessary when the operator has
        # disabled coalescing.  The ``_pending_tokens`` set still
        # coalesces ticks that arrive between this task's scheduling
        # and the lock acquisition below, which is the only coalescing
        # the sub-second path actually needs.
        if _WS_REACTIVE_DEBOUNCE_SECONDS > 0.0:
            await asyncio.sleep(_WS_REACTIVE_DEBOUNCE_SECONDS)
        # Crypto-lane toggle: when the operator has disabled or paused
        # the lane, drop accumulated tick coalescing buckets and skip
        # the per-market payload rebuild entirely. The Binance WS feeds
        # keep flowing into ``_pending_tokens`` / ``_pending_assets``;
        # we just discard them.
        control = await self._read_crypto_control_cached()
        if not _crypto_lane_is_active(control):
            async with self._pending_reactive_lock:
                self._pending_tokens.clear()
                self._pending_assets.clear()
            return
        async with self._pending_reactive_lock:
            tokens = set(self._pending_tokens)
            self._pending_tokens.clear()
            assets = set(self._pending_assets)
            self._pending_assets.clear()
        if not tokens and not assets:
            return
        market_ids = {
            market_id
            for token_id in tokens
            for market_id in self._crypto_token_to_market_ids.get(token_id, set())
            if market_id
        }
        market_ids.update(
            market_id
            for asset in assets
            for market_id in self._crypto_asset_to_market_ids.get(asset, set())
            if market_id
        )
        if not market_ids:
            return
        selected_rows = [
            row
            for row in self._crypto_markets
            if _normalize_market_id(row.get("id") or row.get("slug")) in market_ids
        ]
        if not selected_rows:
            return
        refreshed_rows = self._rebuild_crypto_rows_from_cache(selected_rows)
        await self._queue_ml_pipeline_refresh(refreshed_rows, allow_record=False)
        merged_by_id = {
            _normalize_market_id(row.get("id") or row.get("slug")): row
            for row in self._crypto_markets
        }
        for row in refreshed_rows:
            merged_by_id[_normalize_market_id(row.get("id") or row.get("slug"))] = row
        ordered_ids = [_normalize_market_id(row.get("id") or row.get("slug")) for row in self._crypto_markets]
        self._crypto_markets = [merged_by_id[row_id] for row_id in ordered_ids if row_id in merged_by_id]
        self._crypto_markets_by_lookup = {}
        self._crypto_token_to_market_ids = {}
        self._crypto_asset_to_market_ids = {}
        for row in self._crypto_markets:
            self._index_crypto_market_row(row)
        self._last_crypto_refresh_at = utcnow().isoformat().replace("+00:00", "Z")
        trigger = "reference_ws" if assets and not tokens else "crypto_ws" if tokens and not assets else "crypto_reference_ws"
        self._last_crypto_trigger = trigger
        # Publish a lightweight payload for reactive WS pushes: strip the
        # large history arrays (oracle_history ~80pts, price_history up to
        # ~720pts) to cut payload size on sub-second ticks.  These don't
        # meaningfully change between adjacent reactive ticks; the full
        # arrays ride on the next periodic scan payload, and the frontend
        # merger preserves them across reactive updates.
        lightweight_rows = [
            {
                k: v
                for k, v in row.items()
                if k not in ("oracle_history", "history_tail", "price_history")
            }
            for row in refreshed_rows
        ]
        await self._publish_crypto_snapshot(lightweight_rows, trigger=trigger)
        await self._queue_opportunity_dispatch(
            refreshed_rows,
            trigger=trigger,
            full_source_sweep=False,
        )

    def _rebuild_crypto_rows_from_cache(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        feed_manager = self._feed_manager
        reference_runtime = self._reference_runtime
        rebuilt: list[dict[str, Any]] = []
        now_iso = utcnow().isoformat().replace("+00:00", "Z")
        now_dt = utcnow()
        strict_age = float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 1.0) or 1.0)
        motion_summary_by_asset: dict[str, dict[str, Any]] = {}
        for existing in rows:
            row = dict(existing)
            token_ids = [str(token_id or "").strip() for token_id in (row.get("clob_token_ids") or []) if str(token_id or "").strip()]
            if feed_manager is not None and getattr(feed_manager, "_started", False):
                if len(token_ids) > 0 and feed_manager.cache.is_fresh(token_ids[0], max_age_seconds=strict_age):
                    row["up_price"] = feed_manager.cache.get_mid_price(token_ids[0])
                    row["up_price_source"] = "redis_strict"
                    try:
                        observed_at_epoch = feed_manager.cache.get_observed_at_epoch(token_ids[0])
                    except Exception:
                        observed_at_epoch = None
                    if observed_at_epoch is not None:
                        row["up_price_updated_at"] = datetime.fromtimestamp(
                            float(observed_at_epoch),
                            tz=timezone.utc,
                        ).isoformat().replace("+00:00", "Z")
                if len(token_ids) > 1 and feed_manager.cache.is_fresh(token_ids[1], max_age_seconds=strict_age):
                    row["down_price"] = feed_manager.cache.get_mid_price(token_ids[1])
                    row["down_price_source"] = "redis_strict"
                    try:
                        observed_at_epoch = feed_manager.cache.get_observed_at_epoch(token_ids[1])
                    except Exception:
                        observed_at_epoch = None
                    if observed_at_epoch is not None:
                        row["down_price_updated_at"] = datetime.fromtimestamp(
                            float(observed_at_epoch),
                            tz=timezone.utc,
                        ).isoformat().replace("+00:00", "Z")
                if token_ids:
                    row["history_tail"] = feed_manager.cache.get_price_history(token_ids[0], max_snapshots=20)
            asset = str(row.get("asset") or "").strip().upper()
            oracle = reference_runtime.get_oracle_price(asset) if asset else None
            row["oracle_price"] = oracle.get("price") if oracle else row.get("oracle_price")
            row["oracle_source"] = oracle.get("source") if oracle else row.get("oracle_source")
            row["oracle_updated_at_ms"] = oracle.get("updated_at_ms") if oracle else row.get("oracle_updated_at_ms")
            row["oracle_age_seconds"] = oracle.get("age_seconds") if oracle else row.get("oracle_age_seconds")
            row["oracle_prices_by_source"] = reference_runtime.get_oracle_prices_by_source(asset) if asset else row.get("oracle_prices_by_source")
            row["oracle_history"] = (
                reference_runtime.get_oracle_history(
                    asset,
                    points=_CRYPTO_ORACLE_HISTORY_POINTS,
                    max_age_seconds=_oracle_history_max_age_seconds(row.get("timeframe")),
                )
                if asset
                else row.get("oracle_history")
            )
            if asset:
                motion_summary = motion_summary_by_asset.get(asset)
                if motion_summary is None:
                    motion_summary = reference_runtime.get_oracle_motion_summary(asset)
                    motion_summary_by_asset[asset] = motion_summary
                _enrich_crypto_motion_fields(row, motion_summary)
            start_time = _parse_iso_utc(row.get("start_time"))
            end_time = _parse_iso_utc(row.get("end_time"))
            if start_time is not None and end_time is not None:
                row["is_live"] = start_time <= now_dt < end_time
                row["is_current"] = bool(row["is_live"])
            elif end_time is not None:
                row["is_live"] = now_dt < end_time
            if end_time is not None:
                row["seconds_left"] = max(0, int(round((end_time - now_dt).total_seconds())))
            else:
                row["seconds_left"] = None
            up_price = _to_float(row.get("up_price"))
            down_price = _to_float(row.get("down_price"))
            row["combined"] = (up_price + down_price) if up_price is not None and down_price is not None else None
            row["price_updated_at"] = now_iso
            # Stamp shared per-market derived fields once here so the
            # 3 BTC/ETH crypto strategies can READ them instead of each
            # recomputing extract_oracle_status / regime / age on the
            # event-dispatch hot path.  See enrich_crypto_market_row.
            enrich_crypto_market_row(row, now_ms=int(now_dt.timestamp() * 1000))
            rebuilt.append(row)
        return rebuilt


_market_runtime: MarketRuntime | None = None


def get_market_runtime() -> MarketRuntime:
    global _market_runtime
    if _market_runtime is None:
        _market_runtime = MarketRuntime()
    return _market_runtime


# ── Recorded-event bus tap (Batch C) ────────────────────────────────
#
# Single source of truth for the crypto.update topic on the bus.
# Registered at MarketRuntime.start(); written by the dispatch loop
# right next to the _dispatch_with_per_trader_fanout call so live
# behavior is unchanged but backtest gets a faithful replay stream.


_CRYPTO_UPDATE_TOPIC = "crypto.update.dispatch"


async def _ensure_crypto_update_topic_registered() -> None:
    """Idempotent topic registration for the CRYPTO_UPDATE tap.

    Stored as a parquet topic under the first configured parquet root
    so the operator's existing storage settings control where it lives.
    Without this registration the first publish fails-closed (by
    design — see ``catalog.require_topic``); with it the bus silently
    starts archiving."""
    from services.recorded_event_bus.catalog import register_topic
    from services.external_data.parquet_schema import parquet_root

    root = parquet_root()
    storage_uri = str(root / "recorded_event_bus" / _CRYPTO_UPDATE_TOPIC)
    await register_topic(
        slug=_CRYPTO_UPDATE_TOPIC,
        title="Crypto opportunity-dispatch events (live + replay)",
        description=(
            "Every CRYPTO_UPDATE event the market_runtime fires through "
            "the in-memory event_dispatcher is teed here so the same "
            "stream is replayable.  Closes the un-backtestable gap for "
            "btc_eth_*/crypto_* strategies — they subscribe to this "
            "topic in backtest and see the same envelope they would "
            "have seen live."
        ),
        storage_kind="parquet",
        storage_uri=storage_uri,
        # This topic is HIGH VOLUME — ~3 GB/day at peak crypto activity.
        # Bound it hard so it can never fill the disk again (it grew to
        # ~50 GB with only a 30-day age cap and no size cap, which crashed
        # the host).  max_bytes is the real protection: the pruner trims
        # oldest partition files once the topic exceeds this, independent
        # of age.  Operators can raise either via Data Lab → Topics; the
        # global cap (app_settings.recorded_event_bus_global_max_bytes) is
        # the cross-topic backstop.
        retention_days=7,
        max_bytes=8 * 1024 * 1024 * 1024,  # 8 GB (~2.5 days of replay)
        publishers=["market_runtime"],
        subscribers=[
            "btc_eth_convergence",
            "btc_eth_directional_edge",
            "btc_eth_maker_quote",
            "crypto_5m_midcycle",
            "crypto_entropy_maker",
        ],
    )


async def _publish_crypto_update_to_bus(
    event: "DataEvent",
    copied_for_event: list[dict[str, Any]],
    trigger: Any,
) -> None:
    """Tee a single CRYPTO_UPDATE event into the recorded-event bus.

    The envelope's ``entity_id`` is the dispatch trigger string so
    operators can filter "what fired this batch" without parsing the
    payload.  The ``observed_at_us`` matches the live event's
    timestamp so live and replay see identical truth-times.  The full
    markets payload + trigger live in the bus payload; consumers
    re-shape it exactly as the live ``event_dispatcher`` handlers do.
    """
    # Global recording master switch + per-feature capture toggle — when
    # EITHER is OFF, skip the bus tee so the crypto.update.dispatch topic
    # stops growing.  ``capture_crypto_dispatch`` lets an operator pause
    # just this (high-volume) crypto tee from Data Lab while leaving book /
    # trade / catalog recording running, exactly like the catalog tee
    # gates on ``capture_catalog``.  Both reads are best-effort: a config
    # blip must never break live crypto dispatch, so a read failure
    # fails-OPEN (capture continues) and is swallowed.
    try:
        from services.recording_control import get_recorder_config, is_recording_enabled

        if not await is_recording_enabled():
            return
        cfg = await get_recorder_config()
        if not cfg.get("capture_crypto_dispatch", True):
            return
    except Exception:  # pragma: no cover — never let the switch break dispatch
        pass

    from services.recorded_event_bus import RecordedEvent
    from services.recorded_event_bus import bus as _bus
    # Lazy import the storage attach (otherwise pyarrow loads on every
    # market_runtime cold start regardless of whether the topic is used).
    import services.recorded_event_bus.storage  # noqa: F401

    observed_at_us = int(event.timestamp.timestamp() * 1_000_000)
    envelope = RecordedEvent(
        topic=_CRYPTO_UPDATE_TOPIC,
        entity_id=str(trigger) or "unknown",
        observed_at_us=observed_at_us,
        payload={
            "markets": copied_for_event,
            "trigger": str(trigger),
            "event_source": event.source,
        },
        source="market_runtime",
    )
    await _bus.publish(envelope)
