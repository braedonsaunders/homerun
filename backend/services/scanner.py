from __future__ import annotations

import asyncio
import copy
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Optional, Callable, List, Set

from config import settings
from interfaces import MarketDataProvider
from models import Opportunity, OpportunityFilter
from models.opportunity import AIAnalysis, MispricingType
from models.database import AsyncSessionLocal, ScannerSettings, OpportunityJudgment, ScannerSnapshot
from services.strategy_loader import strategy_loader
from services.opportunity_strategy_catalog import ensure_system_opportunity_strategies_seeded
from services.strategy_sdk import StrategySDK
from services.providers import market_data_provider
from services.pause_state import global_pause_state
from utils.converters import to_iso
from services.market_prioritizer import market_prioritizer, MarketTier
from services.ws_feeds import get_feed_manager
from services.redis_price_cache import redis_price_cache
from services.quality_filter import quality_filter
from services.data_events import DataEvent, EventType
from services.event_dispatcher import event_dispatcher
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

# Persistent single-thread executor for news prefetch embedding work.
# PyTorch/FAISS use thread-local state; a dedicated thread avoids segfaults.
_NEWS_PREFETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="news_prefetch",
)


def _make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware (UTC). Returns None for None input."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ArbitrageScanner:
    """Main scanner that orchestrates arbitrage detection"""

    def __init__(self, data_provider: Optional[MarketDataProvider] = None):
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        self.market_data = data_provider or market_data_provider

        self._running = False
        self._enabled = True
        self._interval_seconds = settings.SCAN_INTERVAL_SECONDS
        self._last_scan: Optional[datetime] = None
        self._last_full_scan: Optional[datetime] = None
        self._last_fast_scan: Optional[datetime] = None
        self._last_catalog_refresh: Optional[datetime] = None
        self._last_full_snapshot_strategy_scan: Optional[datetime] = None
        self._last_full_snapshot_strategy_duration_seconds: Optional[float] = None
        self._last_full_snapshot_strategy_error: Optional[str] = None
        self._last_full_snapshot_strategy_market_count: int = 0
        self._last_full_snapshot_strategy_opportunity_count: int = 0
        self._last_full_snapshot_strategy_count: int = 0
        self._full_snapshot_strategy_running: bool = False
        self._last_fast_scan_duration_seconds: Optional[float] = None
        self._fast_last_started_at: Optional[datetime] = None
        self._fast_last_completed_at: Optional[datetime] = None
        self._heavy_last_started_at: Optional[datetime] = None
        self._heavy_last_completed_at: Optional[datetime] = None
        self._fast_inflight: bool = False
        self._heavy_inflight: bool = False
        self._fast_lane_error: Optional[str] = None
        self._heavy_lane_error: Optional[str] = None
        self._fast_watchdog_timeout_count: int = 0
        self._heavy_watchdog_timeout_count: int = 0
        self._full_snapshot_cursor_index: int = 0
        self._last_full_snapshot_chunk_market_count: int = 0
        self._full_snapshot_cycle_total_markets: int = 0
        self._full_snapshot_cycle_processed_markets: int = 0
        self._full_snapshot_cycle_started_at: Optional[datetime] = None
        self._full_snapshot_cycle_completed_at: Optional[datetime] = None
        self._opportunities: list[Opportunity] = []
        self._scan_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []
        self._activity_callbacks: List[Callable] = []
        self._scan_task: Optional[asyncio.Task] = None
        self._scan_lock: asyncio.Lock = asyncio.Lock()
        self._full_snapshot_lane_lock: asyncio.Lock = asyncio.Lock()

        # Live scanning activity line (streamed to frontend via WebSocket)
        self._current_activity: str = "Idle"

        # Track the running AI scoring task so we can cancel it on pause
        self._ai_scoring_task: Optional[asyncio.Task] = None

        # Auto AI scoring: when False (default), the scanner does NOT
        # automatically score all opportunities with LLM after each scan.
        # Manual per-opportunity analysis (via the Analyze button) still works.
        self._auto_ai_scoring: bool = False

        # Tiered scanning: track scan cycles for fast/full alternation
        self._fast_scan_cycle: int = 0
        self._prioritizer = market_prioritizer

        # Cached full market/event data for use between full scans
        self._cached_events: list = []
        self._cached_markets: list = []
        self._cached_market_by_id: dict[str, object] = {}
        self._cached_prices: dict = {}
        self._token_to_market_ids: dict[str, set[str]] = {}
        self._market_to_event_id: dict[str, str] = {}
        self._event_to_market_ids: dict[str, set[str]] = {}
        self._market_id_to_condition_id: dict[str, str] = {}
        self._condition_id_to_market_id: dict[str, str] = {}

        # Real market price history for opportunity card sparklines.
        # market_id -> [{t: epoch_ms, idx_0: float, idx_1: float, ...}, ...]
        spark_window_hours = max(1, int(settings.SCANNER_SPARKLINE_WINDOW_HOURS))
        spark_sample_seconds = max(10, int(settings.SCANNER_SPARKLINE_SAMPLE_SECONDS))
        spark_max_points = max(30, int(settings.SCANNER_SPARKLINE_MAX_POINTS))
        spark_export_points = max(20, int(settings.SCANNER_SPARKLINE_EXPORT_POINTS))

        self._market_price_history: dict[str, list[dict[str, object]]] = {}
        self._market_history_retention_seconds: int = spark_window_hours * 3600
        retention_points = int(self._market_history_retention_seconds / spark_sample_seconds) + 2
        self._market_history_max_points: int = max(spark_max_points, retention_points)
        self._market_history_export_points: int = min(self._market_history_max_points, spark_export_points)
        # Keep one sample per interval even if price is unchanged, so cards
        # show multi-hour trend shape rather than a single refreshed point.
        self._market_history_sample_interval_ms: int = spark_sample_seconds * 1000
        self._market_token_ids: dict[str, tuple[str, str]] = {}
        self._market_outcome_token_ids: dict[str, tuple[str, ...]] = {}
        self._market_history_backfill_done: set[str] = set()
        self._market_history_backfill_attempt_ms: dict[str, int] = {}
        self._market_history_backfill_retry_ms: int = 5 * 60 * 1000
        self._market_history_backfill_concurrency: int = 8
        self._market_history_backfill_max_markets: int = 120
        self._market_history_backfill_task: Optional[asyncio.Task] = None
        self._market_history_backfill_queue: list[Opportunity] = []

        # Reactive scanning: event set by WS price changes to trigger immediate scan
        self._reactive_trigger: Optional[asyncio.Event] = None
        self._reactive_scan_registered = False
        self._reactive_tokens_lock: Optional[asyncio.Lock] = None
        self._pending_reactive_tokens: dict[str, float] = {}
        self._reactive_backpressure_dropped_tokens: int = 0
        self._reactive_backpressure_dropped_markets: int = 0
        self._last_reactive_batch_tokens: int = 0
        self._last_reactive_batch_markets: int = 0

        # Quality filter audit trail from the last scan cycle
        self._quality_reports: dict = {}

        # Test/runtime override hook for strategy lists.
        self._strategy_overrides: Optional[list] = None
        self._plugins_loaded: bool = False

    @property
    def strategies(self) -> list:
        return self._get_all_strategies()

    @strategies.setter
    def strategies(self, value: list) -> None:
        self._strategy_overrides = list(value or [])

    def set_auto_ai_scoring(self, enabled: bool):
        """Enable or disable automatic AI scoring of all opportunities after each scan."""
        self._auto_ai_scoring = enabled
        print(f"Auto AI scoring {'enabled' if enabled else 'disabled'}")

    @property
    def auto_ai_scoring(self) -> bool:
        return self._auto_ai_scoring

    @property
    def quality_reports(self) -> dict:
        """Quality filter audit trails from the last scan cycle."""
        return self._quality_reports

    def add_callback(self, callback: Callable):
        """Add callback to be notified of new opportunities"""
        self._scan_callbacks.append(callback)

    def add_status_callback(self, callback: Callable):
        """Add callback to be notified of scanner status changes"""
        self._status_callbacks.append(callback)

    def add_activity_callback(self, callback: Callable):
        """Add callback to be notified of scanning activity updates"""
        self._activity_callbacks.append(callback)

    async def _set_activity(self, activity: str):
        """Update the current scanning activity and broadcast to clients."""
        self._current_activity = activity
        for cb in self._activity_callbacks:
            try:
                await cb(activity)
            except Exception:
                pass

    @staticmethod
    def _price_value(raw: Optional[dict]) -> Optional[float]:
        """Extract a usable midpoint-like value from a price payload."""
        if not isinstance(raw, dict):
            return None
        # Failed price fetches are represented as {"error": "..."}; these
        # must not be treated as valid 0.0 prices.
        if raw.get("error") is not None:
            return None
        for key in ("mid", "yes", "price"):
            val = raw.get(key)
            if isinstance(val, (float, int)) and 0 <= val <= 1:
                return float(val)
        bid = raw.get("bid")
        ask = raw.get("ask")
        if isinstance(bid, (float, int)) and isinstance(ask, (float, int)):
            if 0 <= bid <= 1 and 0 <= ask <= 1:
                return float((bid + ask) / 2.0)
            if 0 <= bid <= 1:
                return float(bid)
            if 0 <= ask <= 1:
                return float(ask)
        return None

    @staticmethod
    def _coerce_history_price(raw: object) -> Optional[float]:
        try:
            price = float(raw)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= price <= 1.01):
            return None
        return float(round(min(1.0, max(0.0, price)), 6))

    @classmethod
    def _build_market_history_point(
        cls,
        ts_ms: int,
        outcome_prices: list[Optional[float]],
    ) -> Optional[dict[str, object]]:
        if ts_ms <= 0:
            return None

        normalized = list(outcome_prices)
        while normalized and normalized[-1] is None:
            normalized.pop()
        if not normalized:
            return None

        if len(normalized) == 2:
            if normalized[0] is None and normalized[1] is not None:
                normalized[0] = cls._coerce_history_price(1.0 - float(normalized[1]))
            if normalized[1] is None and normalized[0] is not None:
                normalized[1] = cls._coerce_history_price(1.0 - float(normalized[0]))

        finite_count = sum(1 for value in normalized if value is not None)
        if finite_count < 2:
            return None

        point: dict[str, object] = {"t": float(ts_ms)}
        complete_vector = True
        for idx, value in enumerate(normalized):
            if value is None:
                complete_vector = False
                continue
            point[f"idx_{idx}"] = value

        idx0 = point.get("idx_0")
        idx1 = point.get("idx_1")
        if isinstance(idx0, (int, float)):
            point["yes"] = float(idx0)
        if isinstance(idx1, (int, float)):
            point["no"] = float(idx1)

        if complete_vector:
            point["outcome_prices"] = [point[f"idx_{i}"] for i in range(len(normalized))]
        return point

    @classmethod
    def _normalize_history_point(cls, raw: object, cutoff_ms: int) -> Optional[dict[str, object]]:
        if not isinstance(raw, dict):
            return None

        ts_raw = (
            raw.get("t")
            if raw.get("t") is not None
            else (
                raw.get("ts")
                if raw.get("ts") is not None
                else (raw.get("time") if raw.get("time") is not None else raw.get("timestamp"))
            )
        )
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            return None
        if ts < 10_000_000_000:
            ts *= 1000.0
        ts_ms = int(ts)
        if ts_ms < cutoff_ms:
            return None

        by_index: dict[int, Optional[float]] = {}

        vector_raw = raw.get("outcome_prices")
        if vector_raw is None:
            vector_raw = raw.get("outcomePrices")
        if vector_raw is None:
            vector_raw = raw.get("prices")
        if isinstance(vector_raw, list):
            for idx, value in enumerate(vector_raw):
                by_index[idx] = cls._coerce_history_price(value)

        for key, value in raw.items():
            key_text = str(key)
            if not key_text.startswith("idx_"):
                continue
            idx_text = key_text[4:]
            if not idx_text.isdigit():
                continue
            by_index[int(idx_text)] = cls._coerce_history_price(value)

        yes = cls._coerce_history_price(raw.get("yes"))
        if yes is None:
            yes = cls._coerce_history_price(raw.get("y"))
        if yes is None:
            yes = cls._coerce_history_price(raw.get("p"))
        no = cls._coerce_history_price(raw.get("no"))
        if no is None:
            no = cls._coerce_history_price(raw.get("n"))
        if 0 not in by_index and yes is not None:
            by_index[0] = yes
        if 1 not in by_index and no is not None:
            by_index[1] = no

        if not by_index:
            return None

        max_index = max(by_index.keys())
        values = [by_index.get(i) for i in range(max_index + 1)]
        return cls._build_market_history_point(ts_ms, values)

    @staticmethod
    def _history_point_signature(point: dict[str, object]) -> tuple[float, ...]:
        vector = point.get("outcome_prices")
        if isinstance(vector, list) and vector:
            out: list[float] = []
            for value in vector:
                try:
                    out.append(float(round(float(value), 6)))
                except (TypeError, ValueError):
                    out = []
                    break
            if out:
                return tuple(out)

        indexed: list[tuple[int, float]] = []
        for key, value in point.items():
            key_text = str(key)
            if not key_text.startswith("idx_"):
                continue
            idx_text = key_text[4:]
            if not idx_text.isdigit():
                continue
            try:
                indexed.append((int(idx_text), float(round(float(value), 6))))
            except (TypeError, ValueError):
                continue
        if indexed:
            indexed.sort(key=lambda row: row[0])
            return tuple(price for _, price in indexed)

        try:
            yes = float(round(float(point.get("yes")), 6))
            no = float(round(float(point.get("no")), 6))
            return (yes, no)
        except (TypeError, ValueError):
            return ()

    def _extract_market_outcome_prices(self, market, prices: dict) -> list[Optional[float]]:
        token_ids = [str(token_id or "").strip() for token_id in (getattr(market, "clob_token_ids", None) or [])]
        token_ids = [token_id for token_id in token_ids if token_id]
        existing_raw = list(getattr(market, "outcome_prices", None) or [])

        size = max(len(token_ids), len(existing_raw), 2)
        values: list[Optional[float]] = [None] * size

        for idx, raw_price in enumerate(existing_raw):
            if idx >= len(values):
                break
            parsed = self._coerce_history_price(raw_price)
            if parsed is not None:
                values[idx] = parsed

        for idx, token_id in enumerate(token_ids):
            if idx >= len(values):
                break
            parsed = self._price_value(prices.get(token_id))
            if parsed is None:
                continue
            values[idx] = self._coerce_history_price(parsed)

        return values

    def _extract_market_yes_no_prices(self, market, prices: dict) -> tuple[Optional[float], Optional[float]]:
        values = self._extract_market_outcome_prices(market, prices)
        point = self._build_market_history_point(1, values)
        if point is None:
            return None, None
        yes = point.get("yes")
        no = point.get("no")
        if not isinstance(yes, (int, float)) or not isinstance(no, (int, float)):
            return None, None
        return float(yes), float(no)

    def _update_market_price_history(self, markets: list, prices: dict, ts: datetime) -> None:
        """Append current real market prices to bounded in-memory history."""
        now_ms = int(ts.timestamp() * 1000)
        cutoff_ms = now_ms - int(self._market_history_retention_seconds * 1000)

        for market in markets:
            market_ids = self._market_id_candidates_from_market(market)
            if not market_ids:
                continue
            values = self._extract_market_outcome_prices(market, prices)
            point = self._build_market_history_point(now_ms, values)
            if point is None:
                continue
            point_sig = self._history_point_signature(point)
            if not point_sig:
                continue
            for market_id in market_ids:
                history = self._market_price_history.setdefault(market_id, [])
                if history:
                    last = history[-1]
                    last_sig = self._history_point_signature(last)
                    if last_sig and last_sig == point_sig:
                        last_t = float(last.get("t", 0.0) or 0.0)
                        if (float(point["t"]) - last_t) < self._market_history_sample_interval_ms:
                            last["t"] = float(point["t"])
                            continue
                history.append(point)

                # Trim expired entries in O(n) via bisect-style scan + single slice delete,
                # avoiding the O(n^2) cost of repeated list.pop(0).
                trim_idx = 0
                while trim_idx < len(history) and history[trim_idx].get("t", 0) < cutoff_ms:
                    trim_idx += 1
                if trim_idx:
                    del history[:trim_idx]
                if len(history) > self._market_history_max_points:
                    del history[: len(history) - self._market_history_max_points]

    def _remember_market_tokens(self, markets: list) -> None:
        """Cache token IDs per market for historical backfill calls."""
        for market in markets:
            platform = str(getattr(market, "platform", "polymarket") or "polymarket").lower()
            if platform != "polymarket":
                continue
            token_ids = self._coerce_polymarket_token_ids(getattr(market, "clob_token_ids", None))
            if token_ids is None:
                continue
            for market_id in self._market_id_candidates_from_market(market):
                self._market_outcome_token_ids[market_id] = token_ids
                self._market_token_ids[market_id] = (token_ids[0], token_ids[1])

    @staticmethod
    def _coerce_market_token_ids(raw_tokens: object) -> list[str]:
        parsed = raw_tokens
        if isinstance(parsed, str):
            text = parsed.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = [part.strip() for part in text.split(",") if part.strip()]
        if not isinstance(parsed, (list, tuple)):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in parsed:
            token_id = ""
            if isinstance(item, dict):
                token_id = str(
                    item.get("token_id")
                    or item.get("tokenId")
                    or item.get("asset_id")
                    or item.get("assetId")
                    or item.get("id")
                    or ""
                ).strip()
            else:
                token_id = str(item or "").strip()
            if not token_id or token_id in seen:
                continue
            seen.add(token_id)
            out.append(token_id)
        return out

    @classmethod
    def _coerce_polymarket_token_ids(cls, raw: object) -> Optional[tuple[str, ...]]:
        token_ids = [token_id for token_id in cls._coerce_market_token_ids(raw) if len(token_id) > 20]
        if len(token_ids) < 2:
            return None
        return tuple(token_ids)

    @classmethod
    def _coerce_polymarket_token_pair(cls, raw: object) -> Optional[tuple[str, str]]:
        """Parse polymarket YES/NO token IDs from list/tuple/JSON-string payloads."""
        token_ids = cls._coerce_polymarket_token_ids(raw)
        if token_ids is None:
            return None
        return token_ids[0], token_ids[1]

    @staticmethod
    def _market_id_candidates_from_market(market: object) -> list[str]:
        ids: list[str] = []
        for raw in (
            getattr(market, "id", ""),
            getattr(market, "condition_id", ""),
        ):
            market_id = str(raw or "").strip()
            if not market_id or market_id in ids:
                continue
            ids.append(market_id)
        return ids

    @staticmethod
    def _market_id_candidates_from_payload(market: dict) -> list[str]:
        ids: list[str] = []
        for raw in (
            market.get("id"),
            market.get("condition_id"),
            market.get("conditionId"),
        ):
            market_id = str(raw or "").strip()
            if not market_id or market_id in ids:
                continue
            ids.append(market_id)
        return ids

    def _expand_market_id_aliases(self, market_ids: list[str]) -> list[str]:
        expanded: list[str] = []
        seen: set[str] = set()
        for market_id in market_ids:
            mid = str(market_id or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            expanded.append(mid)
            condition_id = self._market_id_to_condition_id.get(mid)
            if condition_id and condition_id not in seen:
                seen.add(condition_id)
                expanded.append(condition_id)
            canonical_market_id = self._condition_id_to_market_id.get(mid)
            if canonical_market_id and canonical_market_id not in seen:
                seen.add(canonical_market_id)
                expanded.append(canonical_market_id)
        return expanded

    def _market_history_lookup_ids(self, market: dict) -> list[str]:
        base_ids = self._market_id_candidates_from_payload(market)
        return self._expand_market_id_aliases(base_ids)

    def _remember_market_tokens_from_opportunities(self, opportunities: list[Opportunity]) -> None:
        """Cache token IDs from opportunity market dicts."""
        for opp in opportunities:
            for market in opp.markets:
                platform = str(market.get("platform", "polymarket") or "polymarket").lower()
                if platform != "polymarket":
                    continue
                token_ids = self._coerce_polymarket_token_ids(market.get("clob_token_ids"))
                if token_ids is None:
                    continue
                for market_id in self._market_id_candidates_from_payload(market):
                    self._market_outcome_token_ids[market_id] = token_ids
                    self._market_token_ids[market_id] = (token_ids[0], token_ids[1])

    def _rebuild_realtime_graph(self, events: list, markets: list) -> None:
        """Build token->market and event<->market routing maps from cached snapshot."""
        market_by_id: dict[str, object] = {}
        token_to_market_ids: dict[str, set[str]] = {}
        market_to_event_id: dict[str, str] = {}
        event_to_market_ids: dict[str, set[str]] = {}
        market_id_to_condition_id: dict[str, str] = {}
        condition_id_to_market_id: dict[str, str] = {}

        for market in markets:
            market_id = str(getattr(market, "id", "") or "")
            if not market_id:
                continue
            condition_id = str(getattr(market, "condition_id", "") or "").strip()
            market_by_id[market_id] = market
            if condition_id and condition_id != market_id:
                market_id_to_condition_id[market_id] = condition_id
                condition_id_to_market_id[condition_id] = market_id
            for token_id in getattr(market, "clob_token_ids", None) or []:
                token = str(token_id or "").strip()
                if not token:
                    continue
                token_to_market_ids.setdefault(token, set()).add(market_id)

        for event in events:
            event_id = str(getattr(event, "id", "") or getattr(event, "slug", "") or "")
            if not event_id:
                continue
            mids: set[str] = set()
            for market in getattr(event, "markets", None) or []:
                market_id = str(getattr(market, "id", "") or "")
                if not market_id:
                    continue
                condition_id = str(getattr(market, "condition_id", "") or "").strip()
                mids.add(market_id)
                market_to_event_id[market_id] = event_id
                if condition_id and condition_id != market_id:
                    market_to_event_id[condition_id] = event_id
                    market_id_to_condition_id[market_id] = condition_id
                    condition_id_to_market_id[condition_id] = market_id
                if market_id not in market_by_id:
                    market_by_id[market_id] = market
                for token_id in getattr(market, "clob_token_ids", None) or []:
                    token = str(token_id or "").strip()
                    if not token:
                        continue
                    token_to_market_ids.setdefault(token, set()).add(market_id)
            if mids:
                event_to_market_ids[event_id] = mids

        self._cached_market_by_id = market_by_id
        self._token_to_market_ids = token_to_market_ids
        self._market_to_event_id = market_to_event_id
        self._event_to_market_ids = event_to_market_ids
        self._market_id_to_condition_id = market_id_to_condition_id
        self._condition_id_to_market_id = condition_id_to_market_id

    @staticmethod
    def _collect_polymarket_tokens(markets: list) -> list[str]:
        """Collect unique polymarket token IDs from markets in stable order."""
        seen: set[str] = set()
        out: list[str] = []
        for market in markets:
            for token_id in getattr(market, "clob_token_ids", None) or []:
                token = str(token_id or "").strip()
                if not token or len(token) <= 20 or token in seen:
                    continue
                seen.add(token)
                out.append(token)
        return out

    @staticmethod
    def _collect_live_token_ids(markets: list) -> list[str]:
        """Collect unique token IDs from markets in stable order (Polymarket + Kalshi)."""
        seen: set[str] = set()
        out: list[str] = []
        for market in markets:
            for token_id in getattr(market, "clob_token_ids", None) or []:
                token = str(token_id or "").strip()
                if not token or token in seen:
                    continue
                seen.add(token)
                out.append(token)
        return out

    async def _snapshot_ws_prices(self, token_ids: list[str]) -> dict[str, dict]:
        """Return fresh live snapshots for token IDs from Redis + in-memory fallback."""
        if not token_ids or not settings.WS_FEED_ENABLED:
            return {}
        clean_token_ids = [str(token_id).strip() for token_id in token_ids if str(token_id).strip()]
        if not clean_token_ids:
            return {}

        prices: dict[str, dict] = {}
        try:
            prices = await redis_price_cache.read_prices(clean_token_ids)
        except Exception as exc:
            print(f"  Redis live price read failed (using in-memory fallback): {exc}")
            prices = {}
        missing = [token_id for token_id in clean_token_ids if token_id not in prices]
        if not missing:
            return prices

        # Local fallback for same-process WS cache if Redis is slower to catch up.
        try:
            feed_mgr = get_feed_manager()
            if not feed_mgr._started:
                return prices
            for token_id in missing:
                if not feed_mgr.is_fresh(token_id):
                    continue
                mid = feed_mgr.cache.get_mid_price(token_id)
                if mid is None:
                    continue
                bba = feed_mgr.cache.get_best_bid_ask(token_id)
                if bba is None:
                    now_ts = float(time.time())
                    prices[token_id] = {
                        "mid": float(mid),
                        "bid": float(mid),
                        "ask": float(mid),
                        "ts": now_ts,
                        "ingest_ts": now_ts,
                        "exchange_ts": now_ts,
                        "sequence": 0,
                        "is_fresh": True,
                    }
                else:
                    bid, ask = bba
                    now_ts = float(time.time())
                    prices[token_id] = {
                        "mid": float(mid),
                        "bid": float(bid),
                        "ask": float(ask),
                        "ts": now_ts,
                        "ingest_ts": now_ts,
                        "exchange_ts": now_ts,
                        "sequence": 0,
                        "is_fresh": True,
                    }
        except Exception:
            pass

        return prices

    def _apply_live_prices_to_markets(self, markets: list, prices: dict[str, dict]) -> int:
        """Mutate market outcome prices from live token prices to keep fingerprints current."""
        updated = 0
        for market in markets:
            token_ids = [str(token_id or "").strip() for token_id in (getattr(market, "clob_token_ids", None) or [])]
            token_ids = [token_id for token_id in token_ids if token_id]
            if len(token_ids) < 2:
                continue

            current_prices = self._extract_market_outcome_prices(market, prices)
            point = self._build_market_history_point(1, current_prices)
            if point is None:
                continue

            incoming_sig = self._history_point_signature(point)
            existing_raw = list(getattr(market, "outcome_prices", None) or [])
            existing_sig: tuple[float, ...] = ()
            if existing_raw:
                existing_point = self._build_market_history_point(1, [self._coerce_history_price(v) for v in existing_raw])
                if existing_point is not None:
                    existing_sig = self._history_point_signature(existing_point)

            if incoming_sig and existing_sig and incoming_sig == existing_sig:
                continue

            incoming_vector = point.get("outcome_prices")
            if not isinstance(incoming_vector, list) or len(incoming_vector) < 2:
                continue
            market.outcome_prices = [float(v) for v in incoming_vector]
            tokens = getattr(market, "tokens", None) or []
            for idx, token in enumerate(tokens):
                if idx >= len(market.outcome_prices):
                    break
                try:
                    token.price = market.outcome_prices[idx]
                except Exception:
                    pass
            updated += 1
        return updated

    @staticmethod
    def _is_market_active(market: object, now: datetime) -> bool:
        end_date = _make_aware(getattr(market, "end_date", None))
        if end_date is not None and end_date <= now:
            return False

        if bool(getattr(market, "closed", False)):
            return False
        if bool(getattr(market, "resolved", False)):
            return False
        if bool(getattr(market, "archived", False)):
            return False
        if getattr(market, "accepting_orders", None) is False:
            return False
        if getattr(market, "active", True) is False:
            return False

        status = str(getattr(market, "status", "") or "").strip().lower()
        if status in {"closed", "resolved", "settled", "finalized", "inactive", "expired"}:
            return False
        return True

    def _prune_active_catalog(self, events: list, markets: list, now: datetime) -> tuple[list, list]:
        deduped_markets: dict[str, object] = {}
        for market in markets:
            market_id = str(getattr(market, "id", "") or "")
            if not market_id:
                continue
            if not self._is_market_active(market, now):
                continue
            if market_id not in deduped_markets:
                deduped_markets[market_id] = market
                continue
            existing = deduped_markets[market_id]
            if float(getattr(market, "volume", 0.0) or 0.0) > float(getattr(existing, "volume", 0.0) or 0.0):
                deduped_markets[market_id] = market

        active_markets = list(deduped_markets.values())
        active_ids = {str(getattr(market, "id", "") or "") for market in active_markets}

        pruned_events: list = []
        for event in events:
            event_markets = []
            for market in list(getattr(event, "markets", None) or []):
                market_id = str(getattr(market, "id", "") or "")
                if market_id and market_id in active_ids:
                    event_markets.append(deduped_markets[market_id])
            event.markets = event_markets
            if event_markets:
                pruned_events.append(event)
        return pruned_events, active_markets

    @staticmethod
    def _market_priority_key(market: object) -> tuple[float, float, float]:
        volume = float(getattr(market, "volume", 0.0) or 0.0)
        liquidity = float(getattr(market, "liquidity", 0.0) or 0.0)
        has_tokens = 1.0 if list(getattr(market, "clob_token_ids", None) or []) else 0.0
        return (volume, liquidity, has_tokens)

    def _is_tail_end_priority_market(self, market: object, now: datetime) -> bool:
        end_date = _make_aware(getattr(market, "end_date", None))
        if end_date is None:
            return False
        days_to_resolution = (end_date - now).total_seconds() / 86400.0
        if days_to_resolution <= 0.0 or days_to_resolution > 2.0:
            return False

        yes, no = self._extract_market_yes_no_prices(market, {})
        if yes is None or no is None:
            return False
        return max(float(yes), float(no)) >= 0.85

    def _tail_end_priority_key(self, market: object, now: datetime) -> tuple[float, float, float, float]:
        end_date = _make_aware(getattr(market, "end_date", None))
        if end_date is None:
            days_to_resolution = 9999.0
        else:
            days_to_resolution = max(0.0, (end_date - now).total_seconds() / 86400.0)
        yes, no = self._extract_market_yes_no_prices(market, {})
        side_probability = max(float(yes or 0.0), float(no or 0.0))
        liquidity = float(getattr(market, "liquidity", 0.0) or 0.0)
        volume = float(getattr(market, "volume", 0.0) or 0.0)
        time_priority = 1.0 / max(days_to_resolution, 1e-6)
        return (time_priority, side_probability, liquidity, volume)

    def _enforce_catalog_caps(self, events: list, markets: list) -> tuple[list, list]:
        if bool(getattr(settings, "SCANNER_FORCE_FULL_UNIVERSE", True)):
            market_cap = 0
            event_cap = 0
        else:
            market_cap = int(getattr(settings, "MAX_MARKETS_TO_SCAN", 0) or 0)
            event_cap = int(getattr(settings, "MAX_EVENTS_TO_SCAN", 0) or 0)

        capped_markets = list(markets)
        if market_cap > 0 and len(capped_markets) > market_cap:
            capped_markets.sort(key=self._market_priority_key, reverse=True)
            capped_markets = capped_markets[:market_cap]

        kept_market_ids = {str(getattr(market, "id", "") or "") for market in capped_markets}
        capped_events: list = []
        for event in events:
            event_markets = [
                market
                for market in list(getattr(event, "markets", None) or [])
                if str(getattr(market, "id", "") or "") in kept_market_ids
            ]
            if not event_markets:
                continue
            event.markets = event_markets
            capped_events.append(event)

        if event_cap > 0 and len(capped_events) > event_cap:

            def _event_priority_key(event_obj: object) -> tuple[int, float, float]:
                mkts = list(getattr(event_obj, "markets", None) or [])
                return (
                    len(mkts),
                    sum(float(getattr(m, "volume", 0.0) or 0.0) for m in mkts),
                    sum(float(getattr(m, "liquidity", 0.0) or 0.0) for m in mkts),
                )

            capped_events.sort(key=_event_priority_key, reverse=True)
            capped_events = capped_events[:event_cap]
            kept_market_ids = {
                str(getattr(market, "id", "") or "")
                for event in capped_events
                for market in list(getattr(event, "markets", None) or [])
            }
            capped_markets = [
                market for market in capped_markets if str(getattr(market, "id", "") or "") in kept_market_ids
            ]

        return capped_events, capped_markets

    def _trim_runtime_market_caches(self, active_market_ids: set[str]) -> None:
        if not active_market_ids:
            self._cached_prices = {}
            self._market_price_history = {}
            self._market_token_ids = {}
            self._market_outcome_token_ids = {}
            self._market_id_to_condition_id = {}
            self._condition_id_to_market_id = {}
            return

        active_with_aliases = set(active_market_ids)
        for market_id in list(active_market_ids):
            condition_id = self._market_id_to_condition_id.get(market_id)
            if condition_id:
                active_with_aliases.add(condition_id)
            canonical_market_id = self._condition_id_to_market_id.get(market_id)
            if canonical_market_id:
                active_with_aliases.add(canonical_market_id)

        self._cached_prices = {
            token_id: payload
            for token_id, payload in self._cached_prices.items()
            if any(mid in active_market_ids for mid in self._token_to_market_ids.get(token_id, set()))
        }
        self._market_price_history = {
            market_id: history
            for market_id, history in self._market_price_history.items()
            if market_id in active_with_aliases
        }
        self._market_token_ids = {
            market_id: token_pair
            for market_id, token_pair in self._market_token_ids.items()
            if market_id in active_with_aliases
        }
        self._market_outcome_token_ids = {
            market_id: token_ids
            for market_id, token_ids in self._market_outcome_token_ids.items()
            if market_id in active_with_aliases
        }
        self._market_id_to_condition_id = {
            market_id: condition_id
            for market_id, condition_id in self._market_id_to_condition_id.items()
            if market_id in active_market_ids and condition_id in active_with_aliases
        }
        self._condition_id_to_market_id = {
            condition_id: market_id
            for condition_id, market_id in self._condition_id_to_market_id.items()
            if market_id in active_market_ids and condition_id in active_with_aliases
        }
        self._market_history_backfill_done &= active_with_aliases
        self._market_history_backfill_attempt_ms = {
            market_id: ts
            for market_id, ts in self._market_history_backfill_attempt_ms.items()
            if market_id in active_with_aliases
        }

    @staticmethod
    def _price_timestamp(raw: Optional[dict]) -> Optional[float]:
        if not isinstance(raw, dict):
            return None
        for key in ("ingest_ts", "exchange_ts", "ts", "timestamp", "updated_at", "updatedAt"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                ts = float(value)
                if ts > 10_000_000_000:
                    ts /= 1000.0
                return ts
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    continue
                try:
                    return float(text)
                except ValueError:
                    try:
                        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        return parsed.timestamp()
                    except Exception:
                        continue
        return None

    @staticmethod
    def _coerce_market_token_pair(raw_tokens: object) -> tuple[str, str] | None:
        token_ids = ArbitrageScanner._coerce_market_token_ids(raw_tokens)
        if len(token_ids) < 2:
            return None
        yes_token = token_ids[0]
        no_token = token_ids[1]
        if not yes_token or not no_token:
            return None
        return yes_token, no_token

    def _resolve_affected_market_ids(self, changed_token_ids: list[str]) -> list[str]:
        """Resolve changed tokens to a bounded market batch, expanded by event peers."""
        if not changed_token_ids:
            return []
        ordered: list[str] = []
        seen: set[str] = set()

        for token_id in changed_token_ids:
            for market_id in sorted(self._token_to_market_ids.get(token_id, set())):
                if market_id in seen:
                    continue
                seen.add(market_id)
                ordered.append(market_id)

        direct_ids = list(ordered)
        for market_id in direct_ids:
            event_id = self._market_to_event_id.get(market_id)
            if not event_id:
                continue
            for peer_id in sorted(self._event_to_market_ids.get(event_id, set())):
                if peer_id in seen:
                    continue
                seen.add(peer_id)
                ordered.append(peer_id)

        cap = max(10, int(settings.REALTIME_SCAN_MAX_BATCH_MARKETS or 800))
        if len(ordered) > cap:
            self._reactive_backpressure_dropped_markets += len(ordered) - cap
            ordered = ordered[:cap]
        return ordered

    @staticmethod
    def _parse_market_price_timestamp(raw_value: object) -> Optional[float]:
        if isinstance(raw_value, (int, float)):
            ts = float(raw_value)
            if ts > 10_000_000_000:
                ts /= 1000.0
            return ts
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                try:
                    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.timestamp()
                except Exception:
                    return None
        return None

    @staticmethod
    def _format_price_timestamp(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None).isoformat() + "Z"

    def _opportunity_price_max_age_seconds(self) -> float:
        configured = float(getattr(settings, "SCANNER_MARKET_PRICE_MAX_AGE_SECONDS", 0) or 0)
        if configured > 0:
            return configured
        ws_ttl = float(getattr(settings, "WS_PRICE_STALE_SECONDS", 30.0) or 30.0)
        return max(30.0, ws_ttl * 2.0)

    def _market_price_is_stale(self, market: dict, now_ts: float, max_age_seconds: float) -> bool:
        token_ids = self._coerce_market_token_ids(market.get("clob_token_ids"))
        if len(token_ids) < 2:
            return False

        explicit_fresh = market.get("is_price_fresh")
        if isinstance(explicit_fresh, bool):
            return not explicit_fresh

        age_raw = market.get("price_age_seconds")
        if isinstance(age_raw, (int, float)):
            return float(age_raw) > max_age_seconds

        ts = self._parse_market_price_timestamp(market.get("price_updated_at"))
        if ts is None:
            return True
        return (now_ts - ts) > max_age_seconds

    async def refresh_opportunity_prices(
        self,
        opportunities: list[Opportunity],
        *,
        now: Optional[datetime] = None,
        drop_stale: bool = False,
    ) -> list[Opportunity]:
        """Overlay fresh token prices onto opportunity markets from Redis/WS cache."""
        if not opportunities:
            return opportunities

        now_dt = now or datetime.now(timezone.utc)
        now_ts = now_dt.timestamp()
        max_age_seconds = self._opportunity_price_max_age_seconds()

        token_ids: list[str] = []
        seen_tokens: set[str] = set()
        for opp in opportunities:
            for market in opp.markets:
                if not isinstance(market, dict):
                    continue
                market_token_ids = self._coerce_market_token_ids(market.get("clob_token_ids"))
                if len(market_token_ids) < 2:
                    continue
                for token_id in market_token_ids:
                    if token_id not in seen_tokens:
                        seen_tokens.add(token_id)
                        token_ids.append(token_id)

        live_prices: dict[str, dict] = {}
        if token_ids:
            chunk_size = 1200
            for i in range(0, len(token_ids), chunk_size):
                chunk = token_ids[i : i + chunk_size]
                chunk_prices = await self._snapshot_ws_prices(chunk)
                if chunk_prices:
                    live_prices.update(chunk_prices)

        filtered: list[Opportunity] = []
        for opp in opportunities:
            opp_seen_dt = _make_aware(
                getattr(opp, "last_detected_at", None)
                or getattr(opp, "last_seen_at", None)
                or getattr(opp, "detected_at", None)
            )
            opp_seen_ts = opp_seen_dt.timestamp() if opp_seen_dt is not None else None
            opp_last_priced_ts: Optional[float] = None
            market_price_lookup: dict[str, tuple[float, float]] = {}
            for market in opp.markets:
                if not isinstance(market, dict):
                    continue
                token_ids = self._coerce_market_token_ids(market.get("clob_token_ids"))
                yes_val: Optional[float] = None
                no_val: Optional[float] = None
                update_ts: Optional[float] = None
                updated_prices: list[Optional[float]] = []
                existing_prices = market.get("outcome_prices")
                if isinstance(existing_prices, list):
                    updated_prices = [self._coerce_history_price(value) for value in existing_prices]
                if len(token_ids) > len(updated_prices):
                    updated_prices.extend([None] * (len(token_ids) - len(updated_prices)))

                ts_candidates: list[float] = []
                for idx, token_id in enumerate(token_ids):
                    raw = live_prices.get(token_id)
                    live_price = self._price_value(raw)
                    if live_price is not None:
                        while idx >= len(updated_prices):
                            updated_prices.append(None)
                        updated_prices[idx] = self._coerce_history_price(live_price)
                    ts_candidate = self._price_timestamp(raw)
                    if ts_candidate is not None:
                        ts_candidates.append(ts_candidate)

                if len(updated_prices) == 2:
                    if updated_prices[0] is None and updated_prices[1] is not None:
                        updated_prices[0] = self._coerce_history_price(1.0 - float(updated_prices[1]))
                    if updated_prices[1] is None and updated_prices[0] is not None:
                        updated_prices[1] = self._coerce_history_price(1.0 - float(updated_prices[0]))

                if ts_candidates:
                    update_ts = max(ts_candidates)

                history_point = self._build_market_history_point(1, updated_prices)
                if history_point is not None:
                    vector = history_point.get("outcome_prices")
                    if isinstance(vector, list) and len(vector) >= 2:
                        market["outcome_prices"] = [float(value) for value in vector]
                        try:
                            yes_val = float(vector[0])
                            no_val = float(vector[1])
                        except (TypeError, ValueError, IndexError):
                            yes_val = None
                            no_val = None

                if yes_val is not None and no_val is not None:
                    market["yes_price"] = yes_val
                    market["no_price"] = no_val
                    market["current_yes_price"] = yes_val
                    market["current_no_price"] = no_val
                    existing_vector = market.get("outcome_prices")
                    if isinstance(existing_vector, list) and len(existing_vector) >= 2:
                        market["outcome_prices"] = [yes_val, no_val, *existing_vector[2:]]
                    else:
                        market["outcome_prices"] = [yes_val, no_val]
                    market_id = str(market.get("id", "") or "")
                    if market_id:
                        market_price_lookup[market_id] = (yes_val, no_val)

                if update_ts is None:
                    update_ts = self._parse_market_price_timestamp(market.get("price_updated_at"))
                if update_ts is None and token_ids and opp_seen_ts is not None:
                    try:
                        existing_yes = float(market.get("yes_price"))
                        existing_no = float(market.get("no_price"))
                    except (TypeError, ValueError):
                        existing_yes = None
                        existing_no = None
                    if (
                        existing_yes is not None
                        and existing_no is not None
                        and 0.0 <= existing_yes <= 1.0
                        and 0.0 <= existing_no <= 1.0
                    ):
                        update_ts = opp_seen_ts
                if update_ts is None:
                    market["is_price_fresh"] = False if token_ids else True
                    market.setdefault("price_age_seconds", None)
                else:
                    age_seconds = max(0.0, now_ts - update_ts)
                    market["price_updated_at"] = self._format_price_timestamp(update_ts)
                    market["price_age_seconds"] = float(round(age_seconds, 3))
                    market["is_price_fresh"] = age_seconds <= max_age_seconds
                    if opp_last_priced_ts is None or update_ts > opp_last_priced_ts:
                        opp_last_priced_ts = update_ts

            if opp_last_priced_ts is not None:
                opp.last_priced_at = datetime.fromtimestamp(opp_last_priced_ts, tz=timezone.utc)
            elif getattr(opp, "last_priced_at", None) is None and opp_seen_dt is not None:
                opp.last_priced_at = opp_seen_dt

            if market_price_lookup:
                for position in opp.positions_to_take or []:
                    if not isinstance(position, dict):
                        continue
                    market_id = str(
                        position.get("market_id") or position.get("market") or position.get("id") or ""
                    ).strip()
                    if not market_id or market_id not in market_price_lookup:
                        continue
                    yes_val, no_val = market_price_lookup[market_id]
                    outcome = str(position.get("outcome") or "").strip().lower()
                    if outcome == "yes":
                        position["price"] = yes_val
                        position["current_price"] = yes_val
                    elif outcome == "no":
                        position["price"] = no_val
                        position["current_price"] = no_val

            if opp.execution_plan and getattr(opp.execution_plan, "legs", None):
                for leg in opp.execution_plan.legs:
                    market_id = str(getattr(leg, "market_id", "") or "")
                    if market_id not in market_price_lookup:
                        continue
                    yes_val, no_val = market_price_lookup[market_id]
                    outcome = str(getattr(leg, "outcome", "") or "").strip().lower()
                    if outcome == "yes":
                        leg.limit_price = yes_val
                    elif outcome == "no":
                        leg.limit_price = no_val

            if not drop_stale:
                filtered.append(opp)
                continue

            resolution_date = _make_aware(getattr(opp, "resolution_date", None))
            if resolution_date is not None and resolution_date <= now_dt:
                continue
            stale_market_found = any(
                self._market_price_is_stale(market, now_ts, max_age_seconds)
                for market in opp.markets
                if isinstance(market, dict)
            )
            if stale_market_found:
                seen_at = _make_aware(
                    getattr(opp, "last_detected_at", None)
                    or getattr(opp, "last_seen_at", None)
                    or getattr(opp, "detected_at", None)
                )
                if seen_at is None or (now_dt - seen_at).total_seconds() > max_age_seconds:
                    continue
            filtered.append(opp)

        return filtered

    def _full_snapshot_strategy_due(self, now: datetime) -> bool:
        if (
            self._full_snapshot_cycle_total_markets > 0
            and self._full_snapshot_cycle_processed_markets < self._full_snapshot_cycle_total_markets
        ):
            return True
        interval = max(
            int(getattr(settings, "FAST_SCAN_INTERVAL_SECONDS", 15) or 15),
            int(getattr(settings, "SCANNER_FULL_SNAPSHOT_STRATEGY_INTERVAL_SECONDS", 120) or 120),
        )
        if self._last_full_snapshot_strategy_scan is None:
            return True
        return (now - self._last_full_snapshot_strategy_scan).total_seconds() >= float(interval)

    @staticmethod
    def _fast_strategy_timeout_seconds() -> float:
        configured = float(getattr(settings, "SCANNER_FAST_STRATEGY_TIMEOUT_SECONDS", 12.0) or 12.0)
        return max(3.0, configured)

    @staticmethod
    def _full_snapshot_strategy_timeout_seconds() -> float:
        configured = float(getattr(settings, "SCANNER_FULL_SNAPSHOT_STRATEGY_TIMEOUT_SECONDS", 60.0) or 60.0)
        return max(5.0, configured)

    def note_fast_lane_watchdog_timeout(self, timeout_seconds: float) -> None:
        self._fast_watchdog_timeout_count += 1
        self._fast_inflight = False
        self._fast_lane_error = f"watchdog timeout after {timeout_seconds:.1f}s"

    def note_heavy_lane_watchdog_timeout(self, timeout_seconds: float) -> None:
        self._heavy_watchdog_timeout_count += 1
        self._heavy_inflight = False
        self._heavy_lane_error = f"watchdog timeout after {timeout_seconds:.1f}s"

    @staticmethod
    def _lane_watchdog_payload(
        *,
        now: datetime,
        started_at: Optional[datetime],
        inflight: bool,
        threshold_seconds: float,
    ) -> dict:
        if not inflight or started_at is None:
            return {
                "inflight": bool(inflight),
                "started_at": to_iso(started_at),
                "inflight_age_seconds": None,
                "threshold_seconds": float(threshold_seconds),
                "stalled": False,
            }
        age_seconds = max(0.0, (now - started_at).total_seconds())
        return {
            "inflight": True,
            "started_at": to_iso(started_at),
            "inflight_age_seconds": round(age_seconds, 3),
            "threshold_seconds": float(threshold_seconds),
            "stalled": age_seconds > float(threshold_seconds),
        }

    def _select_full_snapshot_markets(self, now: datetime, changed_markets: list, hot_markets: list) -> list:
        if bool(getattr(settings, "SCANNER_FORCE_FULL_UNIVERSE", True)):
            cap = 0
        else:
            cap = int(getattr(settings, "SCANNER_FULL_SNAPSHOT_MAX_MARKETS", 0) or 0)
        pool = [market for market in self._cached_markets if self._is_market_active(market, now)]
        if not pool:
            return []

        selected: list = []
        seen_ids: set[str] = set()

        def _append(markets: list) -> None:
            for market in markets:
                market_id = str(getattr(market, "id", "") or "")
                if not market_id or market_id in seen_ids:
                    continue
                seen_ids.add(market_id)
                selected.append(market)
                if cap > 0 and len(selected) >= cap:
                    return

        _append(changed_markets)
        if cap <= 0 or len(selected) < cap:
            _append(hot_markets)
        if cap <= 0 or len(selected) < cap:
            remaining = sorted(
                [market for market in pool if str(getattr(market, "id", "") or "") not in seen_ids],
                key=self._market_priority_key,
                reverse=True,
            )
            _append(remaining)

        if cap > 0:
            return selected[:cap]
        return selected

    def _get_reactive_trigger(self) -> asyncio.Event:
        if self._reactive_trigger is None:
            self._reactive_trigger = asyncio.Event()
        return self._reactive_trigger

    def _get_reactive_tokens_lock(self) -> asyncio.Lock:
        if self._reactive_tokens_lock is None:
            self._reactive_tokens_lock = asyncio.Lock()
        return self._reactive_tokens_lock

    async def _queue_reactive_tokens(self, changed_tokens: Set[str]) -> None:
        """Queue changed tokens from WS callbacks with bounded backpressure."""
        if not changed_tokens:
            return
        now = time.monotonic()
        cap = max(50, int(settings.REALTIME_SCAN_MAX_PENDING_TOKENS or 2000))
        reactive_lock = self._get_reactive_tokens_lock()
        async with reactive_lock:
            for token_id in changed_tokens:
                token = str(token_id or "").strip()
                if token:
                    self._pending_reactive_tokens[token] = now
            if len(self._pending_reactive_tokens) > cap:
                overflow = len(self._pending_reactive_tokens) - cap
                newest = sorted(
                    self._pending_reactive_tokens.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:cap]
                self._pending_reactive_tokens = dict(newest)
                self._reactive_backpressure_dropped_tokens += overflow
        self._get_reactive_trigger().set()

    async def consume_reactive_tokens(self, max_tokens: Optional[int] = None) -> list[str]:
        """Consume up to max_tokens pending reactive token IDs (newest first)."""
        limit = max(1, int(max_tokens or settings.REALTIME_SCAN_MAX_BATCH_TOKENS or 500))
        reactive_lock = self._get_reactive_tokens_lock()
        async with reactive_lock:
            if not self._pending_reactive_tokens:
                self._last_reactive_batch_tokens = 0
                return []
            ordered = sorted(
                self._pending_reactive_tokens.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            selected = ordered[:limit]
            remaining = ordered[limit:]
            self._pending_reactive_tokens = dict(remaining)
            tokens = [token for token, _ in selected]
            self._last_reactive_batch_tokens = len(tokens)
            if remaining:
                self._get_reactive_trigger().set()
            return tokens

    def _partition_market_refresh_strategies(self) -> tuple[set[str], set[str]]:
        """Split scanner market_data_refresh strategies into incremental vs full sets."""
        incremental: set[str] = set()
        full_snapshot: set[str] = set()
        within_market = str(MispricingType.WITHIN_MARKET.value).lower()

        # When strategy overrides are active (e.g. in tests), partition from those
        # instances instead of the global strategy_loader. An empty override list
        # still signals "override mode" so dispatch is not skipped.
        if self._strategy_overrides is not None:
            for instance in self._strategy_overrides:
                slug = getattr(instance, "slug", None) or getattr(instance, "name", "__override__")
                if getattr(instance, "worker_affinity", "scanner") != "scanner":
                    continue
                subscriptions = set(getattr(instance, "subscriptions", None) or [])
                if "*" in subscriptions:
                    full_snapshot.add(slug)
                    continue
                if EventType.MARKET_DATA_REFRESH not in subscriptions:
                    continue
                mode = str(getattr(instance, "realtime_processing_mode", "auto") or "auto").strip().lower()
                if mode == "incremental":
                    incremental.add(slug)
                    continue
                if mode == "full_snapshot":
                    full_snapshot.add(slug)
                    continue
                mispricing = str(getattr(instance, "mispricing_type", "") or "").strip().lower()
                if mispricing == within_market:
                    incremental.add(slug)
                else:
                    full_snapshot.add(slug)
            # In override mode with no matching strategies, use a sentinel so
            # _dispatch_market_refresh is still invoked (strategies handle themselves).
            if not incremental and not full_snapshot:
                incremental.add("__override__")
            return incremental, full_snapshot

        for slug, loaded in strategy_loader._loaded.items():
            instance = loaded.instance
            if getattr(instance, "worker_affinity", "scanner") != "scanner":
                continue
            subscriptions = set(getattr(instance, "subscriptions", None) or [])
            if "*" in subscriptions:
                full_snapshot.add(slug)
                continue
            if EventType.MARKET_DATA_REFRESH not in subscriptions:
                continue

            mode = str(getattr(instance, "realtime_processing_mode", "auto") or "auto").strip().lower()
            if mode == "incremental":
                incremental.add(slug)
                continue
            if mode == "full_snapshot":
                full_snapshot.add(slug)
                continue

            mispricing = str(getattr(instance, "mispricing_type", "") or "").strip().lower()
            if mispricing == within_market:
                incremental.add(slug)
            else:
                full_snapshot.add(slug)

        return incremental, full_snapshot

    async def _dispatch_market_refresh(
        self,
        event: DataEvent,
        *,
        incremental_slugs: Optional[set[str]] = None,
        full_slugs: Optional[set[str]] = None,
        full_market_snapshot: Optional[list] = None,
        full_prices: Optional[dict[str, dict]] = None,
        handler_timeout_seconds: Optional[float] = None,
    ) -> list[Opportunity]:
        """Dispatch market_data_refresh with per-strategy incremental/full routing."""
        if incremental_slugs is None or full_slugs is None:
            incremental_slugs, full_slugs = self._partition_market_refresh_strategies()
        if not incremental_slugs and not full_slugs:
            return await event_dispatcher.dispatch(
                event,
                handler_timeout_seconds=handler_timeout_seconds,
            )

        all_opportunities: list[Opportunity] = []
        if incremental_slugs:
            all_opportunities.extend(
                await event_dispatcher.dispatch(
                    event,
                    include_strategies=incremental_slugs,
                    handler_timeout_seconds=handler_timeout_seconds,
                )
            )

        if full_slugs:
            snapshot_markets = list(full_market_snapshot if full_market_snapshot is not None else (event.markets or []))
            if not snapshot_markets:
                return all_opportunities
            payload = dict(event.payload or {})
            payload["strategy_batch"] = "full_snapshot"
            full_event = DataEvent(
                event_type=event.event_type,
                source=event.source,
                timestamp=event.timestamp,
                market_id=event.market_id,
                token_id=event.token_id,
                payload=payload,
                old_price=event.old_price,
                new_price=event.new_price,
                markets=snapshot_markets,
                events=event.events,
                prices=dict(full_prices) if full_prices is not None else event.prices,
                scan_mode=event.scan_mode,
                changed_token_ids=event.changed_token_ids,
                changed_market_ids=event.changed_market_ids,
                affected_market_ids=event.affected_market_ids,
            )
            all_opportunities.extend(
                await event_dispatcher.dispatch(
                    full_event,
                    include_strategies=full_slugs,
                    handler_timeout_seconds=handler_timeout_seconds,
                )
            )

        return all_opportunities

    def _merge_market_history_points(self, market_id: str, incoming: list[dict[str, object]], now_ms: int) -> int:
        """Merge incoming history into in-memory store and return merged length."""
        if not incoming:
            return len(self._market_price_history.get(market_id, []))

        cutoff_ms = now_ms - int(self._market_history_retention_seconds * 1000)
        merged_by_ts: dict[int, dict[str, object]] = {}

        for point in self._market_price_history.get(market_id, []):
            normalized = self._normalize_history_point(point, cutoff_ms)
            if normalized is None:
                continue
            merged_by_ts[int(float(normalized["t"]))] = normalized

        for point in incoming:
            normalized = self._normalize_history_point(point, cutoff_ms)
            if normalized is None:
                continue
            merged_by_ts[int(float(normalized["t"]))] = normalized

        merged = [merged_by_ts[k] for k in sorted(merged_by_ts.keys())]
        if len(merged) > self._market_history_max_points:
            merged = merged[-self._market_history_max_points :]
        self._market_price_history[market_id] = merged
        return len(merged)

    async def _backfill_market_history_for_opportunities(self, opportunities: list[Opportunity], now: datetime) -> None:
        """Backfill multi-hour outcome history for visible opportunities."""
        if not opportunities:
            return

        # Local imports avoid circular initialization ordering with the provider layer.
        from services.kalshi_client import kalshi_client
        from services.polymarket import polymarket_client

        now_ms = int(now.timestamp() * 1000)
        window_ms = int(self._market_history_retention_seconds * 1000)
        start_ms = now_ms - window_ms
        start_s = start_ms // 1000
        now_s = now_ms // 1000
        sample_ms = self._market_history_sample_interval_ms

        polymarket_candidates: list[str] = []
        missing_polymarket_candidates: list[str] = []
        kalshi_candidates: list[str] = []
        seen: set[str] = set()
        for opp in opportunities:
            for market in opp.markets:
                market_id = str(market.get("id", "") or "")
                if not market_id or market_id in seen:
                    continue
                seen.add(market_id)
                if market_id in self._market_history_backfill_done:
                    continue
                last_attempt = self._market_history_backfill_attempt_ms.get(market_id, 0)
                if (now_ms - last_attempt) < self._market_history_backfill_retry_ms:
                    continue
                platform = str(market.get("platform", "polymarket") or "polymarket").lower()
                if platform == "kalshi":
                    kalshi_candidates.append(market_id)
                else:
                    if market_id not in self._market_outcome_token_ids:
                        missing_polymarket_candidates.append(market_id)
                    else:
                        polymarket_candidates.append(market_id)
                if (
                    len(polymarket_candidates) + len(kalshi_candidates) + len(missing_polymarket_candidates)
                    >= self._market_history_backfill_max_markets
                ):
                    break
            if (
                len(polymarket_candidates) + len(kalshi_candidates) + len(missing_polymarket_candidates)
                >= self._market_history_backfill_max_markets
            ):
                break

        def _extract_token_ids_from_market_payload(payload: object) -> Optional[tuple[str, ...]]:
            if not isinstance(payload, dict):
                return None

            direct = self._coerce_polymarket_token_ids(
                payload.get("clob_token_ids")
                or payload.get("clobTokenIds")
                or payload.get("token_ids")
                or payload.get("tokenIds")
            )
            if direct is not None:
                return direct

            tokens = payload.get("tokens")
            if not isinstance(tokens, list):
                return None
            inferred_ids: list[str] = []
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                token_id = str(
                    token.get("token_id")
                    or token.get("tokenId")
                    or token.get("asset_id")
                    or token.get("assetId")
                    or token.get("id")
                    or ""
                ).strip()
                if token_id and len(token_id) > 20:
                    inferred_ids.append(token_id)
            return self._coerce_polymarket_token_ids(inferred_ids)

        if missing_polymarket_candidates:
            resolver_semaphore = asyncio.Semaphore(max(1, self._market_history_backfill_concurrency))

            async def _resolve_missing_token_ids(market_id: str) -> bool:
                async with resolver_semaphore:
                    try:
                        market_payload = await polymarket_client.get_market_by_condition_id(market_id)
                    except Exception:
                        market_payload = None

                    token_ids = _extract_token_ids_from_market_payload(market_payload)
                    if token_ids is None:
                        self._market_history_backfill_attempt_ms[market_id] = now_ms
                        return False

                    self._market_outcome_token_ids[market_id] = token_ids
                    self._market_token_ids[market_id] = (token_ids[0], token_ids[1])
                    return True

            resolution_results = await asyncio.gather(
                *[_resolve_missing_token_ids(mid) for mid in missing_polymarket_candidates],
                return_exceptions=True,
            )
            for market_id, resolved in zip(missing_polymarket_candidates, resolution_results):
                if (
                    resolved is True
                    and len(polymarket_candidates) + len(kalshi_candidates) < self._market_history_backfill_max_markets
                ):
                    polymarket_candidates.append(market_id)

        total_candidates = len(polymarket_candidates) + len(kalshi_candidates)
        if total_candidates == 0:
            return

        semaphore = asyncio.Semaphore(self._market_history_backfill_concurrency)

        async def _fetch_polymarket(
            market_id: str,
        ) -> tuple[str, list[dict[str, object]], bool]:
            async with semaphore:
                self._market_history_backfill_attempt_ms[market_id] = now_ms
                token_ids = list(self._market_outcome_token_ids.get(market_id) or ())
                if len(token_ids) < 2:
                    return market_id, [], False

                token_results = await asyncio.gather(
                    *[
                        polymarket_client.get_prices_history(
                            token_id,
                            start_ts=start_s,
                            end_ts=now_s,
                        )
                        for token_id in token_ids
                    ],
                    return_exceptions=True,
                )
                history_by_outcome: list[dict[int, float]] = []
                fetch_success = False
                for result in token_results:
                    by_bucket: dict[int, float] = {}
                    if not isinstance(result, Exception):
                        fetch_success = True
                        for point in result:
                            if not isinstance(point, dict):
                                continue
                            t = point.get("t")
                            p = point.get("p")
                            if t is None or p is None:
                                continue
                            try:
                                ts = int(float(t))
                                price = float(p)
                            except (TypeError, ValueError):
                                continue
                            if ts < start_ms or ts > now_ms or not (0 <= price <= 1.01):
                                continue
                            bucket = start_ms + ((ts - start_ms) // sample_ms) * sample_ms
                            by_bucket[bucket] = price
                    history_by_outcome.append(by_bucket)

                if not fetch_success:
                    return market_id, [], False

                buckets: set[int] = set()
                for by_bucket in history_by_outcome:
                    buckets.update(by_bucket.keys())
                ordered_buckets = sorted(buckets)

                last_prices: list[Optional[float]] = [None] * len(history_by_outcome)
                merged: list[dict[str, object]] = []
                for bucket in ordered_buckets:
                    for idx, by_bucket in enumerate(history_by_outcome):
                        if bucket in by_bucket:
                            last_prices[idx] = self._coerce_history_price(by_bucket[bucket])
                    merged_point = self._build_market_history_point(bucket, list(last_prices))
                    if merged_point is not None:
                        merged.append(merged_point)

                return market_id, merged, fetch_success

        updated = 0
        completed = 0

        def _apply_backfill_results(results: list[tuple[str, list[dict[str, object]], bool]]) -> None:
            nonlocal updated, completed
            for market_id, points, success in results:
                if success:
                    completed += 1
                if not points:
                    continue
                merged_len = self._merge_market_history_points(market_id, points, now_ms)
                if merged_len >= 2:
                    updated += 1
                    self._market_history_backfill_done.add(market_id)

        if polymarket_candidates:
            poly_batch_size = max(1, self._market_history_backfill_concurrency * 2)
            for i in range(0, len(polymarket_candidates), poly_batch_size):
                batch = polymarket_candidates[i : i + poly_batch_size]
                poly_results = await asyncio.gather(*[_fetch_polymarket(mid) for mid in batch])
                _apply_backfill_results(poly_results)

        # Kalshi provides batched candlestick history by market ticker.
        if kalshi_candidates:
            batch_size = 80
            period_minutes = max(
                1,
                int(self._market_history_sample_interval_ms // 60000),
            )
            for i in range(0, len(kalshi_candidates), batch_size):
                batch = kalshi_candidates[i : i + batch_size]
                for market_id in batch:
                    self._market_history_backfill_attempt_ms[market_id] = now_ms

                try:
                    history_map = await kalshi_client.get_market_candlesticks_batch(
                        batch,
                        start_ts=start_s,
                        end_ts=now_s,
                        period_interval=period_minutes,
                        include_latest_before_start=True,
                    )
                    fetch_success = True
                except Exception:
                    history_map = {}
                    fetch_success = False

                for market_id in batch:
                    raw_points = history_map.get(market_id, [])
                    by_bucket: dict[int, dict[str, float]] = {}
                    for point in raw_points:
                        try:
                            ts = int(float(point.get("t", 0)))
                            yes = float(point.get("yes", 0))
                            no = float(point.get("no", 0))
                        except (TypeError, ValueError):
                            continue
                        if ts < start_ms or ts > now_ms:
                            continue
                        if not (0.0 <= yes <= 1.01 and 0.0 <= no <= 1.01):
                            continue
                        bucket = start_ms + ((ts - start_ms) // sample_ms) * sample_ms
                        by_bucket[bucket] = {
                            "t": float(bucket),
                            "yes": float(round(min(1.0, max(0.0, yes)), 6)),
                            "no": float(round(min(1.0, max(0.0, no)), 6)),
                        }
                    merged_points = [by_bucket[k] for k in sorted(by_bucket.keys())]
                    market_fetch_success = fetch_success and market_id in history_map
                    _apply_backfill_results([(market_id, merged_points, market_fetch_success)])

        if updated > 0:
            print(
                f"  Sparkline backfill: hydrated {updated}/{total_candidates} markets "
                f"({completed} successful history fetches)"
            )

    def _queue_market_history_backfill(self, opportunities: list[Opportunity]) -> None:
        """Queue market history backfill to run asynchronously without blocking callers."""
        if not opportunities:
            return

        self._market_history_backfill_queue = list(opportunities)

        task = self._market_history_backfill_task
        if task is not None and not task.done():
            return

        async def _run_queue() -> None:
            while self._market_history_backfill_queue:
                batch = self._market_history_backfill_queue
                self._market_history_backfill_queue = []
                try:
                    await self._backfill_market_history_for_opportunities(batch, datetime.now(timezone.utc))
                    await self._persist_market_history_for_opportunities(batch)
                except Exception as e:
                    print(f"  Async sparkline backfill queue error: {e}")

        backfill_task = asyncio.create_task(_run_queue(), name="scanner_market_history_backfill")
        self._market_history_backfill_task = backfill_task

        def _clear_task(done_task: asyncio.Task) -> None:
            if self._market_history_backfill_task is done_task:
                self._market_history_backfill_task = None

        backfill_task.add_done_callback(_clear_task)

    async def _persist_market_history_for_opportunities(self, opportunities: list[Opportunity]) -> None:
        """Persist backfilled market history so other workers/routes can read it immediately."""
        if not opportunities:
            return
        history = self.get_market_history_for_opportunities(opportunities)
        if not history:
            return

        from services.shared_state import SNAPSHOT_ID

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
                row = result.scalar_one_or_none()
                if row is None:
                    return
                persisted = row.market_history_json if isinstance(row.market_history_json, dict) else {}
                changed = False
                for market_id, points in history.items():
                    if not isinstance(points, list) or len(points) < 2:
                        continue
                    previous = persisted.get(market_id)
                    if previous == points:
                        continue
                    persisted[market_id] = points
                    changed = True
                if not changed:
                    return
                row.market_history_json = persisted
                await session.commit()
            except OperationalError:
                await session.rollback()

    def get_market_history_for_opportunities(
        self, opportunities: list[Opportunity], max_points: Optional[int] = None
    ) -> dict[str, list[dict[str, object]]]:
        """Return compact market history map for markets present in opportunities."""
        market_ids: set[str] = set()
        for opp in opportunities:
            for market in opp.markets:
                for market_id in self._market_history_lookup_ids(market):
                    market_ids.add(market_id)

        out: dict[str, list[dict[str, object]]] = {}
        export_points = self._market_history_export_points if max_points is None else max_points
        limit = max(2, min(self._market_history_max_points, int(export_points)))
        for mid in market_ids:
            hist = self._market_price_history.get(mid, [])
            if hist:
                out[mid] = hist[-limit:]
        return out

    def get_broad_market_history(self, max_markets: int = 500) -> dict[str, list[dict[str, object]]]:
        """Export history for all cached markets, sorted by recency, capped at *max_markets*.

        Unlike ``get_market_history_for_opportunities`` which only returns history
        for markets present in a specific opportunity set, this method exports the
        broadest set of history available so other workers (traders, weather) can
        hydrate their sparklines from the DB even when they run in a subprocess
        with an empty in-memory cache.
        """
        export_points = self._market_history_export_points
        limit = max(2, min(self._market_history_max_points, int(export_points)))

        # Sort markets by most-recent data point (descending) so we keep the
        # most relevant markets when capping at max_markets.
        def _last_ts(hist: list[dict[str, object]]) -> float:
            return float(hist[-1].get("t", 0)) if hist else 0.0

        candidates = [(mid, hist) for mid, hist in self._market_price_history.items() if len(hist) >= 2]
        candidates.sort(key=lambda pair: _last_ts(pair[1]), reverse=True)

        out: dict[str, list[dict[str, object]]] = {}
        for mid, hist in candidates[:max_markets]:
            out[mid] = hist[-limit:]
        return out

    async def attach_price_history_to_opportunities(
        self,
        opportunities: list[Opportunity],
        *,
        now: Optional[datetime] = None,
        timeout_seconds: Optional[float] = 0.0,
        block_for_backfill: bool = False,
    ) -> int:
        """Attach scanner-managed market history without blocking by default."""
        if not opportunities:
            return 0

        ts = now or datetime.now(timezone.utc)
        self._remember_market_tokens_from_opportunities(opportunities)

        # Hydrate local cache from the scanner worker's persisted market history.
        # This is critical for subprocesses (tracked_traders, weather) that have
        # their own scanner singleton with an empty _market_price_history.
        needed_ids: set[str] = set()
        for opp in opportunities:
            for market in opp.markets:
                lookup_ids = self._market_history_lookup_ids(market)
                if lookup_ids and any(mid in self._market_price_history for mid in lookup_ids):
                    continue
                for market_id in lookup_ids:
                    if market_id not in self._market_price_history:
                        needed_ids.add(market_id)
        if needed_ids:
            try:
                await self._hydrate_history_from_db(needed_ids)
            except Exception:
                pass  # non-fatal; backfill will attempt API fetch below

        should_block = bool(block_for_backfill or timeout_seconds is None)
        if should_block:
            try:
                if timeout_seconds is not None and timeout_seconds > 0:
                    await asyncio.wait_for(
                        self._backfill_market_history_for_opportunities(opportunities, ts),
                        timeout=timeout_seconds,
                    )
                else:
                    await self._backfill_market_history_for_opportunities(opportunities, ts)
                await self._persist_market_history_for_opportunities(opportunities)
            except asyncio.TimeoutError:
                print("  Sparkline backfill timed out (shared attach)")
            except Exception as e:
                print(f"  Sparkline backfill error in shared attach: {e}")
        else:
            self._queue_market_history_backfill(opportunities)

        market_history = self.get_market_history_for_opportunities(opportunities)
        attached = 0
        for opp in opportunities:
            for market in opp.markets:
                history: list[dict[str, object]] = []
                for market_id in self._market_history_lookup_ids(market):
                    candidate = market_history.get(market_id, [])
                    if len(candidate) > len(history):
                        history = candidate
                if len(history) < 2:
                    continue
                market["price_history"] = history
                attached += 1
        return attached

    async def _hydrate_history_from_db(self, market_ids: set[str]) -> int:
        """Load persisted market history from the scanner snapshot into local cache."""
        from models.database import AsyncSessionLocal, ScannerSnapshot
        from services.shared_state import SNAPSHOT_ID
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == SNAPSHOT_ID))
            row = result.scalar_one_or_none()
            if row is None:
                return 0
            db_history = row.market_history_json if isinstance(row.market_history_json, dict) else {}
            fallback_history: dict[str, list[dict[str, object]]] = {}
            history_alias_map: dict[str, str] = {}
            if isinstance(row.opportunities_json, list):
                needed_market_ids = set(market_ids)
                for item in row.opportunities_json:
                    if not isinstance(item, dict):
                        continue
                    markets = item.get("markets")
                    if not isinstance(markets, list):
                        continue
                    for market in markets:
                        if not isinstance(market, dict):
                            continue
                        market_id = str(market.get("id", "") or "").strip()
                        condition_id = str(market.get("condition_id") or market.get("conditionId") or "").strip()
                        if market_id and condition_id and market_id != condition_id:
                            history_alias_map[market_id] = condition_id
                            history_alias_map[condition_id] = market_id
                        history = market.get("price_history")
                        if not isinstance(history, list) or len(history) < 2:
                            continue
                        for candidate in (market_id, condition_id):
                            if not candidate:
                                continue
                            if candidate not in needed_market_ids:
                                continue
                            if candidate in fallback_history:
                                continue
                            fallback_history[candidate] = history

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        hydrated = 0
        for mid in market_ids:
            points = db_history.get(mid)
            if not isinstance(points, list) or len(points) < 2:
                alias = history_alias_map.get(mid)
                if alias:
                    points = db_history.get(alias)
            if not isinstance(points, list) or len(points) < 2:
                points = fallback_history.get(mid)
                if not isinstance(points, list) or len(points) < 2:
                    alias = history_alias_map.get(mid)
                    if alias:
                        points = fallback_history.get(alias)
            if not isinstance(points, list) or len(points) < 2:
                continue
            merged = self._merge_market_history_points(mid, points, now_ms)
            if merged >= 2:
                hydrated += 1
        return hydrated

    async def _notify_status_change(self):
        """Notify all status callbacks of a change"""
        status = self.get_status()
        for callback in self._status_callbacks:
            try:
                await callback(status)
            except Exception as e:
                print(f"  Status callback error: {e}")

    async def load_settings(self):
        """Load scanner settings from database"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(ScannerSettings).where(ScannerSettings.id == "default"))
                settings_row = result.scalar_one_or_none()

                if settings_row:
                    self._enabled = settings_row.is_enabled
                    self._interval_seconds = settings_row.scan_interval_seconds
                    # Sync global pause state with persisted setting
                    if self._enabled:
                        global_pause_state.resume()
                    else:
                        global_pause_state.pause()
                    print(f"Loaded scanner settings: enabled={self._enabled}, interval={self._interval_seconds}s")
                else:
                    # Create default settings
                    new_settings = ScannerSettings(
                        id="default",
                        is_enabled=True,
                        scan_interval_seconds=settings.SCAN_INTERVAL_SECONDS,
                    )
                    session.add(new_settings)
                    await session.commit()
                    print("Created default scanner settings")
        except Exception as e:
            print(f"Error loading scanner settings: {e}")

    async def save_settings(self):
        """Save scanner settings to database"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(ScannerSettings).where(ScannerSettings.id == "default"))
                settings_row = result.scalar_one_or_none()

                if settings_row:
                    settings_row.is_enabled = self._enabled
                    settings_row.scan_interval_seconds = self._interval_seconds
                    settings_row.updated_at = utcnow()
                else:
                    settings_row = ScannerSettings(
                        id="default",
                        is_enabled=self._enabled,
                        scan_interval_seconds=self._interval_seconds,
                    )
                    session.add(settings_row)

                await session.commit()
                print(f"Saved scanner settings: enabled={self._enabled}, interval={self._interval_seconds}s")
        except Exception as e:
            print(f"Error saving scanner settings: {e}")

    async def load_plugins(self, source_keys: Optional[list[str]] = None):
        """Load all enabled strategies from the database via unified loader."""
        try:
            async with AsyncSessionLocal() as session:
                await ensure_system_opportunity_strategies_seeded(session)
                result = await strategy_loader.refresh_all_from_db(
                    session=session,
                    source_keys=source_keys,
                    prune_unlisted=bool(source_keys),
                )

            loaded_count = len(result.get("loaded", []))
            error_count = len(result.get("errors", {}))
            self._plugins_loaded = True
        except Exception as e:
            print(f"Error loading strategies: {e}")

    async def _ensure_runtime_strategies_loaded(self) -> None:
        if self._strategy_overrides is not None:
            return
        if self._plugins_loaded:
            return
        await self.load_plugins()

    def _get_all_strategies(self) -> list:
        """Return DB-loaded strategy instances whose worker_affinity is 'scanner'.

        Strategies with other worker affinities (e.g. 'crypto') are handled
        by their respective workers and should not run in the scanner loop.
        """
        if self._strategy_overrides is not None:
            return list(self._strategy_overrides)
        plugin_strategies = strategy_loader.get_all_instances()
        scanner_strategies = [s for s in plugin_strategies if getattr(s, "worker_affinity", "scanner") == "scanner"]
        return scanner_strategies

    def _get_news_edge_helper(self):
        """Return the news_edge strategy instance for prefetch utilities."""
        loaded = strategy_loader.get_strategy("news_edge")
        if loaded and hasattr(loaded.instance, "_build_market_infos"):
            return loaded.instance
        return None

    @staticmethod
    def _strategy_key(strategy) -> str:
        st = getattr(strategy, "strategy_type", "")
        return st if isinstance(st, str) else getattr(st, "value", "")

    def _is_scanner_strategy_active(self, slug: str) -> bool:
        target_slug = str(slug or "").strip().lower()
        if not target_slug:
            return False
        for strategy in self._get_all_strategies():
            if str(self._strategy_key(strategy) or "").strip().lower() == target_slug:
                return True
        return False

    @staticmethod
    def _default_mispricing_for_strategy(strategy_key: str) -> Optional[MispricingType]:
        slug = str(strategy_key or "").strip().lower()
        if not slug:
            return None
        if slug == "settlement_lag":
            return MispricingType.SETTLEMENT_LAG
        if slug in {"combinatorial", "cross_market"}:
            return MispricingType.CROSS_MARKET
        if slug == "news_edge":
            return MispricingType.NEWS_INFORMATION
        return MispricingType.WITHIN_MARKET

    def _hydrate_opportunity_defaults(self, opportunity: Opportunity, strategy: object) -> None:
        if not opportunity.strategy:
            opportunity.strategy = str(self._strategy_key(strategy) or "")
        if opportunity.mispricing_type is None:
            default_mispricing = self._default_mispricing_for_strategy(
                str(opportunity.strategy or self._strategy_key(strategy) or "")
            )
            if default_mispricing is not None:
                opportunity.mispricing_type = default_mispricing

    async def _run_override_strategies(
        self,
        *,
        events: list,
        markets: list,
        prices: dict,
    ) -> list[Opportunity]:
        """Run strategy overrides (test mode). Calls detect() directly."""
        opportunities: list[Opportunity] = []
        for strategy in self._strategy_overrides or []:
            try:
                detect_async = getattr(strategy, "detect_async", None)
                detect = getattr(strategy, "detect", None)
                strategy_opportunities: list[Opportunity] = []

                if inspect.iscoroutinefunction(detect_async):
                    strategy_opportunities = await detect_async(events, markets, prices)
                elif callable(detect):
                    if inspect.iscoroutinefunction(detect):
                        strategy_opportunities = await detect(events, markets, prices)
                    else:
                        strategy_opportunities = detect(events, markets, prices)

                for opportunity in strategy_opportunities or []:
                    if not isinstance(opportunity, Opportunity):
                        continue
                    self._hydrate_opportunity_defaults(opportunity, strategy)
                    opportunities.append(opportunity)
            except Exception as e:
                strategy_name = str(getattr(strategy, "name", self._strategy_key(strategy) or "unknown"))
                print(f"  Strategy {strategy_name} failed: {e}")
        return opportunities

    def _strategy_runtime_status_rows(self) -> list[dict]:
        rows: list[dict] = []
        loaded_slugs: set[str] = set()

        for strategy in self._get_all_strategies():
            strategy_key = str(self._strategy_key(strategy) or "").strip().lower()
            if not strategy_key:
                continue
            loaded_slugs.add(strategy_key)
            rows.append(
                {
                    "name": getattr(strategy, "name", strategy_key.replace("_", " ").title()),
                    "type": strategy_key,
                    "status": "loaded",
                    "error_message": None,
                }
            )

        if self._plugins_loaded:
            for strategy_key, error in strategy_loader._errors.items():
                slug = str(strategy_key or "").strip().lower()
                if not slug or slug in loaded_slugs:
                    continue
                error_text = str(error or "").strip()
                if error_text:
                    first_line = error_text.splitlines()[0].strip()
                    error_text = first_line or error_text
                rows.append(
                    {
                        "name": slug.replace("_", " ").title(),
                        "type": slug,
                        "status": "error",
                        "error_message": error_text or "Unknown strategy load error",
                    }
                )

        rows.sort(key=lambda row: str(row.get("type") or ""))
        return rows

    # ------------------------------------------------------------------
    # Market catalog: fetch from upstream APIs, persist to DB, hydrate
    # ------------------------------------------------------------------

    async def refresh_catalog(self) -> int:
        """Fetch market catalog from upstream APIs and persist to DB.

        This is the slow, HTTP-bound operation (catalog fetch) that runs
        independently on its own timer so scan_fast() is never blocked
        by upstream API slowness.

        Returns the number of markets in the refreshed catalog.
        """
        import time as _time

        _t0 = _time.monotonic()
        print(f"[{utcnow().isoformat()}] Refreshing market catalog...")
        await self._set_activity("Catalog refresh: fetching Polymarket events and markets...")

        try:
            await self._ensure_runtime_strategies_loaded()

            # Phase 1 — Fetch events + markets concurrently
            _phase_t = _time.monotonic()
            events, markets = await asyncio.gather(
                self.market_data.get_all_events(closed=False),
                self.market_data.get_all_markets(active=True),
            )
            print(f"  [timing] Polymarket fetch: {_time.monotonic() - _phase_t:.1f}s")

            now = datetime.now(timezone.utc)

            # Merge event-embedded markets into the flat list (with later hard caps).
            flat_ids = {str(getattr(m, "id", "") or "") for m in markets}
            extra_from_events = 0
            for event in events:
                for market in list(getattr(event, "markets", None) or []):
                    market_id = str(getattr(market, "id", "") or "")
                    if not market_id or market_id in flat_ids:
                        continue
                    markets.append(market)
                    flat_ids.add(market_id)
                    extra_from_events += 1

            # Phase 2 — Fetch Kalshi markets
            if self._is_scanner_strategy_active("cross_platform"):
                await self._set_activity("Catalog refresh: fetching Kalshi markets...")
                _phase_t = _time.monotonic()
                try:
                    kalshi_markets = await self.market_data.get_cross_platform_markets(active=True)
                    markets.extend(kalshi_markets)

                    kalshi_events = await self.market_data.get_cross_platform_events(closed=False)
                    events.extend(kalshi_events)

                    if kalshi_markets:
                        print(f"  Fetched {len(kalshi_events)} Kalshi events and {len(kalshi_markets)} Kalshi markets")
                        print(f"  [timing] Kalshi fetch: {_time.monotonic() - _phase_t:.1f}s")
                except Exception as e:
                    print(f"  Kalshi fetch failed (non-fatal): {e}")

            # Phase 2b — prune closed/resolved/expired, then enforce hard caps.
            events, markets = self._prune_active_catalog(events, markets, now)
            events, markets = self._enforce_catalog_caps(events, markets)
            dedup_msg = f" (+{extra_from_events} from events)" if extra_from_events else ""
            print(f"  Fetched {len(events)} events and {len(markets)} active markets{dedup_msg}")
            await self._set_activity(f"Catalog: {len(events)} events, {len(markets)} active markets")

            # Phase 3 — Read live prices from Redis for ALL tokens
            all_token_ids = self._collect_live_token_ids(markets)
            # Deduplicate
            seen_ids: set[str] = set()
            deduped: list[str] = []
            for tid in all_token_ids:
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    deduped.append(tid)
            all_token_ids = deduped

            prices: dict = {}
            if all_token_ids:
                _phase_t = _time.monotonic()
                await self._set_activity(f"Catalog refresh: reading prices for {len(all_token_ids)} tokens...")
                prices = await self._snapshot_ws_prices(all_token_ids)
                print(f"  Loaded prices for {len(prices)}/{len(all_token_ids)} tokens from Redis")
                print(f"  [timing] Price load: {_time.monotonic() - _phase_t:.1f}s")
            self._apply_live_prices_to_markets(markets, prices)

            # Phase 4 — Update in-memory caches
            self._cached_events = list(events)
            self._cached_markets = list(markets)
            self._cached_prices = dict(prices)
            self._remember_market_tokens(markets)
            self._rebuild_realtime_graph(events, markets)
            self._trim_runtime_market_caches({str(getattr(m, "id", "") or "") for m in markets})
            self._update_market_price_history(markets, prices, now)

            # Phase 5 — Subscribe WS feeds to active tokens
            if settings.WS_FEED_ENABLED:
                try:
                    feed_mgr = get_feed_manager()
                    if feed_mgr._started:
                        poly_tokens = [t for t in self._collect_polymarket_tokens(markets)[:500] if len(t) > 20]
                        if poly_tokens:
                            await feed_mgr.polymarket_feed.subscribe(token_ids=poly_tokens[:200])
                except Exception as e:
                    print(f"  WS subscription failed (non-critical): {e}")

            # Keep MarketMonitor priorities current without triggering extra upstream fetches.
            try:
                from services.market_monitor import market_monitor

                await market_monitor.ingest_snapshot(events, markets)
            except Exception:
                pass

            # Phase 6 — Persist catalog to DB
            duration = _time.monotonic() - _t0
            try:
                from models.database import AsyncSessionLocal
                from services.shared_state import write_market_catalog

                async with AsyncSessionLocal() as session:
                    await write_market_catalog(
                        session,
                        events,
                        markets,
                        duration_seconds=duration,
                    )
            except Exception as e:
                print(f"  Catalog DB persist failed (non-fatal): {e}")

            self._last_catalog_refresh = now
            self._last_full_scan = now

            # Fire background news prefetch if enabled
            if settings.NEWS_EDGE_ENABLED:
                asyncio.create_task(self._prefetch_news_matches(events, markets, prices))

            print(
                f"[{utcnow().isoformat()}] Catalog refresh complete: "
                f"{len(events)} events, {len(markets)} markets in {duration:.1f}s"
            )
            await self._set_activity(f"Catalog refresh complete — {len(events)} events, {len(markets)} markets")
            return len(markets)

        except Exception as e:
            # Persist the error so UI can display catalog health
            try:
                from models.database import AsyncSessionLocal
                from services.shared_state import write_market_catalog

                async with AsyncSessionLocal() as session:
                    await write_market_catalog(session, [], [], error=str(e))
            except Exception:
                pass
            print(f"[{utcnow().isoformat()}] Catalog refresh error: {e}")
            await self._set_activity(f"Catalog refresh error: {e}")
            raise

    async def refresh_catalog_incremental(
        self,
        *,
        force_full: bool = False,
    ) -> dict[str, object]:
        """Incremental market/event sync with periodic full-reconcile fallback."""
        import time as _time
        from models.market import Event

        _t0 = _time.monotonic()
        now = datetime.now(timezone.utc)

        if not self._cached_markets:
            await self._hydrate_catalog_from_db()

        if force_full or not self._cached_markets:
            market_count = await self.refresh_catalog()
            return {
                "mode": "full",
                "event_count": len(self._cached_events),
                "market_count": int(market_count),
                "delta_market_count": int(market_count),
                "delta_event_count": len(self._cached_events),
                "duration_seconds": round(max(0.0, _time.monotonic() - _t0), 3),
            }

        await self._ensure_runtime_strategies_loaded()
        await self._set_activity("Catalog incremental sync: fetching market/event deltas...")

        delta_since_minutes = max(
            1,
            int(getattr(settings, "MARKET_UNIVERSE_INCREMENTAL_SINCE_MINUTES", 5) or 5),
        )
        if self._last_catalog_refresh is not None:
            elapsed_minutes = int(max(0.0, (now - self._last_catalog_refresh).total_seconds()) / 60.0) + 1
            delta_since_minutes = max(delta_since_minutes, elapsed_minutes)
        delta_since_minutes = min(delta_since_minutes, 24 * 60)
        max_event_slugs = max(
            10,
            int(getattr(settings, "MARKET_UNIVERSE_INCREMENTAL_MAX_EVENT_SLUGS", 250) or 250),
        )

        delta_markets: list = []
        try:
            delta_markets = await self.market_data.get_recent_markets(
                since_minutes=delta_since_minutes,
                active=True,
            )
        except Exception as e:
            print(f"  Incremental market fetch failed, falling back to full refresh: {e}")
            market_count = await self.refresh_catalog()
            return {
                "mode": "full",
                "event_count": len(self._cached_events),
                "market_count": int(market_count),
                "delta_market_count": 0,
                "delta_event_count": 0,
                "duration_seconds": round(max(0.0, _time.monotonic() - _t0), 3),
                "error": str(e),
            }

        event_slug_candidates: list[str] = []
        event_slug_seen: set[str] = set()
        for market in delta_markets:
            slug = str(getattr(market, "event_slug", "") or "").strip()
            if not slug or slug in event_slug_seen:
                continue
            event_slug_seen.add(slug)
            event_slug_candidates.append(slug)
            if len(event_slug_candidates) >= max_event_slugs:
                break

        delta_events: list = []
        if event_slug_candidates:
            try:
                delta_events = await self.market_data.get_events_by_slugs(
                    event_slug_candidates,
                    closed=False,
                )
            except Exception as e:
                print(f"  Incremental event fetch failed (non-fatal): {e}")
                delta_events = []

        market_map: dict[str, object] = {}
        for market in self._cached_markets:
            market_id = str(getattr(market, "id", "") or "").strip()
            if market_id:
                market_map[market_id] = market

        touched_market_ids: set[str] = set()
        for market in delta_markets:
            market_id = str(getattr(market, "id", "") or "").strip()
            if not market_id:
                continue
            touched_market_ids.add(market_id)
            if self._is_market_active(market, now):
                market_map[market_id] = market
            else:
                market_map.pop(market_id, None)

        merged_markets = list(market_map.values())
        event_map: dict[str, object] = {}
        for event in self._cached_events:
            key = str(getattr(event, "slug", "") or getattr(event, "id", "") or "").strip()
            if key:
                event_map[key] = event

        touched_event_keys: set[str] = set(event_slug_candidates)
        for event in delta_events:
            key = str(getattr(event, "slug", "") or getattr(event, "id", "") or "").strip()
            if not key:
                continue
            touched_event_keys.add(key)
            if bool(getattr(event, "closed", False)):
                event_map.pop(key, None)
                continue
            event_map[key] = event

        markets_by_slug: dict[str, list] = {}
        for market in merged_markets:
            slug = str(getattr(market, "event_slug", "") or "").strip()
            if not slug:
                continue
            if slug not in markets_by_slug:
                markets_by_slug[slug] = [market]
            else:
                markets_by_slug[slug].append(market)

        for event_key in touched_event_keys:
            linked_markets = markets_by_slug.get(event_key, [])
            if not linked_markets:
                event_map.pop(event_key, None)
                continue
            event = event_map.get(event_key)
            if event is None:
                event = Event(
                    id=event_key,
                    slug=event_key,
                    title=event_key.replace("-", " ").strip().title(),
                    description="",
                    category=None,
                    tags=[],
                    markets=linked_markets,
                    neg_risk=False,
                    active=True,
                    closed=False,
                )
                event_map[event_key] = event
            else:
                event.markets = linked_markets

        merged_events = list(event_map.values())
        merged_events, merged_markets = self._prune_active_catalog(merged_events, merged_markets, now)
        merged_events, merged_markets = self._enforce_catalog_caps(merged_events, merged_markets)

        all_token_ids = self._collect_live_token_ids(merged_markets)
        prices: dict[str, dict] = {}
        if all_token_ids:
            prices = await self._snapshot_ws_prices(all_token_ids)
        self._apply_live_prices_to_markets(merged_markets, prices)

        self._cached_events = list(merged_events)
        self._cached_markets = list(merged_markets)
        self._cached_prices = dict(prices)
        self._remember_market_tokens(merged_markets)
        self._rebuild_realtime_graph(merged_events, merged_markets)
        self._trim_runtime_market_caches({str(getattr(m, "id", "") or "") for m in merged_markets})
        self._update_market_price_history(merged_markets, prices, now)

        if settings.WS_FEED_ENABLED:
            try:
                feed_mgr = get_feed_manager()
                if feed_mgr._started:
                    poly_tokens = [t for t in self._collect_polymarket_tokens(merged_markets)[:500] if len(t) > 20]
                    if poly_tokens:
                        await feed_mgr.polymarket_feed.subscribe(token_ids=poly_tokens[:200])
            except Exception as e:
                print(f"  WS subscription sync failed (non-critical): {e}")

        try:
            from services.market_monitor import market_monitor

            await market_monitor.ingest_snapshot(merged_events, merged_markets)
        except Exception:
            pass

        duration = _time.monotonic() - _t0
        try:
            from services.shared_state import write_market_catalog

            async with AsyncSessionLocal() as session:
                await write_market_catalog(
                    session,
                    merged_events,
                    merged_markets,
                    duration_seconds=duration,
                )
        except Exception as e:
            print(f"  Incremental catalog DB persist failed (non-fatal): {e}")

        self._last_catalog_refresh = now
        return {
            "mode": "incremental",
            "event_count": len(merged_events),
            "market_count": len(merged_markets),
            "delta_market_count": len(touched_market_ids),
            "delta_event_count": len(touched_event_keys),
            "duration_seconds": round(max(0.0, duration), 3),
        }

    async def _hydrate_catalog_from_db(self, *, only_if_newer: bool = False) -> int:
        """Restore market catalog from DB on startup.

        Populates _cached_events, _cached_markets, and derived caches so
        that scan_fast() can run immediately without waiting for the first
        HTTP catalog refresh.  Returns the number of markets loaded.
        """
        try:
            from models.database import AsyncSessionLocal
            from services.shared_state import read_market_catalog

            async with AsyncSessionLocal() as session:
                events, markets, metadata = await read_market_catalog(session)
        except Exception as e:
            print(f"  Catalog hydration from DB failed: {e}")
            return 0

        if not markets:
            return 0

        catalog_age = metadata.get("updated_at")
        if only_if_newer and self._cached_markets:
            if catalog_age is None:
                return 0
            if self._last_catalog_refresh is not None and catalog_age <= self._last_catalog_refresh:
                return 0

        now = datetime.now(timezone.utc)
        events, markets = self._prune_active_catalog(events, markets, now)
        events, markets = self._enforce_catalog_caps(events, markets)

        self._cached_events = events
        self._cached_markets = markets
        self._remember_market_tokens(markets)
        self._rebuild_realtime_graph(events, markets)
        self._trim_runtime_market_caches({str(getattr(m, "id", "") or "") for m in markets})

        if catalog_age:
            self._last_catalog_refresh = catalog_age
            # Also set _last_full_scan so scan_fast doesn't think it
            # has never done a full scan.
            if self._last_full_scan is None:
                self._last_full_scan = catalog_age

        print(
            f"  Hydrated catalog from DB: {len(events)} events, {len(markets)} markets"
            + (" (updated snapshot)" if only_if_newer else "")
        )
        return len(markets)

    async def scan_fast(
        self,
        reactive_token_ids: Optional[list[str]] = None,
        targeted_condition_ids: Optional[list] = None,
    ) -> list[Opportunity]:
        """Fast scan with reactive token batching + timed HOT-tier fallback.

        This lane only runs incremental scanner strategies. Full-snapshot
        strategies run in a separate heavy lane.
        """
        cycle_started = time.monotonic()
        self._fast_last_started_at = datetime.now(timezone.utc)
        self._fast_inflight = True
        self._fast_lane_error = None
        async with self._scan_lock:
            now = datetime.now(timezone.utc)
            reactive_tokens = [str(t or "").strip() for t in (reactive_token_ids or []) if str(t or "").strip()]
            reactive_mode = bool(reactive_tokens)
            mode_label = "reactive" if reactive_mode else "timer"
            print(f"[{now.isoformat()}] Starting fast scan ({mode_label})...")

            if not self._cached_markets:
                print("  Fast scan cache empty; attempting DB catalog hydration...")
                await self._hydrate_catalog_from_db()
                if not self._cached_markets:
                    print("  No catalog available yet (DB empty too); skipping scan cycle")
                    await self._set_activity("Waiting for catalog refresh...")
                    return self._opportunities

            await self._set_activity("Fast scan: preparing market batch...")

            try:
                new_markets: list = []
                if settings.INCREMENTAL_FETCH_ENABLED and not reactive_mode:
                    try:
                        new_markets = await self.market_data.get_recent_markets(since_minutes=5)
                        if new_markets:
                            print(f"  Incremental: {len(new_markets)} recently created markets")
                    except Exception as e:
                        print(f"  Incremental fetch failed (non-fatal): {e}")

                cached_market_ids = {m.id for m in self._cached_markets}
                truly_new = [m for m in new_markets if m.id not in cached_market_ids and self._is_market_active(m, now)]
                if truly_new:
                    self._cached_markets.extend(truly_new)
                    self._cached_events, self._cached_markets = self._prune_active_catalog(
                        self._cached_events,
                        self._cached_markets,
                        now,
                    )
                    self._cached_events, self._cached_markets = self._enforce_catalog_caps(
                        self._cached_events,
                        self._cached_markets,
                    )
                    print(f"  Added {len(truly_new)} brand-new markets to cache")
                    self._remember_market_tokens(self._cached_markets)
                    self._rebuild_realtime_graph(self._cached_events, self._cached_markets)
                    self._trim_runtime_market_caches({str(getattr(m, "id", "") or "") for m in self._cached_markets})

                loop = asyncio.get_running_loop()

                affected_market_ids: list[str] = []
                candidate_markets: list = []
                if reactive_mode:
                    affected_market_ids = self._resolve_affected_market_ids(reactive_tokens)
                    self._last_reactive_batch_markets = len(affected_market_ids)
                    candidate_markets = [
                        self._cached_market_by_id[mid]
                        for mid in affected_market_ids
                        if mid in self._cached_market_by_id
                    ]
                    candidate_markets = [m for m in candidate_markets if self._is_market_active(m, now)]
                    if not candidate_markets:
                        print(
                            "  Reactive batch had no currently cached/active markets; falling back to HOT-tier timer path"
                        )
                        reactive_mode = False

                if not reactive_mode:
                    self._last_reactive_batch_markets = 0

                    def _classify_cached(prioritizer, mkts, ts):
                        prioritizer.update_stability_scores()
                        return prioritizer.classify_all(mkts, ts)

                    tier_map = await loop.run_in_executor(
                        None, _classify_cached, self._prioritizer, self._cached_markets, now
                    )
                    candidate_markets = [m for m in tier_map[MarketTier.HOT] if self._is_market_active(m, now)]
                    affected_market_ids = [str(getattr(m, "id", "") or "") for m in candidate_markets]
                    if not candidate_markets:
                        print("  No HOT-tier markets, skipping fast scan")
                        self._opportunities = await self.refresh_opportunity_prices(
                            self._opportunities,
                            now=now,
                            drop_stale=False,
                        )
                        self._last_scan = now
                        self._last_fast_scan = now
                        return self._opportunities

                    if truly_new:
                        existing_candidate_ids = {str(getattr(m, "id", "") or "") for m in candidate_markets}
                        for market in truly_new:
                            market_id = str(getattr(market, "id", "") or "")
                            if not market_id or market_id in existing_candidate_ids:
                                continue
                            if not self._is_market_active(market, now):
                                continue
                            candidate_markets.append(market)
                            existing_candidate_ids.add(market_id)

                # If targeted condition IDs were requested (e.g. API evaluate
                # endpoint), narrow candidate markets to just those IDs.
                if targeted_condition_ids:
                    _target_set = {cid.lower() for cid in targeted_condition_ids}
                    candidate_markets = [
                        m
                        for m in self._cached_markets
                        if getattr(m, "condition_id", getattr(m, "id", "")).lower() in _target_set
                    ]
                    candidate_markets = [m for m in candidate_markets if self._is_market_active(m, now)]
                    affected_market_ids = [str(getattr(m, "id", "") or "") for m in candidate_markets]
                    print(f"  Targeted scan: narrowed to {len(candidate_markets)} markets")
                elif not reactive_mode:
                    timer_cap = max(10, int(settings.REALTIME_SCAN_MAX_BATCH_MARKETS or 800))
                    if len(candidate_markets) > timer_cap:
                        candidate_markets = sorted(candidate_markets, key=self._market_priority_key, reverse=True)[
                            :timer_cap
                        ]
                        affected_market_ids = [str(getattr(m, "id", "") or "") for m in candidate_markets]

                candidate_token_ids = self._collect_live_token_ids(candidate_markets)
                live_prices: dict[str, dict] = {}
                if reactive_mode:
                    ws_prices = await self._snapshot_ws_prices(candidate_token_ids)
                    live_prices.update(ws_prices)
                    if ws_prices:
                        print(f"  Reactive WS overlay: {len(ws_prices)}/{len(candidate_token_ids)} tokens")
                else:
                    token_sample = candidate_token_ids
                    if token_sample:
                        await self._set_activity(
                            f"Fast scan: reading live prices for {len(token_sample)} hot-tier tokens..."
                        )
                        live_prices = await self._snapshot_ws_prices(token_sample)
                        print(f"  Loaded prices for {len(live_prices)} hot-tier tokens from Redis cache")

                merged_prices = dict(live_prices)
                self._cached_prices.update(live_prices)
                self._apply_live_prices_to_markets(candidate_markets, merged_prices)
                self._update_market_price_history(candidate_markets, merged_prices, now)

                changed_markets = await loop.run_in_executor(
                    None, self._prioritizer.get_changed_markets, candidate_markets
                )
                if not changed_markets:
                    print(f"  All {len(candidate_markets)} candidate markets unchanged, skipping strategies")
                    await self._set_activity(f"Fast scan: {len(candidate_markets)} markets unchanged, skipping")
                    await loop.run_in_executor(None, self._prioritizer.update_after_evaluation, candidate_markets, now)
                    self._opportunities = await self.refresh_opportunity_prices(
                        self._opportunities,
                        now=now,
                        drop_stale=False,
                    )
                    self._last_scan = now
                    self._last_fast_scan = now
                    return self._opportunities

                if reactive_mode:
                    markets_for_strategies = candidate_markets
                    source = "scanner_fast_reactive"
                    scan_mode = "realtime_reactive"
                else:
                    markets_for_strategies = changed_markets
                    source = "scanner_fast_timer"
                    scan_mode = "fast_timer"

                markets_for_strategies = [m for m in markets_for_strategies if self._is_market_active(m, now)]
                changed_market_ids = [str(getattr(m, "id", "") or "") for m in changed_markets]
                affected_ids_payload = [str(getattr(m, "id", "") or "") for m in markets_for_strategies]

                print(
                    f"  Fast scan batch: {len(changed_market_ids)} changed / "
                    f"{len(affected_ids_payload)} dispatched markets ({scan_mode})"
                )

                await self._ensure_runtime_strategies_loaded()
                incremental_slugs, _ = self._partition_market_refresh_strategies()
                if not incremental_slugs:
                    print("  No incremental MARKET_DATA_REFRESH strategies enabled; skipping dispatch")
                    self._opportunities = await self.refresh_opportunity_prices(
                        self._opportunities,
                        now=now,
                        drop_stale=False,
                    )
                    self._last_scan = now
                    self._last_fast_scan = now
                    return self._opportunities

                await self._set_activity(
                    f"Fast scan: running strategies on {len(affected_ids_payload)} markets ({scan_mode})..."
                )

                fast_data_event = DataEvent(
                    event_type=EventType.MARKET_DATA_REFRESH,
                    source=source,
                    timestamp=utcnow(),
                    payload={
                        "scan_mode": scan_mode,
                        "strategy_batch": "incremental",
                        "changed_token_count": len(reactive_tokens) if reactive_mode else 0,
                        "changed_market_count": len(changed_market_ids),
                        "affected_market_count": len(affected_ids_payload),
                    },
                    markets=markets_for_strategies,
                    events=list(self._cached_events),
                    prices=dict(merged_prices),
                    scan_mode=scan_mode,
                    changed_token_ids=list(reactive_tokens) if reactive_mode else None,
                    changed_market_ids=changed_market_ids,
                    affected_market_ids=affected_ids_payload,
                )
                fast_opportunities = await self._dispatch_market_refresh(
                    fast_data_event,
                    incremental_slugs=incremental_slugs,
                    full_slugs=set(),
                    handler_timeout_seconds=self._fast_strategy_timeout_seconds(),
                )

                fast_opportunities = self._deduplicate_cross_strategy(fast_opportunities)
                fast_quality_reports: dict = {}
                fast_filtered: list = []
                for opp in fast_opportunities:
                    strategy_instance = strategy_loader.get_instance(opp.strategy)
                    overrides = (
                        getattr(strategy_instance, "quality_filter_overrides", None) if strategy_instance else None
                    )
                    report = quality_filter.evaluate_opportunity(opp, overrides=overrides)
                    fast_quality_reports[opp.stable_id or opp.id] = report
                    if report.passed:
                        fast_filtered.append(opp)
                fast_opportunities = fast_filtered
                self._quality_reports.update(fast_quality_reports)
                fast_opportunities.sort(key=lambda x: x.roi_percent, reverse=True)
                fast_opportunities = self._apply_opportunity_caps(fast_opportunities)

                def _update_prioritizer_state(prioritizer, mkts, ts):
                    unchanged_count = prioritizer.update_after_evaluation(mkts, ts)
                    prioritizer.compute_attention_scores(mkts)
                    return unchanged_count

                unchanged = await loop.run_in_executor(
                    None,
                    _update_prioritizer_state,
                    self._prioritizer,
                    candidate_markets,
                    now,
                )

                if fast_opportunities:
                    await self._attach_ai_judgments(fast_opportunities)
                    self._opportunities = self._merge_opportunities(fast_opportunities)

                self._opportunities = await self.refresh_opportunity_prices(
                    self._opportunities,
                    now=now,
                    drop_stale=False,
                )
                if self._opportunities:
                    self._opportunities.sort(key=lambda opp: opp.roi_percent, reverse=True)
                    self._opportunities = self._apply_opportunity_caps(self._opportunities)
                    self._queue_market_history_backfill(self._opportunities)

                self._last_scan = now
                self._last_fast_scan = now
                self._fast_scan_cycle += 1

                for callback in self._scan_callbacks:
                    try:
                        await callback(fast_opportunities)
                    except Exception as e:
                        print(f"  Callback error: {e}")

                print(
                    f"[{now.isoformat()}] Fast scan complete ({scan_mode}). "
                    f"{len(fast_opportunities)} detected, "
                    f"{len(self._opportunities)} total in pool "
                    f"({unchanged} unchanged markets skipped)"
                )
                await self._set_activity(
                    f"Fast scan complete — {len(fast_opportunities)} found, {len(self._opportunities)} total"
                )
                return self._opportunities

            except Exception as e:
                self._fast_lane_error = str(e)
                print(f"[{utcnow().isoformat()}] Fast scan error: {e}")
                await self._set_activity(f"Fast scan error: {e}")
                raise
            finally:
                self._fast_inflight = False
                self._fast_last_completed_at = datetime.now(timezone.utc)
                self._last_fast_scan_duration_seconds = round(max(0.0, time.monotonic() - cycle_started), 3)

    async def scan_full_snapshot_strategies(
        self,
        *,
        reason: str = "scheduled",
        targeted_condition_ids: Optional[list[str]] = None,
        force: bool = False,
    ) -> list[Opportunity]:
        """Run only full-snapshot MARKET_DATA_REFRESH strategies on a bounded market set."""
        cycle_started = time.monotonic()
        async with self._full_snapshot_lane_lock:
            now = datetime.now(timezone.utc)

            def _clone_model(value: object) -> object:
                if hasattr(value, "model_copy"):
                    try:
                        return value.model_copy(deep=True)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                return copy.deepcopy(value)

            async with self._scan_lock:
                if not force and not targeted_condition_ids and not self._full_snapshot_strategy_due(now):
                    return self._opportunities

                if not self._cached_markets:
                    await self._hydrate_catalog_from_db()
                    if not self._cached_markets:
                        return self._opportunities

                self._heavy_last_started_at = datetime.now(timezone.utc)
                self._heavy_inflight = True
                self._heavy_lane_error = None
                self._full_snapshot_strategy_running = True
                self._last_full_snapshot_strategy_error = None
                cached_events_snapshot = [_clone_model(event) for event in self._cached_events]
                cached_markets_snapshot = [_clone_model(market) for market in self._cached_markets]

            try:
                await self._ensure_runtime_strategies_loaded()
                _, full_slugs = self._partition_market_refresh_strategies()
                async with self._scan_lock:
                    self._last_full_snapshot_strategy_count = len(full_slugs)
                if not full_slugs:
                    async with self._scan_lock:
                        return self._opportunities

                if targeted_condition_ids:
                    target_set = {
                        str(cid or "").strip().lower() for cid in targeted_condition_ids if str(cid or "").strip()
                    }
                    full_snapshot_markets = [
                        market
                        for market in cached_markets_snapshot
                        if str(getattr(market, "condition_id", getattr(market, "id", "")) or "").lower() in target_set
                        and self._is_market_active(market, now)
                    ]
                else:
                    if bool(getattr(settings, "SCANNER_FORCE_FULL_UNIVERSE", True)):
                        cap = 0
                    else:
                        cap = int(getattr(settings, "SCANNER_FULL_SNAPSHOT_MAX_MARKETS", 0) or 0)
                    active_markets = [market for market in cached_markets_snapshot if self._is_market_active(market, now)]
                    if "tail_end_carry" in full_slugs:
                        tail_priority = [
                            market for market in active_markets if self._is_tail_end_priority_market(market, now)
                        ]
                        tail_priority.sort(key=lambda market: self._tail_end_priority_key(market, now), reverse=True)
                        if cap > 0 and len(tail_priority) >= cap:
                            full_snapshot_markets = tail_priority[:cap]
                        else:
                            seen_ids = {str(getattr(market, "id", "") or "") for market in tail_priority}
                            remaining = [
                                market
                                for market in sorted(active_markets, key=self._market_priority_key, reverse=True)
                                if str(getattr(market, "id", "") or "") not in seen_ids
                            ]
                            full_snapshot_markets = tail_priority + remaining
                            if cap > 0 and len(full_snapshot_markets) > cap:
                                full_snapshot_markets = full_snapshot_markets[:cap]
                    else:
                        full_snapshot_markets = sorted(active_markets, key=self._market_priority_key, reverse=True)
                        if cap > 0 and len(full_snapshot_markets) > cap:
                            full_snapshot_markets = full_snapshot_markets[:cap]

                if not full_snapshot_markets:
                    async with self._scan_lock:
                        self._last_full_snapshot_strategy_market_count = 0
                        self._last_full_snapshot_chunk_market_count = 0
                        self._last_full_snapshot_strategy_opportunity_count = 0
                        return self._opportunities

                universe_markets = full_snapshot_markets
                universe_count = len(universe_markets)
                chunk_size = max(1, int(getattr(settings, "SCANNER_FULL_SNAPSHOT_CHUNK_SIZE", 300) or 300))
                targeted_mode = bool(targeted_condition_ids)
                chunk_start = 0
                chunk_end = universe_count
                chunk_markets = universe_markets
                next_cursor_index = 0
                cycle_completed = True
                cycle_started_at = self._full_snapshot_cycle_started_at or now
                processed_markets = universe_count

                if not targeted_mode:
                    cursor_start = max(0, int(self._full_snapshot_cursor_index or 0))
                    if cursor_start >= universe_count or self._full_snapshot_cycle_total_markets != universe_count:
                        cursor_start = 0
                        cycle_started_at = now
                    elif self._full_snapshot_cycle_started_at is None:
                        cycle_started_at = now

                    chunk_start = cursor_start
                    chunk_end = min(universe_count, cursor_start + chunk_size)
                    chunk_markets = universe_markets[chunk_start:chunk_end]
                    cycle_completed = chunk_end >= universe_count
                    next_cursor_index = 0 if cycle_completed else chunk_end
                    processed_markets = universe_count if cycle_completed else chunk_end

                async with self._scan_lock:
                    self._last_full_snapshot_strategy_market_count = universe_count
                    self._last_full_snapshot_chunk_market_count = len(chunk_markets)

                await self._set_activity(
                    (
                        f"Heavy lane: running full-snapshot chunk "
                        f"{chunk_start + 1}-{chunk_end}/{universe_count} "
                        f"({len(chunk_markets)} markets)..."
                    )
                )

                token_ids = self._collect_live_token_ids(chunk_markets)
                full_snapshot_prices: dict[str, dict] = {}
                if token_ids:
                    full_snapshot_prices = await self._snapshot_ws_prices(token_ids)
                self._apply_live_prices_to_markets(chunk_markets, full_snapshot_prices)

                market_ids = [str(getattr(market, "id", "") or "") for market in chunk_markets]
                full_event = DataEvent(
                    event_type=EventType.MARKET_DATA_REFRESH,
                    source="scanner_full_snapshot",
                    timestamp=utcnow(),
                    payload={
                        "scan_mode": "full_snapshot_heavy",
                        "strategy_batch": "full_snapshot",
                        "reason": reason,
                        "targeted": targeted_mode,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "chunk_size": len(chunk_markets),
                        "total_market_count": universe_count,
                        "affected_market_count": len(market_ids),
                    },
                    markets=chunk_markets,
                    events=cached_events_snapshot,
                    prices=dict(full_snapshot_prices),
                    scan_mode="full_snapshot_heavy",
                    changed_market_ids=market_ids,
                    affected_market_ids=market_ids,
                )

                full_opportunities = await self._dispatch_market_refresh(
                    full_event,
                    incremental_slugs=set(),
                    full_slugs=full_slugs,
                    full_market_snapshot=chunk_markets,
                    full_prices=full_snapshot_prices,
                    handler_timeout_seconds=self._full_snapshot_strategy_timeout_seconds(),
                )
                full_opportunities = self._deduplicate_cross_strategy(full_opportunities)

                full_quality_reports: dict = {}
                full_filtered: list[Opportunity] = []
                for opp in full_opportunities:
                    strategy_instance = strategy_loader.get_instance(opp.strategy)
                    overrides = (
                        getattr(strategy_instance, "quality_filter_overrides", None) if strategy_instance else None
                    )
                    report = quality_filter.evaluate_opportunity(opp, overrides=overrides)
                    full_quality_reports[opp.stable_id or opp.id] = report
                    if report.passed:
                        full_filtered.append(opp)

                full_filtered.sort(key=lambda item: item.roi_percent, reverse=True)
                full_filtered = self._apply_opportunity_caps(full_filtered)

                if full_filtered:
                    await self._attach_ai_judgments(full_filtered)

                async with self._scan_lock:
                    self._cached_prices.update(full_snapshot_prices)
                    self._update_market_price_history(chunk_markets, full_snapshot_prices, now)
                    self._quality_reports.update(full_quality_reports)
                    self._last_full_snapshot_strategy_opportunity_count = len(full_filtered)
                    self._full_snapshot_cycle_total_markets = universe_count
                    self._full_snapshot_cycle_processed_markets = processed_markets
                    self._full_snapshot_cursor_index = next_cursor_index
                    if chunk_start == 0:
                        self._full_snapshot_cycle_started_at = cycle_started_at
                    elif self._full_snapshot_cycle_started_at is None:
                        self._full_snapshot_cycle_started_at = cycle_started_at
                    if cycle_completed:
                        self._full_snapshot_cycle_completed_at = now
                    elif chunk_start == 0:
                        self._full_snapshot_cycle_completed_at = None
                    if full_filtered:
                        self._opportunities = self._merge_opportunities(full_filtered)
                    opportunities_snapshot = [_clone_model(opp) for opp in self._opportunities]

                refreshed_opportunities = await self.refresh_opportunity_prices(
                    opportunities_snapshot,
                    now=now,
                    drop_stale=False,
                )
                if refreshed_opportunities:
                    refreshed_opportunities.sort(key=lambda opp: opp.roi_percent, reverse=True)
                    refreshed_opportunities = self._apply_opportunity_caps(refreshed_opportunities)

                async with self._scan_lock:
                    self._opportunities = refreshed_opportunities
                    if self._opportunities:
                        self._queue_market_history_backfill(self._opportunities)
                    self._last_scan = now
                    self._last_full_snapshot_strategy_scan = now

                for callback in self._scan_callbacks:
                    try:
                        await callback(full_filtered)
                    except Exception as e:
                        print(f"  Callback error: {e}")

                cycle_suffix = " full coverage cycle complete." if cycle_completed else ""
                await self._set_activity(
                    f"Heavy lane complete — {len(full_filtered)} opportunities "
                    f"(chunk {chunk_start + 1}-{chunk_end}/{universe_count}).{cycle_suffix}"
                )
                async with self._scan_lock:
                    return self._opportunities
            except Exception as exc:
                self._heavy_lane_error = str(exc)
                async with self._scan_lock:
                    self._last_full_snapshot_strategy_error = str(exc)
                await self._set_activity(f"Heavy lane error: {exc}")
                raise
            finally:
                async with self._scan_lock:
                    self._full_snapshot_strategy_running = False
                    self._heavy_inflight = False
                    self._heavy_last_completed_at = datetime.now(timezone.utc)
                    self._last_full_snapshot_strategy_duration_seconds = round(
                        max(0.0, time.monotonic() - cycle_started),
                        3,
                    )

    async def _prefetch_news_matches(self, events, markets, prices):
        """Pre-fetch news articles and run semantic matching (no LLM calls).

        This prepares the data so that manual edge analysis from the UI
        is fast — articles are already fetched and matched to markets.
        No paid LLM calls are made here.
        """
        try:
            from services.news.feed_service import news_feed_service
            from services.news.semantic_matcher import semantic_matcher

            # Step 1: Fetch articles (free — RSS/GDELT)
            await news_feed_service.fetch_all()
            all_articles = news_feed_service.get_articles(max_age_hours=settings.NEWS_ARTICLE_TTL_HOURS)

            if not all_articles:
                return

            # Step 2: Build market index
            news_edge = self._get_news_edge_helper()
            if news_edge is None:
                return
            market_infos = news_edge._build_market_infos(events, markets, prices)
            if not market_infos:
                return

            loop = asyncio.get_running_loop()
            executor = _NEWS_PREFETCH_EXECUTOR

            if not semantic_matcher._initialized:
                await loop.run_in_executor(executor, semantic_matcher.initialize)

            await loop.run_in_executor(executor, semantic_matcher.update_market_index, market_infos)

            # Step 3: Embed articles (local ML, free)
            await loop.run_in_executor(executor, semantic_matcher.embed_articles, all_articles)

            # Step 4: Match articles to markets (local, free)
            matches = await loop.run_in_executor(
                executor,
                semantic_matcher.match_articles_to_markets,
                all_articles,
                3,
                settings.NEWS_SIMILARITY_THRESHOLD,
            )

            print(
                f"  News prefetch: {len(all_articles)} articles, "
                f"{len(market_infos)} markets, {len(matches)} matches "
                f"(LLM analysis deferred to manual trigger)"
            )
        except Exception as e:
            print(f"  News prefetch error: {e}")

    def _deduplicate_cross_strategy(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        """Remove duplicate opportunities that cover the same underlying markets.

        When multiple strategies detect the same set of markets, keep only the one
        with the highest ROI — unless a strategy has set allow_deduplication=False,
        in which case its opportunities are always preserved.
        """
        # Separate protected opportunities (allow_deduplication=False)
        protected = []
        candidates = []
        for opp in opportunities:
            strategy_instance = strategy_loader.get_instance(opp.strategy)
            if strategy_instance and not getattr(strategy_instance, "allow_deduplication", True):
                protected.append(opp)
            else:
                candidates.append(opp)

        seen: dict[str, Opportunity] = {}
        for opp in candidates:
            market_ids = sorted(m.get("id", "") for m in opp.markets)
            fingerprint = "|".join(market_ids)
            if fingerprint in seen:
                if opp.roi_percent > seen[fingerprint].roi_percent:
                    seen[fingerprint] = opp
            else:
                seen[fingerprint] = opp

        deduped = list(seen.values()) + protected
        removed = len(opportunities) - len(deduped)
        if removed > 0:
            print(f"  Deduplicated: removed {removed} cross-strategy duplicates")
        return deduped

    @staticmethod
    def _coerce_cap_value(raw_value: object) -> Optional[int]:
        if raw_value is None:
            return None
        if isinstance(raw_value, bool):
            return None
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value

    @staticmethod
    def _strategy_cap_from_instance(instance: object, fallback: int) -> int:
        if instance is None:
            return fallback

        config = getattr(instance, "config", None)
        candidates = []
        if isinstance(config, dict):
            candidates.append(config.get("retention_max_opportunities"))
            candidates.append(config.get("max_opportunities_per_strategy"))
            candidates.append(config.get("max_opportunities"))
        candidates.append(getattr(instance, "retention_max_opportunities", None))
        candidates.append(getattr(instance, "max_opportunities_per_strategy", None))
        candidates.append(getattr(instance, "max_opportunities", None))

        for candidate in candidates:
            cap_value = ArbitrageScanner._coerce_cap_value(candidate)
            if cap_value is not None:
                return cap_value
        return fallback

    @staticmethod
    def _coerce_retention_minutes(raw_value: object) -> Optional[int]:
        parsed = StrategySDK.parse_duration_minutes(raw_value)
        if parsed is None:
            return None
        return max(0, min(int(parsed), 60 * 24 * 90))

    @staticmethod
    def _strategy_ttl_from_instance(instance: object, fallback: int) -> int:
        if instance is None:
            return fallback

        config = getattr(instance, "config", None)
        candidates: list[object] = []
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
            ]
        )

        for candidate in candidates:
            ttl = ArbitrageScanner._coerce_retention_minutes(candidate)
            if ttl is not None:
                return ttl
        return fallback

    def _strategy_cap_for_key(self, strategy_key: str, fallback: int) -> int:
        key = str(strategy_key or "").strip().lower()
        if not key:
            return fallback
        return self._strategy_cap_from_instance(strategy_loader.get_instance(key), fallback)

    def _strategy_ttl_for_key(self, strategy_key: str, fallback: int) -> int:
        key = str(strategy_key or "").strip().lower()
        if not key:
            return fallback
        return self._strategy_ttl_from_instance(strategy_loader.get_instance(key), fallback)

    def _apply_opportunity_caps(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        """Limit pool size globally and per strategy to prevent strategy flood."""
        total_cap = int(getattr(settings, "SCANNER_MAX_OPPORTUNITIES_TOTAL", 0) or 0)
        per_strategy_cap = int(getattr(settings, "SCANNER_MAX_OPPORTUNITIES_PER_STRATEGY", 0) or 0)
        kept: list[Opportunity] = []
        by_strategy: dict[str, int] = {}
        resolved_caps: dict[str, int] = {}

        for opp in opportunities:
            strategy = str(getattr(opp, "strategy", "") or "unknown")
            strategy_cap = resolved_caps.get(strategy)
            if strategy_cap is None:
                strategy_cap = self._strategy_cap_for_key(strategy, per_strategy_cap)
                resolved_caps[strategy] = strategy_cap
            if strategy_cap > 0 and by_strategy.get(strategy, 0) >= strategy_cap:
                continue
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
            kept.append(opp)
            if total_cap > 0 and len(kept) >= total_cap:
                break

        removed = len(opportunities) - len(kept)
        if removed > 0:
            print(
                "  Opportunity caps applied: "
                f"{len(kept)} kept, {removed} removed "
                f"(total_cap={total_cap or 'off'}, per_strategy_cap={per_strategy_cap or 'off'})"
            )
        return kept

    def _merge_opportunities(self, new_opportunities: list[Opportunity]) -> list[Opportunity]:
        """Merge newly detected opportunities into the existing pool.

        Instead of replacing all opportunities on each scan, this method:
        - Adds newly discovered opportunities to the pool
        - Updates existing opportunities (matched by stable_id) with fresh
          market data while preserving original detection time and AI analysis
        - Removes expired opportunities whose resolution date has passed
        """
        now = datetime.now(timezone.utc)

        # Index existing opportunities by stable_id
        existing_map: dict[str, Opportunity] = {opp.stable_id: opp for opp in self._opportunities}

        new_count = 0
        updated_count = 0

        for new_opp in new_opportunities:
            new_opp.last_seen_at = now
            new_opp.last_detected_at = now
            existing = existing_map.get(new_opp.stable_id)
            if existing:
                # Preserve immutable first-detection time and ID while updating recency.
                preserved_first = _make_aware(getattr(existing, "first_detected_at", None) or existing.detected_at) or now
                new_opp.first_detected_at = preserved_first
                new_opp.detected_at = preserved_first
                new_opp.id = existing.id
                if getattr(existing, "last_priced_at", None) and not getattr(new_opp, "last_priced_at", None):
                    new_opp.last_priced_at = existing.last_priced_at
                if not new_opp.markets and existing.markets:
                    new_opp.markets = [dict(m) if isinstance(m, dict) else m for m in existing.markets]
                elif new_opp.markets and existing.markets:
                    previous_by_id: dict[str, dict] = {}
                    for market in existing.markets:
                        if not isinstance(market, dict):
                            continue
                        for market_id in self._market_history_lookup_ids(market):
                            if market_id and market_id not in previous_by_id:
                                previous_by_id[market_id] = market
                    merged_markets: list[dict] = []
                    for market in new_opp.markets:
                        if not isinstance(market, dict):
                            continue
                        previous = None
                        for market_id in self._market_history_lookup_ids(market):
                            candidate = previous_by_id.get(market_id)
                            if candidate is not None:
                                previous = candidate
                                break
                        if previous is None:
                            merged_markets.append(market)
                            continue
                        merged_market = dict(previous)
                        merged_market.update(market)
                        previous_history = previous.get("price_history")
                        current_history = market.get("price_history")
                        if (
                            (not isinstance(current_history, list) or len(current_history) < 2)
                            and isinstance(previous_history, list)
                            and len(previous_history) >= 2
                        ):
                            merged_market["price_history"] = previous_history
                        merged_markets.append(merged_market)
                    if merged_markets:
                        new_opp.markets = merged_markets
                if not new_opp.positions_to_take and existing.positions_to_take:
                    new_opp.positions_to_take = [
                        dict(position) if isinstance(position, dict) else position
                        for position in existing.positions_to_take
                    ]
                if existing.event_id and not new_opp.event_id:
                    new_opp.event_id = existing.event_id
                if existing.event_slug and not new_opp.event_slug:
                    new_opp.event_slug = existing.event_slug
                if existing.event_title and not new_opp.event_title:
                    new_opp.event_title = existing.event_title
                if existing.category and not new_opp.category:
                    new_opp.category = existing.category
                # Preserve AI analysis if not freshly attached from DB
                if existing.ai_analysis and not new_opp.ai_analysis:
                    new_opp.ai_analysis = existing.ai_analysis
                updated_count += 1
            else:
                first_detected = _make_aware(getattr(new_opp, "first_detected_at", None) or new_opp.detected_at) or now
                new_opp.first_detected_at = first_detected
                new_opp.detected_at = first_detected
                new_count += 1
            existing_map[new_opp.stable_id] = new_opp

        # Remove expired/stale opportunities
        def _is_stale(opp: Opportunity) -> bool:
            if opp.last_seen_at is None:
                return False
            fallback_ttl = max(5, int(getattr(settings, "SCANNER_STALE_OPPORTUNITY_MINUTES", 45) or 45))
            ttl = self._strategy_ttl_for_key(opp.strategy, fallback_ttl)
            if ttl <= 0:
                return False
            cutoff = now - timedelta(minutes=ttl)
            return _make_aware(opp.last_seen_at) < cutoff

        merged = [
            opp
            for opp in existing_map.values()
            if (opp.resolution_date is None or _make_aware(opp.resolution_date) > now) and not _is_stale(opp)
        ]

        expired_count = len(existing_map) - len(merged)

        # Sort by ROI
        merged.sort(key=lambda x: x.roi_percent, reverse=True)
        merged = self._apply_opportunity_caps(merged)

        retained = len(merged) - new_count - updated_count
        if retained < 0:
            retained = 0
        parts = []
        if new_count:
            parts.append(f"{new_count} new")
        if updated_count:
            parts.append(f"{updated_count} updated")
        if retained:
            parts.append(f"{retained} retained from prior scans")
        if expired_count:
            parts.append(f"{expired_count} expired removed")
        if parts:
            print(f"  Merge: {', '.join(parts)} -> {len(merged)} total")

        return merged

    # Maximum number of opportunities to score per scan cycle
    AI_SCORE_MAX_PER_SCAN = 50
    # How many LLM calls can run concurrently
    AI_SCORE_CONCURRENCY = 3
    # Don't re-score an opportunity within this many seconds
    AI_SCORE_CACHE_TTL_SECONDS = 300  # 5 minutes

    async def _ai_score_opportunities(self, opportunities: list):
        """Score unscored opportunities using AI (runs in background).

        Judgments are persisted in the OpportunityJudgment DB table (by
        the judge itself) and looked up from there on subsequent scans.

        Cost controls:
        - Limits to AI_SCORE_MAX_PER_SCAN per scan cycle
        - Caps concurrency via AI_SCORE_CONCURRENCY semaphore
        - Skips opportunities already judged within AI_SCORE_CACHE_TTL_SECONDS (DB lookup)
        - Respects cancellation (e.g. on pause) between each scoring call
        """
        try:
            from services.ai.opportunity_judge import opportunity_judge
            import services.shared_state as scanner_shared_state

            # Filter: only unscored (DB dedup already attached scored ones)
            candidates = [o for o in opportunities if o.ai_analysis is None]

            if not candidates:
                return

            # Prioritise by ROI descending — score the best opportunities first
            candidates.sort(key=lambda x: x.roi_percent, reverse=True)
            # Cap the number of LLM calls per scan cycle
            candidates = candidates[: self.AI_SCORE_MAX_PER_SCAN]

            print(f"  AI Judge: scoring {len(candidates)} unscored opportunities...")

            sem = asyncio.Semaphore(self.AI_SCORE_CONCURRENCY)
            persist_lock = asyncio.Lock()

            async def _persist_inline_analysis(opp: Opportunity) -> None:
                if not opp.ai_analysis:
                    return
                # Serialize snapshot patch writes so concurrent scorers don't
                # overwrite each other's updates.
                async with persist_lock:
                    async with AsyncSessionLocal() as session:
                        await scanner_shared_state.update_opportunity_ai_analysis_in_snapshot(
                            session=session,
                            opportunity_id=opp.id,
                            stable_id=opp.stable_id,
                            ai_analysis=opp.ai_analysis.model_dump(mode="json"),
                        )

            async def _score_one(opp):
                async with sem:
                    result = await opportunity_judge.judge_opportunity(opp)
                    opp.ai_analysis = AIAnalysis(
                        overall_score=result.get("overall_score", 0.0),
                        profit_viability=result.get("profit_viability", 0.0),
                        resolution_safety=result.get("resolution_safety", 0.0),
                        execution_feasibility=result.get("execution_feasibility", 0.0),
                        market_efficiency=result.get("market_efficiency", 0.0),
                        recommendation=result.get("recommendation", "review"),
                        reasoning=result.get("reasoning"),
                        risk_factors=result.get("risk_factors", []),
                        judged_at=datetime.now(timezone.utc),
                    )
                    try:
                        await _persist_inline_analysis(opp)
                    except Exception as e:
                        print(f"  AI Judge persist warning: {e}")
                    print(
                        f"  AI Judge: {opp.title[:50]}... "
                        f"-> {result.get('recommendation', 'unknown')} "
                        f"(score: {result.get('overall_score', 0):.2f})"
                    )

            tasks = [asyncio.create_task(_score_one(opp)) for opp in candidates]

            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    for t in tasks:
                        t.cancel()
                    raise
                except Exception as e:
                    print(f"  AI Judge error: {e}")

        except asyncio.CancelledError:
            print("  AI scoring cancelled")
            raise
        except Exception as e:
            print(f"  AI scoring error: {e}")

    def _register_reactive_scanning(self):
        """Register reactive token queueing from price-change signals."""
        if self._reactive_scan_registered:
            return
        if not settings.WS_FEED_ENABLED:
            return
        try:

            async def _on_price_change(event: DataEvent) -> list:
                token = str(event.token_id or "").strip()
                if token:
                    await self._queue_reactive_tokens({token})
                return []

            # Reactive queueing is sourced from cross-process PRICE_CHANGE fanout
            # (crypto worker owns WS ingestion and publishes via event dispatcher).
            event_dispatcher.subscribe("__scanner_reactive__", EventType.PRICE_CHANGE, _on_price_change)

            # Keep local callback wiring when this process also owns WS feeds.
            feed_mgr = get_feed_manager()
            if feed_mgr._started:

                async def _trigger_reactive(changed_tokens: Set[str]):
                    await self._queue_reactive_tokens(changed_tokens)

                feed_mgr.set_reactive_scan_callback(
                    _trigger_reactive,
                    debounce_seconds=float(settings.REALTIME_SCAN_DEBOUNCE_SECONDS),
                )

            self._reactive_scan_registered = True
            print("  Reactive scanning registered (PRICE_CHANGE + local WS callbacks)")
        except Exception as e:
            print(f"  Reactive scanning registration failed: {e}")

    async def _scan_loop(self):
        """Internal scan loop with reactive + tiered polling.

        Catalog refresh runs as a separate background task.  This loop
        always uses scan_fast() which reads from the cached catalog +
        Redis live prices — it never calls upstream HTTP APIs directly.
        """
        # Hydrate catalog from DB so scan_fast works immediately
        await self._hydrate_catalog_from_db()

        # Start background catalog refresh
        catalog_task = asyncio.create_task(self._catalog_refresh_background())

        while self._running:
            if not self._enabled:
                await asyncio.sleep(self._interval_seconds)
                continue

            # Register reactive scanning on first enabled iteration
            self._register_reactive_scanning()

            try:
                reactive_tokens = await self.consume_reactive_tokens()
                await self.scan_fast(reactive_token_ids=reactive_tokens)
            except Exception as e:
                print(f"Scan error: {e}")

            # Wait for either the timer OR a reactive price-change trigger
            reactive_trigger = self._get_reactive_trigger()
            if self._pending_reactive_tokens:
                reactive_trigger.set()
            else:
                reactive_trigger.clear()
            sleep_seconds = settings.FAST_SCAN_INTERVAL_SECONDS
            await self._set_activity(f"Idle — next scan in ≤{sleep_seconds}s (or on price change)")
            try:
                await asyncio.wait_for(reactive_trigger.wait(), timeout=sleep_seconds)
                await self._set_activity("Reactive scan triggered by WS price change")
            except asyncio.TimeoutError:
                pass  # Normal timer-based fallback

        # Clean up background catalog task
        if catalog_task and not catalog_task.done():
            catalog_task.cancel()

    async def _catalog_refresh_background(self):
        """Background loop that refreshes market catalog independently."""
        next_full_reconcile_at: datetime | None = None
        while self._running:
            interval = max(60, settings.FULL_SCAN_INTERVAL_SECONDS)
            full_reconcile_interval = max(
                60,
                int(getattr(settings, "MARKET_UNIVERSE_FULL_RECONCILE_INTERVAL_SECONDS", 900) or 900),
            )
            incremental_timeout = max(
                15,
                int(getattr(settings, "MARKET_UNIVERSE_INCREMENTAL_TIMEOUT_SECONDS", 120) or 120),
            )
            full_timeout = max(
                30,
                int(getattr(settings, "MARKET_UNIVERSE_REFRESH_TIMEOUT_SECONDS", 300) or 300),
            )
            now = datetime.now(timezone.utc)
            force_full = (
                not bool(getattr(settings, "MARKET_UNIVERSE_INCREMENTAL_ENABLED", True))
                or next_full_reconcile_at is None
                or now >= next_full_reconcile_at
            )
            timeout = float(full_timeout if force_full else incremental_timeout)
            try:
                result = await asyncio.wait_for(
                    self.refresh_catalog_incremental(force_full=force_full),
                    timeout=timeout,
                )
                mode = str(result.get("mode") or ("full" if force_full else "incremental"))
                if mode == "full":
                    next_full_reconcile_at = now + timedelta(seconds=full_reconcile_interval)
            except asyncio.TimeoutError:
                print(f"  Catalog refresh timed out after {timeout:.0f}s")
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"  Catalog refresh failed: {e}")
            await asyncio.sleep(interval)

    async def _attach_ai_judgments(self, opportunities: list):
        """Attach existing AI judgments from the DB to opportunity objects.

        Performs a single batch query for latest judgments and matches them
        to opportunities by stable_id. This treats DB-persisted judgments as
        durable state (survives worker restarts and scan cycles).
        """
        if not opportunities:
            return

        try:
            from sqlalchemy import func

            async with AsyncSessionLocal() as session:
                # Get the most recent judgment per opportunity_id.
                subq = (
                    select(
                        OpportunityJudgment.opportunity_id,
                        func.max(OpportunityJudgment.judged_at).label("latest"),
                    )
                    .group_by(OpportunityJudgment.opportunity_id)
                    .subquery()
                )
                rows = (
                    (
                        await session.execute(
                            select(OpportunityJudgment).join(
                                subq,
                                (OpportunityJudgment.opportunity_id == subq.c.opportunity_id)
                                & (OpportunityJudgment.judged_at == subq.c.latest),
                            )
                        )
                    )
                    .scalars()
                    .all()
                )

            # Build stable_id -> AIAnalysis lookup
            judgment_map: dict[str, AIAnalysis] = {}
            for row in rows:
                opp_id = row.opportunity_id or ""
                # Convert opportunity_id to stable_id by stripping trailing _<timestamp>
                parts = opp_id.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    stable_id = parts[0]
                else:
                    stable_id = opp_id

                judgment_map[stable_id] = AIAnalysis(
                    overall_score=row.overall_score or 0.0,
                    profit_viability=row.profit_viability or 0.0,
                    resolution_safety=row.resolution_safety or 0.0,
                    execution_feasibility=row.execution_feasibility or 0.0,
                    market_efficiency=row.market_efficiency or 0.0,
                    recommendation=row.recommendation or "review",
                    reasoning=row.reasoning,
                    risk_factors=row.risk_factors or [],
                    judged_at=row.judged_at,
                )

            # Attach to matching opportunities
            attached = 0
            for opp in opportunities:
                analysis = judgment_map.get(opp.stable_id)
                if analysis:
                    opp.ai_analysis = analysis
                    attached += 1

            if attached:
                print(f"  Attached {attached} existing AI judgments from DB")

        except Exception as e:
            print(f"  Error loading AI judgments from DB: {e}")

    async def start_continuous_scan(self, interval_seconds: int = None):
        """Start continuous scanning loop"""
        # Load persisted settings first
        await self.load_settings()

        # Load strategy plugins from database
        await self.load_plugins()

        if interval_seconds is not None:
            self._interval_seconds = interval_seconds

        self._running = True
        print(f"Starting continuous scan (interval: {self._interval_seconds}s, enabled: {self._enabled})")

        # Run the scan loop
        await self._scan_loop()

    async def start(self):
        """Enable scanning and resume all background services"""
        self._enabled = True
        global_pause_state.resume()
        await self.save_settings()
        await self._notify_status_change()

        # Kick off an immediate catalog refresh + scan in the background so
        # this method returns quickly and doesn't block the API response.
        if self._running:
            asyncio.create_task(self.refresh_catalog())
            asyncio.create_task(self.scan_fast())

    async def pause(self):
        """Pause all background services (scanner, trader orchestrator, wallet tracker, discovery, etc.)."""
        self._enabled = False
        global_pause_state.pause()
        # Cancel any in-flight AI scoring task to stop incurring API costs
        await self._cancel_ai_scoring()
        await self.save_settings()
        await self._notify_status_change()

    async def stop(self):
        """Stop continuous scanning loop completely"""
        self._running = False
        self._enabled = False
        await self._cancel_ai_scoring()

    async def _cancel_ai_scoring(self):
        """Cancel any running AI scoring background task."""
        if self._ai_scoring_task and not self._ai_scoring_task.done():
            self._ai_scoring_task.cancel()
            try:
                await self._ai_scoring_task
            except (asyncio.CancelledError, Exception):
                pass
            print("  AI scoring task cancelled")

    async def set_interval(self, seconds: int):
        """Update scan interval"""
        if seconds < 10:
            seconds = 10  # Minimum 10 seconds
        if seconds > 3600:
            seconds = 3600  # Maximum 1 hour

        self._interval_seconds = seconds
        await self.save_settings()
        await self._notify_status_change()

    def get_status(self) -> dict:
        """Get current scanner status"""
        strategy_rows = self._strategy_runtime_status_rows()
        now = datetime.now(timezone.utc)

        def _age_seconds(dt: Optional[datetime]) -> Optional[float]:
            aware = _make_aware(dt)
            if aware is None:
                return None
            return max(0.0, (now - aware).total_seconds())

        def _p95(values: list[float]) -> Optional[float]:
            if not values:
                return None
            ordered = sorted(values)
            index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
            return round(float(ordered[index]), 3)

        last_fast_scan_age_seconds = _age_seconds(self._last_fast_scan)
        opportunity_price_ages = [
            age
            for age in (
                _age_seconds(_make_aware(getattr(opp, "last_priced_at", None) or opp.detected_at))
                for opp in self._opportunities
            )
            if age is not None
        ]
        opportunity_detected_ages = [
            age
            for age in (
                _age_seconds(_make_aware(getattr(opp, "last_detected_at", None) or opp.detected_at))
                for opp in self._opportunities
            )
            if age is not None
        ]

        fast_watchdog_seconds = max(
            30,
            int(getattr(settings, "SCAN_WATCHDOG_SECONDS", 600) or 600),
            int(getattr(settings, "FAST_SCAN_INTERVAL_SECONDS", 15) or 15) * 3,
        )
        heavy_watchdog_seconds = max(
            30,
            int(getattr(settings, "SCANNER_FULL_SNAPSHOT_WATCHDOG_SECONDS", 180) or 180),
        )
        lane_watchdogs = {
            "fast": self._lane_watchdog_payload(
                now=now,
                started_at=self._fast_last_started_at,
                inflight=self._fast_inflight,
                threshold_seconds=fast_watchdog_seconds,
            ),
            "heavy": self._lane_watchdog_payload(
                now=now,
                started_at=self._heavy_last_started_at,
                inflight=self._heavy_inflight,
                threshold_seconds=heavy_watchdog_seconds,
            ),
        }
        status = {
            "running": self._running,
            "enabled": self._enabled,
            "interval_seconds": self._interval_seconds,
            "auto_ai_scoring": self._auto_ai_scoring,
            "last_scan": to_iso(self._last_scan),
            "last_fast_scan": to_iso(self._last_fast_scan),
            "last_heavy_scan": to_iso(self._last_full_snapshot_strategy_scan),
            "last_fast_scan_age_seconds": (
                round(float(last_fast_scan_age_seconds), 3) if last_fast_scan_age_seconds is not None else None
            ),
            "opportunity_price_age_p95": _p95(opportunity_price_ages),
            "opportunity_last_detected_age_p95": _p95(opportunity_detected_ages),
            "opportunities_count": len(self._opportunities),
            "current_activity": self._current_activity,
            "lane_watchdogs": lane_watchdogs,
            "strategies": strategy_rows,
        }

        # Add WebSocket feed status
        if settings.WS_FEED_ENABLED:
            try:
                feed_mgr = get_feed_manager()
                status["ws_feeds"] = feed_mgr.health_check()
            except Exception:
                status["ws_feeds"] = {"healthy": False, "started": False}

        # Add tiered scanning status
        if settings.TIERED_SCANNING_ENABLED:
            prioritizer_stats = self._prioritizer.get_stats()
            full_coverage_completion_seconds = None
            if self._full_snapshot_cycle_started_at is not None and self._full_snapshot_cycle_completed_at is not None:
                full_coverage_completion_seconds = round(
                    max(0.0, (self._full_snapshot_cycle_completed_at - self._full_snapshot_cycle_started_at).total_seconds()),
                    3,
                )
            coverage_ratio = None
            if self._full_snapshot_cycle_total_markets > 0:
                coverage_ratio = round(
                    float(self._full_snapshot_cycle_processed_markets) / float(self._full_snapshot_cycle_total_markets),
                    6,
                )
            status["tiered_scanning"] = {
                "enabled": True,
                "fast_scan_interval": settings.FAST_SCAN_INTERVAL_SECONDS,
                "full_scan_interval": settings.FULL_SCAN_INTERVAL_SECONDS,
                "full_snapshot_strategy_interval": settings.SCANNER_FULL_SNAPSHOT_STRATEGY_INTERVAL_SECONDS,
                "full_snapshot_strategy_max_markets": settings.SCANNER_FULL_SNAPSHOT_MAX_MARKETS,
                "force_full_universe": bool(getattr(settings, "SCANNER_FORCE_FULL_UNIVERSE", True)),
                "full_snapshot_chunk_size": max(
                    1,
                    int(getattr(settings, "SCANNER_FULL_SNAPSHOT_CHUNK_SIZE", 300) or 300),
                ),
                "fast_strategy_timeout_seconds": self._fast_strategy_timeout_seconds(),
                "full_snapshot_strategy_timeout_seconds": self._full_snapshot_strategy_timeout_seconds(),
                "full_snapshot_strategy_running": self._full_snapshot_strategy_running,
                "realtime_debounce_seconds": settings.REALTIME_SCAN_DEBOUNCE_SECONDS,
                "fast_scan_cycle": self._fast_scan_cycle,
                "last_full_scan": to_iso(self._last_full_scan),
                "last_fast_scan": to_iso(self._last_fast_scan),
                "last_fast_scan_duration_seconds": self._last_fast_scan_duration_seconds,
                "fast_last_started_at": to_iso(self._fast_last_started_at),
                "fast_last_completed_at": to_iso(self._fast_last_completed_at),
                "fast_inflight": self._fast_inflight,
                "fast_lane_error": self._fast_lane_error,
                "fast_watchdog_timeout_count": self._fast_watchdog_timeout_count,
                "last_full_snapshot_strategy_scan": to_iso(self._last_full_snapshot_strategy_scan),
                "last_full_snapshot_strategy_duration_seconds": self._last_full_snapshot_strategy_duration_seconds,
                "last_full_snapshot_strategy_error": self._last_full_snapshot_strategy_error,
                "last_full_snapshot_strategy_market_count": self._last_full_snapshot_strategy_market_count,
                "last_full_snapshot_chunk_market_count": self._last_full_snapshot_chunk_market_count,
                "last_full_snapshot_strategy_opportunity_count": self._last_full_snapshot_strategy_opportunity_count,
                "last_full_snapshot_strategy_count": self._last_full_snapshot_strategy_count,
                "heavy_last_started_at": to_iso(self._heavy_last_started_at),
                "heavy_last_completed_at": to_iso(self._heavy_last_completed_at),
                "heavy_inflight": self._heavy_inflight,
                "heavy_lane_error": self._heavy_lane_error,
                "heavy_watchdog_timeout_count": self._heavy_watchdog_timeout_count,
                "full_snapshot_cursor_index": self._full_snapshot_cursor_index,
                "full_snapshot_cycle_total_markets": self._full_snapshot_cycle_total_markets,
                "full_snapshot_cycle_processed_markets": self._full_snapshot_cycle_processed_markets,
                "full_snapshot_coverage_ratio": coverage_ratio,
                "full_coverage_completion_time": full_coverage_completion_seconds,
                "full_snapshot_cycle_started_at": to_iso(self._full_snapshot_cycle_started_at),
                "full_snapshot_cycle_completed_at": to_iso(self._full_snapshot_cycle_completed_at),
                "lane_watchdogs": lane_watchdogs,
                "cached_markets": len(self._cached_markets),
                "cached_events": len(self._cached_events),
                "pending_reactive_tokens": len(self._pending_reactive_tokens),
                "last_reactive_batch_tokens": self._last_reactive_batch_tokens,
                "last_reactive_batch_markets": self._last_reactive_batch_markets,
                "dropped_reactive_tokens": self._reactive_backpressure_dropped_tokens,
                "dropped_reactive_markets": self._reactive_backpressure_dropped_markets,
                **prioritizer_stats,
            }
            status["coverage_ratio"] = coverage_ratio
            status["full_coverage_completion_time"] = full_coverage_completion_seconds

        return status

    def get_opportunities(self, filter: Optional[OpportunityFilter] = None) -> list[Opportunity]:
        """Get current opportunities with optional filtering"""
        opps = self._opportunities

        if filter:
            if filter.min_profit > 0:
                opps = [o for o in opps if o.roi_percent >= filter.min_profit * 100]
            if filter.max_risk < 1.0:
                opps = [o for o in opps if o.risk_score <= filter.max_risk]
            if filter.strategies:
                opps = [o for o in opps if o.strategy in filter.strategies]
            if filter.min_liquidity > 0:
                opps = [o for o in opps if o.min_liquidity >= filter.min_liquidity]
            if filter.category:
                # Case-insensitive category matching
                category_lower = filter.category.lower()
                opps = [o for o in opps if o.category and o.category.lower() == category_lower]

        return opps

    @property
    def last_scan(self) -> Optional[datetime]:
        return self._last_scan

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def interval_seconds(self) -> int:
        return self._interval_seconds

    def clear_opportunities(self) -> int:
        """Clear all opportunities from memory. Returns count of cleared opportunities."""
        count = len(self._opportunities)
        self._opportunities = []
        print(f"Cleared {count} opportunities from memory")
        return count

    def remove_expired_opportunities(self) -> int:
        """Remove opportunities whose resolution date has passed. Returns count removed."""
        now = datetime.now(timezone.utc)
        before_count = len(self._opportunities)

        self._opportunities = [
            opp for opp in self._opportunities if opp.resolution_date is None or _make_aware(opp.resolution_date) > now
        ]

        removed = before_count - len(self._opportunities)
        if removed > 0:
            print(f"Removed {removed} expired opportunities")
        return removed

    def remove_old_opportunities(self, max_age_minutes: int = 60) -> int:
        """Remove opportunities older than max_age_minutes. Returns count removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        before_count = len(self._opportunities)

        self._opportunities = [
            opp
            for opp in self._opportunities
            if _make_aware(opp.last_detected_at or opp.last_seen_at or opp.detected_at) >= cutoff
        ]

        removed = before_count - len(self._opportunities)
        if removed > 0:
            print(f"Removed {removed} opportunities older than {max_age_minutes} minutes")
        return removed


# Singleton instance
scanner = ArbitrageScanner()
