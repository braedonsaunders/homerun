import asyncio
import json
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Optional, Callable, List

from config import settings
from interfaces import MarketDataProvider
from models import ArbitrageOpportunity, OpportunityFilter
from models.opportunity import AIAnalysis, MispricingType
from models.database import AsyncSessionLocal, ScannerSettings, OpportunityJudgment
from services.strategies import (
    BasicArbStrategy,
    NegRiskStrategy,
    MutuallyExclusiveStrategy,
    ContradictionStrategy,
    MustHappenStrategy,
    MiracleStrategy,
    CombinatorialStrategy,
    SettlementLagStrategy,
    NewsEdgeStrategy,
    CrossPlatformStrategy,
    BayesianCascadeStrategy,
    LiquidityVacuumStrategy,
    EntropyArbStrategy,
    EventDrivenStrategy,
    TemporalDecayStrategy,
    CorrelationArbStrategy,
    MarketMakingStrategy,
    StatArbStrategy,
)
from services.plugin_loader import plugin_loader, PluginValidationError
from services.providers import market_data_provider
from services.pause_state import global_pause_state
from services.market_prioritizer import market_prioritizer, MarketTier
from services.ws_feeds import get_feed_manager
from sqlalchemy import select


def _make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware (UTC). Returns None for None input."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _to_iso_utc_z(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime as canonical UTC ISO string with trailing Z."""
    aware = _make_aware(dt)
    if aware is None:
        return None
    return aware.astimezone(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


class ArbitrageScanner:
    """Main scanner that orchestrates arbitrage detection"""

    def __init__(self, data_provider: Optional[MarketDataProvider] = None):
        self.market_data = data_provider or market_data_provider
        self.strategies = [
            BasicArbStrategy(),
            NegRiskStrategy(),  # Most profitable historically
            MutuallyExclusiveStrategy(),
            ContradictionStrategy(),
            MustHappenStrategy(),
            MiracleStrategy(),  # Swisstony's garbage collection strategy
            CombinatorialStrategy(),  # Cross-market arbitrage via integer programming
            SettlementLagStrategy(),  # Exploit delayed price adjustments (article Part IV)
            CrossPlatformStrategy(),  # Cross-platform arb (Polymarket vs Kalshi)
            BayesianCascadeStrategy(),  # Probability graph belief propagation
            LiquidityVacuumStrategy(),  # Order book imbalance exploitation
            EntropyArbStrategy(),  # Information-theoretic mispricing detection
            EventDrivenStrategy(),  # Price lag after catalyst moves
            TemporalDecayStrategy(),  # Time-decay mispricing in deadline markets
            CorrelationArbStrategy(),  # Mean-reversion on correlated pair spreads
            MarketMakingStrategy(),  # Earn bid-ask spread as liquidity provider
            StatArbStrategy(),  # Statistical edge from ensemble probability signals
        ]

        # Async strategies (require network I/O or LLM calls, run separately)
        self._news_edge_strategy = NewsEdgeStrategy()

        # Mispricing type mapping for strategies that don't set it themselves
        self._strategy_mispricing_map = {
            "basic": MispricingType.WITHIN_MARKET,
            "negrisk": MispricingType.WITHIN_MARKET,
            "mutually_exclusive": MispricingType.WITHIN_MARKET,
            "contradiction": MispricingType.WITHIN_MARKET,
            "must_happen": MispricingType.WITHIN_MARKET,
            "miracle": MispricingType.WITHIN_MARKET,
            "combinatorial": MispricingType.CROSS_MARKET,
            "settlement_lag": MispricingType.SETTLEMENT_LAG,
            "news_edge": MispricingType.NEWS_INFORMATION,
            "cross_platform": MispricingType.CROSS_MARKET,
            "bayesian_cascade": MispricingType.CROSS_MARKET,
            "liquidity_vacuum": MispricingType.WITHIN_MARKET,
            "entropy_arb": MispricingType.WITHIN_MARKET,
            "event_driven": MispricingType.CROSS_MARKET,
            "temporal_decay": MispricingType.WITHIN_MARKET,
            "correlation_arb": MispricingType.CROSS_MARKET,
            "market_making": MispricingType.WITHIN_MARKET,
            "stat_arb": MispricingType.WITHIN_MARKET,
        }
        self._running = False
        self._enabled = True
        self._interval_seconds = settings.SCAN_INTERVAL_SECONDS
        self._last_scan: Optional[datetime] = None
        self._last_full_scan: Optional[datetime] = None
        self._last_fast_scan: Optional[datetime] = None
        self._opportunities: list[ArbitrageOpportunity] = []
        self._scan_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []
        self._activity_callbacks: List[Callable] = []
        self._scan_task: Optional[asyncio.Task] = None

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
        self._cached_prices: dict = {}

        # Real yes/no price history for opportunity card sparklines.
        # market_id -> [{t: epoch_ms, yes: float, no: float}, ...]
        spark_window_hours = max(1, int(settings.SCANNER_SPARKLINE_WINDOW_HOURS))
        spark_sample_seconds = max(10, int(settings.SCANNER_SPARKLINE_SAMPLE_SECONDS))
        spark_max_points = max(30, int(settings.SCANNER_SPARKLINE_MAX_POINTS))
        spark_export_points = max(20, int(settings.SCANNER_SPARKLINE_EXPORT_POINTS))

        self._market_price_history: dict[str, list[dict[str, float]]] = {}
        self._market_history_retention_seconds: int = spark_window_hours * 3600
        retention_points = int(self._market_history_retention_seconds / spark_sample_seconds) + 2
        self._market_history_max_points: int = max(spark_max_points, retention_points)
        self._market_history_export_points: int = min(self._market_history_max_points, spark_export_points)
        # Keep one sample per interval even if price is unchanged, so cards
        # show multi-hour trend shape rather than a single refreshed point.
        self._market_history_sample_interval_ms: int = spark_sample_seconds * 1000
        self._market_token_ids: dict[str, tuple[str, str]] = {}
        self._market_history_backfill_done: set[str] = set()
        self._market_history_backfill_attempt_ms: dict[str, int] = {}
        self._market_history_backfill_retry_ms: int = 5 * 60 * 1000
        self._market_history_backfill_concurrency: int = 8
        self._market_history_backfill_max_markets: int = 120

        # Reactive scanning: event set by WS price changes to trigger immediate scan
        self._reactive_trigger = asyncio.Event()
        self._reactive_scan_registered = False

    def set_auto_ai_scoring(self, enabled: bool):
        """Enable or disable automatic AI scoring of all opportunities after each scan."""
        self._auto_ai_scoring = enabled
        print(f"Auto AI scoring {'enabled' if enabled else 'disabled'}")

    @property
    def auto_ai_scoring(self) -> bool:
        return self._auto_ai_scoring

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

    def _merge_ws_prices(self, http_prices: dict, token_ids: list[str]) -> dict:
        """Merge WebSocket real-time prices with HTTP-fetched prices.

        WS prices take precedence when fresh (within stale TTL).
        Falls back to HTTP prices for tokens not covered by WS.
        """
        if not settings.WS_FEED_ENABLED:
            return http_prices

        try:
            feed_mgr = get_feed_manager()
            if not feed_mgr._started:
                return http_prices

            merged = dict(http_prices)
            ws_hits = 0

            for token_id in token_ids:
                if feed_mgr.is_fresh(token_id):
                    mid = feed_mgr.cache.get_mid_price(token_id)
                    if mid is not None:
                        bba = feed_mgr.cache.get_best_bid_ask(token_id)
                        if bba:
                            bid, ask = bba
                            merged[token_id] = {
                                "mid": mid,
                                "bid": bid,
                                "ask": ask,
                            }
                            ws_hits += 1

            if ws_hits > 0:
                print(f"  WS price overlay: {ws_hits}/{len(token_ids)} tokens from real-time feed")
            return merged
        except Exception as e:
            print(f"  WS price merge failed (using HTTP): {e}")
            return http_prices

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

    def _extract_market_yes_no_prices(self, market, prices: dict) -> tuple[Optional[float], Optional[float]]:
        """Resolve yes/no prices from token mids with model fallback."""
        yes = None
        no = None

        token_ids = getattr(market, "clob_token_ids", None) or []
        if len(token_ids) >= 2:
            yes = self._price_value(prices.get(token_ids[0]))
            no = self._price_value(prices.get(token_ids[1]))

        if yes is None:
            try:
                yes = float(market.yes_price)
            except Exception:
                yes = None
        if no is None:
            try:
                no = float(market.no_price)
            except Exception:
                no = None

        if yes is None or no is None:
            raw = getattr(market, "outcome_prices", None) or []
            if len(raw) >= 2:
                if yes is None:
                    yes = float(raw[0])
                if no is None:
                    no = float(raw[1])

        if yes is None or no is None:
            return None, None
        return yes, no

    def _update_market_price_history(self, markets: list, prices: dict, ts: datetime) -> None:
        """Append current real market prices to bounded in-memory history."""
        now_ms = int(ts.timestamp() * 1000)
        cutoff_ms = now_ms - int(self._market_history_retention_seconds * 1000)

        for market in markets:
            market_id = getattr(market, "id", "")
            if not market_id:
                continue
            yes, no = self._extract_market_yes_no_prices(market, prices)
            if yes is None or no is None:
                continue
            if yes < 0 or no < 0:
                continue
            if yes > 1.01 or no > 1.01:
                continue

            point = {
                "t": float(now_ms),
                "yes": float(round(yes, 6)),
                "no": float(round(no, 6)),
            }
            history = self._market_price_history.setdefault(market_id, [])
            if history:
                last = history[-1]
                if abs(last.get("yes", 0.0) - point["yes"]) < 1e-9 and abs(last.get("no", 0.0) - point["no"]) < 1e-9:
                    last_t = float(last.get("t", 0.0) or 0.0)
                    if (point["t"] - last_t) < self._market_history_sample_interval_ms:
                        last["t"] = point["t"]
                        continue
            history.append(point)

            while history and history[0].get("t", 0) < cutoff_ms:
                history.pop(0)
            if len(history) > self._market_history_max_points:
                del history[: len(history) - self._market_history_max_points]

    def _remember_market_tokens(self, markets: list) -> None:
        """Cache YES/NO token IDs per market for historical backfill calls."""
        for market in markets:
            market_id = str(getattr(market, "id", "") or "")
            if not market_id:
                continue
            platform = str(getattr(market, "platform", "polymarket") or "polymarket")
            if platform != "polymarket":
                continue
            token_pair = self._coerce_polymarket_token_pair(getattr(market, "clob_token_ids", None))
            if token_pair is None:
                continue
            self._market_token_ids[market_id] = token_pair

    @staticmethod
    def _coerce_polymarket_token_pair(raw: object) -> Optional[tuple[str, str]]:
        """Parse polymarket YES/NO token IDs from list/tuple/JSON-string payloads."""
        parsed = raw
        if isinstance(parsed, str):
            text = parsed.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                return None

        if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
            return None

        yes_token = str(parsed[0] or "").strip()
        no_token = str(parsed[1] or "").strip()
        if not yes_token or not no_token or yes_token == no_token:
            return None

        # Real Polymarket token IDs are long hashes/ints; skip synthetic IDs.
        if len(yes_token) <= 20 or len(no_token) <= 20:
            return None
        return yes_token, no_token

    def _remember_market_tokens_from_opportunities(self, opportunities: list[ArbitrageOpportunity]) -> None:
        """Cache YES/NO token IDs from opportunity market dicts."""
        for opp in opportunities:
            for market in opp.markets:
                market_id = str(market.get("id", "") or "")
                if not market_id:
                    continue
                platform = str(market.get("platform", "polymarket") or "polymarket").lower()
                if platform != "polymarket":
                    continue
                token_pair = self._coerce_polymarket_token_pair(market.get("clob_token_ids"))
                if token_pair is None:
                    continue
                self._market_token_ids[market_id] = token_pair

    def _merge_market_history_points(self, market_id: str, incoming: list[dict[str, float]], now_ms: int) -> int:
        """Merge incoming history into in-memory store and return merged length."""
        if not incoming:
            return len(self._market_price_history.get(market_id, []))

        cutoff_ms = now_ms - int(self._market_history_retention_seconds * 1000)
        merged_by_ts: dict[int, dict[str, float]] = {}

        for point in self._market_price_history.get(market_id, []):
            try:
                ts = int(float(point.get("t", 0)))
                yes = float(point.get("yes", 0))
                no = float(point.get("no", 0))
            except (TypeError, ValueError):
                continue
            if ts < cutoff_ms:
                continue
            if not (0 <= yes <= 1.01 and 0 <= no <= 1.01):
                continue
            merged_by_ts[ts] = {"t": float(ts), "yes": yes, "no": no}

        for point in incoming:
            try:
                ts = int(float(point.get("t", 0)))
                yes = float(point.get("yes", 0))
                no = float(point.get("no", 0))
            except (TypeError, ValueError):
                continue
            if ts < cutoff_ms:
                continue
            if not (0 <= yes <= 1.01 and 0 <= no <= 1.01):
                continue
            merged_by_ts[ts] = {
                "t": float(ts),
                "yes": float(round(min(1.0, max(0.0, yes)), 6)),
                "no": float(round(min(1.0, max(0.0, no)), 6)),
            }

        merged = [merged_by_ts[k] for k in sorted(merged_by_ts.keys())]
        if len(merged) > self._market_history_max_points:
            merged = merged[-self._market_history_max_points :]
        self._market_price_history[market_id] = merged
        return len(merged)

    async def _backfill_market_history_for_opportunities(
        self, opportunities: list[ArbitrageOpportunity], now: datetime
    ) -> None:
        """Backfill multi-hour YES/NO history for visible opportunities."""
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
                    if market_id not in self._market_token_ids:
                        continue
                    polymarket_candidates.append(market_id)
                if len(polymarket_candidates) + len(kalshi_candidates) >= self._market_history_backfill_max_markets:
                    break
            if len(polymarket_candidates) + len(kalshi_candidates) >= self._market_history_backfill_max_markets:
                break

        total_candidates = len(polymarket_candidates) + len(kalshi_candidates)
        if total_candidates == 0:
            return

        semaphore = asyncio.Semaphore(self._market_history_backfill_concurrency)

        async def _fetch_polymarket(
            market_id: str,
        ) -> tuple[str, list[dict[str, float]], bool]:
            async with semaphore:
                self._market_history_backfill_attempt_ms[market_id] = now_ms
                yes_token, no_token = self._market_token_ids[market_id]
                yes_res, no_res = await asyncio.gather(
                    polymarket_client.get_prices_history(
                        yes_token,
                        start_ts=start_s,
                        end_ts=now_s,
                    ),
                    polymarket_client.get_prices_history(
                        no_token,
                        start_ts=start_s,
                        end_ts=now_s,
                    ),
                    return_exceptions=True,
                )

                yes_hist = []
                no_hist = []
                if not isinstance(yes_res, Exception):
                    yes_hist = yes_res
                if not isinstance(no_res, Exception):
                    no_hist = no_res
                if not yes_hist and not no_hist:
                    return market_id, [], False

                yes_by_bucket: dict[int, float] = {}
                no_by_bucket: dict[int, float] = {}

                for point in yes_hist:
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
                    yes_by_bucket[bucket] = price

                for point in no_hist:
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
                    no_by_bucket[bucket] = price

                buckets = sorted(set(yes_by_bucket.keys()) | set(no_by_bucket.keys()))
                merged: list[dict[str, float]] = []
                for bucket in buckets:
                    yes = yes_by_bucket.get(bucket)
                    no = no_by_bucket.get(bucket)
                    if yes is None and no is None:
                        continue
                    if yes is None and no is not None and 0 <= no <= 1:
                        yes = 1.0 - no
                    if no is None and yes is not None and 0 <= yes <= 1:
                        no = 1.0 - yes
                    if yes is None or no is None:
                        continue
                    merged.append(
                        {
                            "t": float(bucket),
                            "yes": float(round(min(1.0, max(0.0, yes)), 6)),
                            "no": float(round(min(1.0, max(0.0, no)), 6)),
                        }
                    )
                return market_id, merged, True

        results: list[tuple[str, list[dict[str, float]], bool]] = []
        if polymarket_candidates:
            poly_results = await asyncio.gather(*[_fetch_polymarket(mid) for mid in polymarket_candidates])
            results.extend(poly_results)

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
                    results.append((market_id, merged_points, market_fetch_success))

        updated = 0
        completed = 0
        for market_id, points, success in results:
            if success:
                completed += 1
            if points:
                merged_len = self._merge_market_history_points(market_id, points, now_ms)
                if merged_len >= 2:
                    updated += 1
                    self._market_history_backfill_done.add(market_id)

        if updated > 0:
            print(
                f"  Sparkline backfill: hydrated {updated}/{total_candidates} markets "
                f"({completed} successful history fetches)"
            )

    def get_market_history_for_opportunities(
        self, opportunities: list[ArbitrageOpportunity], max_points: Optional[int] = None
    ) -> dict[str, list[dict[str, float]]]:
        """Return compact market history map for markets present in opportunities."""
        market_ids: set[str] = set()
        for opp in opportunities:
            for market in opp.markets:
                mid = market.get("id", "")
                if mid:
                    market_ids.add(mid)

        out: dict[str, list[dict[str, float]]] = {}
        export_points = self._market_history_export_points if max_points is None else max_points
        limit = max(2, min(self._market_history_max_points, int(export_points)))
        for mid in market_ids:
            hist = self._market_price_history.get(mid, [])
            if hist:
                out[mid] = hist[-limit:]
        return out

    async def attach_price_history_to_opportunities(
        self,
        opportunities: list[ArbitrageOpportunity],
        *,
        now: Optional[datetime] = None,
        timeout_seconds: Optional[float] = 12.0,
    ) -> int:
        """Attach scanner-managed market price history to opportunity market payloads."""
        if not opportunities:
            return 0

        ts = now or datetime.now(timezone.utc)
        self._remember_market_tokens_from_opportunities(opportunities)

        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                await asyncio.wait_for(
                    self._backfill_market_history_for_opportunities(opportunities, ts),
                    timeout=timeout_seconds,
                )
            else:
                await self._backfill_market_history_for_opportunities(opportunities, ts)
        except asyncio.TimeoutError:
            print("  Sparkline backfill timed out (shared attach)")
        except Exception as e:
            print(f"  Sparkline backfill error in shared attach: {e}")

        market_history = self.get_market_history_for_opportunities(opportunities)
        attached = 0
        for opp in opportunities:
            for market in opp.markets:
                market_id = str(market.get("id", "") or "")
                if not market_id:
                    continue
                history = market_history.get(market_id, [])
                if len(history) < 2:
                    continue
                market["price_history"] = history
                attached += 1
        return attached

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

    async def load_plugins(self):
        """Load all enabled plugins from the database into the plugin loader."""
        try:
            from models.database import StrategyPlugin as StrategyPluginModel

            async with AsyncSessionLocal() as session:
                result = await session.execute(select(StrategyPluginModel).where(StrategyPluginModel.enabled))
                plugins = result.scalars().all()

            loaded_count = 0
            for p in plugins:
                try:
                    plugin_loader.load_plugin(p.slug, p.source_code, p.config or None)
                    # Update status in DB
                    async with AsyncSessionLocal() as session:
                        result = await session.execute(
                            select(StrategyPluginModel).where(StrategyPluginModel.id == p.id)
                        )
                        db_plugin = result.scalar_one_or_none()
                        if db_plugin:
                            db_plugin.status = "loaded"
                            db_plugin.error_message = None
                            await session.commit()
                    loaded_count += 1
                except PluginValidationError as e:
                    print(f"  Plugin '{p.slug}' failed to load: {e}")
                    # Update error status in DB
                    async with AsyncSessionLocal() as session:
                        result = await session.execute(
                            select(StrategyPluginModel).where(StrategyPluginModel.id == p.id)
                        )
                        db_plugin = result.scalar_one_or_none()
                        if db_plugin:
                            db_plugin.status = "error"
                            db_plugin.error_message = str(e)
                            await session.commit()
                except Exception as e:
                    print(f"  Plugin '{p.slug}' failed to load (unexpected): {e}")

            if loaded_count > 0:
                print(f"Loaded {loaded_count} strategy plugin(s)")
        except Exception as e:
            print(f"Error loading plugins: {e}")

    def _get_all_strategies(self) -> list:
        """Return built-in strategies + loaded plugin strategies."""
        plugin_strategies = plugin_loader.get_all_strategy_instances()
        # BTC 15-minute high-frequency logic is isolated to crypto-worker.
        # Scanner strategy set must not run btc_eth_highfreq.
        return [s for s in (self.strategies + plugin_strategies) if self._strategy_key(s) != "btc_eth_highfreq"]

    @staticmethod
    def _strategy_key(strategy) -> str:
        st = getattr(strategy, "strategy_type", "")
        return st if isinstance(st, str) else getattr(st, "value", "")

    async def _get_effective_strategies(self) -> list:
        """Apply validation guardrails (demoted strategies are skipped)."""
        all_strategies = self._get_all_strategies()
        try:
            from services.validation_service import validation_service

            demoted = await validation_service.get_demoted_strategy_types()
        except Exception:
            demoted = set()
        if not demoted:
            return all_strategies

        filtered = [s for s in all_strategies if self._strategy_key(s) not in demoted]
        skipped = len(all_strategies) - len(filtered)
        if skipped > 0:
            print(f"  Validation guardrails: skipped {skipped} demoted strategies")
        return filtered

    async def scan_once(self) -> list[ArbitrageOpportunity]:
        """Perform a single scan for arbitrage opportunities"""
        print(f"[{utcnow().isoformat()}] Starting arbitrage scan...")
        await self._set_activity("Fetching Polymarket events and markets...")
        loop = asyncio.get_running_loop()

        try:
            # Fetch events and markets concurrently (they are independent)
            events, markets = await asyncio.gather(
                self.market_data.get_all_events(closed=False),
                self.market_data.get_all_markets(active=True),
            )

            # Filter out markets whose end_date has already passed —
            # these are resolved events awaiting settlement and can't
            # be traded profitably.
            now = datetime.now(timezone.utc)
            markets = [m for m in markets if m.end_date is None or _make_aware(m.end_date) > now]

            # Also prune expired markets inside events so strategies
            # like NegRisk that iterate event.markets don't pick them up.
            for event in events:
                event.markets = [m for m in event.markets if m.end_date is None or _make_aware(m.end_date) > now]

            print(f"  Fetched {len(events)} Polymarket events and {len(markets)} markets")
            await self._set_activity(f"Fetched {len(events)} events, {len(markets)} markets")

            # Fetch Kalshi markets and merge them so ALL strategies
            # (not just cross-platform) can detect opportunities on Kalshi.
            kalshi_market_count = 0
            kalshi_event_count = 0
            if settings.CROSS_PLATFORM_ENABLED:
                await self._set_activity("Fetching Kalshi markets...")
                try:
                    kalshi_markets = await self.market_data.get_cross_platform_markets(active=True)
                    kalshi_markets = [m for m in kalshi_markets if m.end_date is None or _make_aware(m.end_date) > now]
                    kalshi_market_count = len(kalshi_markets)
                    markets.extend(kalshi_markets)

                    kalshi_events = await self.market_data.get_cross_platform_events(closed=False)
                    for ke in kalshi_events:
                        ke.markets = [m for m in ke.markets if m.end_date is None or _make_aware(m.end_date) > now]
                    kalshi_event_count = len(kalshi_events)
                    events.extend(kalshi_events)

                    if kalshi_market_count > 0:
                        print(f"  Fetched {kalshi_event_count} Kalshi events and {kalshi_market_count} Kalshi markets")
                        await self._set_activity(
                            f"Fetched {kalshi_event_count} Kalshi events, {kalshi_market_count} markets"
                        )
                except Exception as e:
                    print(f"  Kalshi fetch failed (non-fatal): {e}")

            # Get live prices for Polymarket tokens only.
            # Kalshi markets already have prices baked in from the API;
            # their synthetic token IDs (e.g. "TICKER_yes") should not
            # be sent to Polymarket's CLOB.
            all_token_ids = []
            for market in markets:
                for tid in market.clob_token_ids:
                    # Polymarket CLOB token IDs are long hex strings (>20 chars).
                    # Kalshi synthetic IDs are short like "TICKER_yes".
                    if len(tid) > 20:
                        all_token_ids.append(tid)

            # Batch price fetching (limit to avoid rate limits)
            prices = {}
            if all_token_ids:
                await self._set_activity(f"Fetching prices for {min(len(all_token_ids), 500)} tokens...")
                # Sample tokens if too many
                token_sample = all_token_ids[:500]
                prices = await self.market_data.get_prices_batch(token_sample)
                print(f"  Fetched prices for {len(prices)} tokens")

            # Overlay WebSocket real-time prices where available
            prices = self._merge_ws_prices(prices, all_token_ids[:500])

            # Cache full data for fast scans between full scans
            self._cached_events = list(events)
            self._cached_markets = list(markets)
            self._cached_prices = dict(prices)
            self._remember_market_tokens(markets)
            self._update_market_price_history(markets, prices, now)

            # Subscribe to WebSocket feeds for active market tokens
            if settings.WS_FEED_ENABLED:
                try:
                    feed_mgr = get_feed_manager()
                    if feed_mgr._started:
                        # Subscribe to all active token IDs for real-time updates
                        # The feed will auto-manage subscriptions
                        poly_tokens = [t for t in all_token_ids[:500] if len(t) > 20]
                        if poly_tokens:
                            await feed_mgr.polymarket_feed.subscribe(token_ids=poly_tokens[:200])
                except Exception as e:
                    print(f"  WS subscription failed (non-critical): {e}")

            # Tiered scanning: classify markets and apply change detection.
            # These are CPU-bound operations on 6000+ markets — run them
            # in the thread pool so the event loop stays responsive.
            markets_to_evaluate = markets
            unchanged_count = 0
            if settings.TIERED_SCANNING_ENABLED:
                try:

                    def _run_prioritizer(prioritizer, mkts, ts):
                        """Run all CPU-bound prioritizer work in a thread."""
                        prioritizer.update_stability_scores()
                        t_map = prioritizer.classify_all(mkts, ts)
                        prioritizer.compute_attention_scores(mkts)
                        changed = prioritizer.get_changed_markets(mkts)
                        return t_map, changed

                    try:
                        tier_map, changed = await loop.run_in_executor(
                            None, _run_prioritizer, self._prioritizer, markets, now
                        )
                    except RuntimeError as e:
                        # Some monitor integrations require an active loop in the
                        # current thread. Retry on the main loop instead of failing
                        # the whole prioritizer pass.
                        if "no current event loop" in str(e).lower():
                            tier_map, changed = _run_prioritizer(self._prioritizer, markets, now)
                        else:
                            raise
                    unchanged_count = len(markets) - len(changed)

                    # For full scans, still run all markets through strategies
                    # but log the change detection stats. The real savings come
                    # from fast scans where we only evaluate changed markets.
                    # However, for COLD-tier markets, we DO skip them if unchanged.
                    cold_ids = {m.id for m in tier_map[MarketTier.COLD]}
                    markets_to_evaluate = [
                        m for m in markets if m.id not in cold_ids or self._prioritizer.has_market_changed(m)
                    ]
                    cold_skipped = len(markets) - len(markets_to_evaluate)

                    print(
                        f"  Tiers: {len(tier_map[MarketTier.HOT])} hot, "
                        f"{len(tier_map[MarketTier.WARM])} warm, "
                        f"{len(tier_map[MarketTier.COLD])} cold "
                        f"({cold_skipped} unchanged cold skipped)"
                    )
                except Exception as e:
                    print(f"  Prioritizer error (non-fatal, using all markets): {e}")
                    markets_to_evaluate = markets

            # Run all strategies concurrently in a thread pool.
            # strategy.detect() is synchronous CPU-bound work; running it
            # on the event loop blocks all async handlers (including
            # GET /api/opportunities) causing the UI to hang during scans.
            all_opportunities = []
            loop = asyncio.get_running_loop()
            await self._set_activity(f"Running {len(self.strategies)} strategies...")

            async def _run_strategy(strategy):
                """Run a single strategy in the default thread-pool executor."""
                return strategy, await loop.run_in_executor(None, strategy.detect, events, markets_to_evaluate, prices)

            all_strategies = await self._get_effective_strategies()
            results = await asyncio.gather(
                *[_run_strategy(s) for s in all_strategies],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    print(f"  Strategy error: {result}")
                    continue
                strategy, opps = result
                # Classify mispricing type if not already set by strategy
                for opp in opps:
                    if opp.mispricing_type is None:
                        opp.mispricing_type = self._strategy_mispricing_map.get(
                            opp.strategy, MispricingType.WITHIN_MARKET
                        )
                all_opportunities.extend(opps)
                print(f"  {strategy.name}: found {len(opps)} opportunities")
                # Record plugin run stats
                strategy_type = getattr(strategy, "strategy_type", "")
                if isinstance(strategy_type, str) and plugin_loader.get_plugin(strategy_type):
                    plugin_loader.record_run(strategy_type, len(opps))

            # Update prioritizer state after evaluation (CPU-bound, run in thread)
            if settings.TIERED_SCANNING_ENABLED:
                try:
                    await loop.run_in_executor(None, self._prioritizer.update_after_evaluation, markets, now)
                except Exception:
                    pass

            # ----------------------------------------------------------
            # STORE opportunities immediately so the API can serve them
            # without waiting for slow LLM-based strategies (News Edge).
            # ----------------------------------------------------------
            all_opportunities = self._deduplicate_cross_strategy(all_opportunities)
            all_opportunities.sort(key=lambda x: x.roi_percent, reverse=True)
            all_opportunities = self._apply_opportunity_caps(all_opportunities)

            # Attach existing AI judgments from the database
            try:
                await asyncio.wait_for(
                    self._attach_ai_judgments(all_opportunities),
                    timeout=15,
                )
            except asyncio.TimeoutError:
                print("  AI judgment attach timed out (15s), continuing without")
            except Exception:
                pass

            self._opportunities = self._merge_opportunities(all_opportunities)
            try:
                await asyncio.wait_for(
                    self._backfill_market_history_for_opportunities(self._opportunities, now),
                    timeout=12,
                )
            except asyncio.TimeoutError:
                print("  Sparkline backfill timed out (12s), continuing with live history")
            except Exception as e:
                print(f"  Sparkline backfill error (non-fatal): {e}")
            self._last_scan = datetime.now(timezone.utc)
            print(f"  Stored {len(self._opportunities)} opportunities")

            # News Edge: LLM-based analysis is now manual only (triggered
            # from the UI via POST /news/edges or POST /news/edges/single)
            # to avoid automatic spend on paid LLM providers.  We still
            # pre-fetch articles + run semantic matching so the data is
            # ready when the user clicks "Analyze".
            if settings.NEWS_EDGE_ENABLED:
                asyncio.create_task(self._prefetch_news_matches(events, markets, prices))

            # AI Intelligence: Score unscored opportunities (non-blocking)
            # Only run if auto_ai_scoring is enabled (opt-in, default OFF).
            # Manual per-opportunity analysis is always available via the UI.
            if self._auto_ai_scoring:
                try:
                    from services.ai import get_llm_manager

                    manager = get_llm_manager()
                    if manager.is_available():
                        # Cancel any still-running scoring task from a previous scan
                        if self._ai_scoring_task and not self._ai_scoring_task.done():
                            self._ai_scoring_task.cancel()
                            try:
                                await self._ai_scoring_task
                            except (asyncio.CancelledError, Exception):
                                pass
                        # Score the full merged pool so retained opps get scored too
                        self._ai_scoring_task = asyncio.create_task(self._ai_score_opportunities(self._opportunities))
                except Exception:
                    pass  # AI scoring is non-critical

            # Notify callbacks (pass only newly detected opportunities)
            for callback in self._scan_callbacks:
                try:
                    await callback(all_opportunities)
                except Exception as e:
                    print(f"  Callback error: {e}")

            self._last_full_scan = now

            scan_suffix = ""
            if settings.TIERED_SCANNING_ENABLED and unchanged_count > 0:
                scan_suffix = f" ({unchanged_count} unchanged markets detected)"

            print(
                f"[{utcnow().isoformat()}] Scan complete. "
                f"{len(all_opportunities)} detected this scan, "
                f"{len(self._opportunities)} total in pool{scan_suffix}"
            )
            await self._set_activity(
                f"Scan complete — {len(all_opportunities)} found, {len(self._opportunities)} total in pool"
            )
            return self._opportunities

        except Exception as e:
            print(f"[{utcnow().isoformat()}] Scan error: {e}")
            await self._set_activity(f"Scan error: {e}")
            raise

    async def scan_fast(self) -> list[ArbitrageOpportunity]:
        """Fast-path scan targeting only HOT-tier and changed markets.

        This runs every FAST_SCAN_INTERVAL_SECONDS (default 15s) between
        full scans. It:
          1. Incrementally fetches only recently created markets (delta)
          2. Uses cached data from the last full scan for existing markets
          3. Re-fetches prices only for HOT-tier markets
          4. Runs strategies only on markets whose prices have changed
          5. Merges results into the main opportunity pool

        Much cheaper than a full scan: fewer API calls, fewer strategy runs.
        """
        now = datetime.now(timezone.utc)
        print(f"[{now.isoformat()}] Starting fast scan (hot-tier + incremental)...")
        await self._set_activity("Fast scan: fetching recent markets...")

        try:
            # 1. Incremental fetch: get only recently created markets
            new_markets: list = []
            if settings.INCREMENTAL_FETCH_ENABLED:
                try:
                    new_markets = await self.market_data.get_recent_markets(since_minutes=5)
                    if new_markets:
                        print(f"  Incremental: {len(new_markets)} recently created markets")
                except Exception as e:
                    print(f"  Incremental fetch failed (non-fatal): {e}")

            # 2. Merge incremental markets into cached data
            cached_market_ids = {m.id for m in self._cached_markets}
            truly_new = [m for m in new_markets if m.id not in cached_market_ids]
            if truly_new:
                self._cached_markets.extend(truly_new)
                print(f"  Added {len(truly_new)} brand-new markets to cache")

            # 3. Update MarketMonitor with the new markets to generate alerts
            try:
                from services.market_monitor import market_monitor

                await market_monitor.get_fresh_opportunities()
            except Exception:
                pass

            # 4. Classify all cached markets into tiers (CPU-bound, run in thread)
            loop = asyncio.get_running_loop()

            def _classify_cached(prioritizer, mkts, ts):
                prioritizer.update_stability_scores()
                return prioritizer.classify_all(mkts, ts)

            tier_map = await loop.run_in_executor(None, _classify_cached, self._prioritizer, self._cached_markets, now)
            hot_markets = tier_map[MarketTier.HOT]

            if not hot_markets:
                print("  No HOT-tier markets, skipping fast scan")
                self._last_fast_scan = now
                return self._opportunities

            # 5. Re-fetch prices for HOT-tier markets only
            hot_token_ids = []
            for market in hot_markets:
                for tid in market.clob_token_ids:
                    if len(tid) > 20:  # Polymarket tokens only
                        hot_token_ids.append(tid)

            hot_prices = {}
            if hot_token_ids:
                await self._set_activity(
                    f"Fast scan: fetching prices for {min(len(hot_token_ids), 200)} hot-tier tokens..."
                )
                hot_prices = await self.market_data.get_prices_batch(hot_token_ids[:200])
                print(f"  Fetched prices for {len(hot_prices)} hot-tier tokens")

            # Overlay WebSocket real-time prices where available
            hot_prices = self._merge_ws_prices(hot_prices, hot_token_ids[:200])

            # Merge with cached prices
            merged_prices = {**self._cached_prices, **hot_prices}
            self._remember_market_tokens(self._cached_markets)
            self._update_market_price_history(hot_markets, merged_prices, now)

            # 6. Change detection: only evaluate markets whose prices moved (CPU-bound)
            changed_markets = await loop.run_in_executor(None, self._prioritizer.get_changed_markets, hot_markets)
            if not changed_markets:
                print(f"  All {len(hot_markets)} hot-tier markets unchanged, skipping strategies")
                await self._set_activity(f"Fast scan: {len(hot_markets)} markets unchanged, skipping")
                self._prioritizer.update_after_evaluation(hot_markets, now)
                self._last_scan = now
                self._last_fast_scan = now
                return self._opportunities

            print(f"  {len(changed_markets)}/{len(hot_markets)} hot-tier markets have price changes")
            await self._set_activity(f"Fast scan: running strategies on {len(changed_markets)} changed markets...")

            # 7. Run strategies on changed markets only (full events kept for context)
            all_markets_for_strategies = [
                m for m in changed_markets if m.end_date is None or _make_aware(m.end_date) > now
            ]
            events_for_strategies = self._cached_events

            loop = asyncio.get_running_loop()
            fast_opportunities = []

            async def _run_strategy(strategy):
                return strategy, await loop.run_in_executor(
                    None,
                    strategy.detect,
                    events_for_strategies,
                    all_markets_for_strategies,
                    merged_prices,
                )

            all_strategies = await self._get_effective_strategies()
            results = await asyncio.gather(
                *[_run_strategy(s) for s in all_strategies],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    continue
                strategy, opps = result
                for opp in opps:
                    if opp.mispricing_type is None:
                        opp.mispricing_type = self._strategy_mispricing_map.get(
                            opp.strategy, MispricingType.WITHIN_MARKET
                        )
                fast_opportunities.extend(opps)
                # Record plugin run stats
                strategy_type = getattr(strategy, "strategy_type", "")
                if isinstance(strategy_type, str) and plugin_loader.get_plugin(strategy_type):
                    plugin_loader.record_run(strategy_type, len(opps))

            fast_opportunities = self._deduplicate_cross_strategy(fast_opportunities)
            fast_opportunities.sort(key=lambda x: x.roi_percent, reverse=True)
            fast_opportunities = self._apply_opportunity_caps(fast_opportunities)

            # 8. Update prioritizer state (CPU-bound, run in thread)
            def _update_prioritizer_state(prioritizer, mkts, ts):
                unch = prioritizer.update_after_evaluation(mkts, ts)
                prioritizer.compute_attention_scores(mkts)
                return unch

            unchanged = await loop.run_in_executor(None, _update_prioritizer_state, self._prioritizer, hot_markets, now)

            # 9. Merge into main pool
            if fast_opportunities:
                await self._attach_ai_judgments(fast_opportunities)
                self._opportunities = self._merge_opportunities(fast_opportunities)
                try:
                    await asyncio.wait_for(
                        self._backfill_market_history_for_opportunities(self._opportunities, now),
                        timeout=6,
                    )
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

            self._last_scan = now
            self._last_fast_scan = now
            self._fast_scan_cycle += 1

            # Notify callbacks
            for callback in self._scan_callbacks:
                try:
                    await callback(fast_opportunities)
                except Exception as e:
                    print(f"  Callback error: {e}")

            print(
                f"[{now.isoformat()}] Fast scan complete. "
                f"{len(fast_opportunities)} detected, "
                f"{len(self._opportunities)} total in pool "
                f"({unchanged} unchanged markets skipped)"
            )
            await self._set_activity(
                f"Fast scan complete — {len(fast_opportunities)} found, {len(self._opportunities)} total"
            )
            return self._opportunities

        except Exception as e:
            print(f"[{utcnow().isoformat()}] Fast scan error: {e}")
            await self._set_activity(f"Fast scan error: {e}")
            raise

    async def _prefetch_news_matches(self, events, markets, prices):
        """Pre-fetch news articles and run semantic matching (no LLM calls).

        This prepares the data so that manual edge analysis from the UI
        is fast — articles are already fetched and matched to markets.
        No paid LLM calls are made here.
        """
        try:
            from services.news.feed_service import news_feed_service
            from services.news.semantic_matcher import semantic_matcher
            from concurrent.futures import ThreadPoolExecutor

            # Step 1: Fetch articles (free — RSS/GDELT)
            await news_feed_service.fetch_all()
            all_articles = news_feed_service.get_articles(max_age_hours=settings.NEWS_ARTICLE_TTL_HOURS)

            if not all_articles:
                return

            # Step 2: Build market index
            market_infos = self._news_edge_strategy._build_market_infos(events, markets, prices)
            if not market_infos:
                return

            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="news_prefetch") as executor:
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

    def _deduplicate_cross_strategy(self, opportunities: list[ArbitrageOpportunity]) -> list[ArbitrageOpportunity]:
        """Remove duplicate opportunities that cover the same underlying markets.

        When multiple strategies detect the same set of markets (e.g., NegRisk
        and settlement_lag both find MI-07 House Election), keep only the one
        with the highest ROI. This prevents the same trade from appearing
        multiple times in the output.

        Uses a market-ID fingerprint: sorted set of market IDs involved.
        """
        seen: dict[str, ArbitrageOpportunity] = {}  # fingerprint -> best opp

        for opp in opportunities:
            # Build a fingerprint from the sorted market IDs
            market_ids = sorted(m.get("id", "") for m in opp.markets)
            fingerprint = "|".join(market_ids)

            if fingerprint in seen:
                # Keep the opportunity with higher ROI
                if opp.roi_percent > seen[fingerprint].roi_percent:
                    seen[fingerprint] = opp
            else:
                seen[fingerprint] = opp

        deduped = list(seen.values())
        removed = len(opportunities) - len(deduped)
        if removed > 0:
            print(f"  Deduplicated: removed {removed} cross-strategy duplicates")
        return deduped

    def _apply_opportunity_caps(self, opportunities: list[ArbitrageOpportunity]) -> list[ArbitrageOpportunity]:
        """Limit pool size globally and per strategy to prevent strategy flood."""
        total_cap = int(getattr(settings, "SCANNER_MAX_OPPORTUNITIES_TOTAL", 0) or 0)
        per_strategy_cap = int(getattr(settings, "SCANNER_MAX_OPPORTUNITIES_PER_STRATEGY", 0) or 0)
        if total_cap <= 0 and per_strategy_cap <= 0:
            return opportunities

        kept: list[ArbitrageOpportunity] = []
        by_strategy: dict[str, int] = {}

        for opp in opportunities:
            strategy = str(getattr(opp, "strategy", "") or "unknown")
            if per_strategy_cap > 0 and by_strategy.get(strategy, 0) >= per_strategy_cap:
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

    def _merge_opportunities(self, new_opportunities: list[ArbitrageOpportunity]) -> list[ArbitrageOpportunity]:
        """Merge newly detected opportunities into the existing pool.

        Instead of replacing all opportunities on each scan, this method:
        - Adds newly discovered opportunities to the pool
        - Updates existing opportunities (matched by stable_id) with fresh
          market data while preserving original detection time and AI analysis
        - Removes expired opportunities whose resolution date has passed
        """
        now = datetime.now(timezone.utc)

        # Index existing opportunities by stable_id
        existing_map: dict[str, ArbitrageOpportunity] = {opp.stable_id: opp for opp in self._opportunities}

        new_count = 0
        updated_count = 0

        for new_opp in new_opportunities:
            new_opp.last_seen_at = now
            existing = existing_map.get(new_opp.stable_id)
            if existing:
                # Preserve original detection time and ID
                new_opp.detected_at = existing.detected_at
                new_opp.id = existing.id
                # Preserve AI analysis if not freshly attached from DB
                if existing.ai_analysis and not new_opp.ai_analysis:
                    new_opp.ai_analysis = existing.ai_analysis
                updated_count += 1
            else:
                new_count += 1
            existing_map[new_opp.stable_id] = new_opp

        # Remove expired opportunities and keep BTC/ETH high-frequency
        # opportunities isolated to the dedicated Crypto surface.
        stale_minutes = max(
            5,
            int(getattr(settings, "SCANNER_STALE_OPPORTUNITY_MINUTES", 45) or 45),
        )
        stale_cutoff = now - timedelta(minutes=stale_minutes)
        merged = [
            opp
            for opp in existing_map.values()
            if (opp.resolution_date is None or _make_aware(opp.resolution_date) > now)
            and (
                opp.last_seen_at is None
                or _make_aware(opp.last_seen_at) >= stale_cutoff
            )
            and opp.strategy != "btc_eth_highfreq"
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

            async def _persist_inline_analysis(opp: ArbitrageOpportunity) -> None:
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
        """Register with WS FeedManager for reactive price-change scanning."""
        if self._reactive_scan_registered:
            return
        if not settings.WS_FEED_ENABLED:
            return
        try:
            feed_mgr = get_feed_manager()
            if feed_mgr._started:

                async def _trigger_reactive():
                    self._reactive_trigger.set()

                feed_mgr.set_reactive_scan_callback(_trigger_reactive, debounce_seconds=2.0)
                self._reactive_scan_registered = True
                print("  Reactive scanning registered (WS price-change triggers)")
        except Exception as e:
            print(f"  Reactive scanning registration failed: {e}")

    async def _scan_loop(self):
        """Internal scan loop with reactive + tiered polling.

        When WS feeds are active, scans are triggered reactively by significant
        price changes (>0.5% move on any subscribed token).  The timer-based
        polling is kept as a fallback with extended intervals:
          - Full scans: every FULL_SCAN_INTERVAL_SECONDS (default 120s)
          - Fast scans: every FAST_SCAN_INTERVAL_SECONDS (default 15s) OR
            immediately on WS price changes (debounced to 2s)
          - Reactive scans skip HTTP price fetching entirely (uses WS cache)

        When TIERED_SCANNING_ENABLED is disabled, falls back to the original
        fixed-interval full scan.
        """
        while self._running:
            if not self._enabled:
                await asyncio.sleep(self._interval_seconds)
                continue

            # Register reactive scanning on first enabled iteration
            self._register_reactive_scanning()

            if not settings.TIERED_SCANNING_ENABLED:
                # Legacy mode: simple fixed-interval full scan
                try:
                    await self.scan_once()
                except Exception as e:
                    print(f"Scan error: {e}")
                await asyncio.sleep(self._interval_seconds)
                continue

            # Tiered scanning mode
            now = datetime.now(timezone.utc)

            # Determine if it's time for a full scan
            full_interval = settings.FULL_SCAN_INTERVAL_SECONDS
            needs_full = self._last_full_scan is None or (now - self._last_full_scan).total_seconds() >= full_interval

            if needs_full:
                try:
                    await self.scan_once()
                except Exception as e:
                    print(f"Full scan error: {e}")
            else:
                # Fast scan (only if we have cached data from a previous full scan)
                if self._cached_markets:
                    # Check if crypto prediction triggers an immediate scan
                    triggered = self._prioritizer.should_trigger_fast_scan(now)
                    if triggered:
                        print("  [TRIGGER] Crypto market creation imminent — fast scan")

                    try:
                        await self.scan_fast()
                    except Exception as e:
                        print(f"Fast scan error: {e}")
                else:
                    # No cached data yet — force a full scan
                    try:
                        await self.scan_once()
                    except Exception as e:
                        print(f"Full scan error (first run): {e}")

            # Wait for either the timer OR a reactive price-change trigger
            self._reactive_trigger.clear()
            sleep_seconds = settings.FAST_SCAN_INTERVAL_SECONDS
            await self._set_activity(f"Idle — next scan in ≤{sleep_seconds}s (or on price change)")
            try:
                # Wait for reactive trigger, but time out after the normal interval
                await asyncio.wait_for(self._reactive_trigger.wait(), timeout=sleep_seconds)
                await self._set_activity("Reactive scan triggered by WS price change")
            except asyncio.TimeoutError:
                pass  # Normal timer-based fallback

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

        # Kick off an immediate scan in the background so this
        # method returns quickly and doesn't block the API response.
        if self._running:
            asyncio.create_task(self.scan_once())

    async def pause(self):
        """Pause all background services (scanner, trader orchestrator, copy trader, wallet tracker, discovery, etc.)."""
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
        status = {
            "running": self._running,
            "enabled": self._enabled,
            "interval_seconds": self._interval_seconds,
            "auto_ai_scoring": self._auto_ai_scoring,
            "last_scan": _to_iso_utc_z(self._last_scan),
            "opportunities_count": len(self._opportunities),
            "current_activity": self._current_activity,
            "strategies": [
                {
                    "name": getattr(s, "name", self._strategy_key(s) or "Unknown"),
                    "type": self._strategy_key(s),
                }
                for s in self.strategies
            ]
            + [
                {
                    "name": self._news_edge_strategy.name,
                    "type": self._strategy_key(self._news_edge_strategy),
                }
            ],
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
            status["tiered_scanning"] = {
                "enabled": True,
                "fast_scan_interval": settings.FAST_SCAN_INTERVAL_SECONDS,
                "full_scan_interval": settings.FULL_SCAN_INTERVAL_SECONDS,
                "fast_scan_cycle": self._fast_scan_cycle,
                "last_full_scan": _to_iso_utc_z(self._last_full_scan),
                "last_fast_scan": _to_iso_utc_z(self._last_fast_scan),
                "cached_markets": len(self._cached_markets),
                "cached_events": len(self._cached_events),
                **prioritizer_stats,
            }

        return status

    def get_opportunities(self, filter: Optional[OpportunityFilter] = None) -> list[ArbitrageOpportunity]:
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

        self._opportunities = [opp for opp in self._opportunities if _make_aware(opp.detected_at) >= cutoff]

        removed = before_count - len(self._opportunities)
        if removed > 0:
            print(f"Removed {removed} opportunities older than {max_age_minutes} minutes")
        return removed


# Singleton instance
scanner = ArbitrageScanner()
