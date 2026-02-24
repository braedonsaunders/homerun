from __future__ import annotations

import asyncio
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict
from datetime import datetime

from models import (
    Opportunity,
    Event,
    ExecutionConstraints,
    ExecutionLeg,
    ExecutionPlan,
    Market,
)
from config import settings
from services.fee_model import fee_model
from services.data_events import DataEvent, EventType

from utils.converters import to_float, to_confidence
from utils.kelly import kalshi_taker_fee, polymarket_taker_fee
from utils.signal_helpers import signal_payload
from utils.utcnow import utcnow as _utcnow, as_utc


def utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return _utcnow()


def make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to timezone-aware UTC."""
    return as_utc(dt)


@dataclass
class DecisionCheck:
    """A single check/gate in the evaluate decision."""

    key: str
    label: str
    passed: bool
    score: float = None
    detail: str = None
    payload: dict = field(default_factory=dict)


@dataclass
class StrategyDecision:
    """Result of evaluate() — whether to trade a signal."""

    decision: str  # selected | skipped | blocked | failed
    reason: str
    score: float = None
    size_usd: float = None
    checks: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)


@dataclass
class ExitDecision:
    """Result of should_exit() — whether to close a position."""

    action: str  # "close" | "hold" | "reduce"
    reason: str
    close_price: float = None
    reduce_fraction: float = None  # For partial exits (0-1)
    payload: dict = field(default_factory=dict)


@dataclass
class ScaleOutTarget:
    trigger_bps: float
    exit_fraction: float


@dataclass
class ScaleOutConfig:
    targets: list
    trailing_stop_bps: float = 80.0
    near_resolution_exit: bool = True
    near_resolution_hours: float = 24.0
    near_resolution_spread_widen_bps: float = 50.0


@dataclass
class ScoringWeights:
    """Declarative scoring formula for the composable evaluate pipeline.

    Strategies set this as a class attribute to opt into the pipeline.
    The composite score is:
        (edge * edge_weight) + (confidence * confidence_weight)
        - (risk_score * risk_penalty) + (market_count * market_count_bonus)
        + (liquidity_score * liquidity_weight) + structural_bonus (if guaranteed)
    """

    edge_weight: float = 0.55
    confidence_weight: float = 30.0
    risk_penalty: float = 8.0
    liquidity_weight: float = 0.0
    liquidity_divisor: float = 5000.0
    market_count_bonus: float = 0.0
    structural_bonus: float = 0.0


@dataclass
class SizingConfig:
    """Declarative sizing formula for the composable evaluate pipeline.

    Position size = base_size * (1 + edge/base_divisor)
                    * (confidence_offset + confidence)
                    * market_scale * risk_scale

    risk_scale = max(risk_floor, 1.0 - risk_score * risk_scale_factor)
    market_scale = 1.0 + min(market_scale_cap, max(0, markets-1)) * market_scale_factor
    """

    base_divisor: float = 100.0
    confidence_offset: float = 0.75
    risk_scale_factor: float = 0.35
    risk_floor: float = 0.55
    market_scale_factor: float = 0.08
    market_scale_cap: int = 4


@dataclass
class EvaluateContext:
    """Typed context passed to strategy.evaluate(). All fields are guaranteed present."""

    params: dict
    trader: Any
    mode: str
    live_market: dict
    source_config: dict

    def get(self, key: str, default=None):
        """Backward-compat dict-style access. Prefer direct attribute access."""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """Backward-compat dict-style subscript access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


class StrategyContext(TypedDict, total=False):
    """Typed structure for strategy context stored alongside every TradeSignal.

    Passed as ``strategy_context_json`` through the signal bridge and available
    in ``evaluate()`` via ``signal.strategy_context_json``. Provides a typed
    contract so strategy authors know exactly what context is available at
    evaluation time.

    All fields are optional (``total=False``). Strategies populate what they
    know and consumers access what they need. Strategy-specific extras go in
    the ``extra`` key to avoid namespace collisions.
    """

    source: str  # Data source ("scanner", "crypto", "weather", ...)
    strategy_type: str  # Strategy slug (mirrors signal.strategy_type)
    roi_percent: float  # Raw theoretical ROI from detect()
    realistic_roi: float  # VWAP-adjusted ROI (when order book data available)
    confidence: float  # Detection-time confidence [0, 1]
    risk_score: float  # Risk score [0, 1] from calculate_risk_score()
    risk_factors: list  # Human-readable list of risk factor strings
    liquidity: float  # Min liquidity across all legs
    edge_percent: float  # Edge in percent (same as roi_percent for arb)
    num_legs: int  # Number of markets / trade legs
    resolution_date: str  # ISO 8601 UTC resolution date string
    days_to_resolution: float  # Fractional days until resolution
    annualized_roi: float  # ROI annualized over days_to_resolution
    is_guaranteed: bool  # True for guaranteed-spread arb
    extra: dict  # Strategy-specific extras (namespaced per strategy)


def _has_custom_detect_async(strategy: "BaseStrategy") -> bool:
    """Return True if *strategy* provides its own ``detect_async`` override.

    The base class ``detect_async`` simply delegates to ``detect``. We skip
    it here so scanner strategies that only implement ``detect`` run in the
    thread-pool rather than on the event loop.
    """
    method = getattr(type(strategy), "detect_async", None)
    if method is None:
        return False
    base_method = getattr(BaseStrategy, "detect_async", None)
    return method is not base_method


def _has_custom_detect_sync(strategy: "BaseStrategy") -> bool:
    """Return True if *strategy* provides its own ``detect_sync`` override.

    The base class ``detect_sync`` delegates to ``detect()``. We use this to
    detect strategies that opted into the new detect_sync() naming convention
    so the scanner can call detect_sync() directly (in the thread-pool).
    """
    method = getattr(type(strategy), "detect_sync", None)
    if method is None:
        return False
    base_method = getattr(BaseStrategy, "detect_sync", None)
    return method is not base_method


class BaseStrategy(ABC):
    """Base class for all trading strategies.

    Strategies set ``strategy_type`` to their unique slug string.

    Full Signal Lifecycle
    =====================

    Every opportunity flows through this pipeline. Understanding it is
    essential for strategy authors — each gate can alter or block your signal.

    1. **Detection** — ``detect()`` / ``detect_async()`` / ``on_event()``
       Your strategy examines market data and returns ``list[Opportunity]``.
       Called by the scanner (MARKET_DATA_REFRESH) or workers (CRYPTO_UPDATE,
       WEATHER_UPDATE, TRADER_ACTIVITY, NEWS_UPDATE, DATA_SOURCE_UPDATE, etc.).

    2. **Quality Filter** — ``QualityFilterPipeline.evaluate_opportunity()``
       10 structural checks (min ROI, directional cap, plausible ROI, max legs,
       per-leg liquidity, hard liquidity floor, min position size, min absolute
       profit, resolution timeframe, annualized ROI). Override thresholds via
       ``quality_filter_overrides`` class attribute. Rejected opportunities are
       still written to the DB with ``quality_passed=False``.

    3. **Cross-Strategy Deduplication** — same market + similar ROI from
       multiple strategies collapses to the best signal. Opt out via
       ``allow_deduplication = False``.

    4. **TradeSignal Upsert** — ``strategy_signal_bridge.bridge_opportunities_to_signals()``
       Opportunity is written to the ``trade_signals`` table as a pending signal.
       Dedupe key = (stable_id, strategy_type, market_id).

    5. **Signal Expiry** — stale signals past ``expires_at`` are skipped.

    6. **Orchestrator Pickup** — ``trader_orchestrator_worker`` polls pending
       signals and calls ``strategy.evaluate(signal, context)``.

    7. **Evaluate** — ``evaluate(signal, context) -> StrategyDecision``
       Your strategy decides whether to trade the signal RIGHT NOW.
       If ``scoring_weights`` is set, uses the composable pipeline
       (``custom_checks`` + ``compute_score`` + ``compute_size``).
       Returns ``StrategyDecision`` with decision = selected|skipped|blocked.

    8. **Risk Manager Gates** (applied by orchestrator, not strategy):
       - Daily loss limit (``BlockReason.RISK_DAILY_LOSS``)
       - Gross exposure cap (``BlockReason.RISK_GROSS_EXPOSURE``)
       - Consecutive loss guard (``BlockReason.RISK_CONSECUTIVE_LOSS``)
       - Trade notional cap (``BlockReason.RISK_TRADE_NOTIONAL``)
       - Open positions limit (``BlockReason.RISK_OPEN_POSITIONS``)
       - Per-market exposure cap (``BlockReason.RISK_MARKET_EXPOSURE``)
       Your ``on_blocked(signal, reason, context)`` fires if blocked.

    9. **Trading Window** — UTC time-of-day gate. ``BlockReason.TRADING_WINDOW``.

    10. **Stacking Guard** — prevents opening a second position on the same
        market. ``BlockReason.STACKING_GUARD``.

    11. **Size Capping** — position size clamped to ``max_trade_notional_usd``.
        Your ``on_size_capped(original, capped, reason)`` fires if capped.

    12. **Order Execution** — the order manager places the trade.

    13. **Position Lifecycle** — ``should_exit(position, market_state)``
        called every cycle for open positions. Default applies TP/SL/trailing/
        max-hold from config.

    Strategy Types
    ==============

    Implement **at least one** of:

    * ``detect()`` — synchronous detection (run in thread-pool executor).
    * ``detect_async()`` — async detection (preferred for I/O-bound work such
      as LLM calls, HTTP requests, or database queries). Awaited on the event loop.
    * Override ``on_event()`` — only needed when your strategy reacts to
      non-standard event types (crypto, weather, news) with custom payload logic.

    Scanner strategies subscribe to ``EventType.MARKET_DATA_REFRESH`` and only
    need to implement ``detect()`` — the default ``on_event()`` handles routing.

    BlockReason Reference
    =====================

    All ``BlockReason`` codes that can be passed to ``on_blocked()``:

    - ``QUALITY_FILTER`` — failed QualityFilterPipeline (ROI, liquidity, etc.)
    - ``RISK_DAILY_LOSS`` — daily realized loss exceeds configured limit
    - ``RISK_GROSS_EXPOSURE`` — total open notional exceeds exposure cap
    - ``RISK_CONSECUTIVE_LOSS`` — too many consecutive losing trades
    - ``RISK_TRADE_NOTIONAL`` — single trade notional exceeds per-trade cap
    - ``RISK_OPEN_POSITIONS`` — max open positions reached
    - ``RISK_MARKET_EXPOSURE`` — per-market exposure cap reached
    - ``TRADING_WINDOW`` — outside configured UTC trading hours
    - ``STACKING_GUARD`` — already have a position on this market
    - ``SIGNAL_EXPIRED`` — signal past its expires_at timestamp
    - ``SCANNER_POOL_CAPACITY`` — scanner opportunity pool is full
    - ``DEDUPLICATION`` — duplicate signal collapsed with a better one
    """

    strategy_type: str  # strategy slug identifier (set to slug by loader)
    name: str
    description: str

    # Default config (overridden by user settings in UI)
    default_config: dict = {}

    # ── Strategy metadata (declare on your subclass) ────────────
    # These let strategies declare their requirements and behavior so the
    # infrastructure can route, filter, and manage them without hardcoding.

    # Mispricing classification — how this strategy finds opportunities
    mispricing_type: Optional[str] = None  # "within_market", "cross_market", "settlement_lag", "news_information"

    # Source key — which data pipeline feeds this strategy
    source_key: str = "scanner"  # "scanner", "crypto", "news", "weather", "traders"

    # Worker affinity — which worker should run this strategy's detect()
    worker_affinity: str = "scanner"  # "scanner", "crypto", "news", "weather", "traders"

    # Opportunity TTL — how long detected opportunities stay valid (minutes)
    # None means use the global SCANNER_STALE_OPPORTUNITY_MINUTES default.
    opportunity_ttl_minutes: Optional[int] = None
    # Retention window alias for config-driven durations like "15m" / "2d".
    retention_window: Optional[str] = None
    # Per-strategy opportunity cap alias for config-driven retention.
    retention_max_opportunities: Optional[int] = None

    # Deduplication control — if True, this strategy's opportunities can be
    # deduplicated against other strategies finding the same markets.
    # Set to False if this strategy provides a unique perspective even on
    # the same markets (e.g., news_edge vs basic on same market).
    allow_deduplication: bool = True

    # Requirements declaration — what data this strategy needs
    requires_order_book: bool = False
    requires_news_data: bool = False
    requires_historical_prices: bool = False

    # Market type preferences — filter markets before passing to detect()
    market_categories: Optional[list[str]] = None  # None means all categories
    requires_resolution_date: bool = False
    binary_only: bool = True  # Most strategies only work with 2-outcome markets

    # Event subscriptions — strategies declare what data events they react to.
    # Use EventType.* constants (e.g. EventType.MARKET_DATA_REFRESH).
    # Unknown values are rejected at registration time by the EventDispatcher.
    subscriptions: list[str] = []
    supports_entry_take_profit_exit: bool = False
    default_open_order_timeout_seconds: Optional[float] = None
    # Realtime scanner routing for MARKET_DATA_REFRESH batches:
    # - "incremental": run on affected-market batches in reactive fast scans
    # - "full_snapshot": always run on the full cached market snapshot
    # - "auto": scanner decides based on strategy metadata (default)
    realtime_processing_mode: str = "auto"

    # Composable evaluate pipeline — set scoring_weights to opt in.
    # When scoring_weights is None, evaluate() uses the base passthrough.
    # When set, evaluate() uses the composable pipeline with custom_checks(),
    # compute_score(), and compute_size() hooks.
    scoring_weights: ScoringWeights | None = None
    sizing_config: SizingConfig | None = None
    scale_out_config: Optional[ScaleOutConfig] = None

    # Strategy-specific fallback defaults for pipeline params.
    # These are used when the param is not provided by the orchestrator.
    # Override on each strategy to preserve its original default behavior.
    pipeline_defaults: dict = {}

    # Per-strategy quality filter overrides. Set any field to override the
    # global platform default for opportunities from this strategy.
    # Example: directional strategies can relax the 120% ROI cap:
    #
    #   from services.quality_filter import QualityFilterOverrides
    #   quality_filter_overrides = QualityFilterOverrides(max_roi_cap=200.0)
    quality_filter_overrides: Any = None  # Optional[QualityFilterOverrides]

    # ── Declarative platform gate hints ─────────────────────────
    # These fields let strategies declare their requirements to the platform
    # so the orchestrator, risk manager, and scheduler can make informed
    # decisions without hardcoding strategy-specific logic.

    # Risk budget — declares this strategy's risk appetite so the risk
    # manager can allocate exposure proportionally across strategies.
    # max_exposure_usd: max gross notional this strategy should have open.
    # max_positions: max concurrent positions from this strategy.
    # max_daily_loss_usd: strategy-specific daily loss circuit breaker.
    # All values are soft hints — the global risk manager caps override them.
    risk_budget: dict = {}
    # Example:
    #   risk_budget = {
    #       "max_exposure_usd": 500.0,
    #       "max_positions": 3,
    #       "max_daily_loss_usd": 50.0,
    #   }

    # Market filters — declarative market selection criteria applied before
    # detect() is called. Reduces noise and lets the scanner skip irrelevant
    # markets early. All fields optional; None means no filter.
    market_filters: dict = {}
    # Example:
    #   market_filters = {
    #       "categories": ["politics", "crypto"],     # only these categories
    #       "min_liquidity_usd": 5000.0,              # skip illiquid markets
    #       "max_resolution_days": 90,                # skip far-future markets
    #       "min_resolution_days": 1,                 # skip about-to-resolve markets
    #       "platforms": ["polymarket"],               # platform filter
    #   }

    # Scheduling — how often this strategy should run and what triggers it.
    # Strategies that set scheduling preferences allow the orchestrator to
    # batch and prioritize strategy execution.
    scheduling: dict = {}
    # Example:
    #   scheduling = {
    #       "run_interval_seconds": 300,   # run every 5 minutes
    #       "run_on_event_only": True,     # only run when a subscribed event fires
    #       "priority": "high",            # high | normal | low
    #   }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate strategy metadata at class definition time.

        Catches configuration errors immediately at import — not at runtime
        when a signal fires. Validates:
        - All declared subscriptions are known EventType values
        - strategy_type is defined on concrete (non-abstract) subclasses

        NOTE: ``__init_subclass__`` is called before the ABC metaclass sets
        ``cls.__abstractmethods__``, so we cannot rely on that attribute to
        detect intermediate base classes. Instead we check whether
        ``strategy_type`` is declared in the class's own ``__dict__`` — if
        it's not, the class is either abstract or an intermediate base, and
        we skip validation. Concrete strategies must explicitly declare
        ``strategy_type = "my_strategy"`` in their class body.
        """
        super().__init_subclass__(**kwargs)
        # Only validate subscriptions if this class itself declares them
        own_subs = cls.__dict__.get("subscriptions")
        if own_subs is not None:
            for et in own_subs:
                if et not in EventType._ALL:
                    raise ValueError(
                        f"{cls.__name__}.subscriptions contains unknown event type '{et}'. "
                        f"Valid types: {sorted(EventType._ALL)}. "
                        f"Use EventType.* constants from services.data_events."
                    )
        own_realtime_mode = cls.__dict__.get("realtime_processing_mode")
        if own_realtime_mode is not None:
            if own_realtime_mode not in {"auto", "incremental", "full_snapshot"}:
                raise TypeError(
                    f"{cls.__name__}.realtime_processing_mode must be one of ['auto', 'incremental', 'full_snapshot']"
                )
        # Only validate strategy_type if this class explicitly declares it.
        # Intermediate base classes (e.g. BaseWeatherStrategy) do not declare
        # strategy_type in their own __dict__ — their concrete subclasses do.
        if "strategy_type" not in cls.__dict__:
            return
        st = cls.__dict__["strategy_type"]
        if not isinstance(st, str) or not st.strip():
            raise TypeError(
                f"{cls.__name__}.strategy_type must be a non-empty string. Example: strategy_type = 'my_strategy'"
            )

    def __init__(self):
        self.fee = settings.POLYMARKET_FEE
        self.min_profit = settings.MIN_PROFIT_THRESHOLD
        self.config = dict(self.default_config)
        # Strategy state — persisted across detect() cycles
        self._state: dict = {}

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[Opportunity]:
        """Detect arbitrage opportunities (sync).

        Backward-compatible name. Prefer overriding detect_sync() for new strategies.

        Called every scan cycle with the full set of active events, markets,
        and live CLOB prices.  Override this for CPU-bound strategies that
        don't need async I/O.

        The default implementation returns an empty list.  Subclasses should
        override ``detect_sync()`` (or ``detect_async()`` for I/O-bound work).
        """
        return []

    def detect_sync(
        self,
        events: list,
        markets: list,
        prices: dict,
    ) -> list:
        """Detect arbitrage opportunities (sync, preferred name for new strategies).

        Alias for detect(). Override detect_sync() or detect_async() — not both.

        detect_sync() runs in a thread pool executor. Use it for CPU-bound or
        simple synchronous detection logic.

        detect_async() runs directly in the event loop. Use it for I/O-bound
        detection (HTTP calls, DB queries, etc.).

        The scanner prefers detect_async() if overridden, otherwise falls back
        to detect_sync() (which delegates to detect() for backward compatibility).
        """
        return self.detect(events, markets, prices)

    async def detect_async(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        """Detect arbitrage opportunities (async, preferred for I/O-bound work).

        The scanner calls this method when it exists.  The default
        implementation delegates to the synchronous ``detect()`` so
        existing strategies continue to work without modification.

        Override this (instead of ``detect()``) when your strategy needs
        to ``await`` coroutines -- e.g. LLM calls via ``services.ai``,
        HTTP requests via ``httpx``, or database queries.
        """
        return self.detect(events, markets, prices)

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        """React to a subscribed DataEvent.

        Called by the EventDispatcher when an event matching one of your
        ``subscriptions`` fires. Return a list of detected opportunities (may be
        empty).

        **Default behaviour**: delegates ``MARKET_DATA_REFRESH`` events to
        ``detect()`` (sync, run in a thread-pool executor) or ``detect_async()``
        if the subclass overrides it. All other event types return ``[]``.

        Scanner strategies that only implement ``detect()`` do **not** need to
        override this method. Override only when your strategy needs custom
        routing — for example, reacting to ``CRYPTO_UPDATE`` or ``NEWS_UPDATE``
        events where the payload structure differs from the standard
        market-data batch.

        Args:
            event: Immutable DataEvent with event_type, source, payload, and
                   type-specific fields (markets, prices, old_price, etc.)

        Returns:
            list of Opportunity objects (empty list if no opportunities).
        """
        if event.event_type == EventType.MARKET_DATA_REFRESH:
            # Priority:
            # 1. detect_async() if the subclass provides its own override → run on event loop
            # 2. detect_sync() if the subclass provides its own override → run in thread-pool
            # 3. detect() (backward-compat name) → run in thread-pool
            if _has_custom_detect_async(self):
                return await self.detect_async(event.events or [], event.markets or [], event.prices or {})
            loop = asyncio.get_running_loop()
            if _has_custom_detect_sync(self):
                return await loop.run_in_executor(
                    None, self.detect_sync, event.events or [], event.markets or [], event.prices or {}
                )
            return await loop.run_in_executor(
                None, self.detect, event.events or [], event.markets or [], event.prices or {}
            )
        return []

    def on_blocked(
        self,
        signal: Any,
        reason: str,
        context: "EvaluateContext | dict",
    ) -> None:
        """Called by the platform whenever a signal is blocked at any gate.

        Args:
            signal: The TradeSignal that was blocked
            reason: A BlockReason constant (e.g. BlockReason.TRADING_WINDOW)
            context: The EvaluateContext for this signal

        Override this to log, alert, or update internal state when your signals are blocked.
        The default implementation does nothing.
        """

    def on_size_capped(
        self,
        original_size: float,
        capped_size: float,
        reason: str,
    ) -> None:
        """Notification hook called when the platform caps this strategy's position size.

        Called by the orchestrator worker when the strategy's ``evaluate()``
        returns a size that exceeds the configured ``max_trade_notional_usd``.
        Use this to log or track size caps for analysis.

        The return value is **ignored** — this hook cannot override the cap.
        It is a pure notification, not a control hook.

        Args:
            original_size: The size the strategy computed in ``evaluate()``.
            capped_size: The size the platform is allowing (after notional cap).
            reason: Human-readable reason for the cap.
        """

    def configure(self, config: dict) -> None:
        """Apply merged config. Called by the loader after instantiation.

        Config cascade: default_config (class attribute) -> DB config (user overrides).
        The loader merges both levels before calling configure(), so ``config``
        here is already the fully-resolved dict. Access config values via
        ``self.config["key"]`` or ``self.config.get("key", default)``.
        """
        merged = dict(self.default_config)
        if config:
            merged.update(config)
        self.config = merged

    @property
    def state(self) -> dict:
        """Persistent state dict — survives across detect() cycles.

        Use this to track rolling averages, momentum signals, prior
        detections, or any cross-cycle data your strategy needs.

        Example:
            self.state["last_prices"] = {market.id: price for ...}
            prior = self.state.get("last_prices", {})
        """
        return self._state

    @state.setter
    def state(self, value: dict) -> None:
        self._state = value

    # ── Execution phase (optional override) ──────────────────

    def evaluate(self, signal: Any, context: "EvaluateContext | dict") -> StrategyDecision:
        """Decide whether to trade a signal RIGHT NOW.

        ``context`` is an ``EvaluateContext`` in all platform calls.

        If scoring_weights is set, uses the composable pipeline
        (custom_checks + compute_score + compute_size). Otherwise
        falls back to the base passthrough implementation.
        """
        if self.scoring_weights is None:
            return self._base_evaluate(signal, context)
        return self._pipeline_evaluate(signal, context)

    def _base_evaluate(self, signal: Any, context: "EvaluateContext | dict") -> StrategyDecision:
        """Base passthrough evaluate — trusts detection layer's edge/confidence.

        Called by the orchestrator when a pending signal from this strategy
        is ready for execution. Override to add custom gating logic.

        ``context`` is an ``EvaluateContext`` in all platform calls.

        Args:
            signal: TradeSignal object with .edge_percent, .confidence,
                    .direction, .entry_price, .payload_json, .strategy_context
            context: EvaluateContext (or dict for backward compat) with
                     params, trader, mode, live_market, source_config fields.

        Returns:
            StrategyDecision with decision="selected"|"skipped"|"blocked"|"failed"
        """
        params = context.get("params") or {}

        edge = float(getattr(signal, "edge_percent", 0) or 0)
        confidence = float(getattr(signal, "confidence", 0) or 0)
        if confidence > 1.0:
            confidence = confidence / 100.0

        min_edge = float(params.get("min_edge_percent", 0))
        min_conf = float(params.get("min_confidence", 0))
        base_size = float(params.get("base_size_usd", 25.0))
        max_size = float(params.get("max_size_usd", 500.0))

        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.1f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
        ]

        if not all(c.passed for c in checks):
            failed = [c for c in checks if not c.passed]
            return StrategyDecision(
                decision="skipped",
                reason=f"Failed: {', '.join(c.key for c in failed)}",
                score=edge * confidence,
                checks=checks,
            )

        size = min(base_size * (1 + edge / 100) * (0.5 + confidence), max_size)

        return StrategyDecision(
            decision="selected",
            reason="Passthrough: detection thresholds met",
            score=edge * confidence,
            size_usd=size,
            checks=checks,
        )

    # ── Composable evaluate pipeline ─────────────────────────

    def custom_checks(
        self, signal: Any, context: "EvaluateContext | dict", params: dict, payload: dict
    ) -> list[DecisionCheck]:
        """Override to add strategy-specific checks beyond the standard pipeline.

        Called during the composable evaluate pipeline after the standard
        edge/confidence/risk checks. Return additional DecisionCheck objects
        that must all pass for the signal to be selected.
        """
        return []

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        """Compute composite score using scoring_weights.

        Override for fully custom scoring logic. Default uses the
        declarative ScoringWeights formula.
        """
        w = self.scoring_weights or ScoringWeights()
        score = (
            (edge * w.edge_weight)
            + (confidence * w.confidence_weight)
            - (risk_score * w.risk_penalty)
            + (min(6, market_count) * w.market_count_bonus)
        )
        if w.liquidity_weight > 0:
            liquidity = float(payload.get("liquidity", 0) or 0)
            score += min(1.0, liquidity / w.liquidity_divisor) * w.liquidity_weight
        if w.structural_bonus > 0 and payload.get("is_guaranteed"):
            score += w.structural_bonus
        return score

    def compute_size(
        self, base_size: float, max_size: float, edge: float, confidence: float, risk_score: float, market_count: int
    ) -> float:
        """Compute position size using sizing_config.

        Override for fully custom sizing logic (e.g. Kelly criterion).
        Default uses the declarative SizingConfig formula.
        """
        cfg = self.sizing_config or SizingConfig()
        risk_scale = max(cfg.risk_floor, 1.0 - (risk_score * cfg.risk_scale_factor))
        market_scale = 1.0 + (min(cfg.market_scale_cap, max(0, market_count - 1)) * cfg.market_scale_factor)
        size = (
            base_size
            * (1.0 + (edge / cfg.base_divisor))
            * (cfg.confidence_offset + confidence)
            * market_scale
            * risk_scale
        )
        return max(1.0, min(max_size, size))

    def _pipeline_evaluate(self, signal: Any, context: "EvaluateContext | dict") -> StrategyDecision:
        """Composable evaluate pipeline implementation."""
        params = context.get("params") or {}
        payload = signal_payload(signal)

        # Merge strategy-specific defaults (pipeline_defaults) with caller params.
        # Caller params take precedence; pipeline_defaults provide fallbacks.
        d = self.pipeline_defaults
        min_edge = to_float(
            params.get("min_edge_percent", d.get("min_edge_percent", 3.0)), d.get("min_edge_percent", 3.0)
        )
        min_conf = to_confidence(
            params.get("min_confidence", d.get("min_confidence", 0.42)), d.get("min_confidence", 0.42)
        )
        max_risk = to_confidence(
            params.get("max_risk_score", d.get("max_risk_score", 0.68)), d.get("max_risk_score", 0.68)
        )
        base_size = max(
            1.0, to_float(params.get("base_size_usd", d.get("base_size_usd", 20.0)), d.get("base_size_usd", 20.0))
        )
        max_size = max(
            base_size, to_float(params.get("max_size_usd", d.get("max_size_usd", 180.0)), d.get("max_size_usd", 180.0))
        )

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])

        # Standard checks
        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "risk_score",
                "Risk score ceiling",
                risk_score <= max_risk,
                score=risk_score,
                detail=f"max={max_risk:.2f}",
            ),
        ]

        # Strategy-specific checks
        checks.extend(self.custom_checks(signal, context, params, payload))

        score = self.compute_score(edge, confidence, risk_score, market_count, payload)

        if not all(c.passed for c in checks):
            failed = [c for c in checks if not c.passed]
            return StrategyDecision(
                "skipped",
                f"{self.name}: {', '.join(c.key for c in failed)}",
                score=score,
                checks=checks,
            )

        size = self.compute_size(base_size, max_size, edge, confidence, risk_score, market_count)

        return StrategyDecision(
            "selected",
            f"{self.name}: signal selected (edge={edge:.2f}%, conf={confidence:.2f})",
            score=score,
            size_usd=size,
            checks=checks,
        )

    # ── Exit/hold phase (optional override) ──────────────────

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Decide whether to close an open position.

        Called every lifecycle cycle for positions opened by this strategy.
        Override to implement custom exit logic (e.g., re-check forecasts,
        monitor correlated markets, decay-based exits).

        Default: delegates to default_exit_check() which applies standard
        TP/SL/trailing/max-hold from strategy config.

        Args:
            position: Position object with:
                .entry_price, .current_price, .highest_price, .lowest_price
                .age_minutes, .pnl_percent, .strategy_context (dict from detect)
                .config (strategy params at time of entry)
            market_state: {
                "current_price": float,
                "market_tradable": bool,
                "is_resolved": bool,
                "winning_outcome": str|None,
            }

        Returns:
            ExitDecision with action="close"|"hold"|"reduce"
            Return None to use default behavior (same as ExitDecision("hold", ...))
        """
        return self.default_exit_check(position, market_state)

    @staticmethod
    def _coerce_seconds_left(value: Any) -> float | None:
        parsed = to_float(value, None)
        if parsed is None:
            return None
        if parsed < 0:
            return None
        return float(parsed)

    @staticmethod
    def _seconds_until_end_time(value: Any) -> float | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        end_time = make_aware(parsed)
        if end_time is None:
            return None
        return max(0.0, (end_time - utcnow()).total_seconds())

    @classmethod
    def _seconds_left_for_position(cls, position: Any, market_state: dict) -> float | None:
        for key in ("seconds_left", "market_seconds_left"):
            parsed = cls._coerce_seconds_left(market_state.get(key))
            if parsed is not None:
                return parsed

        strategy_context = getattr(position, "strategy_context", None)
        if isinstance(strategy_context, dict):
            parsed = cls._coerce_seconds_left(strategy_context.get("seconds_left"))
            if parsed is not None:
                return parsed

        for key in ("end_time", "market_end_time"):
            parsed = cls._seconds_until_end_time(market_state.get(key))
            if parsed is not None:
                return parsed

        if isinstance(strategy_context, dict):
            parsed = cls._seconds_until_end_time(strategy_context.get("end_time"))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _stop_loss_policy(config: dict[str, Any]) -> str:
        raw_policy = config.get("stop_loss_policy")
        if raw_policy is None:
            raw_policy = config.get("stop_loss_mode")
        policy = str(raw_policy or "always").strip().lower()
        if policy in {"near_close", "near_close_only", "close_window"}:
            return "near_close_only"
        return "always"

    @staticmethod
    def _stop_loss_activation_seconds(config: dict[str, Any]) -> float:
        value = config.get("stop_loss_activation_seconds")
        if value is None:
            value = config.get("stop_loss_near_close_seconds")
        parsed = to_float(value, 120.0)
        if parsed is None:
            parsed = 120.0
        return max(0.0, float(parsed))

    def default_exit_check(self, position: Any, market_state: dict) -> ExitDecision:
        """Standard TP/SL/trailing/max-hold exit logic.

        Strategies can call this as a fallback after their custom checks.
        Uses config params: take_profit_pct, stop_loss_pct, trailing_stop_pct,
        max_hold_minutes, min_hold_minutes, resolve_only, stop_loss_policy,
        stop_loss_activation_seconds.
        """
        config = getattr(position, "config", None) or {}
        current_price = market_state.get("current_price")

        # Resolution always wins
        if market_state.get("is_resolved"):
            winning = market_state.get("winning_outcome")
            outcome_idx = getattr(position, "outcome_idx", 0)
            close_price = 1.0 if winning == outcome_idx else 0.0
            return ExitDecision("close", "Market resolved", close_price=close_price)

        if current_price is None:
            return ExitDecision("hold", "No current price available")

        # Resolve-only mode
        if config.get("resolve_only", False):
            return ExitDecision("hold", "Resolve-only mode")

        entry_price = float(getattr(position, "entry_price", 0) or 0)
        age_minutes = float(getattr(position, "age_minutes", 0) or 0)
        min_hold = float(config.get("min_hold_minutes", 0) or 0)
        min_hold_passed = age_minutes >= min_hold

        if not min_hold_passed:
            return ExitDecision("hold", f"Min hold not met ({age_minutes:.0f}/{min_hold:.0f} min)")

        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # Scale-out partial exits (before TP/SL so partial exits fire first)
        soc = self.scale_out_config
        if soc is not None:
            ctx = getattr(position, "strategy_context", None)
            if not isinstance(ctx, dict):
                ctx = {}
                position.strategy_context = ctx
            targets_hit: list = ctx.get("_scale_out_targets_hit", [])
            all_targets_hit = len(targets_hit) >= len(soc.targets)

            if not all_targets_hit:
                for idx, target in enumerate(soc.targets):
                    if idx in targets_hit:
                        continue
                    if pnl_pct >= target.trigger_bps:
                        targets_hit.append(idx)
                        ctx["_scale_out_targets_hit"] = targets_hit
                        return ExitDecision(
                            "reduce",
                            f"Scale-out target {idx + 1} hit ({pnl_pct:.1f}% >= {target.trigger_bps:.1f}%)",
                            close_price=current_price,
                            reduce_fraction=target.exit_fraction,
                        )

            if all_targets_hit:
                # Trailing stop on remainder after all targets hit
                peak_price = ctx.get("_scale_out_peak_price", current_price)
                if current_price > peak_price:
                    peak_price = current_price
                ctx["_scale_out_peak_price"] = peak_price

                if peak_price > entry_price and soc.trailing_stop_bps > 0:
                    trailing_trigger = peak_price * (1.0 - soc.trailing_stop_bps / 10000.0)
                    if current_price <= trailing_trigger:
                        return ExitDecision(
                            "close",
                            f"Scale-out trailing stop ({current_price:.4f} <= {trailing_trigger:.4f})",
                            close_price=current_price,
                        )

                # Near-resolution exit
                if soc.near_resolution_exit:
                    seconds_left = self._seconds_left_for_position(position, market_state)
                    if seconds_left is not None and seconds_left <= soc.near_resolution_hours * 3600.0:
                        spread_change_bps = (
                            abs(current_price - entry_price) / entry_price * 10000.0 if entry_price > 0 else 0.0
                        )
                        if spread_change_bps >= soc.near_resolution_spread_widen_bps:
                            return ExitDecision(
                                "close",
                                f"Near-resolution exit ({seconds_left / 3600.0:.1f}h left, spread {spread_change_bps:.0f}bps)",
                                close_price=current_price,
                            )

        # Take profit
        tp = config.get("take_profit_pct")
        if tp is not None and pnl_pct >= float(tp):
            return ExitDecision("close", f"Take profit hit ({pnl_pct:.1f}% >= {tp}%)", close_price=current_price)

        # Stop loss
        sl = config.get("stop_loss_pct")
        stop_loss_deferred_reason = None
        if sl is not None:
            stop_loss_policy = self._stop_loss_policy(config)
            activation_seconds = self._stop_loss_activation_seconds(config)
            seconds_left = self._seconds_left_for_position(position, market_state)
            stop_loss_armed = True
            if stop_loss_policy == "near_close_only":
                stop_loss_armed = seconds_left is not None and seconds_left <= activation_seconds

            if pnl_pct <= -abs(float(sl)):
                if stop_loss_armed:
                    return ExitDecision("close", f"Stop loss hit ({pnl_pct:.1f}% <= -{sl}%)", close_price=current_price)
                if stop_loss_policy == "near_close_only":
                    if seconds_left is None:
                        stop_loss_deferred_reason = "Stop loss deferred until near close (timing unavailable)"
                    else:
                        stop_loss_deferred_reason = (
                            f"Stop loss deferred ({seconds_left:.0f}s left > {activation_seconds:.0f}s arm window)"
                        )

        # Trailing stop
        trailing = config.get("trailing_stop_pct")
        highest = float(getattr(position, "highest_price", 0) or 0)
        if trailing is not None and float(trailing) > 0 and highest > entry_price:
            trailing_activation_profit_pct = to_float(config.get("trailing_stop_activation_profit_pct"), 0.0)
            if trailing_activation_profit_pct is None:
                trailing_activation_profit_pct = 0.0
            highest_gain_pct = ((highest - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
            if highest_gain_pct < float(trailing_activation_profit_pct):
                trailing = None
        if trailing is not None and float(trailing) > 0 and highest > entry_price:
            trigger_price = highest * (1.0 - float(trailing) / 100.0)
            if current_price <= trigger_price:
                return ExitDecision(
                    "close", f"Trailing stop ({current_price:.4f} <= {trigger_price:.4f})", close_price=current_price
                )

        # Max hold
        max_hold = config.get("max_hold_minutes")
        if max_hold is not None and age_minutes >= float(max_hold):
            return ExitDecision(
                "close", f"Max hold exceeded ({age_minutes:.0f} >= {max_hold} min)", close_price=current_price
            )

        # Market inactive
        if not market_state.get("market_tradable", True) and config.get("close_on_inactive_market", False):
            return ExitDecision("close", "Market inactive", close_price=current_price)

        if stop_loss_deferred_reason is not None:
            return ExitDecision("hold", stop_loss_deferred_reason)

        return ExitDecision("hold", "No exit condition met")

    def calculate_risk_score(
        self, markets: list[Market], resolution_date: Optional[datetime] = None
    ) -> tuple[float, list[str]]:
        """Calculate risk score (0-1) and return risk factors"""
        score = 0.0
        factors = []

        # Time to resolution risk
        if resolution_date:
            resolution_aware = make_aware(resolution_date)
            days_until = (resolution_aware - utcnow()).days
            very_short = getattr(settings, "RISK_VERY_SHORT_DAYS", 2)
            short = getattr(settings, "RISK_SHORT_DAYS", 7)
            long_lockup = getattr(settings, "RISK_LONG_LOCKUP_DAYS", 180)
            ext_lockup = getattr(settings, "RISK_EXTENDED_LOCKUP_DAYS", 90)
            if days_until < very_short:
                score += 0.4
                factors.append(f"Very short time to resolution (<{very_short} days)")
            elif days_until < short:
                score += 0.2
                factors.append(f"Short time to resolution (<{short} days)")
            # Long-duration capital lockup risk
            elif days_until > long_lockup:
                score += 0.3
                factors.append(f"Long capital lockup ({days_until} days to resolution)")
            elif days_until > ext_lockup:
                score += 0.2
                factors.append(f"Extended capital lockup ({days_until} days to resolution)")

        # Liquidity risk
        low_liq = getattr(settings, "RISK_LOW_LIQUIDITY", 1000.0)
        mod_liq = getattr(settings, "RISK_MODERATE_LIQUIDITY", 5000.0)
        min_liquidity = min((m.liquidity for m in markets), default=0)
        if min_liquidity < low_liq:
            score += 0.3
            factors.append(f"Low liquidity (${min_liquidity:.0f})")
        elif min_liquidity < mod_liq:
            score += 0.15
            factors.append(f"Moderate liquidity (${min_liquidity:.0f})")

        # Number of markets (complexity risk) — slippage compounds per leg
        complex_legs = getattr(settings, "RISK_COMPLEX_LEGS", 5)
        multi_legs = getattr(settings, "RISK_MULTIPLE_LEGS", 3)
        if len(markets) > complex_legs:
            score += 0.2
            factors.append(f"Complex trade ({len(markets)} legs — high slippage risk)")
        elif len(markets) > multi_legs:
            score += 0.1
            factors.append(f"Multiple positions ({len(markets)} markets)")

        return min(score, 1.0), factors

    def _calculate_annualized_roi(self, roi_percent: float, resolution_date: Optional[datetime]) -> Optional[float]:
        """Calculate annualized ROI based on days to resolution.

        Returns None if resolution_date is unknown.
        """
        if not resolution_date:
            return None
        resolution_aware = make_aware(resolution_date)
        days_until = max((resolution_aware - utcnow()).total_seconds() / 86400.0, 1.0)
        return roi_percent * (365.0 / days_until)

    @staticmethod
    def fee_adjusted_edge_pct(edge_pct: float, entry_price: float, platform: str = "polymarket") -> float:
        """Return edge percent after platform taker fees."""

        if platform == "polymarket":
            fee = polymarket_taker_fee(entry_price)
        elif platform == "kalshi":
            fee = kalshi_taker_fee(entry_price)
        else:
            fee = 0.0
        return edge_pct - (fee * 100.0)

    def _build_execution_plan(
        self,
        *,
        positions: list[dict[str, Any]],
        markets: list[Market],
    ) -> ExecutionPlan | None:
        if not positions:
            return None

        config = getattr(self, "config", {}) or {}
        market_by_id = {str(m.id): m for m in markets}
        legs: list[ExecutionLeg] = []

        for index, position in enumerate(positions):
            market_id = str(
                position.get("market_id")
                or position.get("id")
                or position.get("market")
                or (markets[index].id if index < len(markets) else "")
            ).strip()
            if not market_id:
                continue

            market = market_by_id.get(market_id)
            action = str(position.get("action") or position.get("side") or "").strip().lower()
            side = "sell" if action.startswith("sell") else "buy"
            outcome = str(position.get("outcome") or "").strip().lower() or None
            limit_price = to_float(position.get("price"), 0.0)
            token_id = str(position.get("token_id") or "").strip() or None
            notional_weight = max(0.0001, to_float(position.get("notional_weight"), 1.0))
            min_fill_ratio = max(0.0, min(1.0, to_float(position.get("min_fill_ratio"), 0.0)))

            legs.append(
                ExecutionLeg(
                    leg_id=f"leg_{index + 1}",
                    market_id=market_id,
                    market_question=str(
                        position.get("market_question") or (market.question if market is not None else "")
                    )
                    or None,
                    token_id=token_id,
                    side=side,
                    outcome=outcome,
                    limit_price=limit_price if limit_price > 0 else None,
                    price_policy=str(position.get("price_policy") or config.get("price_policy") or "maker_limit"),
                    time_in_force=str(position.get("time_in_force") or config.get("time_in_force") or "GTC"),
                    post_only=bool(position.get("post_only") or config.get("post_only", False)),
                    notional_weight=notional_weight,
                    min_fill_ratio=min_fill_ratio,
                    metadata={
                        "position_index": index,
                        "raw_action": action or None,
                    },
                )
            )

        if not legs:
            return None

        policy = str(config.get("execution_policy") or ("PARALLEL_MAKER" if len(legs) > 1 else "SINGLE_LEG"))
        return ExecutionPlan(
            policy=policy,
            time_in_force=str(config.get("time_in_force") or "GTC"),
            legs=legs,
            constraints=ExecutionConstraints(
                max_unhedged_notional_usd=max(0.0, to_float(config.get("max_unhedged_notional_usd"), 0.0)),
                hedge_timeout_seconds=max(1, int(to_float(config.get("hedge_timeout_seconds"), 20))),
                session_timeout_seconds=max(1, int(to_float(config.get("session_timeout_seconds"), 300))),
                max_reprice_attempts=max(0, int(to_float(config.get("max_reprice_attempts"), 3))),
                pair_lock=bool(config.get("pair_lock", len(legs) > 1)),
                leg_fill_tolerance_ratio=max(
                    0.0,
                    min(1.0, to_float(config.get("leg_fill_tolerance_ratio"), 0.02)),
                ),
            ),
            metadata={
                "strategy_type": self.strategy_type,
                "generated_by": "base_strategy.create_opportunity",
            },
        )

    def create_opportunity(
        self,
        title: str,
        description: str,
        total_cost: float,
        markets: list[Market],
        positions: list[dict],
        event: Optional[Event] = None,
        expected_payout: float = 1.0,  # Override for strategies with non-$1 payouts
        is_guaranteed: bool = True,  # False for directional/statistical strategies
        # VWAP-adjusted parameters (all optional for backward compatibility)
        vwap_total_cost: Optional[float] = None,  # Realistic cost from order book
        spread_bps: Optional[float] = None,  # Actual spread in basis points
        fill_probability: Optional[float] = None,  # Probability all legs fill
        # Strategy-specific threshold overrides (high-freq crypto uses thinner books)
        min_liquidity_hard: Optional[float] = None,
        min_position_size: Optional[float] = None,
        min_absolute_profit: Optional[float] = None,
        confidence: Optional[float] = None,
        # ── Escape hatches for non-standard payout models ────────────────────
        # Use these when the default fee/ROI calculations don't apply to your
        # strategy's economics. Only set what you need; defaults are fine otherwise.
        skip_fee_model: bool = False,
        # Pass a pre-computed ROI percent (e.g. 12.5 for 12.5%) to override the
        # standard (net_profit / total_cost) ROI calculation.
        custom_roi_percent: Optional[float] = None,
        # Pass a pre-computed risk score [0, 1] to bypass calculate_risk_score().
        custom_risk_score: Optional[float] = None,
    ) -> Optional[Opportunity]:
        """Create an Opportunity with fee/risk enrichment.

        Args:
            skip_fee_model: When True, uses a zero fee_breakdown. Use for
                directional NO-bet strategies (e.g. Miracle) where the fee
                structure is fundamentally different from the standard
                guaranteed-spread model.
            custom_roi_percent: Pre-computed ROI in percent. When provided,
                overrides the standard (net_profit / total_cost) calculation.
            custom_risk_score: Pre-computed risk score [0, 1]. When provided,
                bypasses calculate_risk_score(). risk_factors will be empty
                unless you set them on the returned opportunity afterward.
        """

        if total_cost <= 0:
            return None

        gross_profit = expected_payout - total_cost

        # --- Comprehensive fee model (gas, spread, multi-leg slippage) ---
        is_negrisk = any(getattr(m, "neg_risk", False) for m in markets)
        total_liquidity = sum(m.liquidity for m in markets)
        entry_prices: list[float] = []
        for position in positions:
            raw_price = position.get("price") if isinstance(position, dict) else None
            parsed_price = to_float(raw_price)
            if parsed_price is None:
                continue
            if parsed_price <= 0:
                continue
            entry_prices.append(float(parsed_price))
        market_platforms = {
            str(getattr(m, "platform", "") or "").strip().lower()
            for m in markets
            if str(getattr(m, "platform", "") or "").strip()
        }
        platform = market_platforms.pop() if len(market_platforms) == 1 else "polymarket"
        if skip_fee_model:
            # Directional/non-standard payout — zero-fee breakdown preserves the
            # _fee_breakdown metadata key without polluting the ROI calculation.
            class _ZeroFees:
                winner_fee = 0.0
                gas_cost_usd = 0.0
                spread_cost = 0.0
                multi_leg_slippage = 0.0
                total_fees = 0.0
                fee_as_pct_of_payout = 0.0

            fee_breakdown = _ZeroFees()
        else:
            fee_breakdown = fee_model.calculate_fees(
                expected_payout=expected_payout,
                num_legs=len(markets),
                is_negrisk=is_negrisk,
                spread_bps=spread_bps,
                total_cost=total_cost,
                maker_mode=settings.FEE_MODEL_MAKER_MODE,
                total_liquidity=total_liquidity,
                platform=platform,
                entry_prices=entry_prices,
            )

        fee = float(fee_breakdown.total_fees)
        net_profit = gross_profit - fee
        roi = (net_profit / total_cost) * 100

        # --- VWAP-adjusted realistic profit ---
        realistic_cost = vwap_total_cost if vwap_total_cost is not None else total_cost
        realistic_gross = expected_payout - realistic_cost
        realistic_net = realistic_gross - fee_breakdown.total_fees
        realistic_roi = (realistic_net / realistic_cost) * 100 if realistic_cost > 0 else 0

        # Use realistic profit for filtering when VWAP data is available
        # (kept for enrichment metadata; hard filters are now in QualityFilterPipeline)

        # Apply custom_roi_percent override (pre-computed by strategy from model)
        if custom_roi_percent is not None:
            roi = float(custom_roi_percent)

        min_roi_percent = float(self.min_profit or 0.0) * 100.0
        if roi < min_roi_percent:
            return None

        if not is_guaranteed and roi > 120.0:
            return None

        # Calculate max position size based on liquidity
        min_liquidity = min((m.liquidity for m in markets), default=0)
        max_position = min_liquidity * 0.1  # Don't exceed 10% of liquidity

        # Calculate risk — use custom_risk_score if provided, otherwise compute
        resolution_date = None
        if markets and markets[0].end_date:
            resolution_date = markets[0].end_date

        if custom_risk_score is not None:
            risk_score = float(max(0.0, min(1.0, custom_risk_score)))
            risk_factors: list[str] = []
        else:
            risk_score, risk_factors = self.calculate_risk_score(markets, resolution_date)

        confidence_score = to_confidence(confidence if confidence is not None else 0.5, 0.5)

        # Build enriched market dicts with VWAP metadata
        market_dicts = []
        for m in markets:
            # Extract outcome labels from tokens (supports multi-outcome markets)
            outcome_labels = [t.outcome for t in m.tokens] if m.tokens else []
            outcome_prices = list(m.outcome_prices) if m.outcome_prices else []
            entry: dict = {
                "id": m.id,
                "condition_id": getattr(m, "condition_id", ""),
                "slug": m.slug,
                "question": m.question,
                "group_item_title": getattr(m, "group_item_title", ""),
                "event_slug": getattr(m, "event_slug", ""),
                "event_ticker": getattr(m, "event_slug", ""),
                "clob_token_ids": list(getattr(m, "clob_token_ids", []) or []),
                "yes_price": m.yes_price,
                "no_price": m.no_price,
                "outcome_labels": outcome_labels,
                "outcome_prices": outcome_prices,
                "liquidity": m.liquidity,
                "volume": getattr(m, "volume", 0.0),
                "platform": getattr(m, "platform", "polymarket"),
            }
            market_dicts.append(entry)

        # Attach realistic-profit metadata to positions_to_take
        enriched_positions = list(positions)  # shallow copy
        if enriched_positions:
            enriched_positions[0] = {
                **enriched_positions[0],
                "_fee_breakdown": {
                    "winner_fee": fee_breakdown.winner_fee,
                    "gas_cost_usd": fee_breakdown.gas_cost_usd,
                    "spread_cost": fee_breakdown.spread_cost,
                    "multi_leg_slippage": fee_breakdown.multi_leg_slippage,
                    "total_fees": fee_breakdown.total_fees,
                    "fee_as_pct_of_payout": fee_breakdown.fee_as_pct_of_payout,
                },
                "_realistic_profit": {
                    "vwap_total_cost": vwap_total_cost,
                    "realistic_gross": realistic_gross,
                    "realistic_net": realistic_net,
                    "realistic_roi": realistic_roi,
                    "theoretical_roi": roi,
                    "spread_bps": spread_bps,
                    "fill_probability": fill_probability,
                },
            }

        roi_type = "guaranteed_spread" if is_guaranteed else "directional_payout"

        return Opportunity(
            strategy=self.strategy_type,
            title=title,
            description=description,
            total_cost=total_cost,
            expected_payout=expected_payout,
            gross_profit=gross_profit,
            fee=fee,
            net_profit=net_profit,
            roi_percent=roi,
            is_guaranteed=is_guaranteed,
            roi_type=roi_type,
            risk_score=risk_score,
            risk_factors=risk_factors,
            confidence=confidence_score,
            markets=market_dicts,
            event_id=event.id if event else None,
            event_slug=event.slug if event else None,
            event_title=event.title if event else None,
            category=event.category if event else None,
            min_liquidity=min_liquidity,
            max_position_size=max_position,
            resolution_date=resolution_date,
            positions_to_take=enriched_positions,
            execution_plan=self._build_execution_plan(
                positions=enriched_positions,
                markets=markets,
            ),
        )
