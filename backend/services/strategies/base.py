from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone

from models import Market, Event, ArbitrageOpportunity
from config import settings
from services.fee_model import fee_model


def utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime"""
    return datetime.now(timezone.utc)


def make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Make a datetime timezone-aware (UTC) if it isn't already"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


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


class BaseStrategy(ABC):
    """Base class for arbitrage detection strategies.

    Strategies set strategy_type to their unique slug string.

    Subclasses must implement at least one of:

    * ``detect()`` -- synchronous detection, run in a thread-pool executor.
    * ``detect_async()`` -- async detection (preferred for I/O-bound work
      such as LLM calls, HTTP requests, or database queries).  Awaited
      directly on the event loop.

    If a strategy implements ``detect_async()``, the scanner will call it
    instead of ``detect()``.  The default ``detect_async()`` falls back to
    ``detect()`` so existing sync strategies work without changes.
    """

    strategy_type: str  # strategy slug identifier
    name: str
    description: str
    key: str = ""  # Set to strategy_type or slug; used by orchestrator for lookups

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

    def __init__(self):
        self.fee = settings.POLYMARKET_FEE
        self.min_profit = settings.MIN_PROFIT_THRESHOLD
        # Strategy state — persisted across detect() cycles
        self._state: dict = {}

    def detect(self, events: list[Event], markets: list[Market], prices: dict[str, dict]) -> list[ArbitrageOpportunity]:
        """Detect arbitrage opportunities (sync).

        Called every scan cycle with the full set of active events, markets,
        and live CLOB prices.  Override this for CPU-bound strategies that
        don't need async I/O.

        The default implementation returns an empty list.  Subclasses should
        override this method or ``detect_async()`` (or both).
        """
        return []

    async def detect_async(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Detect arbitrage opportunities (async, preferred for I/O-bound work).

        The scanner calls this method when it exists.  The default
        implementation delegates to the synchronous ``detect()`` so
        existing strategies continue to work without modification.

        Override this (instead of ``detect()``) when your strategy needs
        to ``await`` coroutines -- e.g. LLM calls via ``services.ai``,
        HTTP requests via ``httpx``, or database queries.
        """
        return self.detect(events, markets, prices)

    def configure(self, config: dict) -> None:
        """Apply user config overrides. Called by the loader after instantiation."""
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

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        """Decide whether to trade a signal RIGHT NOW.

        Called by the orchestrator when a pending signal from this strategy
        is ready for execution. Override to add custom gating logic.

        Default: passthrough — trusts the detection layer's edge/confidence
        and returns 'selected' with base sizing from config.

        Args:
            signal: TradeSignal object with .edge_percent, .confidence,
                    .direction, .entry_price, .payload_json, .strategy_context
            context: {
                "params": dict,        # Strategy config (merged default + user overrides)
                "trader": object,      # Trader ORM row
                "mode": str,           # "paper" or "live"
                "live_market": dict,   # Live CLOB prices (if available)
                "source_config": dict, # Source config from trader
            }

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

    def default_exit_check(self, position: Any, market_state: dict) -> ExitDecision:
        """Standard TP/SL/trailing/max-hold exit logic.

        Strategies can call this as a fallback after their custom checks.
        Uses config params: take_profit_pct, stop_loss_pct, trailing_stop_pct,
        max_hold_minutes, min_hold_minutes, resolve_only.
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

        # Take profit
        tp = config.get("take_profit_pct")
        if tp is not None and pnl_pct >= float(tp):
            return ExitDecision("close", f"Take profit hit ({pnl_pct:.1f}% >= {tp}%)", close_price=current_price)

        # Stop loss
        sl = config.get("stop_loss_pct")
        if sl is not None and pnl_pct <= -abs(float(sl)):
            return ExitDecision("close", f"Stop loss hit ({pnl_pct:.1f}% <= -{sl}%)", close_price=current_price)

        # Trailing stop
        trailing = config.get("trailing_stop_pct")
        highest = float(getattr(position, "highest_price", 0) or 0)
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

        # Multi-leg execution risk
        num_legs = len(markets)
        if num_legs > 1:
            # Each additional leg adds ~5% probability of partial fill
            partial_fill_risk = 1 - (0.95 ** (num_legs - 1))
            score += partial_fill_risk * 0.3  # Weight execution risk
            if partial_fill_risk > 0.2:
                factors.append(
                    f"Multi-leg execution risk ({num_legs} legs, {partial_fill_risk:.0%} partial fill chance)"
                )
            elif num_legs > 1:
                factors.append(f"Multi-leg trade ({num_legs} legs)")

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
    ) -> Optional[ArbitrageOpportunity]:
        """Create an ArbitrageOpportunity if profitable.

        Applies hard rejection filters:
        1. ROI must exceed MIN_PROFIT_THRESHOLD
        2. Min liquidity must exceed MIN_LIQUIDITY_HARD (or min_liquidity_hard override)
        3. Max position size must exceed MIN_POSITION_SIZE (or min_position_size override)
        4. Absolute profit on max position must exceed MIN_ABSOLUTE_PROFIT (or override)
        5. Annualized ROI must exceed MIN_ANNUALIZED_ROI (if resolution date known)
        6. Resolution must be within MAX_RESOLUTION_MONTHS
        """

        gross_profit = expected_payout - total_cost
        fee = expected_payout * self.fee
        net_profit = gross_profit - fee
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        # --- Comprehensive fee model (gas, spread, multi-leg slippage) ---
        is_negrisk = any(getattr(m, "neg_risk", False) for m in markets)
        total_liquidity = sum(m.liquidity for m in markets)
        fee_breakdown = fee_model.calculate_fees(
            expected_payout=expected_payout,
            num_legs=len(markets),
            is_negrisk=is_negrisk,
            spread_bps=spread_bps,
            total_cost=total_cost,
            maker_mode=settings.FEE_MODEL_MAKER_MODE,
            total_liquidity=total_liquidity,
        )

        # --- VWAP-adjusted realistic profit ---
        realistic_cost = vwap_total_cost if vwap_total_cost is not None else total_cost
        realistic_gross = expected_payout - realistic_cost
        realistic_net = realistic_gross - fee_breakdown.total_fees
        realistic_roi = (realistic_net / realistic_cost) * 100 if realistic_cost > 0 else 0

        # Use realistic profit for filtering when VWAP data is available
        effective_roi = realistic_roi if vwap_total_cost is not None else roi

        # Check if profitable after fees
        if effective_roi < self.min_profit * 100:
            return None

        # --- Hard filter: implausible directional ROI ---
        # Directional strategies are not guaranteed spreads. Extremely high
        # ROI readings here are usually payout-model artifacts (e.g. buying a
        # $0.05 leg and treating it like guaranteed $1 settlement).
        if not is_guaranteed:
            # Keep directional ROI guard fixed and conservative. Tying this
            # directly to configurable arbitrage ROI caps can let clearly
            # implausible directional opportunities leak through.
            directional_roi_cap = 120.0
            if effective_roi > directional_roi_cap:
                return None

        # --- Hard filter: suspiciously high ROI ---
        # In efficient prediction markets, genuine arbitrage is 1-5%.
        # ROI > 30% almost always indicates non-exhaustive outcomes, stale
        # order books, or missing data — not a real mispricing.
        # Skip this check for directional/statistical strategies where
        # "ROI" represents potential return, not guaranteed profit.
        if is_guaranteed and effective_roi > settings.MAX_PLAUSIBLE_ROI:
            return None

        # --- Hard filter: too many legs ---
        # Slippage compounds per leg. An 8-leg trade where each leg has
        # even 0.5% slippage loses 4% of margin before execution completes.
        if len(markets) > settings.MAX_TRADE_LEGS:
            return None

        # --- Hard filter: minimum liquidity per leg ---
        # Multi-leg trades need proportionally more liquidity to avoid
        # slippage cascading across legs.
        min_liq_per_leg = getattr(settings, "MIN_LIQUIDITY_PER_LEG", 500.0)
        if len(markets) > 1:
            required_liquidity = min_liq_per_leg * len(markets)
            if total_liquidity < required_liquidity:
                return None

        # Calculate max position size based on liquidity
        min_liquidity = min((m.liquidity for m in markets), default=0)
        max_position = min_liquidity * 0.1  # Don't exceed 10% of liquidity

        # Resolve thresholds (allow strategy-specific overrides for thin-book markets)
        eff_min_liquidity = min_liquidity_hard if min_liquidity_hard is not None else settings.MIN_LIQUIDITY_HARD
        eff_min_position = min_position_size if min_position_size is not None else settings.MIN_POSITION_SIZE
        eff_min_absolute = min_absolute_profit if min_absolute_profit is not None else settings.MIN_ABSOLUTE_PROFIT

        # --- Hard filter: minimum liquidity ---
        if min_liquidity < eff_min_liquidity:
            return None

        # --- Hard filter: minimum position size ---
        if max_position < eff_min_position:
            return None

        # --- Hard filter: minimum absolute profit ---
        # Use realistic net profit for the absolute profit check when available
        effective_net = realistic_net if vwap_total_cost is not None else net_profit
        effective_cost = realistic_cost if vwap_total_cost is not None else total_cost
        absolute_profit = max_position * (effective_net / effective_cost) if effective_cost > 0 else 0
        if absolute_profit < eff_min_absolute:
            return None

        # Calculate risk
        resolution_date = None
        if markets and markets[0].end_date:
            resolution_date = markets[0].end_date

        # --- Hard filter: maximum resolution timeframe ---
        if resolution_date:
            resolution_aware = make_aware(resolution_date)
            days_until = (resolution_aware - utcnow()).total_seconds() / 86400.0
            max_days = settings.MAX_RESOLUTION_MONTHS * 30
            if days_until > max_days:
                return None

            # --- Hard filter: minimum annualized ROI ---
            annualized_roi = self._calculate_annualized_roi(effective_roi, resolution_date)
            if annualized_roi is not None and annualized_roi < settings.MIN_ANNUALIZED_ROI:
                return None

        risk_score, risk_factors = self.calculate_risk_score(markets, resolution_date)

        # Build enriched market dicts with VWAP metadata
        market_dicts = []
        for m in markets:
            # Extract outcome labels from tokens (supports multi-outcome markets)
            outcome_labels = [t.outcome for t in m.tokens] if m.tokens else []
            outcome_prices = list(m.outcome_prices) if m.outcome_prices else []
            entry: dict = {
                "id": m.id,
                "slug": m.slug,
                "question": m.question,
                "group_item_title": getattr(m, "group_item_title", ""),
                "event_slug": getattr(m, "event_slug", ""),
                "event_ticker": getattr(m, "event_slug", ""),
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

        return ArbitrageOpportunity(
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
            markets=market_dicts,
            event_id=event.id if event else None,
            event_slug=event.slug if event else None,
            event_title=event.title if event else None,
            category=event.category if event else None,
            min_liquidity=min_liquidity,
            max_position_size=max_position,
            resolution_date=resolution_date,
            positions_to_take=enriched_positions,
        )
