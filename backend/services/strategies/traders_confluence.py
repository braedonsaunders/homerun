"""
Traders Confluence Strategy

Evaluates tracked trader confluence signals to detect actionable opportunities
where multiple smart wallets are converging on the same market position.

Unlike scanner strategies that detect structural mispricings, this strategy
detects BEHAVIORAL mispricings based on smart money flow patterns.

Pipeline:
  1. Wallet tracker monitors known profitable wallets
  2. Confluence engine detects multi-wallet convergence
  3. Signal bus emits trader flow signals
  4. This strategy filters signals and converts to ArbitrageOpportunity
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from models import ArbitrageOpportunity, Event, Market, Token
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, ScoringWeights, SizingConfig, ExitDecision
from services.data_events import DataEvent
from services.strategy_sdk import StrategySDK
from utils.converters import to_confidence
from functools import partial
from utils.converters import safe_float

logger = logging.getLogger(__name__)

_safe_float_nan = partial(safe_float, reject_nan_inf=True)


class TradersConfluenceStrategy(BaseStrategy):
    """
    Traders Confluence Strategy

    Detects opportunities where multiple tracked profitable wallets
    are converging on the same market position, filtered by
    configurable quality gates.
    """

    strategy_type = "traders_confluence"
    name = "Traders Confluence"
    description = "Detect smart money convergence via tracked wallet confluence analysis"
    mispricing_type = "cross_market"
    source_key = "traders"
    worker_affinity = "traders"
    allow_deduplication = False
    subscriptions = ["trader_activity"]

    # Default config thresholds (overridden by DB config column)
    DEFAULT_CONFIG = {
        "min_edge_percent": 3.0,
        "min_confidence": 0.45,
        "min_confluence_strength": 0.50,
        "min_tier": "high",
        "min_wallet_count": 2,
        "max_entry_price": 0.85,
        "risk_base_score": 0.40,
        "firehose_require_tradable_market": True,
        "firehose_exclude_crypto_markets": True,
        "firehose_require_qualified_source": True,
        "firehose_max_age_minutes": 180,
        "firehose_source_scope": "all",
        "firehose_side_filter": "all",
    }
    default_config = dict(DEFAULT_CONFIG)

    TIER_ORDER = {"low": 0, "medium": 1, "high": 2, "extreme": 3}

    def __init__(self):
        super().__init__()
        self._config = dict(self.DEFAULT_CONFIG)

    def configure(self, config: dict) -> None:
        """Apply user config overrides from the DB config column."""
        if config:
            for key in self.DEFAULT_CONFIG:
                if key in config:
                    self._config[key] = config[key]

    def _effective_config(self) -> dict:
        cfg = dict(self.DEFAULT_CONFIG)
        if isinstance(self._config, dict):
            cfg.update(self._config)
        runtime_cfg = getattr(self, "config", None)
        if isinstance(runtime_cfg, dict):
            cfg.update(runtime_cfg)
        cfg.update(StrategySDK.validate_trader_filter_config(cfg))
        return cfg

    @staticmethod
    def _to_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _has_direction(signal: dict) -> bool:
        outcome = str(signal.get("outcome") or "").strip().upper()
        if outcome in {"YES", "NO"}:
            return True
        signal_type = str(signal.get("signal_type") or "").strip().lower()
        return "buy" in signal_type or "sell" in signal_type or "accumulation" in signal_type

    @staticmethod
    def _price_checks(signal: dict) -> tuple[bool, bool]:
        prices = []
        for key in ("avg_entry_price", "entry_price", "yes_price", "no_price"):
            raw = signal.get(key)
            if raw is None:
                continue
            try:
                parsed = float(raw)
            except Exception:
                continue
            prices.append(parsed)
        if not prices:
            return False, False
        return True, all(0.0 <= price <= 1.0 for price in prices)

    @staticmethod
    def _confidence(signal: dict) -> float:
        direct = signal.get("firehose_confidence")
        if direct is not None:
            return max(0.0, min(1.0, _safe_float_nan(direct, 0.0)))

        strength = signal.get("strength")
        if strength is not None:
            return max(0.0, min(1.0, _safe_float_nan(strength, 0.0)))

        conviction = _safe_float_nan(signal.get("conviction_score"), 0.0)
        if conviction > 1.0:
            conviction = conviction / 100.0
        return max(0.0, min(1.0, conviction))

    @staticmethod
    def _tier_value(tier: object) -> int:
        normalized = StrategySDK.normalize_trader_tier(tier, default="low")
        return TradersConfluenceStrategy.TIER_ORDER.get(normalized, 0)

    @staticmethod
    def _parse_dt(value: object) -> Optional[datetime]:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            try:
                return datetime.fromtimestamp(float(text), tz=timezone.utc)
            except Exception:
                return None

    def evaluate_firehose_signal(
        self,
        signal: dict,
        *,
        cfg: Optional[dict] = None,
    ) -> tuple[bool, list[str], dict[str, bool]]:
        """Evaluate one traders firehose row.

        This gate is the source of truth for Traders opportunities filtering.
        """
        config = cfg or self._effective_config()
        signal = StrategySDK.normalize_trader_signal(signal if isinstance(signal, dict) else {})
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        market_id = str(signal.get("market_id") or "").strip()
        checks["has_market_id"] = bool(market_id)
        if not checks["has_market_id"]:
            reasons.append("missing_market_id")

        wallets = signal.get("wallets") or []
        checks["has_wallets"] = bool(wallets)
        if not checks["has_wallets"]:
            reasons.append("missing_wallets")

        checks["has_direction"] = self._has_direction(signal)
        if not checks["has_direction"]:
            reasons.append("missing_direction")

        has_price_reference, price_in_bounds = self._price_checks(signal)
        checks["has_price_reference"] = has_price_reference
        checks["price_in_bounds"] = price_in_bounds
        if not has_price_reference:
            reasons.append("missing_price_reference")
        elif not price_in_bounds:
            reasons.append("price_out_of_bounds")

        source_flags = signal.get("source_flags") or {}
        qualified = bool(source_flags.get("qualified", True))
        require_qualified = self._to_bool(config.get("firehose_require_qualified_source"), True)
        checks["has_qualified_source"] = qualified or (not require_qualified)
        if require_qualified and not qualified:
            reasons.append("unqualified_wallet_source")

        scope = StrategySDK.normalize_trader_source_scope(config.get("firehose_source_scope"), default="all")
        from_tracked = bool(source_flags.get("from_tracked_traders") or source_flags.get("from_trader_groups"))
        from_pool = bool(source_flags.get("from_pool"))
        if scope == "tracked":
            source_scope_ok = from_tracked
        elif scope == "pool":
            source_scope_ok = from_pool
        else:
            source_scope_ok = from_tracked or from_pool
        checks["matches_source_scope"] = source_scope_ok
        if not source_scope_ok:
            reasons.append("source_scope_mismatch")

        upstream_tradable = bool(signal.get("firehose_market_tradable", signal.get("is_tradeable", True)))
        require_tradable = self._to_bool(config.get("firehose_require_tradable_market"), True)
        checks["upstream_tradable"] = upstream_tradable or (not require_tradable)
        if require_tradable and not upstream_tradable:
            reasons.append("market_not_tradable")

        is_crypto = bool(signal.get("firehose_is_crypto", False))
        exclude_crypto = self._to_bool(config.get("firehose_exclude_crypto_markets"), True)
        checks["not_crypto_market"] = (not is_crypto) or (not exclude_crypto)
        if exclude_crypto and is_crypto:
            reasons.append("crypto_market_excluded")

        max_age_minutes = _safe_float_nan(config.get("firehose_max_age_minutes"), 0.0)
        age_minutes = _safe_float_nan(signal.get("firehose_age_minutes"), 0.0)
        checks["within_age_limit"] = max_age_minutes <= 0 or age_minutes <= max_age_minutes
        if max_age_minutes > 0 and age_minutes > max_age_minutes:
            reasons.append("signal_too_old")

        confidence = self._confidence(signal)
        min_conf = _safe_float_nan(config.get("min_confidence"), 0.0)
        checks["meets_min_confidence"] = confidence >= min_conf
        if confidence < min_conf:
            reasons.append("confidence_below_threshold")

        min_tier = self._tier_value(config.get("min_tier", "high"))
        signal_tier = self._tier_value(signal.get("tier", "low"))
        checks["meets_min_tier"] = signal_tier >= min_tier
        if signal_tier < min_tier:
            reasons.append("tier_below_threshold")

        side_filter = StrategySDK.normalize_trader_side(config.get("firehose_side_filter"), default="all")
        signal_side = StrategySDK.normalize_trader_side(signal.get("side"), default="all")
        checks["matches_side_filter"] = side_filter == "all" or side_filter == signal_side
        if side_filter != "all" and side_filter != signal_side:
            reasons.append("side_filtered")

        min_wallet_count = max(1, int(_safe_float_nan(config.get("min_wallet_count"), 2)))
        wallet_count = int(_safe_float_nan(signal.get("cluster_adjusted_wallet_count"), 0))
        if wallet_count <= 0:
            wallet_count = int(_safe_float_nan(signal.get("wallet_count"), 0))
        checks["meets_min_wallet_count"] = wallet_count >= min_wallet_count
        if wallet_count < min_wallet_count:
            reasons.append("insufficient_wallet_count")

        max_entry_price = _safe_float_nan(config.get("max_entry_price"), 1.0)
        entry_price = signal.get("avg_entry_price")
        if entry_price is None:
            entry_price = signal.get("entry_price")
        parsed_entry = _safe_float_nan(entry_price, 1.0)
        checks["entry_price_within_bounds"] = parsed_entry <= max_entry_price
        if parsed_entry > max_entry_price:
            reasons.append("entry_price_too_high")

        passed = len(reasons) == 0
        return passed, reasons, checks

    def apply_firehose_filters(
        self,
        signals: list[dict],
        *,
        include_filtered: bool = False,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Filter and annotate traders firehose rows for UI/worker consumption."""
        cfg = self._effective_config()
        output: list[dict] = []

        for signal in signals:
            if not isinstance(signal, dict):
                continue
            row = StrategySDK.normalize_trader_signal(dict(signal))
            passed, reasons, checks = self.evaluate_firehose_signal(row, cfg=cfg)
            row["validation"] = {
                "is_valid": passed,
                "is_actionable": passed,
                "is_tradeable": passed,
                "checks": checks,
                "reasons": list(reasons),
            }
            row["is_valid"] = passed
            row["is_actionable"] = passed
            row["is_tradeable"] = passed
            row["validation_reasons"] = list(reasons)

            if passed or include_filtered:
                output.append(row)

        output.sort(
            key=lambda signal: (
                _safe_float_nan(signal.get("conviction_score"), 0.0),
                _safe_float_nan(
                    signal.get("cluster_adjusted_wallet_count", signal.get("wallet_count")),
                    0.0,
                ),
                str(signal.get("last_seen_at") or signal.get("detected_at") or ""),
            ),
            reverse=True,
        )

        if limit is not None:
            try:
                safe_limit = max(1, int(limit))
            except Exception:
                safe_limit = 50
            return output[:safe_limit]
        return output

    def _build_signal_market(self, signal: dict) -> Market:
        market_id = str(signal.get("market_id") or "").strip()
        question = str(signal.get("market_question") or market_id or "Unknown market").strip()
        slug = str(signal.get("market_slug") or market_id).strip()

        side = StrategySDK.normalize_trader_side(signal.get("side"), default="all")
        entry = _safe_float_nan(signal.get("avg_entry_price"), 0.5)
        yes_price = _safe_float_nan(signal.get("yes_price"), -1.0)
        no_price = _safe_float_nan(signal.get("no_price"), -1.0)
        if yes_price < 0.0 or no_price < 0.0:
            if side == "buy":
                yes_price = entry
                no_price = max(0.0, min(1.0, 1.0 - entry))
            elif side == "sell":
                no_price = entry
                yes_price = max(0.0, min(1.0, 1.0 - entry))
            else:
                yes_price = entry
                no_price = max(0.0, min(1.0, 1.0 - entry))
        yes_price = max(0.0, min(1.0, yes_price))
        no_price = max(0.0, min(1.0, no_price))

        yes_token = str(signal.get("yes_token_id") or signal.get("yes_token") or "").strip()
        no_token = str(signal.get("no_token_id") or signal.get("no_token") or "").strip()
        tokens = []
        if yes_token:
            tokens.append(Token(token_id=yes_token, outcome="Yes", price=yes_price))
        if no_token:
            tokens.append(Token(token_id=no_token, outcome="No", price=no_price))

        return Market(
            id=market_id,
            condition_id=market_id,
            question=question,
            slug=slug,
            event_slug=str(signal.get("event_slug") or "").strip(),
            tokens=tokens,
            clob_token_ids=[t.token_id for t in tokens],
            outcome_prices=[yes_price, no_price],
            active=bool(signal.get("is_active", True)),
            closed=False,
            volume=max(0.0, _safe_float_nan(signal.get("market_volume_24h"), 0.0)),
            liquidity=max(0.0, _safe_float_nan(signal.get("market_liquidity"), 0.0)),
            tags=list(signal.get("market_tags") or []),
            platform=str(signal.get("platform") or "polymarket"),
        )

    def _infer_entry_price(self, signal: dict, side: str) -> float:
        direct = _safe_float_nan(signal.get("avg_entry_price"), -1.0)
        if direct >= 0.0:
            return max(0.0, min(1.0, direct))
        if side == "YES":
            return max(0.0, min(1.0, _safe_float_nan(signal.get("yes_price"), 0.5)))
        return max(0.0, min(1.0, _safe_float_nan(signal.get("no_price"), 0.5)))

    def build_opportunities_from_firehose(
        self,
        rows: list[dict],
        *,
        limit: Optional[int] = None,
    ) -> list[ArbitrageOpportunity]:
        if not rows:
            return []

        cfg = self._effective_config()
        opportunities: list[ArbitrageOpportunity] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            signal = StrategySDK.normalize_trader_signal(dict(raw))
            passed, reasons, checks = self.evaluate_firehose_signal(signal, cfg=cfg)
            if not passed:
                continue

            side_label = "YES" if signal.get("side") == "buy" else "NO"
            direction = "buy_yes" if side_label == "YES" else "buy_no"
            market = self._build_signal_market(signal)
            confidence = self._confidence(signal)
            strength = max(0.0, min(1.0, _safe_float_nan(signal.get("strength"), confidence)))
            wallet_count = int(
                _safe_float_nan(signal.get("cluster_adjusted_wallet_count"), _safe_float_nan(signal.get("wallet_count"), 0))
            )
            tier = StrategySDK.normalize_trader_tier(signal.get("tier"), default="low")
            edge_percent = _safe_float_nan(signal.get("edge_percent"), 0.0)
            if edge_percent <= 0.0:
                edge_percent = max(float(cfg.get("min_edge_percent", 3.0)), confidence * 100.0 * 0.12)
            entry_price = self._infer_entry_price(signal, side_label)
            expected_delta = min(0.4, max(0.05, (edge_percent / 100.0) + 0.03))
            expected_payout = min(1.0, entry_price + expected_delta)
            if expected_payout <= entry_price:
                expected_payout = min(1.0, entry_price + 0.05)

            risk_score = float(cfg.get("risk_base_score", 0.4))
            if confidence < 0.6:
                risk_score = min(1.0, risk_score + 0.15)
            if wallet_count <= 2:
                risk_score = min(1.0, risk_score + 0.1)
            if tier == "high":
                risk_score = max(0.0, risk_score - 0.05)
            elif tier == "extreme":
                risk_score = max(0.0, risk_score - 0.1)

            token_id = None
            if market.clob_token_ids:
                idx = 0 if side_label == "YES" else (1 if len(market.clob_token_ids) > 1 else 0)
                token_id = market.clob_token_ids[idx]

            opportunity = self.create_opportunity(
                title=f"Trader Flow: {wallet_count} wallets -> {market.question[:56]}",
                description=(
                    f"{wallet_count} tracked wallets ({tier} tier, {strength:.0%} confluence) "
                    f"converging on {side_label} at ${entry_price:.2f}."
                ),
                total_cost=entry_price,
                expected_payout=expected_payout,
                markets=[market],
                positions=[
                    {
                        "action": "BUY",
                        "outcome": side_label,
                        "price": entry_price,
                        "token_id": token_id,
                        "direction": direction,
                    }
                ],
                event=Event(
                    id=str(signal.get("event_id") or ""),
                    slug=str(signal.get("event_slug") or ""),
                    title=str(signal.get("event_title") or market.question),
                    category=str(signal.get("category") or "") or None,
                    markets=[market],
                ),
                is_guaranteed=False,
                custom_roi_percent=edge_percent,
                custom_risk_score=risk_score,
                confidence=confidence,
            )
            if opportunity is None:
                continue

            detected_at = self._parse_dt(signal.get("detected_at"))
            last_seen_at = self._parse_dt(signal.get("last_seen_at"))
            if detected_at is not None:
                opportunity.detected_at = detected_at
            if last_seen_at is not None:
                opportunity.last_seen_at = last_seen_at

            opportunity.strategy_context = {
                "source_key": "traders",
                "strategy_slug": self.strategy_type,
                "signal_id": str(signal.get("id") or ""),
                "tier": tier,
                "side": signal.get("side"),
                "outcome": str(signal.get("outcome") or "").strip().upper(),
                "wallet_count": wallet_count,
                "confidence": confidence,
                "confluence_strength": strength,
                "edge_percent": edge_percent,
                "source_flags": signal.get("source_flags") or {},
                "source_breakdown": signal.get("source_breakdown") or {},
                "validation": {
                    "is_valid": True,
                    "is_actionable": True,
                    "is_tradeable": True,
                    "checks": checks,
                    "reasons": reasons,
                },
                "firehose": signal,
            }
            opportunity.risk_factors = [
                "Smart money convergence bet (behavioral edge)",
                f"Confluence: {strength:.0%} strength, {wallet_count} wallets, tier {tier}",
            ]
            opportunity.mispricing_type = MispricingType.NEWS_INFORMATION
            opportunities.append(opportunity)

        opportunities.sort(
            key=lambda opp: (
                float(opp.confidence or 0.0),
                float(opp.roi_percent or 0.0),
                str(opp.detected_at or ""),
            ),
            reverse=True,
        )
        if limit is not None:
            try:
                safe_limit = max(1, int(limit))
            except Exception:
                safe_limit = 50
            return opportunities[:safe_limit]
        return opportunities

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """Sync detect - returns empty.

        TradersConfluenceStrategy is fed by the tracked trader pipeline,
        not the main scanner loop. Confluence signals are converted to
        opportunities by detect_from_signals().
        """
        return []

    def detect_from_signals(
        self,
        signals: list[dict],
        markets: list[Market],
        events: list[Event],
    ) -> list[ArbitrageOpportunity]:
        """Convert trader confluence signals into ArbitrageOpportunity objects.

        Each signal dict should contain:
          - market_id: str
          - direction: "buy_yes" | "buy_no"
          - edge_percent: float
          - confidence: float
          - confluence_strength: float (0-1)
          - tier: "low" | "medium" | "high" | "extreme"
          - wallet_count: int
          - total_volume_usd: float
          - wallets: list[str] (wallet addresses/labels)
          - entry_price: float
          - target_price: float
        """
        if not signals:
            return []
        filtered = self.apply_firehose_filters(signals, include_filtered=False, limit=None)
        return self.build_opportunities_from_firehose(filtered, limit=None)

    # ------------------------------------------------------------------
    # on_event()  (event-driven detection from tracked_traders_worker)
    # ------------------------------------------------------------------

    async def on_event(self, event: DataEvent) -> list[ArbitrageOpportunity]:
        if event.event_type != "trader_activity":
            return []
        signals = event.payload.get("signals") or []
        if not signals:
            return []
        filtered = self.apply_firehose_filters(signals, include_filtered=False, limit=None)
        return self.build_opportunities_from_firehose(filtered, limit=None)

    # ------------------------------------------------------------------
    # Evaluate / Should-Exit  (unified strategy interface)
    # ------------------------------------------------------------------

    scoring_weights = ScoringWeights()
    sizing_config = SizingConfig()

    _confluence_strength: float = 0.0

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        source = str(getattr(signal, "source", "") or "").strip().lower()

        payload_channel = str(payload.get("traders_channel") or "").strip().lower()
        if payload_channel:
            channel = payload_channel
        else:
            signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()
            if signal_type == "confluence":
                channel = "confluence"
            else:
                channel = signal_type or "unknown"

        min_confluence_strength = to_confidence(params.get("min_confluence_strength", 0.55), 0.55)
        confluence_strength = to_confidence(payload.get("strength", payload.get("conviction_score", 0.0)), 0.0)

        self._confluence_strength = confluence_strength

        source_ok = source == "traders"
        channel_ok = channel == "confluence"
        channel_threshold_ok = channel == "confluence" and confluence_strength >= min_confluence_strength

        return [
            DecisionCheck("source", "Unified traders source", source_ok, detail="Requires source=traders."),
            DecisionCheck("channel", "Supported traders channel", channel_ok, detail="Requires confluence channel."),
            DecisionCheck(
                "channel_threshold",
                "Channel strength threshold",
                channel_threshold_ok,
                score=confluence_strength,
                detail=f"confluence>={min_confluence_strength:.2f}",
            ),
        ]

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        return (edge * 0.55) + (confidence * 35.0) + (self._confluence_strength * 10.0)

    def compute_size(
        self, base_size: float, max_size: float, edge: float, confidence: float, risk_score: float, market_count: int
    ) -> float:
        size = base_size * (1.0 + (edge / 100.0)) * (0.75 + confidence) * (0.9 + self._confluence_strength)
        return max(1.0, min(max_size, size))

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Traders confluence: standard TP/SL exit."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        return self.default_exit_check(position, market_state)
