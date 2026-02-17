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
from typing import Any, Optional

from config import settings
from models import ArbitrageOpportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import BaseStrategy, DecisionCheck, StrategyDecision, ExitDecision
from services.strategies._evaluate_helpers import to_float, to_confidence, signal_payload
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
        normalized = str(tier or "watch").strip().lower()
        return TradersConfluenceStrategy.TIER_ORDER.get(normalized, 0)

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
        signal_tier = self._tier_value(signal.get("tier", "watch"))
        checks["meets_min_tier"] = signal_tier >= min_tier
        if signal_tier < min_tier:
            reasons.append("tier_below_threshold")

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
            row = dict(signal)
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

        cfg = self._config
        opportunities: list[ArbitrageOpportunity] = []
        market_map = {m.id: m for m in markets}
        event_map: dict[str, Event] = {}
        for event in events:
            for m in event.markets:
                event_map[m.id] = event

        for signal in signals:
            try:
                opp = self._evaluate_signal(signal, market_map, event_map, cfg)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug("Traders Confluence: skipped signal: %s", e)

        if opportunities:
            logger.info(
                "Traders Confluence: %d opportunities from %d signals",
                len(opportunities),
                len(signals),
            )
        return opportunities

    def _evaluate_signal(
        self,
        signal: dict,
        market_map: dict[str, Market],
        event_map: dict[str, Event],
        cfg: dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Evaluate a single confluence signal against config thresholds."""
        edge = float(signal.get("edge_percent", 0))
        confidence = float(signal.get("confidence", 0))
        strength = float(signal.get("confluence_strength", 0))
        tier = str(signal.get("tier", "low")).lower()
        wallet_count = int(signal.get("wallet_count", 0))
        entry_price = float(signal.get("entry_price", 1))

        # Quality gates
        if edge < cfg["min_edge_percent"]:
            return None
        if confidence < cfg["min_confidence"]:
            return None
        if strength < cfg["min_confluence_strength"]:
            return None
        min_tier_val = self.TIER_ORDER.get(cfg["min_tier"], 2)
        signal_tier_val = self.TIER_ORDER.get(tier, 0)
        if signal_tier_val < min_tier_val:
            return None
        if wallet_count < cfg.get("min_wallet_count", 2):
            return None
        if entry_price > cfg["max_entry_price"]:
            return None

        market_id = signal.get("market_id")
        market = market_map.get(market_id) if market_id else None
        if not market:
            return None

        event = event_map.get(market_id)
        direction = signal.get("direction", "buy_yes")
        target_price = float(signal.get("target_price", entry_price))
        total_volume = float(signal.get("total_volume_usd", 0))
        wallets = signal.get("wallets", [])

        # Position details
        side = "YES" if direction == "buy_yes" else "NO"
        token_id = None
        if market.clob_token_ids:
            idx = 0 if direction == "buy_yes" else (1 if len(market.clob_token_ids) > 1 else 0)
            token_id = market.clob_token_ids[idx]

        expected_payout = target_price
        total_cost = entry_price
        gross_profit = expected_payout - total_cost
        fee_amount = expected_payout * self.fee
        net_profit = gross_profit - fee_amount
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        if roi < cfg["min_edge_percent"] / 2:
            return None

        min_liquidity = market.liquidity
        max_position = min(min_liquidity * 0.05, 500.0)

        if max_position < settings.MIN_POSITION_SIZE:
            return None

        # Risk scoring
        risk_score = float(cfg["risk_base_score"])
        risk_factors = [
            "Smart money convergence bet (behavioral edge)",
            f"Confluence: {strength:.0%} strength, {wallet_count} wallets, tier {tier}",
            f"Total tracked volume: ${total_volume:,.0f}",
        ]
        if confidence < 0.6:
            risk_score += 0.15
            risk_factors.append("Moderate confidence")
        if wallet_count <= 2:
            risk_score += 0.1
            risk_factors.append("Minimal wallet convergence")
        if tier == "high":
            risk_score -= 0.05
        elif tier == "extreme":
            risk_score -= 0.1
        risk_score = max(min(risk_score, 1.0), 0.0)

        positions = [
            {
                "action": "BUY",
                "outcome": side,
                "price": entry_price,
                "token_id": token_id,
                "_traders_confluence": {
                    "confluence_strength": strength,
                    "tier": tier,
                    "wallet_count": wallet_count,
                    "total_volume_usd": total_volume,
                    "wallets": wallets[:10],  # Cap for payload size
                    "edge_percent": edge,
                    "confidence": confidence,
                    "direction": direction,
                },
            }
        ]

        market_dict = {
            "id": market.id,
            "slug": market.slug,
            "question": market.question,
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "liquidity": market.liquidity,
        }

        return ArbitrageOpportunity(
            strategy=self.strategy_type,
            title=f"Trader Flow: {wallet_count} wallets → {market.question[:40]}",
            description=(
                f"{wallet_count} tracked wallets ({tier} tier, {strength:.0%} confluence) "
                f"converging on {side} at ${entry_price:.2f} "
                f"(edge: {edge:.1f}%, volume: ${total_volume:,.0f})"
            ),
            total_cost=total_cost,
            expected_payout=expected_payout,
            gross_profit=gross_profit,
            fee=fee_amount,
            net_profit=net_profit,
            roi_percent=roi,
            is_guaranteed=False,
            roi_type="directional_payout",
            risk_score=risk_score,
            risk_factors=risk_factors,
            markets=[market_dict],
            event_id=event.id if event else None,
            event_slug=event.slug if event else None,
            event_title=event.title if event else None,
            category=event.category if event else None,
            min_liquidity=min_liquidity,
            max_position_size=max_position,
            resolution_date=market.end_date,
            mispricing_type=MispricingType.NEWS_INFORMATION,
            positions_to_take=positions,
        )

    # ------------------------------------------------------------------
    # Evaluate / Should-Exit  (unified strategy interface)
    # ------------------------------------------------------------------

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        """Evaluate traders confluence signal with channel strength gating."""
        params = context.get("params") or {}
        payload = signal_payload(signal)

        source = str(getattr(signal, "source", "") or "").strip().lower()

        # Channel derivation: payload.traders_channel -> signal_type fallback
        payload_channel = str(payload.get("traders_channel") or "").strip().lower()
        if payload_channel:
            channel = payload_channel
        else:
            signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()
            if signal_type == "confluence":
                channel = "confluence"
            else:
                channel = signal_type or "unknown"

        min_edge = to_float(params.get("min_edge_percent", 3.0), 3.0)
        min_conf = to_confidence(params.get("min_confidence", 0.48), 0.48)
        min_confluence_strength = to_confidence(
            params.get("min_confluence_strength", 0.55),
            0.55,
        )
        base_size = to_float(params.get("base_size_usd", 18.0), 18.0)
        max_size = max(1.0, to_float(params.get("max_size_usd", 120.0), 120.0))

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        confluence_strength = to_confidence(
            payload.get("strength", payload.get("conviction_score", 0.0)),
            0.0,
        )

        source_ok = source == "traders"
        channel_ok = channel == "confluence"
        channel_threshold_ok = channel == "confluence" and confluence_strength >= min_confluence_strength

        checks = [
            DecisionCheck(
                "source",
                "Unified traders source",
                source_ok,
                detail="Requires source=traders.",
            ),
            DecisionCheck(
                "channel",
                "Supported traders channel",
                channel_ok,
                detail="Requires confluence channel.",
            ),
            DecisionCheck(
                "channel_threshold",
                "Channel strength threshold",
                channel_threshold_ok,
                score=confluence_strength,
                detail=f"confluence>={min_confluence_strength:.2f}",
            ),
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
        ]

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Traders flow filters not met",
                score=(edge * 0.6) + (confidence * 40.0),
                checks=checks,
                payload={
                    "channel": channel,
                    "edge": edge,
                    "confidence": confidence,
                    "confluence_strength": confluence_strength,
                    "min_edge_percent": min_edge,
                    "min_confidence": min_conf,
                    "min_confluence_strength": min_confluence_strength,
                },
            )

        channel_score = confluence_strength
        size = base_size * (1.0 + (edge / 100.0)) * (0.75 + confidence) * (0.9 + channel_score)
        size = max(1.0, min(max_size, size))

        return StrategyDecision(
            decision="selected",
            reason=f"Traders flow {channel} signal selected",
            score=(edge * 0.55) + (confidence * 35.0) + (channel_score * 10.0),
            size_usd=size,
            checks=checks,
            payload={
                "channel": channel,
                "edge": edge,
                "confidence": confidence,
                "confluence_strength": confluence_strength,
                "size_usd": size,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Traders confluence: standard TP/SL exit."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)
        return self.default_exit_check(position, market_state)
