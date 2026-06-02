# SEED TEMPLATE — not the live strategy. The DB `strategies.source_code` is the
# runtime master (loaded by the orchestrator + backtester); edits here only seed
# fresh installs / reset-to-factory. See services/opportunity_strategy_catalog.py.
"""Holding reward yield strategy.

Detects Polymarket markets eligible for holding rewards and surfaces
opportunities to split USDC into YES+NO token pairs via the CTF
``splitPosition`` call.  Capital is preserved regardless of outcome
(one side resolves to $1, the other to $0, combined = $1) while the
position earns ~4% annualized holding rewards from the Polymarket treasury.

Position value is sampled hourly and rewards accrue daily.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from config import settings
from models import Opportunity, Event, Market
from services.strategies.base import (
    BaseStrategy,
    ExitDecision,
    ScoringWeights,
    SizingConfig,
    utcnow,
    make_aware,
)
from services.quality_filter import QualityFilterOverrides
from utils.converters import to_float
from utils.logger import get_logger

logger = get_logger(__name__)

# Polymarket's current holding reward rate (annualized percent).
# Used as a floor when the Gamma API doesn't return per-market rate data.
_DEFAULT_HOLDING_REWARD_APY = 4.0

# Minimum days before resolution to recommend exiting (avoid illiquid merge).
_EXIT_BUFFER_DAYS = 7.0


def _extract_reward_apy(market: Market) -> Optional[float]:
    """Extract annualized reward rate from a market's clob_rewards metadata.

    Polymarket's Gamma API returns ``clobRewards`` as a list of dicts.
    Known field names for the rate:
        - ``rewardsDailyRate`` (fractional daily, e.g. 0.0001096 ~ 4% APY)
        - ``annualizedReward`` (percent, e.g. 4.0)
        - ``dailyRate`` (alias for rewardsDailyRate)

    Returns the annualized APY as a percent, or None if no rate data is found.
    """
    if not market.clob_rewards:
        return None

    for entry in market.clob_rewards:
        if not isinstance(entry, dict):
            continue

        # Check for annualized rate first (most direct).
        annualized = entry.get("annualizedReward") or entry.get("annualized_reward")
        if annualized is not None:
            try:
                apy = float(annualized)
                if apy > 0:
                    return apy
            except (TypeError, ValueError):
                pass

        # Derive from daily rate.
        daily = (
            entry.get("rewardsDailyRate")
            or entry.get("rewards_daily_rate")
            or entry.get("dailyRate")
            or entry.get("daily_rate")
        )
        if daily is not None:
            try:
                daily_f = float(daily)
                if daily_f > 0:
                    return daily_f * 365.0 * 100.0 if daily_f < 1.0 else daily_f * 365.0
            except (TypeError, ValueError):
                pass

    return None


class HoldingRewardYieldStrategy(BaseStrategy):
    """Earn Polymarket holding rewards by splitting USDC into YES+NO pairs."""

    strategy_type = "holding_reward_yield"
    name = "Holding Reward Yield"
    description = (
        "Earn ~4% APY from Polymarket holding rewards on long-dated markets. "
        "Capital is preserved via CTF split (YES+NO = $1) while rewards accrue."
    )
    mispricing_type = "within_market"
    source_key = "scanner"
    subscriptions = ["market_data_refresh"]
    binary_only = True

    quality_filter_overrides = QualityFilterOverrides(
        max_resolution_months=18.0,
        min_roi=0.5,
        min_annualized_roi=0.0,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.3,
        confidence_weight=20.0,
        risk_penalty=6.0,
    )
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.6,
        risk_scale_factor=0.2,
        risk_floor=0.65,
        market_scale_factor=0.0,
    )

    default_config = {
        "min_apy": 2.0,
        "min_liquidity": 5000.0,
        "min_days_to_resolution": 30.0,
        "max_opportunities": 20,
        "min_edge_percent": 1.0,
        "min_confidence": 0.30,
        "max_risk_score": 0.75,
        # APY assumed when the market doesn't publish its own reward rate.
        # Reasonable for current Polymarket holding-reward defaults.
        "default_holding_reward_apy": 4.0,
        # How many days before resolution to flatten the carry position.
        # Carrying right up to settlement reintroduces price-tail risk that
        # the yield doesn't compensate for.
        "exit_buffer_days": 7.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, Any],
    ) -> list[Opportunity]:
        if not settings.HOLDING_REWARD_YIELD_ENABLED:
            return []

        min_apy = float(self.config.get("min_apy", settings.HOLDING_REWARD_MIN_APY))
        min_liquidity = float(self.config.get("min_liquidity", settings.HOLDING_REWARD_MIN_LIQUIDITY))
        min_days = float(self.config.get("min_days_to_resolution", settings.HOLDING_REWARD_MIN_DAYS_TO_RESOLUTION))
        max_opps = int(self.config.get("max_opportunities", 20))

        now = utcnow()
        opportunities: list[Opportunity] = []

        for market in markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) != 2:
                continue
            if not market.accepting_orders or not market.enable_order_book:
                continue
            if market.liquidity < min_liquidity:
                continue
            if market.platform != "polymarket":
                continue

            end_date = make_aware(market.end_date) if market.end_date else None
            if end_date is None:
                continue
            days_to_res = (end_date - now).total_seconds() / 86400.0
            if days_to_res < min_days:
                continue

            reward_eligible = bool(market.clob_rewards) or market.rewards_min_size is not None
            if not reward_eligible:
                continue

            default_apy = float(self.config.get("default_holding_reward_apy", _DEFAULT_HOLDING_REWARD_APY) or _DEFAULT_HOLDING_REWARD_APY)
            apy = _extract_reward_apy(market) or default_apy
            if apy < min_apy:
                continue

            holding_period_yield_pct = apy * (days_to_res / 365.0)
            projected_daily_reward_pct = apy / 365.0

            split_cost = 1.0
            expected_payout = 1.0

            positions = [
                {
                    "market_id": market.id,
                    "market_question": market.question,
                    "action": "split",
                    "outcome": "YES",
                    "price": market.yes_price,
                    "token_id": market.clob_token_ids[0] if market.clob_token_ids else None,
                    "notional_weight": 0.5,
                },
                {
                    "market_id": market.id,
                    "market_question": market.question,
                    "action": "split",
                    "outcome": "NO",
                    "price": market.no_price,
                    "token_id": market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None,
                    "notional_weight": 0.5,
                },
            ]

            opp = self.create_opportunity(
                title=f"Holding Reward Yield: {apy:.1f}% APY",
                description=(
                    f"Split USDC into YES+NO on \"{market.question}\" to earn "
                    f"~{apy:.1f}% annualized holding rewards. "
                    f"Capital preserved via split; {days_to_res:.0f} days to resolution. "
                    f"Projected holding period yield: {holding_period_yield_pct:.2f}%."
                ),
                total_cost=split_cost,
                markets=[market],
                positions=positions,
                expected_payout=expected_payout,
                is_guaranteed=False,
                skip_fee_model=True,
                custom_roi_percent=apy,
                custom_risk_score=self._compute_risk(market, days_to_res, apy),
                confidence=min(0.85, 0.50 + (apy / 20.0)),
            )

            if opp is None:
                continue

            opp.strategy_context = {
                "condition_id": market.condition_id,
                "reward_apy": apy,
                "projected_daily_reward_pct": round(projected_daily_reward_pct, 6),
                "projected_monthly_reward_pct": round(projected_daily_reward_pct * 30.0, 4),
                "holding_period_yield_pct": round(holding_period_yield_pct, 4),
                "holding_days": round(days_to_res, 1),
                "split_merge_action": "split",
                "clob_rewards": market.clob_rewards,
                "rewards_min_size": market.rewards_min_size,
                "rewards_max_spread": market.rewards_max_spread,
            }

            opportunities.append(opp)

            if len(opportunities) >= max_opps:
                break

        opportunities.sort(key=lambda o: o.roi_percent, reverse=True)
        return opportunities[:max_opps]

    def _compute_risk(self, market: Market, days_to_res: float, apy: float) -> float:
        """Compute risk score for a holding reward position.

        Risk is inherently low since capital is preserved via split (YES+NO=$1).
        Primary risks: reward program ending early, gas costs eating into yield,
        and long lock-up reducing capital efficiency.
        """
        risk = 0.15

        if days_to_res > 180:
            risk += 0.05
        if days_to_res > 365:
            risk += 0.05

        if market.liquidity < 10_000:
            risk += 0.05
        elif market.liquidity < 20_000:
            risk += 0.02

        if apy < 3.0:
            risk += 0.03

        return min(0.40, risk)

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        if market_state.get("is_resolved") or market_state.get("resolved"):
            return ExitDecision(
                action="close",
                reason="Market resolved; merge positions to recover USDC",
            )

        if not market_state.get("market_tradable", True):
            return ExitDecision(
                action="close",
                reason="Market no longer tradable; merge positions",
            )

        seconds_left = to_float(market_state.get("seconds_left"))
        exit_buffer_days = float(self.config.get("exit_buffer_days", _EXIT_BUFFER_DAYS) or _EXIT_BUFFER_DAYS)
        if seconds_left is not None and seconds_left < exit_buffer_days * 86400:
            days_left = seconds_left / 86400.0
            return ExitDecision(
                action="close",
                reason=f"Approaching resolution ({days_left:.1f}d left); merge to recover USDC",
            )

        return ExitDecision(action="hold", reason="Holding for reward yield accrual")

    def calculate_risk_score(
        self, markets: list[Market], resolution_date: Optional[datetime]
    ) -> tuple[float, list[str]]:
        risk = 0.15
        factors: list[str] = []

        if resolution_date:
            now = utcnow()
            aware_res = make_aware(resolution_date)
            if aware_res:
                days = (aware_res - now).total_seconds() / 86400.0
                if days > 365:
                    risk += 0.05
                    factors.append("Very long lock-up (>1 year)")
                elif days > 180:
                    risk += 0.03
                    factors.append("Long lock-up (>6 months)")

        min_liq = min((m.liquidity for m in markets), default=0)
        if min_liq < 10_000:
            risk += 0.05
            factors.append(f"Low liquidity (${min_liq:,.0f})")

        if not factors:
            factors.append("Capital preserved via split (YES+NO = $1)")

        return min(0.40, risk), factors
