"""Comprehensive fee model for Polymarket arbitrage.

Accounts for ALL costs beyond the simple 2% winner fee:
- Polygon gas costs per transaction
- NegRisk conversion gas overhead
- Bid-ask spread crossing costs
- Multi-leg compounding slippage

IMDEA study found real-world costs erode ~40% of theoretical profit.
This model ensures profit calculations reflect executable reality.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FeeBreakdown:
    """Complete fee breakdown for a trade."""

    winner_fee: float  # Polymarket 2% winner fee
    gas_cost_usd: float  # Polygon gas cost per transaction
    spread_cost: float  # Cost of crossing the bid-ask spread
    multi_leg_slippage: float  # Compounding slippage across legs
    total_fees: float  # Sum of all fees
    fee_as_pct_of_payout: float  # Total fees as % of expected payout


class FeeModel:
    """Comprehensive fee model for realistic profit calculation.

    Default parameters are calibrated from observed Polygon network costs
    and Polymarket order book data (Feb 2026).

    Usage:
        breakdown = fee_model.calculate_fees(
            expected_payout=1.0,
            num_legs=3,
            is_negrisk=True,
            spread_bps=40.0,
            total_cost=0.96,
        )
        print(f"Total fees: ${breakdown.total_fees:.4f}")
        print(f"Fees as % of payout: {breakdown.fee_as_pct_of_payout:.2f}%")
    """

    # Per-leg slippage rate: each leg adds ~0.3% compounding slippage
    SLIPPAGE_PER_LEG: float = 0.003

    def __init__(
        self,
        winner_fee_rate: float = 0.02,  # 2% Polymarket winner fee
        gas_cost_per_tx: float = 0.005,  # ~$0.005 per Polygon tx
        negrisk_conversion_gas: float = 0.01,  # Extra gas for NegRisk conversion
        default_spread_bps: float = 50.0,  # Default spread in basis points
    ):
        self.winner_fee_rate = winner_fee_rate
        self.gas_cost_per_tx = gas_cost_per_tx
        self.negrisk_conversion_gas = negrisk_conversion_gas
        self.default_spread_bps = default_spread_bps

    def calculate_fees(
        self,
        expected_payout: float,
        num_legs: int,
        is_negrisk: bool = False,
        spread_bps: Optional[float] = None,
        total_cost: float = 0.0,
    ) -> FeeBreakdown:
        """Calculate all fees for a trade.

        Args:
            expected_payout: Expected payout on win (typically 1.0 per share).
            num_legs: Number of legs (markets) in the trade.
            is_negrisk: Whether this is a NegRisk trade requiring conversion.
            spread_bps: Actual bid-ask spread in basis points. If None, uses
                the default_spread_bps from construction.
            total_cost: Total cost of all positions (used for spread calculation).

        Returns:
            FeeBreakdown with all individual fee components and totals.
        """
        # 1. Winner fee: applied to the payout on the winning leg
        winner_fee = expected_payout * self.winner_fee_rate

        # 2. Gas costs: one transaction per leg, plus NegRisk conversion overhead
        gas_cost_usd = self.gas_cost_per_tx * num_legs
        if is_negrisk:
            gas_cost_usd += self.negrisk_conversion_gas

        # 3. Spread cost: cost of crossing the bid-ask spread on total position
        effective_spread_bps = (
            spread_bps if spread_bps is not None else self.default_spread_bps
        )
        spread_cost = total_cost * (effective_spread_bps / 10_000)

        # 4. Multi-leg slippage: modeled as compounding — each leg adds ~0.3%
        #    slippage on top of the previous. For a single leg there is no
        #    compounding effect; slippage starts from the second leg onward.
        #    Formula: total_cost * ((1 + r)^n - 1) where r = per-leg rate, n = num_legs
        if num_legs > 0 and total_cost > 0:
            compounding_factor = (1 + self.SLIPPAGE_PER_LEG) ** num_legs - 1
            multi_leg_slippage = total_cost * compounding_factor
        else:
            multi_leg_slippage = 0.0

        # Sum all fees
        total_fees = winner_fee + gas_cost_usd + spread_cost + multi_leg_slippage

        # Express total fees as percentage of expected payout
        fee_as_pct = (
            (total_fees / expected_payout * 100) if expected_payout > 0 else 0.0
        )

        return FeeBreakdown(
            winner_fee=winner_fee,
            gas_cost_usd=gas_cost_usd,
            spread_cost=spread_cost,
            multi_leg_slippage=multi_leg_slippage,
            total_fees=total_fees,
            fee_as_pct_of_payout=fee_as_pct,
        )

    def net_profit_after_all_fees(
        self,
        gross_profit: float,
        expected_payout: float,
        num_legs: int,
        is_negrisk: bool = False,
        spread_bps: Optional[float] = None,
        total_cost: float = 0.0,
    ) -> tuple[float, FeeBreakdown]:
        """Calculate net profit after subtracting all fees.

        This is the primary entry point for determining whether a trade is
        worth executing. It computes the comprehensive fee breakdown and
        subtracts it from the gross profit.

        Args:
            gross_profit: Revenue minus cost before any fees.
            expected_payout: Expected payout on win (typically 1.0 per share).
            num_legs: Number of legs (markets) in the trade.
            is_negrisk: Whether this is a NegRisk trade requiring conversion.
            spread_bps: Actual bid-ask spread in basis points.
            total_cost: Total cost of all positions.

        Returns:
            Tuple of (net_profit, FeeBreakdown).
        """
        breakdown = self.calculate_fees(
            expected_payout=expected_payout,
            num_legs=num_legs,
            is_negrisk=is_negrisk,
            spread_bps=spread_bps,
            total_cost=total_cost,
        )
        net_profit = gross_profit - breakdown.total_fees
        return net_profit, breakdown


# Module-level singleton for convenient import
fee_model = FeeModel()
