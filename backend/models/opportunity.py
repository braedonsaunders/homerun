from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class StrategyType(str, Enum):
    BASIC = "basic"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    CONTRADICTION = "contradiction"
    NEGRISK = "negrisk"
    MUST_HAPPEN = "must_happen"
    MIRACLE = "miracle"  # Bet against impossible/absurd events
    COMBINATORIAL = "combinatorial"  # Cross-market arbitrage via integer programming
    SETTLEMENT_LAG = (
        "settlement_lag"  # Exploit delayed price updates after outcome determined
    )
    CROSS_PLATFORM = (
        "cross_platform"  # Cross-platform arbitrage (e.g. Polymarket vs Kalshi)
    )


class MispricingType(str, Enum):
    """Classification of mispricing source (from Kroer et al. Part 2, Section IV).

    Market makers choose speed over accuracy, creating three systematic types:
    - WITHIN_MARKET: Sum of probabilities != 1 for multi-condition markets
      (662 NegRisk markets, 42% of all multi-condition, median deviation 0.08)
    - CROSS_MARKET: Dependent markets priced independently, violating constraints
      (1,576 dependent pairs identified, 13 confirmed exploitable)
    - SETTLEMENT_LAG: Prices don't instantly lock after outcome determined
      (windows last minutes to hours, e.g. Assad example)
    """

    WITHIN_MARKET = "within_market"
    CROSS_MARKET = "cross_market"
    SETTLEMENT_LAG = "settlement_lag"


class AIAnalysis(BaseModel):
    """Inline AI judgment data attached to an opportunity."""

    overall_score: float = 0.0
    profit_viability: float = 0.0
    resolution_safety: float = 0.0
    execution_feasibility: float = 0.0
    market_efficiency: float = 0.0
    recommendation: str = "pending"  # strong_execute, execute, review, skip, strong_skip, pending
    reasoning: Optional[str] = None
    risk_factors: list[str] = []
    judged_at: Optional[datetime] = None
    resolution_analyses: list[dict] = []


class ArbitrageOpportunity(BaseModel):
    """Represents a detected arbitrage opportunity"""

    id: str = Field(default_factory=lambda: "")
    stable_id: str = Field(default_factory=lambda: "")  # Persists across scans (no timestamp)
    strategy: StrategyType
    title: str
    description: str

    # Profit metrics
    total_cost: float
    expected_payout: float = 1.0
    gross_profit: float
    fee: float
    net_profit: float
    roi_percent: float

    # Risk assessment
    risk_score: float = Field(ge=0.0, le=1.0, default=0.5)
    risk_factors: list[str] = []

    # Market details
    markets: list[dict] = []  # List of markets involved
    event_id: Optional[str] = None
    event_title: Optional[str] = None
    category: Optional[str] = None

    # Liquidity
    min_liquidity: float = 0.0
    max_position_size: float = 0.0  # How much can be executed

    # Timing
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    resolution_date: Optional[datetime] = None

    # Mispricing classification (from article Part IV)
    mispricing_type: Optional[MispricingType] = None

    # Profit guarantee from Frank-Wolfe (Proposition 4.1)
    guaranteed_profit: Optional[float] = None  # D(μ̂||θ) - g(μ̂)
    capture_ratio: Optional[float] = None  # guaranteed / max profit

    # Execution details
    positions_to_take: list[dict] = []  # What to buy

    # Inline AI analysis (populated by scanner, persisted across scans)
    ai_analysis: Optional[AIAnalysis] = None

    def __init__(self, **data):
        super().__init__(**data)
        market_ids = "_".join([m.get("id", "")[:8] for m in self.markets[:3]])
        if not self.stable_id:
            self.stable_id = f"{self.strategy.value}_{market_ids}"
        if not self.id:
            self.id = f"{self.strategy.value}_{market_ids}_{int(self.detected_at.timestamp())}"


class OpportunityFilter(BaseModel):
    """Filter criteria for opportunities"""

    min_profit: float = 0.0
    max_risk: float = 1.0
    strategies: list[StrategyType] = []
    min_liquidity: float = 0.0
    category: Optional[str] = None
