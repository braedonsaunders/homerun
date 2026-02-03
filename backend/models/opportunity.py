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


class ArbitrageOpportunity(BaseModel):
    """Represents a detected arbitrage opportunity"""
    id: str = Field(default_factory=lambda: "")
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

    # Liquidity
    min_liquidity: float = 0.0
    max_position_size: float = 0.0  # How much can be executed

    # Timing
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    resolution_date: Optional[datetime] = None

    # Execution details
    positions_to_take: list[dict] = []  # What to buy

    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            # Generate ID from strategy and market IDs
            market_ids = "_".join([m.get("id", "")[:8] for m in self.markets[:3]])
            self.id = f"{self.strategy.value}_{market_ids}_{int(self.detected_at.timestamp())}"


class OpportunityFilter(BaseModel):
    """Filter criteria for opportunities"""
    min_profit: float = 0.0
    max_risk: float = 1.0
    strategies: list[StrategyType] = []
    min_liquidity: float = 0.0
