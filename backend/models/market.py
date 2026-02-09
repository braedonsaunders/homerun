from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json


class Token(BaseModel):
    """Represents a YES or NO token in a market"""

    token_id: str
    outcome: str  # "Yes" or "No"
    price: float = 0.0


class Market(BaseModel):
    """Represents a single prediction market"""

    id: str
    condition_id: str
    question: str
    slug: str
    tokens: list[Token] = []
    clob_token_ids: list[str] = []
    outcome_prices: list[float] = []
    active: bool = True
    closed: bool = False
    neg_risk: bool = False
    volume: float = 0.0
    liquidity: float = 0.0
    end_date: Optional[datetime] = None
    platform: str = "polymarket"  # "polymarket" or "kalshi"

    @classmethod
    def from_gamma_response(cls, data: dict) -> "Market":
        """Parse market from Gamma API response"""
        # Parse stringified JSON fields
        clob_token_ids = []
        outcome_prices = []

        if data.get("clobTokenIds"):
            try:
                clob_token_ids = json.loads(data["clobTokenIds"])
            except (json.JSONDecodeError, TypeError):
                pass

        if data.get("outcomePrices"):
            try:
                outcome_prices = [float(p) for p in json.loads(data["outcomePrices"])]
            except (json.JSONDecodeError, TypeError):
                pass

        # Build tokens list
        tokens = []
        outcomes = ["Yes", "No"]
        for i, token_id in enumerate(clob_token_ids):
            price = outcome_prices[i] if i < len(outcome_prices) else 0.0
            outcome = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
            tokens.append(Token(token_id=token_id, outcome=outcome, price=price))

        return cls(
            id=str(data.get("id", "")),
            condition_id=data.get("condition_id", data.get("conditionId", "")),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            tokens=tokens,
            clob_token_ids=clob_token_ids,
            outcome_prices=outcome_prices,
            active=data.get("active", True),
            closed=data.get("closed", False),
            neg_risk=data.get("negRisk", data.get("neg_risk", False)),
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            end_date=data.get("endDate"),
        )

    @property
    def yes_price(self) -> float:
        """Get YES token price"""
        if self.outcome_prices and len(self.outcome_prices) > 0:
            return self.outcome_prices[0]
        return 0.0

    @property
    def no_price(self) -> float:
        """Get NO token price"""
        if self.outcome_prices and len(self.outcome_prices) > 1:
            return self.outcome_prices[1]
        return 0.0


class Event(BaseModel):
    """Represents an event containing one or more markets"""

    id: str
    slug: str
    title: str
    description: str = ""
    category: Optional[str] = None
    markets: list[Market] = []
    neg_risk: bool = False
    active: bool = True
    closed: bool = False

    @classmethod
    def from_gamma_response(cls, data: dict) -> "Event":
        """Parse event from Gamma API response"""
        markets = []
        for m in data.get("markets", []):
            try:
                markets.append(Market.from_gamma_response(m))
            except Exception:
                pass

        # Extract category from tags or category field
        category = None
        if data.get("category"):
            cat = data.get("category")
            # Handle category as dict (API returns {id, label, ...}) or string
            if isinstance(cat, dict):
                category = cat.get("label", cat.get("name", ""))
            else:
                category = cat
        elif data.get("tags"):
            # Tags is usually a list, take the first one as category
            tags = data.get("tags", [])
            if isinstance(tags, list) and len(tags) > 0:
                first_tag = tags[0]
                # Handle tag as dict or string
                if isinstance(first_tag, dict):
                    category = first_tag.get("label", first_tag.get("name", ""))
                else:
                    category = first_tag
            elif isinstance(tags, str):
                category = tags

        return cls(
            id=str(data.get("id", "")),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            category=category,
            markets=markets,
            neg_risk=data.get("negRisk", False),
            active=data.get("active", True),
            closed=data.get("closed", False),
        )
