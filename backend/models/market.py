from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json


def _parse_maybe_json_list(raw: object) -> list[object]:
    """Accept list values directly or parse JSON-encoded list strings."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(parsed, list):
            return parsed
    return []


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
    group_item_title: str = ""
    event_slug: str = ""
    tokens: list[Token] = []
    clob_token_ids: list[str] = []
    outcome_prices: list[float] = []
    active: bool = True
    closed: bool = False
    archived: Optional[bool] = None
    accepting_orders: Optional[bool] = None
    resolved: Optional[bool] = None
    winner: Optional[str] = None
    winning_outcome: Optional[str] = None
    enable_order_book: Optional[bool] = None
    status: Optional[str] = None
    neg_risk: bool = False
    volume: float = 0.0
    liquidity: float = 0.0
    end_date: Optional[datetime] = None
    tags: list[str] = []
    platform: str = "polymarket"  # "polymarket" or "kalshi"

    @classmethod
    def from_gamma_response(cls, data: dict) -> "Market":
        """Parse market from Gamma API response"""
        # Parse stringified JSON fields
        clob_token_ids: list[str] = []
        outcome_prices: list[float] = []

        clob_token_ids_raw = _parse_maybe_json_list(
            data.get("clobTokenIds", data.get("clob_token_ids"))
        )
        for token_id in clob_token_ids_raw:
            token_text = str(token_id or "").strip()
            if token_text:
                clob_token_ids.append(token_text)

        outcome_prices_raw = _parse_maybe_json_list(
            data.get("outcomePrices", data.get("outcome_prices"))
        )
        for price in outcome_prices_raw:
            try:
                outcome_prices.append(float(price))
            except (TypeError, ValueError):
                continue

        # Prefer bestBid/bestAsk from CLOB for the YES side (more accurate
        # than outcomePrices which can be stale on fast-moving markets).
        best_bid = data.get("bestBid")
        best_ask = data.get("bestAsk")
        if best_bid is not None and best_ask is not None:
            try:
                yes_mid = (float(best_bid) + float(best_ask)) / 2.0
                no_mid = 1.0 - yes_mid
                outcome_prices = [yes_mid, no_mid]
            except (ValueError, TypeError):
                pass

        # Binary markets (Up/Down, Yes/No) sometimes omit outcomePrices in API
        if not outcome_prices and data.get("outcomes"):
            outcomes_raw = data["outcomes"]
            if isinstance(outcomes_raw, str):
                try:
                    outcomes_raw = json.loads(outcomes_raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(outcomes_raw, list) and len(outcomes_raw) == 2:
                outcome_prices = [0.5, 0.5]  # Default for binary

        # Build tokens list
        tokens = []
        outcomes = ["Yes", "No"]
        for i, token_id in enumerate(clob_token_ids):
            price = outcome_prices[i] if i < len(outcome_prices) else 0.0
            outcome = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
            tokens.append(Token(token_id=token_id, outcome=outcome, price=price))

        # Extract parent event slug from nested events array or direct field
        event_slug = ""
        if data.get("events"):
            try:
                event_slug = data["events"][0].get("slug", "")
            except (IndexError, AttributeError, TypeError):
                pass
        if not event_slug:
            event_slug = data.get("event_slug", "") or data.get("eventSlug", "")

        tags = []
        raw_tags = data.get("tags", [])
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                if isinstance(tag, str):
                    tags.append(tag)
                elif isinstance(tag, dict):
                    value = tag.get("label") or tag.get("name")
                    if value:
                        tags.append(str(value))
        elif isinstance(raw_tags, str) and raw_tags.strip():
            tags.append(raw_tags.strip())

        return cls(
            id=str(data.get("id", "")),
            condition_id=data.get("condition_id", data.get("conditionId", "")),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            group_item_title=data.get("groupItemTitle", data.get("group_item_title", "")),
            event_slug=event_slug,
            tokens=tokens,
            clob_token_ids=clob_token_ids,
            outcome_prices=outcome_prices,
            active=data.get("active", True),
            closed=data.get("closed", False),
            archived=data.get("archived"),
            accepting_orders=data.get("acceptingOrders", data.get("accepting_orders")),
            resolved=data.get("resolved", data.get("isResolved", data.get("is_resolved"))),
            winner=data.get("winner"),
            winning_outcome=data.get("winningOutcome", data.get("winning_outcome")),
            enable_order_book=data.get("enableOrderBook", data.get("enable_order_book")),
            status=data.get("status", data.get("marketStatus", data.get("market_status"))),
            neg_risk=data.get("negRisk", data.get("neg_risk", False)),
            volume=float(
                data.get("volume") or data.get("volumeNum") or 0
            ),
            liquidity=float(
                data.get("liquidity") or data.get("liquidityNum") or 0
            ),
            end_date=data.get("endDate"),
            tags=tags,
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
    tags: list[str] = []
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

        event_tags = []
        raw_tags = data.get("tags", [])
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                if isinstance(tag, str):
                    event_tags.append(tag)
                elif isinstance(tag, dict):
                    value = tag.get("label") or tag.get("name")
                    if value:
                        event_tags.append(str(value))
        elif isinstance(raw_tags, str) and raw_tags.strip():
            event_tags.append(raw_tags.strip())

        return cls(
            id=str(data.get("id", "")),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            category=category,
            tags=event_tags,
            markets=markets,
            neg_risk=data.get("negRisk", False),
            active=data.get("active", True),
            closed=data.get("closed", False),
        )
