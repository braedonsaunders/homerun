from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class DataEvent:
    """A single data event that strategies can subscribe to.

    Event types:
        price_change       - A market's price changed (from WS feed)
        market_data_refresh - Periodic batch of all market data (replaces scanner poll)
        market_resolved    - A market outcome was determined
        crypto_update      - Crypto market data from crypto worker
        weather_update     - Weather forecast data from weather worker
        trader_activity    - Smart wallet / copy trading signal from traders worker
        news_event         - Breaking news signal from news worker
    """
    event_type: str
    source: str
    timestamp: datetime
    market_id: Optional[str] = None
    token_id: Optional[str] = None
    payload: dict = field(default_factory=dict)
    old_price: Optional[float] = None
    new_price: Optional[float] = None
    markets: Optional[list] = None
    events: Optional[list] = None
    prices: Optional[dict] = None
