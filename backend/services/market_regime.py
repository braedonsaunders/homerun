from __future__ import annotations

import statistics
from collections import deque
from datetime import datetime
from typing import Dict, Optional


_REGIME_PARAMS = {
    "trending": {
        "confidence_multiplier": 1.0,
        "size_multiplier": 1.0,
        "stop_loss_width": 1.0,
        "max_hold_multiplier": 1.0,
    },
    "volatile": {
        "confidence_multiplier": 1.15,
        "size_multiplier": 0.5,
        "stop_loss_width": 1.25,
        "max_hold_multiplier": 0.67,
    },
    "ranging": {
        "confidence_multiplier": 1.0,
        "size_multiplier": 0.7,
        "stop_loss_width": 1.0,
        "max_hold_multiplier": 0.75,
    },
}


class MarketRegimeClassifier:
    def __init__(
        self,
        max_history: int = 200,
        min_points: int = 20,
        trending_cumret_threshold: float = 0.03,
        trending_vol_ratio_max: float = 2.0,
        volatile_stddev_threshold: float = 0.015,
    ):
        self._max_history = max_history
        self._min_points = min_points
        self._trending_cumret_threshold = trending_cumret_threshold
        self._trending_vol_ratio_max = trending_vol_ratio_max
        self._volatile_stddev_threshold = volatile_stddev_threshold
        self._prices: Dict[str, deque] = {}

    def update(self, market_id: str, price: float, timestamp: Optional[datetime] = None) -> None:
        buf = self._prices.get(market_id)
        if buf is None:
            buf = deque(maxlen=self._max_history)
            self._prices[market_id] = buf
        buf.append(price)

    def get_regime(self, market_id: str) -> str:
        buf = self._prices.get(market_id)
        if buf is None or len(buf) < self._min_points:
            return "ranging"

        prices = list(buf)
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices)) if prices[i - 1] != 0]
        if len(returns) < 2:
            return "ranging"

        cumulative_return = abs(prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
        vol = statistics.stdev(returns)
        mean_abs_return = statistics.mean(abs(r) for r in returns)

        if cumulative_return > self._trending_cumret_threshold and (
            mean_abs_return == 0 or vol < self._trending_vol_ratio_max * mean_abs_return
        ):
            return "trending"

        if vol > self._volatile_stddev_threshold:
            return "volatile"

        return "ranging"

    def get_regime_params(self, market_id: str) -> dict:
        regime = self.get_regime(market_id)
        return dict(_REGIME_PARAMS[regime])

    def clear(self, market_id: str) -> None:
        self._prices.pop(market_id, None)


market_regime_classifier = MarketRegimeClassifier()
