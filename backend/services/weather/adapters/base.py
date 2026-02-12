from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class WeatherForecastInput:
    """Normalized weather contract input for model forecasting."""

    location: str
    target_time: datetime
    metric: str
    operator: str
    threshold_c: Optional[float] = None
    threshold_c_low: Optional[float] = None
    threshold_c_high: Optional[float] = None


@dataclass
class WeatherSourceSnapshot:
    """Single forecast source/model value used in consensus building."""

    source_id: str
    provider: str
    model: str
    value_c: Optional[float] = None
    probability: Optional[float] = None
    weight: Optional[float] = None
    target_time: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeatherForecastResult:
    """Forecast result bundle for dual-model consensus."""

    gfs_probability: float
    ecmwf_probability: float
    gfs_value: Optional[float] = None
    ecmwf_value: Optional[float] = None
    source_snapshots: list[WeatherSourceSnapshot] = field(default_factory=list)
    consensus_probability: Optional[float] = None
    consensus_value_c: Optional[float] = None
    source_spread_c: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WeatherModelAdapter(ABC):
    """Forecast provider interface for weather workflow."""

    @abstractmethod
    async def forecast_probability(
        self, contract: WeatherForecastInput
    ) -> WeatherForecastResult:
        """Return GFS/ECMWF probabilities for a normalized weather contract."""
        raise NotImplementedError
