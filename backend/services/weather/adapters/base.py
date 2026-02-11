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


@dataclass
class WeatherForecastResult:
    """Forecast result bundle for dual-model consensus."""

    gfs_probability: float
    ecmwf_probability: float
    gfs_value: Optional[float] = None
    ecmwf_value: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WeatherModelAdapter(ABC):
    """Forecast provider interface for weather workflow."""

    @abstractmethod
    async def forecast_probability(
        self, contract: WeatherForecastInput
    ) -> WeatherForecastResult:
        """Return GFS/ECMWF probabilities for a normalized weather contract."""
        raise NotImplementedError
