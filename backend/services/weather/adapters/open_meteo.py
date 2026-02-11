from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import httpx

from .base import WeatherForecastInput, WeatherForecastResult, WeatherModelAdapter


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _temp_probability(value_c: float, threshold_c: float, operator: str) -> float:
    # ~2C scale keeps transitions realistic instead of binary cliffs.
    scale_c = 2.0
    delta = value_c - threshold_c
    if operator in ("lt", "lte"):
        delta = -delta
    return max(0.0, min(1.0, _sigmoid(delta / scale_c)))


def _temp_range_probability(value_c: float, low_c: float, high_c: float) -> float:
    # Approximate a band probability as CDF(high) - CDF(low) with a smooth
    # logistic CDF around the deterministic model value.
    scale_c = 2.0
    low = min(low_c, high_c)
    high = max(low_c, high_c)
    p_above_low = _sigmoid((value_c - low) / scale_c)
    p_above_high = _sigmoid((value_c - high) / scale_c)
    return max(0.0, min(1.0, p_above_low - p_above_high))


def _precip_probability(value_mm: float, operator: str) -> float:
    # 0mm => very low rain probability, >=2mm => high probability.
    base = _sigmoid((value_mm - 0.8) / 0.5)
    if operator in ("lt", "lte"):
        return 1.0 - base
    return base


class OpenMeteoWeatherAdapter(WeatherModelAdapter):
    """Open-Meteo-backed adapter using GFS + ECMWF model runs."""

    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FC_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout_seconds: float = 15.0):
        self._timeout = timeout_seconds

    async def forecast_probability(
        self, contract: WeatherForecastInput
    ) -> WeatherForecastResult:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                lat, lon, resolved_name = await self._resolve_location(
                    client, contract.location
                )
                gfs_val = await self._fetch_model_value(
                    client, lat, lon, contract.target_time, contract.metric, "gfs_seamless"
                )
                ecmwf_val = await self._fetch_model_value(
                    client, lat, lon, contract.target_time, contract.metric, "ecmwf_ifs04"
                )

            gfs_prob = self._to_probability(gfs_val, contract)
            ecmwf_prob = self._to_probability(ecmwf_val, contract)

            return WeatherForecastResult(
                gfs_probability=gfs_prob,
                ecmwf_probability=ecmwf_prob,
                gfs_value=gfs_val,
                ecmwf_value=ecmwf_val,
                metadata={
                    "provider": "open_meteo",
                    "location": resolved_name or contract.location,
                    "lat": lat,
                    "lon": lon,
                    "target_time": _to_utc_iso(contract.target_time),
                },
            )
        except Exception:
            # Fail-safe neutral forecast so worker never crashes from provider issues.
            return WeatherForecastResult(
                gfs_probability=0.5,
                ecmwf_probability=0.5,
                metadata={"provider": "open_meteo", "fallback": True},
            )

    async def _resolve_location(
        self, client: httpx.AsyncClient, location: str
    ) -> tuple[float, float, Optional[str]]:
        resp = await client.get(self.GEO_URL, params={"name": location, "count": 1})
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results") or []
        if not results:
            raise ValueError(f"Could not geocode location: {location}")
        hit = results[0]
        return float(hit["latitude"]), float(hit["longitude"]), hit.get("name")

    async def _fetch_model_value(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        target_time: datetime,
        metric: str,
        model: str,
    ) -> float:
        hourly_field = "temperature_2m"
        if metric.startswith("precip"):
            hourly_field = "precipitation"

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": hourly_field,
            "start_date": target_time.strftime("%Y-%m-%d"),
            "end_date": target_time.strftime("%Y-%m-%d"),
            "timezone": "UTC",
            "models": model,
            "forecast_days": 16,
        }
        resp = await client.get(self.FC_URL, params=params)
        resp.raise_for_status()
        payload = resp.json() or {}
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        vals = hourly.get(hourly_field) or []
        if not times or not vals:
            raise ValueError("Missing hourly payload")

        target_epoch = int(target_time.replace(tzinfo=timezone.utc).timestamp())

        best_i = 0
        best_diff = None
        for i, ts in enumerate(times):
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                diff = abs(int(dt.timestamp()) - target_epoch)
            except Exception:
                continue
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_i = i

        val = vals[best_i]
        return float(val if val is not None else 0.0)

    def _to_probability(self, model_value: float, contract: WeatherForecastInput) -> float:
        if contract.metric == "temp_range":
            low = contract.threshold_c_low
            high = contract.threshold_c_high
            if low is None or high is None:
                return 0.5
            return _temp_range_probability(model_value, low, high)

        if contract.metric.startswith("temp"):
            threshold = contract.threshold_c if contract.threshold_c is not None else 0.0
            return _temp_probability(model_value, threshold, contract.operator)
        return _precip_probability(model_value, contract.operator)
