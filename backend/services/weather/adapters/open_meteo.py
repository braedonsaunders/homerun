from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Optional

import httpx

from .base import (
    WeatherForecastInput,
    WeatherForecastResult,
    WeatherModelAdapter,
    WeatherSourceSnapshot,
)


OPEN_METEO_MODELS = ("gfs_seamless", "ecmwf_ifs04", "icon_seamless")
OPEN_METEO_BASE_WEIGHTS = {
    "gfs_seamless": 0.38,
    "ecmwf_ifs04": 0.42,
    "icon_seamless": 0.20,
}
TEMP_PROBABILITY_SCALE_C = 2.0


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _to_celsius(value: float, unit: str) -> float:
    if unit.upper() == "F":
        return (value - 32.0) * (5.0 / 9.0)
    return value


def _temp_probability(value_c: float, threshold_c: float, operator: str) -> float:
    # ~2C scale keeps transitions realistic instead of binary cliffs.
    delta = value_c - threshold_c
    if operator in ("lt", "lte"):
        delta = -delta
    return max(0.0, min(1.0, _sigmoid(delta / TEMP_PROBABILITY_SCALE_C)))


def _temp_range_probability(value_c: float, low_c: float, high_c: float) -> float:
    # Approximate a band probability as CDF(high) - CDF(low) with a smooth
    # logistic CDF around the deterministic model value.
    low = min(low_c, high_c)
    high = max(low_c, high_c)
    p_above_low = _sigmoid((value_c - low) / TEMP_PROBABILITY_SCALE_C)
    p_above_high = _sigmoid((value_c - high) / TEMP_PROBABILITY_SCALE_C)
    return max(0.0, min(1.0, p_above_low - p_above_high))


def _precip_probability(value_mm: float, operator: str) -> float:
    # 0mm => very low rain probability, >=2mm => high probability.
    base = _sigmoid((value_mm - 0.8) / 0.5)
    if operator in ("lt", "lte"):
        return 1.0 - base
    return base


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    filtered = {k: max(0.0, float(v)) for k, v in weights.items() if v is not None}
    total = sum(filtered.values())
    if total <= 0:
        count = len(filtered)
        if count == 0:
            return {}
        return {k: 1.0 / count for k in filtered}
    return {k: v / total for k, v in filtered.items()}


def _weighted_average(values: dict[str, float], weights: dict[str, float]) -> Optional[float]:
    common = [k for k in values if k in weights]
    if not common:
        return None
    denom = sum(weights[k] for k in common)
    if denom <= 0:
        return None
    return sum(values[k] * weights[k] for k in common) / denom


class OpenMeteoWeatherAdapter(WeatherModelAdapter):
    """Open-Meteo-backed adapter with weighted multi-source consensus."""

    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FC_URL = "https://api.open-meteo.com/v1/forecast"
    NWS_POINTS_URL = "https://api.weather.gov/points"

    def __init__(self, timeout_seconds: float = 15.0):
        self._timeout = timeout_seconds

    async def forecast_probability(self, contract: WeatherForecastInput) -> WeatherForecastResult:
        try:
            headers = {"User-Agent": "homerun-weather-workflow/1.0"}
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                lat, lon, resolved_name, country_code, tz_name = await self._resolve_location(client, contract.location)

                model_tasks = [
                    self._fetch_model_value(
                        client=client,
                        lat=lat,
                        lon=lon,
                        target_time=contract.target_time,
                        metric=contract.metric,
                        model=model,
                    )
                    for model in OPEN_METEO_MODELS
                ]
                model_results = await asyncio.gather(*model_tasks, return_exceptions=True)

                nws_value_c: Optional[float] = None
                if country_code and country_code.upper() in {"US", "USA", "PR"} and contract.metric.startswith("temp"):
                    nws_value_c = await self._fetch_nws_temperature_c(
                        client=client,
                        lat=lat,
                        lon=lon,
                        target_time=contract.target_time,
                    )

            value_by_source: dict[str, float] = {}
            probability_by_source: dict[str, float] = {}
            snapshots: list[WeatherSourceSnapshot] = []

            for model, result in zip(OPEN_METEO_MODELS, model_results):
                if isinstance(result, Exception):
                    continue
                value_c = float(result)
                source_id = f"open_meteo:{model}"
                value_by_source[source_id] = value_c
                probability_by_source[source_id] = self._to_probability(value_c, contract)

            if nws_value_c is not None:
                source_id = "nws:hourly"
                value_by_source[source_id] = nws_value_c
                probability_by_source[source_id] = self._to_probability(nws_value_c, contract)

            if not probability_by_source:
                return WeatherForecastResult(
                    gfs_probability=0.5,
                    ecmwf_probability=0.5,
                    metadata={"provider": "open_meteo", "fallback": True},
                )

            weights = self._build_source_weights(
                target_time=contract.target_time,
                source_ids=set(probability_by_source.keys()),
            )
            consensus_probability = _weighted_average(probability_by_source, weights)
            consensus_value_c = _weighted_average(value_by_source, weights)

            spread_c = None
            if value_by_source:
                vals = list(value_by_source.values())
                spread_c = max(vals) - min(vals) if vals else None

            for source_id, prob in probability_by_source.items():
                weight = weights.get(source_id)
                value_c = value_by_source.get(source_id)
                provider, model = source_id.split(":", 1)
                snapshots.append(
                    WeatherSourceSnapshot(
                        source_id=source_id,
                        provider=provider,
                        model=model,
                        value_c=value_c,
                        probability=prob,
                        weight=weight,
                        target_time=_to_utc_iso(contract.target_time),
                    )
                )

            gfs_prob = probability_by_source.get("open_meteo:gfs_seamless", 0.5)
            ecmwf_prob = probability_by_source.get("open_meteo:ecmwf_ifs04", gfs_prob)
            gfs_value = value_by_source.get("open_meteo:gfs_seamless")
            ecmwf_value = value_by_source.get("open_meteo:ecmwf_ifs04")

            sources_payload = [
                {
                    "source_id": snap.source_id,
                    "provider": snap.provider,
                    "model": snap.model,
                    "value_c": snap.value_c,
                    "probability": snap.probability,
                    "weight": snap.weight,
                    "target_time": snap.target_time,
                }
                for snap in snapshots
            ]

            return WeatherForecastResult(
                gfs_probability=gfs_prob,
                ecmwf_probability=ecmwf_prob,
                gfs_value=gfs_value,
                ecmwf_value=ecmwf_value,
                source_snapshots=snapshots,
                consensus_probability=consensus_probability,
                consensus_value_c=consensus_value_c,
                source_spread_c=spread_c,
                metadata={
                    "provider": "open_meteo",
                    "location": resolved_name or contract.location,
                    "country_code": country_code,
                    "timezone": tz_name,
                    "lat": lat,
                    "lon": lon,
                    "target_time": _to_utc_iso(contract.target_time),
                    "source_probabilities": probability_by_source,
                    "source_values_c": value_by_source,
                    "source_weights": weights,
                    "forecast_sources": sources_payload,
                    "consensus_probability": consensus_probability,
                    "consensus_value_c": consensus_value_c,
                    "source_spread_c": spread_c,
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
    ) -> tuple[float, float, Optional[str], Optional[str], Optional[str]]:
        resp = await client.get(self.GEO_URL, params={"name": location, "count": 1})
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results") or []
        if not results:
            raise ValueError(f"Could not geocode location: {location}")
        hit = results[0]
        return (
            float(hit["latitude"]),
            float(hit["longitude"]),
            hit.get("name"),
            hit.get("country_code"),
            hit.get("timezone"),
        )

    async def _fetch_model_value(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        target_time: datetime,
        metric: str,
        model: str,
    ) -> float:
        metric_l = (metric or "").lower()
        hourly_field: Optional[str] = None
        daily_field: Optional[str] = None
        if metric_l.startswith("precip"):
            hourly_field = "precipitation"
        elif metric_l in {"temp_max_threshold", "temp_max_range"}:
            daily_field = "temperature_2m_max"
        elif metric_l in {"temp_min_threshold", "temp_min_range"}:
            daily_field = "temperature_2m_min"
        else:
            # Fallback for generic or legacy temperature contracts.
            hourly_field = "temperature_2m"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": target_time.strftime("%Y-%m-%d"),
            "end_date": target_time.strftime("%Y-%m-%d"),
            "timezone": "UTC",
            "models": model,
        }
        if daily_field is not None:
            params["daily"] = daily_field
        if hourly_field is not None:
            params["hourly"] = hourly_field
        resp = await client.get(self.FC_URL, params=params)
        resp.raise_for_status()
        payload = resp.json() or {}
        if daily_field is not None:
            daily = payload.get("daily") or {}
            days = daily.get("time") or []
            vals = daily.get(daily_field) or []
            if not days or not vals:
                raise ValueError("Missing daily payload")

            target_day = target_time.astimezone(timezone.utc).date()
            best_i = 0
            best_diff = None
            for i, day_text in enumerate(days):
                try:
                    day = datetime.fromisoformat(str(day_text)).date()
                except Exception:
                    continue
                diff = abs((day - target_day).days)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_i = i

            val = vals[best_i]
            if val is None:
                raise ValueError("Missing daily model value")
            return float(val)

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
        if val is None:
            raise ValueError("Missing hourly model value")
        return float(val)

    async def _fetch_nws_temperature_c(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        target_time: datetime,
    ) -> Optional[float]:
        try:
            points_resp = await client.get(f"{self.NWS_POINTS_URL}/{lat:.4f},{lon:.4f}")
            points_resp.raise_for_status()
            points_payload = points_resp.json() or {}
            hourly_url = ((points_payload.get("properties") or {}).get("forecastHourly") or "").strip()
            if not hourly_url:
                return None

            hourly_resp = await client.get(hourly_url)
            hourly_resp.raise_for_status()
            payload = hourly_resp.json() or {}
            periods = (payload.get("properties") or {}).get("periods") or []
            if not periods:
                return None

            target_epoch = int(target_time.replace(tzinfo=timezone.utc).timestamp())
            best_temp = None
            best_diff = None
            for period in periods:
                raw_start = period.get("startTime")
                raw_temp = period.get("temperature")
                raw_unit = str(period.get("temperatureUnit") or "F")
                if raw_start is None or raw_temp is None:
                    continue
                try:
                    dt = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00"))
                    diff = abs(int(dt.timestamp()) - target_epoch)
                    temp_c = _to_celsius(float(raw_temp), raw_unit)
                except Exception:
                    continue
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_temp = temp_c
            return best_temp
        except Exception:
            return None

    def _build_source_weights(
        self,
        target_time: datetime,
        source_ids: set[str],
    ) -> dict[str, float]:
        now = datetime.now(timezone.utc)
        tgt = target_time if target_time.tzinfo else target_time.replace(tzinfo=timezone.utc)
        days_ahead = max(0.0, (tgt - now).total_seconds() / 86400.0)

        weights: dict[str, float] = {}
        for model in OPEN_METEO_MODELS:
            source_id = f"open_meteo:{model}"
            if source_id in source_ids:
                weights[source_id] = OPEN_METEO_BASE_WEIGHTS.get(model, 0.0)

        # Near-term (<= 2d): favor GFS updates and NWS if present.
        if days_ahead <= 2.0:
            if "open_meteo:gfs_seamless" in weights:
                weights["open_meteo:gfs_seamless"] += 0.07
            if "open_meteo:ecmwf_ifs04" in weights:
                weights["open_meteo:ecmwf_ifs04"] -= 0.03
            if "open_meteo:icon_seamless" in weights:
                weights["open_meteo:icon_seamless"] -= 0.02
            if "nws:hourly" in source_ids:
                weights["nws:hourly"] = 0.30

        # Medium/long horizon: favor ECMWF more.
        if days_ahead >= 4.0:
            if "open_meteo:ecmwf_ifs04" in weights:
                weights["open_meteo:ecmwf_ifs04"] += 0.08
            if "open_meteo:gfs_seamless" in weights:
                weights["open_meteo:gfs_seamless"] -= 0.04
            if "open_meteo:icon_seamless" in weights:
                weights["open_meteo:icon_seamless"] -= 0.04
            if "nws:hourly" in source_ids:
                weights["nws:hourly"] = 0.08

        if "nws:hourly" in source_ids and "nws:hourly" not in weights:
            weights["nws:hourly"] = 0.20

        return _normalize_weights({k: v for k, v in weights.items() if k in source_ids})

    def _to_probability(self, model_value: float, contract: WeatherForecastInput) -> float:
        if contract.metric.endswith("_range"):
            low = contract.threshold_c_low
            high = contract.threshold_c_high
            if low is None or high is None:
                return 0.5
            return _temp_range_probability(model_value, low, high)

        if contract.metric.startswith("temp"):
            threshold = contract.threshold_c if contract.threshold_c is not None else 0.0
            return _temp_probability(model_value, threshold, contract.operator)
        return _precip_probability(model_value, contract.operator)
