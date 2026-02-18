from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Optional

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


def _select_nws_temperature_c(
    *,
    periods: list[dict],
    target_time: datetime,
    metric: str,
) -> Optional[float]:
    """Select an NWS temperature representative for the contract metric.

    - ``temp_max_*``: max hourly temperature on target UTC day
    - ``temp_min_*``: min hourly temperature on target UTC day
    - fallback/default: nearest-hour temperature to target timestamp
    """
    if not periods:
        return None

    tgt = target_time if target_time.tzinfo else target_time.replace(tzinfo=timezone.utc)
    target_day = tgt.astimezone(timezone.utc).date()
    target_epoch = int(tgt.timestamp())

    day_values: list[float] = []
    nearest_temp: Optional[float] = None
    nearest_diff: Optional[int] = None

    for period in periods:
        if not isinstance(period, dict):
            continue
        raw_start = period.get("startTime")
        raw_temp = period.get("temperature")
        raw_unit = str(period.get("temperatureUnit") or "F")
        if raw_start is None or raw_temp is None:
            continue
        try:
            dt = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00"))
            dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            temp_c = _to_celsius(float(raw_temp), raw_unit)
        except Exception:
            continue

        if dt_utc.date() == target_day:
            day_values.append(temp_c)

        diff = abs(int(dt_utc.timestamp()) - target_epoch)
        if nearest_diff is None or diff < nearest_diff:
            nearest_diff = diff
            nearest_temp = temp_c

    metric_l = (metric or "").lower()
    if day_values:
        if metric_l.startswith("temp_max_"):
            return max(day_values)
        if metric_l.startswith("temp_min_"):
            return min(day_values)

    return nearest_temp


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


def _looks_rate_limited(error_text: str) -> bool:
    text = (error_text or "").lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


class OpenMeteoWeatherAdapter(WeatherModelAdapter):
    """Open-Meteo-backed adapter with weighted multi-source consensus."""

    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FC_URL = "https://api.open-meteo.com/v1/forecast"
    ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
    NWS_POINTS_URL = "https://api.weather.gov/points"

    def __init__(self, timeout_seconds: float = 15.0):
        self._timeout = timeout_seconds
        self._cache_lock: Optional[asyncio.Lock] = None
        self._geo_cache: dict[str, tuple[float, float, Optional[str], Optional[str], Optional[str]]] = {}
        self._model_value_cache: dict[tuple[Any, ...], float] = {}
        self._nws_value_cache: dict[tuple[Any, ...], Optional[float]] = {}
        self._geo_inflight: dict[
            str, asyncio.Task[tuple[float, float, Optional[str], Optional[str], Optional[str]]]
        ] = {}
        self._model_inflight: dict[tuple[Any, ...], asyncio.Task[float]] = {}
        self._nws_inflight: dict[tuple[Any, ...], asyncio.Task[Optional[float]]] = {}

    def _get_cache_lock(self) -> asyncio.Lock:
        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()
        return self._cache_lock

    def clear_cycle_cache(self) -> None:
        """Clear request caches so each scan cycle starts fresh."""
        self._geo_cache.clear()
        self._model_value_cache.clear()
        self._nws_value_cache.clear()

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
                        metric=contract.metric,
                    )

            # Fetch ensemble members (non-blocking, degrades gracefully)
            ensemble_hourly, ensemble_daily_max = await self.fetch_ensemble_members(
                location=contract.location,
                target_time=contract.target_time,
                metric=contract.metric,
            )

            value_by_source: dict[str, float] = {}
            probability_by_source: dict[str, float] = {}
            source_errors: dict[str, str] = {}
            snapshots: list[WeatherSourceSnapshot] = []

            for model, result in zip(OPEN_METEO_MODELS, model_results):
                if isinstance(result, Exception):
                    source_id = f"open_meteo:{model}"
                    source_errors[source_id] = str(result)[:220]
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
                fallback_meta: dict[str, object] = {
                    "provider": "open_meteo",
                    "fallback": True,
                }
                if source_errors:
                    fallback_meta["source_errors"] = source_errors
                    fallback_meta["rate_limited"] = any(_looks_rate_limited(err) for err in source_errors.values())
                return WeatherForecastResult(
                    gfs_probability=0.5,
                    ecmwf_probability=0.5,
                    metadata=fallback_meta,
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

            # Use daily max ensemble for temp_max contracts, hourly otherwise
            metric_l = (contract.metric or "").lower()
            result_ensemble = (
                ensemble_daily_max if metric_l.startswith("temp_max") and ensemble_daily_max else ensemble_hourly
            )
            result_ensemble_daily = ensemble_daily_max or None

            return WeatherForecastResult(
                gfs_probability=gfs_prob,
                ecmwf_probability=ecmwf_prob,
                gfs_value=gfs_value,
                ecmwf_value=ecmwf_value,
                source_snapshots=snapshots,
                consensus_probability=consensus_probability,
                consensus_value_c=consensus_value_c,
                source_spread_c=spread_c,
                ensemble_members=result_ensemble or None,
                ensemble_daily_max=result_ensemble_daily,
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
                    "source_errors": source_errors,
                    "ensemble_member_count": len(result_ensemble) if result_ensemble else 0,
                    "rate_limited": any(_looks_rate_limited(err) for err in source_errors.values()),
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
        cache_key = " ".join(str(location or "").strip().lower().split())
        if not cache_key:
            raise ValueError("Location cannot be empty")

        cache_lock = self._get_cache_lock()
        async with cache_lock:
            cached = self._geo_cache.get(cache_key)
            if cached is not None:
                return cached
            task = self._geo_inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(self._resolve_location_uncached(client=client, location=location))
                self._geo_inflight[cache_key] = task

        try:
            result = await task
        except Exception:
            async with cache_lock:
                if self._geo_inflight.get(cache_key) is task:
                    self._geo_inflight.pop(cache_key, None)
            raise

        async with cache_lock:
            self._geo_cache[cache_key] = result
            if self._geo_inflight.get(cache_key) is task:
                self._geo_inflight.pop(cache_key, None)
        return result

    async def _resolve_location_uncached(
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
        hourly_field, daily_field = self._metric_fields(metric)
        target_utc = (
            target_time.astimezone(timezone.utc)
            if target_time.tzinfo is not None
            else target_time.replace(tzinfo=timezone.utc)
        )
        cache_key = self._forecast_request_cache_key(
            lat=lat,
            lon=lon,
            target_time=target_utc,
            model=model,
            hourly_field=hourly_field,
            daily_field=daily_field,
        )

        cache_lock = self._get_cache_lock()
        async with cache_lock:
            cached = self._model_value_cache.get(cache_key)
            if cached is not None:
                return cached
            task = self._model_inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    self._fetch_model_value_uncached(
                        client=client,
                        lat=lat,
                        lon=lon,
                        target_time=target_utc,
                        model=model,
                        hourly_field=hourly_field,
                        daily_field=daily_field,
                    )
                )
                self._model_inflight[cache_key] = task

        try:
            value = await task
        except Exception:
            async with cache_lock:
                if self._model_inflight.get(cache_key) is task:
                    self._model_inflight.pop(cache_key, None)
            raise

        async with cache_lock:
            self._model_value_cache[cache_key] = value
            if self._model_inflight.get(cache_key) is task:
                self._model_inflight.pop(cache_key, None)
        return value

    @staticmethod
    def _metric_fields(metric: str) -> tuple[Optional[str], Optional[str]]:
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
        return hourly_field, daily_field

    @staticmethod
    def _forecast_request_cache_key(
        *,
        lat: float,
        lon: float,
        target_time: datetime,
        model: str,
        hourly_field: Optional[str],
        daily_field: Optional[str],
    ) -> tuple[Any, ...]:
        target_hour = target_time.replace(minute=0, second=0, microsecond=0).isoformat()
        return (
            round(lat, 4),
            round(lon, 4),
            target_hour,
            model,
            hourly_field or "",
            daily_field or "",
        )

    async def _fetch_model_value_uncached(
        self,
        *,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        target_time: datetime,
        model: str,
        hourly_field: Optional[str],
        daily_field: Optional[str],
    ) -> float:
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
        metric: str,
    ) -> Optional[float]:
        target_utc = (
            target_time.astimezone(timezone.utc)
            if target_time.tzinfo is not None
            else target_time.replace(tzinfo=timezone.utc)
        )
        metric_key = self._nws_metric_key(metric)
        target_key = (
            target_utc.strftime("%Y-%m-%d")
            if metric_key in {"temp_max", "temp_min"}
            else target_utc.replace(minute=0, second=0, microsecond=0).isoformat()
        )
        cache_key = (round(lat, 4), round(lon, 4), metric_key, target_key)

        cache_lock = self._get_cache_lock()
        async with cache_lock:
            if cache_key in self._nws_value_cache:
                return self._nws_value_cache[cache_key]
            task = self._nws_inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    self._fetch_nws_temperature_c_uncached(
                        client=client,
                        lat=lat,
                        lon=lon,
                        target_time=target_utc,
                        metric=metric,
                    )
                )
                self._nws_inflight[cache_key] = task

        try:
            value = await task
        except Exception:
            async with cache_lock:
                if self._nws_inflight.get(cache_key) is task:
                    self._nws_inflight.pop(cache_key, None)
            return None

        async with cache_lock:
            self._nws_value_cache[cache_key] = value
            if self._nws_inflight.get(cache_key) is task:
                self._nws_inflight.pop(cache_key, None)
        return value

    @staticmethod
    def _nws_metric_key(metric: str) -> str:
        metric_l = (metric or "").lower()
        if metric_l.startswith("temp_max_"):
            return "temp_max"
        if metric_l.startswith("temp_min_"):
            return "temp_min"
        return metric_l or "temp"

    async def _fetch_nws_temperature_c_uncached(
        self,
        *,
        client: httpx.AsyncClient,
        lat: float,
        lon: float,
        target_time: datetime,
        metric: str,
    ) -> Optional[float]:
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

        return _select_nws_temperature_c(
            periods=periods,
            target_time=target_time,
            metric=metric,
        )

    async def fetch_ensemble_members(
        self,
        location: str,
        target_time: datetime,
        metric: str,
    ) -> tuple[list[float], list[float]]:
        """Fetch GFS ensemble member values from the Open-Meteo Ensemble API.

        Returns (ensemble_hourly, ensemble_daily_max):
        - ensemble_hourly: per-member value at closest hour to target_time
        - ensemble_daily_max: per-member daily max on target date (for temp_max contracts)
        """
        try:
            headers = {"User-Agent": "homerun-weather-workflow/1.0"}
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                lat, lon, _, _, _ = await self._resolve_location(client, location)
                target_utc = (
                    target_time.astimezone(timezone.utc)
                    if target_time.tzinfo is not None
                    else target_time.replace(tzinfo=timezone.utc)
                )
                date_str = target_utc.strftime("%Y-%m-%d")

                params: dict[str, Any] = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date_str,
                    "end_date": date_str,
                    "timezone": "UTC",
                    "models": "gfs_ensemble_025",
                    "hourly": "temperature_2m",
                }
                resp = await client.get(self.ENSEMBLE_URL, params=params)
                resp.raise_for_status()
                payload = resp.json() or {}

                hourly = payload.get("hourly") or {}
                times = hourly.get("time") or []
                temp_data = hourly.get("temperature_2m") or []

                if not times or not temp_data:
                    return [], []

                # temp_data is a list of lists: one list per ensemble member,
                # OR a flat list if only one member. The Open-Meteo ensemble API
                # returns per-member arrays when multiple members exist.
                # Detect structure: if first element is a list, it's member-based.
                if temp_data and isinstance(temp_data[0], list):
                    # Each element is a member's time series
                    members = temp_data
                else:
                    # Flat array = single member (unusual, treat as 1 member)
                    members = [temp_data]

                # Find closest hour index to target time
                target_epoch = int(target_utc.timestamp())
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

                # Extract per-member value at closest hour
                ensemble_hourly: list[float] = []
                for member_series in members:
                    if best_i < len(member_series) and member_series[best_i] is not None:
                        ensemble_hourly.append(float(member_series[best_i]))

                # Compute per-member daily max across all hours on target day
                ensemble_daily_max: list[float] = []
                target_date = target_utc.date()
                day_indices: list[int] = []
                for i, ts in enumerate(times):
                    try:
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        if dt.date() == target_date:
                            day_indices.append(i)
                    except Exception:
                        continue

                if day_indices:
                    for member_series in members:
                        day_vals = [
                            float(member_series[i])
                            for i in day_indices
                            if i < len(member_series) and member_series[i] is not None
                        ]
                        if day_vals:
                            ensemble_daily_max.append(max(day_vals))

                return ensemble_hourly, ensemble_daily_max

        except Exception:
            return [], []

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
