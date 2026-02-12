"""Temporal baseline anomaly detection using z-scores.

Monitors event frequencies across signal types and countries, maintaining
rolling 30-day baselines with weekday/weekend separation.  When the
current observation deviates significantly (z >= 2.0) from the historical
baseline, a TemporalAnomaly is emitted.

Designed for early detection of geopolitical shifts that prediction
market traders can act on before prices adjust.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config import settings
from .taxonomy_catalog import taxonomy_catalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Signal types that the anomaly detector monitors
MONITORED_SIGNAL_TYPES = taxonomy_catalog.anomaly_monitored_signal_types()

_BASE_MEDIUM_Z = float(
    max(0.5, getattr(settings, "WORLD_INTEL_ANOMALY_THRESHOLD", 1.8) or 1.8)
)

# Z-score severity thresholds
SEVERITY_THRESHOLDS = {
    "medium": _BASE_MEDIUM_Z,
    "high": _BASE_MEDIUM_Z + 1.0,
    "critical": _BASE_MEDIUM_Z + 2.0,
}

# Baseline window: rolling 30 days of daily observations
_BASELINE_WINDOW_DAYS = 30

# Prune observations older than 90 days
_MAX_RETENTION_DAYS = 90
_MIN_BASELINE_POINTS = int(
    max(
        3,
        getattr(settings, "WORLD_INTEL_ANOMALY_MIN_BASELINE_POINTS", 3) or 3,
    )
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DailyObservation:
    """A single day's recorded observation."""

    date: datetime
    value: float
    weekday: int  # 0=Monday, 6=Sunday


@dataclass
class TemporalAnomaly:
    """A detected anomaly where observed value deviates from baseline."""

    signal_type: str
    country: str
    z_score: float
    severity: str  # "normal" | "medium" | "high" | "critical"
    current_value: float
    baseline_mean: float
    baseline_std: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Detects temporal anomalies in event frequency data.

    Maintains per-(signal_type, country) baselines of daily observations,
    split by weekday vs weekend to account for natural periodicity.
    Uses z-scores to flag deviations.
    """

    def __init__(self) -> None:
        # Baselines keyed by (signal_type, country)
        self._baselines: dict[tuple[str, str], list[DailyObservation]] = defaultdict(list)

        # Latest recorded values for anomaly detection (today's running totals)
        self._current_values: dict[tuple[str, str], float] = {}

    # -- Recording -----------------------------------------------------------

    def record_observation(
        self,
        signal_type: str,
        country: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record an observation for a signal type and country.

        This should be called periodically (at least once per collection
        cycle).  Values are accumulated as daily observations.

        Args:
            signal_type: One of MONITORED_SIGNAL_TYPES.
            country: ISO3 country code.
            value: The observed count / rate for this period.
            timestamp: When the observation was taken (defaults to now).
        """
        ts = timestamp or datetime.now(timezone.utc)
        key = (signal_type, country)

        self._current_values[key] = value

        obs = DailyObservation(
            date=ts,
            value=value,
            weekday=ts.weekday(),
        )
        self._baselines[key].append(obs)
        self._prune_baselines(key)

    def _prune_baselines(self, key: tuple[str, str]) -> None:
        """Remove observations older than 90 days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=_MAX_RETENTION_DAYS)
        self._baselines[key] = [
            obs for obs in self._baselines[key] if obs.date >= cutoff
        ]

    # -- Baseline computation ------------------------------------------------

    def _get_baseline_stats(
        self,
        key: tuple[str, str],
        is_weekend: bool,
    ) -> tuple[float, float]:
        """Compute mean and std for the 30-day baseline, filtered by day type.

        Args:
            key: (signal_type, country) tuple.
            is_weekend: If True, use only Sat/Sun observations.

        Returns:
            (mean, std) tuple.  Returns (0.0, 1.0) if insufficient data
            to avoid division-by-zero.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=_BASELINE_WINDOW_DAYS)
        observations = self._baselines.get(key, [])

        # Filter by time window and weekday type
        filtered = [
            obs for obs in observations
            if obs.date >= cutoff
            and (obs.weekday >= 5) == is_weekend
        ]

        if len(filtered) < _MIN_BASELINE_POINTS:
            # Use whatever signal history exists so anomalies can surface
            # before day 5 of runtime.
            if len(filtered) >= 2:
                values = [obs.value for obs in filtered]
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance) if variance > 0 else 1.0
                return mean, max(std, 0.5)
            return 0.0, 1.0

        values = [obs.value for obs in filtered]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1.0

        # Floor std to avoid spurious anomalies from very stable baselines
        std = max(std, mean * 0.1) if mean > 0 else max(std, 0.5)

        return mean, std

    @staticmethod
    def _classify_severity(z_score: float) -> str:
        """Map a z-score to a severity label."""
        abs_z = abs(z_score)
        if abs_z >= SEVERITY_THRESHOLDS["critical"]:
            return "critical"
        elif abs_z >= SEVERITY_THRESHOLDS["high"]:
            return "high"
        elif abs_z >= SEVERITY_THRESHOLDS["medium"]:
            return "medium"
        return "normal"

    # -- Detection -----------------------------------------------------------

    def detect_anomalies(self) -> list[TemporalAnomaly]:
        """Compute z-scores for all active signals and return anomalies.

        An anomaly is emitted for any signal with z >= 2.0 (medium threshold).
        """
        now = datetime.now(timezone.utc)
        is_weekend = now.weekday() >= 5
        anomalies: list[TemporalAnomaly] = []

        for key, current_value in self._current_values.items():
            signal_type, country = key
            mean, std = self._get_baseline_stats(key, is_weekend)

            z_score = (current_value - mean) / std if std > 0 else 0.0
            severity = self._classify_severity(z_score)

            if severity == "normal":
                continue

            direction = "above" if z_score > 0 else "below"
            description = (
                f"{signal_type} in {country} is {abs(z_score):.1f} std devs "
                f"{direction} baseline (current={current_value:.1f}, "
                f"mean={mean:.1f}, std={std:.1f})"
            )

            anomaly = TemporalAnomaly(
                signal_type=signal_type,
                country=country,
                z_score=round(z_score, 2),
                severity=severity,
                current_value=current_value,
                baseline_mean=round(mean, 2),
                baseline_std=round(std, 2),
                detected_at=now,
                description=description,
            )
            anomalies.append(anomaly)

        # Sort by absolute z-score descending
        anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)

        if anomalies:
            logger.info(
                "Anomaly detector found %d anomalies (%d critical)",
                len(anomalies),
                sum(1 for a in anomalies if a.severity == "critical"),
            )
        return anomalies

    def get_anomalies_for_country(self, country: str) -> list[TemporalAnomaly]:
        """Detect anomalies and filter to a specific country."""
        all_anomalies = self.detect_anomalies()
        return [a for a in all_anomalies if a.country == country]

    def get_critical_anomalies(self) -> list[TemporalAnomaly]:
        """Return only anomalies with z >= 3.0 (high or critical)."""
        all_anomalies = self.detect_anomalies()
        return [
            a for a in all_anomalies
            if abs(a.z_score) >= SEVERITY_THRESHOLDS["high"]
        ]

    def export_state(self) -> dict[str, Any]:
        baselines: list[dict[str, Any]] = []
        for (signal_type, country), observations in self._baselines.items():
            rows = [
                {
                    "date": obs.date.isoformat(),
                    "value": obs.value,
                    "weekday": obs.weekday,
                }
                for obs in observations[-180:]
            ]
            baselines.append(
                {
                    "signal_type": signal_type,
                    "country": country,
                    "observations": rows,
                }
            )
        current_values = [
            {
                "signal_type": signal_type,
                "country": country,
                "value": value,
            }
            for (signal_type, country), value in self._current_values.items()
        ]
        return {
            "baselines": baselines,
            "current_values": current_values,
        }

    def import_state(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        baselines = payload.get("baselines") or []
        current_values = payload.get("current_values") or []
        next_baselines: dict[tuple[str, str], list[DailyObservation]] = defaultdict(list)
        next_current: dict[tuple[str, str], float] = {}

        for item in baselines:
            if not isinstance(item, dict):
                continue
            signal_type = str(item.get("signal_type") or "").strip()
            country = str(item.get("country") or "").strip()
            if not signal_type or not country:
                continue
            rows = item.get("observations") or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                date_raw = str(row.get("date") or "").strip()
                try:
                    dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                try:
                    value = float(row.get("value") or 0.0)
                    weekday = int(row.get("weekday") or dt.weekday())
                except Exception:
                    continue
                next_baselines[(signal_type, country)].append(
                    DailyObservation(date=dt, value=value, weekday=weekday)
                )

        for item in current_values:
            if not isinstance(item, dict):
                continue
            signal_type = str(item.get("signal_type") or "").strip()
            country = str(item.get("country") or "").strip()
            if not signal_type or not country:
                continue
            try:
                value = float(item.get("value") or 0.0)
            except Exception:
                continue
            next_current[(signal_type, country)] = value

        if next_baselines:
            self._baselines = next_baselines
            for key in list(self._baselines.keys()):
                self._prune_baselines(key)
        if next_current:
            self._current_values = next_current


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

anomaly_detector = AnomalyDetector()
