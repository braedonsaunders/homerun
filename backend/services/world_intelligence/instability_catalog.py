"""Instability priors and conflict floors catalog."""

from __future__ import annotations

from typing import Any

from .catalog_loader import WorldIntelJsonCatalog

_DEFAULT = {
    "version": 0,
    "updated_at": None,
    "default_regime_multiplier": 1.0,
    "default_baseline_risk": 12.0,
    "ucdp_active_war_floor": 70.0,
    "ucdp_minor_conflict_floor": 50.0,
    "regime_multipliers": {},
    "baseline_risk": {},
    "ucdp_active_wars": [],
    "ucdp_minor_conflicts": [],
}


class InstabilityCatalog:
    def __init__(self) -> None:
        self._catalog = WorldIntelJsonCatalog("instability_priors.json", _DEFAULT)
        self._runtime_active_wars: set[str] | None = None
        self._runtime_minor_conflicts: set[str] | None = None
        self._runtime_source: str | None = None
        self._runtime_year: int | None = None

    def payload(self) -> dict[str, Any]:
        return self._catalog.payload()

    def set_runtime_conflict_lists(
        self,
        *,
        active_wars: list[str] | set[str] | None,
        minor_conflicts: list[str] | set[str] | None,
        source: str | None = None,
        year: int | None = None,
    ) -> None:
        active = {str(v).upper().strip() for v in (active_wars or []) if str(v).strip()}
        minor = {str(v).upper().strip() for v in (minor_conflicts or []) if str(v).strip()}
        self._runtime_active_wars = active or None
        self._runtime_minor_conflicts = minor or None
        self._runtime_source = str(source or "").strip() or None
        try:
            self._runtime_year = int(year) if year is not None else None
        except Exception:
            self._runtime_year = None

    def default_regime_multiplier(self) -> float:
        try:
            return float(self.payload().get("default_regime_multiplier") or 1.0)
        except Exception:
            return 1.0

    def default_baseline_risk(self) -> float:
        try:
            return float(self.payload().get("default_baseline_risk") or 12.0)
        except Exception:
            return 12.0

    def active_war_floor(self) -> float:
        try:
            return float(self.payload().get("ucdp_active_war_floor") or 70.0)
        except Exception:
            return 70.0

    def minor_conflict_floor(self) -> float:
        try:
            return float(self.payload().get("ucdp_minor_conflict_floor") or 50.0)
        except Exception:
            return 50.0

    def regime_multipliers(self) -> dict[str, float]:
        raw = self.payload().get("regime_multipliers") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for iso3, value in raw.items():
            try:
                out[str(iso3).upper().strip()] = float(value)
            except Exception:
                continue
        return out

    def baseline_risk(self) -> dict[str, float]:
        raw = self.payload().get("baseline_risk") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for iso3, value in raw.items():
            try:
                out[str(iso3).upper().strip()] = float(value)
            except Exception:
                continue
        return out

    def active_wars(self) -> set[str]:
        if self._runtime_active_wars is not None:
            return set(self._runtime_active_wars)
        raw = self.payload().get("ucdp_active_wars") or []
        return {str(v).upper().strip() for v in raw if str(v).strip()}

    def minor_conflicts(self) -> set[str]:
        if self._runtime_minor_conflicts is not None:
            return set(self._runtime_minor_conflicts)
        raw = self.payload().get("ucdp_minor_conflicts") or []
        return {str(v).upper().strip() for v in raw if str(v).strip()}

    def runtime_source(self) -> str | None:
        return self._runtime_source

    def runtime_year(self) -> int | None:
        return self._runtime_year


instability_catalog = InstabilityCatalog()
