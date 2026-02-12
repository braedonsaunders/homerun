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

    def payload(self) -> dict[str, Any]:
        return self._catalog.payload()

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
        raw = self.payload().get("ucdp_active_wars") or []
        return {str(v).upper().strip() for v in raw if str(v).strip()}

    def minor_conflicts(self) -> set[str]:
        raw = self.payload().get("ucdp_minor_conflicts") or []
        return {str(v).upper().strip() for v in raw if str(v).strip()}


instability_catalog = InstabilityCatalog()
