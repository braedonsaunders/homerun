"""Signal taxonomy and keyword maps catalog."""

from __future__ import annotations

from typing import Any

from .catalog_loader import WorldIntelJsonCatalog

_DEFAULT = {
    "version": 0,
    "updated_at": None,
    "world_signal_types": [],
    "convergence_signal_types": [],
    "anomaly_monitored_signal_types": [],
    "market_keyword_map": {},
    "convergence_market_keyword_map": {},
    "tension_title_keywords": [],
    "convergence_signal_map": {},
    "acled_event_type_weights": {},
}


class TaxonomyCatalog:
    def __init__(self) -> None:
        self._catalog = WorldIntelJsonCatalog("signal_taxonomy.json", _DEFAULT)

    def payload(self) -> dict[str, Any]:
        return self._catalog.payload()

    def world_signal_types(self) -> set[str]:
        rows = self.payload().get("world_signal_types") or []
        return {str(v).strip() for v in rows if str(v).strip()}

    def convergence_signal_types(self) -> set[str]:
        rows = self.payload().get("convergence_signal_types") or []
        return {str(v).strip() for v in rows if str(v).strip()}

    def anomaly_monitored_signal_types(self) -> set[str]:
        rows = self.payload().get("anomaly_monitored_signal_types") or []
        return {str(v).strip() for v in rows if str(v).strip()}

    def market_keyword_map(self) -> dict[str, list[str]]:
        raw = self.payload().get("market_keyword_map") or {}
        return _coerce_keyword_map(raw)

    def convergence_market_keyword_map(self) -> dict[str, list[str]]:
        raw = self.payload().get("convergence_market_keyword_map") or {}
        return _coerce_keyword_map(raw)

    def tension_title_keywords(self) -> list[str]:
        rows = self.payload().get("tension_title_keywords") or []
        return [str(v).strip().lower() for v in rows if str(v).strip()]

    def convergence_signal_map(self) -> dict[str, str]:
        raw = self.payload().get("convergence_signal_map") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            src = str(key).strip().lower()
            dst = str(value).strip()
            if src and dst:
                out[src] = dst
        return out

    def acled_event_type_weights(self) -> dict[str, float]:
        raw = self.payload().get("acled_event_type_weights") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in raw.items():
            try:
                out[str(key).strip().lower()] = float(value)
            except Exception:
                continue
        return out


def _coerce_keyword_map(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, raw in value.items():
        if not isinstance(raw, list):
            continue
        words = [str(v).strip().lower() for v in raw if str(v).strip()]
        out[str(key).strip()] = words
    return out


taxonomy_catalog = TaxonomyCatalog()
