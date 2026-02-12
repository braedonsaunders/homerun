"""Military identification and vessel parsing catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalog_loader import WorldIntelJsonCatalog

_DEFAULT = {
    "version": 0,
    "updated_at": None,
    "callsign_prefixes": [],
    "icao_ranges": [],
    "callsign_aircraft_types": {},
    "vessel": {
        "military_ship_type_codes": [],
        "name_keywords": [],
        "mid_iso3": {},
    },
    "country_aliases": {},
}


@dataclass(frozen=True)
class IcaoRange:
    start: int
    end: int
    country: str


class MilitaryCatalog:
    def __init__(self) -> None:
        self._catalog = WorldIntelJsonCatalog("military_profiles.json", _DEFAULT)

    def payload(self) -> dict[str, Any]:
        return self._catalog.payload()

    def callsign_prefixes(self) -> list[str]:
        rows = self.payload().get("callsign_prefixes") or []
        out: list[str] = []
        for row in rows:
            text = str(row).strip().upper()
            if text:
                out.append(text)
        return out

    def icao_ranges(self) -> list[IcaoRange]:
        rows = self.payload().get("icao_ranges") or []
        out: list[IcaoRange] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            start = str(row.get("start") or "").strip()
            end = str(row.get("end") or "").strip()
            country = str(row.get("country") or "").strip().upper()
            if not start or not end:
                continue
            try:
                out.append(IcaoRange(start=int(start, 16), end=int(end, 16), country=country))
            except Exception:
                continue
        return out

    def aircraft_type_map(self) -> dict[str, str]:
        raw = self.payload().get("callsign_aircraft_types") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            k = str(key).strip().upper()
            v = str(value).strip()
            if k and v:
                out[k] = v
        return out

    def vessel_ship_type_codes(self) -> set[int]:
        raw = (self.payload().get("vessel") or {}).get("military_ship_type_codes") or []
        out: set[int] = set()
        for row in raw:
            try:
                out.add(int(row))
            except Exception:
                continue
        return out

    def vessel_name_keywords(self) -> list[str]:
        raw = (self.payload().get("vessel") or {}).get("name_keywords") or []
        out: list[str] = []
        for row in raw:
            text = str(row).strip().lower()
            if text:
                out.append(text)
        return out

    def vessel_mid_iso3(self) -> dict[str, str]:
        raw = (self.payload().get("vessel") or {}).get("mid_iso3") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            k = str(key).strip()
            v = str(value).strip().upper()
            if k and len(v) == 3:
                out[k] = v
        return out

    def country_aliases(self) -> dict[str, str]:
        raw = self.payload().get("country_aliases") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            alias = str(key).strip().upper()
            code = str(value).strip().upper()
            if alias and len(code) == 3:
                out[alias] = code
        return out


military_catalog = MilitaryCatalog()
