"""Country code/name normalization catalog for world intelligence."""

from __future__ import annotations

from typing import Any

from .catalog_loader import WorldIntelJsonCatalog
from .military_catalog import military_catalog

_DEFAULT = {
    "countries": [],
}


class CountryCatalog:
    def __init__(self) -> None:
        self._catalog = WorldIntelJsonCatalog("country_reference.json", _DEFAULT)

    def payload(self) -> dict[str, Any]:
        return self._catalog.payload()

    def _rows(self) -> list[dict[str, str]]:
        raw = self.payload()
        countries = raw if isinstance(raw, list) else raw.get("countries", [])
        if not isinstance(countries, list):
            return []
        out: list[dict[str, str]] = []
        for row in countries:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            alpha2 = str(row.get("alpha2") or "").strip().upper()
            alpha3 = str(row.get("alpha3") or "").strip().upper()
            if not name or len(alpha2) != 2 or len(alpha3) != 3:
                continue
            out.append({"name": name, "alpha2": alpha2, "alpha3": alpha3})
        return out

    def alpha2_to_alpha3(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in self._rows():
            out[row["alpha2"]] = row["alpha3"]
        return out

    def alpha3_to_name(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in self._rows():
            out[row["alpha3"]] = row["name"]
        return out

    def name_to_alpha3(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in self._rows():
            out[row["name"].strip().upper()] = row["alpha3"]
        return out

    def normalize_iso3(self, value: str | None) -> str:
        text = str(value or "").strip().upper()
        if not text:
            return ""
        alpha3_to_name = self.alpha3_to_name()
        if len(text) == 3 and text in alpha3_to_name:
            return text

        alpha2_to_alpha3 = self.alpha2_to_alpha3()
        if len(text) == 2 and text in alpha2_to_alpha3:
            return alpha2_to_alpha3[text]

        aliases = military_catalog.country_aliases()
        if text in aliases:
            return aliases[text]

        by_name = self.name_to_alpha3()
        return by_name.get(text, "")

    def country_name(self, value: str | None) -> str:
        iso3 = self.normalize_iso3(value)
        if not iso3:
            return str(value or "").strip()
        return self.alpha3_to_name().get(iso3, iso3)


country_catalog = CountryCatalog()
