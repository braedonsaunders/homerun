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
        self._runtime_rows: list[dict[str, str]] | None = None
        self._runtime_source: str | None = None

    def payload(self) -> dict[str, Any]:
        return self._catalog.payload()

    def set_runtime_rows(
        self,
        rows: list[dict[str, Any]] | None,
        *,
        source: str | None = None,
    ) -> None:
        if not rows:
            self._runtime_rows = None
            self._runtime_source = None
            return
        cleaned: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            alpha2 = str(row.get("alpha2") or "").strip().upper()
            alpha3 = str(row.get("alpha3") or "").strip().upper()
            if not name or len(alpha2) != 2 or len(alpha3) != 3:
                continue
            cleaned.append({"name": name, "alpha2": alpha2, "alpha3": alpha3})
        self._runtime_rows = cleaned or None
        self._runtime_source = str(source or "").strip() or None

    def _rows(self) -> list[dict[str, str]]:
        if self._runtime_rows:
            return list(self._runtime_rows)
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

    def runtime_source(self) -> str | None:
        return self._runtime_source


country_catalog = CountryCatalog()
