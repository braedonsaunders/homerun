"""Default tension pair catalog."""

from __future__ import annotations

from .catalog_loader import WorldIntelJsonCatalog

_DEFAULT = {
    "version": 0,
    "updated_at": None,
    "default_pairs": [],
    "query_names": {},
}


class TensionPairCatalog:
    def __init__(self) -> None:
        self._catalog = WorldIntelJsonCatalog("tension_pairs.json", _DEFAULT)

    def default_pairs(self) -> list[tuple[str, str]]:
        payload = self._catalog.payload()
        rows = payload.get("default_pairs") or []
        out: list[tuple[str, str]] = []
        for row in rows:
            text = str(row).upper().strip()
            if "-" in text:
                left, right = text.split("-", 1)
            elif ":" in text:
                left, right = text.split(":", 1)
            else:
                continue
            a = left.strip()
            b = right.strip()
            if len(a) == 2 and len(b) == 2 and a != b:
                out.append((a, b))
        return out

    def query_names(self) -> dict[str, str]:
        payload = self._catalog.payload()
        raw = payload.get("query_names") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            code = str(key).upper().strip()
            name = str(value).strip()
            if code and name:
                out[code] = name
        return out


tension_pair_catalog = TensionPairCatalog()
