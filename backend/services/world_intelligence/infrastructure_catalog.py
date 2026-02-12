"""Infrastructure dependency graph catalog."""

from __future__ import annotations

from typing import Any

from .catalog_loader import WorldIntelJsonCatalog
from .military_catalog import military_catalog

_DEFAULT = {
    "version": 0,
    "updated_at": None,
    "nodes": [],
    "edges": [],
    "redundancy": {},
    "trade_dependencies": {},
    "country_to_nodes": {},
}


class InfrastructureCatalog:
    def __init__(self) -> None:
        self._catalog = WorldIntelJsonCatalog("infrastructure_graph.json", _DEFAULT)

    def payload(self) -> dict[str, Any]:
        return self._catalog.payload()

    def nodes(self) -> set[str]:
        rows = self.payload().get("nodes") or []
        out: set[str] = set()
        for row in rows:
            node = str(row).strip()
            if node:
                out.add(node)
        return out

    def edges(self) -> list[tuple[str, str, float]]:
        rows = self.payload().get("edges") or []
        out: list[tuple[str, str, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            src = str(row.get("from") or "").strip()
            dst = str(row.get("to") or "").strip()
            if not src or not dst:
                continue
            try:
                weight = float(row.get("weight") or 0.0)
            except Exception:
                continue
            if weight <= 0:
                continue
            out.append((src, dst, weight))
        return out

    def redundancy(self) -> dict[str, float]:
        raw = self.payload().get("redundancy") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for node, value in raw.items():
            try:
                out[str(node)] = max(0.0, min(1.0, float(value)))
            except Exception:
                continue
        return out

    def trade_dependencies(self) -> dict[str, dict[str, float]]:
        raw = self.payload().get("trade_dependencies") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, float]] = {}
        for iso3, deps in raw.items():
            if not isinstance(deps, dict):
                continue
            dep_map: dict[str, float] = {}
            for node, value in deps.items():
                try:
                    dep_map[str(node)] = max(0.0, min(1.0, float(value)))
                except Exception:
                    continue
            code = str(iso3).upper().strip()
            if code and dep_map:
                out[code] = dep_map
        return out

    def country_to_nodes(self) -> dict[str, list[str]]:
        raw = self.payload().get("country_to_nodes") or {}
        if not isinstance(raw, dict):
            return {}
        aliases = military_catalog.country_aliases()
        out: dict[str, list[str]] = {}
        for code_raw, nodes in raw.items():
            if not isinstance(nodes, list):
                continue
            cleaned = [str(n).strip() for n in nodes if str(n).strip()]
            code = str(code_raw).upper().strip()
            if not code or not cleaned:
                continue
            if len(code) == 3 and code.isalpha():
                iso3 = code
            else:
                iso3 = aliases.get(code, "")
            if not iso3:
                continue
            existing = out.setdefault(iso3, [])
            for node in cleaned:
                if node not in existing:
                    existing.append(node)
        return out


infrastructure_catalog = InfrastructureCatalog()
