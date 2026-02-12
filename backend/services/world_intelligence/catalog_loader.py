"""Shared JSON catalog loader for world-intelligence data files."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "world_intelligence"


class WorldIntelJsonCatalog:
    """Loads a world-intelligence JSON file with basic mtime-aware caching."""

    def __init__(self, filename: str, default_payload: dict[str, Any]) -> None:
        self._path = _DATA_ROOT / filename
        self._default = deepcopy(default_payload)
        self._payload: dict[str, Any] = deepcopy(default_payload)
        self._loaded = False
        self._mtime_ns: int | None = None

    def _reload_if_needed(self) -> None:
        if not self._path.exists():
            if not self._loaded:
                logger.warning("World-intel catalog missing: %s", self._path)
                self._payload = deepcopy(self._default)
                self._loaded = True
            return

        try:
            stat = self._path.stat()
            mtime_ns = int(stat.st_mtime_ns)
        except Exception as exc:
            logger.warning("Failed to stat catalog %s: %s", self._path, exc)
            if not self._loaded:
                self._payload = deepcopy(self._default)
                self._loaded = True
            return

        if self._loaded and self._mtime_ns == mtime_ns:
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("catalog root must be an object")
            self._payload = raw
            self._mtime_ns = mtime_ns
            self._loaded = True
        except Exception as exc:
            logger.error("Failed loading world-intel catalog %s: %s", self._path, exc)
            self._payload = deepcopy(self._default)
            self._loaded = True
            self._mtime_ns = mtime_ns

    def payload(self) -> dict[str, Any]:
        self._reload_if_needed()
        return deepcopy(self._payload)
