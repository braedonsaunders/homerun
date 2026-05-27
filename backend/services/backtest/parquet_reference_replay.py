"""Backtest reference-price replay — the historical analogue of
``services.reference_runtime.ReferenceRuntime``.

Live, strategies read the underlying oracle/exchange price off the market
row: ``market_runtime`` stamps ``oracle_prices_by_source`` + ``price_to_beat``
onto each row from ``ReferenceRuntime`` (reference_runtime.py), and the
strategy reads them via ``pick_oracle_source`` / ``market["price_to_beat"]``.

In backtest we must reproduce that EXACTLY, sourced from recorded
``reference__*.parquet`` (written by ``crypto_ohlc_recorder``) instead of a
live WS feed.  This class loads those files for a window and answers
point-in-time queries with the SAME method names ``ReferenceRuntime``
exposes, so the discovery/annotation seam is identical to live — no
divergent code path.

Design:
  * Pure in-memory after load; no DB, no network during replay.
  * As-of semantics: ``get_oracle_prices_by_source(asset, at_ts)`` returns
    the most recent tick per source AT OR BEFORE ``at_ts`` (mirrors how a
    live feed's "latest price" only reflects ticks already received), with
    a freshness ``age_ms`` computed against ``at_ts``.
  * ``price_to_beat(asset, window_start)`` = Chainlink price as-of the
    window open (the resolution baseline), matching the live
    ``get_price_at_time`` lookup market_runtime uses.
"""
from __future__ import annotations

import bisect
from datetime import datetime, timezone
from typing import Any, Iterable

import pyarrow.parquet as pq


def _to_utc_ms(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.astimezone(timezone.utc).timestamp() * 1000)


class ParquetReferenceReplay:
    """Load recorded reference parquet and answer as-of price queries."""

    # source-id → canonical asset are encoded in the series id ref_{coin}_{source}
    def __init__(self, series_files: dict[str, list[str]]):
        # per (asset, source): parallel arrays sorted by observed_ms
        self._obs_ms: dict[tuple[str, str], list[int]] = {}
        self._price: dict[tuple[str, str], list[float]] = {}
        self._src_ms: dict[tuple[str, str], list[int]] = {}
        self._assets: set[str] = set()
        self._sources: set[str] = set()
        self._load(series_files)

    @classmethod
    async def for_window(
        cls,
        *,
        assets: Iterable[str] | None,
        start: datetime,
        end: datetime,
    ) -> "ParquetReferenceReplay":
        """Build from the ProviderDataset catalog via find_reference_coverage."""
        from services.external_data import parquet_scanner as _scan
        try:
            await _scan.ensure_recent_scan(max_age_seconds=60.0)
        except Exception:
            pass
        cov = await _scan.find_reference_coverage(assets=assets, start=start, end=end)
        return cls(cov)

    def _load(self, series_files: dict[str, list[str]]) -> None:
        # accumulate per-key rows then sort once
        acc: dict[tuple[str, str], list[tuple[int, float, int]]] = {}
        for sid, files in (series_files or {}).items():
            for f in files:
                try:
                    d = pq.read_table(
                        f, columns=["observed_at_us", "asset", "source", "price", "source_ts_ms"]
                    ).to_pydict()
                except Exception:
                    continue
                for i in range(len(d.get("observed_at_us", []))):
                    asset = str(d["asset"][i]).upper()
                    source = str(d["source"][i])
                    price = d["price"][i]
                    if price is None or float(price) <= 0:
                        continue
                    obs_ms = int(d["observed_at_us"][i]) // 1000
                    src_ms = int(d["source_ts_ms"][i] or obs_ms)
                    acc.setdefault((asset, source), []).append((obs_ms, float(price), src_ms))
                    self._assets.add(asset)
                    self._sources.add(source)
        for key, rows in acc.items():
            rows.sort(key=lambda r: r[0])
            self._obs_ms[key] = [r[0] for r in rows]
            self._price[key] = [r[1] for r in rows]
            self._src_ms[key] = [r[2] for r in rows]

    @property
    def loaded(self) -> bool:
        return bool(self._obs_ms)

    def coverage(self) -> dict[str, Any]:
        return {
            "assets": sorted(self._assets),
            "sources": sorted(self._sources),
            "series": {f"{a}:{s}": len(self._obs_ms[(a, s)]) for (a, s) in self._obs_ms},
        }

    def _as_of_index(self, key: tuple[str, str], at_ms: int) -> int | None:
        arr = self._obs_ms.get(key)
        if not arr:
            return None
        # rightmost index with obs_ms <= at_ms
        i = bisect.bisect_right(arr, at_ms) - 1
        return i if i >= 0 else None

    # ── ReferenceRuntime-compatible surface (as-of a backtest timestamp) ──
    def get_oracle_prices_by_source(self, asset: str, at_ts: datetime) -> dict[str, dict[str, Any]]:
        asset_u = str(asset or "").strip().upper()
        at_ms = _to_utc_ms(at_ts)
        out: dict[str, dict[str, Any]] = {}
        for (a, s) in self._obs_ms:
            if a != asset_u:
                continue
            idx = self._as_of_index((a, s), at_ms)
            if idx is None:
                continue
            updated_ms = self._src_ms[(a, s)][idx]
            out[s] = {
                "source": s,
                "price": self._price[(a, s)][idx],
                "updated_at_ms": updated_ms,
                "age_ms": max(0.0, float(at_ms - self._obs_ms[(a, s)][idx])),
            }
        return out

    def get_oracle_price(self, asset: str, at_ts: datetime) -> dict[str, Any] | None:
        """Primary price = Chainlink (resolution source) if present, else freshest."""
        by_src = self.get_oracle_prices_by_source(asset, at_ts)
        if not by_src:
            return None
        for pref in ("chainlink", "chainlink_direct"):
            if pref in by_src:
                return by_src[pref]
        # else freshest
        return min(by_src.values(), key=lambda r: r["age_ms"])

    def get_price_at_time(self, asset: str, timestamp_s: float, *, source: str = "chainlink",
                          max_gap_seconds: float = 60.0) -> float | None:
        """Closest recorded price to ``timestamp_s`` for a source (defaults to
        Chainlink — the resolution baseline). Mirrors
        ``ChainlinkFeed.get_price_at_time`` used for price-to-beat."""
        asset_u = str(asset or "").strip().upper()
        key = (asset_u, source)
        arr = self._obs_ms.get(key)
        if not arr:
            return None
        target_ms = int(float(timestamp_s) * 1000.0)
        i = bisect.bisect_left(arr, target_ms)
        best = None; bd = float("inf")
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(arr):
                dd = abs(arr[j] - target_ms)
                if dd < bd:
                    bd = dd; best = self._price[key][j]
        return best if bd <= max_gap_seconds * 1000.0 else None

    def price_to_beat(self, asset: str, window_start: datetime, *, source: str = "chainlink") -> float | None:
        return self.get_price_at_time(asset, _to_utc_ms(window_start) / 1000.0, source=source)
