"""Telonex download + catalog-registration service.

The Telonex API exposes parquet files via 302 redirects to short-lived
presigned URLs.  Each call counts against the operator's quota (5
downloads on the free trial).  This service:

  1. Resolves the presigned URL via :class:`TelonexClient.download_url`
  2. Streams the parquet bytes onto disk under
     ``{parquet_root}/_telonex/data/{exchange}/{channel}/{asset_key}/{date}.parquet``
     (kept as the raw audit copy)
  3. CONVERTS the Telonex-native schema into Homerun's ``SNAPSHOT_SCHEMA``
     and writes the canonical file at
     ``{parquet_root}/telonex/{coin}/{startISO}__{endISO}/snapshots__{asset_id}.parquet``
     so the backtester's ``ParquetBookReplay`` can read it directly.
  4. Upserts a :class:`ProviderDataset` row pointing at the CANONICAL
     window dir with ``storage_type='parquet'`` and the REAL Polymarket
     asset_id in ``token_ids_json`` — drives the Data Lab "Imported
     datasets" panel + Backtest Studio dataset picker, AND is what the
     backtester's ``find_parquet_coverage()`` filters on.
  5. Persists the latest ``X-Downloads-Remaining`` count back to
     ``AppSettings`` so the UI quota pill stays accurate.

Concurrency notes:
  * Synchronous-from-the-caller-perspective today — downloads happen
    inline on the request task.  Multiple days run sequentially so we
    never hit Telonex's per-account parallel cap and so the quota
    counter updates atomically.
  * A future async worker job is trivial to layer on top: bundle the
    range into a ``ProviderImportJob`` and have the worker call
    ``import_range()`` here.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from models.database import AppSettings, AsyncSessionLocal, ProviderDataset
from services.external_data.telonex_client import (
    TelonexAuthError,
    TelonexError,
    TelonexNotFoundError,
    TelonexValidationError,
    build_client_from_settings,
)
from services.external_data.telonex_markets_cache import catalog_dir

logger = logging.getLogger(__name__)


PROVIDER_TELONEX = "telonex"
# Legacy storage_type — kept for the audit-copy registration only.
# The CANONICAL dataset row uses 'parquet' so the backtester's
# ``find_parquet_coverage()`` (which exact-matches storage_type='parquet')
# picks it up automatically.
_STORAGE_TYPE_LEGACY = "telonex_parquet"
_STORAGE_TYPE_CANONICAL = "parquet"


def _infer_coin_from_slug(slug: Optional[str]) -> str:
    """Best-effort coin extraction from a Polymarket slug.

    Used to build the canonical path ``{root}/telonex/{coin}/...``.
    The coin column on ``provider_datasets`` is also surfaced in the
    Backtest Studio dataset picker for quick visual scanning.

    Returns one of {'btc', 'eth', 'sol', 'pred'} — 'pred' is the
    catch-all for prediction markets that aren't crypto-priced (sports,
    politics, weather, etc.) so the dataset still ends up under a
    stable known segment.
    """
    if not slug:
        return "pred"
    s = slug.lower()
    if s.startswith("btc") or "bitcoin" in s or "btc-" in s:
        return "btc"
    if s.startswith("eth") or "ethereum" in s or "eth-" in s:
        return "eth"
    if s.startswith("sol") or "solana" in s or "sol-" in s:
        return "sol"
    return "pred"


def _to_float_or_none(x: Any) -> Optional[float]:
    """Telonex stores prices/sizes as decimal strings — tolerate empties
    and non-numeric junk uniformly across every channel converter."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _write_canonical_snapshot_file(
    converted_table: Any,
    *,
    coin: str,
    real_asset_id: str,
    span_start: datetime,
    span_end: datetime,
) -> tuple[Path, str, int, datetime, datetime]:
    """Common write step — every converter ends with the same pa.Table →
    canonical-path write.  Centralised so the path layout and span-
    truncation rules stay in one place."""
    from services.external_data.parquet_schema import parquet_path_for
    from services.marketdata.writer import write_canonical_table

    # Truncate span to second precision for a clean window-dir slug.
    span_start = span_start.replace(microsecond=0)
    span_end = span_end.replace(microsecond=0)
    if span_end <= span_start:
        span_end = span_start + timedelta(seconds=1)

    dest = parquet_path_for(
        provider=PROVIDER_TELONEX,
        coin=coin,
        token_id=real_asset_id,
        start=span_start,
        end=span_end,
        kind="snapshots",
    )
    # Single canonical writer: schema-validates + lineage-stamps + atomic.
    n = write_canonical_table(
        converted_table, dest_path=dest.file_path, kind="snapshots",
        provider=PROVIDER_TELONEX, compression="snappy",
    )
    return dest.file_path, real_asset_id, n, span_start, span_end


def _convert_book_snapshot_to_canonical(
    src_file: Path,
    *,
    coin: str,
    expected_levels: int,
) -> Optional[tuple[Path, str, int, datetime, datetime]]:
    """Read a Telonex ``book_snapshot_{N}`` parquet (one day, one asset)
    and write the equivalent in Homerun's ``SNAPSHOT_SCHEMA`` at the
    canonical layout.

    ``expected_levels`` is informational only — the converter walks
    columns ``bid_price_0``, ``bid_price_1``, ... until the first
    missing prefix.  This means book_snapshot_5, book_snapshot_25 and
    book_snapshot_full (variable depth) all share one code path.

    Returns ``(dest_path, real_asset_id, n_rows, span_start, span_end)``
    or ``None`` if the file is empty / unreadable.  ``span_start`` /
    ``span_end`` are the actual first/last timestamps in the file —
    used by the caller to compute the dataset row's window so that
    ``find_parquet_coverage()`` only matches backtests whose window
    overlaps the real data span.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from services.external_data.parquet_schema import SNAPSHOT_SCHEMA

    try:
        table = pq.read_table(str(src_file))
    except Exception as exc:
        logger.warning(
            "telonex_import: unreadable parquet %s: %s", src_file, exc,
        )
        return None
    if table.num_rows == 0:
        return None

    cols = table.to_pydict()
    # Defensive — every book_snapshot_N file has timestamp_us, asset_id,
    # and at least one bid/ask level.
    required = ("timestamp_us", "asset_id", "bid_price_0", "ask_price_0")
    for c in required:
        if c not in cols:
            logger.warning(
                "telonex_import: %s missing %s — not a book_snapshot file?",
                src_file, c,
            )
            return None

    real_asset_id = cols["asset_id"][0]
    n = len(cols["timestamp_us"])

    # Detect actual level depth by walking until the first missing
    # column.  book_snapshot_full can vary per file, and book_snapshot_5
    # / _25 should match their expected depth — we accept whatever the
    # file actually contains.
    actual_levels = 0
    while f"bid_price_{actual_levels}" in cols and f"ask_price_{actual_levels}" in cols:
        actual_levels += 1
    if actual_levels == 0:
        return None
    if expected_levels and actual_levels < expected_levels:
        logger.info(
            "telonex_import: %s declared book_snapshot_%d but file only has %d levels",
            src_file, expected_levels, actual_levels,
        )

    # Per-row L2 ladders.  Drop Nones so the float list isn't sparse
    # (the backtester expects parallel-indexed price/size lists with no
    # padding — the first None terminates the ladder for that row).
    def _ladder(prefix: str) -> list[list[float]]:
        out: list[list[float]] = []
        for r in range(n):
            row: list[float] = []
            for i in range(actual_levels):
                v = _to_float_or_none(cols[f"{prefix}{i}"][r])
                if v is not None:
                    row.append(v)
                else:
                    break
            out.append(row)
        return out

    bid_p = _ladder("bid_price_")
    bid_s = _ladder("bid_size_")
    ask_p = _ladder("ask_price_")
    ask_s = _ladder("ask_size_")

    best_bid = [bp[0] if bp else None for bp in bid_p]
    best_ask = [ap[0] if ap else None for ap in ask_p]

    converted = pa.table(
        {
            "token_id":       pa.array([real_asset_id] * n, pa.string()),
            "observed_at_us": pa.array(cols["timestamp_us"], pa.int64()),
            "sequence":       pa.array([None] * n, pa.int64()),
            "best_bid":       pa.array(best_bid, pa.float64()),
            "best_ask":       pa.array(best_ask, pa.float64()),
            "spread_bps":     pa.array([None] * n, pa.float64()),
            "bids_price":     pa.array(bid_p, pa.list_(pa.float64())),
            "bids_size":      pa.array(bid_s, pa.list_(pa.float64())),
            "asks_price":     pa.array(ask_p, pa.list_(pa.float64())),
            "asks_size":      pa.array(ask_s, pa.list_(pa.float64())),
            "trade_price":    pa.array([None] * n, pa.float64()),
            "trade_size":     pa.array([None] * n, pa.float64()),
            "trade_side":     pa.array([None] * n, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )

    first_us = int(cols["timestamp_us"][0])
    last_us = int(cols["timestamp_us"][-1])
    span_start = datetime.fromtimestamp(first_us / 1e6, tz=timezone.utc)
    span_end = datetime.fromtimestamp(last_us / 1e6, tz=timezone.utc)

    return _write_canonical_snapshot_file(
        converted,
        coin=coin,
        real_asset_id=real_asset_id,
        span_start=span_start,
        span_end=span_end,
    )


def _convert_quotes_to_canonical(
    src_file: Path,
    *,
    coin: str,
) -> Optional[tuple[Path, str, int, datetime, datetime]]:
    """Read a Telonex ``quotes`` parquet (best-bid / best-ask only) and
    write the equivalent in Homerun's ``SNAPSHOT_SCHEMA``.

    Quotes are effectively book_snapshot_1: one level per side per row.
    We populate ``best_bid``/``best_ask`` directly and emit a
    single-element ``bids_price``/``asks_price`` list per row so
    book-driven strategies that read the ladder still see something.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from services.external_data.parquet_schema import SNAPSHOT_SCHEMA

    try:
        table = pq.read_table(str(src_file))
    except Exception as exc:
        logger.warning("telonex_import: unreadable parquet %s: %s", src_file, exc)
        return None
    if table.num_rows == 0:
        return None

    cols = table.to_pydict()
    # Telonex quotes columns (per Tardis spec): timestamp_us, asset_id,
    # bid_price, bid_amount (or bid_size), ask_price, ask_amount.
    # Tolerate both _amount and _size naming conventions.
    required = ("timestamp_us", "asset_id")
    for c in required:
        if c not in cols:
            logger.warning("telonex_import: %s missing %s — not quotes?", src_file, c)
            return None

    def _pick(*names: str) -> Optional[list]:
        for nm in names:
            if nm in cols:
                return cols[nm]
        return None

    bid_p_col = _pick("bid_price")
    ask_p_col = _pick("ask_price")
    bid_s_col = _pick("bid_amount", "bid_size")
    ask_s_col = _pick("ask_amount", "ask_size")
    if bid_p_col is None or ask_p_col is None:
        logger.warning("telonex_import: %s missing bid_price/ask_price", src_file)
        return None

    real_asset_id = cols["asset_id"][0]
    n = len(cols["timestamp_us"])

    bid_p = [_to_float_or_none(x) for x in bid_p_col]
    ask_p = [_to_float_or_none(x) for x in ask_p_col]
    bid_s = [_to_float_or_none(x) for x in (bid_s_col or [None] * n)]
    ask_s = [_to_float_or_none(x) for x in (ask_s_col or [None] * n)]

    bids_price_list = [[bp] if bp is not None else [] for bp in bid_p]
    bids_size_list = [
        [bs] if (bid_p[i] is not None and bs is not None) else []
        for i, bs in enumerate(bid_s)
    ]
    asks_price_list = [[ap] if ap is not None else [] for ap in ask_p]
    asks_size_list = [
        [s] if (ask_p[i] is not None and s is not None) else []
        for i, s in enumerate(ask_s)
    ]

    converted = pa.table(
        {
            "token_id":       pa.array([real_asset_id] * n, pa.string()),
            "observed_at_us": pa.array(cols["timestamp_us"], pa.int64()),
            "sequence":       pa.array([None] * n, pa.int64()),
            "best_bid":       pa.array(bid_p, pa.float64()),
            "best_ask":       pa.array(ask_p, pa.float64()),
            "spread_bps":     pa.array([None] * n, pa.float64()),
            "bids_price":     pa.array(bids_price_list, pa.list_(pa.float64())),
            "bids_size":      pa.array(bids_size_list, pa.list_(pa.float64())),
            "asks_price":     pa.array(asks_price_list, pa.list_(pa.float64())),
            "asks_size":      pa.array(asks_size_list, pa.list_(pa.float64())),
            "trade_price":    pa.array([None] * n, pa.float64()),
            "trade_size":     pa.array([None] * n, pa.float64()),
            "trade_side":     pa.array([None] * n, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )

    first_us = int(cols["timestamp_us"][0])
    last_us = int(cols["timestamp_us"][-1])
    span_start = datetime.fromtimestamp(first_us / 1e6, tz=timezone.utc)
    span_end = datetime.fromtimestamp(last_us / 1e6, tz=timezone.utc)

    return _write_canonical_snapshot_file(
        converted,
        coin=coin,
        real_asset_id=real_asset_id,
        span_start=span_start,
        span_end=span_end,
    )


def _convert_trades_to_canonical(
    src_file: Path,
    *,
    coin: str,
) -> Optional[tuple[Path, str, int, datetime, datetime]]:
    """Read a Telonex ``trades`` parquet and write per-trade rows into
    Homerun's ``SNAPSHOT_SCHEMA``.

    Trades have no book state — we populate the ``trade_price``,
    ``trade_size``, ``trade_side`` columns and leave book fields null.
    The backtester's hybrid replay still consumes these rows correctly
    (it routes per-snapshot based on which columns are non-null), and
    trade-flow strategies (VPIN, toxicity) read directly off them.

    Also writes ``onchain_fills`` since the schema is effectively the
    same (timestamp + asset_id + price + size + side, with the
    side semantics differing only by maker/taker convention which we
    normalise to BUY/SELL).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from services.external_data.parquet_schema import SNAPSHOT_SCHEMA

    try:
        table = pq.read_table(str(src_file))
    except Exception as exc:
        logger.warning("telonex_import: unreadable parquet %s: %s", src_file, exc)
        return None
    if table.num_rows == 0:
        return None

    cols = table.to_pydict()
    if "timestamp_us" not in cols or "asset_id" not in cols:
        logger.warning(
            "telonex_import: %s missing timestamp_us/asset_id — not trades?", src_file,
        )
        return None

    def _pick(*names: str) -> Optional[list]:
        for nm in names:
            if nm in cols:
                return cols[nm]
        return None

    price_col = _pick("price", "trade_price", "fill_price")
    size_col = _pick("amount", "size", "trade_amount", "trade_size", "fill_amount")
    side_col = _pick("side", "taker_side", "aggressor_side", "trade_side")
    if price_col is None or size_col is None:
        logger.warning(
            "telonex_import: %s missing price/size — not trades?", src_file,
        )
        return None

    real_asset_id = cols["asset_id"][0]
    n = len(cols["timestamp_us"])

    trade_p = [_to_float_or_none(x) for x in price_col]
    trade_s = [_to_float_or_none(x) for x in size_col]
    # Normalise side to BUY/SELL strings (some channels use buy/sell,
    # b/s, maker/taker; we just lowercase + first-letter map).
    def _norm_side(x: Any) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip().lower()
        if not s:
            return None
        if s.startswith("b"):
            return "BUY"
        if s.startswith("s"):
            return "SELL"
        # maker/taker fills don't carry direction — leave null
        return None

    if side_col is not None:
        trade_side = [_norm_side(x) for x in side_col]
    else:
        trade_side = [None] * n

    converted = pa.table(
        {
            "token_id":       pa.array([real_asset_id] * n, pa.string()),
            "observed_at_us": pa.array(cols["timestamp_us"], pa.int64()),
            "sequence":       pa.array([None] * n, pa.int64()),
            "best_bid":       pa.array([None] * n, pa.float64()),
            "best_ask":       pa.array([None] * n, pa.float64()),
            "spread_bps":     pa.array([None] * n, pa.float64()),
            "bids_price":     pa.array([[] for _ in range(n)], pa.list_(pa.float64())),
            "bids_size":      pa.array([[] for _ in range(n)], pa.list_(pa.float64())),
            "asks_price":     pa.array([[] for _ in range(n)], pa.list_(pa.float64())),
            "asks_size":      pa.array([[] for _ in range(n)], pa.list_(pa.float64())),
            "trade_price":    pa.array(trade_p, pa.float64()),
            "trade_size":     pa.array(trade_s, pa.float64()),
            "trade_side":     pa.array(trade_side, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )

    first_us = int(cols["timestamp_us"][0])
    last_us = int(cols["timestamp_us"][-1])
    span_start = datetime.fromtimestamp(first_us / 1e6, tz=timezone.utc)
    span_end = datetime.fromtimestamp(last_us / 1e6, tz=timezone.utc)

    return _write_canonical_snapshot_file(
        converted,
        coin=coin,
        real_asset_id=real_asset_id,
        span_start=span_start,
        span_end=span_end,
    )


# Channel → converter dispatch.  Lets ``import_range`` stay agnostic
# about which channel the operator picked — every channel runs the same
# download-then-convert-then-register pipeline.
_CHANNEL_CONVERTERS: dict[str, Any] = {
    "book_snapshot_5":    lambda p, *, coin: _convert_book_snapshot_to_canonical(p, coin=coin, expected_levels=5),
    "book_snapshot_25":   lambda p, *, coin: _convert_book_snapshot_to_canonical(p, coin=coin, expected_levels=25),
    "book_snapshot_full": lambda p, *, coin: _convert_book_snapshot_to_canonical(p, coin=coin, expected_levels=0),
    "quotes":             lambda p, *, coin: _convert_quotes_to_canonical(p, coin=coin),
    "trades":             lambda p, *, coin: _convert_trades_to_canonical(p, coin=coin),
    "onchain_fills":      lambda p, *, coin: _convert_trades_to_canonical(p, coin=coin),
}


def _convert_book_snapshot_5_to_canonical(
    src_file: Path,
    *,
    coin: str,
) -> Optional[tuple[Path, str, int, datetime, datetime]]:
    """Back-compat thin wrapper.  Kept so any external callers /
    tests importing the old name keep working — all new code should
    call ``_convert_book_snapshot_to_canonical`` directly."""
    return _convert_book_snapshot_to_canonical(
        src_file, coin=coin, expected_levels=5,
    )


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelonexImportSpec:
    """One asset + channel + date range to fetch.

    Identifier resolution (the API requires exactly one):
      * ``asset_id`` (highest precision)
      * ``market_id`` + ``outcome`` / ``outcome_id``
      * ``slug`` + ``outcome`` / ``outcome_id``

    For Binance, slug / market_id / asset_id all hold the same value
    (the lowercase symbol, e.g. ``btcusdt``) and ``outcome`` is
    irrelevant.
    """

    exchange: str
    channel: str
    start_date: str  # YYYY-MM-DD (inclusive)
    end_date: str    # YYYY-MM-DD (inclusive)
    asset_id: Optional[str] = None
    market_id: Optional[str] = None
    slug: Optional[str] = None
    outcome: Optional[str] = None
    outcome_id: Optional[int] = None

    def validate(self) -> None:
        if not self.exchange:
            raise TelonexValidationError("exchange is required")
        if not self.channel:
            raise TelonexValidationError("channel is required")
        if not (_DATE_RE.match(self.start_date) and _DATE_RE.match(self.end_date)):
            raise TelonexValidationError("dates must be ISO YYYY-MM-DD")
        if not (self.asset_id or self.market_id or self.slug):
            raise TelonexValidationError(
                "provide one of asset_id, market_id, or slug"
            )
        if (self.market_id or self.slug) and self.exchange.lower() == "polymarket":
            if not (self.outcome or self.outcome_id is not None):
                raise TelonexValidationError(
                    "polymarket market_id/slug requires outcome or outcome_id"
                )

    def asset_key(self) -> str:
        """Stable, filesystem-safe key identifying this asset.

        Priority asset_id > market_id+outcome > slug+outcome.  Long
        polymarket asset_ids are hashed to keep paths short.
        """
        if self.asset_id:
            return _safe_asset_segment(self.asset_id)
        outcome_part = (
            self.outcome
            if self.outcome
            else (f"o{self.outcome_id}" if self.outcome_id is not None else "")
        )
        base = self.market_id or self.slug or "unknown"
        seg = base if not outcome_part else f"{base}__{outcome_part}"
        return _safe_asset_segment(seg)

    def external_id(self) -> str:
        """Unique key for ProviderDataset.(provider, external_id)."""
        return f"{self.exchange}:{self.channel}:{self.asset_key()}"

    def label(self) -> str:
        outcome_part = (
            f" / {self.outcome}"
            if self.outcome
            else (f" / outcome_{self.outcome_id}" if self.outcome_id is not None else "")
        )
        asset = self.asset_id or self.market_id or self.slug or "?"
        # Asset IDs can be 70+ chars — truncate for human-readable label.
        if len(asset) > 32:
            asset = asset[:14] + "…" + asset[-8:]
        return f"{self.exchange} · {self.channel} · {asset}{outcome_part}"


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelonexDayResult:
    date: str
    ok: bool
    bytes: int
    path: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class TelonexImportResult:
    spec: TelonexImportSpec
    dataset_id: Optional[str]
    storage_uri: Optional[str]
    days_requested: int
    days_succeeded: int
    days_failed: int
    bytes_downloaded: int
    quota_remaining: Optional[int]
    day_results: list[TelonexDayResult]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def import_range(spec: TelonexImportSpec) -> TelonexImportResult:
    """Download every day in the spec's range and register the dataset.

    This is the only function in the module that spends quota.  One
    HTTP call per requested day; aborts early on the first 403 (quota
    exhausted) but still registers the partial slice that landed.
    """
    spec.validate()

    target_dir = _data_dir_for(spec)
    target_dir.mkdir(parents=True, exist_ok=True)
    days = list(_iter_dates(spec.start_date, spec.end_date))

    client = await build_client_from_settings(require_api_key=True)
    day_results: list[TelonexDayResult] = []
    total_bytes = 0
    succeeded: list[str] = []
    last_remaining: Optional[int] = None
    aborted = False
    # Filled by the conversion step.  Each entry:
    #   (date_str, canonical_file_path, real_asset_id, n_rows, span_start, span_end)
    # Used after the download loop to register the canonical
    # provider_datasets row pointing at the converted files.
    canonical_outputs: list[tuple[str, Path, str, int, datetime, datetime]] = []
    # Telonex enforces "exactly one of asset_id | market_id+outcome |
    # slug+outcome".  When asset_id is set, suppress the others on the
    # wire even if our spec carries them (we still keep ``spec.slug``
    # locally for coin inference / labeling — the API just doesn't want
    # to see both).  Same when market_id is set (slug becomes wire-only
    # noise).
    if spec.asset_id:
        wire_market_id, wire_slug = None, None
        wire_outcome, wire_outcome_id = None, None
    elif spec.market_id:
        wire_market_id = spec.market_id
        wire_slug = None
        wire_outcome, wire_outcome_id = spec.outcome, spec.outcome_id
    else:
        wire_market_id = None
        wire_slug = spec.slug
        wire_outcome, wire_outcome_id = spec.outcome, spec.outcome_id

    try:
        for d in days:
            target_path = target_dir / f"{d}.parquet"
            try:
                result = await client.download_to_path(
                    exchange=spec.exchange,
                    channel=spec.channel,
                    date=d,
                    target_path=target_path,
                    asset_id=spec.asset_id,
                    market_id=wire_market_id,
                    slug=wire_slug,
                    outcome=wire_outcome,
                    outcome_id=wire_outcome_id,
                )
            except TelonexNotFoundError as exc:
                # No data for this day — log + continue with the rest
                # of the range.  Doesn't burn the quota counter (Telonex
                # returns 404 before charging).
                day_results.append(TelonexDayResult(
                    date=d, ok=False, bytes=0, path=None,
                    error=f"no data: {exc}",
                ))
                continue
            except TelonexAuthError as exc:
                # 403 = quota exhausted.  Record the latest remaining
                # value and stop — burning further days won't help.
                last_remaining = exc.downloads_remaining
                day_results.append(TelonexDayResult(
                    date=d, ok=False, bytes=0, path=None, error=str(exc),
                ))
                aborted = True
                break
            except (TelonexError, OSError) as exc:
                day_results.append(TelonexDayResult(
                    date=d, ok=False, bytes=0, path=None, error=str(exc),
                ))
                continue

            total_bytes += int(result.get("bytes") or 0)
            last_remaining = result.get("downloads_remaining") if result.get("downloads_remaining") is not None else last_remaining
            succeeded.append(d)
            day_results.append(TelonexDayResult(
                date=d, ok=True, bytes=int(result.get("bytes") or 0),
                path=str(target_path), error=None,
            ))
            # Convert the Telonex-native parquet → Homerun's
            # SNAPSHOT_SCHEMA at the canonical layout so the
            # backtester can read it directly.  Every documented
            # Telonex channel has a converter registered in
            # ``_CHANNEL_CONVERTERS``; unknown channels fall through to
            # the legacy ``telonex_parquet`` audit-only registration.
            converter = _CHANNEL_CONVERTERS.get(spec.channel)
            if converter is not None:
                try:
                    converted = converter(
                        target_path,
                        coin=_infer_coin_from_slug(spec.slug),
                    )
                    if converted is not None:
                        canonical_outputs.append((d, *converted))
                except Exception:
                    logger.exception(
                        "telonex_import: schema conversion failed for %s "
                        "(channel=%s)",
                        target_path, spec.channel,
                    )
    finally:
        # Always update the cached quota counter when we observed one,
        # even on partial failure.
        try:
            if last_remaining is not None or client.stats().get("last_downloads_remaining") is not None:
                rem = last_remaining if last_remaining is not None else client.stats().get("last_downloads_remaining")
                await _persist_quota(int(rem))
        finally:
            await client.close()

    dataset_id: Optional[str] = None
    storage_uri: Optional[str] = None
    if succeeded:
        # When the converter produced canonical SNAPSHOT_SCHEMA files,
        # register the CANONICAL dataset (storage_type='parquet') —
        # that's what the backtester's find_parquet_coverage() picks
        # up.  Otherwise (unsupported channel, or every conversion
        # raised) fall back to the legacy ``telonex_parquet`` row so
        # the audit copy still appears in the Data Lab.
        if canonical_outputs:
            dataset_id, storage_uri = await _register_canonical_dataset(
                spec, canonical_outputs, total_bytes,
            )
        else:
            dataset_id, storage_uri = await _upsert_dataset(
                spec, target_dir, succeeded, total_bytes,
            )

    return TelonexImportResult(
        spec=spec,
        dataset_id=dataset_id,
        storage_uri=storage_uri,
        days_requested=len(days),
        days_succeeded=len(succeeded),
        days_failed=len(days) - len(succeeded) - (0 if not aborted else 0),
        bytes_downloaded=total_bytes,
        quota_remaining=last_remaining,
        day_results=day_results,
    )


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _data_dir_for(spec: TelonexImportSpec) -> Path:
    return (
        catalog_dir()
        / "data"
        / spec.exchange.lower()
        / spec.channel.lower()
        / spec.asset_key()
    )


_FORBIDDEN_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_asset_segment(value: str, *, max_len: int = 48) -> str:
    """Filesystem-safe representation of a slug / asset_id.

    Polymarket asset_ids are 70+ char decimal strings — preserve full
    fidelity but hash long values to keep path lengths sane on Windows
    (260-char limit unless long-path support is on).
    """
    cleaned = _FORBIDDEN_PATH_CHARS.sub("-", value or "")
    if len(cleaned) <= max_len:
        return cleaned
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    prefix = cleaned[: max_len - 9]  # leave room for "-{8-hex-chars}"
    return f"{prefix}-{h}"


def _iter_dates(start: str, end: str):
    """Inclusive day iterator, ISO YYYY-MM-DD strings."""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    if e < s:
        return
    d = s
    while d <= e:
        yield d.isoformat()
        d = d + timedelta(days=1)


def _file_uri(path: Path) -> str:
    """Cross-platform ``file://`` URI for a filesystem path."""
    abs_path = path.resolve()
    return abs_path.as_uri()


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------


async def _register_canonical_dataset(
    spec: TelonexImportSpec,
    canonical_outputs: list[tuple[str, Path, str, int, datetime, datetime]],
    total_bytes: int,
) -> tuple[str, str]:
    """Register a ``ProviderDataset`` row pointing at the canonical
    SNAPSHOT_SCHEMA parquet files the converter produced.

    Key differences from the legacy ``_upsert_dataset``:
      • ``storage_type='parquet'`` (canonical) so the backtester's
        ``find_parquet_coverage()`` exact-match filter picks it up.
      • ``token_ids_json`` holds the REAL Polymarket asset_id (78-char
        decimal), NOT the synthetic ``telonex:polymarket:...`` token
        the legacy code wrote — so the backtester's per-token routing
        actually matches what live opportunities reference.
      • ``storage_uri`` points at the canonical window directory
        (which may contain Up + Down outcome files side-by-side).
      • ``start_ts/end_ts`` reflect the actual data span (~12 min for
        a 5-min binary), not the requested calendar-day range.
    """
    # All converted files for one import call land in the same
    # window_dir (per the canonical layout).  Pull it from the first
    # entry and trust the rest match (the converter computes the same
    # path from the same span).
    first_path = canonical_outputs[0][1]
    window_dir = first_path.parent
    storage_uri = _file_uri(window_dir)

    # Real asset_ids — deduped while preserving insertion order so the
    # Up outcome comes before Down in the typical 2-outcome import.
    real_asset_ids: list[str] = []
    seen: set[str] = set()
    for _date, _path, asset_id, _n, _s, _e in canonical_outputs:
        if asset_id not in seen:
            real_asset_ids.append(asset_id)
            seen.add(asset_id)

    # Window: the UNION of every converted file's actual span.  For a
    # multi-day import of an ongoing market this might span hours;
    # for a single 5-min binary it's ~12 min.
    span_start = min(s for _d, _p, _a, _n, s, _e in canonical_outputs)
    span_end = max(e for _d, _p, _a, _n, _s, e in canonical_outputs)

    total_rows = sum(n for _d, _p, _a, n, _s, _e in canonical_outputs)
    dates_in_import = sorted({d for d, _p, _a, _n, _s, _e in canonical_outputs})

    external_id = spec.external_id()
    dataset_id = "telonex:" + hashlib.sha1(external_id.encode("utf-8")).hexdigest()[:16]

    payload: dict[str, Any] = {
        "exchange": spec.exchange,
        "channel": spec.channel,
        "asset_id": spec.asset_id,
        "market_id": spec.market_id,
        "slug": spec.slug,
        "outcome": spec.outcome,
        "outcome_id": spec.outcome_id,
        "dates_imported": dates_in_import,
        "last_run_bytes": int(total_bytes),
        "canonical": True,
        "schema_version": "snapshots_v1",
    }

    asset_class = "prediction" if spec.exchange.lower() == "polymarket" else "spot"
    coin = _infer_coin_from_slug(spec.slug)

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderDataset).where(
                    ProviderDataset.provider == PROVIDER_TELONEX,
                    ProviderDataset.external_id == external_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = ProviderDataset(
                id=dataset_id,
                provider=PROVIDER_TELONEX,
                external_id=external_id,
                external_slug=spec.slug,
                coin=coin,
                title=spec.label(),
                asset_class=asset_class,
                token_ids_json=real_asset_ids,
                storage_type=_STORAGE_TYPE_CANONICAL,
                storage_uri=storage_uri,
                start_ts=span_start,
                end_ts=span_end,
                snapshot_count=total_rows,
                trade_count=0,
                last_imported_at=datetime.now(timezone.utc),
                payload_json=payload,
            )
            session.add(row)
        else:
            # Re-import for the same (provider, external_id): merge.
            # Token ids replace (Up + Down are stable per market);
            # window widens (start = min, end = max); snapshot count
            # = sum of converted rows; payload updated.
            row.coin = coin
            row.title = spec.label()
            row.asset_class = asset_class
            row.token_ids_json = real_asset_ids
            row.storage_type = _STORAGE_TYPE_CANONICAL
            row.storage_uri = storage_uri
            row.start_ts = min(row.start_ts or span_start, span_start)
            row.end_ts = max(row.end_ts or span_end, span_end)
            row.snapshot_count = total_rows
            row.last_imported_at = datetime.now(timezone.utc)
            row.payload_json = payload
        await session.commit()

    # Bus catalog: the federated ``polymarket.book.snapshot`` topic
    # already includes ``data/parquet/telonex/`` as one of its members
    # (see SEED_TOPICS in services.recorded_event_bus.catalog), and
    # the external_parquet adapter walks recursively to find every
    # coin / window directory under that root.  So a Telonex import
    # is automatically visible through the bus the moment its files
    # land on disk — no per-channel topic registration required.
    #
    # The legacy ``provider_datasets`` row we write above keeps the
    # operator-facing "what windows have I imported" audit log intact.

    return dataset_id, storage_uri


async def _upsert_dataset(
    spec: TelonexImportSpec,
    target_dir: Path,
    succeeded_dates: list[str],
    total_bytes: int,
) -> tuple[str, str]:
    """Upsert a ProviderDataset row pointing at the directory of day-files.

    Stable ID: ``telonex:{sha1(external_id)[:16]}``.  Re-running the
    same import expands ``start_ts/end_ts`` and the ``payload_json``
    file index rather than creating a duplicate.
    """
    external_id = spec.external_id()
    dataset_id = "telonex:" + hashlib.sha1(external_id.encode("utf-8")).hexdigest()[:16]
    storage_uri = _file_uri(target_dir)

    # Determine the cumulative date window by walking the directory —
    # picks up days from prior import calls.
    all_dates: list[str] = []
    try:
        for p in sorted(target_dir.glob("*.parquet")):
            stem = p.stem
            if _DATE_RE.match(stem):
                all_dates.append(stem)
    except OSError:
        all_dates = list(succeeded_dates)

    if not all_dates:
        all_dates = list(succeeded_dates)
    start_ts = datetime.strptime(min(all_dates), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_ts = datetime.strptime(max(all_dates), "%Y-%m-%d").replace(tzinfo=timezone.utc)

    payload: dict[str, Any] = {
        "exchange": spec.exchange,
        "channel": spec.channel,
        "asset_id": spec.asset_id,
        "market_id": spec.market_id,
        "slug": spec.slug,
        "outcome": spec.outcome,
        "outcome_id": spec.outcome_id,
        "files": all_dates,
        "last_run_bytes": int(total_bytes),
    }

    token_id = f"{PROVIDER_TELONEX}:{spec.exchange}:{spec.channel}:{spec.asset_key()}"
    asset_class = "prediction" if spec.exchange.lower() == "polymarket" else "spot"

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ProviderDataset).where(
                    ProviderDataset.provider == PROVIDER_TELONEX,
                    ProviderDataset.external_id == external_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = ProviderDataset(
                id=dataset_id,
                provider=PROVIDER_TELONEX,
                external_id=external_id,
                external_slug=spec.slug,
                title=spec.label(),
                asset_class=asset_class,
                token_ids_json=[token_id],
                storage_type=_STORAGE_TYPE_LEGACY,
                storage_uri=storage_uri,
                start_ts=start_ts,
                end_ts=end_ts,
                snapshot_count=len(all_dates),
                trade_count=0,
                last_imported_at=datetime.now(timezone.utc),
                payload_json=payload,
            )
            session.add(row)
        else:
            row.title = spec.label()
            row.asset_class = asset_class
            row.token_ids_json = [token_id]
            row.storage_type = _STORAGE_TYPE_LEGACY
            row.storage_uri = storage_uri
            row.start_ts = start_ts
            row.end_ts = end_ts
            row.snapshot_count = len(all_dates)
            row.last_imported_at = datetime.now(timezone.utc)
            row.payload_json = payload
        await session.commit()

    return dataset_id, storage_uri


async def _persist_quota(remaining: int) -> None:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
        if row is None:
            row = AppSettings(id="default")
            session.add(row)
        row.telonex_downloads_remaining = int(remaining)
        row.telonex_downloads_remaining_at = datetime.now(timezone.utc)
        await session.commit()


async def get_quota_snapshot() -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    if row is None:
        return {"remaining": None, "checked_at": None}
    val = getattr(row, "telonex_downloads_remaining", None)
    at = getattr(row, "telonex_downloads_remaining_at", None)
    return {
        "remaining": int(val) if val is not None else None,
        "checked_at": at.isoformat() if at else None,
    }


__all__ = [
    "PROVIDER_TELONEX",
    "TelonexImportSpec",
    "TelonexImportResult",
    "TelonexDayResult",
    "import_range",
    "get_quota_snapshot",
]
