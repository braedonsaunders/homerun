"""The market-data provider contract.

Every source of historical book data — the live WebSocket ingestor, the
polybacktest importer, the telonex importer, and any future vendor — converges
on the SAME canonical output, enforced in two places:

  * :func:`services.marketdata.writer.write_canonical_table` is the only
    function that writes SNAPSHOT/DELTA parquet (schema-validated, lineage-
    stamped, atomic), and
  * a ``ProviderDataset(storage_type='parquet')`` catalog row makes the bytes
    discoverable by :func:`services.marketdata.coverage.resolve_coverage`.

This ``Protocol`` documents that contract structurally (no inheritance
required): a provider knows its ``provider_name`` and how to fetch a window of
its source data and land it as canonical parquet + a catalog row. Adding a new
vendor is "implement this shape and call ``write_canonical_table``" — not
"invent a new write path."
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MarketDataProvider(Protocol):
    """Structural contract for a historical book-data provider."""

    #: Stable provider key — the first path segment under the parquet root
    #: (``{root}/{provider_name}/{coin}/{window}/...``) and the
    #: ``ProviderDataset.provider`` value.
    provider_name: str

    async def fetch_to_canonical(
        self,
        *,
        coin: str,
        market_ids: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """Fetch the requested window from the provider's source and land it as
        canonical SNAPSHOT parquet (via ``write_canonical_table``) plus a
        ``ProviderDataset`` catalog row per market. Returns a summary dict
        (counts / written paths). Implementations are idempotent over a window.
        """
        ...


__all__ = ["MarketDataProvider"]
