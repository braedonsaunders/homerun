"""External market-data provider clients.

Each provider exposes:
  * ``client`` — async HTTP client wrapping the provider's REST API
  * a small mapping of provider-specific market metadata onto our
    canonical ``SNAPSHOT_SCHEMA`` (parquet) book layout

Providers integrated to date:
  * ``polybacktest`` — paid SaaS for Polymarket Up/Down book history +
    Binance reference prices.  See polybacktest_client.py.

Adding a new provider is a matter of writing a sibling module that
exports an async client + a registration helper that the import worker
can dispatch to via the ``provider`` field on ``ProviderImportJob``.
"""

from __future__ import annotations

__all__ = [
    "polybacktest_client",
]
