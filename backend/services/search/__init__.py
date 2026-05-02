"""Global search subsystem.

Provides the ``search_index`` table (one row per searchable entity in
the entire system), the periodic worker that keeps it fresh, the
ranked query engine, and the telemetry log.

Public surface:

* :data:`SearchIndex` / :data:`SearchQueryLog` — SQLAlchemy models
* :func:`run_query` — execute a ranked search (used by ``/search/global``)
* :func:`reindex_all` — full sweep of every entity-type collector
* :func:`reindex_one` — sweep for a single entity type
* :data:`COLLECTORS` — mapping of ``entity_type`` -> async collector

The collectors live in :mod:`services.search.collectors`; the
ranking + headline query lives in :mod:`services.search.service`;
the periodic loop lives in :mod:`services.search.worker`.
"""

from services.search.models import SearchIndex, SearchQueryLog
from services.search.service import run_query, log_query
from services.search.collectors import COLLECTORS, reindex_all, reindex_one

__all__ = [
    "SearchIndex",
    "SearchQueryLog",
    "run_query",
    "log_query",
    "COLLECTORS",
    "reindex_all",
    "reindex_one",
]
