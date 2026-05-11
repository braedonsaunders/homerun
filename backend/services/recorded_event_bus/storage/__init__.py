"""Storage backends for the recorded-event bus.

Two backings supported in v1:

  * :mod:`.parquet_backend` — topic-major partitioned parquet on disk.
    The default for new topics.  Writes go through a batched flush
    queue so the publish hot path enqueues in microseconds; the
    background flush amortises the parquet write cost across hundreds
    of events.

  * :mod:`.sql_adapters` — read-only adapters around the pre-existing
    SQL tables (market_microstructure_snapshots, book_delta_events,
    wallet_monitor_events, opportunity_history).  These topics aren't
    written *through* the bus — the existing recorders own the writes,
    which is why the migration plan is "tap the recorders to also
    publish to the bus, leave the storage path alone."  The adapters
    let the bus *read* those tables as if they were bus-native topics
    so backtest replay sees a unified interface regardless of which
    side the data was written from.

Importing this module attaches the writer + replayer to the singleton
``bus`` — this is the wire-up that makes ``bus.publish`` actually
persist and ``bus.replay`` actually yield.
"""
from __future__ import annotations

from services.recorded_event_bus.bus import bus
from services.recorded_event_bus.storage.parquet_backend import (
    parquet_writer,
    flush_pending_writes,
    shutdown_parquet_backend,
)
from services.recorded_event_bus.storage.replayer import (
    storage_replayer,
)

# Attach.  Doing this at import time means any module that imports
# ``services.recorded_event_bus.storage`` lights up the data plane.
# The bus stays import-cheap (no pyarrow / no SQLAlchemy adapters) for
# strategies that only need to construct envelopes / subscribe.
bus._attach_storage(
    writer=parquet_writer,
    replayer=storage_replayer,
)


__all__ = [
    "flush_pending_writes",
    "shutdown_parquet_backend",
]
