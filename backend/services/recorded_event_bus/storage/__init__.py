"""Storage backends for the recorded-event bus.

Two backings supported in v1:

  * :mod:`.parquet_backend` — topic-major partitioned parquet on disk.
    The default for new topics.  Writes go through a batched flush
    queue so the publish hot path enqueues in microseconds; the
    background flush amortises the parquet write cost across hundreds
    of events.

  * :mod:`.sql_adapters` — read-only adapters around the remaining
    audit SQL tables (wallet_monitor_events, opportunity_history).
    These topics aren't written *through* the bus — the existing
    recorders own the writes, which is why the approach is "tap the
    recorders to also publish to the bus, leave the storage path
    alone."  The adapters let the bus *read* those tables as if they
    were bus-native topics so backtest replay sees a unified interface
    regardless of which side the data was written from.  (Book snapshot
    and delta topics moved fully to the canonical parquet plane in the
    market-data clean cut — they're served by :mod:`.external_parquet`,
    not SQL.)

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
