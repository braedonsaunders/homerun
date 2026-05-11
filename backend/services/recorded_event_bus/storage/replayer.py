"""Unified replayer: every topic flows through the multi-source
replayer.

There used to be a kind-by-kind dispatch here.  That was the wrong
factoring — single-source vs. multi-source ended up as separate
code paths even though the only difference is sources.length.

The new design treats EVERY topic as a list of sources (1..N).
Single-source topics get a fast-path passthrough; multi-source
topics get heap-merged.  The kind-specific readers (parquet, sql,
external_parquet) are now invoked from inside multi_source for
each declared source.

The ``storage_kind`` column on the topic_catalog is preserved as
the operator-facing badge ("primarily sql / parquet / federated")
but is no longer the dispatch key — sources resolved from
``storage_uri`` are.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from services.recorded_event_bus.catalog import TopicSpec
from services.recorded_event_bus.envelope import RecordedEvent
from services.recorded_event_bus.storage.multi_source import multi_source_replayer

logger = logging.getLogger(__name__)


async def storage_replayer(
    spec: TopicSpec,
    window,
) -> AsyncIterator[RecordedEvent]:
    """One entry point for every topic.  Resolves the topic's
    sources from ``storage_uri`` and replays them — single-source
    or multi-source uniformly.  ``memory`` topics yield nothing
    by definition (no replayable backing).
    """
    if spec.storage_kind == "memory":
        return
    async for ev in multi_source_replayer(spec, window):
        yield ev
