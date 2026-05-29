"""The single ordered k-way merge.

The codebase grew four near-identical heap-merge implementations that merge
already-sorted per-source streams into one global-ascending stream:
``HybridBookSource.iter_snapshots``, ``bus.replay``, ``multi_source_replayer``,
and ``external_parquet_replayer``. They differ only in sync-vs-async and the
key function.

This module provides the two canonical merges — synchronous and asynchronous —
keyed by an integer (microsecond timestamp, in practice). Both are *stable*:
when entries from different sources share a key, earlier sources in the
argument order win, giving deterministic ordering (important for reproducible
replays).
"""
from __future__ import annotations

import heapq
from typing import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    TypeVar,
)

T = TypeVar("T")


def ordered_merge(
    sources: Iterable[Iterable[T]],
    *,
    key: Callable[[T], int],
) -> Iterator[T]:
    """Merge already-ascending iterables into one ascending stream by ``key``.

    Thin, stable wrapper over :func:`heapq.merge`. Each input MUST already be
    sorted ascending by ``key``; the merge does not re-sort. Ties are broken by
    source order (first source wins), which ``heapq.merge`` guarantees.
    """
    return heapq.merge(*sources, key=key)


async def aordered_merge(
    sources: Iterable[AsyncIterable[T]],
    *,
    key: Callable[[T], int],
) -> AsyncIterator[T]:
    """Async k-way merge of already-ascending async iterables by ``key``.

    Primes one item from each source, then repeatedly yields the smallest by
    ``key`` and advances that source. Stable: ties break by source order via
    the heap's secondary (source-index) key. Each input MUST be ascending.
    """
    iterators: list[AsyncIterator[T]] = [s.__aiter__() for s in sources]
    # Heap entries: (key, source_index, value, iterator). source_index keeps
    # the sort stable and avoids comparing values when keys tie.
    heap: list[tuple[int, int, T, AsyncIterator[T]]] = []
    for idx, it in enumerate(iterators):
        try:
            first = await it.__anext__()
        except StopAsyncIteration:
            continue
        heap.append((key(first), idx, first, it))
    heapq.heapify(heap)

    while heap:
        k, idx, value, it = heapq.heappop(heap)
        yield value
        try:
            nxt = await it.__anext__()
        except StopAsyncIteration:
            continue
        heapq.heappush(heap, (key(nxt), idx, nxt, it))


__all__ = ["ordered_merge", "aordered_merge"]
