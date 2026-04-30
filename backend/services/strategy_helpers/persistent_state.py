"""Durable per-strategy key/value state for the Strategy SDK.

``BaseStrategy.state`` is an in-memory dict — useful for rolling
counters within a single worker process, but everything is lost on
restart. ``PersistentState`` is the durable counterpart: a thin
wrapper over the ``strategy_persistent_state`` table that strategies
opt into when they need state to survive across restarts.

Design notes
------------
The SDK is consumed by both sync and async strategies. To keep reads
fast and predictable, ``get`` / ``set`` / ``delete`` operate on an
in-memory cache, and persistence is explicit:

* ``await state.load()``  - hydrate the cache from the DB (call once,
  typically in your first ``detect_async`` invocation or on a
  ``start()`` hook).
* ``await state.flush()`` - write any cache entries that have been
  modified since the last flush back to the DB. Idempotent and a
  no-op when nothing is dirty.

This split avoids hidden DB round-trips inside ``detect()`` and lets
strategies decide when to take the durability cost. Loss model: any
``set()`` calls that haven't been flushed are lost on a hard crash.
That matches the contract of ``BaseStrategy.state`` (also volatile),
just with an explicit checkpoint when you want durability.
"""

from __future__ import annotations

import copy
from typing import Any, Optional


_MISSING = object()


class PersistentState:
    """Per-strategy key/value cache backed by ``strategy_persistent_state``.

    Usage::

        from services.strategy_sdk import StrategySDK

        class MyStrategy(BaseStrategy):
            async def detect_async(self, events, markets, prices):
                state = self._state.setdefault(
                    "_persistent",
                    StrategySDK.PersistentState(self.strategy_type),
                )
                if not state.loaded:
                    await state.load()

                last_seen_ts = state.get("last_seen_ts", default=0)
                state.set("last_seen_ts", time.time())

                if state.dirty:
                    await state.flush()

                return []

    The cache stores the last-seen value per key. ``set`` records the
    new value and marks the key dirty; ``flush`` writes only dirty keys
    so frequent reads stay cheap.
    """

    __slots__ = ("_strategy_slug", "_cache", "_dirty", "_loaded", "_session_factory")

    def __init__(
        self,
        strategy_slug: str,
        *,
        session_factory: Optional[Any] = None,
    ) -> None:
        slug = str(strategy_slug or "").strip().lower()
        if not slug:
            raise ValueError("PersistentState requires a non-empty strategy_slug")
        self._strategy_slug: str = slug
        self._cache: dict[str, Any] = {}
        self._dirty: set[str] = set()
        # Tombstones for keys deleted locally that haven't yet been
        # flushed. Tracked in ``_dirty`` for write batching but kept
        # out of ``_cache`` so ``get()`` returns the default.
        self._loaded: bool = False
        # Tests inject a session factory; production paths default to
        # the global ``AsyncSessionLocal`` (resolved lazily so this
        # module imports cleanly in environments where the DB engine
        # hasn't been initialised yet).
        self._session_factory = session_factory

    # ── Properties ─────────────────────────────────────────────

    @property
    def strategy_slug(self) -> str:
        return self._strategy_slug

    @property
    def loaded(self) -> bool:
        """True after ``load()`` has hydrated the cache at least once."""
        return self._loaded

    @property
    def dirty(self) -> bool:
        """True when there are unflushed local writes/deletes."""
        return bool(self._dirty)

    # ── Synchronous cache access ───────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value for ``key``, or ``default`` if absent.

        Returns a deep copy so mutations on the returned object don't
        silently bypass ``set()``.
        """
        value = self._cache.get(key, _MISSING)
        if value is _MISSING:
            return default
        return copy.deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        """Update the cache and mark ``key`` for flush.

        ``value`` must be JSON-serialisable (the underlying column is
        JSON); we don't pre-validate here so callers see the
        serialisation error at flush time rather than burying it in a
        wrapper.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("PersistentState key must be a non-empty string")
        self._cache[key] = copy.deepcopy(value)
        self._dirty.add(key)

    def delete(self, key: str) -> None:
        """Remove ``key`` from the cache and mark it for deletion on flush."""
        if not isinstance(key, str) or not key:
            raise ValueError("PersistentState key must be a non-empty string")
        self._cache.pop(key, None)
        self._dirty.add(key)

    def keys(self) -> list[str]:
        return list(self._cache.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    # ── Async DB roundtrips ────────────────────────────────────

    async def load(self) -> None:
        """Hydrate the cache from the DB.

        Replaces the existing cache (any unflushed local writes are
        dropped). Strategies typically call this once after instantiation;
        re-loading mid-cycle is allowed but uncommon.
        """
        StrategyPersistentState, _ = self._import_models()
        from sqlalchemy import select

        session_factory = self._resolve_session_factory()
        async with session_factory() as session:
            stmt = select(StrategyPersistentState).where(
                StrategyPersistentState.strategy_slug == self._strategy_slug
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        new_cache: dict[str, Any] = {}
        for row in rows:
            new_cache[str(row.key)] = copy.deepcopy(row.value)
        self._cache = new_cache
        self._dirty.clear()
        self._loaded = True

    async def flush(self) -> None:
        """Persist dirty entries to the DB. No-op when ``dirty`` is False."""
        if not self._dirty:
            return
        StrategyPersistentState, _utcnow = self._import_models()
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy import delete

        to_upsert: list[dict[str, Any]] = []
        to_delete: list[str] = []
        for key in self._dirty:
            if key in self._cache:
                to_upsert.append(
                    {
                        "strategy_slug": self._strategy_slug,
                        "key": key,
                        "value": self._cache[key],
                        "updated_at": _utcnow(),
                    }
                )
            else:
                to_delete.append(key)

        session_factory = self._resolve_session_factory()
        async with session_factory() as session:
            if to_upsert:
                stmt = pg_insert(StrategyPersistentState.__table__).values(to_upsert)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["strategy_slug", "key"],
                    set_={
                        "value": stmt.excluded.value,
                        "updated_at": stmt.excluded.updated_at,
                    },
                )
                await session.execute(stmt)
            if to_delete:
                await session.execute(
                    delete(StrategyPersistentState).where(
                        StrategyPersistentState.strategy_slug == self._strategy_slug,
                        StrategyPersistentState.key.in_(to_delete),
                    )
                )
            await session.commit()
        self._dirty.clear()

    # ── Internals ──────────────────────────────────────────────

    def _resolve_session_factory(self):
        if self._session_factory is not None:
            return self._session_factory
        from models.database import AsyncSessionLocal

        return AsyncSessionLocal

    @staticmethod
    def _import_models():
        from models.database import StrategyPersistentState as Model
        from utils.utcnow import utcnow

        return Model, utcnow
