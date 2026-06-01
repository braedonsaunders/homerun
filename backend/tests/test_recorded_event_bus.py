"""End-to-end + unit tests for the recorded-event-bus architecture.

Covers:
  * Envelope: validation, bitemporal leakage guard, ordering, round-trip.
  * Catalog: register/get/list/delete, fail-closed on unknown topics,
    seed idempotency.
  * Bus: live publish/subscribe, replay across topics with time merge,
    entity filter, dedup of duplicate handlers.
  * Storage: parquet round-trip across multiple entities (proves the
    heap-merge fix from Batch B), SQL adapter wraps an existing table.
  * Bridge: legacy ``subscriptions`` resolution to bus topics, topic
    registration gates replay.
  * Cutover: faithful crypto_update event → discovery binner → strategy
    detect_async path (the un-backtestable-strategy gap is closed).

Tests are async + use the real DB (the catalog row table is small +
isolated; tests delete their topics on teardown).  Parquet I/O uses
tempfile.TemporaryDirectory.
"""
from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from services.recorded_event_bus import (
    RecordedEvent, EnvelopeValidationError, parse_topic,
    RecordedEventBus, ReplayWindow,
    register_topic, get_topic, list_topics, delete_topic,
    TopicNotRegisteredError,
)
from services.recorded_event_bus.catalog import (
    require_topic, ensure_seed_topics,
)


# ── Envelope ────────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def _ensure_topic_catalog_schema():
    from models.database import TopicCatalog, async_engine

    async_engine.sync_engine.dispose(close=False)
    async with async_engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: TopicCatalog.__table__.create(sync_conn, checkfirst=True))


class TestEnvelope:
    def test_topic_validation_accepts_well_formed(self):
        assert parse_topic("polymarket.book.snapshot") == ("polymarket", "book", "snapshot")
        assert parse_topic("crypto.update.btc_eth") == ("crypto", "update", "btc_eth")

    def test_topic_validation_rejects_malformed(self):
        for bad in ("NoDots", "polymarket-book", "polymarket.book.", "..foo", "a.b.c.d.e.f.g"):
            with pytest.raises(EnvelopeValidationError):
                parse_topic(bad)

    def test_envelope_round_trip(self):
        e = RecordedEvent(
            topic="polymarket.book.snapshot",
            entity_id="tok_xyz",
            observed_at_us=1_700_000_000_000_000,
            payload={"best_bid": 0.5},
            source="ut",
            sequence=42,
        )
        d = e.to_dict()
        e2 = RecordedEvent.from_dict(d)
        assert e2 == e

    def test_envelope_rejects_backdated_ingest(self):
        # ingested_at_us more than 1s before observed_at_us → leakage hazard
        with pytest.raises(EnvelopeValidationError):
            RecordedEvent(
                topic="a.b.c",
                entity_id="x",
                observed_at_us=2_000_000_000_000_000,
                ingested_at_us=1_700_000_000_000_000,
                payload={},
            )

    def test_envelope_ordering_total(self):
        e1 = RecordedEvent(topic="a.b", entity_id="1", observed_at_us=100, payload={}, sequence=1)
        e2 = RecordedEvent(topic="a.b", entity_id="1", observed_at_us=100, payload={}, sequence=2)
        assert e1.order_key() < e2.order_key()


# ── Catalog ─────────────────────────────────────────────────────────


class TestCatalog:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_seed_topics_are_registered(self):
        await ensure_seed_topics()
        slugs = {t.slug for t in await list_topics()}
        for required in (
            "polymarket.book.snapshot", "polymarket.book.delta",
            "wallet.trade", "opportunity.detected",
        ):
            assert required in slugs

    @pytest.mark.asyncio(loop_scope="module")
    async def test_require_unknown_topic_raises(self):
        with pytest.raises(TopicNotRegisteredError):
            await require_topic("nonsense.fake.topic")

    @pytest.mark.asyncio(loop_scope="module")
    async def test_register_then_delete_round_trip(self):
        slug = "test.envelope.unit"
        try:
            spec = await register_topic(
                slug=slug,
                title="unit test topic",
                storage_kind="memory",
                is_replayable=False,
            )
            assert spec.slug == slug
            assert spec.storage_kind == "memory"
            got = await get_topic(slug)
            assert got is not None
            assert got.title == "unit test topic"
        finally:
            await delete_topic(slug)
        assert await get_topic(slug) is None

    @pytest.mark.asyncio(loop_scope="module")
    async def test_upsert_merges_publishers(self):
        slug = "test.upsert.merge"
        try:
            await register_topic(slug=slug, title="t1", storage_kind="memory",
                                 publishers=["a"], is_replayable=False)
            await register_topic(slug=slug, title="t1", storage_kind="memory",
                                 publishers=["b"], is_replayable=False)
            spec = await get_topic(slug)
            assert set(spec.publishers) == {"a", "b"}
        finally:
            await delete_topic(slug)


# ── Bus + parquet storage round-trip ────────────────────────────────


class TestBusStorageRoundTrip:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_parquet_round_trip_across_entities(self):
        """The fix from Batch B — heap-merge per-partition streams so
        cross-entity replay stays time-ordered."""
        from services.recorded_event_bus import bus
        from services.recorded_event_bus.storage import flush_pending_writes
        import services.recorded_event_bus.storage  # noqa — attach
        slug = "test.parquet.roundtrip"
        with tempfile.TemporaryDirectory() as td:
            uri = str(Path(td) / "topic")
            try:
                await register_topic(
                    slug=slug, title="rt", storage_kind="parquet", storage_uri=uri,
                )
                T0 = int(datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc).timestamp() * 1e6)
                # Interleave entities — alpha at 0,2,4...; beta at 1,3,5
                for i in range(10):
                    await bus.publish(RecordedEvent(
                        topic=slug, entity_id="alpha",
                        observed_at_us=T0 + i * 2000,
                        payload={"i": i},
                    ))
                    await bus.publish(RecordedEvent(
                        topic=slug, entity_id="beta",
                        observed_at_us=T0 + i * 2000 + 1000,
                        payload={"i": i},
                    ))
                await flush_pending_writes()

                got = []
                async for ev in bus.replay(ReplayWindow(
                    start_us=T0 - 1, end_us=T0 + 1_000_000, topics=(slug,),
                )):
                    got.append(ev)
                assert len(got) == 20
                # Must be globally time-ordered, not entity-grouped
                for a, b in zip(got, got[1:]):
                    assert a.observed_at_us <= b.observed_at_us
            finally:
                await delete_topic(slug)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_entity_filter(self):
        from services.recorded_event_bus import bus
        from services.recorded_event_bus.storage import flush_pending_writes
        import services.recorded_event_bus.storage  # noqa
        slug = "test.entity.filter"
        with tempfile.TemporaryDirectory() as td:
            uri = str(Path(td) / "topic")
            try:
                await register_topic(slug=slug, title="ef", storage_kind="parquet", storage_uri=uri)
                T0 = int(datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc).timestamp() * 1e6)
                for ent in ("a", "b", "c"):
                    for i in range(3):
                        await bus.publish(RecordedEvent(
                            topic=slug, entity_id=ent,
                            observed_at_us=T0 + i * 1000,
                            payload={},
                        ))
                await flush_pending_writes()
                # Filter to b only.
                got = []
                async for ev in bus.replay(ReplayWindow(
                    start_us=T0 - 1, end_us=T0 + 1_000_000, topics=(slug,),
                    entity_filter={slug: frozenset(["b"])},
                )):
                    got.append(ev)
                assert len(got) == 3
                assert all(e.entity_id == "b" for e in got)
            finally:
                await delete_topic(slug)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_replay_prunes_out_of_window_files_without_losing_rows(self):
        """Speed optimization guard: replay skips whole files provably outside the
        window (row-group time stats) but must still return EXACTLY the in-window
        events — over-pruning would silently drop rows."""
        from services.recorded_event_bus import bus
        from services.recorded_event_bus.storage import flush_pending_writes
        import services.recorded_event_bus.storage  # noqa: F401
        slug = "test.prune.window"
        with tempfile.TemporaryDirectory() as td:
            uri = str(Path(td) / "topic")
            try:
                await register_topic(slug=slug, title="pr", storage_kind="parquet", storage_uri=uri)
                base = int(datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc).timestamp() * 1e6)
                hour_us = 3_600_000_000
                # 3 batches an hour apart, each flushed to its own file.
                for h in range(3):
                    await bus.publish(RecordedEvent(
                        topic=slug, entity_id="e",
                        observed_at_us=base + h * hour_us, payload={"h": h},
                    ))
                    await flush_pending_writes()
                # Window covering ONLY the middle event — files 0 and 2 are
                # provably outside and must be pruned, file 1 kept.
                mid = base + hour_us
                got = [ev async for ev in bus.replay(ReplayWindow(
                    start_us=mid - 1, end_us=mid + 1, topics=(slug,),
                ))]
                assert len(got) == 1, f"pruning lost/duplicated rows: {len(got)}"
                assert got[0].payload["h"] == 1
            finally:
                await delete_topic(slug)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_flush_loop_drains_partials_on_cadence(self, monkeypatch):
        """Regression: the background flush loop must periodically drain PARTIAL
        (sub-_FLUSH_BATCH_SIZE) buckets.  Otherwise low-volume topics (e.g.
        polymarket.catalog.snapshot, ~1 event/scan) never reach a 500-event
        batch and only ever flush on graceful shutdown — silent data loss + an
        empty parquet stream for backtest replay (the bug this guards)."""
        from services.recorded_event_bus.storage import parquet_backend as pb

        # (a) drain contract — a single-event (partial) bucket is SKIPPED by the
        # full-batch drain but CAUGHT by drain_all (the partial-cadence path).
        ring = pb._WriteRing()
        ring.offer(
            RecordedEvent(topic="t.lv", entity_id="e", observed_at_us=1, payload={}),
            storage_uri="/tmp/x",
        )
        assert ring.drain_full_batches() == [], "partial wrongly flushed by full-batch drain"
        assert len(ring.drain_all()) == 1, "drain_all must flush partial buckets"

        # (b) loop wiring — drain_all is invoked on a fixed cadence, never "never".
        calls: list[bool] = []

        class _Stop(Exception):
            pass

        async def _fake_flush_once(*, drain_all=False):
            calls.append(drain_all)
            if len(calls) >= 11:
                raise _Stop()
            return 0

        monkeypatch.setattr(pb, "_flush_once", _fake_flush_once)
        monkeypatch.setattr(pb, "_FLUSH_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(pb, "_PARTIAL_FLUSH_INTERVAL_SECONDS", 0.05)  # => every 5th tick
        try:
            await pb._flush_loop()
        except _Stop:
            pass
        partial_ticks = [i for i, d in enumerate(calls, start=1) if d]
        assert partial_ticks, "flush loop never drains partials (low-volume topics would never flush)"
        assert partial_ticks == [5, 10], f"unexpected partial-drain cadence: {partial_ticks}"

    @pytest.mark.asyncio(loop_scope="module")
    async def test_catalog_snapshot_records_prices_arg(self, monkeypatch):
        """Regression: the catalog-snapshot publisher must record the per-token
        ``prices`` arg detect() received.  scanner_tick strategies (e.g.
        tail_end_carry) read the book from ``prices``, NOT market.best_bid/ask,
        so without it their backtest replay sees an empty book and never fires
        — the catalog-snapshot stream must carry (markets, events, prices)."""
        from services import shared_state
        import services.recording_control as rc
        import services.recorded_event_bus as reb

        captured: dict = {}

        class _FakeBus:
            async def publish(self, envelope):
                captured["env"] = envelope

        async def _yes():
            return True

        async def _noop_register():
            return None

        monkeypatch.setattr(rc, "is_recording_enabled", _yes)
        monkeypatch.setattr(reb, "bus", _FakeBus())
        monkeypatch.setattr(shared_state, "_ensure_catalog_snapshot_topic_registered", _noop_register)

        await shared_state._publish_catalog_snapshot_to_bus(
            events_payload=[],
            markets_payload=[{"id": "m1", "clob_token_ids": ["tok"]}],
            prices_payload={"tok": {"bid": 0.42, "ask": 0.55}},
            updated_at=datetime.now(timezone.utc),
            duration_seconds=0.0,
            error=None,
        )
        env = captured.get("env")
        assert env is not None, "publisher did not publish an envelope"
        assert env.payload.get("prices") == {"tok": {"bid": 0.42, "ask": 0.55}}, \
            "catalog.snapshot must carry the prices arg for scanner_tick replay"

    @pytest.mark.asyncio(loop_scope="module")
    async def test_live_subscribe_dedups_handlers(self):
        b = RecordedEventBus()  # isolated instance
        seen = []
        async def h(ev): seen.append(ev)
        b.subscribe("a.b.c", h)
        b.subscribe("a.b.c", h)  # duplicate
        # Need to register the topic so publish doesn't fail-closed.
        slug = "test.live.dedup"
        try:
            await register_topic(slug=slug, title="ld", storage_kind="memory", is_replayable=False)
            sub = b.subscribe(slug, h)
            sub2 = b.subscribe(slug, h)
            assert sub is sub2  # idempotent
            await b.publish(RecordedEvent(
                topic=slug, entity_id="x", observed_at_us=int(datetime.now(timezone.utc).timestamp()*1e6),
                payload={},
            ))
            await asyncio.sleep(0.05)
            assert len(seen) == 1  # not 2
        finally:
            await delete_topic(slug)


# ── Bridge / cutover ────────────────────────────────────────────────


class TestBridge:
    def test_resolve_subscriptions_legacy_to_bus(self):
        from services.recorded_event_bus.backtest_bridge import resolve_subscriptions_to_topics
        assert resolve_subscriptions_to_topics(["crypto_update"]) == ("crypto.update.dispatch",)
        assert resolve_subscriptions_to_topics(["wallet_trade"]) == ("wallet.trade",)
        # Unknown legacy → empty (filtered out)
        assert resolve_subscriptions_to_topics(["foo_bar"]) == ()
        # Already-dotted topic passes through
        assert resolve_subscriptions_to_topics(["polymarket.book.snapshot"]) == (
            "polymarket.book.snapshot",
        )
        # Mixed + dedup
        out = resolve_subscriptions_to_topics(
            ["crypto_update", "crypto.update.dispatch", "wallet_trade"]
        )
        assert out == ("crypto.update.dispatch", "wallet.trade")

    @pytest.mark.asyncio(loop_scope="module")
    async def test_bridge_skips_unregistered_topic(self):
        from services.recorded_event_bus.backtest_bridge import replay_events_for_strategy
        class Fake:
            subscriptions = ["nonexistent.topic.xyz"]
        events = []
        async for ev in replay_events_for_strategy(
            strategy=Fake(),
            start_dt=datetime.now(timezone.utc) - timedelta(hours=1),
            end_dt=datetime.now(timezone.utc),
        ):
            events.append(ev)
        assert events == []  # not crash; not yield

    @pytest.mark.asyncio(loop_scope="module")
    async def test_crypto_update_tap_round_trip(self):
        """End-to-end cutover: publish a CRYPTO_UPDATE through the live
        market_runtime path, verify the bridge yields the same event
        for backtest replay."""
        from services.market_runtime import (
            _ensure_crypto_update_topic_registered,
            _publish_crypto_update_to_bus,
        )
        from services.data_events import DataEvent, EventType
        from services.recorded_event_bus.backtest_bridge import replay_events_for_strategy
        from services.recorded_event_bus.storage import flush_pending_writes
        import services.recorded_event_bus.storage  # noqa

        import shutil
        import uuid

        from services.external_data.parquet_schema import parquet_root

        run_id = uuid.uuid4().hex[:8]
        await _ensure_crypto_update_topic_registered()
        BASE = datetime.now(timezone.utc).replace(microsecond=0)
        try:
            for i in range(3):
                ts = BASE - timedelta(minutes=10) + timedelta(seconds=i * 30)
                evt = DataEvent(
                    event_type=EventType.CRYPTO_UPDATE,
                    source="market_runtime",
                    timestamp=ts,
                    payload={},
                )
                await _publish_crypto_update_to_bus(
                    evt,
                    copied_for_event=[{"market_id": f"m_{i}", "mid": 0.5}],
                    trigger=f"unit_test_tick_{run_id}_{i}",
                )
            await flush_pending_writes()

            class FakeCryptoStrategy:
                subscriptions = ["crypto_update"]

            got = []
            async for ev in replay_events_for_strategy(
                strategy=FakeCryptoStrategy(),
                start_dt=BASE - timedelta(minutes=15),
                end_dt=BASE,
            ):
                if ev.entity_id.startswith(f"unit_test_tick_{run_id}_"):
                    got.append(ev)
            assert len(got) == 3
        finally:
            # Don't pollute the real bus data plane with test entity dirs.
            topic_dir = parquet_root() / "recorded_event_bus" / "crypto.update.dispatch"
            for i in range(3):
                shutil.rmtree(topic_dir / f"unit_test_tick_{run_id}_{i}", ignore_errors=True)
