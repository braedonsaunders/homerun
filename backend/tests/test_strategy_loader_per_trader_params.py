"""Plan 0041: per-trader strategy params actually affect signal generation.

These tests pin the shared infrastructure plan 0041 introduced:

- ``BaseStrategy.clone_for_trader`` — produces a per-trader copy with
  isolated runtime state and merged config.
- ``StrategyLoader.get_or_clone_for_trader`` — lazy cache keyed by
  ``(slug, trader_id)``, rebuild on global reload, drop on
  ``invalidate_per_trader``.
- ``StrategyLoader.invalidate_per_trader`` — selective + bulk drops.
- The dispatcher fan-out machinery in ``market_runtime`` tags
  opportunities with ``intended_trader_id`` only when a per-trader
  clone produced them.
- ``intent_runtime.list_unconsumed_signals`` filters by trader id when
  the snapshot is per-trader-scoped, and ignores the filter when the
  scope is ``None`` (legacy multi-trader-visible routing).

The strategy used here is an inline dummy (see ``_ThresholdStrategy``)
because per project policy tests must never bind to a concrete
deployed slug — strategies are DB-driven, user-definable, and can be
absent in any given deployment.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from models.opportunity import Opportunity
from services.data_events import DataEvent, EventType
from services.intent_runtime import IntentRuntime
from services.strategies.base import BaseStrategy
from services.strategy_loader import LoadedStrategy, strategy_loader


# ---------------------------------------------------------------------------
# Inline test strategy fixture — emits an Opportunity per market whose
# ``value`` field exceeds ``self.config["threshold"]``.
# ---------------------------------------------------------------------------


class _ThresholdStrategy(BaseStrategy):
    name = "Threshold Test"
    description = "Plan 0041 inline test fixture"
    source_key = "crypto"
    subscriptions = ["crypto_update"]
    default_config = {"threshold": 100.0}

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        threshold = float(self.config.get("threshold", 100.0))
        markets = (event.payload or {}).get("markets") or []
        out: list[Opportunity] = []
        for market in markets:
            try:
                value = float(market.get("value", 0.0))
            except (TypeError, ValueError):
                continue
            if value <= threshold:
                continue
            out.append(
                Opportunity(
                    strategy="threshold_test_strategy",
                    title=f"value {value} > {threshold}",
                    description="plan 0041 fixture",
                    total_cost=value,
                    gross_profit=1.0,
                    fee=0.0,
                    net_profit=1.0,
                    roi_percent=1.0,
                    markets=[market],
                )
            )
        return out


def _build_loaded_strategy(
    slug: str = "threshold_test_strategy",
    *,
    global_threshold: float = 100.0,
) -> LoadedStrategy:
    instance = _ThresholdStrategy()
    instance.key = slug
    instance.configure({"threshold": global_threshold})
    return LoadedStrategy(
        slug=slug,
        instance=instance,
        class_name="_ThresholdStrategy",
        source_hash="testhash",
        loaded_at=datetime.now(timezone.utc),
        module_name="<plan-0041-inline>",
    )


@pytest.fixture(autouse=True)
def _isolate_loader_state():
    saved_loaded = dict(strategy_loader._loaded)
    saved_per_trader = dict(strategy_loader._per_trader)
    saved_errors = dict(strategy_loader._errors)
    yield
    strategy_loader._loaded = saved_loaded
    strategy_loader._per_trader = saved_per_trader
    strategy_loader._errors = saved_errors


# ---------------------------------------------------------------------------
# Loader: per-trader cache mechanics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_or_clone_for_trader_returns_none_when_global_missing():
    assert strategy_loader.get_or_clone_for_trader(
        "nonexistent_slug", "trader_x", {}
    ) is None


@pytest.mark.unit
def test_get_or_clone_for_trader_returns_none_for_blank_trader_id():
    slug = "threshold_test_strategy"
    strategy_loader._loaded[slug] = _build_loaded_strategy(slug)
    assert strategy_loader.get_or_clone_for_trader(slug, "", {}) is None
    assert strategy_loader.get_or_clone_for_trader(slug, "   ", {}) is None


@pytest.mark.unit
def test_per_trader_clone_isolates_config_and_state():
    """Two traders bound to the same strategy with different
    ``strategy_params`` get independent instances. Modifying one
    trader's ``_state`` does not bleed into the other or into the
    global singleton."""
    slug = "threshold_test_strategy"
    global_loaded = _build_loaded_strategy(slug, global_threshold=100.0)
    strategy_loader._loaded[slug] = global_loaded

    loaded_a = strategy_loader.get_or_clone_for_trader(
        slug, "trader_a", {"threshold": 50.0}
    )
    loaded_b = strategy_loader.get_or_clone_for_trader(
        slug, "trader_b", {"threshold": 150.0}
    )

    assert loaded_a is not None and loaded_b is not None
    instance_a = loaded_a.instance
    instance_b = loaded_b.instance
    assert instance_a is not instance_b
    assert instance_a is not global_loaded.instance
    assert instance_b is not global_loaded.instance
    assert instance_a.config["threshold"] == 50.0
    assert instance_b.config["threshold"] == 150.0
    # Cross-trader state isolation
    instance_a._state["seen"] = "a"
    instance_b._state["seen"] = "b"
    assert instance_a._state["seen"] == "a"
    assert instance_b._state["seen"] == "b"
    assert global_loaded.instance._state.get("seen") is None


@pytest.mark.unit
def test_per_trader_clone_with_empty_params_uses_global_config():
    """A trader whose ``source_configs_json[].strategy_params`` is
    empty falls back to the global strategy's config — does NOT see
    ``default_config`` skipping the global overrides applied at load."""
    slug = "threshold_test_strategy"
    global_loaded = _build_loaded_strategy(slug, global_threshold=250.0)
    strategy_loader._loaded[slug] = global_loaded

    loaded = strategy_loader.get_or_clone_for_trader(slug, "trader_z", {})
    assert loaded is not None
    assert loaded.instance.config["threshold"] == 250.0


@pytest.mark.unit
def test_per_trader_cache_returns_same_instance_on_repeat_access():
    slug = "threshold_test_strategy"
    strategy_loader._loaded[slug] = _build_loaded_strategy(slug)

    first = strategy_loader.get_or_clone_for_trader(slug, "trader_q", {"threshold": 10.0})
    second = strategy_loader.get_or_clone_for_trader(slug, "trader_q", {"threshold": 10.0})
    assert first is second, "cached entry should be returned without rebuild"


@pytest.mark.unit
def test_invalidate_per_trader_drops_specific_trader():
    slug = "threshold_test_strategy"
    strategy_loader._loaded[slug] = _build_loaded_strategy(slug)

    _ = strategy_loader.get_or_clone_for_trader(slug, "trader_a", {"threshold": 50.0})
    _ = strategy_loader.get_or_clone_for_trader(slug, "trader_b", {"threshold": 150.0})
    assert strategy_loader.loaded_per_trader_count() == 2

    removed = strategy_loader.invalidate_per_trader(trader_id="trader_a")
    assert removed == 1
    assert (slug, "trader_a") not in strategy_loader._per_trader
    assert (slug, "trader_b") in strategy_loader._per_trader

    # Re-access trader_a with new config — must rebuild.
    rebuilt = strategy_loader.get_or_clone_for_trader(
        slug, "trader_a", {"threshold": 75.0}
    )
    assert rebuilt is not None
    assert rebuilt.instance.config["threshold"] == 75.0


@pytest.mark.unit
def test_invalidate_per_trader_drops_specific_slug():
    slug_a = "threshold_test_strategy"
    slug_b = "threshold_test_strategy_b"
    strategy_loader._loaded[slug_a] = _build_loaded_strategy(slug_a)
    strategy_loader._loaded[slug_b] = _build_loaded_strategy(slug_b)

    _ = strategy_loader.get_or_clone_for_trader(slug_a, "trader_a", {"threshold": 10.0})
    _ = strategy_loader.get_or_clone_for_trader(slug_b, "trader_a", {"threshold": 20.0})
    assert strategy_loader.loaded_per_trader_count() == 2

    removed = strategy_loader.invalidate_per_trader(slug=slug_a)
    assert removed == 1
    assert (slug_a, "trader_a") not in strategy_loader._per_trader
    assert (slug_b, "trader_a") in strategy_loader._per_trader


@pytest.mark.unit
def test_invalidate_per_trader_bulk_clear():
    slug = "threshold_test_strategy"
    strategy_loader._loaded[slug] = _build_loaded_strategy(slug)

    _ = strategy_loader.get_or_clone_for_trader(slug, "trader_a", {"threshold": 10.0})
    _ = strategy_loader.get_or_clone_for_trader(slug, "trader_b", {"threshold": 20.0})
    assert strategy_loader.loaded_per_trader_count() == 2

    removed = strategy_loader.invalidate_per_trader()
    assert removed == 2
    assert strategy_loader.loaded_per_trader_count() == 0


@pytest.mark.unit
def test_unload_invalidates_per_trader_cache_for_that_slug():
    slug = "threshold_test_strategy"
    strategy_loader._loaded[slug] = _build_loaded_strategy(slug)
    _ = strategy_loader.get_or_clone_for_trader(slug, "trader_a", {"threshold": 10.0})
    assert (slug, "trader_a") in strategy_loader._per_trader

    strategy_loader.unload(slug)
    assert (slug, "trader_a") not in strategy_loader._per_trader


# ---------------------------------------------------------------------------
# on_event semantics — per-trader config controls emission count
# ---------------------------------------------------------------------------


def _crypto_event(values: list[float]) -> DataEvent:
    return DataEvent(
        event_type=EventType.CRYPTO_UPDATE,
        source="plan-0041-test",
        timestamp=datetime.now(timezone.utc),
        payload={
            "markets": [
                {"id": f"market_{i}", "value": v} for i, v in enumerate(values)
            ]
        },
    )


@pytest.mark.unit
def test_per_trader_on_event_emits_independent_opportunity_sets():
    """Trader A with a looser threshold emits more opportunities than
    Trader B with a stricter threshold for the same payload."""
    slug = "threshold_test_strategy"
    strategy_loader._loaded[slug] = _build_loaded_strategy(slug, global_threshold=100.0)

    loaded_a = strategy_loader.get_or_clone_for_trader(
        slug, "trader_a", {"threshold": 50.0}
    )
    loaded_b = strategy_loader.get_or_clone_for_trader(
        slug, "trader_b", {"threshold": 150.0}
    )
    assert loaded_a is not None and loaded_b is not None

    event = _crypto_event([60.0, 110.0, 200.0])

    opps_a = asyncio.run(loaded_a.instance.on_event(event))
    opps_b = asyncio.run(loaded_b.instance.on_event(event))

    # Trader A (threshold 50) accepts all three values.
    assert len(opps_a) == 3
    # Trader B (threshold 150) accepts only the 200 value.
    assert len(opps_b) == 1


# ---------------------------------------------------------------------------
# IntentRuntime list_unconsumed_signals: per-trader scope filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_unconsumed_signals_filters_by_intended_trader_id():
    """When an in-memory snapshot has ``intended_trader_id`` set, only
    the trader whose id matches sees the row. Other traders skip it."""
    runtime = IntentRuntime()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Three synthetic snapshots in the in-memory map:
    # 1. Scoped to trader_a
    # 2. Scoped to trader_b
    # 3. Unscoped (None) — legacy multi-trader-visible row
    base = {
        "source": "crypto",
        "signal_type": "crypto_opportunity",
        "strategy_type": "threshold_test_strategy",
        "market_id": "m1",
        "status": "pending",
        "deferred_until_ws": False,
        "expires_at": None,
        "created_at": now_iso,
        "updated_at": now_iso,
        "required_token_ids": [],
    }
    runtime._signals_by_id["sig_a"] = {
        **base,
        "id": "sig_a",
        "dedupe_key": "sig_a_key",
        "runtime_sequence": 1,
        "intended_trader_id": "trader_a",
    }
    runtime._signals_by_id["sig_b"] = {
        **base,
        "id": "sig_b",
        "dedupe_key": "sig_b_key",
        "runtime_sequence": 2,
        "intended_trader_id": "trader_b",
    }
    runtime._signals_by_id["sig_open"] = {
        **base,
        "id": "sig_open",
        "dedupe_key": "sig_open_key",
        "runtime_sequence": 3,
        "intended_trader_id": None,
    }

    rows_a = await runtime.list_unconsumed_signals(
        trader_id="trader_a", sources=["crypto"], statuses=["pending"]
    )
    rows_b = await runtime.list_unconsumed_signals(
        trader_id="trader_b", sources=["crypto"], statuses=["pending"]
    )
    rows_c = await runtime.list_unconsumed_signals(
        trader_id="trader_c_unbound", sources=["crypto"], statuses=["pending"]
    )

    ids_a = {row.id for row in rows_a}
    ids_b = {row.id for row in rows_b}
    ids_c = {row.id for row in rows_c}

    # Trader A sees its scoped row + the unscoped row, NOT trader B's row.
    assert ids_a == {"sig_a", "sig_open"}
    # Trader B sees its scoped row + the unscoped row, NOT trader A's row.
    assert ids_b == {"sig_b", "sig_open"}
    # A third trader bound to "crypto" (but not the source of any scoped
    # row) sees only the unscoped row.
    assert ids_c == {"sig_open"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_unconsumed_signals_treats_blank_intended_trader_id_as_unscoped():
    """``intended_trader_id`` may arrive as an empty string from the
    JSON-decoded payload. The filter must treat that as unscoped."""
    runtime = IntentRuntime()
    now_iso = datetime.now(timezone.utc).isoformat()
    runtime._signals_by_id["sig"] = {
        "id": "sig",
        "source": "crypto",
        "signal_type": "crypto_opportunity",
        "strategy_type": "threshold_test_strategy",
        "market_id": "m1",
        "status": "pending",
        "deferred_until_ws": False,
        "expires_at": None,
        "created_at": now_iso,
        "updated_at": now_iso,
        "required_token_ids": [],
        "dedupe_key": "sig_key",
        "runtime_sequence": 1,
        "intended_trader_id": "",
    }

    rows = await runtime.list_unconsumed_signals(
        trader_id="any_trader", sources=["crypto"], statuses=["pending"]
    )
    assert {row.id for row in rows} == {"sig"}


# ---------------------------------------------------------------------------
# clone_for_trader contract on BaseStrategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clone_for_trader_default_impl_merges_config():
    """The default ``BaseStrategy.clone_for_trader`` implementation
    merges the global ``self.config`` with the trader override before
    ``configure``."""
    base = _ThresholdStrategy()
    base.configure({"threshold": 100.0, "stable_other_key": "kept"})

    clone = base.clone_for_trader({"threshold": 25.0})
    assert clone is not base
    assert isinstance(clone, _ThresholdStrategy)
    assert clone.config["threshold"] == 25.0
    # Keys from the global config must survive the merge.
    assert clone.config["stable_other_key"] == "kept"


@pytest.mark.unit
def test_clone_for_trader_with_none_returns_clone_with_global_config():
    base = _ThresholdStrategy()
    base.configure({"threshold": 100.0})
    clone = base.clone_for_trader(None)
    assert clone is not base
    assert clone.config["threshold"] == 100.0
