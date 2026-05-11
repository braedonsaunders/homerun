"""Plan 0041: ``market_runtime`` per-trader dispatch fan-out.

Exercises ``_dispatch_with_per_trader_fanout`` end-to-end with a
mocked binding cache + event dispatcher. The strategy used here is
the same inline ``_ThresholdStrategy`` as
``test_strategy_loader_per_trader_params`` — per project policy tests
must not bind to concrete deployed slugs.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.opportunity import Opportunity
from services import market_runtime
from services.data_events import DataEvent, EventType
from services.strategies.base import BaseStrategy
from services.strategy_loader import LoadedStrategy, strategy_loader


# Reuse the inline test strategy contract from the loader test —
# duplicated here to keep the two test files independent.
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


SLUG = "threshold_test_strategy"


@pytest.fixture(autouse=True)
def _isolate_loader_state():
    saved_loaded = dict(strategy_loader._loaded)
    saved_per_trader = dict(strategy_loader._per_trader)
    saved_errors = dict(strategy_loader._errors)
    yield
    strategy_loader._loaded = saved_loaded
    strategy_loader._per_trader = saved_per_trader
    strategy_loader._errors = saved_errors


def _seed_global_strategy(global_threshold: float = 100.0) -> LoadedStrategy:
    instance = _ThresholdStrategy()
    instance.key = SLUG
    instance.configure({"threshold": global_threshold})
    loaded = LoadedStrategy(
        slug=SLUG,
        instance=instance,
        class_name="_ThresholdStrategy",
        source_hash="testhash",
        loaded_at=datetime.now(timezone.utc),
        module_name="<plan-0041-inline>",
    )
    strategy_loader._loaded[SLUG] = loaded
    return loaded


@pytest.fixture
def _patched_dispatcher_subscriptions(monkeypatch):
    """Make the event dispatcher report our inline strategy as subscribed
    to CRYPTO_UPDATE so the fan-out walks the right slug set."""
    monkeypatch.setattr(
        market_runtime.event_dispatcher,
        "subscribed_slugs",
        lambda event_type: {SLUG} if event_type == EventType.CRYPTO_UPDATE else set(),
    )
    # When no per-trader binding is found, the fan-out routes through
    # event_dispatcher.dispatch(..., include_strategies={SLUG}). Stub
    # that call out so the singleton path stays observable but doesn't
    # require a fully-initialised dispatcher.

    async def _stub_dispatch(event, include_strategies=None, **kwargs):
        del kwargs
        if include_strategies is None or SLUG not in include_strategies:
            return []
        instance = strategy_loader.get_instance(SLUG)
        if instance is None:
            return []
        return list(await instance.on_event(event))

    monkeypatch.setattr(market_runtime.event_dispatcher, "dispatch", _stub_dispatch)


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
@pytest.mark.asyncio
async def test_fanout_singleton_path_when_no_bound_traders(
    monkeypatch, _patched_dispatcher_subscriptions
):
    """No traders bound to the slug -> dispatcher uses the singleton via
    ``event_dispatcher.dispatch`` and emits un-tagged opportunities."""
    _seed_global_strategy(global_threshold=100.0)

    async def _no_bindings(source_key: str):
        return {}

    monkeypatch.setattr(
        market_runtime.trader_binding_cache,
        "get_bindings_for_source",
        _no_bindings,
    )

    event = _crypto_event([50.0, 110.0, 200.0])
    opps = await market_runtime._dispatch_with_per_trader_fanout(event)

    # Global threshold 100 -> values 110, 200 pass.
    assert len(opps) == 2
    # Un-tagged emissions (legacy multi-trader-visible routing).
    assert all(opp.intended_trader_id is None for opp in opps)
    # Per-trader cache must remain empty when no bindings exist.
    assert strategy_loader.loaded_per_trader_count() == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fanout_per_trader_path_emits_tagged_opportunities(
    monkeypatch, _patched_dispatcher_subscriptions
):
    """Two traders with different ``min_distance_bps``-equivalent
    overrides produce different opportunity sets, each tagged with
    its own ``intended_trader_id``."""
    _seed_global_strategy(global_threshold=100.0)

    bindings = {
        SLUG: [
            ("trader_a", {"threshold": 50.0}),
            ("trader_b", {"threshold": 150.0}),
        ]
    }

    async def _bindings_for(source_key: str):
        return {slug: list(pairs) for slug, pairs in bindings.items()}

    monkeypatch.setattr(
        market_runtime.trader_binding_cache,
        "get_bindings_for_source",
        _bindings_for,
    )

    event = _crypto_event([60.0, 110.0, 200.0])
    opps = await market_runtime._dispatch_with_per_trader_fanout(event)

    # Trader A (threshold 50): all three values pass -> 3 opps tagged "trader_a"
    # Trader B (threshold 150): only 200 passes      -> 1 opp tagged "trader_b"
    by_trader: dict[str, list[Opportunity]] = {}
    for opp in opps:
        by_trader.setdefault(opp.intended_trader_id or "<unscoped>", []).append(opp)

    assert sorted(by_trader.keys()) == ["trader_a", "trader_b"]
    assert len(by_trader["trader_a"]) == 3
    assert len(by_trader["trader_b"]) == 1
    # Per-trader cache populated.
    assert strategy_loader.loaded_per_trader_count() == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fanout_per_trader_path_falls_back_to_global_when_params_empty(
    monkeypatch, _patched_dispatcher_subscriptions
):
    """A bound trader with empty ``strategy_params`` still goes through
    the per-trader fan-out (so the resulting opportunity is scoped to
    that trader id), but its effective threshold mirrors the global
    config."""
    _seed_global_strategy(global_threshold=120.0)

    bindings = {SLUG: [("trader_only", {})]}

    async def _bindings_for(source_key: str):
        return {slug: list(pairs) for slug, pairs in bindings.items()}

    monkeypatch.setattr(
        market_runtime.trader_binding_cache,
        "get_bindings_for_source",
        _bindings_for,
    )

    event = _crypto_event([60.0, 130.0, 200.0])
    opps = await market_runtime._dispatch_with_per_trader_fanout(event)

    # Global threshold 120 -> values 130, 200 pass; both tagged for the bound trader.
    assert len(opps) == 2
    assert all(opp.intended_trader_id == "trader_only" for opp in opps)
