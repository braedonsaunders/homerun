"""Phase 3 tests: per-strategy ``get_pre_submit_gates()`` migrations.

For each in-tree strategy that we migrated in Phase 3, verify:

1. ``get_pre_submit_gates()`` returns the expected list of Gate names
   (so log queries and telemetry counters keep working).
2. Each gate's declared :class:`CostClass` is sane (L0 here — these
   are all in-memory predicates).
3. A representative input produces the expected pass/fail decision
   for at least one gate per strategy (smoke test).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator.gate_pipeline import (
    CostClass,
    Gate,
    GateContext,
    GateResult,
)


def _ctx(
    *,
    signal_payload: dict | None = None,
    strategy_params: dict | None = None,
    leg: dict | None = None,
    runtime_signal: object | None = None,
) -> GateContext:
    payload = signal_payload or {}
    signal = runtime_signal or SimpleNamespace(payload_json=payload)
    return GateContext(
        runtime_signal=signal,
        decision=None,
        leg=leg,
        live_context={},
        risk_limits={},
        strategy_params=strategy_params or {},
        mode="live",
        extras={"signal_payload": payload},
    )


def _run_sync(gate: Gate, ctx: GateContext) -> GateResult:
    """Run a single gate predicate synchronously — all Phase 3 gates
    are L0 / sync so this is safe."""
    raw = gate.predicate(ctx)
    if hasattr(raw, "__await__"):
        # Defensive — these should be sync.
        import asyncio

        return asyncio.get_event_loop().run_until_complete(raw)
    return raw


# ---------------------------------------------------------------------
# tail_end_carry
# ---------------------------------------------------------------------


def test_tail_end_carry_declares_expected_gates():
    from services.strategies.tail_end_carry import TailEndCarryStrategy

    gates = TailEndCarryStrategy().get_pre_submit_gates()
    assert [g.name for g in gates] == [
        "tail_carry_excluded_keyword",
        "tail_carry_probability_band",
        "tail_carry_min_liquidity",
        "tail_carry_min_upside",
    ]
    assert all(g.cost_class == CostClass.L0_MEMORY for g in gates)
    assert all(isinstance(g, Gate) for g in gates)


def test_tail_end_carry_excluded_keyword_rejects_match():
    from services.strategies.tail_end_carry import TailEndCarryStrategy

    s = TailEndCarryStrategy()
    gate = next(g for g in s.get_pre_submit_gates() if g.name == "tail_carry_excluded_keyword")
    ctx = _ctx(
        signal_payload={
            "market_question": "Will Donald Trump be elected?",
            "title": "Trump 2028",
        },
        strategy_params={"exclude_market_keywords": ["trump"]},
    )
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.reason == "tail_carry_excluded_keyword"
    assert result.detail.get("keyword") == "trump"


def test_tail_end_carry_default_exclusions_block_gap_risk_categories():
    from services.strategies.tail_end_carry import TailEndCarryStrategy

    s = TailEndCarryStrategy()
    gate = next(g for g in s.get_pre_submit_gates() if g.name == "tail_carry_excluded_keyword")

    blocked_markets = [
        ("Exact Score: Any Other Score?", "exact score"),
        ("united-states-vs-paraguay-exact-score-any-other-score", "exact-score"),
        ("United States vs. Paraguay: O/U 4.5", "o/u"),
        ("United States vs. Paraguay over/under 4.5", "over/under"),
        ("Will the price of XRP be between $1.00 and $1.10 on June 13?", "xrp"),
    ]

    for market_question, expected_keyword in blocked_markets:
        result = _run_sync(gate, _ctx(signal_payload={"market_question": market_question}))
        assert result.passed is False
        assert result.reason == "tail_carry_excluded_keyword"
        assert result.detail.get("keyword") == expected_keyword


def test_tail_end_carry_excluded_keyword_passes_no_match():
    from services.strategies.tail_end_carry import TailEndCarryStrategy

    s = TailEndCarryStrategy()
    gate = next(g for g in s.get_pre_submit_gates() if g.name == "tail_carry_excluded_keyword")
    ctx = _ctx(
        signal_payload={"market_question": "Will SpaceX launch by EOY?"},
        strategy_params={"exclude_market_keywords": ["trump"]},
    )
    assert _run_sync(gate, ctx).passed is True


def test_tail_end_carry_probability_band_rejects_below_min():
    from services.strategies.tail_end_carry import TailEndCarryStrategy

    s = TailEndCarryStrategy()
    gate = next(g for g in s.get_pre_submit_gates() if g.name == "tail_carry_probability_band")
    signal = SimpleNamespace(selected_probability=0.50, payload_json={})
    ctx = _ctx(
        runtime_signal=signal,
        strategy_params={"min_probability": 0.85, "max_probability": 0.99},
    )
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.reason == "tail_carry_probability_band"


def test_tail_end_carry_min_liquidity_rejects_thin_market():
    from services.strategies.tail_end_carry import TailEndCarryStrategy

    s = TailEndCarryStrategy()
    gate = next(g for g in s.get_pre_submit_gates() if g.name == "tail_carry_min_liquidity")
    signal = SimpleNamespace(liquidity=200.0, payload_json={})
    ctx = _ctx(runtime_signal=signal, strategy_params={"min_liquidity": 1500.0})
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.detail.get("min_liquidity") == 1500.0


# ---------------------------------------------------------------------
# news_momentum_breakout
# ---------------------------------------------------------------------


def test_news_momentum_breakout_declares_expected_gates():
    from services.strategies.news_momentum_breakout import (
        NewsMomentumBreakoutStrategy,
    )

    gates = NewsMomentumBreakoutStrategy().get_pre_submit_gates()
    assert [g.name for g in gates] == [
        "news_momentum_market_scope",
        "news_momentum_min_liquidity",
        "news_momentum_entry_band",
    ]
    assert all(g.cost_class == CostClass.L0_MEMORY for g in gates)


def test_news_momentum_market_scope_rejects_crypto_market():
    from services.strategies.news_momentum_breakout import (
        NewsMomentumBreakoutStrategy,
    )

    s = NewsMomentumBreakoutStrategy()
    gate = next(
        g for g in s.get_pre_submit_gates() if g.name == "news_momentum_market_scope"
    )
    ctx = _ctx(
        signal_payload={"market_question": "Will Bitcoin hit $100k?"},
        strategy_params={"exclude_crypto_markets": True, "exclude_sports_markets": True},
    )
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.reason == "news_momentum_market_scope"


def test_news_momentum_entry_band_rejects_out_of_band_price():
    from services.strategies.news_momentum_breakout import (
        NewsMomentumBreakoutStrategy,
    )

    s = NewsMomentumBreakoutStrategy()
    gate = next(
        g for g in s.get_pre_submit_gates() if g.name == "news_momentum_entry_band"
    )
    ctx = _ctx(
        leg={"limit_price": 0.85},
        strategy_params={"min_entry_price": 0.18, "max_entry_price": 0.78},
    )
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.reason == "news_momentum_entry_band"


# ---------------------------------------------------------------------
# sports_overreaction_fader
# ---------------------------------------------------------------------


def test_sports_overreaction_fader_declares_expected_gates():
    from services.strategies.sports_overreaction_fader import (
        SportsOverreactionFaderStrategy,
    )

    gates = SportsOverreactionFaderStrategy().get_pre_submit_gates()
    assert [g.name for g in gates] == [
        "sports_fader_min_liquidity",
        "sports_fader_max_spread_bps",
        "sports_fader_min_favorite_prob",
    ]
    assert all(g.cost_class == CostClass.L0_MEMORY for g in gates)


def test_sports_overreaction_fader_min_liquidity_rejects_thin():
    from services.strategies.sports_overreaction_fader import (
        SportsOverreactionFaderStrategy,
    )

    s = SportsOverreactionFaderStrategy()
    gate = next(
        g for g in s.get_pre_submit_gates() if g.name == "sports_fader_min_liquidity"
    )
    signal = SimpleNamespace(liquidity=500.0, payload_json={})
    ctx = _ctx(runtime_signal=signal, strategy_params={"min_liquidity": 2000.0})
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.detail.get("min_liquidity") == 2000.0


def test_sports_overreaction_fader_max_spread_bps_rejects_wide():
    from services.strategies.sports_overreaction_fader import (
        SportsOverreactionFaderStrategy,
    )

    s = SportsOverreactionFaderStrategy()
    gate = next(
        g for g in s.get_pre_submit_gates() if g.name == "sports_fader_max_spread_bps"
    )
    ctx = _ctx(
        signal_payload={"book_spread_bps": 350.0},
        strategy_params={"max_spread_bps": 200.0},
    )
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.detail.get("book_spread_bps") == 350.0


# ---------------------------------------------------------------------
# btc_eth_directional_edge
# ---------------------------------------------------------------------


def test_btc_eth_directional_edge_declares_expected_gates():
    from services.strategies.btc_eth_directional_edge import (
        BtcEthDirectionalEdgeStrategy,
    )

    gates = BtcEthDirectionalEdgeStrategy().get_pre_submit_gates()
    assert [g.name for g in gates] == [
        "btc_eth_edge_source_origin",
        "btc_eth_edge_asset_scope",
        "btc_eth_edge_live_window",
    ]
    assert all(g.cost_class == CostClass.L0_MEMORY for g in gates)


def test_btc_eth_directional_edge_source_origin_rejects_non_crypto():
    from services.strategies.btc_eth_directional_edge import (
        BtcEthDirectionalEdgeStrategy,
    )

    s = BtcEthDirectionalEdgeStrategy()
    gate = next(
        g
        for g in s.get_pre_submit_gates()
        if g.name == "btc_eth_edge_source_origin"
    )
    signal = SimpleNamespace(source="scanner", signal_type="scanner", payload_json={})
    ctx = _ctx(runtime_signal=signal)
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.reason == "btc_eth_edge_source_origin"


def test_btc_eth_directional_edge_source_origin_accepts_crypto_worker():
    from services.strategies.btc_eth_directional_edge import (
        BtcEthDirectionalEdgeStrategy,
    )

    s = BtcEthDirectionalEdgeStrategy()
    gate = next(
        g
        for g in s.get_pre_submit_gates()
        if g.name == "btc_eth_edge_source_origin"
    )
    payload = {"strategy_origin": "crypto_worker"}
    signal = SimpleNamespace(
        source="crypto",
        signal_type="crypto_worker_5m",
        payload_json=payload,
    )
    ctx = _ctx(runtime_signal=signal, signal_payload=payload)
    assert _run_sync(gate, ctx).passed is True


def test_btc_eth_directional_edge_asset_scope_rejects_off_list():
    from services.strategies.btc_eth_directional_edge import (
        BtcEthDirectionalEdgeStrategy,
    )

    s = BtcEthDirectionalEdgeStrategy()
    gate = next(
        g for g in s.get_pre_submit_gates() if g.name == "btc_eth_edge_asset_scope"
    )
    payload = {"live_market": {"asset": "SOL"}}
    ctx = _ctx(
        signal_payload=payload,
        strategy_params={"include_assets": ["BTC", "ETH"]},
    )
    result = _run_sync(gate, ctx)
    assert result.passed is False
    assert result.detail.get("asset") == "sol"
