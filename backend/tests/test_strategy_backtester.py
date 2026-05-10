from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import sqlalchemy

from models.market import Event, Market, Token
from services.strategies.base import BaseStrategy
import services.strategy_backtester as strategy_backtester


@dataclass
class FakeOpportunity:
    id: str
    stable_id: str
    roi_percent: float
    title: str = "backtest"
    strategy_context: dict = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_context, dict):
            self.strategy_context = {}

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "stable_id": self.stable_id,
            "roi_percent": self.roi_percent,
            "title": self.title,
            "strategy_context": dict(self.strategy_context),
        }


class DetectSyncOnlyStrategy(BaseStrategy):
    strategy_type = "unit_sync"
    name = "Unit Sync"
    description = "unit test"

    def detect(self, events, markets, prices):
        raise AssertionError("detect() should not be called when detect_sync() is overridden")

    def detect_sync(self, events, markets, prices):
        return [FakeOpportunity(id="sync_hit", stable_id="sync_hit", roi_percent=3.2)]


class DetectAsyncOnlyStrategy(BaseStrategy):
    strategy_type = "unit_async"
    name = "Unit Async"
    description = "unit test"

    def detect(self, events, markets, prices):
        raise AssertionError("detect() should not be called when detect_async() is overridden")

    async def detect_async(self, events, markets, prices):
        return [FakeOpportunity(id="async_hit", stable_id="async_hit", roi_percent=4.6)]


class ReplaySensitiveStrategy(BaseStrategy):
    strategy_type = "unit_replay"
    name = "Unit Replay"
    description = "unit test"

    def detect(self, events, markets, prices):
        opportunities = []
        for market in markets:
            if str(getattr(market, "id", "")) != "m1":
                continue
            yes = float(getattr(market, "yes_price", 0.0) or 0.0)
            no = float(getattr(market, "no_price", 0.0) or 0.0)
            if yes + no < 0.95:
                opportunities.append(FakeOpportunity(id="replay_hit", stable_id="replay_hit", roi_percent=6.5))
        return opportunities


class EvaluateSelectedStrategy(BaseStrategy):
    strategy_type = "unit_evaluate"
    name = "Unit Evaluate"
    description = "unit test"
    default_config = {
        "require_strict_ws_pricing": True,
        "require_live_market_revalidation": True,
        "max_market_data_age_ms": 1000,
    }

    def detect(self, events, markets, prices):
        return []

    def evaluate(self, signal, context):
        return SimpleNamespace(
            decision="selected",
            reason="selected",
            score=1.0,
            size_usd=25.0,
            checks=[],
        )


class MultiOpportunityStrategy(BaseStrategy):
    strategy_type = "unit_multi"
    name = "Unit Multi"
    description = "unit test"

    def detect(self, events, markets, prices):
        return [
            FakeOpportunity(id="opp_1", stable_id="opp_1", roi_percent=1.0),
            FakeOpportunity(id="opp_2", stable_id="opp_2", roi_percent=2.0),
            FakeOpportunity(id="opp_3", stable_id="opp_3", roi_percent=3.0),
        ]


class _FakeLoader:
    def __init__(self, instance):
        self._instance = instance

    def load(self, slug, source_code, config):
        return SimpleNamespace(instance=self._instance)

    def unload(self, slug):
        return None


def _make_market() -> Market:
    return Market(
        id="m1",
        condition_id="c1",
        question="Will unit test pass?",
        slug="unit-test-market",
        event_slug="event-1",
        tokens=[
            Token(token_id="yes_tok", outcome="Yes", price=0.6),
            Token(token_id="no_tok", outcome="No", price=0.4),
        ],
        clob_token_ids=["yes_tok", "no_tok"],
        outcome_prices=[0.6, 0.4],
        active=True,
        closed=False,
        liquidity=1000.0,
        volume=2000.0,
    )


def _make_event(market: Market) -> Event:
    return Event(
        id="event-1",
        slug="event-1",
        title="Unit Test Event",
        markets=[market],
        active=True,
        closed=False,
    )


def _patch_common(monkeypatch, strategy_instance: BaseStrategy, market: Market, event: Event) -> None:
    monkeypatch.setattr(
        strategy_backtester,
        "validate_strategy_source",
        lambda source: {
            "valid": True,
            "errors": [],
            "warnings": [],
            "class_name": strategy_instance.__class__.__name__,
        },
    )
    monkeypatch.setattr(strategy_backtester, "StrategyLoader", lambda: _FakeLoader(strategy_instance))
    monkeypatch.setattr(strategy_backtester.scanner, "_cached_events", [event], raising=False)
    monkeypatch.setattr(strategy_backtester.scanner, "_cached_markets", [market], raising=False)
    monkeypatch.setattr(strategy_backtester.scanner, "_cached_prices", {}, raising=False)


@pytest.mark.asyncio
async def test_run_strategy_backtest_uses_detect_sync_override(monkeypatch):
    market = _make_market()
    event = _make_event(market)
    strategy = DetectSyncOnlyStrategy()

    _patch_common(monkeypatch, strategy, market, event)
    monkeypatch.setattr(strategy_backtester.scanner, "_market_price_history", {}, raising=False)

    result = await strategy_backtester.run_strategy_backtest(
        source_code="class Dummy: pass",
        slug="sync_test",
        use_ohlc_replay=False,
    )

    assert result.success is True
    assert result.runtime_error is None
    assert result.num_opportunities == 1
    assert result.opportunities[0]["id"] == "sync_hit"


@pytest.mark.asyncio
async def test_run_strategy_backtest_uses_detect_async_override(monkeypatch):
    market = _make_market()
    event = _make_event(market)
    strategy = DetectAsyncOnlyStrategy()

    _patch_common(monkeypatch, strategy, market, event)
    monkeypatch.setattr(strategy_backtester.scanner, "_market_price_history", {}, raising=False)

    result = await strategy_backtester.run_strategy_backtest(
        source_code="class Dummy: pass",
        slug="async_test",
        use_ohlc_replay=False,
    )

    assert result.success is True
    assert result.runtime_error is None
    assert result.num_opportunities == 1
    assert result.opportunities[0]["id"] == "async_hit"


@pytest.mark.asyncio
async def test_run_strategy_backtest_replays_ohlc_when_live_snapshot_empty(monkeypatch):
    market = _make_market()
    event = _make_event(market)
    strategy = ReplaySensitiveStrategy()

    _patch_common(monkeypatch, strategy, market, event)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    history = [
        {"t": float(now_ms - 3_600_000), "yes": 0.44, "no": 0.44},
        {"t": float(now_ms - 1_800_000), "yes": 0.43, "no": 0.43},
    ]
    monkeypatch.setattr(strategy_backtester.scanner, "_market_price_history", {"m1": history}, raising=False)

    result = await strategy_backtester.run_strategy_backtest(
        source_code="class Dummy: pass",
        slug="replay_test",
        use_ohlc_replay=True,
        replay_lookback_hours=24,
        replay_timeframe="30m",
        replay_max_markets=1,
        replay_max_steps=12,
    )

    assert result.success is True
    assert result.runtime_error is None
    assert result.replay_mode == "ohlc_replay"
    assert result.replay_steps > 0
    assert result.replay_markets == 1
    assert result.num_opportunities == 1
    assert result.opportunities[0]["id"] == "replay_hit"
    ctx = result.opportunities[0].get("strategy_context") or {}
    assert "backtest_replay_ts_ms" in ctx


@pytest.mark.asyncio
async def test_run_strategy_backtest_caps_opportunity_output(monkeypatch):
    market = _make_market()
    event = _make_event(market)
    strategy = MultiOpportunityStrategy()

    _patch_common(monkeypatch, strategy, market, event)
    monkeypatch.setattr(strategy_backtester.scanner, "_market_price_history", {}, raising=False)

    result = await strategy_backtester.run_strategy_backtest(
        source_code="class Dummy: pass",
        slug="multi_test",
        use_ohlc_replay=False,
        max_opportunities=2,
    )

    assert result.success is True
    assert result.runtime_error is None
    assert result.num_opportunities == 2
    assert [opp["id"] for opp in result.opportunities] == ["opp_1", "opp_2"]
    assert len(result.quality_reports) == 2
    assert any("truncated to 2 rows from 3 detected" in warning for warning in result.validation_warnings)


class ExitDecisionStrategy(BaseStrategy):
    strategy_type = "unit_exit"
    name = "Unit Exit"
    description = "unit test"

    def detect(self, events, markets, prices):
        return []

    def should_exit(self, position, market_state):
        if position.pnl_percent >= 10:
            return SimpleNamespace(action="close", reason="take profit", close_price=position.current_price)
        if position.pnl_percent >= 2:
            return SimpleNamespace(action="reduce", reason="trim", reduce_fraction=0.5)
        return SimpleNamespace(action="hold", reason="let it run")


class _FakeColumn:
    def desc(self):
        return self

    def __eq__(self, other):
        return ("eq", other)


class _FakeTraderPositionModel:
    status = _FakeColumn()
    first_order_at = _FakeColumn()
    created_at = _FakeColumn()


class _FakeLegacyTraderPositionModel:
    status = _FakeColumn()
    opened_at = _FakeColumn()
    created_at = _FakeColumn()


class _FakeTradeSignalEmissionModel:
    created_at = _FakeColumn()


class _FakeQuery:
    def __init__(self) -> None:
        self.order_by_args = ()

    def where(self, *args):
        return self

    def order_by(self, *args):
        self.order_by_args = args
        return self

    def limit(self, value):
        return self


class _FakeExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeAsyncSession:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, query):
        return _FakeExecuteResult(self._rows)


class _FakeSessionContext:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return _FakeAsyncSession(self._rows)

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_run_evaluate_backtest_skips_live_execution_freshness_gates(monkeypatch):
    strategy = EvaluateSelectedStrategy()
    now = datetime.now(timezone.utc)
    signals = [
        SimpleNamespace(
            id="sig_eval_1",
            market_id="m1",
            source="scanner",
            strategy_type="unit_evaluate",
            direction="buy_yes",
            created_at=now - timedelta(minutes=5),
            payload_json={},
        ),
    ]

    monkeypatch.setattr(
        strategy_backtester,
        "validate_strategy_source",
        lambda source: {
            "valid": True,
            "errors": [],
            "warnings": [],
            "class_name": strategy.__class__.__name__,
        },
    )
    monkeypatch.setattr(strategy_backtester, "StrategyLoader", lambda: _FakeLoader(strategy))

    import models.database as database_models

    monkeypatch.setattr(database_models, "BacktestAsyncSessionLocal", lambda: _FakeSessionContext(signals))
    monkeypatch.setattr(database_models, "TradeSignalEmission", _FakeTradeSignalEmissionModel)

    captured_query: dict[str, object] = {}

    def _fake_select(*args):
        query = _FakeQuery()
        captured_query["model"] = args[0]
        captured_query["query"] = query
        return query

    monkeypatch.setattr(sqlalchemy, "select", _fake_select)

    result = await strategy_backtester.run_evaluate_backtest(
        source_code="class Dummy: pass",
        slug="evaluate_test",
        max_signals=1,
    )

    assert result.success is True
    assert result.runtime_error is None
    assert result.num_signals == 1
    assert result.selected == 1
    assert result.blocked == 0
    assert captured_query["model"] is _FakeTradeSignalEmissionModel

    decision = result.decisions[0]
    assert decision["decision"] == "selected"
    assert any(g["gate"] == "strict_ws_pricing" and g["status"] == "skipped" for g in decision["platform_gates"])
    assert any(
        g["gate"] == "live_market_revalidation" and g["status"] == "skipped"
        for g in decision["platform_gates"]
    )
    assert any(
        g["gate"] == "market_data_freshness" and g["status"] == "skipped"
        for g in decision["platform_gates"]
    )


@pytest.mark.asyncio
async def test_run_exit_backtest_uses_existing_position_columns_and_tracks_actions(monkeypatch):
    strategy = ExitDecisionStrategy()
    now = datetime.now(timezone.utc)

    positions = [
        SimpleNamespace(
            id="p_close",
            market_id="m1",
            market_question="Market close",
            direction="buy_yes",
            mode="shadow",
            total_notional_usd=1000.0,
            avg_entry_price=0.4,
            first_order_at=now - timedelta(minutes=90),
            created_at=now - timedelta(minutes=95),
            payload_json={"entry_price": 0.4, "last_price": 0.45, "strategy_context": {"tag": "a"}},
        ),
        SimpleNamespace(
            id="p_reduce",
            market_id="m2",
            market_question="Market reduce",
            direction="buy_no",
            mode="shadow",
            total_notional_usd=800.0,
            avg_entry_price=0.5,
            first_order_at=now - timedelta(minutes=45),
            created_at=now - timedelta(minutes=50),
            payload_json={"entry_price": 0.5, "last_price": 0.515, "strategy_context": {"tag": "b"}},
        ),
        SimpleNamespace(
            id="p_hold",
            market_id="m3",
            market_question="Market hold",
            direction="buy_yes",
            mode="shadow",
            total_notional_usd=600.0,
            avg_entry_price=0.6,
            first_order_at=now - timedelta(minutes=20),
            created_at=now - timedelta(minutes=25),
            payload_json={"entry_price": 0.6, "last_price": 0.59, "strategy_context": {"tag": "c"}},
        ),
    ]

    monkeypatch.setattr(
        strategy_backtester,
        "validate_strategy_source",
        lambda source: {
            "valid": True,
            "errors": [],
            "warnings": [],
            "class_name": strategy.__class__.__name__,
        },
    )
    monkeypatch.setattr(strategy_backtester, "StrategyLoader", lambda: _FakeLoader(strategy))

    import models.database as database_models

    monkeypatch.setattr(database_models, "BacktestAsyncSessionLocal", lambda: _FakeSessionContext(positions))
    monkeypatch.setattr(database_models, "TraderPosition", _FakeTraderPositionModel)

    captured_query: dict[str, object] = {}

    def _fake_select(*args):
        query = _FakeQuery()
        captured_query["model"] = args[0]
        captured_query["query"] = query
        return query

    monkeypatch.setattr(sqlalchemy, "select", _fake_select)

    result = await strategy_backtester.run_exit_backtest(source_code="class Dummy: pass", slug="exit_test")

    assert result.success is True
    assert result.runtime_error is None
    assert result.num_positions == 3
    assert result.would_close == 1
    assert result.would_reduce == 1
    assert result.would_hold == 1
    assert result.errors == 0
    assert len(result.exit_decisions) == 3

    decisions = {row["position_id"]: row for row in result.exit_decisions}
    assert decisions["p_close"]["action"] == "close"
    assert decisions["p_reduce"]["action"] == "reduce"
    assert decisions["p_reduce"]["reduce_fraction"] == 0.5
    assert decisions["p_hold"]["action"] == "hold"
    assert decisions["p_close"]["age_minutes"] > 0
    assert decisions["p_close"]["mode"] == "shadow"
    assert captured_query["model"] is _FakeTraderPositionModel
    assert len(getattr(captured_query["query"], "order_by_args", ())) == 2


@pytest.mark.asyncio
async def test_run_exit_backtest_supports_opened_at_fallback_column(monkeypatch):
    strategy = ExitDecisionStrategy()
    now = datetime.now(timezone.utc)

    positions = [
        SimpleNamespace(
            id="p_legacy",
            market_id="m_legacy",
            market_question="Legacy opened_at position",
            direction="buy_yes",
            mode="shadow",
            total_notional_usd=500.0,
            avg_entry_price=0.45,
            created_at=now - timedelta(minutes=70),
            payload_json={"entry_price": 0.45, "last_price": 0.5, "strategy_context": {"tag": "legacy"}},
        ),
    ]

    monkeypatch.setattr(
        strategy_backtester,
        "validate_strategy_source",
        lambda source: {
            "valid": True,
            "errors": [],
            "warnings": [],
            "class_name": strategy.__class__.__name__,
        },
    )
    monkeypatch.setattr(strategy_backtester, "StrategyLoader", lambda: _FakeLoader(strategy))

    import models.database as database_models

    monkeypatch.setattr(database_models, "BacktestAsyncSessionLocal", lambda: _FakeSessionContext(positions))
    monkeypatch.setattr(database_models, "TraderPosition", _FakeLegacyTraderPositionModel)

    captured_query: dict[str, object] = {}

    def _fake_select(*args):
        query = _FakeQuery()
        captured_query["model"] = args[0]
        captured_query["query"] = query
        return query

    monkeypatch.setattr(sqlalchemy, "select", _fake_select)

    result = await strategy_backtester.run_exit_backtest(source_code="class Dummy: pass", slug="exit_legacy")

    assert result.success is True
    assert result.runtime_error is None
    assert result.num_positions == 1
    assert result.would_close == 1
    assert len(result.exit_decisions) == 1
    assert captured_query["model"] is _FakeLegacyTraderPositionModel
    assert len(getattr(captured_query["query"], "order_by_args", ())) == 2


# ── Replay-discovery: event-driven strategies ───────────────────────────


class _StrategyWithScope:
    """Minimal strategy stub for scope-extraction tests."""

    def __init__(self, scope_cfg=None):
        self.config = {"traders_scope": scope_cfg} if scope_cfg is not None else {}
        self.accepted_signal_strategy_types: list[str] = []


def test_replay_event_kind_dispatch_by_slug():
    s = _StrategyWithScope()
    assert strategy_backtester._replay_event_kind_for_strategy(
        "traders_copy_trade", s
    ) == "wallet_trade"
    assert strategy_backtester._replay_event_kind_for_strategy(
        "TRADERS_COPY_TRADE", s
    ) == "wallet_trade"
    assert strategy_backtester._replay_event_kind_for_strategy("momentum", s) is None


def test_replay_event_kind_dispatch_by_accepted_types():
    class S:
        accepted_signal_strategy_types = ["traders_copy_trade"]
        config: dict = {}

    assert strategy_backtester._replay_event_kind_for_strategy(
        "renamed_slug", S()
    ) == "wallet_trade"


def test_extract_scope_wallets_normalises_and_dedupes():
    s = _StrategyWithScope({"individual_wallets": ["0xABC", "  0xdef ", "", None, "0xABC"]})
    got = strategy_backtester._extract_scope_wallets(s)
    assert got == {"0xabc", "0xdef"}


def test_extract_scope_wallets_returns_none_when_unset():
    assert strategy_backtester._extract_scope_wallets(_StrategyWithScope()) is None
    assert strategy_backtester._extract_scope_wallets(
        _StrategyWithScope({"individual_wallets": []})
    ) is None


def test_wallet_event_to_strategy_input_matches_live_signal_shape():
    market = {
        "market_id": "c1",
        "market_question": "Will X win?",
        "market_slug": "x",
        "outcome": "Yes",
        "liquidity": 1000.0,
        "token_id": "TOK1",
    }

    class _Ev:
        side = "buy"
        token_id = "TOK1"
        price = 0.62
        size = 100.0
        wallet_address = "0xABC"
        detected_at = datetime(2026, 5, 1, 12, 0, 0)
        tx_hash = "0xtx"
        order_hash = "0xorder"
        log_index = 7
        block_number = 12345
        detection_latency_ms = 250.0

    shaped = strategy_backtester._wallet_event_to_strategy_input(
        _Ev(), market_payload=market
    )
    assert shaped is not None
    assert shaped["copy_event"]["side"] == "BUY"
    assert shaped["copy_event"]["wallet_address"] == "0xabc"
    assert shaped["copy_event"]["confidence"] == 0.70
    assert shaped["source_trade"]["source_notional_usd"] == pytest.approx(0.62 * 100.0)
    assert shaped["source_item_id"] == "0xtx:0xabc:TOK1:BUY:7:0xorder"
    assert shaped["market"] == market


def test_wallet_event_to_strategy_input_rejects_invalid_rows():
    market = {"token_id": "TOK1", "market_id": "c1"}

    class _BadSide:
        side = "huh"
        token_id = "TOK1"
        price = 0.5
        size = 1.0
        wallet_address = "0x1"
        detected_at = datetime(2026, 5, 1, 12, 0, 0)
        tx_hash = ""
        order_hash = ""
        log_index = 0
        block_number = 0
        detection_latency_ms = 0.0

    assert strategy_backtester._wallet_event_to_strategy_input(
        _BadSide(), market_payload=market
    ) is None

    class _ZeroPrice(_BadSide):
        side = "buy"
        price = 0.0

    assert strategy_backtester._wallet_event_to_strategy_input(
        _ZeroPrice(), market_payload=market
    ) is None


def test_build_token_to_market_lookup_handles_json_string_arrays():
    catalog = [
        {
            "condition_id": "c1",
            "question": "Q1",
            "event_slug": "e1",
            "liquidity": 5000.0,
            "clob_token_ids": ["T_A", "T_B"],
            "outcomes": ["Yes", "No"],
        },
        {
            "id": "c2",
            "question": "Q2",
            "clob_token_ids": '["T_C"]',
            "outcomes": '["Win"]',
        },
        {"clob_token_ids": []},
    ]
    lookup = strategy_backtester._build_token_to_market_lookup(catalog)
    assert set(lookup.keys()) == {"T_A", "T_B", "T_C"}
    assert lookup["T_A"]["outcome"] == "Yes"
    assert lookup["T_B"]["outcome"] == "No"
    assert lookup["T_C"]["market_id"] == "c2"
    assert lookup["T_A"]["liquidity"] == 5000.0


class _CopyTradeStrategy(BaseStrategy):
    """Mirrors the public surface of ``traders_copy_trade`` enough for
    ``_replay_discover_opportunities`` to dispatch it as event-driven
    and produce one opportunity per event.
    """

    strategy_type = "traders_copy_trade"
    name = "Copy Trade Test"
    description = "unit test"
    accepted_signal_strategy_types = ["traders_copy_trade"]

    def __init__(self, scope_cfg=None):
        super().__init__()
        # BaseStrategy.__init__ resets self.config from default_config,
        # so apply the scope override AFTER the base init.
        if scope_cfg is not None:
            self.config = dict(self.config)
            self.config["traders_scope"] = scope_cfg

    def detect(self, events, markets, prices):
        out = []
        for ev in events:
            ce = ev.get("copy_event") or {}
            tok = ce.get("token_id")
            if not tok:
                continue
            # Mirror the real strategy's Opportunity shape: it carries
            # ``positions_to_take`` so downstream backtest code can
            # extract the trade intents.
            opp = SimpleNamespace(
                title=f"copy {tok}",
                event_id=ce.get("tx_hash"),
                positions_to_take=[
                    {
                        "token_id": tok,
                        "side": ce.get("side"),
                        "size": float(ce.get("size") or 0.0),
                        "price": float(ce.get("price") or 0.0),
                    }
                ],
                total_cost=float(ce.get("price") or 0.0) * float(ce.get("size") or 0.0),
                expected_roi=1.0,
                risk_score=0.0,
            )
            out.append(opp)
        return out

    async def detect_async(self, events, markets, prices):
        # Mirrors the production strategy: detect_async wraps detect.
        return self.detect(events, markets, prices)


@pytest.mark.asyncio
async def test_replay_discover_feeds_wallet_events_to_event_driven_strategy(monkeypatch):
    """End-to-end: a copy-trade strategy with no live opps should still
    produce synthetic opps when wallet events exist in the window."""

    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    catalog_markets = [
        {
            "id": "c1",
            "question": "Will Q?",
            "clob_token_ids": ["TOK1", "TOK2"],
            "outcomes": ["Yes", "No"],
            "liquidity": 1500.0,
            "active": True,
            "closed": False,
        }
    ]
    monkeypatch.setattr(
        "services.shared_state._read_market_catalog_file",
        lambda: ([], catalog_markets, {}),
    )

    class _StubEvent:
        def __init__(self, ts, side, token_id, tx):
            self.detected_at = ts
            self.side = side
            self.token_id = token_id
            self.price = 0.55
            self.size = 50.0
            self.wallet_address = "0xtracked"
            self.tx_hash = tx
            self.order_hash = ""
            self.log_index = 0
            self.block_number = 1
            self.detection_latency_ms = 100.0

    stub_events = [
        _StubEvent(start + timedelta(hours=2), "buy", "TOK1", "0xa"),
        _StubEvent(start + timedelta(hours=10), "sell", "TOK2", "0xb"),
        _StubEvent(start + timedelta(hours=20), "buy", "TOK1", "0xc"),
    ]

    async def _stub_loader(**_kw):
        return list(stub_events), False

    monkeypatch.setattr(
        strategy_backtester,
        "_load_wallet_events_for_replay",
        _stub_loader,
    )

    # Stub AsyncSessionLocal so the SET statement_timeout call is a no-op.
    class _StubSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        async def execute(self, *args, **kwargs):
            class _R:
                def scalar_one(self): return 0
                def all(self): return []
                def scalars(self):
                    class _S:
                        def all(self): return []
                        def first(self): return None
                    return _S()
            return _R()

    monkeypatch.setattr(
        "models.database.AsyncSessionLocal",
        lambda: _StubSession(),
    )

    strategy = _CopyTradeStrategy(
        scope_cfg={"individual_wallets": ["0xtracked"]}
    )

    opps = await strategy_backtester._replay_discover_opportunities(
        strategy=strategy,
        slug="traders_copy_trade",
        start_dt=start,
        end_dt=end,
        sample_interval_seconds=1800,
        max_ticks=96,
    )

    # Each of the 3 wallet events should produce one synthetic opp.
    assert len(opps) == 3, f"expected 3 synthetic opps, got {len(opps)}"
    assert all(o.strategy_type == "traders_copy_trade" for o in opps)
    # Detected_at should be ordered (events were ordered chronologically)
    times = [o.detected_at for o in opps]
    assert times == sorted(times)


class _PlainDetectScannerStrategy(BaseStrategy):
    """Mirrors the override pattern used by ~20 scanner strategies in
    this codebase (tail_end_carry, stat_arb, news_momentum_breakout,
    every BTC/ETH variant, etc.): only ``detect()`` is overridden,
    not ``detect_sync`` / ``detect_async``.

    Discovery used to early-return for these strategies, which is why
    backtests of tail_end_carry etc. produced near-zero opps even when
    live had been firing for a week.  This test pins the fix.
    """

    strategy_type = "unit_plain_detect"
    name = "Plain detect()"
    description = "unit test"

    def detect(self, events, markets, prices):
        out = []
        for m in markets:
            tok_ids = list(getattr(m, "clob_token_ids", []) or [])
            for token_id in tok_ids:
                px = prices.get(token_id)
                if not isinstance(px, dict):
                    continue
                if (px.get("mid") or 0.0) <= 0.0:
                    continue
                out.append(SimpleNamespace(
                    title=f"plain_hit {token_id}",
                    event_id=token_id,
                    positions_to_take=[
                        {"token_id": token_id, "side": "BUY", "price": px["mid"], "size": 10.0}
                    ],
                    total_cost=px["mid"] * 10.0,
                    expected_roi=1.0,
                    risk_score=0.0,
                ))
        return out


@pytest.mark.asyncio
async def test_replay_discover_runs_strategies_that_only_override_detect(monkeypatch):
    """Regression: tail_end_carry-style strategies (custom detect(), no
    custom detect_sync/detect_async) must NOT be early-returned by
    replay discovery.  Previously this skipped ~20 scanner strategies."""

    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=4)

    catalog_markets = [
        {
            "id": "c1",
            "question": "Will Q?",
            "clob_token_ids": ["TOK1"],
            "outcomes": ["Yes"],
            "active": True,
            "closed": False,
        }
    ]
    monkeypatch.setattr(
        "services.shared_state._read_market_catalog_file",
        lambda: ([], catalog_markets, {}),
    )

    # Stub the per-tick price grid so TOK1 has a state at tick 1 onward.
    async def _stub_grid(*, session, token_ids, ticks, end_dt, use_deltas):
        return {
            "TOK1": [
                None if i == 0 else {
                    "best_bid": 0.50,
                    "best_ask": 0.52,
                    "mid": 0.51,
                    "price": 0.51,
                    "spread_bps": 20.0,
                    "observed_at": ticks[0],
                }
                for i in range(len(ticks))
            ]
        }

    monkeypatch.setattr(
        strategy_backtester, "_build_per_tick_prices_grid", _stub_grid
    )

    # The probe runs even when grid is stubbed; short-circuit to avoid DB.
    async def _no_probe(**_kw):
        return False

    monkeypatch.setattr(
        strategy_backtester, "_probe_should_prefer_deltas", _no_probe
    )

    class _StubSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        async def execute(self, *a, **kw):
            class _R:
                def scalar_one(self): return 0
                def all(self): return []
                def scalars(self):
                    class _S:
                        def all(self): return []
                        def first(self): return None
                    return _S()
            return _R()

    monkeypatch.setattr(
        "models.database.BacktestAsyncSessionLocal", lambda: _StubSession()
    )

    strategy = _PlainDetectScannerStrategy()
    opps = await strategy_backtester._replay_discover_opportunities(
        strategy=strategy,
        slug="unit_plain_detect",
        start_dt=start,
        end_dt=end,
        sample_interval_seconds=600,
        max_ticks=12,
    )
    # 4 hours / 600s = 24 raw ticks but capped at 12; tick 0 has no
    # state so detect runs on ticks 1..11 → 11 opps.  Allow >0 for
    # robustness against tick math drift.
    assert len(opps) > 0, "plain-detect() strategy produced 0 opps — regression!"
    # All synthetic opps must carry the strategy slug + a token-shaped
    # positions_to_take so the downstream matching engine can use them.
    for o in opps:
        assert o.strategy_type == "unit_plain_detect"
        assert o.positions_data["positions_to_take"]
        assert o.positions_data["positions_to_take"][0]["token_id"] == "TOK1"


@pytest.mark.asyncio
async def test_replay_discover_skips_event_loading_for_book_strategies(monkeypatch):
    """Sanity: book-driven strategies don't trigger the wallet-event
    load path even when WalletMonitorEvent is populated.  This catches
    accidental dispatch flips."""

    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    catalog_markets = [
        {"id": "c1", "clob_token_ids": ["TOK1"], "outcomes": ["Yes"], "active": True}
    ]
    monkeypatch.setattr(
        "services.shared_state._read_market_catalog_file",
        lambda: ([], catalog_markets, {}),
    )

    load_called = {"flag": False}

    async def _should_not_run(**_kw):
        load_called["flag"] = True
        return [], False

    monkeypatch.setattr(
        strategy_backtester,
        "_load_wallet_events_for_replay",
        _should_not_run,
    )

    class _StubSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        async def execute(self, *args, **kwargs):
            class _R:
                def scalar_one(self): return 0
                def all(self): return []
                def scalars(self):
                    class _S:
                        def all(self): return []
                        def first(self): return None
                    return _S()
            return _R()

    monkeypatch.setattr(
        "models.database.AsyncSessionLocal",
        lambda: _StubSession(),
    )

    # A vanilla book-driven strategy.
    strategy = DetectAsyncOnlyStrategy()
    await strategy_backtester._replay_discover_opportunities(
        strategy=strategy,
        slug="async_test",
        start_dt=start,
        end_dt=end,
        sample_interval_seconds=600,
        max_ticks=12,
    )
    assert load_called["flag"] is False, (
        "wallet event loader should not run for non-event-driven strategies"
    )
