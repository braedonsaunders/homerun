from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

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
