import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.signal_bus import _strategy_runtime_metadata  # noqa: E402


def _stub_loader(monkeypatch, source_key: str, subscriptions=None):
    monkeypatch.setattr(
        "services.strategy_loader.strategy_loader.get_strategy",
        lambda slug: SimpleNamespace(
            instance=SimpleNamespace(
                source_key=source_key,
                subscriptions=subscriptions or [],
            )
        ),
    )


def test_scanner_source_uses_ws_current(monkeypatch):
    _stub_loader(monkeypatch, "scanner")
    opp = SimpleNamespace(strategy="my_scanner_strategy")
    assert _strategy_runtime_metadata(opp)["execution_activation"] == "ws_current"


def test_crypto_source_uses_immediate(monkeypatch):
    _stub_loader(monkeypatch, "crypto")
    opp = SimpleNamespace(strategy="my_crypto_strategy")
    assert _strategy_runtime_metadata(opp)["execution_activation"] == "immediate"


def test_traders_source_uses_immediate(monkeypatch):
    _stub_loader(monkeypatch, "traders", subscriptions=["trader_activity"])
    opp = SimpleNamespace(strategy="my_copy_trade_strategy")
    metadata = _strategy_runtime_metadata(opp)
    assert metadata["source_key"] == "traders"
    assert metadata["execution_activation"] == "immediate"


def test_unknown_source_falls_back_to_ws_post_arm_tick(monkeypatch):
    _stub_loader(monkeypatch, "future_unknown_source")
    opp = SimpleNamespace(strategy="some_future_strategy")
    assert _strategy_runtime_metadata(opp)["execution_activation"] == "ws_post_arm_tick"


def test_missing_strategy_returns_empty_metadata(monkeypatch):
    monkeypatch.setattr(
        "services.strategy_loader.strategy_loader.get_strategy",
        lambda slug: None,
    )
    opp = SimpleNamespace(strategy="not_loaded")
    assert _strategy_runtime_metadata(opp) == {}
