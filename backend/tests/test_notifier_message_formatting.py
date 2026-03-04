import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.notifier import TelegramNotifier
from services import notifier as notifier_module


class _FakeResult:
    def __init__(self, *, rows=None, scalar_value=None):
        self._rows = list(rows or [])
        self._scalar_value = scalar_value

    def all(self):
        return list(self._rows)

    def scalar(self):
        return self._scalar_value


class _FakeSession:
    def __init__(self, results):
        self._results = list(results)

    async def execute(self, *_args, **_kwargs):
        if not self._results:
            raise AssertionError("No fake result available for execute()")
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_position_close_alert_has_compact_layout_and_usd_totals(monkeypatch):
    notifier = TelegramNotifier()
    captured: list[str] = []

    async def _enqueue(text: str) -> None:
        captured.append(text)

    async def _name_map(_session, _trader_ids: set[str]) -> dict[str, str]:
        return {
            "trader-1": "Alpha",
            "trader-2": "Beta",
        }

    monkeypatch.setattr(notifier, "_enqueue", _enqueue)
    monkeypatch.setattr(notifier, "_load_trader_name_map", _name_map)

    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    orders = [
        SimpleNamespace(
            id="order-1",
            trader_id="trader-1",
            status="resolved_win",
            actual_profit=15.50,
            market_question="Will CPI come in below 3%?",
            market_id="market-1",
            payload_json={"position_close": {"close_trigger": "take_profit"}},
            updated_at=now,
            created_at=now,
            mode="shadow",
        ),
        SimpleNamespace(
            id="order-2",
            trader_id="trader-2",
            status="resolved_loss",
            actual_profit=-4.25,
            market_question="Will BTC close above $120k this week?",
            market_id="market-2",
            payload_json={"position_close": {"close_trigger": "stop_loss"}},
            updated_at=now,
            created_at=now,
            mode="shadow",
        ),
    ]

    await notifier._send_position_close_alerts(
        session=object(),
        orders=orders,
        mode="shadow",
    )

    assert len(captured) == 1
    plain = captured[0].replace("\\", "").replace("*", "")
    assert "Won: $15.50" in plain
    assert "Lost: $4.25" in plain
    assert "Net: +$11.25" in plain
    assert "1) Alpha · WIN · +$15.50" in plain
    assert "2) Beta · LOSS · -$4.25" in plain
    assert "Triggers:" in plain
    assert "take_profit" in plain
    assert "stop_loss" in plain


@pytest.mark.asyncio
async def test_timeline_summary_has_compact_realized_won_lost_net():
    notifier = TelegramNotifier()
    session = _FakeSession(
        [
            _FakeResult(rows=[("selected", 6), ("blocked", 2)]),
            _FakeResult(rows=[("submitted", 4, 120.0), ("resolved_win", 2, 80.0)]),
            _FakeResult(rows=[("resolved_win", 2, 35.0), ("resolved_loss", 1, -12.5)]),
            _FakeResult(scalar_value=3),
        ]
    )

    start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)
    summary = await notifier._build_timeline_summary(
        session=session,
        start=start,
        end=end,
        control=SimpleNamespace(mode="shadow"),
        snapshot=SimpleNamespace(traders_running=2, traders_total=5),
        title="Autotrader Performance (24h)",
    )

    plain = summary.replace("\\", "").replace("*", "")
    assert "Won: $35.00" in plain
    assert "Lost: $12.50" in plain
    assert "Net: +$22.50" in plain
    assert "Decisions:" in plain
    assert "Orders:" in plain
    assert "Realized:" in plain


@pytest.mark.asyncio
async def test_status_message_has_daily_and_realized_usd_breakdown(monkeypatch):
    notifier = TelegramNotifier()

    class _SessionContext:
        async def __aenter__(self):
            return _FakeSession(
                [
                    _FakeResult(rows=[("resolved_win", 20.0), ("resolved_loss", -8.5)]),
                ]
            )

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(notifier_module, "AsyncSessionLocal", lambda: _SessionContext())
    monkeypatch.setattr(
        notifier_module,
        "read_orchestrator_control",
        AsyncMock(
            return_value={
                "mode": "shadow",
                "is_enabled": True,
                "is_paused": False,
                "kill_switch": False,
            }
        ),
    )
    monkeypatch.setattr(
        notifier_module,
        "read_orchestrator_snapshot",
        AsyncMock(
            return_value={
                "running": True,
                "traders_running": 3,
                "traders_total": 6,
                "decisions_count": 42,
                "orders_count": 18,
                "open_orders": 5,
                "daily_pnl": -12.30,
                "last_run_at": datetime(2026, 3, 2, 11, 30, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 3, 2, 11, 31, tzinfo=timezone.utc),
            }
        ),
    )

    message = await notifier._telegram_status_message()
    plain = message.replace("\\", "").replace("*", "")

    assert "Daily: -$12.30" in plain
    assert "24h Won: $20.00" in plain
    assert "24h Lost: $8.50" in plain
    assert "24h Net: +$11.50" in plain
