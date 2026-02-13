import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_maintenance


@pytest.mark.asyncio
async def test_flush_data_requires_confirm():
    session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())

    with pytest.raises(HTTPException) as excinfo:
        await routes_maintenance.flush_data(
            routes_maintenance.FlushDataRequest(target="scanner", confirm=False),
            session=session,
        )

    assert excinfo.value.status_code == 400
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_flush_all_calls_all_handlers(monkeypatch):
    session = object()
    flush_scanner = AsyncMock(return_value={"scanner_snapshot_opportunities": 3})
    flush_weather = AsyncMock(return_value={"weather_trade_intents": 2})
    flush_news = AsyncMock(return_value={"news_article_cache": 5})
    flush_trader_orchestrator = AsyncMock(return_value={"trade_signal_snapshots": 1})

    monkeypatch.setattr(routes_maintenance, "_flush_scanner_data", flush_scanner)
    monkeypatch.setattr(routes_maintenance, "_flush_weather_data", flush_weather)
    monkeypatch.setattr(routes_maintenance, "_flush_news_data", flush_news)
    monkeypatch.setattr(routes_maintenance, "_flush_trader_orchestrator_runtime_data", flush_trader_orchestrator)

    result = await routes_maintenance._run_flush(session, "all")

    assert list(result.keys()) == ["scanner", "weather", "news", "trader_orchestrator"]
    flush_scanner.assert_awaited_once_with(session)
    flush_weather.assert_awaited_once_with(session)
    flush_news.assert_awaited_once_with(session)
    flush_trader_orchestrator.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_flush_data_commits_and_returns_payload(monkeypatch):
    session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
    run_flush = AsyncMock(
        return_value={
            "scanner": {"scanner_snapshot_opportunities": 4},
        }
    )
    monkeypatch.setattr(routes_maintenance, "_run_flush", run_flush)

    response = await routes_maintenance.flush_data(
        routes_maintenance.FlushDataRequest(target="scanner", confirm=True),
        session=session,
    )

    run_flush.assert_awaited_once_with(session, "scanner")
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()
    assert response["status"] == "success"
    assert response["target"] == "scanner"
    assert "protected_datasets" in response
    assert any("trader_orders" in value for value in response["protected_datasets"])


@pytest.mark.asyncio
async def test_flush_data_rolls_back_on_error(monkeypatch):
    session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
    monkeypatch.setattr(routes_maintenance, "_run_flush", AsyncMock(side_effect=RuntimeError("boom")))

    with pytest.raises(HTTPException) as excinfo:
        await routes_maintenance.flush_data(
            routes_maintenance.FlushDataRequest(target="news", confirm=True),
            session=session,
        )

    assert excinfo.value.status_code == 500
    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once()
