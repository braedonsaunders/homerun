import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.wallet_ws_monitor import (
    WalletWebSocketMonitor,
    _build_rpc_candidates,
    _exception_text,
)
from services.ws_feeds import KalshiWSFeed, PriceCache


def test_exception_text_falls_back_to_repr_for_empty_message():
    err = TimeoutError()
    assert _exception_text(err) == repr(err)


def test_build_rpc_candidates_deduplicates_and_orders_urls():
    urls = _build_rpc_candidates("https://polygon-rpc.com")
    assert urls[0] == "https://polygon-rpc.com"
    assert len(urls) == len(set(urls))


@pytest.mark.asyncio
async def test_handle_block_does_not_advance_cursor_on_rpc_failure(monkeypatch):
    monitor = WalletWebSocketMonitor()
    monitor.add_wallet("0x1111111111111111111111111111111111111111")
    monitor._last_processed_block = 100

    async def _raise_timeout(_: str):
        raise TimeoutError()

    monkeypatch.setattr(monitor, "_get_logs_for_block", _raise_timeout)

    await monitor._handle_block(101)

    assert monitor._last_processed_block == 100
    assert monitor._stats["blocks_processed"] == 0
    assert monitor._stats["errors"] == 1


@pytest.mark.asyncio
async def test_handle_block_advances_cursor_when_rpc_succeeds(monkeypatch):
    monitor = WalletWebSocketMonitor()
    monitor.add_wallet("0x1111111111111111111111111111111111111111")
    monitor._last_processed_block = 100

    async def _return_logs(_: str):
        return []

    monkeypatch.setattr(monitor, "_get_logs_for_block", _return_logs)

    await monitor._handle_block(101)

    assert monitor._last_processed_block == 101
    assert monitor._stats["blocks_processed"] == 1
    assert monitor._stats["errors"] == 0


@pytest.mark.asyncio
async def test_kalshi_ws_feed_start_skips_when_credentials_missing(monkeypatch):
    feed = KalshiWSFeed(cache=PriceCache())
    monkeypatch.setattr("services.ws_feeds.WEBSOCKETS_AVAILABLE", True)
    monkeypatch.setattr(feed, "_load_auth_headers", AsyncMock(return_value={}))

    await feed.start()

    assert feed._state.value == "closed"
    assert feed._run_task is None


@pytest.mark.asyncio
async def test_kalshi_ws_feed_starts_when_credentials_available(monkeypatch):
    feed = KalshiWSFeed(cache=PriceCache())
    monkeypatch.setattr("services.ws_feeds.WEBSOCKETS_AVAILABLE", True)
    monkeypatch.setattr(
        feed,
        "_load_auth_headers",
        AsyncMock(return_value={"Authorization": "Bearer test"}),
    )
    monkeypatch.setattr(feed, "_run_loop", AsyncMock(return_value=None))

    await feed.start()

    assert feed._run_task is not None
    await asyncio.sleep(0)
