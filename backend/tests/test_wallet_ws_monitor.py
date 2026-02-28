import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.wallet_ws_monitor import (
    CTF_EXCHANGE_ADDRESSES,
    ORDER_FILLED_TOPIC,
    WalletMonitorEvent,
    WalletWebSocketMonitor,
    _build_rpc_candidates,
    _determine_trade_side_and_details,
    _exception_text,
    _parse_order_filled_log,
)
from services.ws_feeds import KalshiWSFeed, PriceCache


def _word(value: int) -> str:
    return f"{value:064x}"


def test_exception_text_falls_back_to_repr_for_empty_message():
    err = TimeoutError()
    assert _exception_text(err) == repr(err)


def test_wallet_monitor_event_detected_at_uses_utc_datetime_type():
    detected_at_column = WalletMonitorEvent.__table__.c.detected_at
    assert detected_at_column.type.__class__.__name__ == "UTCDateTime"


def test_build_rpc_candidates_deduplicates_and_orders_urls():
    urls = _build_rpc_candidates("https://polygon-rpc.com")
    assert urls[0] == "https://polygon-rpc.com"
    assert len(urls) == len(set(urls))


def test_build_rpc_candidates_normalizes_ws_primary_to_https():
    urls = _build_rpc_candidates("wss://polygon-bor-rpc.publicnode.com")
    assert urls[0] == "https://polygon-bor-rpc.publicnode.com"
    assert all(url.startswith(("http://", "https://")) for url in urls)


def test_build_rpc_candidates_ignores_invalid_scheme():
    urls = _build_rpc_candidates("ftp://polygon-rpc.com")
    assert "ftp://polygon-rpc.com" not in urls
    assert "https://polygon-rpc.com" in urls


def test_parse_order_filled_log_handles_indexed_layout():
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
    token_id = int("17534298117009088653925892532629674654011194647421335094162202923073472881781")
    data = "0x" + "".join(
        [
            _word(0),
            _word(token_id),
            _word(6_919_996),
            _word(8_987_008),
            _word(0),
        ]
    )
    parsed = _parse_order_filled_log(
        {
            "topics": [
                ORDER_FILLED_TOPIC,
                "0x" + ("11" * 32),
                "0x" + ("0" * 24) + maker[2:],
                "0x" + ("0" * 24) + taker[2:],
            ],
            "data": data,
        }
    )

    assert parsed is not None
    assert parsed["maker"] == maker
    assert parsed["taker"] == taker
    assert parsed["maker_asset_id"] == "0"
    assert parsed["taker_asset_id"] == str(token_id)

    side, token, size, price = _determine_trade_side_and_details(parsed, maker)
    assert side == "BUY"
    assert token == str(token_id)
    assert size == pytest.approx(8.987008)
    assert price == pytest.approx(0.7699999811, rel=1e-6)


def test_parse_order_filled_log_handles_legacy_layout():
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
    token_id = int("31855172254305735984935876906797193204237964575325148621271964334710253764045")
    maker_word = ("0" * 24) + maker[2:]
    taker_word = ("0" * 24) + taker[2:]
    data = "0x" + "".join(
        [
            maker_word,
            taker_word,
            _word(token_id),
            _word(0),
            _word(1_210_000),
            _word(1_197_900),
            _word(0),
        ]
    )
    parsed = _parse_order_filled_log(
        {
            "topics": [
                ORDER_FILLED_TOPIC,
                "0x" + ("22" * 32),
            ],
            "data": data,
        }
    )

    assert parsed is not None
    assert parsed["maker"] == maker
    assert parsed["taker"] == taker
    assert parsed["maker_asset_id"] == str(token_id)
    assert parsed["taker_asset_id"] == "0"

    side, token, size, price = _determine_trade_side_and_details(parsed, maker)
    assert side == "SELL"
    assert token == str(token_id)
    assert size == pytest.approx(1.21)
    assert price == pytest.approx(0.99)


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
async def test_get_logs_for_block_uses_all_exchange_addresses(monkeypatch):
    monitor = WalletWebSocketMonitor()
    captured = {}

    async def _rpc_request(payload, *, method: str, block_hex: str = ""):
        captured["payload"] = payload
        captured["method"] = method
        captured["block_hex"] = block_hex
        return {"result": []}

    monkeypatch.setattr(monitor, "_rpc_request", _rpc_request)

    result = await monitor._get_logs_for_block("0x123")

    assert result == []
    assert captured["method"] == "eth_getLogs"
    assert captured["block_hex"] == "0x123"
    params = captured["payload"]["params"][0]
    assert params["address"] == list(CTF_EXCHANGE_ADDRESSES)
    assert params["topics"] == [ORDER_FILLED_TOPIC]


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
