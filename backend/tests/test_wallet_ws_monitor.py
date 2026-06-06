import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.wallet_ws_monitor import (
    ORDER_FILLED_TOPIC,
    POLYMARKET_EXCHANGE_ADDRESSES_V2,
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


def _v2_order_filled_log(
    *,
    maker: str,
    taker: str,
    side: int,
    token_id: int,
    maker_amount: int,
    taker_amount: int,
    fee: int = 0,
    order_hash: str = "0x" + ("11" * 32),
    builder_word: str = "00" * 32,
    metadata_word: str = "00" * 32,
) -> dict:
    """Build a CLOB V2 ``OrderFilled`` log payload (4 topics + 7 data words)."""
    data = "0x" + "".join(
        [
            _word(side),
            _word(token_id),
            _word(maker_amount),
            _word(taker_amount),
            _word(fee),
            builder_word,
            metadata_word,
        ]
    )
    return {
        "topics": [
            ORDER_FILLED_TOPIC,
            order_hash,
            "0x" + ("0" * 24) + maker[2:],
            "0x" + ("0" * 24) + taker[2:],
        ],
        "data": data,
    }


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


def test_build_rpc_candidates_excludes_default_auth_required_public_fallbacks():
    urls = _build_rpc_candidates("https://polygon-rpc.com")
    assert "https://rpc.ankr.com/polygon" not in urls


def test_build_rpc_candidates_does_not_reintroduce_excluded_endpoints():
    urls = _build_rpc_candidates(
        "https://polygon-bor-rpc.publicnode.com",
        excluded_urls={"https://polygon-rpc.com"},
    )
    assert "https://polygon-rpc.com" not in urls
    assert urls[0] == "https://polygon-bor-rpc.publicnode.com"


def test_build_rpc_candidates_normalizes_ws_primary_to_https():
    urls = _build_rpc_candidates("wss://polygon-bor-rpc.publicnode.com")
    assert urls[0] == "https://polygon-bor-rpc.publicnode.com"
    assert all(url.startswith(("http://", "https://")) for url in urls)


def test_build_rpc_candidates_ignores_invalid_scheme():
    urls = _build_rpc_candidates("ftp://polygon-rpc.com")
    assert "ftp://polygon-rpc.com" not in urls
    assert all(url.startswith(("http://", "https://")) for url in urls)
    assert len(urls) > 0


def test_parse_order_filled_log_decodes_v2_buy_from_maker_perspective():
    """Maker side=BUY, leader is the maker → leader's effective side is BUY.

    Maker pays 6.919996 USDC, receives 8.987008 outcome tokens at
    price 0.77 USDC per token.
    """
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0xc0d63c0d63c0d63c0d63c0d63c0d63c0d63c0d63"
    token_id = 17534298117009088653925892532629674654011194647421335094162202923073472881781
    parsed = _parse_order_filled_log(
        _v2_order_filled_log(
            maker=maker,
            taker=taker,
            side=0,
            token_id=token_id,
            maker_amount=6_919_996,
            taker_amount=8_987_008,
        )
    )

    assert parsed is not None
    assert parsed["maker"] == maker
    assert parsed["taker"] == taker
    assert parsed["side"] == 0
    assert parsed["token_id"] == str(token_id)
    assert parsed["builder"] == "0x" + "00" * 32
    assert parsed["metadata"] == "0x" + "00" * 32

    side, token, size, price = _determine_trade_side_and_details(parsed, maker)
    assert side == "BUY"
    assert token == str(token_id)
    assert size == pytest.approx(8.987008)
    assert price == pytest.approx(0.77, rel=1e-4)


def test_parse_order_filled_log_decodes_v2_sell_from_maker_perspective():
    """Maker side=SELL, leader is the maker → leader's effective side is SELL.

    Maker gives 1.21 outcome tokens, receives 1.1979 USDC at
    price 0.99 USDC per token.
    """
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0xc0d63c0d63c0d63c0d63c0d63c0d63c0d63c0d63"
    token_id = 31855172254305735984935876906797193204237964575325148621271964334710253764045
    parsed = _parse_order_filled_log(
        _v2_order_filled_log(
            maker=maker,
            taker=taker,
            side=1,
            token_id=token_id,
            maker_amount=1_210_000,
            taker_amount=1_197_900,
        )
    )

    assert parsed is not None
    assert parsed["side"] == 1
    assert parsed["token_id"] == str(token_id)

    side, token, size, price = _determine_trade_side_and_details(parsed, maker)
    assert side == "SELL"
    assert token == str(token_id)
    assert size == pytest.approx(1.21)
    assert price == pytest.approx(0.99)


def test_parse_order_filled_log_decodes_v2_buy_from_taker_perspective():
    """Maker side=BUY, leader is the taker → leader's effective side is SELL.

    The maker buys 8.987008 tokens for 6.919996 USDC; from the taker's
    perspective the taker is selling those tokens at price 0.77.
    """
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0xa11ce0a11ce0a11ce0a11ce0a11ce0a11ce0a11c"
    token_id = 17534298117009088653925892532629674654011194647421335094162202923073472881781
    parsed = _parse_order_filled_log(
        _v2_order_filled_log(
            maker=maker,
            taker=taker,
            side=0,
            token_id=token_id,
            maker_amount=6_919_996,
            taker_amount=8_987_008,
        )
    )
    assert parsed is not None

    side, token, size, price = _determine_trade_side_and_details(parsed, taker)
    assert side == "SELL"
    assert token == str(token_id)
    assert size == pytest.approx(8.987008)
    assert price == pytest.approx(0.77, rel=1e-4)


def test_parse_order_filled_log_decodes_v2_sell_from_taker_perspective():
    """Maker side=SELL, leader is the taker → leader's effective side is BUY.

    The maker sells 1.21 tokens for 1.1979 USDC; from the taker's
    perspective the taker is buying those tokens at price 0.99.
    """
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0xa11ce0a11ce0a11ce0a11ce0a11ce0a11ce0a11c"
    token_id = 31855172254305735984935876906797193204237964575325148621271964334710253764045
    parsed = _parse_order_filled_log(
        _v2_order_filled_log(
            maker=maker,
            taker=taker,
            side=1,
            token_id=token_id,
            maker_amount=1_210_000,
            taker_amount=1_197_900,
        )
    )
    assert parsed is not None

    side, token, size, price = _determine_trade_side_and_details(parsed, taker)
    assert side == "BUY"
    assert token == str(token_id)
    assert size == pytest.approx(1.21)
    assert price == pytest.approx(0.99)


def test_parse_order_filled_log_propagates_builder_and_metadata_words():
    """Both ``bytes32`` payload tail words should round-trip as 0x-hex."""
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0xc0d63c0d63c0d63c0d63c0d63c0d63c0d63c0d63"
    builder_hex = "ab" * 32
    metadata_hex = "cd" * 32
    parsed = _parse_order_filled_log(
        _v2_order_filled_log(
            maker=maker,
            taker=taker,
            side=0,
            token_id=42,
            maker_amount=10,
            taker_amount=10,
            builder_word=builder_hex,
            metadata_word=metadata_hex,
        )
    )
    assert parsed is not None
    assert parsed["builder"] == "0x" + builder_hex
    assert parsed["metadata"] == "0x" + metadata_hex


_REAL_LOGS_FIXTURE = (
    Path(__file__).parent / "fixtures" / "polymarket_v2_order_filled_logs.json"
)


def _load_real_v2_logs() -> list[dict]:
    """Load the captured live ``eth_getLogs`` response.

    Provenance lives inside the fixture's ``_provenance`` block —
    block number, transaction hash, polygonscan URL. Reviewers can
    re-derive every expected value below by visiting the polygonscan
    page for the captured tx and decoding the Logs tab against the V2
    ``OrderFilled`` ABI.
    """
    with _REAL_LOGS_FIXTURE.open() as fh:
        payload = json.load(fh)
    return list(payload["logs"])


def test_parse_order_filled_log_v2_real_block_buy_leg():
    """Pin the parser against a real BUY ``OrderFilled`` event captured
    from Polygon (tx 0x36bbcf…0766, logIndex 0x169). If Polygon ever
    changes the on-chain encoding, this assertion catches it before
    synthetic ``_v2_order_filled_log()`` builders silently agree with
    themselves."""
    log = _load_real_v2_logs()[0]
    parsed = _parse_order_filled_log(log)
    assert parsed is not None

    assert parsed["order_hash"] == (
        "0xa2d356c1739a2bd23409138cb471030a5aab8defcff2e6b5b7c26eddcf0c1bff"
    )
    assert parsed["maker"] == "0x43b3467e41f2f8d1da53fe712685b20095bc0ccc"
    assert parsed["taker"] == "0xe8b5e506aeadb393310523a24ddc726a7d8de05c"
    assert parsed["side"] == 0  # BUY
    assert parsed["token_id"] == str(
        int(
            "e12fcbb2d900f097795bc3226d4e7754ef649c45f0f5da1951c5b8f6625b9925",
            16,
        )
    )
    assert parsed["maker_amount_filled"] == 1_232_800  # 1.2328 USDC base units
    assert parsed["taker_amount_filled"] == 1_840_000  # 1.84 outcome tokens
    assert parsed["fee"] == 0
    assert parsed["builder"] == "0x" + "00" * 32
    assert parsed["metadata"] == "0x" + "00" * 32

    # Maker perspective: paid 1.2328 USDC for 1.84 tokens at price 0.67.
    side, token, size, price = _determine_trade_side_and_details(
        parsed, parsed["maker"]
    )
    assert side == "BUY"
    assert token == parsed["token_id"]
    assert size == pytest.approx(1.84)
    assert price == pytest.approx(0.67)

    # Taker perspective is the inverse: sold 1.84 tokens for 1.2328 USDC.
    side, _token, size, price = _determine_trade_side_and_details(
        parsed, parsed["taker"]
    )
    assert side == "SELL"
    assert size == pytest.approx(1.84)
    assert price == pytest.approx(0.67)


def test_parse_order_filled_log_v2_real_block_sell_leg():
    """Pin the parser against a real SELL ``OrderFilled`` event from
    the same captured tx (logIndex 0x16b)."""
    log = _load_real_v2_logs()[1]
    parsed = _parse_order_filled_log(log)
    assert parsed is not None

    assert parsed["order_hash"] == (
        "0x4edc4412b1da873522ec58ffffc94a74e9922657a600d2459a4d32b1d2fe1804"
    )
    assert parsed["maker"] == "0xc89d5c0f4d12aa83475b2b7804995578c46d9dc0"
    assert parsed["taker"] == "0xe8b5e506aeadb393310523a24ddc726a7d8de05c"
    assert parsed["side"] == 1  # SELL
    assert parsed["token_id"] == str(
        int(
            "6d516a601989d5875423553ba46a7a1ee5ade7a033a7bfc441ba99045d76159f",
            16,
        )
    )
    assert parsed["maker_amount_filled"] == 3_160_000  # 3.16 outcome tokens
    assert parsed["taker_amount_filled"] == 1_042_800  # 1.0428 USDC base units
    assert parsed["fee"] == 0

    # Maker perspective: gave 3.16 tokens, received 1.0428 USDC at 0.33.
    side, token, size, price = _determine_trade_side_and_details(
        parsed, parsed["maker"]
    )
    assert side == "SELL"
    assert token == parsed["token_id"]
    assert size == pytest.approx(3.16)
    assert price == pytest.approx(0.33)

    # Taker perspective: bought 3.16 tokens for 1.0428 USDC at 0.33.
    side, _token, size, price = _determine_trade_side_and_details(
        parsed, parsed["taker"]
    )
    assert side == "BUY"
    assert size == pytest.approx(3.16)
    assert price == pytest.approx(0.33)


def test_real_v2_log_address_in_v2_exchange_set():
    """Sanity: every fixture log's ``address`` field belongs to the V2
    exchange set we filter against. Catches a future regression where
    we accidentally re-introduced a V1 address."""
    v2_set = {addr.lower() for addr in POLYMARKET_EXCHANGE_ADDRESSES_V2}
    for log in _load_real_v2_logs():
        assert log["address"].lower() in v2_set


def test_real_v2_log_topic_matches_module_constant():
    """Sanity: every fixture log's ``topics[0]`` equals our
    ``ORDER_FILLED_TOPIC`` constant. If Polymarket ever re-keys the
    event signature, this assertion fails before production
    ``eth_getLogs`` filters silently start returning zero events."""
    for log in _load_real_v2_logs():
        assert log["topics"][0] == ORDER_FILLED_TOPIC


def test_parse_order_filled_log_rejects_v1_layout():
    """V1 emitted 2 topics + 7 data words. Must return ``None`` now —
    accepting it would be dead code (V1 contracts emit zero events on
    Polygon since the 2026-04-28 cutover) and would silently misroute
    the payload through a non-V2 schema."""
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0xc0d63c0d63c0d63c0d63c0d63c0d63c0d63c0d63"
    maker_word = ("0" * 24) + maker[2:]
    taker_word = ("0" * 24) + taker[2:]
    data = "0x" + "".join(
        [maker_word, taker_word, _word(0), _word(123), _word(1), _word(1), _word(0)]
    )
    log = {
        "topics": [ORDER_FILLED_TOPIC, "0x" + ("22" * 32)],
        "data": data,
    }
    assert _parse_order_filled_log(log) is None


def test_parse_order_filled_log_rejects_v1_indexed_layout():
    """V1's later 4-topics + 5 data-words layout must also be rejected."""
    maker = "0xb83f717cd03598dde412bc63e8ec1f18914fb5b2"
    taker = "0xc0d63c0d63c0d63c0d63c0d63c0d63c0d63c0d63"
    data = "0x" + "".join(
        [_word(0), _word(123), _word(1), _word(1), _word(0)]
    )
    log = {
        "topics": [
            ORDER_FILLED_TOPIC,
            "0x" + ("11" * 32),
            "0x" + ("0" * 24) + maker[2:],
            "0x" + ("0" * 24) + taker[2:],
        ],
        "data": data,
    }
    assert _parse_order_filled_log(log) is None


@pytest.mark.asyncio
async def test_handle_block_does_not_advance_cursor_on_rpc_failure(monkeypatch):
    monitor = WalletWebSocketMonitor()
    monitor.add_wallet("0x1111111111111111111111111111111111111111")
    monitor._last_processed_block = 100

    async def _raise_timeout(_, from_block_hex=None):
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

    async def _return_logs(_, from_block_hex=None):
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
    assert params["address"] == list(POLYMARKET_EXCHANGE_ADDRESSES_V2)
    assert params["topics"] == [ORDER_FILLED_TOPIC]


def test_wallet_monitor_endpoint_eviction_is_sticky():
    monitor = WalletWebSocketMonitor()
    monitor._http_rpc_url = "https://polygon-rpc.com"
    monitor._rpc_urls = _build_rpc_candidates(monitor._http_rpc_url)

    monitor._evict_rpc_endpoint("https://polygon-rpc.com")
    monitor._refresh_rpc_candidates()

    assert "https://polygon-rpc.com" in monitor._evicted_rpc_urls
    assert "https://polygon-rpc.com" not in monitor._rpc_urls
    assert monitor._http_rpc_url != "https://polygon-rpc.com"


@pytest.mark.asyncio
async def test_wallet_monitor_invalid_block_range_is_treated_as_transient(monkeypatch):
    monitor = WalletWebSocketMonitor()
    monitor._http_rpc_url = "https://polygon-bor-rpc.publicnode.com"
    monitor._rpc_urls = [monitor._http_rpc_url]

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "error": {
                    "code": -32000,
                    "message": "invalid block range params",
                }
            }

    class _FakeClient:
        async def post(self, endpoint, json):
            return _FakeResponse()

    async def _get_rpc_client():
        return _FakeClient()

    monkeypatch.setattr(monitor, "_get_rpc_client", _get_rpc_client)

    result = await monitor._rpc_request(
        {"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs", "params": [{}]},
        method="eth_getLogs",
        block_hex="0x123",
    )

    assert result is None
    assert monitor._rpc_urls == ["https://polygon-bor-rpc.publicnode.com"]
    assert "https://polygon-bor-rpc.publicnode.com" not in monitor._evicted_rpc_urls
    assert monitor._rpc_failure_streak == 0


# ---------------------------------------------------------------------------
# 2026-05-05: per-endpoint 429 cooldown
# ---------------------------------------------------------------------------


def test_rpc_endpoint_in_cooldown_returns_false_when_no_cooldown_set():
    monitor = WalletWebSocketMonitor()
    assert monitor._rpc_endpoint_in_cooldown("https://polygon-rpc.com") is False


def test_cool_down_rpc_endpoint_marks_endpoint_skipped():
    monitor = WalletWebSocketMonitor()
    cooldown = monitor._cool_down_rpc_endpoint("https://polygon-rpc.com")
    assert cooldown >= 30.0  # First 429 → at least the base 30s window
    assert monitor._rpc_endpoint_in_cooldown("https://polygon-rpc.com") is True


def test_cool_down_rpc_endpoint_backs_off_exponentially():
    monitor = WalletWebSocketMonitor()
    first = monitor._cool_down_rpc_endpoint("https://polygon-rpc.com")
    second = monitor._cool_down_rpc_endpoint("https://polygon-rpc.com")
    third = monitor._cool_down_rpc_endpoint("https://polygon-rpc.com")
    # Each consecutive 429 doubles the cooldown (30 → 60 → 120 → ...).
    assert second > first
    assert third > second


def test_cool_down_rpc_endpoint_honors_retry_after_header():
    monitor = WalletWebSocketMonitor()
    # First 429 base would be 30s; provider asks 90s → honor 90s.
    cooldown = monitor._cool_down_rpc_endpoint(
        "https://polygon-rpc.com",
        retry_after_seconds=90.0,
    )
    assert cooldown >= 90.0


def test_cool_down_rpc_endpoint_caps_at_max():
    monitor = WalletWebSocketMonitor()
    # 11 consecutive cooldowns ⇒ 30 * 2^10 = 30720s, must clamp.
    for _ in range(11):
        cooldown = monitor._cool_down_rpc_endpoint("https://polygon-rpc.com")
    # Final cooldown must respect the 600s ceiling.
    assert cooldown <= 600.0


def test_clear_rpc_endpoint_cooldown_resets_attempts_and_window():
    monitor = WalletWebSocketMonitor()
    monitor._cool_down_rpc_endpoint("https://polygon-rpc.com")
    monitor._cool_down_rpc_endpoint("https://polygon-rpc.com")
    monitor._clear_rpc_endpoint_cooldown("https://polygon-rpc.com")
    assert monitor._rpc_endpoint_in_cooldown("https://polygon-rpc.com") is False
    # The next 429 should restart from the base window, not from where we
    # left off.
    next_cooldown = monitor._cool_down_rpc_endpoint("https://polygon-rpc.com")
    assert next_cooldown >= 30.0
    assert next_cooldown < 90.0  # Not 4x escalated; back to fresh


@pytest.mark.asyncio
async def test_rpc_request_skips_endpoints_in_cooldown(monkeypatch):
    """If endpoint A is in cooldown, _rpc_request must skip it and use B."""
    monitor = WalletWebSocketMonitor()
    monitor._http_rpc_url = "https://a.example"
    monitor._rpc_urls = ["https://a.example", "https://b.example"]
    monitor._cool_down_rpc_endpoint("https://a.example")

    posted_to: list[str] = []

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"jsonrpc": "2.0", "id": 1, "result": "0x1"}

    class _FakeClient:
        async def post(self, endpoint, json):
            posted_to.append(endpoint)
            return _FakeResponse()

    async def _get_rpc_client():
        return _FakeClient()

    monkeypatch.setattr(monitor, "_get_rpc_client", _get_rpc_client)

    result = await monitor._rpc_request(
        {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []},
        method="eth_blockNumber",
    )

    assert result == {"jsonrpc": "2.0", "id": 1, "result": "0x1"}
    # Only the non-cooldowned endpoint received traffic.
    assert posted_to == ["https://b.example"]


@pytest.mark.asyncio
async def test_rpc_request_clears_cooldown_attempts_on_success(monkeypatch):
    """A successful request through a previously-throttled endpoint must
    reset the attempt counter so the next 429 starts from the base window."""
    monitor = WalletWebSocketMonitor()
    monitor._http_rpc_url = "https://a.example"
    monitor._rpc_urls = ["https://a.example"]
    # Pretend we've already cooled this endpoint down twice.
    monitor._rpc_endpoint_cooldown_attempts["https://a.example"] = 2
    # ...but the cooldown window has expired so the request goes through.

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"jsonrpc": "2.0", "id": 1, "result": "0x1"}

    class _FakeClient:
        async def post(self, endpoint, json):
            return _FakeResponse()

    async def _get_rpc_client():
        return _FakeClient()

    monkeypatch.setattr(monitor, "_get_rpc_client", _get_rpc_client)

    await monitor._rpc_request(
        {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []},
        method="eth_blockNumber",
    )

    assert monitor._rpc_endpoint_cooldown_attempts.get("https://a.example") in (None, 0)


@pytest.mark.asyncio
async def test_rpc_request_429_response_cools_endpoint(monkeypatch):
    """A 429 from one endpoint cools it down, falls through to the next."""
    import httpx

    monitor = WalletWebSocketMonitor()
    monitor._http_rpc_url = "https://a.example"
    monitor._rpc_urls = ["https://a.example", "https://b.example"]

    class _FakeResponse:
        def __init__(self, *, status: int):
            self.status_code = status
            self.headers = {"Retry-After": "45"} if status == 429 else {}

        def raise_for_status(self):
            if self.status_code == 429:
                raise httpx.HTTPStatusError(
                    "429",
                    request=httpx.Request("POST", "https://a.example"),
                    response=httpx.Response(429, headers={"Retry-After": "45"}),
                )

        def json(self):
            return {"jsonrpc": "2.0", "id": 1, "result": "0x1"}

    posted_to: list[str] = []

    class _FakeClient:
        async def post(self, endpoint, json):
            posted_to.append(endpoint)
            if endpoint == "https://a.example":
                # Construct a real httpx.Response with the 429 headers so
                # the wrapper's headers.get("Retry-After") path works.
                response = httpx.Response(
                    429,
                    headers={"Retry-After": "45"},
                    request=httpx.Request("POST", endpoint),
                )
                raise httpx.HTTPStatusError(
                    "429 Too Many Requests",
                    request=response.request,
                    response=response,
                )
            return _FakeResponse(status=200)

    async def _get_rpc_client():
        return _FakeClient()

    monkeypatch.setattr(monitor, "_get_rpc_client", _get_rpc_client)

    result = await monitor._rpc_request(
        {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []},
        method="eth_blockNumber",
    )

    assert result == {"jsonrpc": "2.0", "id": 1, "result": "0x1"}
    # Confirm we tried A (got 429), then fell through to B.
    assert posted_to == ["https://a.example", "https://b.example"]
    # A is now in cooldown — Retry-After 45s honored as the floor.
    assert monitor._rpc_endpoint_in_cooldown("https://a.example") is True


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
