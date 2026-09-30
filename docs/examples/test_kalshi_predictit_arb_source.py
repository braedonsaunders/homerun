"""Tests for the kalshi-predictit-arb data source (docs/examples/kalshi_predictit_arb_source.py).

Covers: offline demo mode (no network, no key), homerun SDK integration (the
data-source validator, ``DataSourceLoader`` and ``preview_data_source``), the
x402 402 -> sign -> retry flow with an unpadded base64url payment header, and
the payment refusal rules. No real network or keys: HTTP goes through
``httpx.MockTransport`` and signing uses a throwaway ``Account.create()``.

Run from ``backend/``: ``python -m pytest ../docs/examples/test_kalshi_predictit_arb_source.py``
"""

import base64
import json
import sys
from pathlib import Path

import httpx
import pytest

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parents[1] / "backend"
for _path in (str(_BACKEND), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import kalshi_predictit_arb_source as arb  # noqa: E402
from models.database import DataSource  # noqa: E402
from services.data_source_loader import data_source_loader, validate_data_source_source  # noqa: E402
from services.data_source_runner import preview_data_source  # noqa: E402
from services.data_source_sdk import BaseDataSource  # noqa: E402

SOURCE_CODE = (_HERE / "kalshi_predictit_arb_source.py").read_text(encoding="utf-8")
SLUG = "kalshi_predictit_arb_test"
PAY_TO = "0x" + "ab" * 20
LIVE_BODY = {
    "opportunities": [{"pair": "LIVE-TEST A <-> B", "best_direction": {"net_yield_c": 2.5}, "executable": True}]
}


def _requirement(**overrides):
    requirement = {
        "scheme": "exact",
        "network": "eip155:8453",
        "amount": "20000",
        "asset": arb.BASE_USDC,
        "payTo": PAY_TO,
        "maxTimeoutSeconds": 60,
        "extra": {"name": "USD Coin", "version": "2"},
    }
    requirement.update(overrides)
    return requirement


def _required(accepts):
    return {"x402Version": 2, "resource": {"url": arb.DEFAULT_URL}, "accepts": accepts}


def _b64(obj, urlsafe=False, pad=True):
    raw = json.dumps(obj).encode()
    text = (base64.urlsafe_b64encode(raw) if urlsafe else base64.b64encode(raw)).decode()
    return text if pad else text.rstrip("=")


class FakeServer:
    """402 on the first unpaid request, 200 once a payment header is present."""

    def __init__(self, required, variant="header", always_402=False):
        self.required = required
        self.variant = variant
        self.always_402 = always_402
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if "PAYMENT-SIGNATURE" in request.headers and not self.always_402:
            return httpx.Response(200, json=LIVE_BODY, headers={"PAYMENT-RESPONSE": _b64({"success": True})})
        if self.variant == "body":
            return httpx.Response(402, json=self.required)
        urlsafe = self.variant == "urlsafe"
        return httpx.Response(402, headers={"PAYMENT-REQUIRED": _b64(self.required, urlsafe=urlsafe, pad=not urlsafe)})


def _source(signer=None, server=None, **config):
    source = arb.KalshiPredictItArbSource()
    source.configure(config)
    source.payment_signer = signer
    if server is not None:
        source.transport = httpx.MockTransport(server)
    return source


@pytest.fixture
def no_network(monkeypatch):
    async def _blocked(*args, **kwargs):
        raise AssertionError("network call attempted in demo mode")

    monkeypatch.setattr(httpx.AsyncClient, "send", _blocked)


@pytest.fixture
def signer():
    pytest.importorskip("eth_account")
    from eth_account import Account
    from kalshi_predictit_arb_signer import EthAccountSigner

    return EthAccountSigner(private_key=Account.create().key)


# ---------------------------------------------------------------------------
# Fixture / demo mode
# ---------------------------------------------------------------------------


def test_sample_fixture_is_labelled_fictional_and_matches_embedded_copy():
    fixture = json.loads((_HERE / "kalshi_predictit_arb_sample.json").read_text(encoding="utf-8"))
    assert fixture == arb.SAMPLE_RESPONSE
    assert "NOT real market data" in fixture["_notice"]
    assert all(row["event"].startswith("SAMPLE") for row in fixture["opportunities"])
    assert fixture["opportunities"][0]["kalshi"]["ticker"] == "SAMPLE-SENATEXX-26-D"


def test_defaults():
    source = arb.KalshiPredictItArbSource()
    assert isinstance(source, BaseDataSource)
    assert source.demo is True
    assert source.config["max_usd_per_call"] == 0.02
    assert source._max_atomic() == 20000
    assert "private_key" not in source.config


@pytest.mark.asyncio
async def test_demo_mode_makes_no_network_calls(no_network):
    records = await _source().fetch_async()
    assert [r["executable"] for r in records] == [True, True]
    assert all(r["title"].startswith("[SAMPLE DATA] SAMPLE") for r in records)
    assert all("sample" in r["tags"] and r["sample_data"] is True for r in records)


@pytest.mark.asyncio
async def test_demo_mode_all_filters_and_missing_fields(no_network):
    records = await _source(mode="all").fetch_async()
    assert len(records) == 5
    missing = records[-1]
    assert missing["net_yield_c"] is None and missing["executable"] is False
    assert missing["best_direction"] is None
    assert "n/a" in missing["summary"]
    assert len(await _source(mode="all", q="senate").fetch_async()) == 2
    assert len(await _source(mode="all", limit=1).fetch_async()) == 1
    with pytest.raises(ValueError):
        await _source(mode="bogus").fetch_async()


async def _titles(rows):
    source = _source()

    async def scan():
        return {"opportunities": rows}

    source.scan = scan
    return [r["title"] for r in await source.fetch_async()]


@pytest.mark.asyncio
async def test_event_is_preferred_over_pair():
    assert await _titles([{"event": "E", "pair": "P"}]) == ["E"]


@pytest.mark.asyncio
async def test_pair_is_used_when_event_is_missing():
    assert await _titles([{"pair": "P"}]) == ["P"]


@pytest.mark.asyncio
async def test_rows_without_event_or_pair_are_skipped():
    assert await _titles([{"executable": True}, {"event": "", "pair": None}, {"event": " "}]) == []


# ---------------------------------------------------------------------------
# homerun SDK integration
# ---------------------------------------------------------------------------


def test_source_passes_homerun_validator():
    result = validate_data_source_source(SOURCE_CODE)
    assert result["valid"], result["errors"]
    assert result["class_name"] == "KalshiPredictItArbSource"
    assert result["capabilities"]["has_fetch_async"] is True
    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_preview_data_source_normalizes_records(no_network):
    config = {"mode": "all", "limit": 25}
    source = DataSource(slug=SLUG, source_code=SOURCE_CODE, config=config, class_name=None)
    try:
        preview = await preview_data_source(source, max_records=25)
        runtime = data_source_loader.get_runtime(SLUG)
        assert isinstance(runtime.instance, BaseDataSource)
        assert runtime.instance.config["max_usd_per_call"] == 0.02
    finally:
        data_source_loader.unload(SLUG)
    assert preview["source_slug"] == SLUG
    assert preview["total_fetched"] == 5
    first = preview["records"][0]
    assert first["category"] == "cross_venue_arb"
    assert first["source"] == "kalshi-predictit-arb"
    assert first["observed_at"] and first["external_id"]
    assert first["tags"] == ["kalshi", "predictit", "arbitrage", "sample"]
    assert first["payload"]["sample_data"] is True
    assert first["payload"]["net_yield_c"] == 3.1 and first["payload"]["executable"] is True
    assert first["payload"]["best_direction"]["net_yield_c"] == 3.1
    assert first["payload"]["best_direction"]["net_yield_pct"] == 3.4
    assert len({r["external_id"] for r in preview["records"]}) == 5


@pytest.mark.asyncio
async def test_sandboxed_runtime_with_signer_pays_once(signer):
    server = FakeServer(_required([_requirement()]))
    runtime = data_source_loader.load(slug=SLUG, source_code=SOURCE_CODE, config={})
    try:
        runtime.instance.payment_signer = signer
        runtime.instance.transport = httpx.MockTransport(server)
        preview = await preview_data_source(DataSource(slug=SLUG, source_code=SOURCE_CODE, config={}))
    finally:
        data_source_loader.unload(SLUG)
    assert len(server.requests) == 2
    assert preview["records"][0]["title"] == "LIVE-TEST A <-> B"
    assert "sample" not in preview["records"][0]["tags"]


# ---------------------------------------------------------------------------
# 402 -> sign -> retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["header", "urlsafe", "body"])
async def test_402_sign_retry(signer, variant):
    from eth_account import Account
    from eth_account.messages import encode_typed_data

    requirement = _requirement()
    server = FakeServer(_required([requirement]), variant=variant)
    source = _source(signer, server, q="senate")
    records = await source.fetch_async()

    assert records[0]["title"] == "LIVE-TEST A <-> B"
    assert source.last_payment_response == {"success": True}
    assert len(server.requests) == 2
    assert "PAYMENT-SIGNATURE" not in server.requests[0].headers
    assert server.requests[1].url.params["q"] == "senate"

    header = server.requests[1].headers["PAYMENT-SIGNATURE"]
    assert server.requests[1].headers["X-PAYMENT"] == header
    assert not set("+/=") & set(header)
    assert header == header.rstrip("=")

    payload = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
    assert payload["x402Version"] == 2
    assert payload["accepted"] == requirement
    assert payload["resource"] == {"url": arb.DEFAULT_URL}
    auth = payload["payload"]["authorization"]
    assert auth["from"] == signer.address
    assert auth["to"] == PAY_TO
    assert auth["value"] == "20000"
    assert all(isinstance(auth[key], str) for key in auth)
    assert int(auth["validBefore"]) - int(auth["validAfter"]) == 660
    assert auth["nonce"].startswith("0x") and len(auth["nonce"]) == 66

    message = {
        "from": auth["from"],
        "to": auth["to"],
        "value": int(auth["value"]),
        "validAfter": int(auth["validAfter"]),
        "validBefore": int(auth["validBefore"]),
        "nonce": bytes.fromhex(auth["nonce"][2:]),
    }
    typed = arb.build_typed_data(requirement, message)
    assert typed["domain"]["chainId"] == 8453
    recovered = Account.recover_message(
        encode_typed_data(full_message=typed), signature=payload["payload"]["signature"]
    )
    assert recovered == signer.address


@pytest.mark.asyncio
async def test_second_402_raises_payment_failed_after_one_retry(signer):
    server = FakeServer(_required([_requirement()]), always_402=True)
    with pytest.raises(arb.PaymentFailed):
        await _source(signer, server).fetch_async()
    assert len(server.requests) == 2


@pytest.mark.asyncio
async def test_first_acceptable_requirement_is_chosen(signer):
    good = _requirement(amount="10000")
    server = FakeServer(_required([_requirement(network="eip155:1"), good]))
    await _source(signer, server).fetch_async()
    header = server.requests[1].headers["PAYMENT-SIGNATURE"]
    assert arb.decode_b64_json(header)["accepted"] == good


# ---------------------------------------------------------------------------
# Refusals (checked before anything is signed)
# ---------------------------------------------------------------------------


class RecordingSigner:
    address = "0x" + "11" * 20

    def __init__(self):
        self.calls = 0

    def sign_typed_data(self, typed_data):
        self.calls += 1
        return "0x" + "00" * 65


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accepts, needle",
    [
        ([_requirement(network="eip155:1")], "not Base"),
        ([_requirement(network="solana")], "not Base"),
        ([_requirement(scheme="upto")], "not exact"),
        ([_requirement(asset="0x" + "cd" * 20)], "not Base USDC"),
        ([_requirement(amount="20001")], "exceeds cap"),
        ([_requirement(amount="1000000")], "exceeds cap"),
        ([_requirement(amount="0")], "must be positive"),
        ([_requirement(amount="abc")], "invalid amount"),
        ([_requirement(payTo="0x123")], "invalid payTo"),
        ([_requirement(payTo="0x" + "zz" * 20)], "invalid payTo"),
        ([_requirement(payTo=None)], "invalid payTo"),
        ([], "no payment requirements"),
        (None, "no payment requirements"),
    ],
)
async def test_refusals(accepts, needle):
    signer = RecordingSigner()
    required = _required(accepts)
    if accepts is None:
        del required["accepts"]
    server = FakeServer(required)
    with pytest.raises(arb.PaymentRefused, match=needle):
        await _source(signer, server).fetch_async()
    assert signer.calls == 0
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_lower_cap_refuses_default_price():
    server = FakeServer(_required([_requirement()]))
    with pytest.raises(arb.PaymentRefused, match="exceeds cap 10000"):
        await _source(RecordingSigner(), server, max_usd_per_call=0.01).fetch_async()


@pytest.mark.asyncio
async def test_402_without_requirements_is_refused():
    def handler(request):
        return httpx.Response(402, text="pay up")

    with pytest.raises(arb.PaymentRefused, match="without payment requirements"):
        await _source(RecordingSigner(), handler).fetch_async()


# ---------------------------------------------------------------------------
# Header codec
# ---------------------------------------------------------------------------


def test_encode_header_is_unpadded_base64url():
    payload = {"x402Version": 2, "blob": "\xff\xfe>>>???" * 7}
    header = arb.encode_payment_header(payload)
    assert not set("+/=") & set(header)
    assert arb.decode_b64_json(header) == payload


@pytest.mark.parametrize("urlsafe", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_decode_is_tolerant(urlsafe, pad):
    obj = {"accepts": [{"x": "\xff\xfe>>>???"}], "n": 1}
    assert arb.decode_b64_json(_b64(obj, urlsafe=urlsafe, pad=pad)) == obj


@pytest.mark.parametrize("value", [None, "", "!!!not-base64!!!", _b64("x")[:-3] + "@@"])
def test_decode_garbage_returns_none(value):
    assert arb.decode_b64_json(value) is None
