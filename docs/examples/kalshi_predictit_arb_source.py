"""Data Source: kalshi-predictit-arb (Kalshi <-> PredictIt cross-venue arbitrage feed).

A ``BaseDataSource`` that surfaces an external cross-venue scanner as data-source
records. The scanner entity-resolves Kalshi markets against PredictIt contracts
(state/district/office/party/strike/cycle gating, party-mismatch hard reject),
prices both arb directions after fees (Kalshi taker ``ceil(7*P*(1-P))`` cents per
leg; PredictIt 10% of winning-leg profit plus 5% withdrawal) and reports the
worst-case net across settlement outcomes; ``executable`` is true only at >= 1c
net. Gaps are indicative until checked against live depth, fees and resolution
equivalence. Homerun has no PredictIt venue, so records are signals only.

Offline by default
------------------

With no payment signer attached the source makes **no network calls** and
returns ``SAMPLE_RESPONSE`` below: fictional pairs and prices, NOT real market
data (the same content as ``kalshi_predictit_arb_sample.json``).

This file passes homerun's data-source validator (``services.data_source_loader``),
so it can be pasted into the Data Sources UI or created with
``StrategySDK.create_data_source(slug="kalshi_predictit_arb", source_kind="python",
source_code=...)``. That sandbox blocks ``os``/``open``/``eth_account`` imports,
so inside it the source runs in demo mode only.

Live mode (paid, opt-in)
------------------------

The live feed is a paid third-party API: $0.02 USDC per call via x402 v2
("exact" scheme, USDC on Base). It is used only when host code attaches a
signer (see ``kalshi_predictit_arb_signer.py``). Payment requirements are
refused unless they are Base / exact / Base USDC / 0 < amount <= the
``max_usd_per_call`` cap (default 0.02) with a valid ``payTo``. The signed
authorization is sent once, as unpadded base64url, in ``PAYMENT-SIGNATURE`` and
``X-PAYMENT``; a second 402 raises ``PaymentFailed``.

Disclosure: ``kalshi-predictit-arb`` is built and operated by Team Takatini.
"""

from __future__ import annotations

import base64
import json
import random
import time
from decimal import Decimal
from typing import Any

import httpx

from services.data_source_sdk import BaseDataSource

DEFAULT_URL = "https://x402.bankr.bot/0x69fb671637ed68881f66b9ebf305ec3ef5574f65/kalshi-predictit-arb"
BASE_NETWORKS = ("eip155:8453", "base")
BASE_CHAIN_ID = 8453
BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_DECIMALS = 6
MODES = ("opportunities", "all")
SOURCE = "kalshi-predictit-arb"

# Fictional SAMPLE data, NOT real market data. Kept identical to
# kalshi_predictit_arb_sample.json (the sandbox blocks open()).
SAMPLE_RESPONSE: dict[str, Any] = {
    "_notice": "SAMPLE DATA for offline demo/tests only. Fictional pairs and prices, NOT real market data.",
    "q": "",
    "mode": "all",
    "opportunities": [
        {
            "event": "SAMPLE State XX Senate winner? - Democratic",
            "match_score": 0.74,
            "kalshi": {"ticker": "SAMPLE-SENATEXX-26-D", "yes_bid_c": 59, "yes_ask_c": 60},
            "predictit": {
                "market": "SAMPLE Which party will win the XX Senate race?",
                "contract": "Democratic",
                "yes_ask_c": 68,
                "no_ask_c": 31,
                "position_limit_usd": 850,
            },
            "best_direction": {
                "direction": "YES_KALSHI__NO_PREDICTIT",
                "kalshi_price_c": 60,
                "predictit_price_c": 31,
                "stake_c": 91,
                "raw_spread_c": 9,
                "net_if_yes_wins_c": 3.1,
                "net_if_no_wins_c": 5.2,
                "net_yield_c": 3.1,
                "net_yield_pct": 3.4,
                "fees": {"kalshi_taker_c": 2, "predictit_profit_fee_c": 6.9, "predictit_withdrawal_drag_c": 3.45},
            },
            "days_to_settlement": 120,
            "roc_annualized_pct": 10.3,
            "executable": True,
            "liquidity": {"vwap_buy_c": {"100": 60, "500": 60, "1000": 61}, "book_depth_usd": 12000.0},
            "slippage_ok_100": True,
        },
        {
            "event": "SAMPLE District YY-01 House winner? - Republican",
            "match_score": 0.69,
            "kalshi": {"ticker": "SAMPLE-HOUSEYY01-26-R", "yes_bid_c": 44, "yes_ask_c": 45},
            "predictit": {
                "market": "SAMPLE YY-01 House race",
                "contract": "Republican",
                "yes_ask_c": 47,
                "no_ask_c": 47,
                "position_limit_usd": 850,
            },
            "best_direction": {
                "direction": "YES_KALSHI__NO_PREDICTIT",
                "kalshi_price_c": 45,
                "predictit_price_c": 47,
                "stake_c": 92,
                "raw_spread_c": 8,
                "net_if_yes_wins_c": 1.2,
                "net_if_no_wins_c": 2.4,
                "net_yield_c": 1.2,
                "net_yield_pct": 1.3,
                "fees": {"kalshi_taker_c": 2, "predictit_profit_fee_c": 5.3, "predictit_withdrawal_drag_c": 2.65},
            },
            "days_to_settlement": 400,
            "roc_annualized_pct": 1.2,
            "executable": True,
            "liquidity": {"vwap_buy_c": {"100": 45, "500": 46, "1000": 46}, "book_depth_usd": 4000.0},
            "slippage_ok_100": True,
        },
        {
            "event": "SAMPLE State ZZ governor party winner? - Democratic",
            "match_score": 0.66,
            "kalshi": {"ticker": "SAMPLE-GOVPARTYZZ-26-D", "yes_bid_c": 38, "yes_ask_c": 39},
            "predictit": {
                "market": "SAMPLE ZZ Governor race",
                "contract": "Democratic",
                "yes_ask_c": 41,
                "no_ask_c": 41,
                "position_limit_usd": 850,
            },
            "best_direction": {
                "direction": "YES_KALSHI__NO_PREDICTIT",
                "kalshi_price_c": 39,
                "predictit_price_c": 41,
                "stake_c": 80,
                "raw_spread_c": 20,
                "net_if_yes_wins_c": 0.4,
                "net_if_no_wins_c": 1.1,
                "net_yield_c": 0.4,
                "net_yield_pct": 0.5,
                "fees": {"kalshi_taker_c": 2, "predictit_profit_fee_c": 5.9, "predictit_withdrawal_drag_c": 2.95},
            },
            "days_to_settlement": 400,
            "roc_annualized_pct": 0.5,
            "executable": False,
            "liquidity": {"vwap_buy_c": {"100": 39, "500": 40, "1000": 40}, "book_depth_usd": 2500.0},
            "slippage_ok_100": True,
        },
        {
            "event": "SAMPLE State WW Senate winner? - Republican",
            "match_score": 0.71,
            "kalshi": {"ticker": "SAMPLE-SENATEWW-26-R", "yes_bid_c": 80, "yes_ask_c": 81},
            "predictit": {
                "market": "SAMPLE WW Senate race",
                "contract": "Republican",
                "yes_ask_c": 83,
                "no_ask_c": 10,
                "position_limit_usd": 850,
            },
            "best_direction": {
                "direction": "YES_KALSHI__NO_PREDICTIT",
                "kalshi_price_c": 81,
                "predictit_price_c": 10,
                "stake_c": 91,
                "raw_spread_c": 9,
                "net_if_yes_wins_c": -2.0,
                "net_if_no_wins_c": 4.3,
                "net_yield_c": -2.0,
                "net_yield_pct": -2.2,
                "fees": {"kalshi_taker_c": 2, "predictit_profit_fee_c": 8.1, "predictit_withdrawal_drag_c": 4.05},
            },
            "days_to_settlement": 400,
            "roc_annualized_pct": -2.0,
            "executable": False,
            "liquidity": {"vwap_buy_c": {"100": 81, "500": 82, "1000": 82}, "book_depth_usd": 2500.0},
            "slippage_ok_100": False,
        },
        {"event": "SAMPLE record with missing fields (client must tolerate this)"},
    ],
    "stats": {
        "pairs_evaluated": 5,
        "executable": 2,
        "kalshi_markets_scanned": 5,
        "predictit_contracts_scanned": 5,
        "duration_ms": 0,
    },
    "fetched_at": "2026-01-01T00:00:00.000Z",
}

TRANSFER_WITH_AUTHORIZATION_TYPES: dict[str, list[dict[str, str]]] = {
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ]
}


class PaymentRefused(Exception):
    """No acceptable payment requirement (network, scheme, asset, amount, cap, payTo)."""


class PaymentFailed(Exception):
    """A payment was sent but the server still answered 402."""


# ---------------------------------------------------------------------------
# Response helpers (tolerant of missing / unknown fields)
# ---------------------------------------------------------------------------


def normalise_response(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    opportunities = data.get("opportunities")
    if not isinstance(opportunities, list):
        opportunities = []
    data["opportunities"] = [row for row in opportunities if isinstance(row, dict)]
    return data


def net_yield(opportunity: dict[str, Any], key: str = "net_yield_c") -> float | None:
    best = opportunity.get("best_direction")
    if not isinstance(best, dict):
        return None
    try:
        return float(best[key])
    except (KeyError, TypeError, ValueError):
        return None


def is_executable(opportunity: dict[str, Any]) -> bool:
    return opportunity.get("executable") is True


# ---------------------------------------------------------------------------
# x402 v2 payment helpers (no signing here: the signer is injected)
# ---------------------------------------------------------------------------


def encode_payment_header(payload: dict[str, Any]) -> str:
    """Unpadded base64url, as used by the live-tested client."""
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_b64_json(value: str | None) -> Any:
    """Tolerant decode: standard or urlsafe base64, padded or not."""
    if not value:
        return None
    try:
        text = str(value).strip().replace("-", "+").replace("_", "/")
        return json.loads(base64.b64decode(text + "=" * (-len(text) % 4), validate=True))
    except (ValueError, TypeError):
        return None


def parse_payment_required(response: httpx.Response) -> dict[str, Any]:
    required = decode_b64_json(response.headers.get("PAYMENT-REQUIRED"))
    if required is None:
        try:
            required = response.json()
        except ValueError:
            required = None
    if not isinstance(required, dict):
        raise PaymentRefused("402 response without payment requirements")
    return required


def _requirement_amount(requirement: dict[str, Any]) -> Any:
    # v2 uses 'amount', v1 used 'maxAmountRequired'
    return requirement.get("amount", requirement.get("maxAmountRequired"))


def refusal_reason(requirement: Any, max_atomic: int) -> str | None:
    if not isinstance(requirement, dict):
        return "malformed requirement"
    if requirement.get("network") not in BASE_NETWORKS:
        return f"network {requirement.get('network')!r} is not Base"
    if requirement.get("scheme") != "exact":
        return f"scheme {requirement.get('scheme')!r} is not exact"
    if str(requirement.get("asset", "")).lower() != BASE_USDC.lower():
        return f"asset {requirement.get('asset')!r} is not Base USDC"
    raw = _requirement_amount(requirement)
    try:
        amount = int(raw)
    except (TypeError, ValueError):
        return f"invalid amount {raw!r}"
    if amount <= 0:
        return f"amount {amount} must be positive"
    if amount > max_atomic:
        return f"amount {amount} exceeds cap {max_atomic}"
    pay_to = requirement.get("payTo")
    if not (isinstance(pay_to, str) and pay_to.startswith("0x") and len(pay_to) == 42):
        return f"invalid payTo {pay_to!r}"
    try:
        int(pay_to[2:], 16)
    except ValueError:
        return f"invalid payTo {pay_to!r}"
    return None


def select_requirement(accepts: Any, max_atomic: int) -> dict[str, Any]:
    if not accepts or not isinstance(accepts, list):
        raise PaymentRefused("402 response offered no payment requirements")
    reasons = []
    for requirement in accepts:
        reason = refusal_reason(requirement, max_atomic)
        if reason is None:
            return requirement
        reasons.append(reason)
    raise PaymentRefused("; ".join(reasons))


def build_authorization(payer: str, requirement: dict[str, Any], now: int | None = None) -> dict[str, Any]:
    now = int(time.time()) if now is None else int(now)
    return {
        "from": payer,
        "to": requirement["payTo"],
        "value": int(_requirement_amount(requirement)),
        "validAfter": now - 600,
        "validBefore": now + int(requirement.get("maxTimeoutSeconds") or 60),
        "nonce": random.SystemRandom().getrandbits(256).to_bytes(32, "big"),
    }


def build_typed_data(requirement: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    extra = requirement.get("extra") or {}
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            **TRANSFER_WITH_AUTHORIZATION_TYPES,
        },
        "primaryType": "TransferWithAuthorization",
        "domain": {
            "name": extra.get("name") or "USD Coin",
            "version": extra.get("version") or "2",
            "chainId": BASE_CHAIN_ID,
            "verifyingContract": requirement["asset"],
        },
        "message": authorization,
    }


def build_payment_payload(
    requirement: dict[str, Any],
    resource: Any,
    authorization: dict[str, Any],
    signature: str,
) -> dict[str, Any]:
    return {
        "x402Version": 2,
        "resource": resource,
        "accepted": requirement,
        "payload": {
            "signature": signature if str(signature).startswith("0x") else f"0x{signature}",
            "authorization": {
                "from": authorization["from"],
                "to": authorization["to"],
                "value": str(authorization["value"]),
                "validAfter": str(authorization["validAfter"]),
                "validBefore": str(authorization["validBefore"]),
                "nonce": "0x" + authorization["nonce"].hex(),
            },
        },
    }


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------


class KalshiPredictItArbSource(BaseDataSource):
    name = "Kalshi-PredictIt Arb (kalshi-predictit-arb)"
    description = (
        "Cross-venue Kalshi/PredictIt pairs priced after fees (worst case across outcomes). "
        "Offline SAMPLE data unless a payment signer is attached; live feed is paid ($0.02 USDC/call via x402)."
    )
    default_config = {
        "q": "",
        "limit": 10,
        "mode": "opportunities",
        "endpoint": DEFAULT_URL,
        "max_usd_per_call": 0.02,
        "timeout_seconds": 15.0,
    }

    # Host code (outside the DB sandbox) may attach a signer object with an
    # ``address`` attribute and ``sign_typed_data(typed_data) -> "0x..."``.
    # Without one the source stays in offline demo mode.
    payment_signer: Any = None
    # Test hook: an httpx transport; production uses the SDK's shared client.
    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self) -> None:
        super().__init__()
        self.last_payment_response: Any = None

    @property
    def demo(self) -> bool:
        return self.payment_signer is None

    def _max_atomic(self) -> int:
        cap = self.config.get("max_usd_per_call", 0.02)
        try:
            return int(Decimal(str(cap)) * 10**USDC_DECIMALS)
        except Exception:
            return int(Decimal("0.02") * 10**USDC_DECIMALS)

    def _params(self) -> dict[str, Any]:
        mode = str(self.config.get("mode") or "opportunities")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        params: dict[str, Any] = {"limit": self._as_int(self.config.get("limit"), 10, 1, 25), "mode": mode}
        q = str(self.config.get("q") or "").strip()
        if q:
            params["q"] = q
        return params

    def _scan_sample(self, params: dict[str, Any]) -> dict[str, Any]:
        data = normalise_response(json.loads(json.dumps(SAMPLE_RESPONSE)))
        rows = data["opportunities"]
        if params["mode"] == "opportunities":
            rows = [row for row in rows if is_executable(row)]
        words = str(params.get("q") or "").lower().split()
        if words:
            rows = [row for row in rows if all(word in json.dumps(row).lower() for word in words)]
        data["opportunities"] = rows[: params["limit"]]
        data["sample_data"] = True
        return data

    async def scan(self) -> dict[str, Any]:
        params = self._params()
        if self.demo:
            return self._scan_sample(params)
        if self.transport is not None:
            async with httpx.AsyncClient(transport=self.transport) as client:
                return await self._scan_live(client, params)
        return await self._scan_live(self.http_client, params)

    async def _scan_live(self, client: httpx.AsyncClient, params: dict[str, Any]) -> dict[str, Any]:
        url = str(self.config.get("endpoint") or DEFAULT_URL)
        timeout = self._as_float(self.config.get("timeout_seconds"), 15.0)
        response = await client.get(url, params=params, timeout=timeout)
        if response.status_code == 402:
            required = parse_payment_required(response)
            requirement = select_requirement(required.get("accepts"), self._max_atomic())
            authorization = build_authorization(str(self.payment_signer.address), requirement)
            signature = self.payment_signer.sign_typed_data(build_typed_data(requirement, authorization))
            payload = build_payment_payload(
                requirement, required.get("resource") or {"url": url}, authorization, signature
            )
            header = encode_payment_header(payload)
            response = await client.get(
                url,
                params=params,
                headers={"PAYMENT-SIGNATURE": header, "X-PAYMENT": header},
                timeout=timeout,
            )
            if response.status_code == 402:
                raise PaymentFailed(f"Payment not accepted: {response.text[:200]}")
            self.last_payment_response = decode_b64_json(response.headers.get("PAYMENT-RESPONSE"))
        response.raise_for_status()
        return normalise_response(response.json())

    async def fetch_async(self) -> list[dict[str, Any]]:
        response = await self.scan()
        sample = bool(response.get("sample_data"))
        observed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        records = []
        for row in response["opportunities"]:
            # Live feed rows carry "event", older rows "pair"; rows with neither are skipped.
            pair = str(row.get("event") or row.get("pair") or "").strip()
            if not pair:
                continue
            best = row.get("best_direction")
            yield_c = net_yield(row)
            yield_pct = net_yield(row, "net_yield_pct")
            prefix = "[SAMPLE DATA] " if sample else ""
            state = "executable" if is_executable(row) else "not executable"
            yield_text = "n/a" if yield_c is None else f"{yield_c:.2f}c"
            pct_text = "" if yield_pct is None else f" ({yield_pct:.2f}%)"
            records.append(
                {
                    "external_id": self._stable_hash(SOURCE, pair),
                    "title": f"{prefix}{pair}",
                    "summary": (
                        f"{prefix}Worst-case net {yield_text}{pct_text} after fees, {state}. "
                        "Indicative until checked against depth, fees and resolution."
                    ),
                    "category": "cross_venue_arb",
                    "source": SOURCE,
                    "url": None,
                    "observed_at": observed,
                    "tags": ["kalshi", "predictit", "arbitrage"] + (["sample"] if sample else []),
                    # Extra keys are kept in the record's payload_json (homerun convention).
                    "pair": pair,
                    "best_direction": best if isinstance(best, dict) else None,
                    "net_yield_c": yield_c,
                    "net_yield_pct": yield_pct,
                    "executable": is_executable(row),
                    "sample_data": sample,
                }
            )
        return records

    def transform(self, item: dict[str, Any]) -> dict[str, Any]:
        return item
