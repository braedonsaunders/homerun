# kalshi-predictit-arb data source (example)

A homerun **data source** (`BaseDataSource`) that surfaces an external
Kalshi <-> PredictIt cross-venue arbitrage scanner as data-source records, so
strategies can read it with `DataSourceSDK.get_records(source_slug="kalshi_predictit_arb")`.

The scanner entity-resolves Kalshi markets against PredictIt contracts
(state/district/office/party/strike/cycle gating, party-mismatch hard reject),
prices both arb directions after fees (Kalshi taker `ceil(7*P*(1-P))` cents
per leg; PredictIt 10% of winning-leg profit plus 5% withdrawal) and reports the
worst-case net across settlement outcomes. `executable` is true only at >= 1c
net. Gaps are indicative until checked against live depth, fees and resolution
equivalence.

**This is a docs example, not a core service.** Nothing in `backend/` imports
it. Homerun has no PredictIt venue, so records are signals only; it complements
(does not replace) the built-in Polymarket <-> Kalshi `cross_platform` strategy.

Disclosure: `kalshi-predictit-arb` is built and operated by Team Takatini.
The live feed is a **paid** third-party API ($0.02 USDC per call on Base, via
x402). Docs: https://mastertyrone.github.io/kalshi-predictit-arb/. Client: https://github.com/mastertyrone/kalshi-predictit-arb

## Files

| File | Purpose |
| --- | --- |
| `kalshi_predictit_arb_source.py` | The data source. Passes homerun's data-source validator; paste into the Data Sources UI or pass to `StrategySDK.create_data_source`. |
| `kalshi_predictit_arb_signer.py` | Optional EIP-712 signer for live mode (host code, needs `eth-account`). |
| `run_kalshi_predictit_arb.py` | Runnable example using homerun's own loader and `preview_data_source`. |
| `kalshi_predictit_arb_sample.json` | Fictional SAMPLE fixture (NOT real market data) used by demo mode. |
| `test_kalshi_predictit_arb_source.py` | Tests (no network, no real keys). |

## Offline by default

With no payment signer attached the source makes **no network calls** and
returns fictional SAMPLE records (titles prefixed `[SAMPLE DATA]`, tag
`sample`). Inside homerun's data-source sandbox (which blocks `os`, `open` and
`eth_account`) it therefore always runs in demo mode.

```bash
cd backend
python ../docs/examples/run_kalshi_predictit_arb.py
python ../docs/examples/run_kalshi_predictit_arb.py --mode all --q senate --json
```

## Register in homerun

Data Sources UI -> new Python source -> paste `kalshi_predictit_arb_source.py`,
slug `kalshi_predictit_arb`. Or:

```python
await StrategySDK.create_data_source(
    slug="kalshi_predictit_arb",
    source_key="custom",
    source_kind="python",
    source_code=source_code,
    config={"mode": "opportunities", "limit": 10, "q": ""},
)
```

Record shape: `category="cross_venue_arb"`, `source="kalshi-predictit-arb"`,
`tags=["kalshi", "predictit", "arbitrage"]` (+ `"sample"` in demo mode), and
`pair`, `best_direction`, `net_yield_c`, `net_yield_pct`, `executable`,
`sample_data` in the record payload.

## Live mode (paid, opt-in)

Live mode only happens when host code attaches a signer to the loaded
instance; the example does this only if `X402_WALLET_KEY` is set:

```bash
pip install eth-account   # or: pip install -r backend/requirements-trading.txt
X402_WALLET_KEY=0x...  python ../docs/examples/run_kalshi_predictit_arb.py
```

Use a dedicated low-balance wallet. Payment requirements are refused unless
they are Base (`eip155:8453`) / scheme `exact` / Base USDC
(`0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`) / `0 < amount <= max_usd_per_call`
(default `0.02`) with a valid `payTo`; an empty `accepts` list is refused. The
signed EIP-3009 authorization is sent once, as unpadded base64url, in both
`PAYMENT-SIGNATURE` and `X-PAYMENT`; a second 402 raises `PaymentFailed`.
Incoming `PAYMENT-REQUIRED` / `PAYMENT-RESPONSE` headers are decoded tolerantly
(standard or urlsafe base64, padded or not).

## Run the tests

```bash
pip install -r backend/requirements.txt pytest pytest-asyncio eth-account
cd backend && python -m pytest ../docs/examples/test_kalshi_predictit_arb_source.py
```

Signing tests skip when `eth-account` is not installed; they use a throwaway
`Account.create()` key and `httpx.MockTransport`, never the real endpoint.
