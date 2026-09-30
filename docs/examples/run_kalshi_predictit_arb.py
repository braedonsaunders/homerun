"""Run the kalshi-predictit-arb data source through homerun's own loader.

Offline by default: with no ``X402_WALLET_KEY`` set this makes no network calls
and prints fictional SAMPLE records (NOT real market data)::

    cd backend && python ../docs/examples/run_kalshi_predictit_arb.py
    python ../docs/examples/run_kalshi_predictit_arb.py --mode all --q senate

Live mode is opt-in and paid ($0.02 USDC per call on Base, via x402): set
``X402_WALLET_KEY`` to a dedicated low-balance wallet key and install
``eth-account``. Payments above ``--max-usd-per-call`` (default 0.02) are refused.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parents[1] / "backend"
for _path in (str(_BACKEND), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from models.database import DataSource  # noqa: E402
from services.data_source_loader import data_source_loader, validate_data_source_source  # noqa: E402
from services.data_source_runner import preview_data_source  # noqa: E402

SLUG = "kalshi_predictit_arb"
SOURCE_FILE = _HERE / "kalshi_predictit_arb_source.py"


async def run(config: dict) -> dict:
    source_code = SOURCE_FILE.read_text(encoding="utf-8")
    validation = validate_data_source_source(source_code)
    if not validation["valid"]:
        raise SystemExit("Validation failed: " + "; ".join(validation["errors"]))

    runtime = data_source_loader.load(slug=SLUG, source_code=source_code, config=config)
    try:
        if os.environ.get("X402_WALLET_KEY"):
            from kalshi_predictit_arb_signer import EthAccountSigner

            runtime.instance.payment_signer = EthAccountSigner()
            print(f"LIVE mode (paid, cap ${config['max_usd_per_call']}/call) as {runtime.instance.payment_signer}")
        else:
            print("DEMO mode: no X402_WALLET_KEY set, no network calls, fictional SAMPLE data.")
        source = DataSource(slug=SLUG, source_code=source_code, config=config, class_name=validation["class_name"])
        return await preview_data_source(source, max_records=25)
    finally:
        data_source_loader.unload(SLUG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q", default="", help="keyword filter")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", choices=("opportunities", "all"), default="opportunities")
    parser.add_argument("--max-usd-per-call", type=float, default=0.02)
    parser.add_argument("--json", action="store_true", help="print the full preview as JSON")
    args = parser.parse_args(argv)

    config = {"q": args.q, "limit": args.limit, "mode": args.mode, "max_usd_per_call": args.max_usd_per_call}
    preview = asyncio.run(run(config))
    if args.json:
        print(json.dumps(preview, indent=2, default=str))
    else:
        for record in preview["records"]:
            print(f"- {record['title']}\n    {record['summary']}\n    tags={record['tags']}")
        print(f"{len(preview['records'])} record(s)")
    print(
        "\nTo register in homerun: Data Sources UI -> New (python), paste kalshi_predictit_arb_source.py, "
        f"slug '{SLUG}'; or StrategySDK.create_data_source(slug='{SLUG}', source_key='custom', "
        "source_kind='python', source_code=...). Strategies read it via "
        f"DataSourceSDK.get_records(source_slug='{SLUG}')."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
