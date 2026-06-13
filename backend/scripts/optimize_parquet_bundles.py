r"""Rewrite bundled parquet files with canonical row-group sizing.

Run from the repository root or backend directory:

    python backend/scripts/optimize_parquet_bundles.py --root D:\homerundata --workers 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.external_data.parquet_compactor import optimize_bundle_row_groups  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Rewrite parquet bundles whose row groups are smaller than the canonical target."
    )
    parser.add_argument("--root", required=True, help="Parquet root containing bundled window directories.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of bundle optimization workers.",
    )
    args = parser.parse_args()

    result = optimize_bundle_row_groups(Path(args.root), workers=max(1, int(args.workers)))
    payload = asdict(result)
    payload["root"] = str(result.root)
    print(json.dumps(payload, indent=2))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
