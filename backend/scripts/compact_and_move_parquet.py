r"""Compact canonical parquet market data and move a parquet root.

Run from the repository root or backend directory:

    python backend/scripts/compact_and_move_parquet.py --source C:\homerun\data\parquet --dest D:\homerundata
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

from services.external_data.parquet_compactor import migrate_parquet_root  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Compact canonical book parquet into bundles and move a parquet root."
    )
    parser.add_argument("--source", required=True, help="Existing parquet root.")
    parser.add_argument("--dest", required=True, help="Destination parquet root.")
    parser.add_argument(
        "--keep-sources",
        action="store_true",
        help="Copy/compact without deleting verified source files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of window compaction workers. Use 1 for strictly sequential migration.",
    )
    args = parser.parse_args()

    result = migrate_parquet_root(
        Path(args.source),
        Path(args.dest),
        delete_sources=not bool(args.keep_sources),
        workers=max(1, int(args.workers)),
    )
    payload = asdict(result)
    payload["source_root"] = str(result.source_root)
    payload["dest_root"] = str(result.dest_root)
    print(json.dumps(payload, indent=2))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
