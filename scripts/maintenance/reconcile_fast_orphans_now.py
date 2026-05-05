"""One-shot trigger for ``reconcile_orphaned_fast_submissions``.

Run when the in-startup sweep didn't clear orphan ``in_flight`` /
``clob_exception`` / ``post_update_failed`` skeleton rows for a fast
trader, leaving the trader's ``max_open_orders`` cap occupied.

Usage (from repo root, with the worker venv active):

    python -m scripts.maintenance.reconcile_fast_orphans_now \
        --trader-id fb1e2fc1e6bb47fbb5dd199dafc671d2

Flags:
  --trader-id TRADER_ID   Required. Run reconcile for this trader.
  --min-age-seconds N     Optional. Default 30; matches the runtime sweep.
  --dry-run               List eligible rows but skip the venue call + update.

Output: prints the structured result returned by the reconcile function.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


async def _main(trader_id: str, min_age_seconds: float, dry_run: bool) -> int:
    from models.database import AsyncSessionLocal, TraderOrder  # noqa: E402
    from sqlalchemy import select  # noqa: E402
    from services.trader_orchestrator_state import (  # noqa: E402
        reconcile_orphaned_fast_submissions,
    )

    if dry_run:
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(
                        TraderOrder.id,
                        TraderOrder.status,
                        TraderOrder.market_id,
                        TraderOrder.created_at,
                        TraderOrder.payload_json,
                    )
                    .where(TraderOrder.trader_id == trader_id)
                    .where(TraderOrder.mode == "live")
                    .where(TraderOrder.provider_clob_order_id.is_(None))
                )
            ).all()

        eligible: list[dict[str, Any]] = []
        for row in rows:
            payload = row.payload_json if isinstance(row.payload_json, dict) else None
            if not payload or not payload.get("fast_tier"):
                continue
            state = str(payload.get("fast_submission_state") or "").strip()
            if state not in {"in_flight", "clob_exception", "post_update_failed"}:
                continue
            eligible.append(
                {
                    "id": str(row.id),
                    "status": row.status,
                    "market_id": row.market_id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "fast_submission_state": state,
                    "fast_idempotency_key": payload.get("fast_idempotency_key"),
                }
            )
        print(json.dumps({"dry_run": True, "eligible_rows": eligible}, indent=2, default=str))
        return 0

    async with AsyncSessionLocal() as session:
        result = await reconcile_orphaned_fast_submissions(
            session,
            trader_id=trader_id,
            min_age_seconds=min_age_seconds,
            commit=True,
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trader-id", required=True)
    parser.add_argument("--min-age-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_main(args.trader_id, args.min_age_seconds, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
