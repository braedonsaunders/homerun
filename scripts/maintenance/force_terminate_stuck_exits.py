"""Force-terminate stuck pending exits that are poisoning reconciliation.

Background: a TraderOrder whose ``payload_json.pending_live_exit`` has
``retry_count >= _FAILED_EXIT_MAX_RETRIES`` (5) and a stale
``last_attempt_at`` timestamp can keep the position lifecycle code re-
entering the same exit path every reconciliation cycle.  When the
underlying market has resolved or the order book is dry, every retry
times out and ``Live reconciliation timed out for trader=... 30s``
fires repeatedly — one trader's stuck row holds up its entire
reconcile cycle.

This script lists candidates and lets the operator force-mark them
terminal so the lifecycle path stops touching them.  It is read-only
unless ``--apply`` is passed.

Usage (from repo root, with the worker venv active):

    # List every candidate for a trader
    python -m scripts.maintenance.force_terminate_stuck_exits \\
        --trader-id fb1e2fc1e6bb47fbb5dd199dafc671d2

    # Apply the change for a specific order (preferred — surgical)
    python -m scripts.maintenance.force_terminate_stuck_exits \\
        --trader-id fb1e2fc1e6bb47fbb5dd199dafc671d2 \\
        --order-id 07f3f733-... \\
        --apply

    # Apply across all candidates for the trader (use sparingly)
    python -m scripts.maintenance.force_terminate_stuck_exits \\
        --trader-id fb1e2fc1e6bb47fbb5dd199dafc671d2 \\
        --apply-all

Effect: sets ``status='failed'`` and stamps a ``manual_terminate`` block
in ``payload.pending_live_exit`` so the audit trail records why the row
was force-marked.  The order's ``actual_profit`` is NOT touched — that
stays whatever the verifier last computed.  If the position later
turns out to have been fillable after all, run
``rerecompute_phantom_resolved_pnl`` or operator_writeoff to correct
the P&L.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


_STUCK_ACTIVE_STATUSES = (
    "submitted", "executed", "completed", "open", "pending", "placing", "queued"
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_pending_exit(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a candidate descriptor if the row has a stuck exit, else None."""
    pending_exit = payload.get("pending_live_exit")
    if not isinstance(pending_exit, dict):
        return None
    retry_count = int(pending_exit.get("retry_count") or 0)
    status = str(pending_exit.get("status") or "").strip().lower()
    last_error = pending_exit.get("last_error") or ""
    last_attempt = pending_exit.get("last_attempt_at") or pending_exit.get("triggered_at")
    if status not in {"pending", "submitted", "failed", "blocked_min_notional"}:
        return None
    if retry_count < 5:
        return None
    return {
        "retry_count": retry_count,
        "status": status,
        "last_error": str(last_error)[:140],
        "last_attempt": last_attempt,
        "next_retry_at": pending_exit.get("next_retry_at"),
    }


async def _fetch_candidates(trader_id: str) -> list[dict[str, Any]]:
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, TraderOrder

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(TraderOrder)
                .where(TraderOrder.trader_id == trader_id)
                .where(TraderOrder.mode == "live")
                .where(TraderOrder.status.in_(_STUCK_ACTIVE_STATUSES))
            )
        ).scalars().all()

    candidates: list[dict[str, Any]] = []
    for row in rows:
        payload = row.payload_json if isinstance(row.payload_json, dict) else {}
        descriptor = _classify_pending_exit(payload)
        if descriptor is None:
            continue
        candidates.append(
            {
                "order_id": str(row.id),
                "market_id": str(row.market_id or ""),
                "token_id": (
                    (payload.get("leg") or {}).get("token_id")
                    if isinstance(payload.get("leg"), dict)
                    else None
                ),
                "status": str(row.status),
                "market_question": str(row.market_question or "")[:80],
                **descriptor,
            }
        )
    return candidates


async def _apply_force_terminate(
    trader_id: str,
    order_ids: list[str],
    operator_note: str,
) -> int:
    from models.database import AsyncSessionLocal, TraderOrder

    if not order_ids:
        return 0
    written = 0
    async with AsyncSessionLocal() as session:
        for order_id in order_ids:
            row = await session.get(TraderOrder, order_id)
            if row is None or str(row.trader_id) != trader_id:
                continue
            payload = dict(row.payload_json or {})
            pending_exit = dict(payload.get("pending_live_exit") or {})
            pending_exit["status"] = "manual_terminate"
            pending_exit["manual_terminate"] = {
                "applied_at": _utcnow_iso(),
                "applied_by": "force_terminate_stuck_exits.py",
                "prior_retry_count": int(pending_exit.get("retry_count") or 0),
                "prior_last_error": str(pending_exit.get("last_error") or "")[:200],
                "operator_note": operator_note,
            }
            payload["pending_live_exit"] = pending_exit
            row.payload_json = payload
            row.status = "failed"
            row.error_message = (
                f"force_terminate_stuck_exits: retry_count="
                f"{pending_exit['manual_terminate']['prior_retry_count']} "
                f"({operator_note})"
            )[:255]
            written += 1
        await session.commit()
    return written


async def _main(args: argparse.Namespace) -> int:
    candidates = await _fetch_candidates(args.trader_id)
    print(f"trader_id={args.trader_id}  candidates={len(candidates)}")
    if not candidates:
        return 0
    print()
    print(f"{'order_id':>36} | {'retries':>7} | {'status':>10} | {'last_attempt':>20} | last_error")
    print("-" * 130)
    for c in candidates:
        print(
            f"{c['order_id'][:36]:>36} | "
            f"{c['retry_count']:>7} | "
            f"{c['status']:>10} | "
            f"{(c['last_attempt'] or '')[:20]:>20} | "
            f"{c['last_error']}"
        )

    if args.order_id:
        if not any(c["order_id"] == args.order_id for c in candidates):
            print(f"\norder_id={args.order_id} is not in the candidate set; refusing.")
            return 1
        target_ids = [args.order_id]
    elif args.apply_all:
        target_ids = [c["order_id"] for c in candidates]
    else:
        print("\n[dry-run] Pass --order-id ... --apply OR --apply-all to actually mark these terminal.")
        return 0

    if not (args.apply or args.apply_all):
        print(f"\n[dry-run] Would terminate {len(target_ids)} order(s).  Re-run with --apply.")
        return 0

    written = await _apply_force_terminate(
        args.trader_id, target_ids, operator_note=args.note or "stuck_exit_unblock"
    )
    print(f"\nMarked {written} order(s) as failed/manual_terminate.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trader-id", required=True)
    parser.add_argument(
        "--order-id",
        help="Surgical: terminate this single order (preferred over --apply-all).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the change (use with --order-id).",
    )
    parser.add_argument(
        "--apply-all",
        action="store_true",
        help="Apply to every candidate.  Use sparingly.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Free-form audit note attached to manual_terminate.operator_note.",
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
