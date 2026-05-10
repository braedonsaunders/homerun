"""One-shot Plan 0035 simulation against Plan 0033 evidence.

Replays the 33 cancelled `Sandbox - Tail-End` orders from
`docs/plans/work-artifacts/0033-book-snapshot-join.csv` through the
post-fix `_chase_up_execution_caps` reducer and reports how many would
have crossed their book's best ask once the entry-band conflation is
removed.

Each row in the CSV is a tail-end carry leg with the strategy's
`target_price` materialised as `ctx_max_entry_price` (per
`tail_end_carry.py:809-810`, `max_execution_price = max_entry_price =
target_price` when chase-up is enabled). Pre-fix, the chase-up reducer
collapsed `shadow_limit_price` to `min(target_price, max_probability,
derived(min_upside_percent))` which on this bot was always the
`max_probability=0.905` cap. Post-fix, only the execution-price caps
remain, so `shadow_limit_price = target_price = ctx_max_entry_price`.

Recovered = `post_fix_shadow_limit >= book_best_ask` AND a real book
snapshot is present. The rows without a snapshot
(`verdict=no_book_snapshot`) are excluded from the denominator —
they're indeterminate either way.

Deletable after the Plan 0035 verdict lands in
`docs/operational/runtime-tweaks.md`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from services.trader_orchestrator.order_manager import _chase_up_execution_caps  # noqa: E402

CSV_PATH = REPO_ROOT / "docs" / "plans" / "work-artifacts" / "0033-book-snapshot-join.csv"
BOT_MAX_PROBABILITY = 0.905
BOT_MIN_UPSIDE_PERCENT = 6.0


def _post_fix_shadow_limit(*, signal_price: float, target_price: float) -> float:
    """Mirror the post-fix shadow chase-up branch in `submit_execution_leg`.

    The leg dict written by `tail_end_carry.py:804-815` carries
    `max_execution_price = max_entry_price = target_price` when
    chase-up is enabled.  `_chase_up_execution_caps` returns
    `[target_price, target_price]`; their min is `target_price`.  The
    chase-up branch then lifts `shadow_limit_price` from the signal
    price to that cap (or 1.0 if no cap survived the filter).
    """
    leg = {
        "max_execution_price": target_price,
        "max_entry_price": target_price,
    }
    params = {
        "max_probability": BOT_MAX_PROBABILITY,
        "min_upside_percent": BOT_MIN_UPSIDE_PERCENT,
    }
    caps = _chase_up_execution_caps(leg=leg, metadata={}, params=params)
    tightest = min(caps, default=None)
    initial = float(signal_price)
    if tightest is None:
        return 1.0
    if tightest > initial:
        return float(tightest)
    return initial


def main() -> int:
    if not CSV_PATH.exists():
        print(f"Missing: {CSV_PATH}", file=sys.stderr)
        return 2

    bucket_a_total = 0
    bucket_a_recovered = 0
    bucket_c_total = 0
    bucket_c_recovered = 0
    indeterminate = 0

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict = row["verdict"].strip()
            signal_price = float(row["leg_signal_price"])
            target_price = float(row["ctx_max_entry_price"])
            best_ask_raw = row["book_best_ask"].strip()

            if verdict == "no_book_snapshot" or not best_ask_raw:
                indeterminate += 1
                continue

            best_ask = float(best_ask_raw)
            post_fix_limit = _post_fix_shadow_limit(
                signal_price=signal_price,
                target_price=target_price,
            )
            crosses = post_fix_limit >= best_ask

            if verdict == "config_gated_chase_would_help":
                bucket_a_total += 1
                if crosses:
                    bucket_a_recovered += 1
            elif verdict == "book_above_chase_cap":
                bucket_c_total += 1
                if crosses:
                    bucket_c_recovered += 1
            else:
                print(f"Unknown verdict: {verdict!r}", file=sys.stderr)
                return 2

    total_evidenced = bucket_a_total + bucket_c_total
    print(f"Plan 0035 cap-split simulation against Plan 0033 evidence")
    print(f"  Bucket A (config-gated):      {bucket_a_recovered:>2} / {bucket_a_total:>2} recovered")
    print(f"  Bucket C (book above target): {bucket_c_recovered:>2} / {bucket_c_total:>2} recovered")
    print(f"  Indeterminate (no snapshot):  {indeterminate:>2}")
    print(f"  Total evidenced:              {bucket_a_recovered + bucket_c_recovered:>2} / {total_evidenced:>2} recovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
