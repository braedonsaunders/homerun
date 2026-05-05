"""Last-line-of-defense pre-trade safety floors.

Background
----------
Strategies on this platform are loaded by ``exec``-ing
``Strategy.source_code`` from the database (see
:mod:`services.strategy_loader`).  That means a fix made in the .py
file under ``services/strategies/`` does NOT propagate to the running
process unless the corresponding ``Strategy.source_code`` row is
re-seeded — and the catalog explicitly does NOT overwrite existing
rows on startup (``ensure_system_opportunity_strategies_seeded``
preserves user edits).

The 2026-05-05 review uncovered a category of strategy-level safety
guards (e.g. ``min_entry_price`` floors on contrarian maker bets)
where the .py-level fix was real but invisible to the live system
because the DB-stored source_code lacked the gate.  Operators could
also disable the gate inadvertently from the UI.

This module enforces those floors at the **submit boundary** —
inside :func:`services.trader_orchestrator.order_manager.submit_execution_leg`,
right before the CLOB call — using a hard table that can only be
changed by deploying new code.  Strategies cannot loosen these floors
via config.  The result is defense-in-depth: even if a stale
strategy_versions row, a misconfigured UI override, or a bug in the
strategy's own gate logic lets a low-quality order through, this
layer rejects it before the venue ever sees the request.

Design
------
* Per-strategy entry-price floors and ceilings, keyed by strategy
  slug, with a default of "no floor" so unknown strategies are
  unaffected.
* Returns a structured :class:`SafetyAssessment` so the caller can
  log the decision, propagate it as an order ``error_message``, and
  surface it as a ``buy_pre_submit_gate``-style skip without disturbing
  cap accounting.
* No exceptions are raised; ``passed=False`` is the rejection signal.

Adding a new floor
------------------
Edit ``_STRATEGY_ENTRY_PRICE_FLOORS`` (or the ceiling map) below,
include a comment with the empirical justification + date, and add a
test in ``tests/test_execution_safety.py``.  Code review should
verify the floor matches the strategy's documented profitability
window.  Floors should be conservative — they're the line below
which the platform refuses to trade, regardless of operator config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Hard floors / ceilings (defense-in-depth tables)
# ---------------------------------------------------------------------------

# Per-strategy minimum BUY entry price. The strategy will be REFUSED
# at the submit boundary if a buy order targets this strategy with an
# entry price strictly below the listed value. Default behavior for
# strategies not listed here: no floor enforced.
#
# Each entry MUST cite the empirical justification + date. These are
# safety floors, not optimization targets — pick the value below which
# the strategy is structurally unprofitable, not the optimum.
_STRATEGY_ENTRY_PRICE_FLOORS: dict[str, float] = {
    # 2026-05-05 live data: bucket-by-bucket P&L for a single trading day
    # showed only the 0.80+ entry-price bucket was profitable
    # (5/0 wins, +$4.74). Below 0.50: 4/19 wins, -$58. Below 0.20: 0/3
    # wins, -$8. Asymmetric payoff (lose 100% on adverse resolution,
    # capture only the spread on wins) makes contrarian cheap-side
    # entries structurally a losing bet against market consensus.
    "crypto_entropy_maker": 0.80,
}

# Per-strategy maximum BUY entry price. Currently empty — left here
# as the symmetric extension point. Settlement-risk-driven ceilings
# (e.g. "never buy at >= 0.99 because Polymarket's redemption window
# can land you holding losing tokens") would live here.
_STRATEGY_ENTRY_PRICE_CEILINGS: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyAssessment:
    """Outcome of a pre-submit safety check.

    Attributes:
        passed: True iff the order may proceed to the venue.
        reason: Stable identifier for the violation (or ``"ok"``).
            Used as a status code by the caller for routing /
            telemetry; do not use for human-readable messages.
        message: Human-readable description suitable for log lines
            and ``trader_order.error_message``.
        floor: The numeric floor that was checked, when applicable.
        ceiling: The numeric ceiling that was checked, when applicable.
        observed: The value actually seen on the order, when applicable.
    """

    passed: bool
    reason: str
    message: str
    floor: Optional[float] = None
    ceiling: Optional[float] = None
    observed: Optional[float] = None


def assert_buy_entry_price_within_safety_bounds(
    *,
    strategy_slug: Optional[str],
    entry_price: Optional[float],
) -> SafetyAssessment:
    """Verify a BUY order's entry price clears the per-strategy hard floor.

    This is the canonical pre-submit safety check for entry-price
    bounds. It runs from
    :func:`services.trader_orchestrator.order_manager.submit_execution_leg`
    immediately before the CLOB call, regardless of which path
    constructed the order.

    Args:
        strategy_slug: Lower-cased strategy identifier. Unknown / empty
            slugs are treated as "no floor enforced" so the safety
            layer never blocks unrelated strategies.
        entry_price: The order's effective entry price (Polymarket
            scale: 0..1). ``None`` is treated as missing — the safety
            layer doesn't fabricate a violation when upstream code
            forgot to populate the field.

    Returns:
        :class:`SafetyAssessment` with ``passed=False`` when the floor
        or ceiling is violated; ``passed=True`` otherwise.

    Notes:
        * No exception is raised. Callers MUST inspect ``passed``.
        * The function is pure / synchronous / O(1).
        * Strategies CAN'T loosen the floor via UI config — these are
          hardcoded so they survive UI edits AND stale strategy_versions
          rows in the DB.
    """
    slug_norm = (strategy_slug or "").strip().lower()
    if not slug_norm:
        return SafetyAssessment(
            passed=True,
            reason="ok",
            message="No strategy slug supplied; safety floor not applicable.",
        )

    # ``None`` entry_price → upstream gate didn't populate the field.
    # Don't fabricate a violation; let the existing buy-gate / token-id
    # validation handle it. The safety layer is for orders the system
    # *thinks* are well-formed but which violate a hard policy floor.
    if entry_price is None:
        return SafetyAssessment(
            passed=True,
            reason="ok",
            message="Entry price not supplied; safety floor not applicable.",
        )

    try:
        price_value = float(entry_price)
    except (TypeError, ValueError):
        return SafetyAssessment(
            passed=True,
            reason="ok",
            message="Entry price not numeric; safety floor not applicable.",
        )

    floor = _STRATEGY_ENTRY_PRICE_FLOORS.get(slug_norm)
    ceiling = _STRATEGY_ENTRY_PRICE_CEILINGS.get(slug_norm)

    if floor is not None and price_value < float(floor):
        return SafetyAssessment(
            passed=False,
            reason="entry_price_below_safety_floor",
            message=(
                f"Strategy '{slug_norm}' entry price {price_value:.4f} "
                f"is below the safety floor {float(floor):.4f}. "
                f"Order refused at submit boundary."
            ),
            floor=float(floor),
            observed=price_value,
        )

    if ceiling is not None and price_value > float(ceiling):
        return SafetyAssessment(
            passed=False,
            reason="entry_price_above_safety_ceiling",
            message=(
                f"Strategy '{slug_norm}' entry price {price_value:.4f} "
                f"is above the safety ceiling {float(ceiling):.4f}. "
                f"Order refused at submit boundary."
            ),
            ceiling=float(ceiling),
            observed=price_value,
        )

    return SafetyAssessment(
        passed=True,
        reason="ok",
        message=(
            f"Strategy '{slug_norm}' entry price {price_value:.4f} "
            f"within safety bounds."
        ),
        floor=float(floor) if floor is not None else None,
        ceiling=float(ceiling) if ceiling is not None else None,
        observed=price_value,
    )


def get_strategy_entry_price_floor(strategy_slug: Optional[str]) -> Optional[float]:
    """Read-only accessor for the configured floor (or None).

    Useful for UI surfacing ("This strategy has a hard floor at X")
    and for tests that pin the table contents.
    """
    slug_norm = (strategy_slug or "").strip().lower()
    if not slug_norm:
        return None
    return _STRATEGY_ENTRY_PRICE_FLOORS.get(slug_norm)


def get_strategy_entry_price_ceiling(strategy_slug: Optional[str]) -> Optional[float]:
    """Read-only accessor for the configured ceiling (or None)."""
    slug_norm = (strategy_slug or "").strip().lower()
    if not slug_norm:
        return None
    return _STRATEGY_ENTRY_PRICE_CEILINGS.get(slug_norm)
