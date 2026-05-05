"""Last-line-of-defense pre-trade safety floors.

Background
----------
Strategies on this platform are DB-backed and user-managed: their
source code lives in ``Strategy.source_code``, their gate config
lives in ``Strategy.config``. Operators add, edit, fork, and remove
strategies through the UI; the platform itself MUST NOT assume any
specific strategy slug exists.

That makes any guard whose only home is the strategy's own
``_score_market`` gate fragile: a fix to the .py file under
``services/strategies/`` doesn't propagate to the running system
because the catalog explicitly preserves user edits across reseeds,
and a UI edit can disable a gate the operator considered protective.

This module enforces operator-installed safety floors at the
**submit boundary** (inside
:func:`services.trader_orchestrator.order_manager.submit_execution_leg`,
right before the CLOB call). Strategies cannot loosen these floors
via UI config or strategy source-code edits — only an explicit call
to :func:`register_strategy_entry_price_floor` at startup or runtime
installs one.

Design
------
* Per-strategy entry-price floors and ceilings, registered by slug.
* Empty by default — the platform makes NO assumptions about which
  strategies exist or what their floors should be. Operators install
  floors as part of their deployment policy (loader script, admin
  endpoint, settings migration, etc.).
* :class:`SafetyAssessment` carries the structured result (no
  exceptions raised) so callers log the decision, propagate it as an
  order ``error_message``, and surface it as a SKIPPED order without
  disturbing cap accounting.

API
---
* :func:`register_strategy_entry_price_floor` — install or update a
  floor for a strategy slug.
* :func:`register_strategy_entry_price_ceiling` — install or update a
  ceiling for a strategy slug.
* :func:`unregister_strategy_entry_price_floor` /
  :func:`unregister_strategy_entry_price_ceiling` — remove operator-
  installed bounds.
* :func:`assert_buy_entry_price_within_safety_bounds` — the canonical
  pre-submit check.
* :func:`get_strategy_entry_price_floor` /
  :func:`get_strategy_entry_price_ceiling` — read-only accessors for
  UI surfacing and diagnostics.
* :func:`clear_all_strategy_entry_price_bounds` — testing-only helper
  to reset the registry between cases.

Concurrency
-----------
The registry is mutated only at install / update time, which is
expected to be infrequent (startup, admin actions). Reads on the hot
path are dict lookups against the live mapping — Python dict reads
are atomic enough for our use here, and the cost of a stale read on
a register/unregister race is one stale safety decision (which is
the same outcome as making the change one cycle later). No lock is
held across the hot-path check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Operator-installed floor / ceiling registries.
#
# Both are EMPTY by default. The platform makes no assumption that any
# particular strategy slug exists. Operators install bounds via the
# ``register_*`` functions below as part of their deployment policy.
# ---------------------------------------------------------------------------

_STRATEGY_ENTRY_PRICE_FLOORS: dict[str, float] = {}
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


def _normalize_slug(strategy_slug: Optional[str]) -> str:
    return (strategy_slug or "").strip().lower()


def register_strategy_entry_price_floor(
    strategy_slug: str,
    floor: float,
) -> None:
    """Install or update the BUY entry-price floor for a strategy slug.

    Args:
        strategy_slug: The strategy slug, case-insensitive. Empty / None
            slugs are silently ignored (no floor installed).
        floor: Minimum acceptable BUY entry price. Polymarket scale
            (0..1). Must be a finite number; non-finite values are
            silently ignored.

    Calling with the same slug overwrites the previous floor.
    """
    slug_norm = _normalize_slug(strategy_slug)
    if not slug_norm:
        return
    try:
        floor_value = float(floor)
    except (TypeError, ValueError):
        return
    if floor_value != floor_value or floor_value in (float("inf"), float("-inf")):
        # NaN / inf — refuse to install.
        return
    _STRATEGY_ENTRY_PRICE_FLOORS[slug_norm] = floor_value


def register_strategy_entry_price_ceiling(
    strategy_slug: str,
    ceiling: float,
) -> None:
    """Install or update the BUY entry-price ceiling for a strategy slug.

    Same semantics as :func:`register_strategy_entry_price_floor` but
    for the upper bound. Useful for settlement-risk-driven ceilings
    (e.g. "never buy at >= 0.99").
    """
    slug_norm = _normalize_slug(strategy_slug)
    if not slug_norm:
        return
    try:
        ceiling_value = float(ceiling)
    except (TypeError, ValueError):
        return
    if ceiling_value != ceiling_value or ceiling_value in (float("inf"), float("-inf")):
        return
    _STRATEGY_ENTRY_PRICE_CEILINGS[slug_norm] = ceiling_value


def unregister_strategy_entry_price_floor(strategy_slug: str) -> None:
    """Remove the operator-installed floor for a strategy slug.

    No-op if the slug isn't registered.
    """
    _STRATEGY_ENTRY_PRICE_FLOORS.pop(_normalize_slug(strategy_slug), None)


def unregister_strategy_entry_price_ceiling(strategy_slug: str) -> None:
    """Remove the operator-installed ceiling for a strategy slug.

    No-op if the slug isn't registered.
    """
    _STRATEGY_ENTRY_PRICE_CEILINGS.pop(_normalize_slug(strategy_slug), None)


def clear_all_strategy_entry_price_bounds() -> None:
    """Wipe the floor + ceiling registries.

    Intended for tests and explicit deployment-time resets. Production
    code should prefer :func:`unregister_strategy_entry_price_floor` /
    :func:`unregister_strategy_entry_price_ceiling` so unrelated bounds
    aren't accidentally cleared.
    """
    _STRATEGY_ENTRY_PRICE_FLOORS.clear()
    _STRATEGY_ENTRY_PRICE_CEILINGS.clear()


def assert_buy_entry_price_within_safety_bounds(
    *,
    strategy_slug: Optional[str],
    entry_price: Optional[float],
) -> SafetyAssessment:
    """Verify a BUY order's entry price clears the operator-installed
    per-strategy hard floor and ceiling.

    This is the canonical pre-submit safety check for entry-price
    bounds. It runs from
    :func:`services.trader_orchestrator.order_manager.submit_execution_leg`
    immediately before the CLOB call, regardless of which path
    constructed the order.

    Args:
        strategy_slug: Strategy identifier (case-insensitive). Empty /
            ``None`` slugs are treated as "no floor enforced" so the
            safety layer never blocks orders whose strategy attribution
            is missing — that's an upstream telemetry issue, not a
            safety violation.
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
        * If no floor or ceiling is registered for the slug, the check
          passes — the registry is empty by default.
    """
    slug_norm = _normalize_slug(strategy_slug)
    if not slug_norm:
        return SafetyAssessment(
            passed=True,
            reason="ok",
            message="No strategy slug supplied; safety floor not applicable.",
        )

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
    """Read-only accessor for the registered floor (or None)."""
    slug_norm = _normalize_slug(strategy_slug)
    if not slug_norm:
        return None
    return _STRATEGY_ENTRY_PRICE_FLOORS.get(slug_norm)


def get_strategy_entry_price_ceiling(strategy_slug: Optional[str]) -> Optional[float]:
    """Read-only accessor for the registered ceiling (or None)."""
    slug_norm = _normalize_slug(strategy_slug)
    if not slug_norm:
        return None
    return _STRATEGY_ENTRY_PRICE_CEILINGS.get(slug_norm)
