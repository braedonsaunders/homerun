"""Dispatch ``on_fill`` / ``on_partial_fill`` / ``on_cancel`` to the
right strategy class instance.

This is the platform-side glue.  Strategy authors override the hooks
on their ``BaseStrategy`` subclass; the platform calls into here at
the canonical fill / cancel sites, and we look up the strategy by
slug, instantiate (or reuse the loader-cached instance), and invoke
the hook in a try/except.

Hooks are no-ops by default (``BaseStrategy`` provides empty
implementations) so missing them is never a runtime error — they
are pure observation.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any


logger = logging.getLogger("strategy_callbacks")


def _resolve_strategy_instance(strategy_slug: str) -> Any | None:
    if not strategy_slug:
        return None
    try:
        # Process-shared singleton — populated by the strategy registry
        # boot path on each worker plane.  If the slug isn't loaded in
        # this process (e.g. a news-plane callback firing for a
        # trading-plane-only strategy), get_instance returns None and
        # we silently no-op.
        from services.strategy_loader import strategy_loader

        return strategy_loader.get_instance(strategy_slug)
    except Exception:
        logger.debug("Failed to resolve strategy instance for slug=%s", strategy_slug, exc_info=True)
        return None


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        try:
            await value
        except Exception:
            logger.exception("Strategy callback raised on await")


async def dispatch_on_fill(
    *,
    strategy_slug: str,
    order: Any,
    mode: str,
    filled_shares: float,
    average_price: float,
    notional_usd: float,
    ensemble_snapshot: dict | None,
) -> None:
    instance = _resolve_strategy_instance(strategy_slug)
    if instance is None or not hasattr(instance, "on_fill"):
        return
    try:
        result = instance.on_fill(
            order,
            mode=mode,
            filled_shares=filled_shares,
            average_price=average_price,
            notional_usd=notional_usd,
            ensemble_snapshot=ensemble_snapshot,
        )
        await _maybe_await(result)
    except Exception:
        logger.exception("Strategy %s.on_fill raised", strategy_slug)


async def dispatch_on_partial_fill(
    *,
    strategy_slug: str,
    order: Any,
    mode: str,
    filled_shares: float,
    remaining_shares: float,
    average_price: float,
) -> None:
    instance = _resolve_strategy_instance(strategy_slug)
    if instance is None or not hasattr(instance, "on_partial_fill"):
        return
    try:
        result = instance.on_partial_fill(
            order,
            mode=mode,
            filled_shares=filled_shares,
            remaining_shares=remaining_shares,
            average_price=average_price,
        )
        await _maybe_await(result)
    except Exception:
        logger.exception("Strategy %s.on_partial_fill raised", strategy_slug)


async def dispatch_on_cancel(
    *,
    strategy_slug: str,
    order: Any,
    mode: str,
    reason: str,
    unfilled_shares: float,
) -> None:
    instance = _resolve_strategy_instance(strategy_slug)
    if instance is None or not hasattr(instance, "on_cancel"):
        return
    try:
        result = instance.on_cancel(
            order,
            mode=mode,
            reason=reason,
            unfilled_shares=unfilled_shares,
        )
        await _maybe_await(result)
    except Exception:
        logger.exception("Strategy %s.on_cancel raised", strategy_slug)
