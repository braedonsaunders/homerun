"""Lazy strategy registry exports.

Avoid eager registry imports at package init time to prevent import cycles with
DB strategy loader initialization.
"""

from __future__ import annotations

from typing import Any


def get_strategy(*args: Any, **kwargs: Any):
    from .registry import get_strategy as _get_strategy

    return _get_strategy(*args, **kwargs)


def list_strategy_keys(*args: Any, **kwargs: Any):
    from .registry import list_strategy_keys as _list_strategy_keys

    return _list_strategy_keys(*args, **kwargs)


__all__ = ["get_strategy", "list_strategy_keys"]
