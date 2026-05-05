"""User-facing report renderers.

Currently exports the wallet-strategy reverse-engineer executive PDF.
Future reports (drift dashboards, monthly summaries, etc.) live here
too — keep them template-driven so changes are HTML/CSS, not Python.
"""
from __future__ import annotations

__all__ = ["wallet_strategy_report"]
