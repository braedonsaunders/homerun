"""Strategy reverse-engineering subsystem.

Owns the long-running pipeline that takes a wallet address, profiles
the trader, picks an appropriate dataset (Data Lab integration), runs
an LLM agent loop that iteratively writes ``BaseStrategy`` Python
source, backtests each candidate against the chosen dataset, scores
the result against the wallet's actual fills, and refines until either
``target_score`` or ``max_iterations``.

The user-visible deliverable is a Python strategy that can be promoted
into the live strategy library plus an optional executive PDF report.

Public surface:
  * ``service.enqueue_job``                — start a new run
  * ``service.run_job``                    — execute (called by worker)
  * ``service.cancel_job``                 — cooperative cancel
  * ``service.list_jobs``, ``get_job``     — read state
  * ``service.list_iterations``            — read per-iteration audit

The agent loop and tool registry live in :mod:`agent` and :mod:`tools`.
"""

from __future__ import annotations

__all__ = [
    "service",
    "agent",
    "tools",
]
