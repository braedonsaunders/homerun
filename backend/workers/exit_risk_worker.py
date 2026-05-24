"""Trading-plane worker entry for the isolated exit-risk loop.

Thin ``start_loop`` shim so the worker host (``workers/host.py``) owns the
task lifecycle, consistent with the other trading-plane workers. The actual
fast stop-loss evaluation/execution lives in
``services.trader_orchestrator.exit_risk_loop.ExitRiskLoop``.
"""

from __future__ import annotations


async def start_loop() -> None:
    from services.trader_orchestrator.exit_risk_loop import get_exit_risk_loop

    await get_exit_risk_loop().run_forever()
