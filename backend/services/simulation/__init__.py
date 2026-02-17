"""Simulation package.

This package preserves the legacy `services.simulation` module surface while
adding execution simulator components under `services.simulation.*`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_legacy_module():
    legacy_path = Path(__file__).resolve().parent.parent / "simulation.py"
    spec = importlib.util.spec_from_file_location("services._simulation_legacy", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load legacy simulation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()

# Legacy exports used across the codebase.
SimulationService = _legacy.SimulationService
SlippageModel = _legacy.SlippageModel
simulation_service = _legacy.simulation_service
AsyncSessionLocal = _legacy.AsyncSessionLocal
settings = _legacy.settings
asyncio = _legacy.asyncio

from .execution_simulator import ExecutionSimulator, execution_simulator
from .fill_models import FillConfig, FillModel
from .historical_data_provider import HistoricalDataProvider

__all__ = [
    "SimulationService",
    "SlippageModel",
    "simulation_service",
    "AsyncSessionLocal",
    "settings",
    "asyncio",
    "ExecutionSimulator",
    "execution_simulator",
    "FillConfig",
    "FillModel",
    "HistoricalDataProvider",
]
