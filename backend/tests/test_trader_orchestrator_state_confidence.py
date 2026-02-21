import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base
from services.trader_orchestrator_state import _normalize_trader_payload
from tests.postgres_test_db import build_postgres_session_factory


async def _build_session_factory(_tmp_path: Path):
    return await build_postgres_session_factory(Base, "trader_state_confidence")


@pytest.mark.asyncio
async def test_normalize_trader_payload_converts_percent_min_confidence(tmp_path):
    payload = {
        "name": "Crypto HF Trader",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "crypto_15m",
                "strategy_params": {"min_confidence": 45, "min_edge_percent": 3.0},
            }
        ],
    }
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        normalized = await _normalize_trader_payload(session, payload)
    await engine.dispose()
    assert normalized["source_configs"][0]["strategy_params"]["min_confidence"] == 0.45


@pytest.mark.asyncio
async def test_normalize_trader_payload_preserves_fraction_min_confidence(tmp_path):
    payload = {
        "name": "Crypto HF Trader",
        "source_configs": [
            {
                "source_key": "crypto",
                "strategy_key": "crypto_15m",
                "strategy_params": {"min_confidence": 0.45},
            }
        ],
    }
    engine, session_factory = await _build_session_factory(tmp_path)
    async with session_factory() as session:
        normalized = await _normalize_trader_payload(session, payload)
    await engine.dispose()
    assert normalized["source_configs"][0]["strategy_params"]["min_confidence"] == 0.45
