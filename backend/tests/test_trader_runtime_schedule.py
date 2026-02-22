import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategy_sdk import StrategySDK
from services.trader_orchestrator.decision_gates import is_within_trading_schedule_utc


def _utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def test_validate_trader_runtime_metadata_defaults_include_schedule():
    cfg = StrategySDK.validate_trader_runtime_metadata({})
    schedule = cfg["trading_schedule_utc"]
    assert schedule["enabled"] is False
    assert schedule["days"] == ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    assert schedule["start_time"] == "00:00"
    assert schedule["end_time"] == "23:59"
    assert schedule["start_date"] is None
    assert schedule["end_date"] is None
    assert schedule["end_at"] is None


def test_validate_trader_runtime_metadata_normalizes_schedule_values():
    cfg = StrategySDK.validate_trader_runtime_metadata(
        {
            "trading_schedule_utc": {
                "enabled": "true",
                "days": ["Monday", "fri", "SUN", "bad"],
                "start_time": "13:00",
                "end_time": "14:30",
                "start_date": "2026-02-01",
                "end_date": "2026-02-28",
                "end_at": "2026-02-28T21:15:00Z",
            }
        }
    )
    assert cfg["trading_schedule_utc"] == {
        "enabled": True,
        "days": ["mon", "fri", "sun"],
        "start_time": "13:00",
        "end_time": "14:30",
        "start_date": "2026-02-01",
        "end_date": "2026-02-28",
        "end_at": "2026-02-28T21:15:00Z",
    }


def test_trading_schedule_blocks_outside_time_weekday_and_date():
    metadata = {
        "trading_schedule_utc": {
            "enabled": True,
            "days": ["mon", "tue", "wed", "thu", "fri"],
            "start_time": "13:00",
            "end_time": "14:00",
            "start_date": "2026-02-01",
            "end_date": "2026-02-28",
            "end_at": None,
        }
    }

    assert is_within_trading_schedule_utc(metadata, _utc(2026, 2, 20, 13, 30)) is True
    assert is_within_trading_schedule_utc(metadata, _utc(2026, 2, 20, 14, 1)) is False
    assert is_within_trading_schedule_utc(metadata, _utc(2026, 2, 21, 13, 30)) is False
    assert is_within_trading_schedule_utc(metadata, _utc(2026, 3, 1, 13, 30)) is False


def test_trading_schedule_respects_end_at_and_overnight_windows():
    metadata = {
        "trading_schedule_utc": {
            "enabled": True,
            "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            "start_time": "23:00",
            "end_time": "02:00",
            "start_date": None,
            "end_date": None,
            "end_at": "2026-02-21T01:30:00Z",
        }
    }

    assert is_within_trading_schedule_utc(metadata, _utc(2026, 2, 21, 0, 30)) is True
    assert is_within_trading_schedule_utc(metadata, _utc(2026, 2, 21, 1, 30)) is False
    assert is_within_trading_schedule_utc(metadata, _utc(2026, 2, 21, 15, 0)) is False
