from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, field_validator

from services.anomaly_detector import anomaly_detector, AnomalyType, Severity
from utils.validation import validate_eth_address

anomaly_router = APIRouter()


class AnalyzeWalletRequest(BaseModel):
    address: str

    @field_validator("address")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_eth_address(v)


class FindProfitableRequest(BaseModel):
    min_trades: int = 50
    min_win_rate: float = 0.6
    min_pnl: float = 1000.0
    max_anomaly_score: float = 0.5


# ==================== WALLET ANALYSIS ====================

@anomaly_router.post("/analyze")
async def analyze_wallet(request: AnalyzeWalletRequest):
    """
    Analyze a wallet for trading patterns and anomalies.

    Returns comprehensive analysis including:
    - Trading statistics (win rate, ROI, PnL)
    - Detected strategies being used
    - Anomaly detection results
    - Recommendation for whether to copy
    """
    analysis = await anomaly_detector.analyze_wallet(request.address)

    return {
        "wallet": analysis.address,
        "stats": {
            "total_trades": analysis.total_trades,
            "win_rate": analysis.win_rate,
            "total_pnl": analysis.total_pnl,
            "avg_roi": analysis.avg_roi,
            "max_roi": analysis.max_roi,
            "avg_hold_time_hours": analysis.avg_hold_time_hours,
            "trade_frequency_per_day": analysis.trade_frequency_per_day,
            "markets_traded": analysis.markets_traded
        },
        "strategies_detected": analysis.strategies_detected,
        "anomaly_score": analysis.anomaly_score,
        "anomalies": analysis.anomalies,
        "is_profitable_pattern": analysis.is_profitable_pattern,
        "recommendation": analysis.recommendation
    }


@anomaly_router.get("/analyze/{wallet_address}")
async def analyze_wallet_get(wallet_address: str):
    """Analyze a wallet (GET method for convenience)"""
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    analysis = await anomaly_detector.analyze_wallet(address)

    return {
        "wallet": analysis.address,
        "stats": {
            "total_trades": analysis.total_trades,
            "win_rate": analysis.win_rate,
            "total_pnl": analysis.total_pnl,
            "avg_roi": analysis.avg_roi,
            "max_roi": analysis.max_roi
        },
        "strategies_detected": analysis.strategies_detected,
        "anomaly_score": analysis.anomaly_score,
        "anomalies": analysis.anomalies,
        "is_profitable_pattern": analysis.is_profitable_pattern,
        "recommendation": analysis.recommendation
    }


# ==================== FIND PROFITABLE WALLETS ====================

@anomaly_router.post("/find-profitable")
async def find_profitable_wallets(request: FindProfitableRequest):
    """
    Find wallets with profitable patterns that aren't suspicious.

    This is used to discover wallets worth copying.
    Filters out wallets with high anomaly scores (suspicious activity).
    """
    wallets = await anomaly_detector.find_profitable_wallets(
        min_trades=request.min_trades,
        min_win_rate=request.min_win_rate,
        min_pnl=request.min_pnl,
        max_anomaly_score=request.max_anomaly_score
    )

    return {
        "count": len(wallets),
        "wallets": [{
            "address": w.address,
            "win_rate": w.win_rate,
            "total_pnl": w.total_pnl,
            "avg_roi": w.avg_roi,
            "strategies": w.strategies_detected,
            "anomaly_score": w.anomaly_score,
            "recommendation": w.recommendation
        } for w in wallets]
    }


# ==================== ANOMALIES ====================

@anomaly_router.get("/anomalies")
async def get_anomalies(
    severity: Optional[str] = Query(default=None, description="Filter by severity: low, medium, high, critical"),
    anomaly_type: Optional[str] = Query(default=None, description="Filter by anomaly type"),
    limit: int = Query(default=100, ge=1, le=500)
):
    """Get detected anomalies"""
    # Validate severity
    if severity and severity not in [s.value for s in Severity]:
        raise HTTPException(status_code=400, detail=f"Invalid severity. Must be one of: {[s.value for s in Severity]}")

    # Validate anomaly type
    if anomaly_type and anomaly_type not in [t.value for t in AnomalyType]:
        raise HTTPException(status_code=400, detail=f"Invalid anomaly type. Must be one of: {[t.value for t in AnomalyType]}")

    anomalies = await anomaly_detector.get_anomalies(
        severity=severity,
        anomaly_type=anomaly_type,
        limit=limit
    )

    return {
        "count": len(anomalies),
        "anomalies": anomalies
    }


@anomaly_router.get("/anomaly-types")
async def get_anomaly_types():
    """Get all anomaly types and their descriptions"""
    return {
        "types": [
            {
                "type": t.value,
                "category": "statistical" if t.value in ["impossible_win_rate", "unusual_roi", "perfect_timing", "statistically_impossible"]
                    else "pattern" if t.value in ["front_running", "wash_trading", "coordinated_trading"]
                    else "behavioral"
            }
            for t in AnomalyType
        ],
        "severities": [s.value for s in Severity]
    }


# ==================== QUICK CHECKS ====================

@anomaly_router.get("/check/{wallet_address}")
async def quick_check_wallet(wallet_address: str):
    """
    Quick check if a wallet is suspicious.

    Returns a simple pass/fail with basic stats.
    Use /analyze for full analysis.
    """
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    analysis = await anomaly_detector.analyze_wallet(address)

    is_suspicious = analysis.anomaly_score > 0.5
    critical_anomalies = [a for a in analysis.anomalies if a["severity"] == "critical"]

    return {
        "wallet": address,
        "is_suspicious": is_suspicious,
        "anomaly_score": analysis.anomaly_score,
        "critical_anomalies": len(critical_anomalies),
        "win_rate": analysis.win_rate,
        "total_pnl": analysis.total_pnl,
        "verdict": "AVOID" if is_suspicious else "OK",
        "summary": analysis.recommendation
    }
