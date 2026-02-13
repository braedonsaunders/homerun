"""
Kalshi Account API Routes

Endpoints for Kalshi account management, authentication, balance,
positions, and order placement.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.kalshi_client import kalshi_client
from models.database import AsyncSessionLocal, AppSettings
from sqlalchemy import select
from utils.logger import get_logger
from utils.secrets import decrypt_secret

logger = get_logger(__name__)
router = APIRouter(prefix="/kalshi", tags=["Kalshi"])


# ==================== REQUEST/RESPONSE MODELS ====================


class KalshiLoginRequest(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None


class KalshiAccountStatus(BaseModel):
    platform: str = "kalshi"
    authenticated: bool = False
    member_id: Optional[str] = None
    email: Optional[str] = None
    balance: Optional[dict] = None
    positions_count: int = 0


# ==================== ENDPOINTS ====================


@router.get("/status")
async def get_kalshi_status():
    """Get Kalshi account connection status and balance summary."""
    try:
        # Try to initialize from stored credentials if not yet authenticated
        if not kalshi_client.is_authenticated:
            await _try_auto_login()

        status = await kalshi_client.get_account_status()

        # Add email from settings for display
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
            settings = result.scalar_one_or_none()
            if settings:
                status["email"] = settings.kalshi_email

        return status
    except Exception as e:
        logger.error("Failed to get Kalshi status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
async def login_kalshi(request: KalshiLoginRequest):
    """Authenticate with Kalshi.

    Supports two auth methods:
    1. Email + Password: Performs login API call to get bearer token
    2. API Key: Sets the key directly for bearer auth
    """
    try:
        success = await kalshi_client.initialize_auth(
            email=request.email,
            password=request.password,
            api_key=request.api_key,
        )
        if success:
            return {
                "status": "success",
                "message": "Kalshi authentication successful",
                "authenticated": True,
                "member_id": kalshi_client._auth_member_id,
            }
        return {
            "status": "error",
            "message": "Kalshi authentication failed. Check your credentials.",
            "authenticated": False,
        }
    except Exception as e:
        logger.error("Kalshi login error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout")
async def logout_kalshi():
    """Clear Kalshi authentication."""
    kalshi_client.logout()
    return {"status": "success", "message": "Logged out of Kalshi"}


@router.get("/balance")
async def get_kalshi_balance():
    """Get Kalshi account balance."""
    if not kalshi_client.is_authenticated:
        await _try_auto_login()

    if not kalshi_client.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated with Kalshi. Configure credentials in Settings.",
        )

    balance = await kalshi_client.get_balance()
    if balance is None:
        raise HTTPException(status_code=502, detail="Failed to fetch Kalshi balance")
    return balance


@router.get("/positions")
async def get_kalshi_positions():
    """Get current Kalshi portfolio positions."""
    if not kalshi_client.is_authenticated:
        await _try_auto_login()

    if not kalshi_client.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated with Kalshi. Configure credentials in Settings.",
        )

    positions = await kalshi_client.get_positions()
    return positions


@router.get("/orders")
async def get_kalshi_orders(status: Optional[str] = None):
    """Get Kalshi order history."""
    if not kalshi_client.is_authenticated:
        await _try_auto_login()

    if not kalshi_client.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated with Kalshi. Configure credentials in Settings.",
        )

    orders = await kalshi_client.get_orders(status=status)
    return orders


# ==================== HELPERS ====================


async def _try_auto_login():
    """Try to authenticate using stored credentials from settings."""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
            settings = result.scalar_one_or_none()
            if not settings:
                return

            api_key = decrypt_secret(settings.kalshi_api_key)
            password = decrypt_secret(settings.kalshi_password)

            if api_key:
                await kalshi_client.initialize_auth(api_key=api_key)
            elif settings.kalshi_email and password:
                await kalshi_client.initialize_auth(
                    email=settings.kalshi_email,
                    password=password,
                )
    except Exception as e:
        logger.error("Kalshi auto-login failed", error=str(e))
