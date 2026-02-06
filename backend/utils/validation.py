import re
from typing import Optional
from pydantic import BaseModel, field_validator, Field
from fastapi import HTTPException


# Ethereum address regex
ETH_ADDRESS_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")


def validate_eth_address(address: str) -> str:
    """Validate Ethereum address format"""
    if not address:
        raise ValueError("Address cannot be empty")

    address = address.strip()

    if not ETH_ADDRESS_REGEX.match(address):
        raise ValueError(f"Invalid Ethereum address format: {address}")

    return address


def validate_positive_number(value: float, name: str) -> float:
    """Validate that a number is positive"""
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def validate_percentage(value: float, name: str) -> float:
    """Validate that a value is a valid percentage (0-100)"""
    if value < 0 or value > 100:
        raise ValueError(f"{name} must be between 0 and 100")
    return value


def validate_limit(value: int, max_limit: int = 1000) -> int:
    """Validate pagination limit"""
    if value < 1:
        return 1
    if value > max_limit:
        return max_limit
    return value


class WalletAddressParam(BaseModel):
    """Validated wallet address parameter"""
    address: str

    @field_validator("address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        return validate_eth_address(v)


class PaginationParams(BaseModel):
    """Validated pagination parameters"""
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class OpportunityFilterParams(BaseModel):
    """Validated opportunity filter parameters"""
    min_profit: float = Field(default=0.0, ge=0.0, le=100.0)
    max_risk: float = Field(default=1.0, ge=0.0, le=1.0)
    min_liquidity: float = Field(default=0.0, ge=0.0)
    strategy: Optional[str] = None


class SimulationParams(BaseModel):
    """Validated simulation parameters"""
    initial_capital: float = Field(default=10000.0, ge=100.0, le=10000000.0)
    position_size_percent: float = Field(default=5.0, ge=0.1, le=100.0)
    max_positions: int = Field(default=10, ge=1, le=100)
    slippage_tolerance: float = Field(default=0.5, ge=0.0, le=10.0)


class CopyTradingParams(BaseModel):
    """Validated copy trading parameters"""
    source_wallet: str
    enabled: bool = True
    copy_threshold_roi: float = Field(default=2.5, ge=0.0, le=100.0)
    max_position_size: float = Field(default=1000.0, ge=10.0, le=1000000.0)
    copy_delay_seconds: int = Field(default=5, ge=0, le=300)
    slippage_tolerance: float = Field(default=1.0, ge=0.0, le=10.0)

    @field_validator("source_wallet")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_eth_address(v)


def validate_request(model: BaseModel, data: dict) -> BaseModel:
    """Validate request data against a Pydantic model"""
    try:
        return model(**data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
