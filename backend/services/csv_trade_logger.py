"""
CSV Append-Only Trade Logger

Independent audit trail for all trades. Survives database corruption,
openable in Excel/Google Sheets for quick manual review.

Columns: timestamp,context,token_id,side,whale_shares,bot_shares,
whale_price,bot_price,status,fill_percent,slippage_bps,opportunity_id,strategy
"""

import csv
import asyncio
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("csv_trade_logger")

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "data" / "trades.csv"

CSV_COLUMNS = [
    "timestamp", "context", "token_id", "side", "whale_shares", "bot_shares",
    "whale_price", "bot_price", "status", "fill_percent", "slippage_bps",
    "opportunity_id", "strategy", "tier", "category", "total_ms",
]


@dataclass
class TradeLogEntry:
    timestamp: str
    context: str  # "auto_trader", "copy_trader"
    token_id: str
    side: str
    whale_shares: float = 0.0
    bot_shares: float = 0.0
    whale_price: float = 0.0
    bot_price: float = 0.0
    status: str = ""  # SUCCESS, EXEC_FAIL, CB_BLOCKED, SKIPPED_SMALL, PROB_SKIP, DEPTH_BLOCKED
    fill_percent: float = 0.0
    slippage_bps: float = 0.0
    opportunity_id: str = ""
    strategy: str = ""
    tier: int = 0
    category: str = ""
    total_ms: float = 0.0


class CSVTradeLogger:
    def __init__(self, csv_path: Path = None):
        self._path = csv_path or DEFAULT_CSV_PATH
        self._lock = asyncio.Lock()
        self._initialized = False

    def _ensure_file(self):
        if self._initialized:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with open(self._path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)
        self._initialized = True

    async def log_trade(self, entry: TradeLogEntry):
        async with self._lock:
            try:
                self._ensure_file()
                with open(self._path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        entry.timestamp, entry.context, entry.token_id, entry.side,
                        entry.whale_shares, entry.bot_shares, entry.whale_price,
                        entry.bot_price, entry.status, entry.fill_percent,
                        entry.slippage_bps, entry.opportunity_id, entry.strategy,
                        entry.tier, entry.category, entry.total_ms,
                    ])
            except Exception as e:
                logger.error("CSV write failed", error=str(e))

    async def log_quick(self, context: str, token_id: str, side: str, status: str,
                        bot_shares: float = 0, bot_price: float = 0, **kwargs):
        entry = TradeLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            context=context, token_id=token_id, side=side, status=status,
            bot_shares=bot_shares, bot_price=bot_price, **kwargs,
        )
        await self.log_trade(entry)

    def get_path(self) -> str:
        return str(self._path)

    def get_line_count(self) -> int:
        if not self._path.exists():
            return 0
        with open(self._path) as f:
            return sum(1 for _ in f) - 1  # Subtract header


csv_trade_logger = CSVTradeLogger()
