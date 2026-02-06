"""
Telegram Notification Service

Sends formatted Telegram messages for key trading events:
- High-ROI arbitrage opportunities
- Trade executions and results
- Circuit breaker triggers
- Daily P&L summaries
- Error/warning alerts

Uses Telegram Bot API via httpx with rate limiting (20 msgs/min)
to stay within Telegram's rate limits.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import httpx
from sqlalchemy import select

from config import settings
from models.database import AsyncSessionLocal, AppSettings
from utils.logger import get_logger

logger = get_logger("notifier")

TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_MESSAGES_PER_MINUTE = 20


# ── MarkdownV2 helpers ──────────────────────────────────────────────


def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2.

    All of these characters must be escaped outside of code spans:
    _ * [ ] ( ) ~ ` > # + - = | { } . !
    """
    special = r"\_*[]()~`>#+=|{}.!-"
    escaped = []
    for ch in str(text):
        if ch in special:
            escaped.append(f"\\{ch}")
        else:
            escaped.append(ch)
    return "".join(escaped)


def _code(text: str) -> str:
    """Wrap text in an inline code span (no escaping needed inside)."""
    return f"`{text}`"


def _bold(text: str) -> str:
    """Wrap already-escaped text in bold markers."""
    return f"*{text}*"


# ── Daily stats accumulator ─────────────────────────────────────────


@dataclass
class DailyStats:
    """Tracks daily activity for the summary notification."""

    date: str = ""
    opportunities_detected: int = 0
    opportunities_acted_on: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_pnl: float = 0.0
    best_trade_roi: Optional[float] = None
    best_trade_strategy: Optional[str] = None
    worst_trade_roi: Optional[float] = None
    worst_trade_strategy: Optional[str] = None
    strategy_breakdown: dict = field(default_factory=dict)

    def reset(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        self.date = today
        self.opportunities_detected = 0
        self.opportunities_acted_on = 0
        self.trades_won = 0
        self.trades_lost = 0
        self.total_pnl = 0.0
        self.best_trade_roi = None
        self.best_trade_strategy = None
        self.worst_trade_roi = None
        self.worst_trade_strategy = None
        self.strategy_breakdown = {}


# ── Notifier service ────────────────────────────────────────────────


class TelegramNotifier:
    """
    Singleton Telegram notification service.

    Reads credentials from the AppSettings database table first, then
    falls back to ``config.py`` environment variables.  All public
    ``notify_*`` methods are fire-and-forget safe -- they silently
    degrade when credentials are missing or the Telegram API is
    unreachable.
    """

    def __init__(self):
        self._bot_token: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._notifications_enabled: bool = False
        self._notify_on_opportunity: bool = True
        self._notify_on_trade: bool = True
        self._notify_min_roi: float = 5.0

        # Rate limiting: sliding window of send timestamps
        self._send_timestamps: deque[float] = deque()
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._queue_task: Optional[asyncio.Task] = None

        # Daily stats
        self._daily_stats = DailyStats()
        self._daily_summary_task: Optional[asyncio.Task] = None

        # HTTP client (created lazily)
        self._http_client: Optional[httpx.AsyncClient] = None

        # State
        self._running = False
        self._started = False

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self):
        """Initialise the notifier: load settings, start queue worker."""
        if self._started:
            logger.warning("Notifier already started")
            return

        await self._load_settings()

        if not self._bot_token or not self._chat_id:
            logger.info(
                "Telegram credentials not configured -- "
                "notifier will stay dormant until credentials are set"
            )
        else:
            logger.info("Telegram notifier credentials loaded")

        self._http_client = httpx.AsyncClient(timeout=15.0)
        self._running = True
        self._started = True

        # Background worker drains the rate-limited queue
        self._queue_task = asyncio.create_task(self._queue_worker())

        # Daily summary scheduler
        self._daily_stats.reset()
        self._daily_summary_task = asyncio.create_task(self._daily_summary_scheduler())

        # Register with scanner and auto_trader
        self._register_callbacks()

        logger.info("Telegram notifier started")

    def stop(self):
        """Gracefully stop the notifier."""
        self._running = False
        if self._queue_task and not self._queue_task.done():
            self._queue_task.cancel()
        if self._daily_summary_task and not self._daily_summary_task.done():
            self._daily_summary_task.cancel()
        logger.info("Telegram notifier stopped")

    async def shutdown(self):
        """Close the HTTP client (call during app shutdown)."""
        self.stop()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ── Settings ─────────────────────────────────────────────────

    async def _load_settings(self):
        """Load Telegram settings from the database, falling back to config.py."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AppSettings).where(AppSettings.id == "default")
                )
                row = result.scalar_one_or_none()

                if row:
                    self._bot_token = (
                        row.telegram_bot_token or settings.TELEGRAM_BOT_TOKEN
                    )
                    self._chat_id = row.telegram_chat_id or settings.TELEGRAM_CHAT_ID
                    self._notifications_enabled = bool(row.notifications_enabled)
                    self._notify_on_opportunity = bool(row.notify_on_opportunity)
                    self._notify_on_trade = bool(row.notify_on_trade)
                    self._notify_min_roi = (
                        float(row.notify_min_roi)
                        if row.notify_min_roi is not None
                        else 5.0
                    )
                    return
        except Exception as exc:
            logger.warning(
                "Could not load notifier settings from DB, using config.py",
                error=str(exc),
            )

        # Fallback to config.py
        self._bot_token = settings.TELEGRAM_BOT_TOKEN
        self._chat_id = settings.TELEGRAM_CHAT_ID
        self._notifications_enabled = bool(self._bot_token and self._chat_id)

    async def reload_settings(self):
        """Public helper so the settings API can trigger a reload."""
        await self._load_settings()
        logger.info("Notifier settings reloaded")

    # ── Callback registration ────────────────────────────────────

    def _register_callbacks(self):
        """Hook into scanner and auto_trader callback systems."""
        from services.scanner import scanner
        from services.auto_trader import auto_trader

        scanner.add_callback(self._on_opportunities)
        auto_trader.add_callback(self._on_trade_event)
        logger.info("Notifier callbacks registered with scanner and auto_trader")

    # ── Scanner callback ─────────────────────────────────────────

    async def _on_opportunities(self, opportunities):
        """Called by the scanner after each scan cycle."""
        if not self._notifications_enabled or not self._notify_on_opportunity:
            # Still count for daily stats
            self._daily_stats.opportunities_detected += len(opportunities)
            return

        self._daily_stats.opportunities_detected += len(opportunities)

        for opp in opportunities:
            if opp.roi_percent >= self._notify_min_roi:
                await self.notify_opportunity(opp)

    # ── Auto-trader callback ─────────────────────────────────────

    async def _on_trade_event(self, event: str, data: dict):
        """Called by auto_trader on trade lifecycle events."""
        if event == "trade_executed":
            trade = data.get("trade")
            opp = data.get("opportunity")
            if trade and opp:
                self._daily_stats.opportunities_acted_on += 1

                strategy_key = opp.strategy.value
                breakdown = self._daily_stats.strategy_breakdown
                if strategy_key not in breakdown:
                    breakdown[strategy_key] = {"count": 0, "pnl": 0.0}
                breakdown[strategy_key]["count"] += 1

                if self._notifications_enabled and self._notify_on_trade:
                    await self.notify_trade_executed(trade, opp)

        elif event == "trade_resolved":
            trade = data.get("trade")
            if trade:
                pnl = trade.actual_profit or 0.0
                self._daily_stats.total_pnl += pnl

                if pnl >= 0:
                    self._daily_stats.trades_won += 1
                else:
                    self._daily_stats.trades_lost += 1

                roi = (pnl / trade.total_cost * 100) if trade.total_cost else 0.0
                strategy_name = (
                    trade.strategy.value
                    if hasattr(trade.strategy, "value")
                    else str(trade.strategy)
                )

                if (
                    self._daily_stats.best_trade_roi is None
                    or roi > self._daily_stats.best_trade_roi
                ):
                    self._daily_stats.best_trade_roi = roi
                    self._daily_stats.best_trade_strategy = strategy_name
                if (
                    self._daily_stats.worst_trade_roi is None
                    or roi < self._daily_stats.worst_trade_roi
                ):
                    self._daily_stats.worst_trade_roi = roi
                    self._daily_stats.worst_trade_strategy = strategy_name

                breakdown = self._daily_stats.strategy_breakdown
                if strategy_name in breakdown:
                    breakdown[strategy_name]["pnl"] += pnl

        elif event == "circuit_breaker":
            reason = data.get("reason", "Unknown trigger")
            if self._notifications_enabled:
                await self.notify_circuit_breaker(reason)

    # ── Public notification methods ──────────────────────────────

    async def notify_opportunity(self, opp) -> None:
        """Send a notification for a high-ROI opportunity."""
        signal = (
            "\u2705" if opp.roi_percent >= 10 else "\U0001f7e1"
        )  # green check / yellow circle

        strategy = _escape_md(opp.strategy.value.replace("_", " ").title())
        roi = _escape_md(f"{opp.roi_percent:.2f}%")
        cost = _escape_md(f"${opp.total_cost:.2f}")
        profit = _escape_md(f"${opp.net_profit:.4f}")
        risk = _escape_md(f"{opp.risk_score:.2f}")
        liquidity = _escape_md(f"${opp.min_liquidity:,.0f}")
        title = _escape_md(opp.title[:120])

        lines = [
            f"{signal} {_bold('New Opportunity')}",
            "",
            f"{_bold('Strategy:')} {strategy}",
            f"{_bold('ROI:')} {roi}",
            f"{_bold('Cost:')} {cost}",
            f"{_bold('Net Profit:')} {profit}",
            f"{_bold('Risk Score:')} {risk}",
            f"{_bold('Liquidity:')} {liquidity}",
        ]

        if opp.guaranteed_profit is not None:
            gp = _escape_md(f"${opp.guaranteed_profit:.4f}")
            lines.append(f"{_bold('Guaranteed:')} {gp}")
        if opp.mispricing_type:
            mt = _escape_md(opp.mispricing_type.value.replace("_", " ").title())
            lines.append(f"{_bold('Type:')} {mt}")

        lines.extend(["", title])

        await self._enqueue("\n".join(lines))

    async def notify_trade_executed(self, trade, opp) -> None:
        """Send a notification when a trade is executed."""
        mode_label = (
            trade.mode.value if hasattr(trade.mode, "value") else str(trade.mode)
        )
        mode_upper = mode_label.upper()
        signal = (
            "\U0001f4b0" if mode_upper == "LIVE" else "\U0001f4dd"
        )  # money bag / memo

        strategy = _escape_md(
            trade.strategy.value
            if hasattr(trade.strategy, "value")
            else str(trade.strategy)
        )
        roi = _escape_md(f"{opp.roi_percent:.2f}%")
        size = _escape_md(f"${trade.total_cost:.2f}")
        expected = _escape_md(f"${trade.expected_profit:.4f}")
        mode = _escape_md(mode_upper)
        title = _escape_md(opp.title[:120])

        lines = [
            f"{signal} {_bold('Trade Executed')} \\[{mode}\\]",
            "",
            f"{_bold('Strategy:')} {_escape_md(strategy)}",
            f"{_bold('ROI:')} {roi}",
            f"{_bold('Size:')} {size}",
            f"{_bold('Expected Profit:')} {expected}",
        ]

        if trade.guaranteed_profit is not None:
            gp = _escape_md(f"${trade.guaranteed_profit:.4f}")
            lines.append(f"{_bold('Guaranteed:')} {gp}")
        if trade.mispricing_type:
            mt = _escape_md(trade.mispricing_type.replace("_", " ").title())
            lines.append(f"{_bold('Type:')} {mt}")

        lines.extend(["", title])

        await self._enqueue("\n".join(lines))

    async def notify_circuit_breaker(self, reason: str) -> None:
        """Send an alert when the circuit breaker triggers."""
        lines = [
            f"\u26a0\ufe0f {_bold('Circuit Breaker Triggered')}",
            "",
            f"{_bold('Reason:')} {_escape_md(reason)}",
            "",
            _escape_md("Auto-trading paused. Manual review recommended."),
        ]
        await self._enqueue("\n".join(lines))

    async def notify_error(self, component: str, message: str) -> None:
        """Send an error/warning alert."""
        if not self._notifications_enabled:
            return
        lines = [
            f"\u274c {_bold('Error Alert')}",
            "",
            f"{_bold('Component:')} {_escape_md(component)}",
            f"{_bold('Message:')} {_escape_md(message[:500])}",
        ]
        await self._enqueue("\n".join(lines))

    async def notify_daily_summary(self) -> None:
        """Send the daily P&L summary."""
        stats = self._daily_stats
        pnl_signal = (
            "\U0001f4c8" if stats.total_pnl >= 0 else "\U0001f4c9"
        )  # chart up / chart down

        lines = [
            f"{pnl_signal} {_bold('Daily Summary')} \\- {_escape_md(stats.date)}",
            "",
            f"{_bold('Opportunities Detected:')} {_escape_md(str(stats.opportunities_detected))}",
            f"{_bold('Opportunities Acted On:')} {_escape_md(str(stats.opportunities_acted_on))}",
            f"{_bold('Trades Won:')} {_escape_md(str(stats.trades_won))}",
            f"{_bold('Trades Lost:')} {_escape_md(str(stats.trades_lost))}",
            f"{_bold('P&L:')} {_escape_md(f'${stats.total_pnl:+.2f}')}",
        ]

        if stats.best_trade_roi is not None:
            lines.append(
                f"{_bold('Best Trade:')} {_escape_md(f'{stats.best_trade_roi:+.2f}%')} "
                f"\\({_escape_md(stats.best_trade_strategy or 'N/A')}\\)"
            )
        if stats.worst_trade_roi is not None:
            lines.append(
                f"{_bold('Worst Trade:')} {_escape_md(f'{stats.worst_trade_roi:+.2f}%')} "
                f"\\({_escape_md(stats.worst_trade_strategy or 'N/A')}\\)"
            )

        if stats.strategy_breakdown:
            lines.append("")
            lines.append(_bold("Strategy Breakdown:"))
            for strat, info in sorted(stats.strategy_breakdown.items()):
                name = _escape_md(strat.replace("_", " ").title())
                count = _escape_md(str(info["count"]))
                pnl = _escape_md(f"${info['pnl']:+.2f}")
                lines.append(f"  {name}: {count} trades, {pnl}")

        await self._enqueue("\n".join(lines))

    # ── Rate limiting & queue ────────────────────────────────────

    def _can_send_now(self) -> bool:
        """Check if we are within the rate limit window."""
        now = time.monotonic()
        # Purge timestamps older than 60 s
        while self._send_timestamps and self._send_timestamps[0] < now - 60:
            self._send_timestamps.popleft()
        return len(self._send_timestamps) < MAX_MESSAGES_PER_MINUTE

    def _record_send(self):
        self._send_timestamps.append(time.monotonic())

    async def _enqueue(self, text: str):
        """Add a message to the send queue."""
        if not self._bot_token or not self._chat_id:
            return
        await self._message_queue.put(text)

    async def _queue_worker(self):
        """Background task that drains the message queue respecting rate limits."""
        while self._running:
            try:
                # Wait for a message (with timeout so we can check _running)
                try:
                    message = await asyncio.wait_for(
                        self._message_queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Wait until rate limit allows sending
                while not self._can_send_now():
                    await asyncio.sleep(1.0)
                    if not self._running:
                        return

                await self._send_telegram(message)
                self._record_send()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Queue worker error", error=str(exc))
                await asyncio.sleep(1.0)

    # ── Telegram API ─────────────────────────────────────────────

    async def _send_telegram(self, text: str) -> bool:
        """Send a single message via the Telegram Bot API."""
        if not self._bot_token or not self._chat_id:
            logger.debug("Telegram credentials not configured, skipping message")
            return False

        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=15.0)

        url = f"{TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        }

        try:
            resp = await self._http_client.post(url, json=payload)
            if resp.status_code == 200:
                logger.debug("Telegram message sent successfully")
                return True

            # Telegram returns 429 for rate limiting
            if resp.status_code == 429:
                body = resp.json()
                retry_after = body.get("parameters", {}).get("retry_after", 5)
                logger.warning(
                    "Telegram rate limited, will retry", retry_after=retry_after
                )
                await asyncio.sleep(retry_after)
                # Re-queue the message
                await self._message_queue.put(text)
                return False

            logger.warning(
                "Telegram API error",
                status=resp.status_code,
                body=resp.text[:300],
            )
            return False

        except httpx.TimeoutException:
            logger.warning("Telegram API request timed out")
            return False
        except Exception as exc:
            logger.error("Failed to send Telegram message", error=str(exc))
            return False

    # ── Daily summary scheduler ──────────────────────────────────

    async def _daily_summary_scheduler(self):
        """Send a daily summary at ~00:00 UTC and reset counters."""
        while self._running:
            try:
                now = datetime.utcnow()
                # Next midnight UTC
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                wait_seconds = (tomorrow - now).total_seconds()

                await asyncio.sleep(wait_seconds)
                if not self._running:
                    return

                if self._notifications_enabled:
                    await self.notify_daily_summary()

                self._daily_stats.reset()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Daily summary scheduler error", error=str(exc))
                await asyncio.sleep(60)

    # ── Utility ──────────────────────────────────────────────────

    async def send_test_message(self) -> bool:
        """Send a test message to verify the Telegram configuration."""
        await self._load_settings()
        if not self._bot_token or not self._chat_id:
            return False

        text = (
            f"\u2705 {_bold('Homerun Notifier')}\n\n"
            f"Test message received\\.\n"
            f"Notifications are working correctly\\."
        )
        return await self._send_telegram(text)


# Singleton instance
notifier = TelegramNotifier()
