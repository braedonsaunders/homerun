"""Telegram notification service focused on trader-orchestrator activity."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from sqlalchemy import and_, func, or_, select

from config import settings
from models.database import (
    AppSettings,
    AsyncSessionLocal,
    Trader,
    TraderDecision,
    TraderEvent,
    TraderOrder,
    TraderOrchestratorControl,
    TraderOrchestratorSnapshot,
)
from utils.logger import get_logger
from utils.secrets import decrypt_secret
from utils.utcnow import utcnow

logger = get_logger("notifier")

TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_MESSAGES_PER_MINUTE = 20
MONITOR_POLL_SECONDS = 5
SETTINGS_REFRESH_SECONDS = 30
DEFAULT_SUMMARY_INTERVAL_MINUTES = 60


def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"\_*[]()~`>#+=|{}.!-"
    escaped: list[str] = []
    for ch in str(text):
        if ch in special:
            escaped.append(f"\\{ch}")
        else:
            escaped.append(ch)
    return "".join(escaped)


def _bold(text: str) -> str:
    return f"*{text}*"


def _clamp_int(value: object, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def _to_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


class TelegramNotifier:
    """Telegram notifier that tracks autotrader-only runtime activity."""

    def __init__(self) -> None:
        self._bot_token: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._notifications_enabled: bool = False

        self._notify_autotrader_orders: bool = False
        self._notify_autotrader_issues: bool = True
        self._notify_autotrader_timeline: bool = True
        self._summary_interval_minutes: int = DEFAULT_SUMMARY_INTERVAL_MINUTES
        self._summary_per_trader: bool = False

        self._send_timestamps: deque[float] = deque()
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._queue_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None

        self._http_client: Optional[httpx.AsyncClient] = None

        self._running = False
        self._started = False
        self._delivery_ready = False

        self._autotrader_active: bool = False
        self._last_summary_at: Optional[datetime] = None
        self._last_snapshot_error: Optional[str] = None
        self._last_settings_reload_monotonic: float = 0.0

        self._last_order_cursor: Optional[tuple[datetime, str]] = None
        self._last_event_cursor: Optional[tuple[datetime, str]] = None

    async def start(self) -> None:
        """Initialize notifier runtime and start background workers."""
        if self._started:
            logger.warning("Notifier already started")
            return

        await self._load_settings()
        await self._prime_cursors()

        if not self._bot_token or not self._chat_id:
            logger.info("Telegram credentials not configured -- notifier will stay dormant until credentials are set")
        else:
            logger.info("Telegram notifier credentials loaded")

        self._http_client = httpx.AsyncClient(timeout=15.0)
        self._running = True
        self._started = True
        self._last_settings_reload_monotonic = time.monotonic()

        self._queue_task = asyncio.create_task(self._queue_worker())
        self._monitor_task = asyncio.create_task(self._autotrader_monitor_loop())

        logger.info("Telegram notifier started (autotrader mode)")

    def stop(self) -> None:
        """Stop notifier tasks."""
        self._running = False
        for task in (self._queue_task, self._monitor_task):
            if task and not task.done():
                task.cancel()
        logger.info("Telegram notifier stopped")

    async def shutdown(self) -> None:
        """Stop and close network resources."""
        self.stop()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _load_settings(self) -> None:
        """Load notifier configuration from DB with environment fallback."""
        row: Optional[AppSettings] = None
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                row = result.scalar_one_or_none()
        except Exception as exc:
            logger.warning(
                "Could not load notifier settings from DB, using config.py",
                error=str(exc),
            )

        if row is None:
            self._bot_token = settings.TELEGRAM_BOT_TOKEN
            self._chat_id = settings.TELEGRAM_CHAT_ID
            self._notifications_enabled = bool(self._bot_token and self._chat_id)
            self._notify_autotrader_orders = False
            self._notify_autotrader_issues = True
            self._notify_autotrader_timeline = True
            self._summary_interval_minutes = DEFAULT_SUMMARY_INTERVAL_MINUTES
            self._summary_per_trader = False
            return

        self._bot_token = decrypt_secret(row.telegram_bot_token) or settings.TELEGRAM_BOT_TOKEN
        self._chat_id = row.telegram_chat_id or settings.TELEGRAM_CHAT_ID
        self._notifications_enabled = bool(row.notifications_enabled)

        # Legacy bridge: if new toggles are missing for any reason, fall back to old flags.
        self._notify_autotrader_orders = bool(getattr(row, "notify_autotrader_orders", bool(row.notify_on_trade)))
        self._notify_autotrader_issues = bool(getattr(row, "notify_autotrader_issues", True))
        self._notify_autotrader_timeline = bool(
            getattr(row, "notify_autotrader_timeline", bool(row.notify_on_opportunity))
        )
        self._summary_interval_minutes = _clamp_int(
            getattr(
                row,
                "notify_autotrader_summary_interval_minutes",
                DEFAULT_SUMMARY_INTERVAL_MINUTES,
            ),
            5,
            1440,
            DEFAULT_SUMMARY_INTERVAL_MINUTES,
        )
        self._summary_per_trader = bool(getattr(row, "notify_autotrader_summary_per_trader", False))

    async def reload_settings(self) -> None:
        await self._load_settings()
        logger.info("Notifier settings reloaded")

    async def _prime_cursors(self) -> None:
        """Avoid replaying old DB records when notifier starts."""
        try:
            async with AsyncSessionLocal() as session:
                latest_order = (
                    await session.execute(
                        select(TraderOrder.created_at, TraderOrder.id)
                        .order_by(TraderOrder.created_at.desc(), TraderOrder.id.desc())
                        .limit(1)
                    )
                ).first()
                if latest_order and latest_order[0] is not None:
                    self._last_order_cursor = (_to_utc(latest_order[0]), str(latest_order[1]))

                latest_event = (
                    await session.execute(
                        select(TraderEvent.created_at, TraderEvent.id)
                        .order_by(TraderEvent.created_at.desc(), TraderEvent.id.desc())
                        .limit(1)
                    )
                ).first()
                if latest_event and latest_event[0] is not None:
                    self._last_event_cursor = (_to_utc(latest_event[0]), str(latest_event[1]))

                snapshot = await session.get(TraderOrchestratorSnapshot, "latest")
                if snapshot is not None:
                    self._last_snapshot_error = str(snapshot.last_error).strip() if snapshot.last_error else None

                control = await session.get(TraderOrchestratorControl, "default")
                self._autotrader_active = bool(
                    control is not None and bool(control.is_enabled) and not bool(control.is_paused)
                )
                if self._autotrader_active:
                    self._last_summary_at = utcnow()
        except Exception as exc:
            logger.debug("Notifier cursor priming skipped", error=str(exc))

    async def _autotrader_monitor_loop(self) -> None:
        """Poll trader-orchestrator state and emit Telegram notifications."""
        while self._running:
            try:
                now_monotonic = time.monotonic()
                if now_monotonic - self._last_settings_reload_monotonic >= SETTINGS_REFRESH_SECONDS:
                    await self._load_settings()
                    self._last_settings_reload_monotonic = now_monotonic

                delivery_ready = bool(self._notifications_enabled and self._bot_token and self._chat_id)

                if delivery_ready and not self._delivery_ready:
                    # Avoid replaying stale history when notifications are
                    # enabled after running with delivery disabled.
                    await self._prime_cursors()
                    self._delivery_ready = True

                if not delivery_ready:
                    self._delivery_ready = False
                    await asyncio.sleep(MONITOR_POLL_SECONDS)
                    continue

                async with AsyncSessionLocal() as session:
                    control = await session.get(TraderOrchestratorControl, "default")
                    snapshot = await session.get(TraderOrchestratorSnapshot, "latest")

                    active = bool(control is not None and bool(control.is_enabled) and not bool(control.is_paused))
                    mode = str(getattr(control, "mode", "paper") or "paper").upper()

                    traders_running = int(getattr(snapshot, "traders_running", 0) if snapshot else 0)
                    traders_total = int(getattr(snapshot, "traders_total", 0) if snapshot else 0)

                    if active and not self._autotrader_active:
                        await self._enqueue(
                            self._format_runtime_state_message(
                                title="Autotrader Active",
                                mode=mode,
                                traders_running=traders_running,
                                traders_total=traders_total,
                            )
                        )
                        self._last_summary_at = utcnow()
                    elif not active and self._autotrader_active:
                        await self._enqueue(
                            self._format_runtime_state_message(
                                title="Autotrader Paused",
                                mode=mode,
                                traders_running=traders_running,
                                traders_total=traders_total,
                            )
                        )
                        self._last_summary_at = None

                    self._autotrader_active = active

                    new_events = await self._load_new_trader_events(session)
                    new_orders = await self._load_new_trader_orders(session)

                    if self._notify_autotrader_issues:
                        await self._send_issue_alerts(
                            session=session,
                            events=new_events,
                            orders=new_orders,
                            snapshot=snapshot,
                        )

                    if active and self._notify_autotrader_orders:
                        await self._send_order_activity_alert(
                            session=session,
                            orders=new_orders,
                            mode=mode,
                        )

                    if active and self._notify_autotrader_timeline:
                        await self._maybe_send_timeline_summary(
                            session=session,
                            control=control,
                            snapshot=snapshot,
                        )
                    elif not active:
                        self._last_summary_at = None

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Autotrader notifier monitor error", error=str(exc))

            await asyncio.sleep(MONITOR_POLL_SECONDS)

    async def _load_new_trader_orders(self, session) -> list[TraderOrder]:
        query = select(TraderOrder).order_by(
            TraderOrder.created_at.asc(),
            TraderOrder.id.asc(),
        )

        if self._last_order_cursor is not None:
            cursor_ts, cursor_id = self._last_order_cursor
            query = query.where(
                or_(
                    TraderOrder.created_at > cursor_ts.replace(tzinfo=None),
                    and_(
                        TraderOrder.created_at == cursor_ts.replace(tzinfo=None),
                        TraderOrder.id > cursor_id,
                    ),
                )
            )

        rows = (await session.execute(query.limit(200))).scalars().all()
        if rows:
            last = rows[-1]
            if last.created_at is not None:
                self._last_order_cursor = (_to_utc(last.created_at), str(last.id))
        return rows

    async def _load_new_trader_events(self, session) -> list[TraderEvent]:
        query = select(TraderEvent).order_by(
            TraderEvent.created_at.asc(),
            TraderEvent.id.asc(),
        )

        if self._last_event_cursor is not None:
            cursor_ts, cursor_id = self._last_event_cursor
            query = query.where(
                or_(
                    TraderEvent.created_at > cursor_ts.replace(tzinfo=None),
                    and_(
                        TraderEvent.created_at == cursor_ts.replace(tzinfo=None),
                        TraderEvent.id > cursor_id,
                    ),
                )
            )

        rows = (await session.execute(query.limit(200))).scalars().all()
        if rows:
            last = rows[-1]
            if last.created_at is not None:
                self._last_event_cursor = (_to_utc(last.created_at), str(last.id))
        return rows

    @staticmethod
    def _is_issue_event(event: TraderEvent) -> bool:
        severity = str(event.severity or "").strip().lower()
        if severity in {"warn", "warning", "error", "critical"}:
            return True

        event_type = str(event.event_type or "").strip().lower()
        payload = event.payload_json or {}

        if event_type == "kill_switch":
            return bool(payload.get("enabled", True))

        if event_type == "live_preflight":
            return str(payload.get("status") or "").lower() == "failed"

        return False

    @staticmethod
    def _is_issue_order(order: TraderOrder) -> bool:
        status = str(order.status or "").strip().lower()
        if status in {"failed", "rejected", "cancelled", "error"}:
            return True
        if order.error_message:
            return True
        return False

    async def _load_trader_name_map(
        self,
        session,
        trader_ids: set[str],
    ) -> dict[str, str]:
        if not trader_ids:
            return {}
        rows = (await session.execute(select(Trader.id, Trader.name).where(Trader.id.in_(tuple(trader_ids))))).all()
        return {str(row[0]): str(row[1]) for row in rows}

    async def _send_issue_alerts(
        self,
        *,
        session,
        events: list[TraderEvent],
        orders: list[TraderOrder],
        snapshot: Optional[TraderOrchestratorSnapshot],
    ) -> None:
        trader_ids = {str(item.trader_id) for item in list(events) + list(orders) if getattr(item, "trader_id", None)}
        trader_names = await self._load_trader_name_map(session, trader_ids)

        for event in events:
            if not self._is_issue_event(event):
                continue
            await self._enqueue(self._format_issue_event_message(event=event, trader_names=trader_names))

        for order in orders:
            if not self._is_issue_order(order):
                continue
            await self._enqueue(self._format_issue_order_message(order=order, trader_names=trader_names))

        snapshot_error = None
        if snapshot is not None and snapshot.last_error:
            snapshot_error = str(snapshot.last_error).strip()

        if snapshot_error != self._last_snapshot_error:
            if snapshot_error:
                await self._enqueue(self._format_worker_issue_message(snapshot_error))
            self._last_snapshot_error = snapshot_error

    async def _send_order_activity_alert(
        self,
        *,
        session,
        orders: list[TraderOrder],
        mode: str,
    ) -> None:
        regular_orders = [row for row in orders if not self._is_issue_order(row)]
        if not regular_orders:
            return

        status_counts: dict[str, int] = defaultdict(int)
        notional = 0.0
        trader_counts: dict[str, int] = defaultdict(int)
        trader_ids: set[str] = set()

        for row in regular_orders:
            status_counts[str(row.status or "unknown").lower()] += 1
            notional += float(row.notional_usd or 0.0)
            if row.trader_id:
                trader_id = str(row.trader_id)
                trader_ids.add(trader_id)
                trader_counts[trader_id] += 1

        trader_names = await self._load_trader_name_map(session, trader_ids)

        status_parts = [
            f"{_escape_md(status)}: {_escape_md(str(count))}" for status, count in sorted(status_counts.items())
        ]

        top_traders = sorted(
            trader_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        trader_parts = [
            f"{_escape_md(trader_names.get(tid, tid[:8]))} {_escape_md(str(count))}" for tid, count in top_traders
        ]

        lines = [
            f"📈 {_bold('Autotrader Orders')}",
            "",
            f"{_bold('Mode:')} {_escape_md(mode)}",
            f"{_bold('New Orders:')} {_escape_md(str(len(regular_orders)))}",
            f"{_bold('Status Mix:')} {_escape_md(', '.join(status_parts) if status_parts else 'none')}",
            f"{_bold('Notional:')} {_escape_md(_format_money(notional))}",
        ]
        if trader_parts:
            lines.append(f"{_bold('Top Traders:')} {_escape_md(', '.join(trader_parts))}")

        await self._enqueue("\n".join(lines))

    async def _maybe_send_timeline_summary(
        self,
        *,
        session,
        control: Optional[TraderOrchestratorControl],
        snapshot: Optional[TraderOrchestratorSnapshot],
    ) -> None:
        now = utcnow()
        interval = timedelta(minutes=self._summary_interval_minutes)

        if self._last_summary_at is None:
            self._last_summary_at = now
            return

        if now - self._last_summary_at < interval:
            return

        start = self._last_summary_at
        end = now

        summary_message = await self._build_timeline_summary(
            session=session,
            start=start,
            end=end,
            control=control,
            snapshot=snapshot,
        )
        await self._enqueue(summary_message)
        self._last_summary_at = now

    async def _build_timeline_summary(
        self,
        *,
        session,
        start: datetime,
        end: datetime,
        control: Optional[TraderOrchestratorControl],
        snapshot: Optional[TraderOrchestratorSnapshot],
    ) -> str:
        start_naive = _to_utc(start).replace(tzinfo=None)
        end_naive = _to_utc(end).replace(tzinfo=None)

        decision_rows = (
            await session.execute(
                select(TraderDecision.decision, func.count(TraderDecision.id))
                .where(
                    TraderDecision.created_at >= start_naive,
                    TraderDecision.created_at < end_naive,
                )
                .group_by(TraderDecision.decision)
            )
        ).all()
        decision_counts = {str(row[0] or "unknown").lower(): int(row[1] or 0) for row in decision_rows}

        order_rows = (
            await session.execute(
                select(
                    TraderOrder.status,
                    func.count(TraderOrder.id),
                    func.coalesce(func.sum(TraderOrder.notional_usd), 0.0),
                )
                .where(
                    TraderOrder.created_at >= start_naive,
                    TraderOrder.created_at < end_naive,
                )
                .group_by(TraderOrder.status)
            )
        ).all()
        order_counts = {str(row[0] or "unknown").lower(): int(row[1] or 0) for row in order_rows}
        total_notional = float(sum(float(row[2] or 0.0) for row in order_rows))

        resolved_pnl = float(
            (
                await session.execute(
                    select(func.coalesce(func.sum(TraderOrder.actual_profit), 0.0)).where(
                        TraderOrder.updated_at >= start_naive,
                        TraderOrder.updated_at < end_naive,
                        TraderOrder.status.in_(("resolved_win", "resolved_loss")),
                    )
                )
            ).scalar()
            or 0.0
        )

        issue_count = int(
            (
                await session.execute(
                    select(func.count(TraderEvent.id)).where(
                        TraderEvent.created_at >= start_naive,
                        TraderEvent.created_at < end_naive,
                        func.lower(TraderEvent.severity).in_(("warn", "warning", "error", "critical")),
                    )
                )
            ).scalar()
            or 0
        )

        mode = str(getattr(control, "mode", "paper") or "paper").upper()
        traders_running = int(getattr(snapshot, "traders_running", 0) if snapshot else 0)
        traders_total = int(getattr(snapshot, "traders_total", 0) if snapshot else 0)

        lines = [
            f"🕒 {_bold('Autotrader Timeline')}",
            "",
            f"{_bold('Window:')} {_escape_md(_to_utc(start).strftime('%Y-%m-%d %H:%M'))} -> {_escape_md(_to_utc(end).strftime('%Y-%m-%d %H:%M'))} UTC",
            f"{_bold('Mode:')} {_escape_md(mode)}",
            f"{_bold('Traders Running:')} {_escape_md(str(traders_running))}/{_escape_md(str(traders_total))}",
            f"{_bold('Decisions:')} {_escape_md(self._format_counts(decision_counts))}",
            f"{_bold('Orders:')} {_escape_md(self._format_counts(order_counts))}",
            f"{_bold('Notional:')} {_escape_md(_format_money(total_notional))}",
            f"{_bold('Resolved P&L:')} {_escape_md(f'{resolved_pnl:+.2f}')}",
            f"{_bold('Issues:')} {_escape_md(str(issue_count))}",
        ]

        if self._summary_per_trader:
            trader_lines = await self._build_per_trader_summary_lines(
                session=session,
                start_naive=start_naive,
                end_naive=end_naive,
            )
            if trader_lines:
                lines.append("")
                lines.append(_bold("Per Trader:"))
                lines.extend(trader_lines)

        return "\n".join(lines)

    async def _build_per_trader_summary_lines(
        self,
        *,
        session,
        start_naive: datetime,
        end_naive: datetime,
    ) -> list[str]:
        order_rows = (
            await session.execute(
                select(
                    TraderOrder.trader_id,
                    func.count(TraderOrder.id),
                    func.coalesce(func.sum(TraderOrder.notional_usd), 0.0),
                )
                .where(
                    TraderOrder.created_at >= start_naive,
                    TraderOrder.created_at < end_naive,
                )
                .group_by(TraderOrder.trader_id)
            )
        ).all()

        decision_rows = (
            await session.execute(
                select(
                    TraderDecision.trader_id,
                    TraderDecision.decision,
                    func.count(TraderDecision.id),
                )
                .where(
                    TraderDecision.created_at >= start_naive,
                    TraderDecision.created_at < end_naive,
                )
                .group_by(TraderDecision.trader_id, TraderDecision.decision)
            )
        ).all()

        issue_rows = (
            await session.execute(
                select(TraderEvent.trader_id, func.count(TraderEvent.id))
                .where(
                    TraderEvent.created_at >= start_naive,
                    TraderEvent.created_at < end_naive,
                    func.lower(TraderEvent.severity).in_(("warn", "warning", "error", "critical")),
                )
                .group_by(TraderEvent.trader_id)
            )
        ).all()

        trader_ids: set[str] = set()
        for row in order_rows:
            if row[0]:
                trader_ids.add(str(row[0]))
        for row in decision_rows:
            if row[0]:
                trader_ids.add(str(row[0]))
        for row in issue_rows:
            if row[0]:
                trader_ids.add(str(row[0]))

        if not trader_ids:
            return []

        names = await self._load_trader_name_map(session, trader_ids)

        decision_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in decision_rows:
            trader_id = str(row[0]) if row[0] else "unknown"
            decision = str(row[1] or "unknown").lower()
            decision_map[trader_id][decision] += int(row[2] or 0)

        issue_map = {(str(row[0]) if row[0] else "unknown"): int(row[1] or 0) for row in issue_rows}

        lines: list[str] = []
        sorted_orders = sorted(order_rows, key=lambda row: int(row[1] or 0), reverse=True)

        for row in sorted_orders[:12]:
            trader_id = str(row[0]) if row[0] else "unknown"
            order_count = int(row[1] or 0)
            notional = float(row[2] or 0.0)
            decision_counts = decision_map.get(trader_id, {})
            selected_count = int(decision_counts.get("selected", 0))
            blocked_count = int(decision_counts.get("blocked", 0))
            skipped_count = int(decision_counts.get("skipped", 0))
            issues = int(issue_map.get(trader_id, 0))

            trader_name = names.get(trader_id, trader_id[:8])
            line = (
                f"- {_escape_md(trader_name)}: "
                f"orders {_escape_md(str(order_count))}, "
                f"notional {_escape_md(_format_money(notional))}, "
                f"selected {_escape_md(str(selected_count))}, "
                f"blocked {_escape_md(str(blocked_count))}, "
                f"skipped {_escape_md(str(skipped_count))}, "
                f"issues {_escape_md(str(issues))}"
            )
            lines.append(line)

        return lines

    @staticmethod
    def _format_counts(counts: dict[str, int]) -> str:
        if not counts:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in sorted(counts.items(), key=lambda item: item[0]))

    def _format_runtime_state_message(
        self,
        *,
        title: str,
        mode: str,
        traders_running: int,
        traders_total: int,
    ) -> str:
        return "\n".join(
            [
                f"ℹ️ {_bold(_escape_md(title))}",
                "",
                f"{_bold('Mode:')} {_escape_md(mode)}",
                f"{_bold('Traders Running:')} {_escape_md(str(traders_running))}/{_escape_md(str(traders_total))}",
            ]
        )

    def _format_issue_event_message(
        self,
        *,
        event: TraderEvent,
        trader_names: dict[str, str],
    ) -> str:
        severity = str(event.severity or "warn").upper()
        trader_name = "System"
        if event.trader_id:
            trader_name = trader_names.get(str(event.trader_id), str(event.trader_id)[:8])

        created_at = _to_utc(event.created_at) or utcnow()
        message = str(event.message or "No details provided")

        lines = [
            f"⚠️ {_bold('Autotrader Issue')}",
            "",
            f"{_bold('Severity:')} {_escape_md(severity)}",
            f"{_bold('Type:')} {_escape_md(str(event.event_type or 'event'))}",
            f"{_bold('Trader:')} {_escape_md(trader_name)}",
            f"{_bold('Message:')} {_escape_md(message[:400])}",
            f"{_bold('Time:')} {_escape_md(created_at.strftime('%Y-%m-%d %H:%M:%S'))} UTC",
        ]
        return "\n".join(lines)

    def _format_issue_order_message(
        self,
        *,
        order: TraderOrder,
        trader_names: dict[str, str],
    ) -> str:
        trader_name = "Unknown"
        if order.trader_id:
            trader_name = trader_names.get(str(order.trader_id), str(order.trader_id)[:8])

        status = str(order.status or "unknown").upper()
        created_at = _to_utc(order.created_at) or utcnow()
        reason = str(order.error_message or order.reason or "Order failure")
        market_question = str(order.market_question or order.market_id or "unknown market")

        lines = [
            f"❌ {_bold('Autotrader Order Issue')}",
            "",
            f"{_bold('Trader:')} {_escape_md(trader_name)}",
            f"{_bold('Status:')} {_escape_md(status)}",
            f"{_bold('Market:')} {_escape_md(market_question[:180])}",
            f"{_bold('Reason:')} {_escape_md(reason[:400])}",
            f"{_bold('Time:')} {_escape_md(created_at.strftime('%Y-%m-%d %H:%M:%S'))} UTC",
        ]
        return "\n".join(lines)

    def _format_worker_issue_message(self, error_text: str) -> str:
        lines = [
            f"❌ {_bold('Autotrader Worker Error')}",
            "",
            f"{_bold('Error:')} {_escape_md(error_text[:700])}",
            f"{_bold('Time:')} {_escape_md(utcnow().strftime('%Y-%m-%d %H:%M:%S'))} UTC",
        ]
        return "\n".join(lines)

    def _can_send_now(self) -> bool:
        now = time.monotonic()
        while self._send_timestamps and self._send_timestamps[0] < now - 60:
            self._send_timestamps.popleft()
        return len(self._send_timestamps) < MAX_MESSAGES_PER_MINUTE

    def _record_send(self) -> None:
        self._send_timestamps.append(time.monotonic())

    async def _enqueue(self, text: str) -> None:
        if not self._bot_token or not self._chat_id:
            return
        if not text:
            return

        body = text.strip()
        if len(body) <= 3900:
            await self._message_queue.put(body)
            return

        lines = body.splitlines()
        chunk = ""
        for line in lines:
            candidate = f"{chunk}\n{line}" if chunk else line
            if len(candidate) <= 3900:
                chunk = candidate
                continue
            if chunk:
                await self._message_queue.put(chunk)
            chunk = line[:3900]
        if chunk:
            await self._message_queue.put(chunk)

    async def _queue_worker(self) -> None:
        while self._running:
            try:
                try:
                    message = await asyncio.wait_for(self._message_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

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

    async def _send_telegram(self, text: str) -> bool:
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

            if resp.status_code == 429:
                body = resp.json()
                retry_after = body.get("parameters", {}).get("retry_after", 5)
                logger.warning("Telegram rate limited, will retry", retry_after=retry_after)
                await asyncio.sleep(retry_after)
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

    async def send_test_message(self) -> bool:
        """Send an immediate test message using currently stored settings."""
        await self._load_settings()
        if not self._bot_token or not self._chat_id:
            return False

        text = (
            f"✅ {_bold('Homerun Autotrader Notifier')}\n\n"
            "Test message received\\.\n"
            "Autotrader Telegram alerts are configured correctly\\."
        )
        return await self._send_telegram(text)


notifier = TelegramNotifier()
