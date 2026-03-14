"""Telegram notification service focused on trader-orchestrator activity."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from typing import Any, Optional

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
    TraderPosition,
)
from services.pause_state import global_pause_state
from services.trader_orchestrator_state import (
    OPEN_ORDER_STATUSES,
    ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS,
    arm_live_start,
    cleanup_trader_open_orders,
    consume_live_arm_token,
    create_live_preflight,
    create_trader_event,
    get_daily_realized_pnl,
    get_gross_exposure,
    get_trader,
    list_traders,
    read_orchestrator_control,
    read_orchestrator_snapshot,
    set_trader_paused,
    sync_trader_position_inventory,
    update_orchestrator_control,
    write_orchestrator_snapshot,
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
COMMAND_POLL_TIMEOUT_SECONDS = 25
COMMAND_LOOP_IDLE_SECONDS = 3
CLOSE_ALERT_MARKER_LIMIT = 4000
CLOSE_ALERT_BATCH_LIMIT = 10
TELEGRAM_UPDATE_DRAIN_LIMIT = 50
REALIZED_ORDER_STATUSES = {"resolved_win", "resolved_loss", "closed_win", "closed_loss", "win", "loss"}
ISSUE_ORDER_STATUSES = {"failed", "rejected", "cancelled", "error"}
ACTIVE_POSITION_STATUS = "open"
AI_CHAT_MAX_HISTORY = 20
AI_CHAT_MAX_TOKENS = 1024


def _escape_html(text: Any) -> str:
    return html_escape(str(text), quote=False)


def _bold(text: str) -> str:
    return f"<b>{text}</b>"


def _inline_code(text: Any) -> str:
    return f"<code>{_escape_html(text)}</code>"


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


def _format_signed_money(value: float) -> str:
    amount = float(value or 0.0)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):,.2f}"


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return f"{text[: limit - 3]}..."


def _compact_join(parts: list[str], *, max_items: int) -> str:
    compact_parts = [str(part).strip() for part in parts if str(part).strip()]
    if not compact_parts:
        return "none"
    selected = compact_parts[:max_items]
    overflow = len(compact_parts) - len(selected)
    if overflow > 0:
        selected.append(f"+{overflow} more")
    return " · ".join(selected)


def _mono(text: str) -> str:
    return f"<pre>{_escape_html(text)}</pre>"


def _pad_right(text: str, width: int) -> str:
    return text + " " * max(0, width - len(text))


def _pad_left(text: str, width: int) -> str:
    return " " * max(0, width - len(text)) + text


def _build_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    """Build a borderless fixed-width table for Telegram <pre> blocks."""
    if not rows:
        return "(empty)"
    col_count = len(headers)
    if align is None:
        align = ["l"] * col_count
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:col_count]):
            widths[i] = max(widths[i], len(cell))
    def _fmt_row(cells: list[str]) -> str:
        parts = []
        for i in range(col_count):
            cell = cells[i] if i < len(cells) else ""
            w = widths[i]
            if i < len(align) and align[i] == "r":
                parts.append(_pad_left(cell, w))
            else:
                parts.append(_pad_right(cell, w))
        return "  ".join(parts)

    separator = "  ".join("-" * width for width in widths)
    lines = [_fmt_row(headers), separator]
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def _normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode == "paper":
        return "shadow"
    if mode in {"shadow", "live"}:
        return mode
    return "other"


def _mode_label(value: Any) -> str:
    mode = _normalize_mode(value)
    if mode == "other":
        return "OTHER"
    return mode.upper()


def _mode_label_from_orders(orders: list[TraderOrder], fallback: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in orders:
        counts[_mode_label(getattr(row, "mode", None))] += 1
    if not counts:
        return fallback
    if len(counts) == 1:
        return next(iter(counts.keys()))
    parts = [f"{label} {counts[label]}" for label in sorted(counts.keys())]
    return f"MIXED ({', '.join(parts)})"


class TelegramNotifier:
    """Telegram notifier that tracks autotrader-only runtime activity."""

    def __init__(self) -> None:
        self._bot_token: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._notifications_enabled: bool = False

        self._notify_autotrader_orders: bool = False
        self._notify_autotrader_closes: bool = True
        self._notify_autotrader_issues: bool = True
        self._notify_autotrader_timeline: bool = True
        self._summary_interval_minutes: int = DEFAULT_SUMMARY_INTERVAL_MINUTES
        self._summary_per_trader: bool = False

        self._send_timestamps: deque[float] = deque()
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._queue_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._command_task: Optional[asyncio.Task] = None

        self._http_client: Optional[httpx.AsyncClient] = None

        self._running = False
        self._started = False
        self._delivery_ready = False

        self._autotrader_active: bool = False
        self._last_summary_at: Optional[datetime] = None
        self._last_snapshot_error: Optional[str] = None
        self._last_settings_reload_monotonic: float = 0.0

        self._last_order_cursor: Optional[tuple[datetime, str]] = None
        self._last_order_update_cursor: Optional[tuple[datetime, str]] = None
        self._last_event_cursor: Optional[tuple[datetime, str]] = None
        self._telegram_update_offset: Optional[int] = None
        self._close_alert_markers: dict[str, str] = {}
        self._ws_alert_sent: bool = False

        self._ai_chat_histories: dict[str, list[dict[str, str]]] = {}

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
        await self._prime_telegram_update_offset()

        self._queue_task = asyncio.create_task(self._queue_worker())
        self._monitor_task = asyncio.create_task(self._autotrader_monitor_loop())
        self._command_task = asyncio.create_task(self._telegram_command_loop())

        logger.info("Telegram notifier started (autotrader mode)")

    def stop(self) -> None:
        """Stop notifier tasks."""
        self._running = False
        for task in (self._queue_task, self._monitor_task, self._command_task):
            if task and not task.done():
                task.cancel()
        logger.info("Telegram notifier stopped")

    async def shutdown(self) -> None:
        """Stop and close network resources."""
        self.stop()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _replace_http_client(self) -> None:
        existing_client = self._http_client
        self._http_client = httpx.AsyncClient(timeout=15.0)
        if existing_client is not None:
            try:
                await existing_client.aclose()
            except Exception:
                pass

    async def _load_settings(self) -> None:
        """Load notifier configuration from DB with environment fallback."""
        row: Optional[AppSettings] = None
        had_existing_settings = any(
            (
                self._bot_token,
                self._chat_id,
                self._notifications_enabled,
                self._last_settings_reload_monotonic > 0.0,
            )
        )
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                row = result.scalar_one_or_none()
        except Exception as exc:
            if had_existing_settings:
                logger.warning(
                    "Could not refresh notifier settings from DB; keeping current notifier settings",
                    exc_info=exc,
                )
                return
            logger.warning(
                "Could not load notifier settings from DB, using config.py",
                exc_info=exc,
            )

        if row is None:
            self._bot_token = settings.TELEGRAM_BOT_TOKEN
            self._chat_id = settings.TELEGRAM_CHAT_ID
            self._notifications_enabled = bool(self._bot_token and self._chat_id)
            self._notify_autotrader_orders = False
            self._notify_autotrader_closes = True
            self._notify_autotrader_issues = True
            self._notify_autotrader_timeline = True
            self._summary_interval_minutes = DEFAULT_SUMMARY_INTERVAL_MINUTES
            self._summary_per_trader = False
            return

        db_bot_token = decrypt_secret(row.telegram_bot_token)
        db_bot_token = str(db_bot_token).strip() if db_bot_token is not None else ""
        db_chat_id = str(getattr(row, "telegram_chat_id", "") or "").strip()
        env_bot_token = str(getattr(settings, "TELEGRAM_BOT_TOKEN", "") or "").strip()
        env_chat_id = str(getattr(settings, "TELEGRAM_CHAT_ID", "") or "").strip()

        # Runtime precedence: DB non-null override > env > default(None).
        self._bot_token = db_bot_token or env_bot_token or None
        self._chat_id = db_chat_id or env_chat_id or None
        if row.notifications_enabled is None:
            self._notifications_enabled = bool(self._bot_token and self._chat_id)
        else:
            self._notifications_enabled = bool(row.notifications_enabled)

        # Legacy bridge: if new toggles are missing for any reason, fall back to old flags.
        self._notify_autotrader_orders = bool(getattr(row, "notify_autotrader_orders", bool(row.notify_on_trade)))
        self._notify_autotrader_closes = bool(getattr(row, "notify_autotrader_closes", True))
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

                latest_order_update = (
                    await session.execute(
                        select(TraderOrder.updated_at, TraderOrder.id)
                        .where(TraderOrder.updated_at.is_not(None))
                        .order_by(TraderOrder.updated_at.desc(), TraderOrder.id.desc())
                        .limit(1)
                    )
                ).first()
                if latest_order_update and latest_order_update[0] is not None:
                    self._last_order_update_cursor = (_to_utc(latest_order_update[0]), str(latest_order_update[1]))

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
                self._close_alert_markers = {}
        except Exception as exc:
            logger.debug("Notifier cursor priming skipped", exc_info=exc)

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
                    mode = _mode_label(getattr(control, "mode", "shadow"))

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
                    updated_orders = await self._load_updated_trader_orders(session)

                    if active and self._notify_autotrader_issues:
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

                    if active and self._notify_autotrader_closes:
                        await self._send_position_close_alerts(
                            session=session,
                            orders=updated_orders,
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

                # WS feed health monitoring (runs regardless of orchestrator state)
                try:
                    from services.ws_feeds import get_feed_manager
                    ws_health = get_feed_manager().health_check()
                    poly_failures = ws_health.get("polymarket", {}).get("stats", {}).get("consecutive_failures", 0)
                    if poly_failures >= 10 and not self._ws_alert_sent:
                        self._ws_alert_sent = True
                        await self._enqueue(
                            f"⚠️ {_bold('WebSocket Feed Down')}\n"
                            "Polymarket WS disconnected.\n"
                            f"{_escape_html(poly_failures)} consecutive failures.\n"
                            "Auto-trader paused."
                        )
                    elif poly_failures == 0 and self._ws_alert_sent:
                        self._ws_alert_sent = False
                        await self._enqueue(
                            f"✅ {_bold('WebSocket Feed Recovered')}\n"
                            "Polymarket WS reconnected."
                        )
                except Exception:
                    pass  # WS feeds may not be initialized in this process

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Autotrader notifier monitor error", exc_info=exc)

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

    async def _load_updated_trader_orders(self, session) -> list[TraderOrder]:
        query = (
            select(TraderOrder)
            .where(TraderOrder.updated_at.is_not(None))
            .order_by(TraderOrder.updated_at.asc(), TraderOrder.id.asc())
        )

        if self._last_order_update_cursor is not None:
            cursor_ts, cursor_id = self._last_order_update_cursor
            query = query.where(
                or_(
                    TraderOrder.updated_at > cursor_ts.replace(tzinfo=None),
                    and_(
                        TraderOrder.updated_at == cursor_ts.replace(tzinfo=None),
                        TraderOrder.id > cursor_id,
                    ),
                )
            )

        rows = (await session.execute(query.limit(300))).scalars().all()
        if rows:
            last = rows[-1]
            if last.updated_at is not None:
                self._last_order_update_cursor = (_to_utc(last.updated_at), str(last.id))
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
        if status in ISSUE_ORDER_STATUSES:
            return True
        if order.error_message:
            return True
        return False

    @staticmethod
    def _is_realized_order(order: TraderOrder) -> bool:
        return str(order.status or "").strip().lower() in REALIZED_ORDER_STATUSES

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
        mode_label = _mode_label_from_orders(regular_orders, _mode_label(mode))

        status_counts: dict[str, int] = defaultdict(int)
        status_notional: dict[str, float] = defaultdict(float)
        notional = 0.0
        trader_counts: dict[str, int] = defaultdict(int)
        trader_notional: dict[str, float] = defaultdict(float)
        trader_ids: set[str] = set()

        for row in regular_orders:
            status = str(row.status or "unknown").lower()
            order_notional = float(row.notional_usd or 0.0)

            status_counts[status] += 1
            status_notional[status] += order_notional
            notional += order_notional
            if row.trader_id:
                trader_id = str(row.trader_id)
                trader_ids.add(trader_id)
                trader_counts[trader_id] += 1
                trader_notional[trader_id] += order_notional

        trader_names = await self._load_trader_name_map(session, trader_ids)
        status_parts = [
            f"{status.upper()} {status_counts[status]}/{_format_money(status_notional[status])}"
            for status in sorted(status_counts.keys())
        ]

        top_traders = sorted(
            trader_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        trader_parts = [
            f"{_truncate_text(trader_names.get(tid, tid[:8]), 12)} {count}/{_format_money(trader_notional.get(tid, 0.0))}"
            for tid, count in top_traders
        ]

        lines = [
            f"📈 {_bold('Autotrader Orders')} · {_escape_html(mode_label)}",
            f"{_bold('New:')} {_escape_html(len(regular_orders))} · {_bold('Notional:')} {_escape_html(_format_money(notional))}",
            f"{_bold('Mix:')} {_escape_html(_compact_join(status_parts, max_items=3))}",
        ]
        if trader_parts:
            lines.append(f"{_bold('Top:')} {_escape_html(_compact_join(trader_parts, max_items=2))}")

        await self._enqueue("\n".join(lines))

    async def _send_position_close_alerts(
        self,
        *,
        session,
        orders: list[TraderOrder],
        mode: str,
    ) -> None:
        realized_rows: dict[str, TraderOrder] = {}

        for row in orders:
            if not self._is_realized_order(row):
                continue
            updated_at = _to_utc(row.updated_at) or _to_utc(row.created_at) or utcnow()
            marker = f"{str(row.status or '').lower()}@{updated_at.isoformat()}"
            order_id = str(row.id)
            if self._close_alert_markers.get(order_id) == marker:
                continue
            self._close_alert_markers[order_id] = marker
            realized_rows[order_id] = row

        if not realized_rows:
            return

        while len(self._close_alert_markers) > CLOSE_ALERT_MARKER_LIMIT:
            oldest = next(iter(self._close_alert_markers.keys()), None)
            if oldest is None:
                break
            self._close_alert_markers.pop(oldest, None)

        rows = sorted(
            realized_rows.values(),
            key=lambda row: (_to_utc(row.updated_at) or _to_utc(row.created_at) or utcnow(), str(row.id)),
        )
        mode_label = _mode_label_from_orders(rows, _mode_label(mode))
        trader_ids = {str(row.trader_id) for row in rows if row.trader_id}
        trader_names = await self._load_trader_name_map(session, trader_ids)

        title = "Autotrader Position Close" if len(rows) == 1 else "Autotrader Position Closes"
        realized_won = 0.0
        realized_lost = 0.0
        close_lines: list[str] = []
        trigger_parts: list[str] = []
        display_limit = min(CLOSE_ALERT_BATCH_LIMIT, 5)

        for index, row in enumerate(rows, start=1):
            trader_name = trader_names.get(str(row.trader_id), str(row.trader_id)[:8] if row.trader_id else "Unknown")
            status = str(row.status or "closed").lower()
            status_label = status.replace("_", " ").upper()
            pnl = float(row.actual_profit or 0.0)
            market = str(row.market_question or row.market_id or "unknown market")
            payload = row.payload_json if isinstance(row.payload_json, dict) else {}
            position_close = payload.get("position_close") if isinstance(payload, dict) else {}
            close_trigger = ""
            if isinstance(position_close, dict):
                close_trigger = str(position_close.get("close_trigger") or "").strip()

            if pnl >= 0:
                realized_won += pnl
            else:
                realized_lost += abs(pnl)

            if index <= display_limit:
                status_short = "WIN" if "win" in status else "LOSS" if "loss" in status else _truncate_text(status_label, 10)
                close_lines.append(
                    f"{_escape_html(index)}) {_escape_html(_truncate_text(trader_name, 13))} · "
                    f"{_escape_html(status_short)} · {_escape_html(_format_signed_money(pnl))}"
                )
                close_lines.append(f"   {_escape_html(_truncate_text(market, 56))}")
            if close_trigger:
                trigger_parts.append(
                    f"{index}:{_truncate_text(close_trigger, 20)}"
                )

        net_realized = realized_won - realized_lost
        lines = [
            f"✅ {_bold(title)} · {_escape_html(mode_label)}",
            (
                f"{_bold('Count:')} {_escape_html(len(rows))} · "
                f"{_bold('Won:')} {_escape_html(_format_money(realized_won))} · "
                f"{_bold('Lost:')} {_escape_html(_format_money(realized_lost))} · "
                f"{_bold('Net:')} {_escape_html(_format_signed_money(net_realized))}"
            ),
        ]
        lines.extend(close_lines)
        if trigger_parts:
            lines.append(f"{_bold('Triggers:')} {_escape_html(_compact_join(trigger_parts, max_items=3))}")

        remaining = len(rows) - display_limit
        if remaining > 0:
            lines.append(f"{_bold('More:')} {_escape_html(remaining)} close(s)")

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
        title: str = "Autotrader Timeline",
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

        realized_rows = (
            await session.execute(
                select(
                    TraderOrder.status,
                    func.count(TraderOrder.id),
                    func.coalesce(func.sum(TraderOrder.actual_profit), 0.0),
                )
                .where(
                    TraderOrder.updated_at >= start_naive,
                    TraderOrder.updated_at < end_naive,
                    TraderOrder.status.in_(tuple(REALIZED_ORDER_STATUSES)),
                )
                .group_by(TraderOrder.status)
            )
        ).all()

        realized_won = 0.0
        realized_lost = 0.0
        for row in realized_rows:
            pnl = float(row[2] or 0.0)
            if pnl >= 0:
                realized_won += pnl
            else:
                realized_lost += abs(pnl)
        realized_net = realized_won - realized_lost

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

        mode = _mode_label(getattr(control, "mode", "shadow"))
        traders_running = int(getattr(snapshot, "traders_running", 0) if snapshot else 0)
        traders_total = int(getattr(snapshot, "traders_total", 0) if snapshot else 0)

        decision_parts = [
            f"{key.upper()} {decision_counts[key]}"
            for key in sorted(decision_counts.keys(), key=lambda name: (-decision_counts[name], name))
        ]
        order_parts = [
            f"{str(row[0] or 'unknown').upper()} {int(row[1] or 0)}/{_format_money(float(row[2] or 0.0))}"
            for row in sorted(order_rows, key=lambda item: (-int(item[1] or 0), str(item[0] or "unknown")))
        ]
        realized_parts = [
            f"{str(row[0] or 'unknown').upper()} {int(row[1] or 0)}/{_format_signed_money(float(row[2] or 0.0))}"
            for row in sorted(realized_rows, key=lambda item: (-int(item[1] or 0), str(item[0] or "unknown")))
        ]

        lines = [
            f"🕒 {_bold(_escape_html(title))}",
            f"{_bold('Window:')} {_escape_html(_to_utc(start).strftime('%Y-%m-%d %H:%M'))} -> {_escape_html(_to_utc(end).strftime('%Y-%m-%d %H:%M'))} UTC",
            f"{_bold('Mode:')} {_escape_html(mode)} · {_bold('Traders:')} {_escape_html(traders_running)}/{_escape_html(traders_total)}",
            f"{_bold('Notional:')} {_escape_html(_format_money(total_notional))} · {_bold('Issues:')} {_escape_html(issue_count)}",
            (
                f"{_bold('Won:')} {_escape_html(_format_money(realized_won))} · "
                f"{_bold('Lost:')} {_escape_html(_format_money(realized_lost))} · "
                f"{_bold('Net:')} {_escape_html(_format_signed_money(realized_net))}"
            ),
        ]

        if decision_counts:
            lines.append(f"{_bold('Decisions:')} {_escape_html(_compact_join(decision_parts, max_items=4))}")
        if order_counts:
            lines.append(f"{_bold('Orders:')} {_escape_html(_compact_join(order_parts, max_items=4))}")
        if realized_rows:
            lines.append(f"{_bold('Realized:')} {_escape_html(_compact_join(realized_parts, max_items=4))}")

        if self._summary_per_trader:
            trader_lines = await self._build_per_trader_summary_lines(
                session=session,
                start_naive=start_naive,
                end_naive=end_naive,
            )
            if trader_lines:
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
                f"- {_escape_html(trader_name)}: "
                f"orders {_escape_html(order_count)}, "
                f"notional {_escape_html(_format_money(notional))}, "
                f"selected {_escape_html(selected_count)}, "
                f"blocked {_escape_html(blocked_count)}, "
                f"skipped {_escape_html(skipped_count)}, "
                f"issues {_escape_html(issues)}"
            )
            lines.append(line)

        return lines

    @staticmethod
    def _format_utc_timestamp(value: Any) -> str:
        if isinstance(value, datetime):
            dt = _to_utc(value)
            if dt is not None:
                return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "n/a"

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
                f"ℹ️ {_bold(_escape_html(title))}",
                "",
                f"{_bold('Mode:')} {_escape_html(mode)}",
                f"{_bold('Traders Running:')} {_escape_html(traders_running)}/{_escape_html(traders_total)}",
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
            f"{_bold('Severity:')} {_escape_html(severity)}",
            f"{_bold('Type:')} {_escape_html(str(event.event_type or 'event'))}",
            f"{_bold('Trader:')} {_escape_html(trader_name)}",
            f"{_bold('Message:')} {_escape_html(message[:400])}",
            f"{_bold('Time:')} {_escape_html(created_at.strftime('%Y-%m-%d %H:%M:%S'))} UTC",
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
            f"{_bold('Trader:')} {_escape_html(trader_name)}",
            f"{_bold('Status:')} {_escape_html(status)}",
            f"{_bold('Market:')} {_escape_html(market_question[:180])}",
            f"{_bold('Reason:')} {_escape_html(reason[:400])}",
            f"{_bold('Time:')} {_escape_html(created_at.strftime('%Y-%m-%d %H:%M:%S'))} UTC",
        ]
        return "\n".join(lines)

    def _format_worker_issue_message(self, error_text: str) -> str:
        lines = [
            f"❌ {_bold('Autotrader Worker Error')}",
            "",
            f"{_bold('Error:')} {_escape_html(error_text[:700])}",
            f"{_bold('Time:')} {_escape_html(utcnow().strftime('%Y-%m-%d %H:%M:%S'))} UTC",
        ]
        return "\n".join(lines)

    async def _prime_telegram_update_offset(self) -> None:
        if not self._bot_token or not self._chat_id:
            self._telegram_update_offset = None
            return
        for _ in range(TELEGRAM_UPDATE_DRAIN_LIMIT):
            updates = await self._fetch_telegram_updates(timeout_seconds=0)
            if not updates:
                break

    async def _fetch_telegram_updates(self, *, timeout_seconds: int) -> list[dict[str, Any]]:
        if not self._bot_token:
            return []
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=15.0)

        url = f"{TELEGRAM_API_BASE}/bot{self._bot_token}/getUpdates"
        payload: dict[str, Any] = {
            "timeout": max(0, int(timeout_seconds)),
            "limit": 100,
            "allowed_updates": ["message"],
        }
        if self._telegram_update_offset is not None:
            payload["offset"] = int(self._telegram_update_offset)

        try:
            response = await self._http_client.post(url, json=payload, timeout=timeout_seconds + 10)
            if response.status_code == 429:
                retry_after = 5
                try:
                    body = response.json()
                    retry_after = int(body.get("parameters", {}).get("retry_after", 5) or 5)
                except Exception:
                    retry_after = 5
                logger.warning("Telegram getUpdates rate limited", retry_after=retry_after)
                await asyncio.sleep(max(1, retry_after))
                return []

            if response.status_code != 200:
                logger.warning("Telegram getUpdates failed", status=response.status_code, body=response.text[:250])
                return []

            body = response.json()
            if not isinstance(body, dict) or not body.get("ok"):
                logger.warning("Telegram getUpdates returned invalid payload")
                return []

            result = body.get("result")
            if not isinstance(result, list):
                return []

            updates = [item for item in result if isinstance(item, dict)]
            if updates:
                update_ids = [int(item.get("update_id")) for item in updates if isinstance(item.get("update_id"), int)]
                if update_ids:
                    self._telegram_update_offset = max(update_ids) + 1
            return updates
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException:
            return []
        except httpx.NetworkError as exc:
            logger.warning("Telegram getUpdates network error (%s); resetting HTTP client", type(exc).__name__)
            await self._replace_http_client()
            return []
        except Exception as exc:
            logger.error("Failed to poll Telegram updates", exc_info=exc)
            return []

    async def _telegram_command_loop(self) -> None:
        while self._running:
            try:
                now_monotonic = time.monotonic()
                if now_monotonic - self._last_settings_reload_monotonic >= SETTINGS_REFRESH_SECONDS:
                    await self._load_settings()
                    self._last_settings_reload_monotonic = now_monotonic

                if not self._bot_token or not self._chat_id:
                    await asyncio.sleep(COMMAND_LOOP_IDLE_SECONDS)
                    continue

                updates = await self._fetch_telegram_updates(timeout_seconds=COMMAND_POLL_TIMEOUT_SECONDS)
                for update in updates:
                    await self._process_telegram_update(update)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Telegram command loop failed", exc_info=exc)
                await asyncio.sleep(COMMAND_LOOP_IDLE_SECONDS)

    async def _process_telegram_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return

        chat = message.get("chat")
        if not isinstance(chat, dict):
            return
        chat_id = str(chat.get("id") or "").strip()
        if not chat_id:
            return

        expected_chat_id = str(self._chat_id or "").strip()
        if expected_chat_id and chat_id != expected_chat_id:
            logger.warning("Ignoring Telegram command from unexpected chat", chat_id=chat_id)
            return

        text = str(message.get("text") or "").strip()
        if not text:
            return

        operator = "telegram"
        sender = message.get("from")
        if isinstance(sender, dict):
            username = str(sender.get("username") or "").strip()
            sender_id = str(sender.get("id") or "").strip()
            if username:
                operator = f"telegram:{username}"
            elif sender_id:
                operator = f"telegram:{sender_id}"

        try:
            response = await self._handle_telegram_command(text=text, operator=operator, chat_id=chat_id)
        except Exception as exc:
            logger.error("Telegram command processing failed", exc_info=exc)
            response = "❌ Telegram command failed.\nPlease retry or check backend logs."
        if response:
            await self._enqueue(response)

    async def _handle_telegram_command(self, *, text: str, operator: str, chat_id: str) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return ""

        parts = stripped.split()
        head = str(parts[0] or "").strip()
        args = [str(item or "").strip() for item in parts[1:] if str(item or "").strip()]

        if head.startswith("/"):
            command = head.split("@", 1)[0].lower()
        else:
            command = head.lower()

        if command in {"/help", "/commands", "help", "commands"}:
            return self._telegram_help_message()
        if command in {"/status", "status"}:
            return await self._telegram_status_message()
        if command in {"/performance", "/pnl", "performance", "pnl"}:
            hours_arg = args[0] if args else None
            return await self._telegram_performance_message(hours_arg=hours_arg)
        if command in {"/stop", "stop", "/off", "off", "/disable", "disable"}:
            return await self._telegram_stop_autotrader(operator=operator)
        if command in {"/start", "start", "/on", "on", "/enable", "enable"}:
            mode_arg = self._parse_telegram_start_args(args)
            return await self._telegram_start_autotrader(
                mode_arg=mode_arg,
                operator=operator,
            )
        if command in {"/autotrader", "autotrader"}:
            if not args:
                return f"Usage: {_inline_code('/autotrader on [shadow|live]')} or {_inline_code('/autotrader off')}"
            action = args[0].strip().lower()
            action_args = args[1:]
            if action in {"on", "start", "enable", "enabled", "resume"}:
                mode_arg = self._parse_telegram_start_args(action_args)
                return await self._telegram_start_autotrader(
                    mode_arg=mode_arg,
                    operator=operator,
                )
            if action in {"off", "stop", "disable", "disabled", "pause"}:
                return await self._telegram_stop_autotrader(operator=operator)
            return f"Usage: {_inline_code('/autotrader on [shadow|live]')} or {_inline_code('/autotrader off')}"
        if command in {"/killswitch", "/kill", "killswitch", "kill"}:
            if not args:
                return f"Usage: {_inline_code('/killswitch on')} or {_inline_code('/killswitch off')}"
            toggle = args[0].strip().lower()
            if toggle in {"on", "1", "true", "enable", "enabled"}:
                return await self._telegram_set_kill_switch(enabled=True, operator=operator)
            if toggle in {"off", "0", "false", "disable", "disabled"}:
                return await self._telegram_set_kill_switch(enabled=False, operator=operator)
            return f"Usage: {_inline_code('/killswitch on')} or {_inline_code('/killswitch off')}"
        if command in {"/traders", "traders"}:
            return await self._telegram_traders_message()
        if command in {"/trader", "trader"}:
            if not args:
                return f"Usage: {_inline_code('/trader <name or id>')}"
            return await self._telegram_trader_detail_message(query=" ".join(args))
        if command in {"/orders", "orders"}:
            limit_arg = args[0] if args else None
            return await self._telegram_orders_message(limit_arg=limit_arg)
        if command in {"/positions", "positions"}:
            return await self._telegram_positions_message()
        if command in {"/exposure", "exposure"}:
            return await self._telegram_exposure_message()
        if command in {"/pause"}:
            if not args:
                return f"Usage: {_inline_code('/pause <trader name or id>')}"
            return await self._telegram_pause_trader(query=" ".join(args), operator=operator)
        if command in {"/resume"}:
            if not args:
                return f"Usage: {_inline_code('/resume <trader name or id>')}"
            return await self._telegram_resume_trader(query=" ".join(args), operator=operator)
        if command in {"/close"}:
            if not args:
                return f"Usage: {_inline_code('/close <trader name or id>')} - closes all open positions for the trader"
            return await self._telegram_close_trader_positions(query=" ".join(args), operator=operator)
        if command in {"/sell", "/cancel"}:
            if not args:
                return f"Usage: {_inline_code('/sell <order_id>')} - cancel/close an open order"
            return await self._telegram_sell_order(order_id=args[0], operator=operator)
        if command in {"/ask", "ask"}:
            if not args:
                return f"Usage: {_inline_code('/ask <question>')} - ask AI about your traders and data"
            return await self._telegram_ai_chat(
                message=" ".join(args),
                chat_id=chat_id,
            )
        if command in {"/clear"}:
            self._ai_chat_histories.pop(chat_id, None)
            return "Chat history cleared."

        # If not a recognized command, route to AI chat
        if not head.startswith("/"):
            return await self._telegram_ai_chat(message=stripped, chat_id=chat_id)

        return f"Unknown command.\nUse {_inline_code('/help')} for available Telegram commands."

    @staticmethod
    def _parse_telegram_start_args(args: list[str]) -> Optional[str]:
        mode_arg: Optional[str] = None
        if args:
            first = args[0].lower()
            if first in {"shadow", "live"}:
                mode_arg = first
        return mode_arg

    @staticmethod
    def _telegram_help_message() -> str:
        return "\n".join(
            [
                f"🤖 {_bold('Homerun Telegram Commands')}",
                "",
                _bold("Status & Monitoring"),
                f"{_inline_code('/status')} - orchestrator state and metrics",
                f"{_inline_code('/traders')} - list all traders with status",
                f"{_inline_code('/trader <name>')} - detailed trader view",
                f"{_inline_code('/orders [N]')} - recent orders (default 10)",
                f"{_inline_code('/positions')} - all open positions",
                f"{_inline_code('/exposure')} - gross exposure breakdown",
                f"{_inline_code('/performance [hours]')} - timeline summary (default 24h)",
                "",
                _bold("Orchestrator Control"),
                f"{_inline_code('/start [shadow|live]')} - start autotrader",
                f"{_inline_code('/stop')} - stop autotrader",
                f"{_inline_code('/autotrader on|off')} - quick power toggle",
                f"{_inline_code('/killswitch on|off')} - emergency kill switch",
                "",
                _bold("Trader Control"),
                f"{_inline_code('/pause <trader>')} - pause a trader",
                f"{_inline_code('/resume <trader>')} - resume a trader",
                f"{_inline_code('/close <trader>')} - close all open positions",
                f"{_inline_code('/sell <order_id>')} - cancel/close an order",
                "",
                _bold("AI Chat"),
                f"{_inline_code('/ask <question>')} - ask AI about your data",
                "Or just type a message to chat with AI",
                f"{_inline_code('/clear')} - reset AI chat history",
            ]
        )

    async def _telegram_status_message(self) -> str:
        async with AsyncSessionLocal() as session:
            control = await read_orchestrator_control(session)
            snapshot = await read_orchestrator_snapshot(session)
            cutoff = (utcnow() - timedelta(hours=24)).replace(tzinfo=None)
            realized_rows = (
                await session.execute(
                    select(
                        TraderOrder.status,
                        func.coalesce(func.sum(TraderOrder.actual_profit), 0.0),
                    )
                    .where(
                        TraderOrder.updated_at >= cutoff,
                        TraderOrder.status.in_(tuple(REALIZED_ORDER_STATUSES)),
                    )
                    .group_by(TraderOrder.status)
                )
            ).all()
            exposure = await get_gross_exposure(session)

        realized_won = 0.0
        realized_lost = 0.0
        for row in realized_rows:
            pnl = float(row[1] or 0.0)
            if pnl >= 0:
                realized_won += pnl
            else:
                realized_lost += abs(pnl)
        realized_24h = realized_won - realized_lost

        mode = _mode_label(control.get("mode", "shadow"))
        enabled = bool(control.get("is_enabled", False))
        paused = bool(control.get("is_paused", True))
        running = bool(snapshot.get("running", False))
        state = "ACTIVE" if enabled and not paused else "PAUSED"
        kill_switch = bool(control.get("kill_switch", False))
        traders_running = int(snapshot.get("traders_running", 0) or 0)
        traders_total = int(snapshot.get("traders_total", 0) or 0)
        decisions_count = int(snapshot.get("decisions_count", 0) or 0)
        orders_count = int(snapshot.get("orders_count", 0) or 0)
        open_orders = int(snapshot.get("open_orders", 0) or 0)
        daily_pnl = float(snapshot.get("daily_pnl", 0.0) or 0.0)
        last_run = self._format_utc_timestamp(snapshot.get("last_run_at"))

        table_rows = [
            ["State", f"{state} / {'running' if running else 'idle'}"],
            ["Mode", mode],
            ["Kill Switch", "ON" if kill_switch else "OFF"],
            ["Traders", f"{traders_running}/{traders_total}"],
            ["Decisions", str(decisions_count)],
            ["Orders", f"{orders_count} ({open_orders} open)"],
            ["Exposure", _format_money(exposure)],
            ["Daily PnL", _format_signed_money(daily_pnl)],
            ["24h Won", _format_money(realized_won)],
            ["24h Lost", _format_money(realized_lost)],
            ["24h Net", _format_signed_money(realized_24h)],
            ["Last Run", last_run],
        ]
        table = _build_table(["Metric", "Value"], table_rows)
        return f"📊 {_bold('Autotrader Status')}\n{_mono(table)}"

    async def _telegram_performance_message(self, *, hours_arg: Optional[str]) -> str:
        hours = _clamp_int(hours_arg, 1, 168, 24)
        end = utcnow()
        start = end - timedelta(hours=hours)

        async with AsyncSessionLocal() as session:
            control_row = await session.get(TraderOrchestratorControl, "default")
            snapshot_row = await session.get(TraderOrchestratorSnapshot, "latest")
            return await self._build_timeline_summary(
                session=session,
                start=start,
                end=end,
                control=control_row,
                snapshot=snapshot_row,
                title=f"Autotrader Performance ({hours}h)",
            )

    async def _telegram_start_autotrader(
        self,
        *,
        mode_arg: Optional[str],
        operator: str,
    ) -> str:
        if global_pause_state.is_paused:
            return "❌ Global pause is active.\nResume workers before starting autotrader."

        mode = str(mode_arg or "").strip().lower()
        if mode and mode not in {"shadow", "live"}:
            return f"❌ Invalid mode.\nUse {_inline_code('shadow')} or {_inline_code('live')}."

        async with AsyncSessionLocal() as session:
            control_before = await read_orchestrator_control(session)
            selected_mode = mode or str(control_before.get("mode") or "shadow").strip().lower() or "shadow"
            if selected_mode not in {"shadow", "live"}:
                selected_mode = "shadow"
            if selected_mode == "live":
                preflight = await create_live_preflight(
                    session,
                    requested_mode="live",
                    requested_by=operator,
                )
                if str(preflight.get("status") or "").strip().lower() != "passed":
                    failed_checks = preflight.get("failed_checks") or []
                    failed_ids = [
                        str(check.get("id") or "").strip()
                        for check in failed_checks
                        if isinstance(check, dict) and str(check.get("id") or "").strip()
                    ]
                    failed_summary = ", ".join(failed_ids[:6]) if failed_ids else "preflight checks failed"
                    return (
                        "❌ Live preflight failed.\n"
                        f"Checks: {_escape_html(failed_summary)}.\n"
                        f"Review live setup in the UI and retry {_inline_code('/autotrader on live')}."
                    )

                arm_response = await arm_live_start(
                    session,
                    preflight_id=str(preflight.get("preflight_id") or ""),
                    ttl_seconds=300,
                    requested_by=operator,
                )
                arm_token = str(arm_response.get("arm_token") or "").strip()
                if not arm_token:
                    return "❌ Live start failed: arm token was not issued."
                if not await consume_live_arm_token(session, arm_token):
                    return f"❌ Live start failed: arm token expired.\nRetry {_inline_code('/autotrader on live')}."

                control = await update_orchestrator_control(
                    session,
                    is_enabled=True,
                    is_paused=False,
                    mode="live",
                    requested_run_at=utcnow(),
                )
                await write_orchestrator_snapshot(
                    session,
                    running=False,
                    enabled=True,
                    current_activity="Live start command queued from Telegram",
                    interval_seconds=int(
                        control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS
                    ),
                )
                await create_trader_event(
                    session,
                    event_type="live_started",
                    source="telegram",
                    operator=operator,
                    message="Live trading started via Telegram",
                    payload={
                        "mode": "live",
                        "preflight_id": preflight.get("preflight_id"),
                    },
                )
                return "✅ Autotrader start queued.\nMode: LIVE"

            control = await update_orchestrator_control(
                session,
                is_enabled=True,
                is_paused=False,
                mode=selected_mode,
                requested_run_at=utcnow(),
            )
            await write_orchestrator_snapshot(
                session,
                running=False,
                enabled=True,
                current_activity="Start command queued from Telegram",
                interval_seconds=int(
                    control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS
                ),
            )
            await create_trader_event(
                session,
                event_type="started",
                source="telegram",
                operator=operator,
                message=f"Trader orchestrator started in {selected_mode} mode via Telegram",
                payload={
                    "mode": selected_mode,
                },
            )

        return f"✅ Autotrader start queued.\nMode: {_escape_html(selected_mode.upper())}"

    async def _telegram_stop_autotrader(self, *, operator: str) -> str:
        async with AsyncSessionLocal() as session:
            control = await update_orchestrator_control(
                session,
                is_enabled=False,
                is_paused=True,
                requested_run_at=None,
            )
            await write_orchestrator_snapshot(
                session,
                running=False,
                enabled=False,
                current_activity="Manual stop requested from Telegram",
                interval_seconds=int(
                    control.get("run_interval_seconds") or ORCHESTRATOR_DEFAULT_RUN_INTERVAL_SECONDS
                ),
            )
            await create_trader_event(
                session,
                event_type="stopped",
                source="telegram",
                operator=operator,
                message="Trader orchestrator stopped via Telegram",
            )
        return "🛑 Autotrader stopped."

    async def _telegram_set_kill_switch(self, *, enabled: bool, operator: str) -> str:
        async with AsyncSessionLocal() as session:
            await update_orchestrator_control(session, kill_switch=bool(enabled))
            await create_trader_event(
                session,
                event_type="kill_switch",
                severity="warn" if enabled else "info",
                source="telegram",
                operator=operator,
                message="Kill switch updated via Telegram",
                payload={"enabled": bool(enabled)},
            )
        if enabled:
            return "⛔ Kill switch enabled."
        return "✅ Kill switch disabled."

    async def _resolve_trader(self, session, query: str) -> tuple[str | None, str | None]:
        """Resolve a trader by name substring or ID prefix. Returns (trader_id, trader_name) or (None, None)."""
        q = query.strip().lower()
        traders = await list_traders(session)
        exact = [t for t in traders if t["name"].lower() == q or t["id"] == q]
        if exact:
            return exact[0]["id"], exact[0]["name"]
        partial = [t for t in traders if q in t["name"].lower() or t["id"].startswith(q)]
        if len(partial) == 1:
            return partial[0]["id"], partial[0]["name"]
        if len(partial) > 1:
            return None, None
        return None, None

    async def _telegram_traders_message(self) -> str:
        async with AsyncSessionLocal() as session:
            traders = await list_traders(session)

        if not traders:
            return "No traders configured."

        rows = []
        for t in traders:
            name = _truncate_text(t["name"], 18)
            mode = t.get("mode", "?").upper()[:3]
            enabled = "ON" if t.get("is_enabled") else "OFF"
            paused = "PAUSED" if t.get("is_paused") else "ACTIVE"
            status = f"{enabled}/{paused}"
            interval = str(t.get("interval_seconds", "?")) + "s"
            rows.append([name, mode, status, interval])

        table = _build_table(["Name", "Mod", "Status", "Intv"], rows)
        return f"📋 {_bold('Traders')}\n{_mono(table)}"

    async def _telegram_trader_detail_message(self, *, query: str) -> str:
        async with AsyncSessionLocal() as session:
            trader_id, trader_name = await self._resolve_trader(session, query)
            if trader_id is None:
                traders = await list_traders(session)
                partial = [t for t in traders if query.lower() in t["name"].lower()]
                if len(partial) > 1:
                    names = ", ".join(t["name"] for t in partial[:5])
                    return f"Multiple traders match '{_escape_html(query)}': {_escape_html(names)}.\nBe more specific."
                return f"Trader '{_escape_html(query)}' not found."

            trader = await get_trader(session, trader_id)
            if not trader:
                return f"Trader '{_escape_html(query)}' not found."

            daily_pnl = await get_daily_realized_pnl(session, trader_id=trader_id)

            open_positions = (
                await session.execute(
                    select(func.count(TraderPosition.id))
                    .where(
                        TraderPosition.trader_id == trader_id,
                        func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS,
                    )
                )
            ).scalar() or 0

            open_orders_count = (
                await session.execute(
                    select(func.count(TraderOrder.id))
                    .where(
                        TraderOrder.trader_id == trader_id,
                        func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(OPEN_ORDER_STATUSES)),
                    )
                )
            ).scalar() or 0

            total_orders = (
                await session.execute(
                    select(func.count(TraderOrder.id))
                    .where(TraderOrder.trader_id == trader_id)
                )
            ).scalar() or 0

        enabled = "ON" if trader.get("is_enabled") else "OFF"
        paused = "PAUSED" if trader.get("is_paused") else "ACTIVE"
        mode = str(trader.get("mode", "shadow")).upper()
        interval = trader.get("interval_seconds", "?")
        last_run = self._format_utc_timestamp(trader.get("last_run_at"))
        risk = trader.get("risk_limits", {})

        detail_rows = [
            ["Status", f"{enabled} / {paused}"],
            ["Mode", mode],
            ["Interval", f"{interval}s"],
            ["Open Positions", str(open_positions)],
            ["Open Orders", str(open_orders_count)],
            ["Total Orders", str(total_orders)],
            ["Daily PnL", _format_signed_money(daily_pnl)],
            ["Last Run", last_run],
        ]
        if risk.get("max_daily_loss_usd"):
            detail_rows.append(["Max Daily Loss", _format_money(risk["max_daily_loss_usd"])])
        if risk.get("max_trade_notional_usd"):
            detail_rows.append(["Max Notional", _format_money(risk["max_trade_notional_usd"])])

        table = _build_table(["Field", "Value"], detail_rows)
        return f"🔍 {_bold(_escape_html(trader.get('name', trader_id)))}\n{_mono(table)}"

    async def _telegram_orders_message(self, *, limit_arg: str | None) -> str:
        limit = _clamp_int(limit_arg, 1, 30, 10)
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(TraderOrder)
                    .order_by(TraderOrder.created_at.desc())
                    .limit(limit)
                )
            ).scalars().all()

            if not rows:
                return "No orders found."

            trader_ids = {str(r.trader_id) for r in rows if r.trader_id}
            names = await self._load_trader_name_map(session, trader_ids)

        table_rows = []
        for r in rows:
            trader = _truncate_text(names.get(str(r.trader_id), str(r.trader_id)[:8] if r.trader_id else "?"), 10)
            status = str(r.status or "?").upper()[:8]
            market = _truncate_text(r.market_question or r.market_id or "?", 22)
            notional = _format_money(float(r.notional_usd or 0))
            pnl = _format_signed_money(float(r.actual_profit or 0)) if r.actual_profit is not None else "-"
            oid = str(r.id)[:8]
            table_rows.append([oid, trader, status, notional, pnl, market])

        table = _build_table(
            ["ID", "Trader", "Status", "Notional", "PnL", "Market"],
            table_rows,
            align=["l", "l", "l", "r", "r", "l"],
        )
        return f"📋 {_bold(f'Recent Orders ({len(rows)})')}\n{_mono(table)}"

    async def _telegram_positions_message(self) -> str:
        async with AsyncSessionLocal() as session:
            positions = (
                await session.execute(
                    select(TraderPosition)
                    .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
                    .order_by(TraderPosition.total_notional_usd.desc())
                    .limit(30)
                )
            ).scalars().all()

            if not positions:
                return "No open positions."

            trader_ids = {str(p.trader_id) for p in positions if p.trader_id}
            names = await self._load_trader_name_map(session, trader_ids)

        table_rows = []
        total_notional = 0.0
        for p in positions:
            trader = _truncate_text(names.get(str(p.trader_id), str(p.trader_id)[:8] if p.trader_id else "?"), 10)
            market = _truncate_text(p.market_question or p.market_id or "?", 24)
            direction = str(p.direction or "?").upper()[:6]
            mode = str(p.mode or "?").upper()[:3]
            notional = _format_money(float(p.total_notional_usd or 0))
            total_notional += float(p.total_notional_usd or 0)
            orders = str(p.open_order_count or 0)
            table_rows.append([trader, direction, mode, notional, orders, market])

        table = _build_table(
            ["Trader", "Dir", "Mod", "Notional", "#", "Market"],
            table_rows,
            align=["l", "l", "l", "r", "r", "l"],
        )
        footer = f"Total: {_format_money(total_notional)} across {len(positions)} positions"
        return f"📊 {_bold('Open Positions')}\n{_mono(table)}\n{_escape_html(footer)}"

    async def _telegram_exposure_message(self) -> str:
        async with AsyncSessionLocal() as session:
            total_exposure = await get_gross_exposure(session)
            shadow_exposure = await get_gross_exposure(session, mode="shadow")
            live_exposure = await get_gross_exposure(session, mode="live")
            daily_pnl = await get_daily_realized_pnl(session)

            trader_exposure_rows = (
                await session.execute(
                    select(
                        TraderOrder.trader_id,
                        func.coalesce(func.sum(TraderOrder.notional_usd), 0.0),
                        func.count(TraderOrder.id),
                    )
                    .where(func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(OPEN_ORDER_STATUSES)))
                    .group_by(TraderOrder.trader_id)
                    .order_by(func.coalesce(func.sum(TraderOrder.notional_usd), 0.0).desc())
                    .limit(15)
                )
            ).all()

            trader_ids = {str(r[0]) for r in trader_exposure_rows if r[0]}
            names = await self._load_trader_name_map(session, trader_ids)

        summary_rows = [
            ["Total Exposure", _format_money(total_exposure)],
            ["Shadow", _format_money(shadow_exposure)],
            ["Live", _format_money(live_exposure)],
            ["Daily PnL", _format_signed_money(daily_pnl)],
        ]
        summary = _build_table(["Metric", "Value"], summary_rows)

        if trader_exposure_rows:
            per_trader_rows = []
            for r in trader_exposure_rows:
                name = _truncate_text(names.get(str(r[0]), str(r[0])[:8] if r[0] else "?"), 16)
                exp = _format_money(float(r[1] or 0))
                count = str(int(r[2] or 0))
                per_trader_rows.append([name, exp, count])
            per_trader = _build_table(["Trader", "Exposure", "Orders"], per_trader_rows, align=["l", "r", "r"])
            return f"💰 {_bold('Exposure')}\n{_mono(summary)}\n{_bold('Per Trader')}\n{_mono(per_trader)}"

        return f"💰 {_bold('Exposure')}\n{_mono(summary)}"

    async def _telegram_pause_trader(self, *, query: str, operator: str) -> str:
        async with AsyncSessionLocal() as session:
            trader_id, trader_name = await self._resolve_trader(session, query)
            if trader_id is None:
                return f"Trader '{_escape_html(query)}' not found or ambiguous."
            result = await set_trader_paused(session, trader_id, True)
            if result is None:
                return f"Trader '{_escape_html(query)}' not found."
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="trader_paused",
                source="telegram",
                operator=operator,
                message="Trader paused via Telegram",
            )
        return f"⏸️ Trader '{_escape_html(trader_name or query)}' paused."

    async def _telegram_resume_trader(self, *, query: str, operator: str) -> str:
        async with AsyncSessionLocal() as session:
            trader_id, trader_name = await self._resolve_trader(session, query)
            if trader_id is None:
                return f"Trader '{_escape_html(query)}' not found or ambiguous."
            result = await set_trader_paused(session, trader_id, False)
            if result is None:
                return f"Trader '{_escape_html(query)}' not found."
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="trader_resumed",
                source="telegram",
                operator=operator,
                message="Trader resumed via Telegram",
            )
        return f"▶️ Trader '{_escape_html(trader_name or query)}' resumed."

    async def _telegram_close_trader_positions(self, *, query: str, operator: str) -> str:
        async with AsyncSessionLocal() as session:
            trader_id, trader_name = await self._resolve_trader(session, query)
            if trader_id is None:
                return f"Trader '{_escape_html(query)}' not found or ambiguous."

            result = await cleanup_trader_open_orders(
                session,
                trader_id=trader_id,
                scope="all",
                target_status="cancelled",
                reason="closed via Telegram",
            )
            updated = int(result.get("updated", 0))
            matched = int(result.get("matched", 0))
            by_mode = result.get("by_mode", {})
            await sync_trader_position_inventory(session, trader_id=trader_id)
            await create_trader_event(
                session,
                trader_id=trader_id,
                event_type="trader_positions_cleanup",
                severity="warn",
                source="telegram",
                operator=operator,
                message=f"All positions closed via Telegram: {updated}/{matched}",
                payload=result,
            )

        mode_detail = ""
        shadow_count = int(by_mode.get("shadow", 0))
        live_count = int(by_mode.get("live", 0))
        if shadow_count or live_count:
            mode_detail = f"\nShadow: {_escape_html(shadow_count)} · Live: {_escape_html(live_count)}"

        return (
            f"🗑️ Trader '{_escape_html(trader_name or query)}'\n"
            f"Matched: {_escape_html(matched)} · Closed: {_escape_html(updated)}"
            f"{mode_detail}"
        )

    async def _telegram_sell_order(self, *, order_id: str, operator: str) -> str:
        oid = order_id.strip()
        async with AsyncSessionLocal() as session:
            # Try exact match first, then prefix match
            order = (
                await session.execute(
                    select(TraderOrder).where(TraderOrder.id == oid)
                )
            ).scalar_one_or_none()

            if order is None:
                candidates = (
                    await session.execute(
                        select(TraderOrder)
                        .where(TraderOrder.id.startswith(oid))
                        .limit(5)
                    )
                ).scalars().all()
                if len(candidates) == 1:
                    order = candidates[0]
                elif len(candidates) > 1:
                    ids = ", ".join(str(c.id)[:12] for c in candidates)
                    return f"Ambiguous order ID. Matches: {_escape_html(ids)}"
                else:
                    return f"Order '{_escape_html(oid)}' not found."

            current_status = str(order.status or "").lower()
            if current_status not in OPEN_ORDER_STATUSES:
                return (
                    f"Order {_escape_html(str(order.id)[:12])} is already "
                    f"{_escape_html(current_status.upper())}. Cannot close."
                )

            order_mode = _normalize_mode(order.mode)
            provider_cancelled = False

            # For live orders, attempt provider-side cancel first
            if order_mode == "live":
                payload = order.payload_json if isinstance(order.payload_json, dict) else {}
                cancel_targets = []
                clob_id = str(payload.get("provider_clob_order_id") or "").strip()
                prov_id = str(payload.get("provider_order_id") or "").strip()
                if clob_id:
                    cancel_targets.append(clob_id)
                if prov_id and prov_id not in cancel_targets:
                    cancel_targets.append(prov_id)
                if cancel_targets:
                    try:
                        from services.trader_orchestrator.order_manager import cancel_live_provider_order
                        for target in cancel_targets:
                            if await cancel_live_provider_order(target):
                                provider_cancelled = True
                                break
                    except Exception as exc:
                        logger.warning("Provider cancel failed for order", order_id=order.id, exc_info=exc)

            order.status = "cancelled"
            order.reason = f"Cancelled via Telegram by {operator}"
            order.updated_at = utcnow().replace(tzinfo=None)
            await session.commit()

            trader_name = "?"
            if order.trader_id:
                names = await self._load_trader_name_map(session, {str(order.trader_id)})
                trader_name = names.get(str(order.trader_id), str(order.trader_id)[:8])

            market = _truncate_text(order.market_question or order.market_id or "?", 40)

            await create_trader_event(
                session,
                trader_id=order.trader_id,
                event_type="order_cancelled",
                source="telegram",
                operator=operator,
                message=f"Order {order.id} cancelled via Telegram",
                payload={
                    "order_id": order.id,
                    "mode": order_mode,
                    "provider_cancelled": provider_cancelled,
                },
            )

        provider_note = ""
        if order_mode == "live":
            provider_note = (
                f"\n{_bold('Provider:')} "
                f"{_escape_html('cancelled' if provider_cancelled else 'cancel failed - check manually')}"
            )

        return (
            f"✅ Order cancelled\n"
            f"{_bold('ID:')} {_escape_html(str(order.id)[:12])}\n"
            f"{_bold('Trader:')} {_escape_html(trader_name)}\n"
            f"{_bold('Mode:')} {_escape_html(order_mode.upper())}\n"
            f"{_bold('Market:')} {_escape_html(market)}"
            f"{provider_note}"
        )

    async def _telegram_ai_chat(self, *, message: str, chat_id: str) -> str:
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage
        except Exception:
            return "AI is not available. Check AI provider configuration."

        try:
            manager = get_llm_manager()
        except Exception:
            return "AI is not initialized. Check AI provider configuration."

        if not manager.is_available():
            return "No AI provider configured. Set up an API key in Settings."

        context_text = await self._build_ai_context()

        system_prompt = (
            "You are the Homerun AI assistant, accessible via Telegram. Homerun is a prediction market "
            "arbitrage platform that scans Polymarket and Kalshi for mispricings.\n\n"
            "You have access to the current system state below. Answer the user's questions about traders, "
            "positions, orders, P&L, and strategy performance. Be concise — your responses will be displayed "
            "on a small mobile screen. Use short sentences and bullet points. Do NOT use markdown formatting "
            "like bold, italic, or links — just plain text. Numbers should use $ signs and commas. "
            "Keep responses under 600 characters when possible.\n\n"
            f"=== CURRENT SYSTEM STATE ===\n{context_text}"
        )

        history = self._ai_chat_histories.get(chat_id, [])
        messages = [LLMMessage(role="system", content=system_prompt)]
        for msg in history[-AI_CHAT_MAX_HISTORY:]:
            messages.append(LLMMessage(role=msg["role"], content=msg["content"]))
        messages.append(LLMMessage(role="user", content=message))

        try:
            response = await manager.chat(
                messages=messages,
                max_tokens=AI_CHAT_MAX_TOKENS,
                purpose="telegram_chat",
            )
        except Exception as exc:
            logger.error("AI chat failed", exc_info=exc)
            return "AI request failed. Please try again."

        answer = response.content.strip()
        if not answer:
            return "AI returned an empty response."

        if chat_id not in self._ai_chat_histories:
            self._ai_chat_histories[chat_id] = []
        self._ai_chat_histories[chat_id].append({"role": "user", "content": message})
        self._ai_chat_histories[chat_id].append({"role": "assistant", "content": answer})

        # Keep history bounded
        if len(self._ai_chat_histories[chat_id]) > AI_CHAT_MAX_HISTORY * 2:
            self._ai_chat_histories[chat_id] = self._ai_chat_histories[chat_id][-AI_CHAT_MAX_HISTORY * 2:]

        return _escape_html(answer)

    async def _build_ai_context(self) -> str:
        """Build a text summary of the current system state for AI context."""
        parts = []
        try:
            async with AsyncSessionLocal() as session:
                control = await read_orchestrator_control(session)
                snapshot = await read_orchestrator_snapshot(session)

                enabled = bool(control.get("is_enabled", False))
                paused = bool(control.get("is_paused", True))
                state = "ACTIVE" if enabled and not paused else "PAUSED"
                mode = str(control.get("mode", "shadow")).upper()
                kill_switch = bool(control.get("kill_switch", False))
                traders_running = int(snapshot.get("traders_running", 0) or 0)
                traders_total = int(snapshot.get("traders_total", 0) or 0)
                daily_pnl = float(snapshot.get("daily_pnl", 0.0) or 0.0)
                open_orders_snap = int(snapshot.get("open_orders", 0) or 0)

                parts.append(
                    f"Orchestrator: {state}, Mode: {mode}, Kill Switch: {'ON' if kill_switch else 'OFF'}, "
                    f"Traders: {traders_running}/{traders_total}, Daily PnL: {_format_signed_money(daily_pnl)}, "
                    f"Open Orders: {open_orders_snap}"
                )

                traders = await list_traders(session)
                if traders:
                    trader_lines = []
                    for t in traders:
                        t_enabled = "ON" if t.get("is_enabled") else "OFF"
                        t_paused = "PAUSED" if t.get("is_paused") else "ACTIVE"
                        t_mode = str(t.get("mode", "?")).upper()
                        t_interval = t.get("interval_seconds", "?")
                        trader_lines.append(
                            f"  - {t['name']} (ID: {t['id'][:12]}): {t_enabled}/{t_paused}, "
                            f"Mode: {t_mode}, Interval: {t_interval}s"
                        )
                    parts.append("Traders:\n" + "\n".join(trader_lines))

                exposure = await get_gross_exposure(session)
                daily_realized = await get_daily_realized_pnl(session)
                parts.append(f"Gross Exposure: {_format_money(exposure)}, Daily Realized PnL: {_format_signed_money(daily_realized)}")

                positions = (
                    await session.execute(
                        select(TraderPosition)
                        .where(func.lower(func.coalesce(TraderPosition.status, "")) == ACTIVE_POSITION_STATUS)
                        .order_by(TraderPosition.total_notional_usd.desc())
                        .limit(20)
                    )
                ).scalars().all()

                if positions:
                    trader_ids = {str(p.trader_id) for p in positions if p.trader_id}
                    names = await self._load_trader_name_map(session, trader_ids)
                    pos_lines = []
                    for p in positions:
                        tname = names.get(str(p.trader_id), str(p.trader_id)[:8] if p.trader_id else "?")
                        mkt = _truncate_text(p.market_question or p.market_id or "?", 50)
                        pos_lines.append(
                            f"  - {tname}: {str(p.direction or '?').upper()} on \"{mkt}\", "
                            f"Notional: {_format_money(float(p.total_notional_usd or 0))}, "
                            f"Mode: {str(p.mode or '?').upper()}, Orders: {p.open_order_count or 0}"
                        )
                    parts.append(f"Open Positions ({len(positions)}):\n" + "\n".join(pos_lines))
                else:
                    parts.append("Open Positions: None")

                recent_orders = (
                    await session.execute(
                        select(TraderOrder)
                        .order_by(TraderOrder.created_at.desc())
                        .limit(10)
                    )
                ).scalars().all()

                if recent_orders:
                    order_trader_ids = {str(r.trader_id) for r in recent_orders if r.trader_id}
                    order_names = await self._load_trader_name_map(session, order_trader_ids)
                    order_lines = []
                    for r in recent_orders:
                        tname = order_names.get(str(r.trader_id), str(r.trader_id)[:8] if r.trader_id else "?")
                        mkt = _truncate_text(r.market_question or r.market_id or "?", 40)
                        pnl_str = _format_signed_money(float(r.actual_profit)) if r.actual_profit is not None else "pending"
                        created = _to_utc(r.created_at).strftime("%H:%M") if r.created_at else "?"
                        order_lines.append(
                            f"  - [{str(r.id)[:8]}] {tname}: {str(r.status or '?').upper()}, "
                            f"${float(r.notional_usd or 0):,.2f}, PnL: {pnl_str}, Market: \"{mkt}\", At: {created}"
                        )
                    parts.append(f"Recent Orders ({len(recent_orders)}):\n" + "\n".join(order_lines))

        except Exception as exc:
            logger.error("Failed to build AI context", exc_info=exc)
            parts.append("(Error loading system state)")

        return "\n\n".join(parts)

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
                logger.error("Queue worker error", exc_info=exc)
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
            "parse_mode": "HTML",
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
            logger.error("Failed to send Telegram message", exc_info=exc)
            return False

    async def send_test_message(self) -> bool:
        """Send an immediate test message using currently stored settings."""
        await self._load_settings()
        if not self._bot_token or not self._chat_id:
            return False

        text = (
            f"✅ {_bold('Homerun Autotrader Notifier')}\n\n"
            "Test message received.\n"
            "Autotrader Telegram alerts are configured correctly."
        )
        return await self._send_telegram(text)


notifier = TelegramNotifier()
