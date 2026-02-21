from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from models.database import AppSettings, AsyncSessionLocal
from utils.passwords import verify_password
from utils.utcnow import utcnow

UI_LOCK_SESSION_COOKIE = "homerun_ui_lock_session"


@dataclass
class UILockSettingsSnapshot:
    enabled: bool
    idle_timeout_minutes: int
    password_hash: str | None


@dataclass
class UILockSession:
    last_activity_at: datetime


class UILockService:
    def __init__(self) -> None:
        self._settings_lock = asyncio.Lock()
        self._settings_cache: UILockSettingsSnapshot | None = None
        self._settings_loaded_at_monotonic = 0.0
        self._settings_ttl_seconds = 5.0

        self._sessions_lock = asyncio.Lock()
        self._sessions: dict[str, UILockSession] = {}

    def mark_settings_dirty(self) -> None:
        self._settings_loaded_at_monotonic = 0.0
        self._settings_cache = None

    async def invalidate_all_sessions(self) -> None:
        async with self._sessions_lock:
            self._sessions.clear()

    async def invalidate_session(self, token: str | None) -> None:
        if not token:
            return
        async with self._sessions_lock:
            self._sessions.pop(token, None)

    async def _load_settings_from_db(self) -> UILockSettingsSnapshot:
        async with AsyncSessionLocal() as session:
            row = (await session.execute(select(AppSettings).where(AppSettings.id == "default"))).scalar_one_or_none()
        if row is None:
            return UILockSettingsSnapshot(enabled=False, idle_timeout_minutes=15, password_hash=None)
        timeout_minutes = int(getattr(row, "ui_lock_idle_timeout_minutes", 15) or 15)
        timeout_minutes = max(1, min(timeout_minutes, 24 * 60))
        return UILockSettingsSnapshot(
            enabled=bool(getattr(row, "ui_lock_enabled", False)),
            idle_timeout_minutes=timeout_minutes,
            password_hash=getattr(row, "ui_lock_password_hash", None),
        )

    async def get_settings(self, *, force_refresh: bool = False) -> UILockSettingsSnapshot:
        now_monotonic = asyncio.get_running_loop().time()
        if (
            not force_refresh
            and self._settings_cache is not None
            and (now_monotonic - self._settings_loaded_at_monotonic) < self._settings_ttl_seconds
        ):
            return self._settings_cache

        async with self._settings_lock:
            now_monotonic = asyncio.get_running_loop().time()
            if (
                not force_refresh
                and self._settings_cache is not None
                and (now_monotonic - self._settings_loaded_at_monotonic) < self._settings_ttl_seconds
            ):
                return self._settings_cache
            snapshot = await self._load_settings_from_db()
            self._settings_cache = snapshot
            self._settings_loaded_at_monotonic = now_monotonic
            return snapshot

    async def _is_session_valid(self, token: str, timeout_minutes: int) -> bool:
        now = utcnow()
        max_idle_seconds = timeout_minutes * 60
        async with self._sessions_lock:
            session = self._sessions.get(token)
            if session is None:
                return False
            idle_seconds = (now - session.last_activity_at).total_seconds()
            if idle_seconds >= max_idle_seconds:
                self._sessions.pop(token, None)
                return False
            return True

    async def _touch_session(self, token: str) -> bool:
        now = utcnow()
        async with self._sessions_lock:
            session = self._sessions.get(token)
            if session is None:
                return False
            session.last_activity_at = now
            return True

    async def is_token_unlocked(self, token: Optional[str]) -> bool:
        settings = await self.get_settings()
        if not settings.enabled:
            return True
        if not settings.password_hash:
            return False
        if not token:
            return False
        return await self._is_session_valid(token, settings.idle_timeout_minutes)

    async def status(self, token: Optional[str]) -> dict[str, object]:
        settings = await self.get_settings()
        unlocked = await self.is_token_unlocked(token)
        return {
            "enabled": settings.enabled,
            "locked": bool(settings.enabled and not unlocked),
            "idle_timeout_minutes": settings.idle_timeout_minutes,
            "has_password": bool(settings.password_hash),
        }

    async def unlock(self, password: str) -> tuple[bool, str | None, str]:
        settings = await self.get_settings(force_refresh=True)
        if not settings.enabled:
            return True, None, "UI lock is disabled."
        if not settings.password_hash:
            return False, None, "UI lock password is not configured."
        if not verify_password(password, settings.password_hash):
            return False, None, "Incorrect password."

        token = secrets.token_urlsafe(32)
        now = utcnow()
        async with self._sessions_lock:
            self._sessions[token] = UILockSession(last_activity_at=now)
        return True, token, "Unlocked"

    async def record_activity(self, token: Optional[str]) -> bool:
        settings = await self.get_settings()
        if not settings.enabled:
            return True
        if not token:
            return False
        is_valid = await self._is_session_valid(token, settings.idle_timeout_minutes)
        if not is_valid:
            return False
        return await self._touch_session(token)


ui_lock_service = UILockService()
