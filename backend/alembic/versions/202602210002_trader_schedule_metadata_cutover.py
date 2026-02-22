"""Cut over trader runtime metadata from trading_window_utc to trading_schedule_utc.

Revision ID: 202602210002
Revises: 202602210001
Create Date: 2026-02-21 12:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names, table_names


revision = "202602210002"
down_revision = "202602210001"
branch_labels = None
depends_on = None


def _coerce_hhmm(value: object, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    parts = text.split(":")
    if len(parts) < 2:
        return default
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except Exception:
        return default
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return default
    return f"{hour:02d}:{minute:02d}"


def _normalize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if "trading_schedule_utc" in metadata and isinstance(metadata.get("trading_schedule_utc"), dict):
        if "trading_window_utc" not in metadata:
            return metadata
        next_metadata = dict(metadata)
        next_metadata.pop("trading_window_utc", None)
        return next_metadata

    trading_window = metadata.get("trading_window_utc")
    if not isinstance(trading_window, dict):
        return metadata

    next_metadata = dict(metadata)
    next_metadata.pop("trading_window_utc", None)
    next_metadata["trading_schedule_utc"] = {
        "enabled": True,
        "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        "start_time": _coerce_hhmm(trading_window.get("start"), "00:00"),
        "end_time": _coerce_hhmm(trading_window.get("end"), "23:59"),
        "start_date": None,
        "end_date": None,
        "end_at": None,
    }
    return next_metadata


def upgrade() -> None:
    if "traders" not in table_names():
        return
    if "metadata_json" not in column_names("traders"):
        return

    bind = op.get_bind()
    traders = sa.table(
        "traders",
        sa.column("id", sa.String()),
        sa.column("metadata_json", sa.JSON()),
    )

    rows = list(bind.execute(sa.select(traders.c.id, traders.c.metadata_json)).mappings().all())
    for row in rows:
        trader_id = row.get("id")
        metadata_json = row.get("metadata_json")
        if not trader_id or not isinstance(metadata_json, dict):
            continue
        normalized = _normalize_metadata(metadata_json)
        if normalized == metadata_json:
            continue
        bind.execute(
            traders.update().where(traders.c.id == trader_id).values(metadata_json=normalized)
        )


def downgrade() -> None:
    pass
