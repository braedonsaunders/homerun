"""Backfill datasource retention defaults by source family and kind.

Revision ID: 202602190006
Revises: 202602190005
Create Date: 2026-02-19 11:15:00.000000
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

from alembic import op
import sqlalchemy as sa
from alembic_helpers import column_names, table_names


revision = "202602190006"
down_revision = "202602190005"
branch_labels = None
depends_on = None

_RETENTION_BOUNDS: dict[str, tuple[int, int]] = {
    "max_records": (1, 250_000),
    "max_age_days": (1, 3650),
}

_DEFAULT_RETENTION_BY_KIND: dict[str, dict[str, int]] = {
    "python": {"max_records": 25_000, "max_age_days": 180},
    "rss": {"max_records": 7_500, "max_age_days": 30},
    "rest_api": {"max_records": 15_000, "max_age_days": 45},
}

_EVENTS_RETENTION_POLICY: dict[str, int] = {"max_records": 50_000, "max_age_days": 365}
_EVENTS_ALL_RETENTION_POLICY: dict[str, int] = {"max_records": 100_000, "max_age_days": 365}
_STORIES_RSS_RETENTION_POLICY: dict[str, int] = {"max_records": 7_500, "max_age_days": 30}
_STORIES_REST_RETENTION_POLICY: dict[str, int] = {"max_records": 15_000, "max_age_days": 45}
_FALLBACK_RETENTION_POLICY: dict[str, int] = {"max_records": 10_000, "max_age_days": 90}


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_json_dict(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _normalize_retention(retention: object) -> dict[str, int]:
    if not isinstance(retention, dict):
        return {}

    normalized: dict[str, int] = {}
    for key, bounds in _RETENTION_BOUNDS.items():
        raw_value = retention.get(key)
        if raw_value in (None, ""):
            continue
        low, high = bounds
        try:
            parsed = int(float(raw_value))
        except (TypeError, ValueError):
            continue
        if parsed < low or parsed > high:
            continue
        normalized[key] = parsed
    return normalized


def _default_retention_policy(*, slug: object, source_key: object, source_kind: object) -> dict[str, int]:
    normalized_slug = str(slug or "").strip().lower()
    normalized_source_key = str(source_key or "").strip().lower()
    normalized_source_kind = str(source_kind or "").strip().lower()

    if normalized_slug == "events_all":
        return dict(_EVENTS_ALL_RETENTION_POLICY)
    if normalized_source_key == "events" or normalized_slug.startswith("events_"):
        return dict(_EVENTS_RETENTION_POLICY)
    if normalized_source_key == "stories":
        if normalized_source_kind == "rss" or normalized_slug.startswith("stories_google_"):
            return dict(_STORIES_RSS_RETENTION_POLICY)
        return dict(_STORIES_REST_RETENTION_POLICY)

    by_kind = _DEFAULT_RETENTION_BY_KIND.get(normalized_source_kind)
    if by_kind is not None:
        return dict(by_kind)
    return dict(_FALLBACK_RETENTION_POLICY)


def upgrade() -> None:
    if "data_sources" not in table_names():
        return

    existing_columns = set(column_names("data_sources"))
    required = {"id", "slug", "source_key", "source_kind", "retention"}
    if not required.issubset(existing_columns):
        return

    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id, slug, source_key, source_kind, retention FROM data_sources")).fetchall()
    if not rows:
        return

    data_sources = sa.table(
        "data_sources",
        sa.column("id", sa.String()),
        sa.column("retention", sa.JSON()),
        sa.column("updated_at", sa.DateTime()),
    )

    has_updated_at = "updated_at" in existing_columns
    now_value = _utcnow_naive()

    for row in rows:
        source_id = str(row[0] or "").strip()
        if not source_id:
            continue

        current_raw = _parse_json_dict(row[4])
        current_normalized = _normalize_retention(current_raw)
        defaults = _default_retention_policy(slug=row[1], source_key=row[2], source_kind=row[3])
        merged = dict(defaults)
        merged.update(current_normalized)

        if current_raw == merged:
            continue

        update_values: dict[str, object] = {"retention": merged}
        if has_updated_at:
            update_values["updated_at"] = now_value

        bind.execute(data_sources.update().where(data_sources.c.id == source_id).values(**update_values))


def downgrade() -> None:
    return
