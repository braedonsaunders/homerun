"""AutoTrader source-first config cutover and traders-source unification.

Revision ID: 202602140005
Revises: 202602140004
Create Date: 2026-02-14 23:59:59.000000
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202602140005"
down_revision = "202602140004"
branch_labels = None
depends_on = None


_SOURCE_MATRIX: dict[str, set[str]] = {
    "scanner": {"opportunity_general", "opportunity_structural"},
    "crypto": {"crypto_5m", "crypto_15m", "crypto_1h", "crypto_4h"},
    "news": {"news_reaction"},
    "weather": {"weather_consensus", "weather_alerts"},
    "traders": {"traders_flow"},
}
_SOURCE_DEFAULT_STRATEGY: dict[str, str] = {
    "scanner": "opportunity_general",
    "crypto": "crypto_15m",
    "news": "news_reaction",
    "weather": "weather_consensus",
    "traders": "traders_flow",
}
_OMNI_MAP: dict[str, str] = {
    "crypto": "crypto_15m",
    "scanner": "opportunity_general",
    "news": "news_reaction",
    "weather": "weather_consensus",
    "traders": "traders_flow",
}
_SOURCE_ALIASES: dict[str, str] = {
    "tracked_traders": "traders",
    "pool_traders": "traders",
    "insider": "traders",
}


def _table_names() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return set(inspector.get_table_names())


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _normalize_strategy_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key in {"strategy.default", "default"}:
        return "crypto_15m"
    return key


def _normalize_source_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    key = _SOURCE_ALIASES.get(key, key)
    if key == "world_intelligence":
        return ""
    return key


def _normalize_strategy_params(value: Any) -> dict[str, Any]:
    params = value if isinstance(value, dict) else {}
    out = dict(params)
    if "min_confidence" in out:
        try:
            parsed = float(out["min_confidence"])
            if parsed > 1.0:
                parsed = parsed / 100.0
            out["min_confidence"] = max(0.0, min(1.0, parsed))
        except Exception:
            out["min_confidence"] = 0.0
    return out


def _canonical_sources(raw_sources: Any, old_strategy_key: str) -> list[str]:
    values = raw_sources if isinstance(raw_sources, list) else []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        source_key = _normalize_source_key(raw)
        if not source_key or source_key in seen:
            continue
        if source_key not in _SOURCE_MATRIX:
            continue
        seen.add(source_key)
        out.append(source_key)

    if out:
        return out
    if old_strategy_key.startswith("crypto_"):
        return ["crypto"]
    if old_strategy_key == "news_reaction":
        return ["news"]
    if old_strategy_key == "traders_flow":
        return ["traders"]
    return ["scanner"]


def _map_strategy(old_strategy_key: str, source_key: str) -> str:
    if old_strategy_key == "omni_aggressive":
        return _OMNI_MAP.get(source_key, _SOURCE_DEFAULT_STRATEGY[source_key])
    if old_strategy_key in _SOURCE_MATRIX[source_key]:
        return old_strategy_key
    return _SOURCE_DEFAULT_STRATEGY[source_key]


def _build_source_configs(old_strategy_key: Any, raw_sources: Any, raw_params: Any) -> list[dict[str, Any]]:
    normalized_strategy = _normalize_strategy_key(old_strategy_key)
    sources = _canonical_sources(raw_sources, normalized_strategy)
    params = _normalize_strategy_params(raw_params)

    source_configs: list[dict[str, Any]] = []
    for source_key in sources:
        config: dict[str, Any] = {
            "source_key": source_key,
            "strategy_key": _map_strategy(normalized_strategy, source_key),
            "strategy_params": dict(params),
        }
        if source_key == "traders":
            config["traders_scope"] = {
                "modes": ["tracked", "pool"],
                "individual_wallets": [],
                "group_ids": [],
            }
        source_configs.append(config)
    return source_configs


def _backfill_traders_source_configs() -> None:
    bind = op.get_bind()
    trader_table = sa.table(
        "traders",
        sa.column("id", sa.String()),
        sa.column("strategy_key", sa.String()),
        sa.column("sources_json", sa.JSON()),
        sa.column("params_json", sa.JSON()),
        sa.column("source_configs_json", sa.JSON()),
    )
    rows = bind.execute(
        sa.select(
            trader_table.c.id,
            trader_table.c.strategy_key,
            trader_table.c.sources_json,
            trader_table.c.params_json,
            trader_table.c.source_configs_json,
        )
    ).mappings().all()

    for row in rows:
        source_configs = _build_source_configs(
            row.get("strategy_key"),
            row.get("sources_json"),
            row.get("params_json"),
        )
        bind.execute(
            trader_table.update()
            .where(trader_table.c.id == row["id"])
            .values(source_configs_json=source_configs)
        )


def _canonicalize_trade_signal_sources() -> None:
    bind = op.get_bind()
    signal_table = sa.table(
        "trade_signals",
        sa.column("id", sa.String()),
        sa.column("source", sa.String()),
        sa.column("signal_type", sa.String()),
        sa.column("dedupe_key", sa.String()),
    )
    rows = bind.execute(
        sa.select(
            signal_table.c.id,
            signal_table.c.source,
            signal_table.c.signal_type,
            signal_table.c.dedupe_key,
        ).where(
            signal_table.c.source.in_(["traders", "tracked_traders", "pool_traders", "insider"])
        )
    ).mappings().all()

    reserved_dedupe_keys: set[str] = {
        str(row["dedupe_key"])
        for row in rows
        if str(row.get("source") or "").strip().lower() == "traders"
    }

    for row in rows:
        source = str(row.get("source") or "").strip().lower()
        if source not in {"tracked_traders", "pool_traders", "insider"}:
            continue

        channel = "confluence"
        base_key = str(row.get("dedupe_key") or "").strip() or str(row.get("id") or "")
        new_dedupe = f"{channel}:{base_key}"
        if new_dedupe in reserved_dedupe_keys:
            new_dedupe = f"{channel}:{base_key}:{str(row.get('id') or '')[:8]}"
        reserved_dedupe_keys.add(new_dedupe)

        bind.execute(
            signal_table.update()
            .where(signal_table.c.id == row["id"])
            .values(source="traders", dedupe_key=new_dedupe)
        )


def _rebuild_trade_signal_snapshots() -> None:
    bind = op.get_bind()
    snapshot_table = sa.table(
        "trade_signal_snapshots",
        sa.column("source", sa.String()),
        sa.column("pending_count", sa.Integer()),
        sa.column("selected_count", sa.Integer()),
        sa.column("submitted_count", sa.Integer()),
        sa.column("executed_count", sa.Integer()),
        sa.column("skipped_count", sa.Integer()),
        sa.column("expired_count", sa.Integer()),
        sa.column("failed_count", sa.Integer()),
        sa.column("latest_signal_at", sa.DateTime()),
        sa.column("oldest_pending_at", sa.DateTime()),
        sa.column("freshness_seconds", sa.Float()),
        sa.column("updated_at", sa.DateTime()),
        sa.column("stats_json", sa.JSON()),
    )
    signal_table = sa.table(
        "trade_signals",
        sa.column("source", sa.String()),
        sa.column("status", sa.String()),
        sa.column("created_at", sa.DateTime()),
    )

    bind.execute(snapshot_table.delete())

    rows = bind.execute(
        sa.select(
            signal_table.c.source,
            signal_table.c.status,
            sa.func.count().label("count"),
            sa.func.max(signal_table.c.created_at).label("latest_created"),
            sa.func.min(
                sa.case(
                    (signal_table.c.status == "pending", signal_table.c.created_at),
                    else_=None,
                )
            ).label("oldest_pending"),
        ).group_by(signal_table.c.source, signal_table.c.status)
    ).mappings().all()

    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = str(row.get("source") or "")
        status = str(row.get("status") or "")
        count = int(row.get("count") or 0)
        stats = by_source.setdefault(
            source,
            {
                "pending_count": 0,
                "selected_count": 0,
                "submitted_count": 0,
                "executed_count": 0,
                "skipped_count": 0,
                "expired_count": 0,
                "failed_count": 0,
                "latest_signal_at": None,
                "oldest_pending_at": None,
            },
        )
        key = f"{status}_count"
        if key in stats:
            stats[key] = count
        latest_created = row.get("latest_created")
        oldest_pending = row.get("oldest_pending")
        if latest_created is not None and (
            stats["latest_signal_at"] is None or latest_created > stats["latest_signal_at"]
        ):
            stats["latest_signal_at"] = latest_created
        if oldest_pending is not None and (
            stats["oldest_pending_at"] is None or oldest_pending < stats["oldest_pending_at"]
        ):
            stats["oldest_pending_at"] = oldest_pending

    now = datetime.utcnow()
    for source, stats in by_source.items():
        latest_signal_at = stats.get("latest_signal_at")
        freshness = (
            float((now - latest_signal_at).total_seconds())
            if isinstance(latest_signal_at, datetime)
            else None
        )
        total = sum(
            int(stats.get(key, 0) or 0)
            for key in (
                "pending_count",
                "selected_count",
                "submitted_count",
                "executed_count",
                "skipped_count",
                "expired_count",
                "failed_count",
            )
        )
        bind.execute(
            snapshot_table.insert().values(
                source=source,
                pending_count=int(stats.get("pending_count") or 0),
                selected_count=int(stats.get("selected_count") or 0),
                submitted_count=int(stats.get("submitted_count") or 0),
                executed_count=int(stats.get("executed_count") or 0),
                skipped_count=int(stats.get("skipped_count") or 0),
                expired_count=int(stats.get("expired_count") or 0),
                failed_count=int(stats.get("failed_count") or 0),
                latest_signal_at=latest_signal_at,
                oldest_pending_at=stats.get("oldest_pending_at"),
                freshness_seconds=freshness,
                updated_at=now,
                stats_json={"total": total},
            )
        )


def upgrade() -> None:
    table_names = _table_names()
    if "traders" not in table_names:
        return

    trader_columns = _column_names("traders")
    if "source_configs_json" not in trader_columns:
        op.add_column(
            "traders",
            sa.Column(
                "source_configs_json",
                sa.JSON(),
                nullable=True,
                server_default=sa.text("'[]'"),
            ),
        )

    _backfill_traders_source_configs()

    if "trade_signals" in table_names:
        _canonicalize_trade_signal_sources()
    if "trade_signal_snapshots" in table_names and "trade_signals" in table_names:
        _rebuild_trade_signal_snapshots()


def downgrade() -> None:
    # Intentional no-op for safety and rollback simplicity.
    pass
